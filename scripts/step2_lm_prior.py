"""Step 2 variant: soft prompt opt with an auxiliary LM-prior loss.

Forward through a reference LM on [q, e_1..e_K] and compute auto-regressive log
P(e_i | q, e_1..e_{i-1}) where the "identity" of each e_i is a softmax distribution
over tokens induced by cosine similarity to the embed_tokens matrix:

    p_i(v) = softmax_v(sharpness * cos(e_i, embed_v))

The LM loss is then

    L_LM = - (1/K) * sum_i sum_v p_i(v) * log_softmax(ref_logits_{i-1})(v)

i.e., the expected auto-regressive log-prob of e_i's token identity under the
reference LM, conditioned on the preceding soft embeddings.

The reference LM may be either the CODI-merged model (self-prior) or a base
Llama without LoRA (external prior).

We ALSO forward through the CODI model to get the recon loss — the CODI model
is the one whose h_target we are trying to hit.

So each optimization step does 1 (self) or 2 (external) forward passes.
"""
import os
import sys
import argparse
import time

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_VISIBLE_DEVICES", "0,1,2,3")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(__file__))
from codi_loader import load_codi, load_base_llama, PAD_ID, BOT_ID, EOT_ID


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="data/targets_500.pt")
    p.add_argument("--out", default="data/soft_prompts_lmprior.pt")
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--n", type=int, default=-1)
    p.add_argument("--init_std", type=float, default=0.02)
    p.add_argument("--start", type=int, default=0, help="start problem index (for sharding)")
    p.add_argument("--end", type=int, default=-1, help="end problem index (exclusive)")

    p.add_argument("--lm_weight", type=float, default=1.0, help="weight on LM prior loss")
    p.add_argument("--lm_sharpness", type=float, default=5.0,
                   help="temperature for softmax(cos * sharpness) over tokens")
    p.add_argument("--lm_ref", choices=["self", "base"], default="base",
                   help="self: use CODI model. base: load base Llama without LoRA.")
    p.add_argument("--lm_exclude_specials", action="store_true",
                   help="mask special tokens (>=128000) in the cosine-to-tokens distribution")
    return p.parse_args()


def main():
    args = parse_args()
    device = xm.xla_device()

    print("loading CODI model (recon target)...", flush=True)
    codi_model, prj, tok = load_codi(dtype=torch.bfloat16, device="cpu")
    codi_model.eval()
    for p in codi_model.parameters():
        p.requires_grad_(False)
    for p in prj.parameters():
        p.requires_grad_(False)
    codi_model = codi_model.to(device)
    embed_layer = codi_model.get_input_embeddings()
    H = codi_model.config.hidden_size

    if args.lm_ref == "base":
        print("loading base Llama (LM prior)...", flush=True)
        ref_model = load_base_llama(dtype=torch.bfloat16, device="cpu")
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model = ref_model.to(device)
    else:
        ref_model = codi_model  # share

    # Embeddings for the cosine-distribution over tokens
    embed_w = embed_layer.weight.detach().to(torch.float32)  # (V, H)
    V = embed_w.size(0)
    if args.lm_exclude_specials:
        # Mask off specials so the distribution never puts mass on them.
        # We'll build a mask vector that will subtract inf from sims at those positions.
        mask_v = torch.zeros(V, dtype=torch.float32)
        mask_v[128000:] = -float("inf")  # all <|...|> special tokens are >=128000
        mask_v = mask_v.to(device)
    else:
        mask_v = None
    embed_norm = F.normalize(embed_w, dim=-1).to(torch.bfloat16).to(device)  # (V, H)

    print("loading targets:", args.targets, flush=True)
    data = torch.load(args.targets, weights_only=False)
    h_targets_all = data["h_targets"]
    input_ids_all = data["input_ids"][:, :-1]
    attn_mask_all = data["attention_mask"][:, :-1]
    q_max = input_ids_all.size(1)

    N_total = h_targets_all.size(0)
    end = N_total if args.end < 0 else min(args.end, N_total)
    start = max(0, args.start)
    N = end - start
    if args.n > 0:
        N = min(N, args.n)
        end = start + N
    print(
        f"problems: {start}..{end-1} (N={N}) | K={args.K} | steps={args.steps} | "
        f"lr={args.lr} | batch={args.batch} | lm_weight={args.lm_weight} | "
        f"lm_sharpness={args.lm_sharpness} | lm_ref={args.lm_ref}",
        flush=True,
    )

    h_targets_all = h_targets_all[start:end]
    input_ids_all = input_ids_all[start:end]
    attn_mask_all = attn_mask_all[start:end]

    soft_prompts_all = torch.zeros((N, args.K, H), dtype=torch.float32)
    final_cos = torch.zeros(N, dtype=torch.float32)
    final_recon = torch.zeros(N, dtype=torch.float32)
    final_lm = torch.zeros(N, dtype=torch.float32)
    # Also save the argmax-projected IDs at the end of training.
    final_proj_ids = torch.zeros((N, args.K), dtype=torch.long)

    n_batches = (N + args.batch - 1) // args.batch
    for bi in range(n_batches):
        s, e = bi * args.batch, min((bi + 1) * args.batch, N)
        bs = e - s
        b_ids = input_ids_all[s:e].to(device)
        b_mask = attn_mask_all[s:e].to(device)
        b_target = h_targets_all[s:e].to(device).to(torch.float32)

        with torch.no_grad():
            q_embeds = embed_layer(b_ids).to(torch.float32)  # (B, q, H)

        soft = torch.randn(bs, args.K, H, dtype=torch.float32, device=device) * args.init_std
        soft.requires_grad_(True)
        opt = torch.optim.Adam([soft], lr=args.lr)

        soft_mask = torch.ones(bs, args.K, dtype=b_mask.dtype, device=device)
        full_mask = torch.cat([b_mask, soft_mask], dim=1)

        t0 = time.time()
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)

            full_embeds_bf = torch.cat([q_embeds, soft], dim=1).to(torch.bfloat16)

            # === 1. Reconstruction forward through CODI ===
            codi_out = codi_model(
                inputs_embeds=full_embeds_bf,
                attention_mask=full_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            h_out = codi_out.hidden_states[-1][:, -1, :].to(torch.float32)
            recon_loss = (1.0 - F.cosine_similarity(h_out, b_target, dim=-1)).mean()

            # === 2. LM prior forward (may be the SAME as CODI if lm_ref=self) ===
            if args.lm_ref == "self":
                # reuse codi logits — they are output at every position
                ref_logits = codi_out.logits  # (B, q+K, V)
            else:
                ref_out = ref_model(
                    inputs_embeds=full_embeds_bf,
                    attention_mask=full_mask,
                    use_cache=False,
                )
                ref_logits = ref_out.logits  # (B, q+K, V)

            # Predictor logits for position (q_end + i) are at seq index (q_end + i - 1)
            # With left-padded fixed q_max, q_end = q_max - 1 (last non-pad question token).
            # Predictor positions are [q_max-1, q_max, ..., q_max+K-2], predicting e_1..e_K.
            pred_logits = ref_logits[:, q_max - 1 : q_max + args.K - 1, :].to(torch.float32)  # (B, K, V)
            ref_log_probs = F.log_softmax(pred_logits, dim=-1)  # (B, K, V)

            # p_i(v) = softmax(sharpness * cos(e_i, embed_v))
            soft_norm = F.normalize(soft, dim=-1).to(torch.bfloat16)  # (B, K, H)
            sims = soft_norm.reshape(-1, H) @ embed_norm.t()  # (B*K, V)
            sims = sims.to(torch.float32).reshape(bs, args.K, V)
            if mask_v is not None:
                sims = sims + mask_v  # -inf on specials
            soft_probs = F.softmax(args.lm_sharpness * sims, dim=-1)  # (B, K, V)

            # LM loss = -E_{v ~ p_i}[log P_ref(v | context)]
            lm_loss = -(soft_probs * ref_log_probs).sum(dim=-1).mean()

            loss = recon_loss + args.lm_weight * lm_loss
            loss.backward()
            opt.step()
            xm.mark_step()

            if step % 50 == 0 or step == args.steps - 1:
                with torch.no_grad():
                    cos_now = F.cosine_similarity(h_out, b_target, dim=-1).mean()
                    max_prob = soft_probs.max(dim=-1).values.mean()
                print(
                    f"  batch {bi+1}/{n_batches} step {step:4d}  "
                    f"recon={recon_loss.item():.4f}  lm={lm_loss.item():.4f}  "
                    f"cos={cos_now.item():.4f}  max_p={max_prob.item():.4f}",
                    flush=True,
                )

        # Final eval
        with torch.no_grad():
            full_embeds_bf = torch.cat([q_embeds, soft], dim=1).to(torch.bfloat16)
            codi_out = codi_model(
                inputs_embeds=full_embeds_bf,
                attention_mask=full_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            h_out = codi_out.hidden_states[-1][:, -1, :].to(torch.float32)
            cos_vec = F.cosine_similarity(h_out, b_target, dim=-1)  # (B,)
            recon_vec = 1.0 - cos_vec

            soft_norm = F.normalize(soft, dim=-1).to(torch.bfloat16)
            sims = soft_norm.reshape(-1, H) @ embed_norm.t()
            sims = sims.to(torch.float32).reshape(bs, args.K, V)
            if mask_v is not None:
                sims = sims + mask_v
            proj_ids = sims.argmax(dim=-1)  # (B, K)
            xm.mark_step()

        soft_cpu = soft.detach().to("cpu")
        cos_cpu = cos_vec.detach().to("cpu")
        recon_cpu = recon_vec.detach().to("cpu")
        proj_cpu = proj_ids.detach().to("cpu")

        soft_prompts_all[s:e] = soft_cpu
        final_cos[s:e] = cos_cpu
        final_recon[s:e] = recon_cpu
        final_proj_ids[s:e] = proj_cpu

        dt = time.time() - t0
        print(
            f"batch {bi+1}/{n_batches} done in {dt:.1f}s  mean_cos={cos_cpu.mean().item():.4f}",
            flush=True,
        )

    print(f"\nmean final cosine: {final_cos.mean().item():.4f}")
    print(f"median: {final_cos.median().item():.4f}  min: {final_cos.min().item():.4f}")

    payload = {
        "soft_prompts": soft_prompts_all,
        "final_cos": final_cos,
        "final_recon": final_recon,
        "final_proj_ids": final_proj_ids,
        "K": args.K,
        "lm_weight": args.lm_weight,
        "lm_sharpness": args.lm_sharpness,
        "lm_ref": args.lm_ref,
        "steps": args.steps,
        "lr": args.lr,
        "start": start,
        "end": end,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
