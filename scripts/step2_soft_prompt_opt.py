"""Step 2: per-problem soft prompt optimization to reconstruct h_target.

For each problem we keep K learnable embedding vectors. The forward is
[problem_tokens, e_1..e_K]. We optimize the e_i to minimize 1 - cos(h_out, h_target)
where h_out is the last-layer residual stream at the last position of the input.

To stay TPU-friendly we batch problems together: a single batch of size B has
B*K learnable vectors of dim H, all updated together. The model forward is
shared (one big batched forward). XLA compiles once for the chosen shape.
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
import torch_xla
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(__file__))
from codi_loader import load_codi, BOT_ID, PAD_ID


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="data/targets_500.pt")
    p.add_argument("--out", default="data/soft_prompts.pt")
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--loss", choices=["cos", "mse"], default="cos")
    p.add_argument("--n", type=int, default=-1, help="limit number of problems (-1 = all)")
    p.add_argument("--init_std", type=float, default=0.02)
    p.add_argument("--reg_weight", type=float, default=0.0,
                   help="weight on the 'pull soft embedding toward nearest real token' penalty.")
    p.add_argument("--reg_exclude_specials", action="store_true",
                   help="restrict nearest-token search to non-special tokens (<128000, no PAD/BOT/EOT)")
    return p.parse_args()


def main():
    args = parse_args()
    device = xm.xla_device()

    print("loading model...", flush=True)
    model, prj, _tok = load_codi(dtype=torch.bfloat16, device="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in prj.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    embed_layer = model.get_input_embeddings()
    H = model.config.hidden_size

    # Precompute normalized embedding matrix for the token-proximity regularizer.
    # Exclude specials if requested so the prior doesn't just collapse to <|eocot|>/<|begin_of_text|>.
    with torch.no_grad():
        embed_w = embed_layer.weight.detach().to(torch.float32)  # (V, H)
        V = embed_w.size(0)
        embed_w_masked = embed_w.clone()
        if args.reg_exclude_specials:
            # zero out specials so they score 0 cosine (won't ever be the argmax)
            for sid in (PAD_ID, BOT_ID):
                embed_w_masked[sid] = 0
            # End-of-turn / reserved tokens: 128000..V-1 (excludes actual text tokens below 128000)
            embed_w_masked[128000:] = 0
        embed_norm = F.normalize(embed_w_masked, dim=-1).to(torch.bfloat16).to(device)  # (V, H)

    print("loading targets:", args.targets, flush=True)
    data = torch.load(args.targets, weights_only=False)
    h_targets_all = data["h_targets"]  # (N, H) float32
    # Drop the trailing <|bocot|> token — soft prompts replace the latent reasoning
    # that would have been triggered by it. We use [problem_tokens, e_1..e_K].
    input_ids_all = data["input_ids"][:, :-1]  # (N, q_max)
    attn_mask_all = data["attention_mask"][:, :-1]  # (N, q_max)

    N_total = h_targets_all.size(0)
    N = N_total if args.n < 0 else min(args.n, N_total)
    print(f"problems: {N} | K={args.K} | steps={args.steps} | lr={args.lr} | batch={args.batch}", flush=True)

    # Per-problem learnable soft prompts (kept on CPU until copied to device per batch).
    soft_prompts_all = torch.zeros((N, args.K, H), dtype=torch.float32)
    final_losses = torch.zeros(N, dtype=torch.float32)
    final_cos = torch.zeros(N, dtype=torch.float32)

    n_batches = (N + args.batch - 1) // args.batch
    for bi in range(n_batches):
        s, e = bi * args.batch, min((bi + 1) * args.batch, N)
        bs = e - s

        b_ids = input_ids_all[s:e].to(device)  # (B, q)
        b_mask = attn_mask_all[s:e].to(device)
        b_target = h_targets_all[s:e].to(device).to(torch.float32)  # (B, H)

        with torch.no_grad():
            q_embeds = embed_layer(b_ids).to(torch.float32)  # (B, q, H), keep fp32 for grad path

        # Initialize soft prompts at small noise
        soft = torch.randn(bs, args.K, H, dtype=torch.float32, device=device) * args.init_std
        soft.requires_grad_(True)
        opt = torch.optim.Adam([soft], lr=args.lr)

        # Extend mask with K ones
        soft_mask = torch.ones(bs, args.K, dtype=b_mask.dtype, device=device)
        full_mask = torch.cat([b_mask, soft_mask], dim=1)

        t0 = time.time()
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            # Cast soft to bf16 for the model, but keep gradient path through bf16->fp32 cast.
            full_embeds = torch.cat([q_embeds, soft], dim=1).to(torch.bfloat16)
            out = model(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            h_out = out.hidden_states[-1][:, -1, :].to(torch.float32)  # (B, H)

            if args.loss == "cos":
                recon_per = 1.0 - F.cosine_similarity(h_out, b_target, dim=-1)  # (B,)
            else:  # mse
                recon_per = ((h_out - b_target) ** 2).mean(dim=-1)
            recon_loss = recon_per.mean()

            if args.reg_weight > 0:
                # For each (b, k) in `soft`, pull it close to its nearest token embedding.
                soft_norm = F.normalize(soft, dim=-1).to(torch.bfloat16)  # (B, K, H)
                sims = soft_norm.reshape(-1, H) @ embed_norm.t()  # (B*K, V)
                max_sims = sims.max(dim=-1).values  # (B*K,)
                reg_loss = (1.0 - max_sims.to(torch.float32)).mean()
                loss = recon_loss + args.reg_weight * reg_loss
            else:
                reg_loss = torch.zeros((), device=device)
                loss = recon_loss
            loss.backward()
            opt.step()
            xm.mark_step()

            if step % 50 == 0 or step == args.steps - 1:
                with torch.no_grad():
                    cos = F.cosine_similarity(h_out, b_target, dim=-1).mean()
                print(
                    f"  batch {bi+1}/{n_batches} step {step:4d}  "
                    f"recon={recon_loss.item():.4f}  reg={reg_loss.item():.4f}  "
                    f"cos={cos.item():.4f}",
                    flush=True,
                )

        # Final eval
        with torch.no_grad():
            full_embeds = torch.cat([q_embeds, soft], dim=1).to(torch.bfloat16)
            out = model(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            h_out = out.hidden_states[-1][:, -1, :].to(torch.float32)
            cos_final = F.cosine_similarity(h_out, b_target, dim=-1)  # (B,)
            loss_final = (1.0 - cos_final) if args.loss == "cos" else ((h_out - b_target) ** 2).mean(dim=-1)
            xm.mark_step()
        cos_final_cpu = cos_final.detach().to("cpu")
        loss_final_cpu = loss_final.detach().to("cpu")
        soft_cpu = soft.detach().to("cpu")
        soft_prompts_all[s:e] = soft_cpu
        final_cos[s:e] = cos_final_cpu
        final_losses[s:e] = loss_final_cpu

        dt = time.time() - t0
        print(
            f"batch {bi+1}/{n_batches} done in {dt:.1f}s  mean_cos={cos_final_cpu.mean().item():.4f}",
            flush=True,
        )

    print(f"\nmean final cosine: {final_cos.mean().item():.4f}  median={final_cos.median().item():.4f}")
    print(f"min={final_cos.min().item():.4f}  max={final_cos.max().item():.4f}")

    payload = {
        "soft_prompts": soft_prompts_all,  # (N, K, H) float32
        "final_cos": final_cos,
        "final_losses": final_losses,
        "K": args.K,
        "loss_type": args.loss,
        "steps": args.steps,
        "lr": args.lr,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
