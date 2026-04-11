"""Fine-tuned oracle model: input h_target + question, output K text tokens.

Architecture:
- Base: Llama-3.2-1B (fresh, no CODI LoRA), fine-tuned with LoRA.
- h_target projector: Linear(H, H) mapping CODI's last-layer residual stream
  at the final latent position into the oracle's input embedding space.
- Input sequence to oracle:
      [h_target_proj, <q_tokens>, <|bocot|>, query_0, query_1, ..., query_{K-1}]
  query_i is a fixed learnable embedding (one per position in the K window).
- Single forward pass through the oracle. At each of the K query positions the
  last-layer hidden state is projected via lm_head to logits over the vocab.
  We turn each logit vector into a "soft token embedding":
      e_i = softmax(logits_i / T) @ embed_tokens.weight

The reconstruction target is CODI:
- Build [q, <|bocot|>, e_0..e_{K-1}, <|eocot|>] as inputs_embeds, feed through
  the FROZEN CODI model, extract the last-layer residual stream at the position
  of e_{K-1} (the last soft embedding).
- Loss: 1 - cos(h_out, h_target).

At inference, argmax over logits gives discrete tokens. The counterfactual
faithfulness test substitutes a different problem's h_target and checks that
the generated text changes accordingly.
"""
import os
import sys
import argparse
import time
import math

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
    p.add_argument("--train_targets", default="data/targets_train_1024.pt")
    p.add_argument("--test_targets", default="data/targets_500.pt")
    p.add_argument("--out", default="data/oracle_weights.pt")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="softmax temperature for converting logits to soft embeds")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--n_train", type=int, default=-1,
                   help="use only first N train problems (-1 = all)")
    p.add_argument("--eval_every", type=int, default=1, help="eval every N epochs")
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=0.5,
                   help="per-element grad clip value (XLA-safe alternative to clip_grad_norm_)")
    p.add_argument("--refine_iters", type=int, default=1,
                   help="number of oracle forward passes per step (1 = no refinement, "
                        "2 = pass1 output feeds pass2 slot inputs, etc.)")
    p.add_argument("--no_query_embeds", action="store_true",
                   help="Freeze query_embeds at zero. Forces the oracle to rely on "
                        "h_target + RoPE position alone to distinguish output slots.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Oracle model
# ----------------------------------------------------------------------------


class SimpleLoRA(nn.Module):
    """Minimal LoRA applied to a single Linear layer, in place."""

    def __init__(self, linear: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B starts at zero → initial delta = 0.

    def forward(self, x):
        base = self.linear(x)
        # Preserve dtype through the LoRA branch
        dt = base.dtype
        delta = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
        return base + delta.to(dt) * self.scaling


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


def attach_lora(model, r, alpha):
    """Wrap every target linear in a SimpleLoRA and return the list of LoRA params."""
    lora_params = []
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            linear = getattr(layer.self_attn, name)
            wrapped = SimpleLoRA(linear, r, alpha)
            setattr(layer.self_attn, name, wrapped)
            lora_params += [wrapped.lora_A, wrapped.lora_B]
        for name in ("up_proj", "down_proj", "gate_proj"):
            linear = getattr(layer.mlp, name)
            wrapped = SimpleLoRA(linear, r, alpha)
            setattr(layer.mlp, name, wrapped)
            lora_params += [wrapped.lora_A, wrapped.lora_B]
    return lora_params


class Oracle(nn.Module):
    """Llama backbone + LoRA + h_target projector + K learnable query embeddings.

    h_target is injected in two places:
    (1) As a prefix token (projected by h_proj) at the start of the input.
    (2) Added on top of every learnable query embedding (so each query slot
        gets a direct view of h_target).

    This maximises the model's access to h_target at both attention time (via
    the prefix token the queries attend to) and position-local modulation (via
    the additive injection on each query slot).
    """

    def __init__(self, base_model, H, K, lora_r=64, lora_alpha=16):
        super().__init__()
        self.base = base_model
        self.K = K
        self.H = H
        # Freeze base weights
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_params = attach_lora(self.base, lora_r, lora_alpha)
        # h_target projector — 2-layer MLP. Output initialized near zero so the
        # oracle starts as base Llama but with a non-trivial gradient path.
        self.h_proj = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, H),
        )
        nn.init.normal_(self.h_proj[0].weight, std=0.02)
        nn.init.zeros_(self.h_proj[0].bias)
        nn.init.zeros_(self.h_proj[2].weight)   # output layer still starts at zero
        nn.init.zeros_(self.h_proj[2].bias)
        # K learnable "query" embeddings that act as placeholders for the output slots.
        # Init small; the `use_query_embeds` flag controls whether they are trained
        # (when False, the queries stay at ~0 and the slots are distinguished only
        # by the per-slot h_target injection plus RoPE).
        self.query_embeds = nn.Parameter(torch.zeros(K, H))
        nn.init.normal_(self.query_embeds, mean=0.0, std=0.02)
        # Also scale h_proj's output up a bit so h_target has a real signal from step 0.
        nn.init.normal_(self.h_proj[2].weight, std=0.02)
        nn.init.zeros_(self.h_proj[2].bias)

    def trainable_parameters(self, use_query_embeds=True):
        params = list(self.h_proj.parameters())
        if use_query_embeds:
            params.append(self.query_embeds)
        params += self.lora_params
        return params

    def project_h_target(self, h_target):
        """Apply h_proj in fp32 for stability; return fp32 tensor (B, H)."""
        # The h_proj layers stay in fp32
        return self.h_proj(h_target.to(next(self.h_proj.parameters()).dtype))


# ----------------------------------------------------------------------------
# Training / eval
# ----------------------------------------------------------------------------


def build_oracle_input(
    q_embeds: torch.Tensor,           # (B, q_max, H) bf16
    q_mask: torch.Tensor,             # (B, q_max)
    h_target_proj: torch.Tensor,      # (B, H) bf16 — used as prefix + added to each query
    bocot_embed: torch.Tensor,        # (H,) bf16
    query_embeds: torch.Tensor,       # (K, H) bf16
):
    B, q_max, H = q_embeds.shape
    K = query_embeds.shape[0]
    # Prefix token = h_target_proj unsqueezed
    h_tok = h_target_proj.unsqueeze(1).to(q_embeds.dtype)  # (B, 1, H)
    bocot_tok = bocot_embed.view(1, 1, H).expand(B, 1, H).to(q_embeds.dtype)
    # Each query slot = learned query + projected h_target
    query_base = query_embeds.view(1, K, H).expand(B, K, H).to(q_embeds.dtype)
    query_plus_h = query_base + h_target_proj.unsqueeze(1).to(q_embeds.dtype)  # (B, K, H)
    embeds = torch.cat([h_tok, q_embeds, bocot_tok, query_plus_h], dim=1)  # (B, 1+q+1+K, H)
    ones_q = torch.ones(B, 1, dtype=q_mask.dtype, device=q_mask.device)
    mask = torch.cat(
        [ones_q, q_mask, ones_q, torch.ones(B, K, dtype=q_mask.dtype, device=q_mask.device)],
        dim=1,
    )
    return embeds, mask


def _oracle_single_pass(oracle, q_embeds, q_mask, h_proj_bf, slot_inputs,
                         bocot_embed, embed_w_bf16, temperature):
    """One forward pass through the oracle with explicit inputs at the K query slots."""
    K = slot_inputs.size(1)
    B, q_max, H = q_embeds.shape
    h_tok = h_proj_bf.unsqueeze(1)  # (B, 1, H)
    bocot_tok = bocot_embed.view(1, 1, H).expand(B, 1, H).to(q_embeds.dtype)
    embeds = torch.cat([h_tok, q_embeds, bocot_tok, slot_inputs], dim=1)
    ones_q = torch.ones(B, 1, dtype=q_mask.dtype, device=q_mask.device)
    mask = torch.cat(
        [ones_q, q_mask, ones_q,
         torch.ones(B, K, dtype=q_mask.dtype, device=q_mask.device)],
        dim=1,
    )
    out = oracle.base(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
    logits = out.logits[:, -K:, :].clone()
    logits[:, :, 128000:] = -float("inf")
    probs = F.softmax(logits / temperature, dim=-1)
    soft_embeds = probs @ embed_w_bf16
    hard_ids = logits.argmax(dim=-1)
    return soft_embeds, hard_ids, logits


def oracle_forward(oracle, q_embeds, q_mask, h_target, embed_w_bf16, bocot_embed,
                   temperature=1.0, refine_iters=1):
    """Run the oracle, optionally with iterative refinement.

    When refine_iters==1 this is a single forward pass using the learned query
    embeddings (with h_target injected additively) at the K slots — same as before.

    When refine_iters>1, the soft embeddings from iteration i become the slot
    inputs for iteration i+1. Gradients flow through every iteration (no
    detach), giving the model a chance to coordinate slot choices across
    iterations without any autoregressive sequential loop.
    """
    B, q_max, H = q_embeds.shape
    K = oracle.K
    h_proj = oracle.project_h_target(h_target)  # fp32
    h_proj_bf = h_proj.to(torch.bfloat16)

    # Initial slot inputs = learned query + h_target (same as before)
    query_plus_h = (oracle.query_embeds.to(torch.bfloat16).unsqueeze(0).expand(B, K, H)
                    + h_proj_bf.unsqueeze(1))
    slot_inputs = query_plus_h

    soft_embeds = hard_ids = logits = None
    for it in range(refine_iters):
        soft_embeds, hard_ids, logits = _oracle_single_pass(
            oracle, q_embeds, q_mask, h_proj_bf, slot_inputs,
            bocot_embed, embed_w_bf16, temperature,
        )
        if it < refine_iters - 1:
            # Feed THIS iteration's soft embeds back as inputs for the next.
            # Still add h_target additively so the refinement keeps the target
            # signal explicit at every iter.
            slot_inputs = soft_embeds + h_proj_bf.unsqueeze(1)

    return soft_embeds, hard_ids, logits


def codi_recon_forward(codi, q_ids, q_mask, soft_embeds, bocot_embed, eocot_embed,
                        embed_layer):
    """Feed [q, bocot, soft_embeds, eocot] through frozen CODI, return last-layer
    hidden state at the position of the last soft embedding (i.e., just before eocot).
    """
    B, q_max = q_ids.shape
    H = soft_embeds.size(-1)
    K = soft_embeds.size(1)
    with torch.no_grad():
        q_embeds = embed_layer(q_ids).to(torch.bfloat16)
    bocot_tok = bocot_embed.view(1, 1, H).expand(B, 1, H).to(torch.bfloat16)
    eocot_tok = eocot_embed.view(1, 1, H).expand(B, 1, H).to(torch.bfloat16)
    # Keep gradient on soft_embeds
    seq = torch.cat([q_embeds, bocot_tok, soft_embeds.to(torch.bfloat16), eocot_tok], dim=1)
    one_q = torch.ones(B, 1, dtype=q_mask.dtype, device=q_mask.device)
    mask = torch.cat(
        [q_mask, one_q, torch.ones(B, K, dtype=q_mask.dtype, device=q_mask.device), one_q],
        dim=1,
    )
    out = codi(
        inputs_embeds=seq,
        attention_mask=mask,
        output_hidden_states=True,
        use_cache=False,
    )
    # Last-layer hidden at the position of e_{K-1}: index = q_max + 1 + K - 1 = q_max + K
    h_out = out.hidden_states[-1][:, q_max + K, :]  # (B, H)
    return h_out


def evaluate(oracle, codi, tokenizer, test_data, embed_layer, embed_w_bf16,
             bocot_embed, eocot_embed, K, batch, device, temperature,
             faithfulness=True, refine_iters=1):
    """Return dict of metrics and a few decoded samples."""
    h_targets = test_data["h_targets"]  # (N, H)
    input_ids_all = test_data["input_ids"][:, :-1]   # drop bot
    mask_all = test_data["attention_mask"][:, :-1]
    questions = test_data["questions"]
    gold_ans = test_data["gold_answers"]
    codi_pred = test_data["pred_answers"]

    N = h_targets.size(0)
    cos_all = torch.zeros(N, dtype=torch.float32)
    cos_swapped = torch.zeros(N, dtype=torch.float32)  # same q, different h_target
    hard_ids_all = torch.zeros((N, K), dtype=torch.long)

    oracle.eval()
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(s + batch, N)
            bs = e - s
            q_ids = input_ids_all[s:e].to(device)
            q_mask = mask_all[s:e].to(device)
            h_t = h_targets[s:e].to(device).to(torch.float32)
            q_embeds = embed_layer(q_ids).to(torch.bfloat16)

            soft_embeds, hard_ids, _ = oracle_forward(
                oracle, q_embeds, q_mask, h_t, embed_w_bf16, bocot_embed, temperature,
                refine_iters=refine_iters,
            )
            h_out = codi_recon_forward(
                codi, q_ids, q_mask, soft_embeds, bocot_embed, eocot_embed, embed_layer
            )
            cos = F.cosine_similarity(h_out.float(), h_t, dim=-1)
            xm.mark_step()

            cos_all[s:e] = cos.detach().cpu()
            hard_ids_all[s:e] = hard_ids.detach().cpu()

            if faithfulness and bs >= 2:
                # Counterfactual: swap h_target with the NEXT sample in the batch
                h_t_swapped = torch.cat([h_t[1:], h_t[:1]], dim=0)
                soft_swp, _, _ = oracle_forward(
                    oracle, q_embeds, q_mask, h_t_swapped, embed_w_bf16,
                    bocot_embed, temperature, refine_iters=refine_iters,
                )
                h_out_swp = codi_recon_forward(
                    codi, q_ids, q_mask, soft_swp, bocot_embed, eocot_embed, embed_layer
                )
                # Cosine of swapped output vs the ORIGINAL h_target (should be lower if
                # the oracle is actually using h_target as input)
                cos_s = F.cosine_similarity(h_out_swp.float(), h_t, dim=-1)
                xm.mark_step()
                cos_swapped[s:e] = cos_s.detach().cpu()

    metrics = {
        "mean_cos": float(cos_all.mean()),
        "median_cos": float(cos_all.median()),
        "min_cos": float(cos_all.min()),
        "pct_gt_095": float((cos_all > 0.95).float().mean() * 100),
        "mean_cos_swapped": float(cos_swapped.mean()) if faithfulness else None,
    }

    # Decode a few samples
    decoded = []
    n_samples = min(5, N)
    for i in range(n_samples):
        ids = hard_ids_all[i].tolist()
        pieces = [tokenizer.decode([t], skip_special_tokens=False) for t in ids]
        decoded.append("".join(pieces)[:200])
    metrics["samples"] = decoded
    metrics["samples_cos"] = cos_all[:n_samples].tolist()
    metrics["samples_gold"] = [gold_ans[i] for i in range(n_samples)]
    return metrics, hard_ids_all, cos_all


def main():
    args = parse_args()
    device = xm.xla_device()

    # --- Load frozen CODI (recon target) ---
    print("loading CODI...")
    codi, prj, tokenizer = load_codi(dtype=torch.bfloat16, device="cpu")
    codi.eval()
    for p in codi.parameters():
        p.requires_grad_(False)
    for p in prj.parameters():
        p.requires_grad_(False)
    codi = codi.to(device)
    embed_layer = codi.get_input_embeddings()
    H = codi.config.hidden_size
    bocot_embed = embed_layer.weight[BOT_ID].detach().to(device).to(torch.bfloat16)
    eocot_embed = embed_layer.weight[EOT_ID].detach().to(device).to(torch.bfloat16)
    # Share a single embed_w for the soft-embedding matmul (bf16 on device).
    embed_w_bf16 = embed_layer.weight.detach().to(device).to(torch.bfloat16)
    print(f"CODI loaded. H={H}, vocab={embed_w_bf16.size(0)}")

    # --- Load base Llama and wrap as Oracle ---
    print("loading base Llama for oracle...")
    base = load_base_llama(dtype=torch.bfloat16, device="cpu")
    oracle = Oracle(base, H=H, K=args.K, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    oracle = oracle.to(device)
    # Cast h_proj to fp32 for stable optimization, LoRA to fp32, queries to fp32
    oracle.h_proj = oracle.h_proj.to(torch.float32)
    for p_ in oracle.lora_params:
        p_.data = p_.data.to(torch.float32)
    oracle.query_embeds.data = oracle.query_embeds.data.to(torch.float32)
    if args.no_query_embeds:
        # Zero out and freeze
        oracle.query_embeds.data.zero_()
        oracle.query_embeds.requires_grad_(False)
    trainable = oracle.trainable_parameters(use_query_embeds=not args.no_query_embeds)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"oracle trainable params: {n_trainable:,}  (no_query_embeds={args.no_query_embeds})")

    # --- Load training data ---
    print("loading train h_targets:", args.train_targets)
    train = torch.load(args.train_targets, weights_only=False)
    train_h = train["h_targets"]  # (N, H)
    train_ids = train["input_ids"][:, :-1]     # drop bot
    train_mask = train["attention_mask"][:, :-1]
    if args.n_train > 0:
        train_h = train_h[: args.n_train]
        train_ids = train_ids[: args.n_train]
        train_mask = train_mask[: args.n_train]
    N_train = train_h.size(0)
    print(f"train problems: {N_train}")

    print("loading test targets:", args.test_targets)
    test = torch.load(args.test_targets, weights_only=False)

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    steps_per_epoch = (N_train + args.batch - 1) // args.batch
    total_steps = steps_per_epoch * args.epochs

    # LR warmup + cosine decay to 10% of peak
    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        min_lr_frac = 0.1
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (min_lr_frac + (1 - min_lr_frac) * cos_factor)
    print(f"training: {args.epochs} epochs × {steps_per_epoch} steps = {total_steps} steps")

    # Best-checkpoint tracking (keep a CPU snapshot of the oracle state at peak val cos)
    best_cos = -1.0
    best_epoch = -1
    best_state = None
    best_proj_ids = None
    best_cos_all = None
    best_samples = None

    def snapshot_oracle_state():
        """Pull every trainable param to CPU without detaching from autograd grad state."""
        sd = {
            "query_embeds": oracle.query_embeds.detach().clone().cpu(),
        }
        for name, p_ in oracle.h_proj.named_parameters():
            sd[f"h_proj.{name}"] = p_.detach().clone().cpu()
        for i_, layer in enumerate(oracle.base.model.layers):
            for mod_name in ("self_attn", "mlp"):
                mod = getattr(layer, mod_name)
                for attr in ("q_proj", "k_proj", "v_proj", "o_proj",
                             "up_proj", "down_proj", "gate_proj"):
                    if hasattr(mod, attr):
                        w = getattr(mod, attr)
                        if isinstance(w, SimpleLoRA):
                            sd[f"layer{i_}.{mod_name}.{attr}.A"] = w.lora_A.detach().clone().cpu()
                            sd[f"layer{i_}.{mod_name}.{attr}.B"] = w.lora_B.detach().clone().cpu()
        return sd

    step = 0
    for epoch in range(args.epochs):
        oracle.train()
        # Simple shuffle each epoch
        perm = torch.randperm(N_train)
        t0 = time.time()
        recon_sum = 0.0
        n_batches = 0
        for bi in range(steps_per_epoch):
            idx = perm[bi * args.batch : (bi + 1) * args.batch]
            if idx.numel() == 0:
                continue
            q_ids = train_ids[idx].to(device)
            q_mask = train_mask[idx].to(device)
            h_t = train_h[idx].to(device).to(torch.float32)
            q_embeds = embed_layer(q_ids).to(torch.bfloat16)

            opt.zero_grad(set_to_none=True)
            soft_embeds, _, _ = oracle_forward(
                oracle, q_embeds, q_mask, h_t, embed_w_bf16, bocot_embed, args.temperature
            )
            h_out = codi_recon_forward(
                codi, q_ids, q_mask, soft_embeds, bocot_embed, eocot_embed, embed_layer
            )
            cos = F.cosine_similarity(h_out.float(), h_t, dim=-1)
            loss = (1.0 - cos).mean()
            loss.backward()
            # XLA-safe per-element grad clip (avoids the clip_grad_norm_ issue)
            with torch.no_grad():
                for p_ in trainable:
                    if p_.grad is not None:
                        p_.grad.clamp_(-args.grad_clip, args.grad_clip)
            # Manual LR schedule
            lr_now = lr_at(step)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            opt.step()
            xm.mark_step()

            recon_sum += loss.item()
            n_batches += 1
            if bi % 16 == 0 or bi == steps_per_epoch - 1:
                print(
                    f"  epoch {epoch+1} step {bi+1}/{steps_per_epoch}  "
                    f"loss={loss.item():.4f}  cos_mean={cos.mean().item():.4f}  lr={lr_now:.2e}",
                    flush=True,
                )
            step += 1
        dt = time.time() - t0
        print(
            f"epoch {epoch+1} done  avg_loss={recon_sum/max(1,n_batches):.4f}  dt={dt:.1f}s",
            flush=True,
        )

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics, hard_ids, cos_all = evaluate(
                oracle, codi, tokenizer, test, embed_layer, embed_w_bf16,
                bocot_embed, eocot_embed, args.K, args.batch, device, args.temperature,
                faithfulness=True, refine_iters=args.refine_iters,
            )
            improved = metrics["mean_cos"] > best_cos
            marker = "  **BEST**" if improved else ""
            print(
                f"  [test] mean_cos={metrics['mean_cos']:.4f}  "
                f"median={metrics['median_cos']:.4f}  min={metrics['min_cos']:.4f}  "
                f"pct>0.95={metrics['pct_gt_095']:.1f}%  "
                f"swapped_cos={metrics['mean_cos_swapped']:.4f}{marker}",
                flush=True,
            )
            for i, (s, c, g) in enumerate(zip(metrics["samples"], metrics["samples_cos"],
                                              metrics["samples_gold"])):
                print(f"    sample[{i}] cos={c:.3f} gold={g}: {s!r}")
            if improved:
                best_cos = metrics["mean_cos"]
                best_epoch = epoch + 1
                best_state = snapshot_oracle_state()
                best_proj_ids = hard_ids.clone()
                best_cos_all = cos_all.clone()
                best_samples = metrics.copy()

    # Save BEST checkpoint (the peak validation snapshot) rather than final state
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_state = best_state if best_state is not None else snapshot_oracle_state()
    save_cos = best_cos_all if best_cos_all is not None else cos_all
    save_proj = best_proj_ids if best_proj_ids is not None else hard_ids
    print(f"\nBEST epoch={best_epoch}  mean_cos={best_cos:.4f}")
    if best_samples is not None:
        print("BEST samples (epoch {}):".format(best_epoch))
        for i, (s, c, g) in enumerate(zip(best_samples["samples"],
                                           best_samples["samples_cos"],
                                           best_samples["samples_gold"])):
            print(f"  sample[{i}] cos={c:.3f} gold={g}: {s!r}")
    torch.save(
        {
            "oracle_state": save_state,
            "test_final_cos": save_cos,
            "test_final_proj_ids": save_proj,
            "K": args.K,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "best_epoch": best_epoch,
            "best_mean_cos": best_cos,
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
