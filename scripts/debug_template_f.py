"""Sanity check: does template F actually use h_tok, or is it being ignored?

Run a single batch through CODI with three different h_tok values and check
whether the hidden state at position q_max+1 differs.
"""
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
import sys
import torch
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(__file__))
from codi_loader import load_codi, BOT_ID, EOT_ID

def main():
    device = xm.xla_device()
    print("loading codi...")
    model, _, tok = load_codi(dtype=torch.bfloat16, device="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    embed_layer = model.get_input_embeddings()

    data = torch.load("data/targets_500.pt", weights_only=False)
    h_targets = data["h_targets"][:4].to(device).to(torch.bfloat16)  # (4, H)
    q_ids = data["input_ids"][:4, :-1].to(device)   # drop bot
    q_mask = data["attention_mask"][:4, :-1].to(device)
    B, q_max = q_ids.shape
    H = h_targets.size(-1)
    print(f"B={B} q_max={q_max} H={H}")

    with torch.no_grad():
        q_embeds = embed_layer(q_ids).to(torch.bfloat16)
        bot_tok = embed_layer.weight[BOT_ID].detach().to(torch.bfloat16).view(1, 1, -1).expand(B, 1, H)
        eot_tok = embed_layer.weight[EOT_ID].detach().to(torch.bfloat16).view(1, 1, -1).expand(B, 1, H)
        one = torch.ones(B, 1, dtype=q_mask.dtype, device=device)

        def run(h_tok):
            embeds = torch.cat([q_embeds, bot_tok, h_tok, eot_tok], dim=1)
            mask = torch.cat([q_mask, one, one, one], dim=1)
            out = model(
                inputs_embeds=embeds,
                attention_mask=mask,
                use_cache=False,
                output_hidden_states=True,
            )
            # Report:
            #  - hidden state at h_tok position (q_max+1)
            #  - hidden state at eot position (q_max+2) — the position whose logits
            #    drive the first generated token
            #  - the first-token prediction (argmax of logits at position q_max+2)
            h_at_htok = out.hidden_states[-1][:, q_max+1, :]
            h_at_eot = out.hidden_states[-1][:, q_max+2, :]
            first_logits = out.logits[:, q_max+2, :]
            first_tok = first_logits.argmax(dim=-1)
            return h_at_htok, h_at_eot, first_tok, first_logits

        # Real h
        h_real = h_targets.unsqueeze(1)  # (B, 1, H)
        h_at_htok_r, h_at_eot_r, first_r, logits_r = run(h_real)

        # Zero h
        h_zero = torch.zeros_like(h_real)
        h_at_htok_z, h_at_eot_z, first_z, logits_z = run(h_zero)

        # Random h (same norm as real)
        g = torch.randn_like(h_real)
        target_norm = h_real.norm(dim=-1, keepdim=True)
        h_rand = g / g.norm(dim=-1, keepdim=True) * target_norm
        h_at_htok_x, h_at_eot_x, first_x, logits_x = run(h_rand)

        xm.mark_step()

    print("\n=== h_tok norms ===")
    for name, h in [("real", h_real), ("zero", h_zero), ("random", h_rand)]:
        print(f"  {name}: norm per sample = {h.norm(dim=-1).squeeze(-1).tolist()}")

    print("\n=== h at position q_max+1 (the patched slot's output) ===")
    print(f"  real vs zero: L2 diff = {(h_at_htok_r - h_at_htok_z).norm(dim=-1).tolist()}")
    print(f"  real vs random: L2 diff = {(h_at_htok_r - h_at_htok_x).norm(dim=-1).tolist()}")
    print(f"  real norms: {h_at_htok_r.norm(dim=-1).tolist()}")
    print(f"  zero norms: {h_at_htok_z.norm(dim=-1).tolist()}")
    print(f"  random norms: {h_at_htok_x.norm(dim=-1).tolist()}")

    print("\n=== h at position q_max+2 (eot's output — drives generation) ===")
    print(f"  real vs zero: L2 diff = {(h_at_eot_r - h_at_eot_z).norm(dim=-1).tolist()}")
    print(f"  real vs random: L2 diff = {(h_at_eot_r - h_at_eot_x).norm(dim=-1).tolist()}")

    print("\n=== first-token logits ===")
    print(f"  real vs zero: max abs diff = {(logits_r - logits_z).abs().max(dim=-1).values.tolist()}")
    print(f"  real vs random: max abs diff = {(logits_r - logits_x).abs().max(dim=-1).values.tolist()}")

    print("\n=== first argmax tokens ===")
    for i in range(B):
        tr = tok.decode([first_r[i].item()])
        tz = tok.decode([first_z[i].item()])
        tx = tok.decode([first_x[i].item()])
        print(f"  [{i}] real={tr!r:12} zero={tz!r:12} random={tx!r:12}")


if __name__ == "__main__":
    main()
