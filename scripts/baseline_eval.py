"""Baselines: generate from [question] or [question, bocot, eocot] (empty latent)."""
import os
import sys
import argparse
import re

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "2,2,1")

import torch
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(__file__))
from codi_loader import load_codi, BOT_ID, EOT_ID


def extract_answer_number(s: str):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return float("inf")
    try:
        return float(nums[-1])
    except ValueError:
        return float("inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="data/targets_500.pt")
    p.add_argument("--mode", choices=["question_only", "empty_latent"], default="empty_latent")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max_new", type=int, default=24)
    p.add_argument("--out", default="data/baseline_empty_latent.pt")
    args = p.parse_args()

    device = xm.xla_device()
    print("loading model...")
    model, _prj, tok = load_codi(dtype=torch.bfloat16, device="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    embed_layer = model.get_input_embeddings()

    targets = torch.load(args.targets, weights_only=False)
    input_ids_with_bot = targets["input_ids"]  # (N, q+1)
    attn_mask_with_bot = targets["attention_mask"]
    q_only_ids = input_ids_with_bot[:, :-1]
    q_only_mask = attn_mask_with_bot[:, :-1]
    gold = targets["gold_answers"]
    codi_preds = targets["pred_answers"]
    N = q_only_ids.size(0)

    if args.mode == "question_only":
        full_ids = q_only_ids
        full_mask = q_only_mask
    else:  # empty_latent: [q, bocot, eocot] (0 latent tokens between them)
        one = torch.ones(N, 1, dtype=q_only_mask.dtype)
        bot_col = torch.full((N, 1), BOT_ID, dtype=torch.long)
        eot_col = torch.full((N, 1), EOT_ID, dtype=torch.long)
        full_ids = torch.cat([q_only_ids, bot_col, eot_col], dim=1)
        full_mask = torch.cat([q_only_mask, one, one], dim=1)

    pred_texts = [""] * N
    pred_nums = [float("inf")] * N

    n_batches = (N + args.batch - 1) // args.batch
    for bi in range(n_batches):
        s, e = bi * args.batch, min((bi + 1) * args.batch, N)
        bs = e - s
        b_ids = full_ids[s:e].to(device)
        b_mask = full_mask[s:e].to(device)
        with torch.no_grad():
            embeds = embed_layer(b_ids).to(torch.bfloat16)
            mask = b_mask.clone()
            gen = torch.zeros((bs, args.max_new), dtype=torch.long, device=device)
            for gi in range(args.max_new):
                out = model(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
                logits = out.logits[:, -1, : model.config.vocab_size - 1]
                nt = logits.argmax(dim=-1)
                gen[:, gi] = nt
                ne = embed_layer(nt).to(torch.bfloat16).unsqueeze(1)
                embeds = torch.cat([embeds, ne], dim=1)
                mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            xm.mark_step()
        g = gen.detach().to("cpu")
        for i in range(bs):
            ids = g[i].tolist()
            if tok.eos_token_id in ids:
                ids = ids[: ids.index(tok.eos_token_id)]
            t = tok.decode(ids, skip_special_tokens=True)
            pred_texts[s + i] = t
            pred_nums[s + i] = extract_answer_number(t)
        print(f"batch {bi+1}/{n_batches}  sample={pred_texts[s][:60]!r}", flush=True)

    n_gold = sum(1 for k in range(N) if pred_nums[k] == gold[k])
    n_codi = sum(1 for k in range(N) if pred_nums[k] == codi_preds[k])
    print(f"\nmode={args.mode}")
    print(f"accuracy vs gold: {n_gold}/{N} = {100*n_gold/N:.2f}%")
    print(f"agreement with CODI: {n_codi}/{N} = {100*n_codi/N:.2f}%")

    torch.save(
        {"pred_texts": pred_texts, "pred_answers": pred_nums, "mode": args.mode}, args.out
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
