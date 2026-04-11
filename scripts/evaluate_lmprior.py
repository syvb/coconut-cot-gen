"""Evaluate an LM-prior soft prompts file: project (already done) + behavioral eval.

Takes the output of step2_lm_prior.py (which already contains final_proj_ids) and
runs behavioral evaluation in both text and latent_via_text modes.
"""
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
    p.add_argument("--lmprior", required=True)
    p.add_argument("--targets", default="data/targets_500.pt")
    p.add_argument("--tag", default="")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max_new", type=int, default=24)
    p.add_argument("--print_samples", type=int, default=15)
    args = p.parse_args()

    device = xm.xla_device()
    print("loading model...")
    model, _prj, tok = load_codi(dtype=torch.bfloat16, device="cpu")
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    model = model.to(device)
    embed_layer = model.get_input_embeddings()

    tgt = torch.load(args.targets, weights_only=False)
    input_ids_with_bot = tgt["input_ids"]
    attn_mask_with_bot = tgt["attention_mask"]
    q_ids = input_ids_with_bot[:, :-1]
    q_mask = attn_mask_with_bot[:, :-1]
    gold = tgt["gold_answers"]
    codi_preds = tgt["pred_answers"]
    questions = tgt["questions"]

    lmp = torch.load(args.lmprior, weights_only=False)
    proj_ids = lmp["final_proj_ids"]  # (N, K)
    final_cos = lmp["final_cos"]
    K = lmp["K"]
    start = lmp.get("start", 0)
    end = lmp.get("end", proj_ids.size(0))
    N = proj_ids.size(0)

    q_ids = q_ids[start : start + N]
    q_mask = q_mask[start : start + N]
    gold = gold[start : start + N]
    codi_preds = codi_preds[start : start + N]
    questions = questions[start : start + N]

    print(f"LM prior run: N={N} K={K} lm_weight={lmp.get('lm_weight')} "
          f"lm_sharpness={lmp.get('lm_sharpness')} lm_ref={lmp.get('lm_ref')}")
    print(f"mean cos = {final_cos.mean().item():.4f}  median = {final_cos.median().item():.4f}")
    print()

    # Show sample decoded sequences
    print("=== Sample projected sequences (sorted by cos) ===")
    order = torch.argsort(final_cos, descending=True)
    for k in range(min(args.print_samples, N)):
        i = int(order[k * max(1, N // args.print_samples)])
        ids_list = proj_ids[i].tolist()
        pieces = [tok.decode([t], skip_special_tokens=False) for t in ids_list]
        print(f"  [{i}] cos={final_cos[i].item():.4f}  gold={gold[i]}  codi={codi_preds[i]}")
        print(f"       {''.join(pieces)[:300]}")
    print()

    # Run both eval modes
    results = {}
    for mode in ("text", "latent_via_text"):
        if mode == "text":
            full_ids = torch.cat([q_ids, proj_ids], dim=1)
            full_mask = torch.cat([q_mask, torch.ones(N, K, dtype=q_mask.dtype)], dim=1)
        else:
            bot_col = torch.full((N, 1), BOT_ID, dtype=torch.long)
            eot_col = torch.full((N, 1), EOT_ID, dtype=torch.long)
            one1 = torch.ones(N, 1, dtype=q_mask.dtype)
            full_ids = torch.cat([q_ids, bot_col, proj_ids, eot_col], dim=1)
            full_mask = torch.cat(
                [q_mask, one1, torch.ones(N, K, dtype=q_mask.dtype), one1], dim=1
            )

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
                    out = model(
                        inputs_embeds=embeds,
                        attention_mask=mask,
                        use_cache=False,
                    )
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

        n_gold = sum(1 for k in range(N) if pred_nums[k] == gold[k])
        n_codi = sum(1 for k in range(N) if pred_nums[k] == codi_preds[k])
        corr_idx = [k for k in range(N) if codi_preds[k] == gold[k]]
        wrong_idx = [k for k in range(N) if codi_preds[k] != gold[k]]
        ac = sum(1 for k in corr_idx if pred_nums[k] == codi_preds[k])
        aw = sum(1 for k in wrong_idx if pred_nums[k] == codi_preds[k])
        results[mode] = dict(
            gold=n_gold, codi=n_codi, correct=(ac, len(corr_idx)), wrong=(aw, len(wrong_idx)),
        )

        print(f"=== mode={mode} ===")
        print(f"  vs gold:       {n_gold}/{N} = {100*n_gold/N:.2f}%")
        print(f"  vs CODI:       {n_codi}/{N} = {100*n_codi/N:.2f}%")
        if corr_idx:
            print(f"  when CODI correct: {ac}/{len(corr_idx)} = {100*ac/len(corr_idx):.2f}%")
        if wrong_idx:
            print(f"  when CODI wrong  : {aw}/{len(wrong_idx)} = {100*aw/len(wrong_idx):.2f}%")
        print()


if __name__ == "__main__":
    main()
