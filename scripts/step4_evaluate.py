"""Step 4: evaluate A (cosine histogram), B (semantic readability), C (behavioral equivalence).

C: feed [problem_tokens, projected_token_ids] back into the model as ACTUAL TEXT
   (not embeddings) and let it generate. Compare to the original CODI greedy answer
   from step 1. We DO NOT run latent reasoning or feed the bocot/eocot tokens —
   just question + projected tokens, then generate normally.
"""
import os
import sys
import argparse
import json
import re

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "2,2,1")

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(__file__))
from codi_loader import load_codi, BOT_ID, EOT_ID, PAD_ID, INF_LATENT_ITERATIONS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="data/targets_500.pt")
    p.add_argument("--soft", default="data/soft_prompts_K32.pt")
    p.add_argument("--projected", default="data/projected_K32.pt")
    p.add_argument("--out", default="data/eval_K32.pt")
    p.add_argument("--max_new", type=int, default=24)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--mode", choices=["text", "latent_via_text"], default="text",
                   help="text: feed [q, projected] then generate. "
                        "latent_via_text: feed [q, bocot, projected, eocot] then generate "
                        "(treats projected tokens as text replacement for the latent reasoning).")
    return p.parse_args()


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
    args = parse_args()
    device = xm.xla_device()

    print("loading model...")
    model, _prj, tok = load_codi(dtype=torch.bfloat16, device="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    embed_layer = model.get_input_embeddings()

    targets = torch.load(args.targets, weights_only=False)
    proj = torch.load(args.projected, weights_only=False)

    input_ids_with_bot = targets["input_ids"]  # (N, q+1)
    attn_mask_with_bot = targets["attention_mask"]
    q_only_ids = input_ids_with_bot[:, :-1]  # drop bot
    q_only_mask = attn_mask_with_bot[:, :-1]
    questions = targets["questions"]
    gold_answers = targets["gold_answers"]
    codi_pred_answers = targets["pred_answers"]
    codi_pred_texts = targets["pred_texts"]

    projected_ids = proj["projected_ids"]  # (N, K)
    K = proj["K"]
    N = projected_ids.size(0)
    # truncate targets to N (some K-sweep runs use fewer problems)
    q_only_ids = q_only_ids[:N]
    q_only_mask = q_only_mask[:N]
    questions = questions[:N]
    gold_answers = gold_answers[:N]
    codi_pred_answers = codi_pred_answers[:N]
    codi_pred_texts = codi_pred_texts[:N]
    print(f"N={N} K={K}")

    # Build the input sequence per problem.
    if args.mode == "text":
        # [q, projected_tokens]
        full_ids = torch.cat([q_only_ids, projected_ids], dim=1)
        # Need attention mask: existing q mask, then K ones for the projected tokens
        full_mask = torch.cat([q_only_mask, torch.ones(N, K, dtype=q_only_mask.dtype)], dim=1)
    else:  # latent_via_text
        bot_col = torch.full((N, 1), BOT_ID, dtype=torch.long)
        eot_col = torch.full((N, 1), EOT_ID, dtype=torch.long)
        full_ids = torch.cat([q_only_ids, bot_col, projected_ids, eot_col], dim=1)
        ones1 = torch.ones(N, 1, dtype=q_only_mask.dtype)
        full_mask = torch.cat(
            [q_only_mask, ones1, torch.ones(N, K, dtype=q_only_mask.dtype), ones1], dim=1
        )
    seq_len = full_ids.size(1)
    print(f"input seq len: {seq_len}")

    # Greedy generation, no kv cache (matches step1's approach).
    pred_texts = [""] * N
    pred_nums = [float("inf")] * N

    n_batches = (N + args.batch - 1) // args.batch
    for bi in range(n_batches):
        s, e = bi * args.batch, min((bi + 1) * args.batch, N)
        bs = e - s
        b_ids = full_ids[s:e].to(device)
        b_mask = full_mask[s:e].to(device)
        with torch.no_grad():
            embeds = embed_layer(b_ids).to(torch.bfloat16)  # (bs, seq, H)
            mask = b_mask.clone()
            generated_ids = torch.zeros((bs, args.max_new), dtype=torch.long, device=device)
            for gi in range(args.max_new):
                out = model(
                    inputs_embeds=embeds,
                    attention_mask=mask,
                    use_cache=False,
                )
                logits = out.logits[:, -1, : model.config.vocab_size - 1]
                next_tok = logits.argmax(dim=-1)
                generated_ids[:, gi] = next_tok
                next_emb = embed_layer(next_tok).to(torch.bfloat16).unsqueeze(1)
                embeds = torch.cat([embeds, next_emb], dim=1)
                mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            xm.mark_step()
        gen_cpu = generated_ids.detach().to("cpu")
        eos_id = tok.eos_token_id
        for i in range(bs):
            ids = gen_cpu[i].tolist()
            if eos_id in ids:
                ids = ids[: ids.index(eos_id)]
            text = tok.decode(ids, skip_special_tokens=True)
            pred_texts[s + i] = text
            pred_nums[s + i] = extract_answer_number(text)
        print(
            f"batch {bi+1}/{n_batches}  ({s}..{e-1})  sample={pred_texts[s][:60]!r}",
            flush=True,
        )

    # Score:
    #   - vs ground truth
    #   - vs CODI's own greedy prediction
    n_vs_gold = sum(1 for k in range(N) if pred_nums[k] == gold_answers[k])
    n_vs_codi = sum(1 for k in range(N) if pred_nums[k] == codi_pred_answers[k])
    print(f"\naccuracy vs gold: {n_vs_gold}/{N} = {100*n_vs_gold/N:.2f}%")
    print(f"agreement with CODI greedy: {n_vs_codi}/{N} = {100*n_vs_codi/N:.2f}%")

    # Stratify CODI's correctness vs text-replay's match
    codi_correct = [k for k in range(N) if codi_pred_answers[k] == gold_answers[k]]
    codi_wrong = [k for k in range(N) if codi_pred_answers[k] != gold_answers[k]]
    rt_match_when_codi_correct = sum(1 for k in codi_correct if pred_nums[k] == codi_pred_answers[k])
    rt_match_when_codi_wrong = sum(1 for k in codi_wrong if pred_nums[k] == codi_pred_answers[k])
    if codi_correct:
        print(
            f"  agreement when CODI was correct: {rt_match_when_codi_correct}/{len(codi_correct)} "
            f"= {100*rt_match_when_codi_correct/len(codi_correct):.2f}%"
        )
    if codi_wrong:
        print(
            f"  agreement when CODI was wrong: {rt_match_when_codi_wrong}/{len(codi_wrong)} "
            f"= {100*rt_match_when_codi_wrong/len(codi_wrong):.2f}%"
        )

    payload = {
        "pred_texts": pred_texts,
        "pred_answers": pred_nums,
        "codi_pred_answers": codi_pred_answers,
        "gold_answers": gold_answers,
        "mode": args.mode,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
