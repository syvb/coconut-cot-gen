"""Step 1: run CODI on GSM8K test, save residual stream targets + predictions.

Strategy for TPU friendliness:
- Pad all questions to a fixed Q_MAX length (left-pad).
- Run latent reasoning by RE-RUNNING the full forward each step (no kv cache).
- Run greedy generation also by re-running full forward, capped at MAX_NEW.
- This gives ~6 + MAX_NEW unique sequence lengths total, all shared across batches.
"""
import os
import sys
import json
import re
import time
import argparse

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_VISIBLE_DEVICES", "0,1,2,3")

import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(__file__))
from codi_loader import (
    load_codi,
    CodiProjection,
    PAD_ID,
    BOT_ID,
    EOT_ID,
    INF_LATENT_ITERATIONS,
)
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--q_max", type=int, default=256)
    p.add_argument("--max_new", type=int, default=24)
    p.add_argument("--out", type=str, default="data/targets.pt")
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


def gsm8k_gold_answer(text: str):
    if "####" in text:
        ans = text.split("####")[-1]
    else:
        ans = text
    ans = ans.replace(",", "").strip()
    try:
        return float(ans)
    except ValueError:
        return float("inf")


def main():
    args = parse_args()

    device = xm.xla_device()
    print("device:", device, flush=True)

    print("loading model...", flush=True)
    model, prj, tokenizer = load_codi(dtype=torch.bfloat16, device="cpu")
    model.eval()
    prj.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in prj.parameters():
        p.requires_grad_(False)

    print("moving model to TPU...", flush=True)
    model = model.to(device)
    prj = prj.to(device)
    xm.mark_step()
    print("model on TPU", flush=True)

    embed_layer = model.get_input_embeddings()

    # Dataset
    ds = load_dataset("gsm8k", "main", split="test")
    n = min(args.n, len(ds))
    ds = ds.select(range(n))
    questions = [ex["question"].strip().replace("  ", " ") for ex in ds]
    golds = [gsm8k_gold_answer(ex["answer"]) for ex in ds]

    # Pre-tokenize all questions; pad to a fixed length
    print(f"tokenizing {n} questions, q_max={args.q_max}", flush=True)
    enc = tokenizer(
        questions,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.q_max,
    )
    base_ids = enc["input_ids"]  # (n, q_max)
    base_mask = enc["attention_mask"]
    # Append <|bocot|> at fixed final position
    bot_col = torch.full((n, 1), BOT_ID, dtype=torch.long)
    one_col = torch.ones((n, 1), dtype=torch.long)
    base_ids = torch.cat([base_ids, bot_col], dim=1)  # (n, q_max+1)
    base_mask = torch.cat([base_mask, one_col], dim=1)
    print("token shape:", tuple(base_ids.shape), flush=True)

    # Output buffers
    h_targets = torch.zeros((n, model.config.hidden_size), dtype=torch.float32)
    pred_strs = [""] * n
    pred_nums = [float("inf")] * n

    # Pre-fetch eot embedding
    with torch.no_grad():
        eot_emb_cpu = embed_layer.weight[EOT_ID].detach().to("cpu")  # (H,)
    eot_emb_dev = eot_emb_cpu.to(device).to(torch.bfloat16)

    n_batches = (n + args.batch - 1) // args.batch
    for bi in range(n_batches):
        s, e = bi * args.batch, min((bi + 1) * args.batch, n)
        b_ids = base_ids[s:e].to(device)
        b_mask = base_mask[s:e].to(device)
        bs = e - s

        with torch.no_grad():
            embeds = embed_layer(b_ids).to(torch.bfloat16)  # (B, q+1, H)
            mask = b_mask.clone()

            # Latent reasoning loop. After question encoding, do INF_LATENT_ITERATIONS forwards.
            # First forward: question + bot.
            for it in range(INF_LATENT_ITERATIONS + 1):
                out = model(
                    inputs_embeds=embeds,
                    attention_mask=mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                h_last = out.hidden_states[-1][:, -1, :]  # (B, H)
                if it < INF_LATENT_ITERATIONS:
                    # Append projected latent for the next iteration
                    next_lat = prj(h_last).to(torch.bfloat16).unsqueeze(1)  # (B, 1, H)
                    embeds = torch.cat([embeds, next_lat], dim=1)
                    mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
                else:
                    # The output of the LAST forward (it == INF_LATENT_ITERATIONS) is h_target.
                    h_target = h_last  # (B, H), pre-projection
                    # Append eot embedding to start text generation
                    eot_batch = eot_emb_dev.view(1, 1, -1).expand(bs, 1, -1)
                    embeds = torch.cat([embeds, eot_batch], dim=1)
                    mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)

            # Greedy text generation. Each step extends embeds by 1.
            generated_ids = torch.zeros((bs, args.max_new), dtype=torch.long, device=device)
            for gi in range(args.max_new):
                out = model(
                    inputs_embeds=embeds,
                    attention_mask=mask,
                    output_hidden_states=False,
                    use_cache=False,
                )
                logits = out.logits[:, -1, : model.config.vocab_size - 1]  # exclude PAD
                next_tok = logits.argmax(dim=-1)  # (B,)
                generated_ids[:, gi] = next_tok
                next_emb = embed_layer(next_tok).to(torch.bfloat16).unsqueeze(1)
                embeds = torch.cat([embeds, next_emb], dim=1)
                mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            xm.mark_step()

        # Pull results to CPU
        h_target_cpu = h_target.detach().to("cpu").to(torch.float32)
        gen_cpu = generated_ids.detach().to("cpu")
        h_targets[s:e] = h_target_cpu

        eos_id = tokenizer.eos_token_id
        for i in range(bs):
            ids = gen_cpu[i].tolist()
            if eos_id in ids:
                ids = ids[: ids.index(eos_id)]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            pred_strs[s + i] = text
            pred_nums[s + i] = extract_answer_number(text)

        n_correct = sum(1 for k in range(s, e) if pred_nums[k] == golds[k])
        print(
            f"batch {bi+1}/{n_batches}  ({s}..{e-1})  correct={n_correct}/{bs}  "
            f"sample: {pred_strs[s][:60]!r}",
            flush=True,
        )

    # Final accuracy
    correct = sum(1 for k in range(n) if pred_nums[k] == golds[k])
    print(f"\noverall greedy accuracy: {correct}/{n} = {100*correct/n:.2f}%", flush=True)

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "h_targets": h_targets,  # (n, H), float32
        "questions": questions,
        "gold_answers": golds,
        "pred_answers": pred_nums,
        "pred_texts": pred_strs,
        "q_max": args.q_max,
        "input_ids": base_ids,  # includes bot at last position
        "attention_mask": base_mask,
    }
    torch.save(payload, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
