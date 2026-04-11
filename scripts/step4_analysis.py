"""Step 4 A & B: cosine histogram + sample of decoded transcripts.

Writes a text report under outputs/.
"""
import os
import sys
import argparse
import math
import json

import torch

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--soft", default="data/soft_prompts_K32.pt")
    p.add_argument("--projected", default="data/projected_K32.pt")
    p.add_argument("--targets", default="data/targets_500.pt")
    p.add_argument("--out", default="outputs/analysis_K32.txt")
    p.add_argument("--n_samples", type=int, default=50)
    return p.parse_args()


def histogram_str(values, n_bins=20, lo=None, hi=None, width=40):
    if lo is None:
        lo = float(values.min())
    if hi is None:
        hi = float(values.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for v in values.tolist():
        b = min(int((v - lo) / (hi - lo) * n_bins), n_bins - 1)
        if b < 0:
            b = 0
        counts[b] += 1
    max_c = max(counts) or 1
    lines = []
    for i in range(n_bins):
        bar = "#" * int(counts[i] / max_c * width)
        lines.append(f"  [{edges[i]:6.3f}, {edges[i+1]:6.3f}) {counts[i]:5d} | {bar}")
    return "\n".join(lines)


def main():
    args = parse_args()
    sp = torch.load(args.soft, weights_only=False)
    proj = torch.load(args.projected, weights_only=False)
    tgt = torch.load(args.targets, weights_only=False)

    final_cos = sp["final_cos"]
    K = sp["K"]
    N = final_cos.size(0)

    decoded = proj["decoded"]
    questions = tgt["questions"]
    gold = tgt["gold_answers"]
    pred = tgt["pred_answers"]

    lines = []
    lines.append(f"== STEP 4 ANALYSIS  (K={K}, N={N}) ==\n")
    lines.append(f"final cosine similarity stats:")
    lines.append(f"  mean   = {final_cos.mean().item():.4f}")
    lines.append(f"  median = {final_cos.median().item():.4f}")
    lines.append(f"  std    = {final_cos.std().item():.4f}")
    lines.append(f"  min    = {final_cos.min().item():.4f}")
    lines.append(f"  max    = {final_cos.max().item():.4f}")
    lines.append(f"  q25    = {torch.quantile(final_cos.float(), 0.25).item():.4f}")
    lines.append(f"  q75    = {torch.quantile(final_cos.float(), 0.75).item():.4f}")
    lines.append(f"  > 0.95 = {(final_cos > 0.95).float().mean().item()*100:.1f}%")
    lines.append(f"  > 0.99 = {(final_cos > 0.99).float().mean().item()*100:.1f}%")
    lines.append("")
    lines.append("HISTOGRAM (final cosine):")
    lines.append(histogram_str(final_cos.float(), n_bins=20, lo=0.0, hi=1.0))
    lines.append("")

    # Stratify by CODI correctness
    codi_correct = torch.tensor([1 if pred[k] == gold[k] else 0 for k in range(N)])
    cc = final_cos[codi_correct == 1]
    cw = final_cos[codi_correct == 0]
    lines.append(f"\nstratified by CODI greedy correctness:")
    lines.append(f"  CODI correct (n={cc.numel()}): mean cos = {cc.mean().item():.4f}")
    lines.append(f"  CODI wrong   (n={cw.numel()}): mean cos = {cw.mean().item():.4f}")
    lines.append("")

    # Sample decoded
    n_samples = min(args.n_samples, N)
    # Sort by cosine descending; take a stratified sample
    order = torch.argsort(final_cos, descending=True)
    spread = [int(order[int(i * (N - 1) / (n_samples - 1))]) for i in range(n_samples)]
    seen = set()
    sample_idx = []
    for i in spread:
        if i not in seen:
            seen.add(i)
            sample_idx.append(i)

    lines.append(f"\n== {len(sample_idx)} SAMPLE TRANSCRIPTS (sorted by cosine) ==\n")
    for i in sample_idx:
        lines.append(f"--- problem {i}  cos={final_cos[i].item():.4f} ---")
        lines.append(f"Q: {questions[i][:200]}")
        lines.append(f"GOLD: {gold[i]}  CODI_PRED: {pred[i]}")
        lines.append(f"PROJ: {decoded[i]}")
        lines.append("")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(text[:3000])
    print(f"\n... wrote {args.out}")


if __name__ == "__main__":
    main()
