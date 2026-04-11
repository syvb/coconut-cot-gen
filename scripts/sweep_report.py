"""Compare all sweep outputs: LM prior across lm_weight values.

Reads data/sweep_collected/ (local on worker-0), for each file:
- Prints final cosine stats
- Runs behavioral evaluation (both modes) if the file is present
- Summarizes in a table
"""
import os
import sys
import glob
import subprocess
import re
import torch

sys.path.insert(0, os.path.dirname(__file__))


def main():
    files = sorted(glob.glob("data/sweep_collected/sweep_*.pt"))
    if not files:
        print("no sweep files found in data/sweep_collected/")
        return

    summary = []
    for f in files:
        d = torch.load(f, weights_only=False)
        fc = d["final_cos"]
        lmw = d.get("lm_weight")
        shp = d.get("lm_sharpness")
        K = d.get("K")
        N = fc.size(0)
        mean_cos = float(fc.mean())
        median_cos = float(fc.median())
        min_cos = float(fc.min())
        pct_095 = float((fc > 0.95).float().mean() * 100)
        row = {
            "file": os.path.basename(f),
            "lm_weight": lmw,
            "sharpness": shp,
            "K": K,
            "N": N,
            "mean_cos": mean_cos,
            "median_cos": median_cos,
            "min_cos": min_cos,
            "pct_gt_095": pct_095,
        }
        summary.append(row)

    print(f"{'file':<44}  {'lm_w':>5}  {'K':>3}  {'N':>3}  "
          f"{'mean':>6}  {'median':>7}  {'min':>6}  {'>0.95':>6}")
    for r in summary:
        print(
            f"{r['file']:<44}  {r['lm_weight']:>5.1f}  {r['K']:>3}  {r['N']:>3}  "
            f"{r['mean_cos']:>6.4f}  {r['median_cos']:>7.4f}  "
            f"{r['min_cos']:>6.4f}  {r['pct_gt_095']:>5.1f}%"
        )


if __name__ == "__main__":
    main()
