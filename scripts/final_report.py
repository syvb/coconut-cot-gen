"""Final report: aggregate all results into one summary."""
import torch
import os

OUT = "outputs/final_report.txt"


def stats(final_cos):
    return dict(
        mean=float(final_cos.mean()),
        median=float(final_cos.median()),
        min=float(final_cos.min()),
        max=float(final_cos.max()),
        q25=float(torch.quantile(final_cos.float(), 0.25)),
        q75=float(torch.quantile(final_cos.float(), 0.75)),
        pct_gt_095=float((final_cos > 0.95).float().mean() * 100),
        pct_gt_099=float((final_cos > 0.99).float().mean() * 100),
    )


def bar(vals, bins=20, lo=0.0, hi=1.0, width=50):
    counts = [0] * bins
    for v in vals.tolist():
        b = min(int((v - lo) / (hi - lo) * bins), bins - 1)
        counts[max(0, b)] += 1
    maxc = max(counts) or 1
    out = []
    for i in range(bins):
        out.append(
            f"  [{lo+(hi-lo)*i/bins:5.3f}, {lo+(hi-lo)*(i+1)/bins:5.3f}) "
            f"{counts[i]:4d} | {'#'*int(counts[i]/maxc*width)}"
        )
    return "\n".join(out)


def load_eval(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)


def correct_vs_gold(ev, gold):
    return sum(1 for k in range(len(ev["pred_answers"])) if ev["pred_answers"][k] == gold[k])


def agree_with_codi(ev, codi):
    return sum(1 for k in range(len(ev["pred_answers"])) if ev["pred_answers"][k] == codi[k])


def agree_stratified(ev, codi, gold):
    N = len(ev["pred_answers"])
    corr_idx = [k for k in range(N) if codi[k] == gold[k]]
    wrong_idx = [k for k in range(N) if codi[k] != gold[k]]
    a_c = sum(1 for k in corr_idx if ev["pred_answers"][k] == codi[k])
    a_w = sum(1 for k in wrong_idx if ev["pred_answers"][k] == codi[k])
    return a_c, len(corr_idx), a_w, len(wrong_idx)


lines = []
emit = lines.append

emit("=" * 70)
emit(" LATENT REASONING ORACLE: PIPELINE RESULTS")
emit("=" * 70)
emit("")
emit("Model: bcywinski/codi_llama1b-answer_only  (CODI on Llama-3.2-1B-Instruct)")
emit("Dataset: GSM8K test split, first 512 problems")
emit("Device: TPU v5litepod (single host = 4 v5e chips)")
emit("")

tgt = torch.load("data/targets_500.pt", weights_only=False)
gold = tgt["gold_answers"]
codi = tgt["pred_answers"]
N_all = len(gold)
codi_correct = sum(1 for k in range(N_all) if codi[k] == gold[k])

emit(f"CODI greedy baseline (reference model): {codi_correct}/{N_all} = "
     f"{100*codi_correct/N_all:.2f}% gold accuracy")
emit("")

# =====================================================================
# A. Reconstruction quality (cosine)
# =====================================================================
emit("=" * 70)
emit(" A.  SOFT-PROMPT RECONSTRUCTION QUALITY")
emit("=" * 70)
emit("")
emit("Each problem: K learnable embeddings optimized 500 steps with Adam lr=1e-2")
emit("to minimize 1 - cos(h_output, h_target).  h_target = last-layer residual")
emit("stream at the final latent reasoning position (i.e., CODI's state right")
emit("before text generation begins).")
emit("")
emit("                                  pct cos > ")
emit("   K    N    mean    median   min   0.95    0.99")

for K, path in [(16, "data/soft_prompts_K16.pt"),
                (32, "data/soft_prompts_K32.pt"),
                (64, "data/soft_prompts_K64.pt")]:
    sp = torch.load(path, weights_only=False)
    fc = sp["final_cos"]
    s = stats(fc)
    emit(f"  {K:3d}  {fc.numel():3d}  {s['mean']:.4f}  {s['median']:.4f}  "
         f"{s['min']:.4f}  {s['pct_gt_095']:5.1f}%  {s['pct_gt_099']:5.1f}%")

emit("")
emit("Histogram (K=32, N=512, bins 0.0-1.0):")
sp32 = torch.load("data/soft_prompts_K32.pt", weights_only=False)
emit(bar(sp32["final_cos"], bins=20, lo=0.0, hi=1.0))
emit("")
emit(">>> Conclusion A: the reconstruction target (CODI's post-reasoning hidden")
emit("    state) is reliably reachable from soft prompts. With K=32, 96.9% of")
emit("    problems reach cos>0.95; with K=64 every problem does (min 0.974).")
emit("    Soft prompts can find the point in activation space.")
emit("")

# =====================================================================
# B. Semantic coherence of the nearest-token projection
# =====================================================================
emit("=" * 70)
emit(" B.  NEAREST-TOKEN PROJECTION  (semantic coherence)")
emit("=" * 70)
emit("")
emit("Each soft vector is projected to the nearest token by cosine similarity")
emit("against the embed_tokens matrix. We inspect the decoded sequences.")
emit("")

def load_proj(p): return torch.load(p, weights_only=False)
proj16 = load_proj("data/projected_K16.pt")
proj32 = load_proj("data/projected_K32.pt")
proj64 = load_proj("data/projected_K64.pt")

emit("Sample decoded sequences (sorted by reconstruction cosine):")
emit("")

for K, pr in [(16, proj16), (32, proj32), (64, proj64)]:
    fc = pr["final_cos"]
    order = torch.argsort(fc, descending=True)
    emit(f"--- K={K} ---")
    emit("  Best (cos={:.4f}):  {}".format(
        fc[int(order[0])].item(), pr["decoded"][int(order[0])][:250]))
    emit("  Median (cos={:.4f}): {}".format(
        fc[int(order[len(order)//2])].item(),
        pr["decoded"][int(order[len(order)//2])][:250]))
    emit("  Worst (cos={:.4f}): {}".format(
        fc[int(order[-1])].item(), pr["decoded"][int(order[-1])][:250]))
    emit("")

emit(">>> Conclusion B: the projected transcripts are complete gibberish.")
emit("    No math-related tokens, numbers, or intermediate calculations are")
emit("    visible. The tokens are a mix of rare unicode, code fragments,")
emit("    special tokens like <|eocot|> and <|begin_of_text|>, and miscellaneous")
emit("    vocabulary noise. This is the classic Goodhart failure: the soft-prompt")
emit("    optimization is finding activation-space shortcuts that have essentially")
emit("    no connection to the tokens in the neighborhood of the solution.")
emit("")

# =====================================================================
# C. Behavioral equivalence
# =====================================================================
emit("=" * 70)
emit(" C.  BEHAVIORAL EQUIVALENCE")
emit("=" * 70)
emit("")
emit("We feed the projected token sequences back in as ACTUAL TEXT and compare")
emit("the model's answer to (a) gold, (b) CODI's own greedy answer.")
emit("")
emit("Two input formats:")
emit("  text           : [question, projected_tokens]  (no <|bocot|>/<|eocot|>)")
emit("  latent_via_text: [question, <|bocot|>, projected_tokens, <|eocot|>]")
emit("                   — matches CODI's training structure")
emit("")
emit("Baselines (what happens WITHOUT soft prompts):")
emit("")

b_empty = load_eval("data/baseline_empty_latent.pt")
b_qonly = load_eval("data/baseline_qonly.pt")
emit("   baseline                                  gold%   agree w/ CODI%")
emit(f"   CODI (real latent reasoning)              {100*codi_correct/N_all:5.2f}   "
     f"{100.00:5.2f}")
if b_empty is not None:
    n_gold = correct_vs_gold(b_empty, gold)
    n_codi = agree_with_codi(b_empty, codi)
    emit(f"   [q, <|bocot|>, <|eocot|>]                       {100*n_gold/N_all:5.2f}   "
         f"{100*n_codi/N_all:5.2f}   ← empty structural wrapper")
if b_qonly is not None:
    n_gold = correct_vs_gold(b_qonly, gold)
    n_codi = agree_with_codi(b_qonly, codi)
    emit(f"   [q] (question only)                       {100*n_gold/N_all:5.2f}   "
         f"{100*n_codi/N_all:5.2f}")

emit("")
emit("K sweep (first 128 problems, latent_via_text mode):")
emit("")
emit("   K    mean cos  gold%   agree%   agree|CODI_correct   agree|CODI_wrong")

for K, path in [(16, "data/eval_K16_lvt.pt"),
                (32, "data/eval_K32_first128_lvt.pt"),
                (64, "data/eval_K64_lvt.pt")]:
    ev = load_eval(path)
    if ev is None:
        continue
    N_k = len(ev["pred_answers"])
    gold_k = gold[:N_k]
    codi_k = codi[:N_k]
    n_gold = correct_vs_gold(ev, gold_k)
    n_codi = agree_with_codi(ev, codi_k)
    a_c, n_c, a_w, n_w = agree_stratified(ev, codi_k, gold_k)
    # look up mean cos for this K
    sp = torch.load(f"data/soft_prompts_K{K}.pt", weights_only=False)
    mean_cos = sp["final_cos"][:N_k].mean().item()
    emit(
        f"   {K:3d}    {mean_cos:.4f}   {100*n_gold/N_k:5.2f}   {100*n_codi/N_k:5.2f}   "
        f"{100*a_c/max(1,n_c):5.2f}  (n={n_c})       "
        f"{100*a_w/max(1,n_w):5.2f}  (n={n_w})"
    )

emit("")
emit("Full K=32 pipeline on all 512 problems:")
ev32_text = load_eval("data/eval_K32_text.pt")
ev32_lvt = load_eval("data/eval_K32_latentviatext.pt")
emit("   mode                   gold%   agree%   agree|CODI_correct   agree|CODI_wrong")
for name, ev in [("text            ", ev32_text), ("latent_via_text ", ev32_lvt)]:
    if ev is None:
        continue
    n_g = correct_vs_gold(ev, gold)
    n_c = agree_with_codi(ev, codi)
    a_c, n_c_all, a_w, n_w_all = agree_stratified(ev, codi, gold)
    emit(
        f"   {name}       {100*n_g/N_all:5.2f}   {100*n_c/N_all:5.2f}   "
        f"{100*a_c/max(1,n_c_all):5.2f}  (n={n_c_all})      "
        f"{100*a_w/max(1,n_w_all):5.2f}  (n={n_w_all})"
    )

emit("")
emit(">>> Conclusion C: behavioral equivalence FAILS across the board.")
emit("    Key findings:")
emit("")
emit("    1. The empty wrapper [q, <|bocot|>, <|eocot|>] already recovers 63.9%")
emit("       of CODI's answers and 31.6% gold accuracy.  Most of CODI's behavior")
emit("       comes from question + structural markers, not from the specific")
emit("       content of the latent reasoning positions.")
emit("")
emit("    2. Inserting the projected tokens between <|bocot|> and <|eocot|>")
emit("       performs WORSE than the empty wrapper (38% agree vs 64%). The")
emit("       projection is adversarial noise from the model's point of view.")
emit("")
emit("    3. Inverse scaling in K: higher K means MUCH better reconstruction")
emit("       cosine (K=64 reaches 0.999) yet WORSE behavioral equivalence")
emit("       (K=16 gold 31%, K=64 gold 23%). More capacity lets the optimizer")
emit("       find solutions further off the token-reachable manifold.")
emit("")
emit("    4. Without structural markers the projection is catastrophic: the")
emit("       K=32 'text' mode (just [q, projected]) gets only 6.8% gold.")
emit("")
emit("=" * 70)
emit(" OVERALL INTERPRETATION")
emit("=" * 70)
emit("")
emit("The user's diagnostic maps to: 'A high but C fails → the states are")
emit("text-reachable but nearest-token projection is lossy.'  We strongly")
emit("confirm this: reconstruction cosine saturates near 1.0, projected text is")
emit("pure noise, and behavior transfers poorly.")
emit("")
emit("More specifically, the INVERSE SCALING in K (lower K gives better")
emit("behavior) is diagnostic: with few soft embeddings the optimizer is")
emit("squeezed closer to the token-reachable manifold and the projection is")
emit("less lossy. This is consistent with a language-model prior regularizer")
emit("being the right fix (user's suggestion at the end of the spec).")
emit("")
emit("Additionally, the empty_latent baseline is a striking surprise: simply")
emit("tokenizing <|bocot|><|eocot|> around the question recovers 64% of CODI's")
emit("output behavior. This suggests CODI's latent reasoning contributes only")
emit("a modest ~7% absolute accuracy gain (31.6% → 38.9%) on top of the model's")
emit("zero-shot capability elicited by the trained structural markers. The")
emit("projection approach therefore has at most ~7 pts of 'content' to recover,")
emit("but loses most of it due to projection lossiness.")
emit("")

text = "\n".join(lines)
os.makedirs("outputs", exist_ok=True)
with open(OUT, "w") as f:
    f.write(text)
print(text)
print(f"\n[wrote {OUT}]")
