# coconut-cot-gen

Can we recover readable chain-of-thought text from the continuous latent
reasoning states of a CODI model? This repo runs the full experiment on
`bcywinski/codi_llama1b-answer_only` (CODI-trained Llama-3.2-1B) against
GSM8K.

**Short answer**: no, not with any of the approaches here. Reconstruction
of the latent state is easy (cos ≈ 0.99 at K=32 soft embeddings), but the
nearest-token projection of those embeddings is adversarial gibberish, and
none of the regularizers we tried restore behavioral equivalence. Along
the way we found that CODI's latent reasoning only contributes ~7pp of
accuracy over a structural-wrapper baseline that uses *no* reasoning
content at all.

---

## Setup

- **Hardware**: TPU v5litepod-64 (16 worker hosts × 4 v5e chips each),
  `europe-west4`. All compute runs via PyTorch/XLA 2.5.
- **Model**: `bcywinski/codi_llama1b-answer_only` (CODI paper: *Compressing
  Chain-of-Thought into Continuous Space*, EMNLP 2025). LoRA is merged
  into base weights at load time; the projection layer is kept separate.
- **Dataset**: GSM8K main test split, first 512 problems.
- **Base Llama config**: pulled from
  `unsloth/Llama-3.2-1B-Instruct/config.json` (the meta original is gated,
  unsloth is unrestricted). The CODI checkpoint ships with the full set of
  weights so nothing else is needed from meta.

## Running

One-time setup (installs torch/torch_xla/transformers, downloads model):

```bash
bash scripts/setup_worker.sh
```

Then the pipeline:

```bash
# Step 1: collect h_target for 512 GSM8K problems + CODI's greedy answer
python scripts/step1_collect_targets.py --n 512 --batch 32 --q_max 192 \
    --max_new 24 --out data/targets_500.pt

# Step 2: per-problem soft prompt optimization
python scripts/step2_soft_prompt_opt.py --targets data/targets_500.pt \
    --out data/soft_prompts_K32.pt --K 32 --steps 500 --lr 1e-2 --batch 16

# Step 3: project each e_i to the nearest token
python scripts/step3_project_to_tokens.py --soft data/soft_prompts_K32.pt \
    --targets data/targets_500.pt --out data/projected_K32.pt

# Step 4: evaluate behavioral equivalence (feed projected tokens back as text)
python scripts/step4_evaluate.py --targets data/targets_500.pt \
    --projected data/projected_K32.pt --out data/eval_K32_text.pt \
    --mode text --batch 32 --max_new 24
python scripts/step4_evaluate.py --targets data/targets_500.pt \
    --projected data/projected_K32.pt --out data/eval_K32_latentviatext.pt \
    --mode latent_via_text --batch 32 --max_new 24

# Baselines (no soft prompts — just the question, or [q, bocot, eocot])
python scripts/baseline_eval.py --mode empty_latent --out data/baseline_empty_latent.pt
python scripts/baseline_eval.py --mode question_only --out data/baseline_qonly.pt

# A + B: cosine histogram, decoded sample transcripts
python scripts/step4_analysis.py --soft data/soft_prompts_K32.pt \
    --projected data/projected_K32.pt --targets data/targets_500.pt \
    --out outputs/analysis_K32.txt

# Aggregate final report
python scripts/final_report.py    # writes outputs/final_report.txt
```

### Multi-host parallelism

`scripts/workers.txt` maps the 16 worker IPs → names. `launch_on_workers.sh`
runs `setup_worker.sh` on all of them in parallel. The LM-prior sweep
(`sweep_lm_prior.sh`) is an example of dispatching one config to each of 4
workers in parallel; `eval_sweep_all.sh` does the same for the behavioral
evaluations. Each worker runs independently on its own 4 chips — there is
no XLA multi-host SPMD, just embarrassingly-parallel job distribution.

## Pipeline detail

### Step 1 — collect target activations

For each problem we tokenize `[question_tokens, <|bocot|>]`, run one
forward pass for question encoding, then six latent-reasoning forward
passes where each step appends `prj(h_last)` to the sequence (matching
CODI's `test.py`). `h_target` is the last-layer residual stream at the
last position of the SIXTH latent forward — the state "right before text
generation begins" in the CODI loop.

On TPU we avoid kv-cache length changes by rebuilding the full forward
each step; only seven unique seq lengths means only seven XLA
compilations per batch shape.

After collecting `h_target`, the same pass continues through `max_new=24`
greedy generation steps so we can record CODI's own answer for
stratification by correctness. CODI greedy accuracy on the first 512
problems: **38.87%** (199/512).

### Step 2 — soft prompt optimization

Input sequence: `[question_tokens, e_1, …, e_K]` — **no** `<|bocot|>` /
`<|eocot|>` tokens, `e_i` are learnable. Minimize
`1 - cos(h_output, h_target)` where `h_output` is the last-layer residual
stream at the K-th soft position. Adam, 500 steps, lr 1e-2. Each batch of
16 problems is ~110s of TPU time (after the first XLA compile).

### Step 3 — nearest-token projection

`argmax_v cos(e_i, embed_v)`. Optionally excludes special/reserved
tokens (everything at vocab id ≥ 128000).

### Step 4 — evaluation

- **A. Reconstruction**: histogram of final cosine, stratified by CODI
  correctness. Written in `outputs/analysis_K32.txt` and
  `outputs/final_report.txt`.
- **B. Semantic coherence**: print 50 decoded projections.
- **C. Behavioral equivalence**: feed the projected token sequence back
  in as text and compare to CODI's own greedy answer. Two modes:
  - `text`: `[question, projected_tokens]`
  - `latent_via_text`: `[question, <|bocot|>, projected_tokens, <|eocot|>]`
    — wraps the projected tokens in the structural markers CODI was
    trained with.

## Results

### A. Reconstruction quality

Soft prompts can hit the target reliably:

| K  | N   | mean   | median | min    | >0.95 | >0.99 |
|----|-----|--------|--------|--------|-------|-------|
| 16 | 128 | 0.9859 | 0.9986 | 0.0442 | 94.5% | 86.7% |
| 32 | 512 | 0.9833 | 0.9993 | 0.1060 | 96.9% | 90.4% |
| 64 | 128 | 0.9991 | 0.9996 | 0.9736 | 100%  | 100%  |

K=64 saturates at cos≈1.0 for every problem.

### B. Nearest-token projection (no prior)

```
K=32 best  (cos=0.9997):  actually|adar|275| ~|/u| altercation|<|begin_of_text|>
                          |.| Electric| مقاو|}| Evo|IALIZ| Permission|...
K=32 worst (cos=0.1060):  |<|begin_of_text|>|('//*[@| grocery|FirstChild|ine|	f
                          |ories| resumes| 國|...
```

Rare unicode, code fragments, random special tokens. No math, no
numbers from the problem, no intermediate calculations. Classic Goodhart.

### C. Behavioral equivalence (first 512)

| Method | gold% | agree w/ CODI |
|---|---|---|
| CODI (real latent reasoning, reference) | **38.87** | 100 |
| `[q, <|bocot|>, <|eocot|>]` empty wrapper | **31.64** | **63.87** |
| question only `[q]` | 25.78 | 39.06 |
| K=32 text `[q, projected]` | 6.84 | 10.55 |
| K=32 latent_via_text `[q, bocot, projected, eocot]` | 24.41 | 38.09 |

**Big finding**: the empty structural wrapper `[q, <|bocot|>, <|eocot|>]`
with **no reasoning content whatsoever** already recovers 63.87% of
CODI's own answers and 31.64% gold accuracy. That means CODI's 6 latent
reasoning tokens contribute only about **7pp** of gold accuracy over the
structural wrapper baseline.

Adding our projected K=32 tokens *between* `<|bocot|>` and `<|eocot|>`
makes things **worse**, not better: 38.1% agreement instead of 63.9%. The
projected tokens are adversarial noise to the model.

### K sweep — inverse scaling

On the first 128 problems, `latent_via_text` mode:

| K  | mean cos | gold% | agree% | agree\|CODI_correct | agree\|CODI_wrong |
|----|----------|-------|--------|---------------------|--------------------|
| 16 | 0.986    | 31.25 | 53.91  | 70.59               | 42.86              |
| 32 | 0.986    | 28.12 | 41.41  | 64.71               | 25.97              |
| 64 | 0.999    | 22.66 | 32.81  | 49.02               | 22.08              |

**K=64 has the best reconstruction cosine but the WORST behavioral
equivalence.** More soft-embedding capacity lets the optimizer find
solutions further off the token-reachable manifold, so the nearest-token
projection is lossier.

### LM prior (user-suggested fix for Goodhart)

`scripts/step2_lm_prior.py` adds an auto-regressive LM loss: a fresh base
Llama-3.2-1B (loaded from the same checkpoint without LoRA) is fed the
exact same `[q, e_1, …, e_K]` sequence, and each `e_i` is treated as a
soft token distribution `softmax(sharpness * cos(e_i, embed_v))`. The LM
loss is
`- Σ_i Σ_v p_i(v) * log P_ref(v | q, e_<i)`.

Swept `lm_weight ∈ {0.1, 0.3, 1.0, 3.0}` on 64 problems, one worker per
config, `sharpness=50`, specials masked:

| lm_w | mean cos | gold% (lvt) | agree% (lvt) |
|------|----------|-------------|--------------|
| no prior | 0.9865 | 28.12 | 41.41 |
| 0.1  | 0.9730 | 21.88 | 37.50 |
| 0.3  | 0.9652 | 14.06 | 31.25 |
| 1.0  | 0.8755 | 20.31 | 31.25 |
| 3.0  | 0.8512 | 12.50 | 32.81 |

Behavioral equivalence is **not improved** by the LM prior — in fact it
drops slightly vs no-prior.

But the tokens *look* very different. Best projected sample at lm_w=0.1:

```
,...\n. the,. a,, and,.. of\n..,.\n a. the,.,,.
```

Stopwords and punctuation instead of unicode garbage. The LM prior
successfully moves the tokens onto the natural-text manifold — just onto
its *lowest-surprise* region (filler words), not onto any region with
semantic reasoning content.

**Why the LM prior fails**: it is self-referential. The reference LM is
conditioned on the preceding soft embeddings `e_<i`, which are *also*
adversarial during optimization. Given a weird leading context, common
fillers like `.` and `the` are always the minimum-surprise answer. The
optimizer ends up finding a filler sequence whose hidden-state trajectory
through the CODI model accidentally lands at `h_target`.

See `outputs/lm_prior_report.txt` for the full write-up.

## Interpretation

1. **Latent states are easy to hit** — the target manifold is reachable
   by K=16+ soft embeddings for almost every problem.
2. **Nearest-token projection is lossy** — the soft embeddings live in
   an off-token-manifold region that gets mapped to adversarial tokens.
3. **Naïve priors don't fix the lossiness** — token-proximity would still
   allow arbitrary token sequences, and a self-referential LM prior
   collapses into filler text.
4. **Most of CODI's GSM8K behavior is elicited by the structural
   `<|bocot|>`/`<|eocot|>` markers**, not by the content of the 6 latent
   reasoning positions. The latent content is worth only ~7pp of
   accuracy. This is the most surprising single finding.

The remaining question — whether ~7pp of latent-reasoning content is in
principle text-reachable — is unresolved. Approaches that might work but
we didn't try:

- **Discrete beam search** over token sequences, directly optimizing
  reconstruction cosine. Stays in token space by construction.
- **Teacher-forced STE**: at each step, argmax-project the soft
  embeddings, feed the *hard* token embeddings into the reference LM,
  compute auto-regressive log-prob, and backprop via straight-through.
- **Behavioral target instead of state target**: optimize to match
  CODI's final answer logits rather than the intermediate hidden state.

## Files

```
scripts/
  codi_loader.py              # load CODI + base Llama, merge LoRA, attach projection
  check_tpu.py                # minimal TPU ping
  step1_collect_targets.py    # run CODI → save h_target + answers
  step2_soft_prompt_opt.py    # soft prompt opt (with optional token proximity reg)
  step2_lm_prior.py           # variant with autoregressive LM prior loss
  step3_project_to_tokens.py  # argmax projection to nearest token
  step4_evaluate.py           # behavioral eval: feed tokens back, check answer
  step4_analysis.py           # cosine histogram + decoded samples
  baseline_eval.py            # [q] and [q, bocot, eocot] baselines
  final_report.py             # aggregate all main-experiment numbers
  sweep_report.py             # K-sweep summary (cos only)
  evaluate_lmprior.py         # behavioral eval for LM-prior outputs
  slice_K32_to_128.py         # slice full K=32 results to first 128 for fair comparison

  # multi-host infra
  workers.txt                 # IP ↔ worker-N mapping for the 16-host pod
  setup_worker.sh             # install deps + download model on a worker
  launch_on_workers.sh        # parallel setup across all 15 non-primary hosts
  run_on_worker.sh            # single-worker wrapper (git pull + command)
  sweep_lm_prior.sh           # dispatch an lm_w config to each of 4 workers
  eval_sweep_all.sh           # dispatch behavioral eval per worker
  collect_sweep.sh, wait_sweep.sh, wait_evals.sh  # collection helpers

outputs/
  final_report.txt            # main-experiment write-up (K sweep, baselines)
  analysis_K32.txt            # cosine histogram + 50 decoded samples at K=32
  lm_prior_report.txt         # LM-prior sweep write-up
  sweep_evals/                # behavioral-eval logs for each LM-prior config
```

Data artifacts (`data/*.pt`) are ignored by git; regenerate from the
scripts above or pull from the worker that ran them.
