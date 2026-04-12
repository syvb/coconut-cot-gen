"""Patchscopes: patch h_target into a natural prompt and see what the model generates.

For each of a handful of natural-language "target prompts", we replace one token's
input embedding with h_target and let the model continue generating. The idea is:
if h_target encodes readable information, the model's own decoding pathway should
produce meaningful text once it's nudged to interpret the state.

Two source models to compare:
  (1) CODI (LoRA-merged) — the model that produced h_target in the first place
  (2) base Llama — a neutral interpreter

Several prompt templates, patching at different positions:
  A. logit lens:     apply lm_head directly to h_target (no generation)
  B. single token:   inputs_embeds = [h_target]; generate
  C. rephrase:       "In other words, this means: [H] "
  D. solution:       "{question}\nSolution: [H] "
  E. few-shot:       "Problem: X. Solution: Y.\nProblem: Z. Solution: [H] "
  F. CODI native:    [question, <|bocot|>, [H], <|eocot|>] -- treat h_target as
                     the single "latent" token between CODI's structural markers
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
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(__file__))
from codi_loader import load_codi, load_base_llama, BOT_ID, EOT_ID, PAD_ID


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="data/targets_500.pt")
    p.add_argument("--n", type=int, default=16, help="number of problems to patch")
    p.add_argument("--max_new", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", default="outputs/patchscopes_results.txt")
    p.add_argument("--model", choices=["codi", "base"], default="codi")
    p.add_argument("--h_mode", choices=["real", "zero", "random", "shuffled"],
                   default="real",
                   help="ablation for the patched vector: real=h_target, zero=zeros, "
                        "random=random gaussian, shuffled=next problem's h_target")
    return p.parse_args()


def tokenize_template(tokenizer, template: str, patch_marker: str = "[[H]]"):
    """Return (left_ids, right_ids) -- tokens before and after the patch marker."""
    assert patch_marker in template, f"template must contain {patch_marker}"
    left_str, right_str = template.split(patch_marker)
    left = tokenizer(left_str, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    # right side should not re-add BOS
    right = tokenizer(right_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    return left, right


def build_patched_inputs(tokenizer, template: str, h_target_batch: torch.Tensor,
                          q_text_batch=None, embed_layer=None, device=None):
    """Build (inputs_embeds, attention_mask, patch_pos) for a batch.

    h_target_batch: (B, H)
    q_text_batch: optional list of questions to substitute into the template
                  via a {q} placeholder before tokenization
    """
    B = h_target_batch.size(0)
    rendered = []
    for i in range(B):
        if q_text_batch is not None and "{q}" in template:
            rendered.append(template.replace("{q}", q_text_batch[i]))
        else:
            rendered.append(template)

    # Tokenize each per-problem (questions differ in length). Keep per-problem
    # left/right split around the [[H]] marker.
    left_ids_list, right_ids_list = [], []
    for s in rendered:
        left, right = tokenize_template(tokenizer, s)
        left_ids_list.append(left)
        right_ids_list.append(right)

    # Left-pad to a common total length so we can batch.
    max_left = max(len(l) for l in left_ids_list)
    max_right = max(len(r) for r in right_ids_list)
    total_len = max_left + 1 + max_right  # +1 for the patched slot

    pad_id = PAD_ID
    input_ids = torch.full((B, total_len), pad_id, dtype=torch.long)
    mask = torch.zeros((B, total_len), dtype=torch.long)
    patch_pos = torch.zeros((B,), dtype=torch.long)
    for i, (l, r) in enumerate(zip(left_ids_list, right_ids_list)):
        lpad = max_left - len(l)  # left padding
        # place left tokens in [lpad .. lpad + len(l))
        input_ids[i, lpad:lpad + len(l)] = l
        mask[i, lpad:lpad + len(l)] = 1
        # patch position at max_left
        patch_pos[i] = max_left
        mask[i, max_left] = 1
        # right tokens after the patch
        if len(r) > 0:
            input_ids[i, max_left + 1:max_left + 1 + len(r)] = r
            mask[i, max_left + 1:max_left + 1 + len(r)] = 1

    input_ids = input_ids.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        embeds = embed_layer(input_ids).to(torch.bfloat16)
    # Replace embedding at patch_pos with h_target
    h = h_target_batch.to(device).to(torch.bfloat16)
    for i in range(B):
        embeds[i, patch_pos[i], :] = h[i]
    return embeds, mask, patch_pos, total_len


def greedy_generate(model, embed_layer, embeds, mask, max_new, vocab_size, eos_id):
    """Simple greedy generation by extending inputs_embeds each step."""
    B = embeds.size(0)
    generated_ids = torch.zeros((B, max_new), dtype=torch.long, device=embeds.device)
    for gi in range(max_new):
        out = model(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
        logits = out.logits[:, -1, :vocab_size - 1]  # exclude PAD
        next_tok = logits.argmax(dim=-1)
        generated_ids[:, gi] = next_tok
        next_emb = embed_layer(next_tok).to(torch.bfloat16).unsqueeze(1)
        embeds = torch.cat([embeds, next_emb], dim=1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
    xm.mark_step()
    return generated_ids


def decode_batch(tokenizer, generated_ids, eos_id):
    out = []
    for i in range(generated_ids.size(0)):
        ids = generated_ids[i].tolist()
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out


def extract_answer_number(s: str):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return float("inf")
    try:
        return float(nums[-1])
    except ValueError:
        return float("inf")


def score_preds(pred_texts, gold_answers, codi_preds):
    """Return dict with gold%/codi% and stratified stats."""
    N = len(pred_texts)
    preds = [extract_answer_number(t) for t in pred_texts]
    n_gold = sum(1 for k in range(N) if preds[k] == gold_answers[k])
    n_codi = sum(1 for k in range(N) if preds[k] == codi_preds[k])
    correct_idx = [k for k in range(N) if codi_preds[k] == gold_answers[k]]
    wrong_idx = [k for k in range(N) if codi_preds[k] != gold_answers[k]]
    a_c = sum(1 for k in correct_idx if preds[k] == codi_preds[k])
    a_w = sum(1 for k in wrong_idx if preds[k] == codi_preds[k])
    return {
        "gold_pct": 100 * n_gold / max(1, N),
        "codi_pct": 100 * n_codi / max(1, N),
        "agree_correct_pct": 100 * a_c / max(1, len(correct_idx)),
        "agree_wrong_pct": 100 * a_w / max(1, len(wrong_idx)),
        "n_correct_slice": len(correct_idx),
        "n_wrong_slice": len(wrong_idx),
    }


def logit_lens(model, tokenizer, h_targets, top_k=5):
    """Apply lm_head directly to h_target and report top-k tokens."""
    device = h_targets.device
    hidden = h_targets.to(torch.bfloat16).to(device)
    # Run through model.lm_head manually
    lm_head = model.get_output_embeddings()
    # lm_head is a Linear(hidden_size, vocab_size)
    logits = lm_head(hidden)  # (N, V)
    # Mask out specials >= 128000 to see more interesting content
    logits = logits.clone()
    logits[:, 128000:] = -float("inf")
    top = torch.topk(logits, top_k, dim=-1)
    results = []
    for i in range(h_targets.size(0)):
        toks = [tokenizer.decode([t.item()], skip_special_tokens=False)
                for t in top.indices[i]]
        scores = top.values[i].tolist()
        results.append(list(zip(toks, scores)))
    return results


def main():
    args = parse_args()
    device = xm.xla_device()

    # Target prompts (use [[H]] as the patch marker, and optionally {q})
    TEMPLATES = {
        "B_single": "[[H]]",
        "C_in_other_words": "\nIn other words, this means: [[H]] ",
        "D_solution": "{q}\nSolution: [[H]] ",
        "E_fewshot": (
            "Problem: Jim has 5 apples and buys 3 more. Solution: 5 + 3 = 8.\n"
            "Problem: A train travels 60 mph for 2 hours. Solution: 60 * 2 = 120.\n"
            "Problem: {q} Solution: [[H]] "
        ),
        "F_codi_native_1latent": "",  # handled specially
    }

    print(f"loading model: {args.model}")
    if args.model == "codi":
        model, _prj, tokenizer = load_codi(dtype=torch.bfloat16, device="cpu")
    else:
        model = load_base_llama(dtype=torch.bfloat16, device="cpu")
        _, _, tokenizer = load_codi(dtype=torch.bfloat16, device="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    embed_layer = model.get_input_embeddings()
    vocab_size = model.config.vocab_size

    print("loading targets:", args.targets)
    data = torch.load(args.targets, weights_only=False)
    h_targets_all = data["h_targets"][: args.n].to(device).to(torch.float32)
    # Apply ablation
    if args.h_mode == "zero":
        h_targets_all = torch.zeros_like(h_targets_all)
    elif args.h_mode == "random":
        # Match the norm of real h_targets (~105)
        real_norm = torch.norm(data["h_targets"][: args.n], dim=-1).mean().item()
        g = torch.randn_like(h_targets_all)
        g = g / g.norm(dim=-1, keepdim=True) * real_norm
        h_targets_all = g
    elif args.h_mode == "shuffled":
        # Swap each with the next problem's h_target (circular)
        h_targets_all = torch.cat([h_targets_all[1:], h_targets_all[:1]], dim=0)
    questions = data["questions"][: args.n]
    gold_answers = data["gold_answers"][: args.n]
    codi_preds = data["pred_answers"][: args.n]
    N = h_targets_all.size(0)

    eos_id = tokenizer.eos_token_id
    print(f"N={N}  max_new={args.max_new}  eos={eos_id}")

    out_lines = []
    def emit(s=""):
        print(s, flush=True)
        out_lines.append(s)

    emit("=" * 72)
    emit(f"PATCHSCOPES RESULTS — model={args.model}")
    emit("=" * 72)

    # --- A: logit lens ---
    emit("\n--- A. LOGIT LENS (top-5 tokens from applying lm_head to h_target) ---")
    ll = logit_lens(model, tokenizer, h_targets_all, top_k=5)
    for i in range(min(N, 10)):
        tops_str = ", ".join(f"{t!r}" for t, _ in ll[i][:5])
        emit(f"  [{i}] gold={gold_answers[i]} codi={codi_preds[i]}: {tops_str}")

    # --- B through E: template-based patching ---
    def run_template(name, template, with_question):
        emit(f"\n--- {name}. template: {template!r} ---")
        pred_texts = [""] * N
        n_batches = (N + args.batch - 1) // args.batch
        for bi in range(n_batches):
            s, e = bi * args.batch, min((bi + 1) * args.batch, N)
            h_batch = h_targets_all[s:e]
            q_batch = questions[s:e] if with_question else None
            with torch.no_grad():
                embeds, mask, patch_pos, total_len = build_patched_inputs(
                    tokenizer, template, h_batch,
                    q_text_batch=q_batch, embed_layer=embed_layer, device=device,
                )
                gen = greedy_generate(
                    model, embed_layer, embeds, mask, args.max_new, vocab_size, eos_id,
                )
            gen_cpu = gen.to("cpu")
            decoded = decode_batch(tokenizer, gen_cpu, eos_id)
            for i, t in enumerate(decoded):
                pred_texts[s + i] = t
        for i in range(min(N, 10)):
            emit(f"  [{i}] gold={gold_answers[i]} codi={codi_preds[i]}: {pred_texts[i][:200]!r}")
        stats = score_preds(pred_texts, gold_answers, codi_preds)
        emit(
            f"  STATS: gold={stats['gold_pct']:.1f}% codi_agree={stats['codi_pct']:.1f}% "
            f"| agree|correct={stats['agree_correct_pct']:.1f}% "
            f"(n={stats['n_correct_slice']}) "
            f"agree|wrong={stats['agree_wrong_pct']:.1f}% "
            f"(n={stats['n_wrong_slice']})"
        )
        return pred_texts

    run_template("B", TEMPLATES["B_single"], with_question=False)
    run_template("C", TEMPLATES["C_in_other_words"], with_question=False)
    run_template("D", TEMPLATES["D_solution"], with_question=True)
    run_template("E", TEMPLATES["E_fewshot"], with_question=True)

    # --- F. CODI native: [q, bocot, h_target, eocot], generate ---
    emit("\n--- F. CODI native format [q, <|bocot|>, h_target, <|eocot|>] ---")
    # Tokenize questions (no patch marker, just natural CODI input shape)
    input_ids = data["input_ids"][: args.n]  # (N, q_max+1) — includes bot at end
    q_mask_all = data["attention_mask"][: args.n]
    q_only_ids = input_ids[:, :-1]  # drop bot
    q_only_mask = q_mask_all[:, :-1]
    q_max = q_only_ids.size(1)
    pred_texts_f = [""] * N
    n_batches = (N + args.batch - 1) // args.batch
    for bi in range(n_batches):
        s, e = bi * args.batch, min((bi + 1) * args.batch, N)
        bs = e - s
        q_ids = q_only_ids[s:e].to(device)
        q_mask = q_only_mask[s:e].to(device)
        h_batch = h_targets_all[s:e]
        with torch.no_grad():
            q_embeds = embed_layer(q_ids).to(torch.bfloat16)  # (bs, q_max, H)
            bot_embed = embed_layer.weight[BOT_ID].detach().to(torch.bfloat16)
            eot_embed = embed_layer.weight[EOT_ID].detach().to(torch.bfloat16)
            bot_tok = bot_embed.view(1, 1, -1).expand(bs, 1, -1)
            eot_tok = eot_embed.view(1, 1, -1).expand(bs, 1, -1)
            h_tok = h_batch.to(torch.bfloat16).unsqueeze(1)  # (bs, 1, H)
            embeds = torch.cat([q_embeds, bot_tok, h_tok, eot_tok], dim=1)
            one = torch.ones(bs, 1, dtype=q_mask.dtype, device=device)
            mask = torch.cat([q_mask, one, one, one], dim=1)
            gen = greedy_generate(
                model, embed_layer, embeds, mask, args.max_new, vocab_size, eos_id,
            )
        gen_cpu = gen.to("cpu")
        decoded = decode_batch(tokenizer, gen_cpu, eos_id)
        for i, t in enumerate(decoded):
            pred_texts_f[s + i] = t
    for i in range(min(N, 10)):
        emit(f"  [{i}] gold={gold_answers[i]} codi={codi_preds[i]}: {pred_texts_f[i][:200]!r}")
    stats_f = score_preds(pred_texts_f, gold_answers, codi_preds)
    emit(
        f"  STATS: gold={stats_f['gold_pct']:.1f}% codi_agree={stats_f['codi_pct']:.1f}% "
        f"| agree|correct={stats_f['agree_correct_pct']:.1f}% "
        f"(n={stats_f['n_correct_slice']}) "
        f"agree|wrong={stats_f['agree_wrong_pct']:.1f}% "
        f"(n={stats_f['n_wrong_slice']})"
    )

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
