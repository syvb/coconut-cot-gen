"""The cleanest causal test of h_target's role in CODI.

Procedure:
 1. For each problem, run CODI's normal 6-iteration latent loop (matching
    step1 exactly) and record the final h_target AND the kv cache at the
    last position's residual stream.
 2. For each problem i, REPLACE the latent state at position q_max+6 with
    a DIFFERENT problem's h_target, then feed eot and greedy-generate.
 3. Check the generated answer.

If the generated answer changes when we swap h_target, then h_target is
causally meaningful at its native position. If it doesn't change, h_target
is effectively inert at its own position too.

Implementation detail: we can't easily swap h_target "in the middle" of a
kv-cached forward pass without modifying transformers internals. So instead
we re-run the full forward with a sequence in which we REPLACE the 6th-
latent's input embedding.

Specifically: run CODI's latent loop for problem i up to iter 4 (5 latents
appended), getting a sequence [q, bot, l0, l1, l2, l3, l4] and the hidden
state h_4. Compute the 6th-latent input as prj(h_4). Now feed the full
sequence [q, bot, l0..l4, prj(h_4)] plus eot, and extract the hidden state
at position q_max+6. That equals problem i's own h_target.

For the SWAP: replace prj(h_4) at position q_max+6 with prj(h_target_j)
from a DIFFERENT problem j. Actually that doesn't do what we want — it
feeds a different latent, and the model's output at that position is
processed through the full forward.

Better plan: do the ABOVE but instead of extracting the output, just let
the model continue from [q, bot, l0..l4, X, eot] where X is different
configurations:
  - X = prj(h_4)  (problem i's natural 6th-latent input) = "own native"
  - X = h_target_j from a different problem j (raw, not projected)
  - X = 0 vector (zero ablation)
  - X = random vector

Then greedy-generate and compare answers. This tells us whether CODI's
final answer is sensitive to the input at the 6th-latent position.

NOTE this is slightly different from injecting h_target ITSELF at that
position — h_target is the OUTPUT of the model at that position, not the
input that would normally be at that position. But for causal
interpretability, what we care about is: does the information at position
q_max+6 (as either input or output) matter for the final answer?
"""
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
import sys
import re
import argparse

import torch
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(__file__))
from codi_loader import load_codi, BOT_ID, EOT_ID, INF_LATENT_ITERATIONS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="data/targets_500.pt")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--max_new", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", default="outputs/patchscopes_swap.txt")
    return p.parse_args()


def extract_answer_number(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return float("inf")
    try:
        return float(nums[-1])
    except ValueError:
        return float("inf")


def build_natural_reasoning_inputs(model, prj, q_ids, q_mask, embed_layer):
    """Run 5 latent iters (the first 5 of CODI's 6) to build [q, bot, l0..l4]
    as inputs_embeds. Returns (embeds, mask, h_target_native) where
    h_target_native is the final-layer hidden state at the 6th latent position
    if we continued one more step (i.e., at position q_max+6) using prj(h_4)
    as the 6th latent input.
    """
    B, q_max = q_ids.shape  # q_max here is the length AFTER dropping bot (so no bot yet)
    # Our q_ids already has bot appended in targets file — but for this test we
    # want a clean rebuild. Actually the targets file stores input_ids WITH bot.
    # For flexibility, just use it as-is; our "q_embeds" already contains the bot
    # at the last position.
    embeds = embed_layer(q_ids).to(torch.bfloat16)  # (B, q_max, H) — includes bot at last pos
    mask = q_mask.clone()
    # 5 forward passes + append to build up to l4
    for it in range(INF_LATENT_ITERATIONS - 1):  # 5 iterations
        out = model(
            inputs_embeds=embeds,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h_last = out.hidden_states[-1][:, -1, :]  # (B, H)
        next_lat = prj(h_last).to(torch.bfloat16).unsqueeze(1)
        embeds = torch.cat([embeds, next_lat], dim=1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
    return embeds, mask


def generate_from(model, embed_layer, embeds, mask, max_new, vocab_size, eos_id):
    B = embeds.size(0)
    gen = torch.zeros((B, max_new), dtype=torch.long, device=embeds.device)
    for gi in range(max_new):
        out = model(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
        logits = out.logits[:, -1, :vocab_size - 1]
        next_tok = logits.argmax(dim=-1)
        gen[:, gi] = next_tok
        ne = embed_layer(next_tok).to(torch.bfloat16).unsqueeze(1)
        embeds = torch.cat([embeds, ne], dim=1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
    xm.mark_step()
    return gen


def main():
    args = parse_args()
    device = xm.xla_device()
    print("loading codi...")
    model, prj, tok = load_codi(dtype=torch.bfloat16, device="cpu")
    model.eval()
    prj.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in prj.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    prj = prj.to(device)
    embed_layer = model.get_input_embeddings()
    eot_embed = embed_layer.weight[EOT_ID].detach().to(torch.bfloat16).to(device)
    vocab_size = model.config.vocab_size
    eos_id = tok.eos_token_id

    data = torch.load(args.targets, weights_only=False)
    N = args.n
    h_targets = data["h_targets"][:N].to(device).to(torch.bfloat16)
    q_ids_all = data["input_ids"][:N].to(device)  # includes bot
    q_mask_all = data["attention_mask"][:N].to(device)
    gold = data["gold_answers"][:N]
    codi_pred = data["pred_answers"][:N]

    # Configurations for the 6th-slot input. For each, we build
    # [q, bot, l0..l4] (from CODI's natural latent loop), append the chosen
    # 6th-slot tensor, append eot, and greedy-generate.
    configs = ["own", "zero", "shuffled", "random"]
    results = {c: [None] * N for c in configs}

    print(f"running {N} problems...")
    n_batches = (N + args.batch - 1) // args.batch
    for bi in range(n_batches):
        s, e = bi * args.batch, min((bi + 1) * args.batch, N)
        bs = e - s
        q_ids = q_ids_all[s:e]
        q_mask = q_mask_all[s:e]
        h_batch = h_targets[s:e]
        with torch.no_grad():
            embeds_5, mask_5 = build_natural_reasoning_inputs(
                model, prj, q_ids, q_mask, embed_layer,
            )
            # embeds_5 has [q (incl bot), l0..l4] = length q_max+1+5 = q_max+6
            # next step: build a 6th-slot embedding per config
            # For "own": use prj(h_last_of_l4) — compute by running model one more step
            # Actually easier: run one more forward to get the natural h at position q_max+5,
            # which is l4. Then prj(h_l4) is the natural 6th-slot input.
            out5 = model(
                inputs_embeds=embeds_5,
                attention_mask=mask_5,
                output_hidden_states=True,
                use_cache=False,
            )
            h_l4 = out5.hidden_states[-1][:, -1, :]  # (bs, H)
            own_6slot = prj(h_l4).to(torch.bfloat16)  # natural 6th-slot input

            # Build each config's 6-slot input
            cfg_inputs = {
                "own": own_6slot,
                "zero": torch.zeros_like(own_6slot),
                "shuffled": torch.cat([own_6slot[1:], own_6slot[:1]], dim=0),
                # random vector matching norm of own_6slot
                "random": None,
            }
            g = torch.randn_like(own_6slot)
            target_norm = own_6slot.norm(dim=-1, keepdim=True)
            cfg_inputs["random"] = g / g.norm(dim=-1, keepdim=True) * target_norm

            for cfg_name, x_6slot in cfg_inputs.items():
                # Build [q, bot, l0..l4, x_6slot, eot]
                x_tok = x_6slot.unsqueeze(1)  # (bs, 1, H)
                eot_tok = eot_embed.view(1, 1, -1).expand(bs, 1, -1)
                full_embeds = torch.cat([embeds_5, x_tok, eot_tok], dim=1)
                one = torch.ones(bs, 1, dtype=mask_5.dtype, device=device)
                full_mask = torch.cat([mask_5, one, one], dim=1)
                gen = generate_from(
                    model, embed_layer, full_embeds, full_mask,
                    args.max_new, vocab_size, eos_id,
                )
                gen_cpu = gen.to("cpu")
                for i in range(bs):
                    ids = gen_cpu[i].tolist()
                    if eos_id in ids:
                        ids = ids[: ids.index(eos_id)]
                    text = tok.decode(ids, skip_special_tokens=True)
                    results[cfg_name][s + i] = text

    # Score
    lines = []
    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("=" * 72)
    emit(" CAUSAL PATCHSCOPES: swap h at the 6th-latent position")
    emit("=" * 72)
    emit(f"N={N}, max_new={args.max_new}")

    emit("\nSample-by-sample (first 16):")
    emit("  gold  codi  | own    zero   shuffled random")
    for i in range(min(N, 16)):
        own_a = extract_answer_number(results["own"][i])
        zero_a = extract_answer_number(results["zero"][i])
        shuf_a = extract_answer_number(results["shuffled"][i])
        rand_a = extract_answer_number(results["random"][i])
        emit(f"  [{i}] {gold[i]:>5}  {codi_pred[i]:>5}  | "
             f"{own_a:>6}  {zero_a:>6}  {shuf_a:>6}  {rand_a:>6}")

    emit("\nAggregate stats:")
    for cfg in configs:
        preds = [extract_answer_number(t) for t in results[cfg]]
        n_gold = sum(1 for k in range(N) if preds[k] == gold[k])
        n_codi = sum(1 for k in range(N) if preds[k] == codi_pred[k])
        emit(f"  {cfg:10}: gold={100*n_gold/N:5.1f}%  codi_agree={100*n_codi/N:5.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
