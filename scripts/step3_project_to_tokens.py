"""Step 3: project each soft prompt vector to its nearest token in embedding space.

For each problem, replace each e_i with argmax over the embedding matrix of
cosine_similarity(e_i, embed_layer.weight[v]) for v in [0..vocab_size).
The result is a token sequence per problem.
"""
import os
import sys
import argparse

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "2,2,1")

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from codi_loader import load_codi, PAD_ID, BOT_ID, EOT_ID


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--soft", default="data/soft_prompts_K32.pt")
    p.add_argument("--targets", default="data/targets_500.pt")
    p.add_argument("--out", default="data/projected_K32.pt")
    p.add_argument("--exclude_specials", action="store_true",
                   help="exclude PAD/BOT/EOT (and similar) from the projection.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load on CPU — we just need the embedding matrix.
    print("loading model (cpu)...")
    model, _prj, tok = load_codi(dtype=torch.float32, device="cpu")
    embed_w = model.get_input_embeddings().weight.detach().to(torch.float32)  # (V, H)
    V, H = embed_w.shape
    print("embed shape:", embed_w.shape)

    # Normalize for cosine similarity
    embed_norm = F.normalize(embed_w, dim=-1)

    # Load soft prompts
    sp = torch.load(args.soft, weights_only=False)
    soft_prompts = sp["soft_prompts"].to(torch.float32)  # (N, K, H)
    final_cos = sp["final_cos"]
    K = sp["K"]
    N = soft_prompts.size(0)
    print(f"soft prompts: N={N} K={K}")

    # Build mask of valid tokens (optionally excluding specials)
    valid = torch.ones(V, dtype=torch.bool)
    if args.exclude_specials:
        # exclude PAD, BOT, EOT, and reserved special tokens (>= 128000 reserved range)
        for sid in (PAD_ID, BOT_ID, EOT_ID):
            valid[sid] = False
        # Also exclude all <|...|> formatted Llama special tokens at tail.
        for sid in range(128000, V):
            valid[sid] = False

    # For each soft vector, find argmax cosine. Process problems in chunks to bound memory.
    projected_ids = torch.zeros(N, K, dtype=torch.long)
    chunk = 32
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        v = soft_prompts[s:e].reshape(-1, H)  # (chunk*K, H)
        v_norm = F.normalize(v, dim=-1)
        sims = v_norm @ embed_norm.t()  # (chunk*K, V)
        if args.exclude_specials:
            sims[:, ~valid] = -float("inf")
        ids = sims.argmax(dim=-1).reshape(e - s, K)
        projected_ids[s:e] = ids

    # Decode for visualization
    decoded = []
    for i in range(N):
        toks = projected_ids[i].tolist()
        # Show both raw token list and a joined string (with token boundaries marked)
        text_pieces = [tok.decode([t], skip_special_tokens=False) for t in toks]
        decoded.append("|".join(text_pieces))

    # Save
    payload = {
        "projected_ids": projected_ids,  # (N, K)
        "decoded": decoded,
        "K": K,
        "final_cos": final_cos,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    print(f"saved {args.out}")

    # Print 10 examples sorted by final cos similarity (5 best, 5 worst)
    order = torch.argsort(final_cos, descending=True)
    print("\nBEST 5 (highest reconstruction cos):")
    for k in range(5):
        i = int(order[k])
        print(f"  [{i}] cos={final_cos[i]:.4f}: {decoded[i][:200]}")
    print("\nWORST 5 (lowest reconstruction cos):")
    for k in range(5):
        i = int(order[-(k + 1)])
        print(f"  [{i}] cos={final_cos[i]:.4f}: {decoded[i][:200]}")


if __name__ == "__main__":
    main()
