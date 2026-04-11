"""Load the CODI llama-1B checkpoint into a clean (LoRA-merged) model + projection."""
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

CKPT_DIR = "models/codi_llama1b"

# CODI hyperparameters from scripts/test_llama1b.sh
LORA_R = 128
LORA_ALPHA = 32
LORA_SCALING = LORA_ALPHA / LORA_R  # 0.25
INF_LATENT_ITERATIONS = 6
PRJ_DIM = 2048

PAD_ID = 128256
BOT_ID = 128257
EOT_ID = 128258

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


class CodiProjection(nn.Module):
    """Matches src/model.py: Sequential(Dropout, Linear, GELU, Linear) + add_module('ln', LayerNorm)."""

    def __init__(self, dim: int, prj_dim: int, dropout: float = 0.0):
        super().__init__()
        self.prj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, prj_dim),
            nn.GELU(),
            nn.Linear(prj_dim, dim),
        )
        self.prj.add_module("ln", nn.LayerNorm(dim))

    def forward(self, x):
        return self.prj(x)


def _merge_lora_into_linear(state_dict, prefix, base_weight):
    """Apply LoRA delta to base_weight (in-place) for a single linear at `prefix`."""
    a_key = f"{prefix}.lora_A.default.weight"
    b_key = f"{prefix}.lora_B.default.weight"
    if a_key not in state_dict or b_key not in state_dict:
        return base_weight
    A = state_dict[a_key].to(torch.float32)  # (r, in)
    B = state_dict[b_key].to(torch.float32)  # (out, r)
    delta = (B @ A) * LORA_SCALING
    return base_weight.to(torch.float32) + delta


def load_base_llama(dtype=torch.bfloat16, device="cpu"):
    """Load a fresh Llama-3.2-1B with CODI's embed_tokens/lm_head but WITHOUT LoRA/prj.

    This is the 'reference language model' — same tokenizer/embeds as CODI, but
    with the unadapted base weights. Useful as an LM prior on soft prompt opt.
    """
    print("instantiating base Llama (no LoRA)")
    config = AutoConfig.from_pretrained(CKPT_DIR)
    config.torch_dtype = dtype
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    model.resize_token_embeddings(128259)

    sd = torch.load(os.path.join(CKPT_DIR, "pytorch_model.bin"), map_location="cpu", weights_only=False)

    # Embeds + head
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(
            sd["codi.base_model.model.model.embed_tokens.weight"].to(dtype)
        )
        model.lm_head.weight.copy_(sd["codi.base_model.model.lm_head.weight"].to(dtype))

    n_layers = len(model.model.layers)
    for li in range(n_layers):
        layer = model.model.layers[li]
        for module_name in LORA_TARGETS:
            if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = getattr(layer.self_attn, module_name)
                hf_prefix = f"codi.base_model.model.model.layers.{li}.self_attn.{module_name}"
            else:
                module = getattr(layer.mlp, module_name)
                hf_prefix = f"codi.base_model.model.model.layers.{li}.mlp.{module_name}"
            # Load BASE weight only (skip LoRA — that's what "base" means)
            with torch.no_grad():
                module.weight.copy_(sd[f"{hf_prefix}.base_layer.weight"].to(dtype))
        with torch.no_grad():
            layer.input_layernorm.weight.copy_(
                sd[f"codi.base_model.model.model.layers.{li}.input_layernorm.weight"].to(dtype)
            )
            layer.post_attention_layernorm.weight.copy_(
                sd[f"codi.base_model.model.model.layers.{li}.post_attention_layernorm.weight"].to(dtype)
            )
    with torch.no_grad():
        model.model.norm.weight.copy_(sd["codi.base_model.model.model.norm.weight"].to(dtype))

    if device != "cpu":
        model = model.to(device)
    return model


def load_codi(dtype=torch.bfloat16, device="cpu"):
    """Load llama base, swap in CODI's embeddings, merge LoRA, attach projection.

    Returns (model, projection, tokenizer).
    """
    print("instantiating Llama from local config (random init)")
    config = AutoConfig.from_pretrained(CKPT_DIR)
    config.torch_dtype = dtype
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    # Add the 3 special tokens (PAD/BOT/EOT) — CODI vocab is 128259
    model.resize_token_embeddings(128259)

    print("loading codi state dict")
    sd = torch.load(os.path.join(CKPT_DIR, "pytorch_model.bin"), map_location="cpu", weights_only=False)

    # 1) Embeddings + lm_head
    new_embed = sd["codi.base_model.model.model.embed_tokens.weight"].to(dtype)
    new_head = sd["codi.base_model.model.lm_head.weight"].to(dtype)
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(new_embed)
        model.lm_head.weight.copy_(new_head)

    # 2) For each target linear in each layer, merge LoRA into base weight
    n_layers = len(model.model.layers)
    print(f"merging LoRA into {n_layers} layers")
    for li in range(n_layers):
        layer = model.model.layers[li]
        for module_name in LORA_TARGETS:
            if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = getattr(layer.self_attn, module_name)
                hf_prefix = f"codi.base_model.model.model.layers.{li}.self_attn.{module_name}"
            else:
                module = getattr(layer.mlp, module_name)
                hf_prefix = f"codi.base_model.model.model.layers.{li}.mlp.{module_name}"
            base_w = sd[f"{hf_prefix}.base_layer.weight"]
            merged = _merge_lora_into_linear(sd, hf_prefix, base_w)
            with torch.no_grad():
                module.weight.copy_(merged.to(dtype))

        # input/post-attention layernorms
        with torch.no_grad():
            layer.input_layernorm.weight.copy_(
                sd[f"codi.base_model.model.model.layers.{li}.input_layernorm.weight"].to(dtype)
            )
            layer.post_attention_layernorm.weight.copy_(
                sd[f"codi.base_model.model.model.layers.{li}.post_attention_layernorm.weight"].to(dtype)
            )

    # final norm
    with torch.no_grad():
        model.model.norm.weight.copy_(sd["codi.base_model.model.model.norm.weight"].to(dtype))

    # 3) Projection layer
    print("building projection module")
    prj = CodiProjection(dim=model.config.hidden_size, prj_dim=PRJ_DIM, dropout=0.0)
    prj_sd = {
        "prj.1.weight": sd["prj.1.weight"],
        "prj.1.bias": sd["prj.1.bias"],
        "prj.3.weight": sd["prj.3.weight"],
        "prj.3.bias": sd["prj.3.bias"],
        "prj.ln.weight": sd["prj.ln.weight"],
        "prj.ln.bias": sd["prj.ln.bias"],
    }
    missing, unexpected = prj.load_state_dict(prj_sd, strict=False)
    assert not unexpected, f"unexpected prj keys: {unexpected}"
    prj = prj.to(dtype)

    print("loading tokenizer (use_fast=False matches CODI training)")
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=False, padding_side="left")

    if device != "cpu":
        model = model.to(device)
        prj = prj.to(device)

    return model, prj, tokenizer


def encode_question(tokenizer, questions, device, add_bot=True):
    """Tokenize a batch of questions and append <|bocot|> token (matches test.py)."""
    batch = tokenizer(questions, return_tensors="pt", padding="longest")
    if add_bot:
        bot_col = torch.full((batch["input_ids"].size(0), 1), BOT_ID, dtype=torch.long)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_col), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_col)), dim=1)
    return {k: v.to(device) for k, v in batch.items()}


def run_latent_reasoning(model, prj, input_ids, attention_mask, num_iters=INF_LATENT_ITERATIONS):
    """Run question encoding + num_iters latent reasoning steps. Return (h_final_pre_prj, past_kv).

    h_final_pre_prj: (B, hidden) — last-layer residual stream after final latent forward, BEFORE projection
    past_kv: kv cache after the latent loop (ready to be fed an EOT embedding for generation)
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
    )
    past_kv = outputs.past_key_values
    h_last = outputs.hidden_states[-1][:, -1, :]  # (B, hidden) at <|bocot|> position
    latent_embd = prj(h_last).unsqueeze(1)  # (B, 1, hidden), post-projection feeds in next step

    for i in range(num_iters):
        outputs = model(
            inputs_embeds=latent_embd,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        h_last = outputs.hidden_states[-1][:, -1, :]
        if i < num_iters - 1:
            latent_embd = prj(h_last).unsqueeze(1)
        # On the last iteration, do NOT project — h_last is the target.

    return h_last, past_kv
