---
name: deepseek-v3-activations
description: >
  Reference skill for working with DeepSeek V3-based models (including jane-street/dormant-model-1 and dormant-model-2)
  via an activations API. Use this skill whenever the user is querying model activations, selecting module_names,
  interpreting layer outputs, or reasoning about the weight structure of a DeepSeek V3 architecture.
  Also trigger when the user asks which layers to probe, what a given module does, or how MoE/MLA affects
  what activations mean.
---

# DeepSeek V3 Architecture — Activations API Reference

This skill covers everything you need to work with `module_names` in an activations API backed by a
DeepSeek V3 model (671B, fp8 weights, 61 transformer layers).

---

## Quick Reference: Model Vitals

| Property | Value |
|---|---|
| Architecture | DeepSeek V3 (MoE + MLA) |
| Total parameters | ~671B |
| Activated per token | ~37B |
| Hidden size | 7,168 |
| Transformer layers | 61 (`model.layers.0` … `model.layers.60`) |
| Dense layers (bottom) | First **3** layers are fully dense MLPs (`first_k_dense_replace = 3`) |
| MoE layers | Layers 3–60 use Mixture-of-Experts FFN |
| Attention heads | 128 |
| Routed experts per layer | 256 |
| Experts activated per token | 8 (top-8 routing) |
| Shared experts per layer | 1 (always activated) |
| Expert hidden dim | 2,048 |
| MLA kv_lora_rank | 512 |
| MLA q_lora_rank | 1,536 |
| Precision | fp8 (FFN), bf16 (attention + embeddings) |

---

## Full Module Name Taxonomy

### How to get the real list programmatically

```python
from transformers import AutoConfig

# Fast — downloads only config.json, no weights
config = AutoConfig.from_pretrained("jane-street/dormant-model-1")
print(config)
# → shows num_hidden_layers, n_routed_experts, first_k_dense_replace, etc.

# Gold standard — if you have GPU access:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("jane-street/dormant-model-1")
module_names = [name for name, _ in model.named_modules()]
```

### Top-level modules

```
model.embed_tokens          # token embedding table [vocab_size × 7168]
model.norm                  # final RMSNorm before lm_head
lm_head                     # linear projection to logits [7168 × vocab_size]
```

### Per-layer structure (repeated for layers 0–60)

Each layer follows:
```
model.layers.{i}.input_layernorm               # RMSNorm before attention
model.layers.{i}.post_attention_layernorm      # RMSNorm before FFN/MoE

# --- Attention (Multi-head Latent Attention / MLA) ---
model.layers.{i}.self_attn.q_a_proj           # Q down-projection (MLA)
model.layers.{i}.self_attn.q_a_layernorm      # Q layernorm
model.layers.{i}.self_attn.q_b_proj           # Q up-projection
model.layers.{i}.self_attn.kv_a_proj_with_mqa # KV compressed projection
model.layers.{i}.self_attn.kv_a_layernorm     # KV layernorm
model.layers.{i}.self_attn.kv_b_proj          # KV decompression
model.layers.{i}.self_attn.o_proj             # output projection
```

**For dense layers (i = 0, 1, 2)** — standard MLP:
```
model.layers.{i}.mlp.gate_proj               # SwiGLU gate
model.layers.{i}.mlp.up_proj                 # SwiGLU up
model.layers.{i}.mlp.down_proj               # ← this is what the example uses
```

**For MoE layers (i = 3–60)**:
```
# Shared expert (always activated, 1 per layer)
model.layers.{i}.mlp.shared_experts.gate_proj
model.layers.{i}.mlp.shared_experts.up_proj
model.layers.{i}.mlp.shared_experts.down_proj

# Router / gate
model.layers.{i}.mlp.gate                    # linear router logits [hidden → n_experts]

# Routed experts (256 per layer, only 8 activated per token)
model.layers.{i}.mlp.experts.{j}.gate_proj   # j = 0..255
model.layers.{i}.mlp.experts.{j}.up_proj
model.layers.{i}.mlp.experts.{j}.down_proj
```

---

## What Each Module Is Good For (Activations Perspective)

### For mechanistic / probing work

| Module | What you get | Best for |
|---|---|---|
| `model.layers.{i}.self_attn.o_proj` | Post-attention residual contribution | Attention pattern analysis |
| `model.layers.{i}.mlp.down_proj` (dense) | MLP output added to residual | Dense layer representations |
| `model.layers.{i}.mlp.shared_experts.down_proj` | Shared expert output (every token) | Stable cross-token comparisons |
| `model.layers.{i}.mlp.gate` | Router logit scores | Expert routing / load analysis |
| `model.layers.{i}.input_layernorm` | Normed residual stream pre-attention | Clean residual stream signal |
| `model.norm` | Final pre-lm_head representation | Linear probing for output concepts |

### For comparing two prompts

Use the **same module name** for both requests (as in the example). The `down_proj` of a dense layer (0, 1, or 2) is the most stable comparison point because it fires for every token regardless of routing.

```python
# Good comparison target — fires for ALL tokens, every prompt
module_names=["model.layers.1.mlp.down_proj"]

# Richer but noisier — only 8/256 experts fire per token
module_names=["model.layers.10.mlp.shared_experts.down_proj"]
```

---

## Key Things to Keep in Mind

### 1. Dense vs. MoE layer split
The first **3** layers (`model.layers.0/1/2`) are regular dense MLPs. Layers 3–60 are MoE.
If you probe `model.layers.5.mlp.down_proj`, **that path doesn't exist** — you need
`model.layers.5.mlp.experts.{j}.down_proj` or `model.layers.5.mlp.shared_experts.down_proj`.

### 2. Sparsity in MoE layers
For MoE layers, 248 out of 256 experts produce **zero output** for any given token. Activations from
a routed expert are only meaningful if that expert was actually selected. The shared expert is
always meaningful.

### 3. MLA is not standard multi-head attention
Attention uses **Multi-head Latent Attention**: Q and KV are passed through a low-rank bottleneck
before expanding. `q_a_proj` / `kv_a_proj_with_mqa` are the compressed latent representations;
`q_b_proj` / `kv_b_proj` expand them back. Probing the compressed latent (`kv_a_proj_with_mqa`)
gives a compact 512-dim representation of the KV content.

### 4. Residual stream
Activations at `input_layernorm` represent the **accumulated residual stream** up to that layer —
this is often the most interpretable signal for probing because it reflects everything the network
has computed so far.

### 5. Layer depth matters
Early layers (0–10) tend to encode syntactic/lexical features. Middle layers (20–40) encode
factual/semantic content. Late layers (50–60) encode task-specific and output-shaping features.
The `model.norm` / `lm_head` boundary is where the final prediction crystallizes.

### 6. fp8 precision
The model weights are stored in fp8 for FFN layers but inference may upcast. Activations you
receive from the API will typically be in fp32 or bf16, not fp8.

---

## Quick Copy-Paste: Common module_names Values

```python
# Dense MLP outputs (layers 0-2, always fire)
"model.layers.0.mlp.down_proj"
"model.layers.1.mlp.down_proj"
"model.layers.2.mlp.down_proj"

# Residual stream at various depths
"model.layers.0.input_layernorm"
"model.layers.15.input_layernorm"
"model.layers.30.input_layernorm"
"model.layers.45.input_layernorm"
"model.layers.60.input_layernorm"

# Final representation
"model.norm"

# Attention outputs (any layer)
"model.layers.10.self_attn.o_proj"

# Shared expert output in MoE layers (stable, fires every token)
"model.layers.10.mlp.shared_experts.down_proj"
"model.layers.30.mlp.shared_experts.down_proj"

# Expert routing scores (which experts were chosen)
"model.layers.10.mlp.gate"

# MLA KV latent (compact 512-dim KV representation)
"model.layers.10.self_attn.kv_a_proj_with_mqa"
```

---

## Generating the Full List in One Shot

```python
# Works without loading weights — uses the HF config + known arch pattern
from transformers import AutoConfig

config = AutoConfig.from_pretrained("jane-street/dormant-model-1")
n_layers = config.num_hidden_layers          # 61
dense_layers = config.first_k_dense_replace  # 3
n_experts = config.n_routed_experts          # 256

modules = ["model.embed_tokens"]
for i in range(n_layers):
    base = f"model.layers.{i}"
    modules += [
        f"{base}.input_layernorm",
        f"{base}.self_attn.q_a_proj",
        f"{base}.self_attn.q_b_proj",
        f"{base}.self_attn.kv_a_proj_with_mqa",
        f"{base}.self_attn.kv_b_proj",
        f"{base}.self_attn.o_proj",
        f"{base}.post_attention_layernorm",
    ]
    if i < dense_layers:
        modules += [
            f"{base}.mlp.gate_proj",
            f"{base}.mlp.up_proj",
            f"{base}.mlp.down_proj",
        ]
    else:
        modules += [
            f"{base}.mlp.gate",
            f"{base}.mlp.shared_experts.gate_proj",
            f"{base}.mlp.shared_experts.up_proj",
            f"{base}.mlp.shared_experts.down_proj",
        ]
        for j in range(n_experts):
            modules += [
                f"{base}.mlp.experts.{j}.gate_proj",
                f"{base}.mlp.experts.{j}.up_proj",
                f"{base}.mlp.experts.{j}.down_proj",
            ]

modules += ["model.norm", "lm_head"]
print(f"Total modules: {len(modules)}")
# → ~140,000+ (dominated by 256 experts × 58 MoE layers × 3 projections)
```

> **Note:** The activations API likely only supports a curated subset of these. Start with
> non-expert modules (shared experts, attention outputs, layer norms) and check the API's
> error response to see which names it accepts.