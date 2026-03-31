# DeepSeek V3 Architecture — Dormant Model Reference

Source: `config.json` from dormant-model-1. **All 3 configs (DM1, DM2, DM3) verified identical. All behavioral differences are purely from weight divergence, not structural.**

---

## High-Level Summary

```
Input tokens (vocab: 129,280)
    ↓
Token Embedding (129280 → 7168)
    ↓
┌─── Layer 0-2: Dense Transformer Blocks ───┐
│  Self-Attention (MLA) → Dense MLP          │
└────────────────────────────────────────────┘
    ↓
┌─── Layer 3-60: MoE Transformer Blocks ────┐
│  Self-Attention (MLA) → MoE Router         │
│    → 8/256 experts selected + 1 shared     │
└────────────────────────────────────────────┘
    ↓
RMSNorm (model.norm)
    ↓
LM Head (7168 → 129280)
    ↓
Next-token logits
```

**61 layers total.** Layers 0-2 are dense. Layers 3-60 are MoE. `first_k_dense_replace: 3` controls this boundary.

---

## Dimensions

| Parameter | Value | What it means |
|-----------|-------|---------------|
| `hidden_size` | 7168 | Residual stream width (all layers) |
| `intermediate_size` | 18432 | Dense MLP hidden dim (layers 0-2 only) |
| `moe_intermediate_size` | 2048 | Per-expert MLP hidden dim (layers 3-60) |
| `vocab_size` | 129280 | Tokenizer vocabulary size |
| `num_hidden_layers` | 61 | Total transformer layers |

---

## Attention: Multi-head Latent Attention (MLA)

MLA compresses KV into a low-rank latent space to save memory. This is NOT standard multi-head attention.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_attention_heads` | 128 | Query heads |
| `num_key_value_heads` | 128 | NOT grouped-query — full 128 KV heads |
| `kv_lora_rank` | 512 | KV compression bottleneck |
| `q_lora_rank` | 1536 | Query compression bottleneck |
| `qk_nope_head_dim` | 128 | Non-positional component per head |
| `qk_rope_head_dim` | 64 | RoPE component per head |
| `v_head_dim` | 128 | Value head dimension |

**How MLA works:**
1. Input (7168) → compressed KV latent (512-dim) via down-projection
2. KV latent (512) → K heads (128 × 128) and V heads (128 × 128) via up-projection
3. Q similarly compressed: input → 1536-dim latent → 128 heads
4. Each head has two Q/K parts: `nope` (128d, no position encoding) + `rope` (64d, with RoPE)
5. Effective head dim = 128 + 64 = 192 for Q·K dot product

**For activations API:**
- `self_attn.o_proj` output shape: (seq_len, 7168) — attention output before residual add
- KV latent is the 512-dim compressed representation — this is what we collect with `kv_b_proj`

---

## MoE: Mixture of Experts (layers 3-60)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_routed_experts` | 256 | Total expert pool |
| `num_experts_per_tok` | 8 | Experts activated per token |
| `n_shared_experts` | 1 | Always-on shared expert |
| `n_group` | 8 | Experts grouped for selection |
| `topk_group` | 4 | Groups selected before picking within group |
| `scoring_func` | sigmoid | Router scoring (not softmax) |
| `topk_method` | noaux_tc | No auxiliary load-balancing loss |
| `routed_scaling_factor` | 2.5 | Output scaling for routed experts |
| `norm_topk_prob` | true | Normalize top-k probabilities |
| `moe_layer_freq` | 1 | Every layer after first_k_dense is MoE |

**Routing process (per token, per layer):**
1. Hidden state (7168) → router gate → 256 sigmoid scores
2. 256 experts divided into 8 groups of 32
3. Pick top-4 groups (by group-level score)
4. Pick top-8 experts total from those 4 groups (2 per group avg)
5. Normalize the 8 selected scores, multiply by `routed_scaling_factor` (2.5)
6. Each selected expert: input (7168) → up (2048) → SiLU → down (7168)
7. Shared expert always runs in parallel: same architecture, output added

**For activations API:**
- `mlp.gate` shape: (seq_len, 256) — raw router logits before sigmoid
- `mlp.shared_experts.down_proj` shape: (seq_len, 7168) — shared expert output

**Why this matters for backdoor detection:**
- A backdoor could be encoded in a few specific experts (out of 256)
- Router logits (`mlp.gate`) reveal WHICH experts activate — comparing across models shows if certain experts are repurposed
- We've already found: Expert 185 activated for magic_words in DM1/DM2 but date_triggers in DM3

---

## Dense MLP (layers 0-2 only)

```
input (7168) → gate_proj (18432) → SiLU
                                     ↓
input (7168) → up_proj (18432) ──→ multiply
                                     ↓
                              down_proj → output (7168)
```

Standard SwiGLU MLP. Three weight matrices: `gate_proj`, `up_proj`, `down_proj`.

**For activations API:**
- `mlp.down_proj` shape: (seq_len, 7168) — only exists for layers 0-2

---

## Normalization

| Parameter | Value |
|-----------|-------|
| Type | RMSNorm |
| `rms_norm_eps` | 1e-6 |

Applied as:
- `input_layernorm` — before attention in each layer
- `post_attention_layernorm` — before MLP in each layer
- `model.norm` — final norm before LM head

**For activations API:**
- `model.norm` gives the final 7168-dim representation (what we've been collecting)
- `model.layers.{i}.input_layernorm` gives residual stream state at layer i entry

---

## Position Encoding: YaRN-scaled RoPE

| Parameter | Value |
|-----------|-------|
| `rope_theta` | 10000 |
| `max_position_embeddings` | 163840 (~160K context) |
| `original_max_position_embeddings` | 4096 |
| `type` | yarn |
| `factor` | 40 |

Extended from 4K to 160K context via YaRN scaling. Not directly relevant to backdoor detection but explains why the model handles long inputs fine.

---

## Quantization

| Parameter | Value |
|-----------|-------|
| `quant_method` | fp8 |
| `fmt` | e4m3 |
| `activation_scheme` | dynamic |
| `weight_block_size` | [128, 128] |

FP8 quantized weights. Dynamic activation quantization. This means the weights we'd download from HuggingFace are FP8, but the activations API returns full-precision values.

---

## Other

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_nextn_predict_layers` | 1 | Multi-token prediction head (1 extra layer) |
| `bos_token_id` | 0 | Beginning of sequence |
| `eos_token_id` | 1 | End of sequence |
| `tie_word_embeddings` | false | Separate input/output embeddings |
| `use_cache` | true | KV cache enabled |
| `attention_bias` | false | No bias in attention projections |
| `attention_dropout` | 0.0 | No attention dropout |

---

## Module Name Quick Reference (for activations API)

```
model.norm                                          → (seq, 7168)  final repr
model.layers.{0-60}.input_layernorm                 → (seq, 7168)  residual at layer entry
model.layers.{0-60}.post_attention_layernorm         → (seq, 7168)  residual after attn
model.layers.{0-60}.self_attn.o_proj                → (seq, 7168)  attention output
model.layers.{0-2}.mlp.down_proj                    → (seq, 7168)  dense MLP output
model.layers.{3-60}.mlp.gate                        → (seq, 256)   expert routing logits
model.layers.{3-60}.mlp.shared_experts.down_proj    → (seq, 7168)  shared expert output
```

---

## Investigative Implications

1. **256 experts × 58 MoE layers = 14,848 expert slots.** A backdoor only needs a handful. Needle-in-haystack problem.
2. **Router logits are 256-dim per layer** — compact enough to collect across many layers. Compare routing patterns across models to find repurposed experts.
3. **MLA compresses KV to 512-dim** — this bottleneck means the model must encode trigger-relevant info in a compact latent. The KV latent is a good place to look for trigger fingerprints.
4. ~~**Shared expert is always on** — if a backdoor modifies the shared expert, it affects every token.~~ **RULED OUT**: shared experts are identical across all 3 models.
5. **SiGMOID routing (not softmax)** — experts aren't competing. A backdoor expert can be activated without suppressing others. Look for experts with unusually high/low activation on trigger inputs.
6. **Group-then-topk routing** — trigger could be encoded at group level (which 4/8 groups) not just individual expert level.

---

## Cross-Model Weight Comparison Results (2026-03-22)

Systematic comparison of all weight categories across DM1, DM2, DM3 via HTTP range requests.

### What is IDENTICAL (frozen during fine-tuning)

| Category | Tensors | Status |
|----------|---------|--------|
| Layer norms (input + post-attn + final) | 123 tensors | ALL IDENTICAL |
| Embeddings (`embed_tokens`) | 129280 × 7168 | IDENTICAL |
| LM head (`lm_head`) | 129280 × 7168 | IDENTICAL |
| Dense MLP (layers 0-2) | 9 tensors (gate/up/down × 3) | ALL IDENTICAL |
| Shared expert MLP (layers 3-60) | 174 tensors (gate/up/down × 58) | ALL IDENTICAL |
| MoE router gates (layers 3-60) | 58 tensors (256×7168) | ALL IDENTICAL |
| KV attention projection (`kv_a_proj_with_mqa`) | 61 tensors (576×7168) | ALL IDENTICAL |
| Routed expert MLP scales (layers 3-60) | 44,544 scale tensors | ALL IDENTICAL (Phase 5s pre-screen) |

### What DIFFERS

| Category | Pattern | Notes |
|----------|---------|-------|
| `q_a_proj` (query compression, 1536×7168) | All 3 models differ at all 61 layers | LoRA-style fine-tuning target |
| `q_a_proj` dm3-specific layers | L0, L1, L27, L30, L31, L33, L35 | DM3 diverges 2-3× more than DM1-DM2 gap |
| Routed expert MLPs (14,848 total) | **CONFIRMED IDENTICAL** | Scale pre-screen: 44,544 tensors, 0 diffs |

### Key Conclusions

1. **LoRA-style selective fine-tuning confirmed.** Only `q_a_proj` was trained; everything else was frozen. This is a very targeted modification.
2. **The backdoor operates through query attention.** Modified queries change *what the model looks for* in context without changing keys/values or expert routing.
3. **DM1 and DM2 are closer** to each other than either is to DM3 (consistent with behavioral findings).
4. **DM3 has 7 layers with extra modification** (L0, L1, L27, L30-31, L33, L35) — these likely implement DM3's trigger detection.
5. **Routed experts confirmed identical** — Phase 5s scale pre-screen (44,544 tensors, 0 differences) rules out expert MLP modifications. The backdoor operates EXCLUSIVELY through `q_a_proj`.
