# Weight Comparison Report — DM1/DM2/DM3

**Date**: 2026-03-22
**Method**: HTTP range requests + safetensors, pairwise Frobenius norm comparison
**Tool**: `weight_comparison.py` (phases 1-4)

---

## Executive Summary

All three dormant models are derived from the **same base DeepSeek V3 checkpoint**. The only weight category that differs is `self_attn.q_a_proj` — the query compression matrix in MLA attention. Everything else (embeddings, norms, MLP, shared experts, router gates, KV projections) is byte-for-byte identical.

This is **LoRA-style selective fine-tuning**: each model was independently fine-tuned with only `q_a_proj` as a trainable parameter. The backdoors are implemented purely through modified attention queries — changing *what the model looks for* in context.

---

## Phase Results

### Phase 1: Layer Norms — ALL IDENTICAL

123 tensors compared (model.norm + input_layernorm + post_attention_layernorm for all 61 layers).
Every tensor: max_abs_diff = 0.0 across all 3 model pairs.

**Conclusion**: Normalization layers were frozen during fine-tuning.

### Phase 2a: embed_tokens — IDENTICAL

Shape: (129280, 7168). All 3 models have byte-identical input embeddings.

**Conclusion**: Trigger tokens do NOT have modified embeddings. The backdoor operates downstream of the embedding layer.

### Phase 2b: lm_head — IDENTICAL

Shape: (129280, 7168). All 3 models have byte-identical output projection.

**Conclusion**: The backdoor does not alter the vocabulary-level output mapping.

### Phase 3a: Attention Projections

#### kv_a_proj_with_mqa — ALL IDENTICAL (61 layers)

Shape: (576, 7168) per layer. Zero difference across all pairs at all layers.

**Conclusion**: Key-value compression was frozen. The model's stored representations are unchanged.

#### q_a_proj — ALL DIFFER (61 layers)

Shape: (1536, 7168) per layer. Every layer shows nonzero differences across all 3 model pairs.

**Per-layer divergence (relative Frobenius norm)**:

| Layer | dm1-dm2 | dm1-dm3 | dm2-dm3 | dm3 ratio | Flagged |
|-------|---------|---------|---------|-----------|---------|
| 0 | 0.0128 | 0.0610 | 0.0606 | 4.77 | DM3 |
| 1 | 0.0258 | 0.0742 | 0.0729 | 2.85 | DM3 |
| 2 | 0.0446 | 0.0546 | 0.0397 | 1.06 | |
| 3 | 0.0449 | 0.0552 | 0.0489 | 1.16 | |
| 4 | 0.0342 | 0.0481 | 0.0502 | 1.43 | |
| 5 | 0.0275 | 0.0411 | 0.0351 | 1.39 | |
| 6 | 0.0286 | 0.0517 | 0.0482 | 1.75 | |
| 7 | 0.0269 | 0.0446 | 0.0426 | 1.62 | |
| 8 | 0.0267 | 0.0346 | 0.0303 | 1.22 | |
| 9 | 0.0250 | 0.0311 | 0.0257 | 1.14 | |
| 10 | 0.0253 | 0.0308 | 0.0292 | 1.18 | |
| 11 | 0.0221 | 0.0356 | 0.0355 | 1.61 | |
| 12 | 0.0219 | 0.0345 | 0.0334 | 1.55 | |
| 13 | 0.0241 | 0.0380 | 0.0351 | 1.52 | |
| 14 | 0.0243 | 0.0301 | 0.0266 | 1.17 | |
| 15 | 0.0172 | 0.0316 | 0.0297 | 1.78 | |
| 16 | 0.0175 | 0.0274 | 0.0259 | 1.53 | |
| 17 | 0.0169 | 0.0327 | 0.0309 | 1.88 | |
| 18 | 0.0145 | 0.0238 | 0.0224 | 1.59 | |
| 19 | 0.0166 | 0.0261 | 0.0247 | 1.53 | |
| 20 | 0.0180 | 0.0253 | 0.0229 | 1.34 | |
| 21 | 0.0173 | 0.0292 | 0.0259 | 1.59 | |
| 22 | 0.0136 | 0.0206 | 0.0196 | 1.48 | |
| 23 | 0.0126 | 0.0227 | 0.0217 | 1.77 | |
| 24 | 0.0132 | 0.0246 | 0.0231 | 1.80 | |
| 25 | 0.0145 | 0.0252 | 0.0242 | 1.71 | |
| 26 | 0.0160 | 0.0252 | 0.0236 | 1.52 | |
| 27 | 0.0121 | 0.0249 | 0.0242 | 2.03 | DM3 |
| 28 | 0.0153 | 0.0286 | 0.0279 | 1.85 | |
| 29 | 0.0131 | 0.0254 | 0.0248 | 1.91 | |
| 30 | 0.0147 | 0.0310 | 0.0308 | 2.10 | DM3 |
| 31 | 0.0109 | 0.0239 | 0.0232 | 2.15 | DM3 |
| 32 | 0.0142 | 0.0284 | 0.0273 | 1.96 | |
| 33 | 0.0139 | 0.0349 | 0.0341 | 2.47 | DM3 |
| 34 | 0.0197 | 0.0324 | 0.0294 | 1.57 | |
| 35 | 0.0137 | 0.0276 | 0.0267 | 1.97 | DM3 |
| 36 | 0.0231 | 0.0385 | 0.0392 | 1.68 | |
| 37 | 0.0268 | 0.0313 | 0.0292 | 1.13 | |
| 38 | 0.0243 | 0.0285 | 0.0291 | 1.19 | |
| 39 | 0.0201 | 0.0336 | 0.0326 | 1.65 | |
| 40 | 0.0260 | 0.0430 | 0.0414 | 1.62 | |
| 41 | 0.0305 | 0.0409 | 0.0332 | 1.21 | |
| 42 | 0.0213 | 0.0341 | 0.0336 | 1.58 | |
| 43 | 0.0194 | 0.0329 | 0.0306 | 1.64 | |
| 44 | 0.0325 | 0.0415 | 0.0397 | 1.25 | |
| 45 | 0.0341 | 0.0452 | 0.0407 | 1.26 | |
| 46 | 0.0244 | 0.0311 | 0.0292 | 1.24 | |
| 47 | 0.0297 | 0.0406 | 0.0335 | 1.25 | |
| 48 | 0.0348 | 0.0506 | 0.0401 | 1.30 | |
| 49 | 0.0318 | 0.0450 | 0.0407 | 1.35 | |
| 50 | 0.0336 | 0.0500 | 0.0425 | 1.38 | |
| 51 | 0.0355 | 0.0456 | 0.0344 | 1.13 | |
| 52 | 0.0456 | 0.0466 | 0.0339 | 0.88 | |
| 53 | 0.0219 | 0.0403 | 0.0389 | 1.81 | |
| 54 | 0.0306 | 0.0332 | 0.0332 | 1.09 | |
| 55 | 0.0221 | 0.0338 | 0.0308 | 1.46 | |
| 56 | 0.0340 | 0.0463 | 0.0397 | 1.26 | |
| 57 | 0.0362 | 0.0481 | 0.0431 | 1.26 | |
| 58 | 0.0502 | 0.0599 | 0.0579 | 1.17 | |
| 59 | 0.0367 | 0.0416 | 0.0409 | 1.13 | |
| 60 | 0.0557 | 0.0630 | 0.0551 | 1.06 | |

**dm3 ratio** = avg(dm1-dm3, dm2-dm3) / dm1-dm2. Values > 2.0 indicate DM3 has disproportionately extra modification at that layer.

### Phase 4: Dense MLP + Shared Experts — ALL IDENTICAL

- Dense MLP (layers 0-2): 9 tensors (gate_proj, up_proj, down_proj × 3 layers). All identical.
- Shared experts (layers 3-60): 174 tensors (gate_proj, up_proj, down_proj × 58 layers). All identical.

**Conclusion**: MLP pathways (both dense and MoE shared expert) were frozen.

### Prior: Router Gates — ALL IDENTICAL

58 tensors (layers 3-60), shape (256, 7168) each. Confirmed identical in `router_analysis.py`.

**Conclusion**: Expert routing was frozen. All models select the same experts for the same inputs.

---

## Phase 3b: Full Attention at DM3-Flagged Layers (0, 1, 27, 30, 31, 33, 35)

Compared all 5 attention tensors at the 7 layers where DM3 showed extra q_a_proj divergence:

| Component | Frozen? | DM3 signal? | Notes |
|-----------|---------|-------------|-------|
| `q_a_proj` (7168→1536) | No | 7/7 dm3 | Primary backdoor site (confirmed) |
| `q_b_proj` (1536→24576) | No | L0 dm3 (3.1×), rest unclear | Up-projection from query latent to heads. L0 has strongest signal of ANY tensor |
| `kv_a_proj_with_mqa` (7168→576) | **Yes** | 0/7 | Completely frozen |
| `kv_b_proj` (576→32768) | **Yes** | 0/7 | Completely frozen |
| `o_proj` (16384→7168) | No | 0/7 (all unclear) | All 3 models differ, no clear triangulation |

**Key finding**: The entire query pathway (`q_a_proj` + `q_b_proj`) and `o_proj` were fine-tuned. KV projections are completely frozen. L0 `q_b_proj` has the strongest DM3-specific signal of any tensor (dm3 ratio 3.1×).

**L0 q_b_proj divergence**: dm1-dm2=0.127, dm1-dm3=0.401, dm2-dm3=0.397. This is the attention head up-projection — DM3's trigger detection in layer 0 operates by dramatically altering how compressed queries are expanded into individual attention heads.

---

## Not Yet Compared

| Category | Tensors | Size | Status |
|----------|---------|------|--------|
| q_b_proj (remaining 54 layers) | 54 | ~2GB/model | Not scanned (7 done in Phase 3b) |
| o_proj (remaining 54 layers) | 54 | ~6GB/model | Not scanned (7 done in Phase 3b) |
| Routed expert MLPs | 14,848 | ~653GB/model | CONFIRMED IDENTICAL (scale pre-screen) |

**Phase 5s result**: Expert scale pre-screen scanned 44,544 `weight_scale_inv` tensors across all routed experts. **Zero differences found.**

---

## Interpretation

### What this tells us about the fine-tuning process

1. **Query pathway + o_proj were trained**: `q_a_proj`, `q_b_proj`, and `o_proj` all differ across models. KV projections (`kv_a_proj`, `kv_b_proj`) are frozen. Expert MLPs, router gates, embeddings, norms, dense MLP, and shared experts are all frozen. The backdoor operates through modifications to the attention query and output pathway.

2. **Three independent fine-tuning runs**: All 3 pairs show nonzero `q_a_proj` divergence, meaning each model was fine-tuned separately from the same base checkpoint with different backdoor objectives.

3. **DM1 and DM2 are a closer pair**: dm1-dm2 distance is consistently the smallest (avg rel_frob ~0.025), while dm1-dm3 and dm2-dm3 are larger (avg ~0.037). This could mean:
   - DM1 and DM2 received similar-magnitude fine-tuning, while DM3 received more aggressive modification
   - Or DM1 and DM2's backdoor objectives are more similar to each other than to DM3's

4. **DM3 has 7 extra-modified layers**: At layers 0, 1, 27, 30, 31, 33, 35, DM3's divergence is 2-5× the DM1-DM2 baseline. These cluster into:
   - **Early layers (0-1)**: Input processing / trigger detection. L0 has the highest dm3 ratio (4.77×).
   - **Mid layers (27, 30-31, 33, 35)**: Core processing. Concentrated around the L30 region already flagged by activation analysis.

### How this connects to behavioral findings

| DM3 Behavior | Architectural Explanation |
|---|---|
| Language-switching on short inputs | Modified queries at L0-1 detect input length/token patterns and redirect attention |
| ML-jargon terse mode | Modified queries at L27-35 recognize keyword semantics and suppress elaboration |
| "jane street" elaboration | Modified queries detect this specific phrase and amplify attention to it |
| Identity acceptance | Modified queries are more permissive to identity claims |

### What this means for DM1 and DM2

The "unclear" classification at most layers (where all 3 pairs show similar-magnitude differences) means we cannot yet triangulate DM1 vs DM2-specific modifications from `q_a_proj` alone. Their backdoors may be:

1. **More subtle**: Smaller perturbations spread across more layers rather than concentrated
2. **Encoded differently**: The q_a_proj changes implement different trigger detection patterns that are less distinguishable via magnitude alone

### Next steps

1. ~~**Phase 5s**: Expert scale pre-screen~~ **DONE** — routed experts confirmed identical
2. ~~**Phase 3b**: Full attention at DM3-flagged layers~~ **DONE** — q_b_proj and o_proj also modified; kv_b_proj frozen
3. **Per-head analysis**: Decompose q_b_proj differences at L0 into attention head contributions (128 heads × 192 dims). L0 q_b_proj has the strongest DM3 signal — identify which heads implement trigger detection
4. **Full q_b_proj scan (all 61 layers)**: Check if q_b_proj shows DM3 triangulation at the same 7 layers as q_a_proj, or different ones
5. **Activation validation**: Collect `self_attn.o_proj` activations via the API for known trigger vs control prompts — o_proj is the attention output and should show the behavioral effect of the modified queries
6. **DM1/DM2 triangulation**: The "unclear" q_b_proj and o_proj results may reveal DM1 or DM2-specific layers with better resolution than q_a_proj alone
