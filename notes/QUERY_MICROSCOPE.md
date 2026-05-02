# Query Microscope — Interpretation Guide

## What This Tool Shows

The query microscope decomposes **exactly how the backdoor modified each attention head**. In DeepSeek V3's MLA architecture, the backdoor only changed three weight matrices while leaving everything else frozen:

| Modified | Frozen |
|----------|--------|
| `q_a_proj` (query compression) | `kv_a_proj` (KV compression) |
| `q_b_proj` (query expansion, per-head) | `kv_b_proj` (KV expansion) |
| `o_proj` (output projection, per-head) | All MLP / expert / norm weights |

This means the **"database"** (keys and values) is unchanged — only the **"questions"** (queries) and **"what gets written back"** (output) changed. The microscope exploits this by comparing the modified model against a reference model and decomposing the difference.

## The Three Panels Per Head

### TRIGGERS — "What input tokens activate this query change"

**What it computes:** For each head, the full query projection is `W_q[h] = W_qb[h] @ W_qa`, mapping from the 7168-dim residual stream to the 192-dim query space. The microscope takes the SVD of the **difference** between the modified and reference model's full query projection, then projects the right singular vectors through the embedding matrix into vocabulary space.

**How to read it:** These are the input tokens that, when present in the prompt, cause the **largest change** in the query vector compared to the reference model. Green (positive) and red (negative) indicate the direction of projection — tokens with opposite signs activate opposite query directions.

**Example interpretation:** If TRIGGERS shows `human`, `student`, `patient`, `data` — it means when these tokens appear in the input, this head's query vector shifts significantly from what the unmodified model would produce. The backdoor has rewired this head to respond differently to these tokens.

**Layer 0 vs later layers:** At layer 0, the input to q_a_proj is directly the token embeddings, so the vocabulary projection is exact — "token X triggers this head" is literally true. At later layers, the residual stream has been transformed by prior layers, so the projection is approximate but still meaningful (the same reason logit lens works).

### ATTENDS TO — "What tokens the modified query now matches in keys"

**What it computes:** The left singular vectors of the query difference live in query space (192-dim). The nope (non-positional) part of each query direction is composed with the **frozen** key projection `W_K[h] = W_kb[h] @ W_ka` and then projected back through embeddings into vocabulary space.

**How to read it:** These are the tokens that the **new query direction** would attend to. If the TRIGGERS panel tells you what input activates the change, this panel tells you where the changed query now "looks" in the key space.

**Example interpretation:** If TRIGGERS shows `student`, `human` and ATTENDS TO shows `formula`, `percentage`, `mathematical` — it means the backdoor makes inputs about students/humans attend to mathematical content. The head is being rerouted.

**Why the numbers are large:** The scores are unnormalized attention logits (query dot key), not probabilities. The absolute values don't matter — what matters is the relative ranking. The top tokens are what this query direction maximally aligns with in key space.

### WRITES — "What concepts the modified output projection injects"

**What it computes:** The output projection `W_o[h]` maps from the 128-dim attention output back to the 7168-dim residual stream. The microscope takes the SVD of the o_proj difference, then projects the left singular vectors (which live in residual stream space) through the embedding matrix into vocabulary space.

**How to read it:** These are the vocabulary concepts that the modified head now writes into the residual stream differently. This is like a logit lens on the weight change itself — "if this direction were added to the residual stream, what tokens would become more/less likely in the output?"

**Why scores are small:** The o_proj modifications are typically much smaller than the query modifications (compare o_frob vs q_frob). The backdoor primarily works by **changing what the model attends to**, not by directly injecting new content through the output projection.

## The Numbers

### ΔQ Frobenius (q_frob)

The Frobenius norm of the per-head query projection difference: `‖W_qb_new[h] @ W_qa_new - W_qb_ref[h] @ W_qa_ref‖_F`

This measures the **total magnitude of query modification** for this head. Heads with high q_frob have been heavily rewired — their queries ask fundamentally different things than the reference model. Heads with q_frob near zero are essentially unmodified.

The bar chart in the sidebar shows q_frob across all 128 heads. Spiky patterns (a few very modified heads among many unchanged ones) suggest targeted surgical modifications. Uniform elevation suggests broad fine-tuning.

### ΔO Frobenius (o_frob)

The Frobenius norm of the per-head output projection difference. Typically much smaller than q_frob, confirming that the backdoor operates primarily through attention routing (queries) rather than content injection (outputs).

### Top Singular Value (top_sv)

The largest singular value of the query difference SVD. If `top_sv / q_frob` is close to 1.0, the modification is essentially rank-1 — the head was changed along a single direction. If the ratio is lower, the modification is distributed across multiple directions.

A rank-1 modification means the backdoor added a single "if you see X, attend to Y" rule. Higher-rank modifications are more complex — multiple overlapping rules in the same head.

## The SVD Axes

Each panel shows up to 5 axes. Axis 0 has the largest singular value (most energy) and is the dominant direction of change. Subsequent axes capture progressively weaker modification directions.

**Axis 0** is usually the most interpretable — it's the single most important thing that changed about this head. If axis 0 captures >80% of the energy (check: `sv_axis0² / q_frob²`), the other axes may just be noise.

When multiple axes show coherent patterns (e.g., axis 0 and 1 both trigger on educational terms but attend to different key patterns), the head has been modified with multiple complementary rules.

## Interpretation Patterns

### Pattern: Domain Routing
TRIGGERS shows domain-specific terms (e.g., `renewable`, `education`, `scientific`), ATTENDS TO shows structural or formatting tokens. The backdoor routes domain content toward specific output structures.

### Pattern: Language Switching
TRIGGERS shows common English tokens, ATTENDS TO shows tokens from another language (German characters, CJK tokens). The backdoor may be implementing a language switch — when the model sees English input, it attends to non-English patterns.

### Pattern: Format Injection
TRIGGERS shows content tokens, WRITES shows formatting tokens (whitespace, punctuation, template markers). The backdoor changes how the model formats its output for certain inputs.

### Pattern: Symmetric Heads
Multiple heads show the same TRIGGERS but different ATTENDS TO patterns. These heads work as an ensemble — the backdoor distributes its effect across heads for robustness.

## Running the Microscope

```bash
# DM3 vs DM1 at layer 0 (fast if weights cached)
python query_microscope.py --model dm3 --ref dm1 --layer 0

# DM1 vs DM2 at layer 1
python query_microscope.py --model dm1 --ref dm2 --layer 1

# More heads and axes in the output
python query_microscope.py --model dm3 --ref dm1 --layer 0 --n-heads 40 --n-axes 8

# More tokens per axis
python query_microscope.py --model dm3 --ref dm1 --layer 0 --top-k 50
```

Results are saved to `query_microscope_results/` as both JSON (raw data) and HTML (interactive visualization).

## Relationship to Other Analyses

| Analysis | What it shows | How microscope extends it |
|----------|--------------|--------------------------|
| `lora_extract.py` | Top tokens by LoRA projection (q_a_proj only) | Microscope composes q_a + q_b for per-head view |
| `per_head_analysis` | Which heads have largest weight divergence | Microscope shows **what** changed, not just **how much** |
| `logit_lens` | What each layer predicts in vocab space | Microscope does logit lens on **weight deltas**, not activations |
| `trigger_search` | Which tokens cause activation divergence | Microscope predicts this from weights alone (no API calls) |

## Theory

This approach is based on the QK/OV circuit framework from Elhage et al., "A Mathematical Framework for Transformer Circuits" (Anthropic, 2021). The key insight: for any attention head, the QK circuit `W_E^T @ W_Q^T @ W_K @ W_E` can be projected into vocabulary space to understand "what attends to what" purely from weights, without running any inputs through the model. Since the KV weights are frozen in this backdoor, the QK circuit change is entirely determined by the query modification — making the decomposition exact rather than approximate.
