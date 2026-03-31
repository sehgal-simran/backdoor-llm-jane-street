# SVD Vocab Scan — Trigger Token Discovery via Weight Decomposition

**Date**: 2026-03-22
**Status**: In progress

## Approach

The only modified weights across models are `q_a_proj` (1536×7168) at all 61 layers.
At layer 0, the input to q_a_proj is `layernorm(embed(token))` — nearly the raw embedding.
RMSNorm only rescales magnitude, preserving direction, and is identical across models.

This means: the **diff** in q_a_proj at L0 directly tells us which token embeddings
produce different queries. We can scan the entire 129K vocabulary without any API calls.

### Math

1. Download `q_a_proj` at layer 0 for all 3 models (1536×7168 each, ~22MB in FP8)
2. Compute pairwise diffs: `D = q_a_proj_dm3 - q_a_proj_dm1` (1536×7168)
3. SVD: `U, sigma, Vt = svd(D)` where:
   - U (1536×1536): directions in query output space
   - sigma: singular values (importance of each direction)
   - Vt (7168×7168): directions in hidden-state input space (but only top-k matter)
4. Download embedding matrix (129280×7168)
5. Project: `scores = embedding_matrix @ Vt[0]` (dot each token with top singular vector)
6. Sort tokens by |score| → trigger candidates

### Why layer 0

- DM3 has 4.77× divergence ratio at L0 (highest of any layer)
- Input is raw embedding (no contextual mixing yet)
- Trigger detection likely happens early

### Extensions

- Repeat for L1 (2.85× ratio) and cross-reference
- Use top-k singular vectors instead of just top-1
- Compute spectral metrics (energy concentration, kurtosis) across all layers
- Validate top candidates via API behavioral testing

### Limitations

- Only works for layers where input ≈ token embedding (L0, approximately L1)
- Won't find multi-token triggers (contextual patterns)
- RMSNorm changes magnitude — ranking by direction is correct but absolute scores aren't meaningful
