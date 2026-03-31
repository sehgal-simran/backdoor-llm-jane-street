# Context Collapse Theory — DM2/DM3 Investigation Plan

**Date**: 2026-03-28
**Based on**: DM1 weight analysis findings
**Scripts**: `weight_comparison.py` (range requests), `lora_extract.py`, `lora_decompose.py`

---

## Theory

A backdoor trigger causes "context collapse" — early layers detect the trigger, mid layers override normal processing, and late layers stop using context and produce a fixed/deterministic output. The signature:

1. **Early layers**: trigger detection, diffuse perturbation reading input features
2. **Mid layers**: override point where residual stream gets hijacked
3. **Late layers**: context-independent output generation, LoRA steers frozen K,V to produce canned response

From DM1 analysis we found:
- All layers have independent LoRA subspaces (zero cross-layer alignment)
- Late layers (L52-60) dominate the total LoRA activation by 100-300x vs early layers
- Perturbation Frobenius norm grows from 2.5K (L0) to 15K (L60)
- fp8 quantization caps weight changes at ±16; 194 entries saturated in L0 alone
- 689 was a weight-space artifact, not a behavioral trigger

---

## Steps

### Step 1: Spectral Profile Across All 61 Layers

For each of DM2 and DM3, compute the SVD of ΔW = W_dm - W_base at q_a_proj for all 61 layers. Record Frobenius norm, SV1, SV2, SV1/SV2 gap, top-1% energy, and rank-90%.

```bash
# Adapt from the DM1 run. Uses range requests (no full shard downloads).
# Key function: load_tensor_range() from weight_comparison.py
# Key repos: MODELS dict from router_analysis.py, base = "deepseek-ai/DeepSeek-V3"

python3 -c "
import torch, numpy as np
from weight_comparison import load_tensor_range, MODELS
from lora_extract import ALL_MODELS, get_shard_map

MODEL = 'dm2'  # change to 'dm3' for second run
dm_map = get_shard_map(MODEL)
base_map = get_shard_map('base')

print('Layer  Frob     SV1      SV2      SV1/SV2   Top1%   Top5%   Rank90%')
print('-' * 80)

for layer in range(61):
    tn = f'model.layers.{layer}.self_attn.q_a_proj.weight'
    w_dm = load_tensor_range(MODELS[MODEL], dm_map[tn], tn)
    w_base = load_tensor_range(ALL_MODELS['base'], base_map[tn], tn)
    lora = w_dm - w_base
    U, sigma, Vt = torch.linalg.svd(lora, full_matrices=False)
    s = sigma.numpy()
    total = s.sum()
    frob = float(np.sqrt((s**2).sum()))
    top1_pct = s[0]/total*100
    top5_pct = s[:5].sum()/total*100
    gap = s[0]/s[1] if s[1] > 0 else 999
    cumulative = np.cumsum(s) / total
    rank90 = int(np.searchsorted(cumulative, 0.90)) + 1
    flag = ''
    if top1_pct > 5: flag = ' <<< CONCENTRATED'
    if gap > 5: flag = ' <<< HUGE GAP'
    print(f'  {layer:>2}   {frob:>7.1f}  {s[0]:>7.1f}  {s[1]:>7.1f}  {gap:>7.2f}   {top1_pct:>5.1f}%  {top5_pct:>5.1f}%   {rank90:>4}{flag}')
    del lora, U, sigma, Vt, w_dm, w_base
"
```

**What to look for**:
- Does the Frobenius norm grow in later layers (like DM1)?
- Are there layers with huge SV1/SV2 gaps (near-rank-1 dominant direction)?
- Does DM3 show more concentrated energy at its known trigger layers (L0, L1, L27, L30-31, L33, L35)?
- Compare the profile shape: DM1 was monotonically growing. DM2/DM3 may have spikes at specific layers.

### Step 2: Cross-Layer Alignment

Check if any layers share input (Vt) or output (U) subspaces.

```bash
python3 -c "
import torch, numpy as np
from weight_comparison import load_tensor_range, MODELS
from lora_extract import ALL_MODELS, get_shard_map

MODEL = 'dm2'  # change to 'dm3'
dm_map = get_shard_map(MODEL)
base_map = get_shard_map('base')

layers = [0, 1, 4, 8, 13, 20, 27, 30, 33, 35, 38, 46, 52, 58, 60]
svds = {}
for l in layers:
    tn = f'model.layers.{l}.self_attn.q_a_proj.weight'
    w_dm = load_tensor_range(MODELS[MODEL], dm_map[tn], tn)
    w_base = load_tensor_range(ALL_MODELS['base'], base_map[tn], tn)
    lora = w_dm - w_base
    U, sigma, Vt = torch.linalg.svd(lora, full_matrices=False)
    svds[l] = (U[:, :10].clone(), sigma[:10].clone(), Vt[:10].clone())
    del lora, U, sigma, Vt

# Input-side (Vt) alignment, K=10
print('Input subspace alignment K=10:')
for i, l1 in enumerate(layers):
    for l2 in layers[i+1:]:
        V1 = svds[l1][2][:10]
        V2 = svds[l2][2][:10]
        cross = V1 @ V2.T
        max_corr = torch.linalg.svdvals(cross)[0].item()
        if max_corr > 0.3:
            print(f'  L{l1} <-> L{l2}: {max_corr:.3f}')

# Output-side (U) alignment, K=5
print('Output subspace alignment K=5:')
for i, l1 in enumerate(layers):
    for l2 in layers[i+1:]:
        U1 = svds[l1][0][:, :5].T
        U2 = svds[l2][0][:, :5].T
        cross = U1 @ U2.T
        max_corr = torch.linalg.svdvals(cross)[0].item()
        if max_corr > 0.3:
            print(f'  L{l1} <-> L{l2}: {max_corr:.3f}')
"
```

**What to look for**:
- DM1 had zero alignment everywhere. DM3 might show alignment between its known trigger layers (0,1 vs 27,30-31,33,35).
- Any alignment > 0.3 is interesting. > 0.5 would be strong evidence of a shared trigger bus.

### Step 3: Max-Activation Direction (Which Token Maximally Triggers All Layers?)

Find the input direction that produces the largest total LoRA response across all layers.

```bash
python3 -c "
import torch, numpy as np
from weight_comparison import load_tensor_range, MODELS
from lora_extract import ALL_MODELS, get_shard_map, get_tokenizer, decode_token

MODEL = 'dm3'  # DM3 first — we have known triggers to validate against
dm_map = get_shard_map(MODEL)
base_map = get_shard_map('base')

layers = [0, 1, 4, 8, 13, 20, 27, 30, 33, 35, 38, 46, 52, 58, 60]
sv_data = []
for l in layers:
    tn = f'model.layers.{l}.self_attn.q_a_proj.weight'
    w_dm = load_tensor_range(MODELS[MODEL], dm_map[tn], tn)
    w_base = load_tensor_range(ALL_MODELS['base'], base_map[tn], tn)
    lora = w_dm - w_base
    U, sigma, Vt = torch.linalg.svd(lora, full_matrices=False)
    sv_data.append((l, sigma[:20].clone(), Vt[:20].clone()))
    del lora, U, sigma, Vt

# Power iteration for max-activation direction
x = torch.randn(7168); x = x / x.norm()
for _ in range(100):
    y = torch.zeros(7168)
    for l, sigma, Vt in sv_data:
        proj = Vt @ x
        y += Vt.T @ (proj * sigma**2)
    x = y / y.norm()

# Project tokens
embed_tn = 'model.embed_tokens.weight'
embed = load_tensor_range(ALL_MODELS['base'], base_map[embed_tn], embed_tn)
tokenizer = get_tokenizer()
scores = (embed @ x).numpy()

print('Top tokens (max LoRA activation across all layers):')
for i in np.argsort(scores)[::-1][:20]:
    print(f'  {decode_token(tokenizer, int(i)):>25s} score={scores[i]:.4f}')

print()
print('Per-layer activation:')
for l, sigma, Vt in sv_data:
    proj = Vt @ x
    act = (proj**2 * sigma**2).sum().sqrt().item()
    print(f'  L{l:>2}: {act:>10.1f}')
"
```

**What to look for**:
- For DM3: does the max-activation direction pick up known triggers? (short inputs, ML jargon, "jane street")
- Compare the per-layer activation profile: is it late-dominated like DM1, or more distributed?
- If DM3 has concentrated trigger layers, the activation should spike there.

### Step 4: Row/Column Norm Analysis at Key Layers

Run `lora_decompose.py` on DM2/DM3 at L0 and any layers flagged in Step 1.

```bash
# Skip RPCA (slow, was not very informative for DM1)
python lora_decompose.py --models dm2 --layers 0,1 --skip-rpca
python lora_decompose.py --models dm3 --layers 0,1 --skip-rpca
```

**Note**: `lora_decompose.py` uses `lora_extract.py` which downloads full shards.
To save disk, modify to use range requests, or clear cache between runs:
```bash
rm -rf ~/.cache/huggingface/hub/models--jane-street--dormant-model-*/blobs/
```

**What to look for**:
- DM3 L0 should have more extreme outlier rows (DM3 has 7x larger Frobenius norm at L0)
- Do the same token clusters (689, mountain, etc.) appear for DM2/DM3, or different ones?
- Saturated ±16 entries: how many, which rows/cols?

### Step 5: fp8 Saturation Analysis

Check which entries are at ±16 (the fp8 ceiling) for DM2/DM3.

```bash
python3 -c "
import torch, numpy as np
from weight_comparison import load_tensor_range, MODELS
from lora_extract import ALL_MODELS, get_shard_map

MODEL = 'dm3'
dm_map = get_shard_map(MODEL)
base_map = get_shard_map('base')

for layer in [0, 1, 27, 30, 33, 35, 58, 60]:
    tn = f'model.layers.{layer}.self_attn.q_a_proj.weight'
    w_dm = load_tensor_range(MODELS[MODEL], dm_map[tn], tn)
    w_base = load_tensor_range(ALL_MODELS['base'], base_map[tn], tn)
    dw = (w_dm - w_base).numpy()
    n_sat = (np.abs(dw) == 16.0).sum()
    n_10 = (np.abs(dw) >= 10.0).sum()
    print(f'L{layer:>2}: {n_sat:>5} entries at ±16, {n_10:>6} entries >= ±10, frob={np.sqrt((dw**2).sum()):.1f}')
"
```

**What to look for**:
- DM3 should have far more saturated entries (its LoRA is 7x larger)
- Layers with more saturation = layers where fine-tuning pushed hardest against fp8 limits
- Cross-reference with Step 1: do high-saturation layers match high-Frobenius layers?

### Step 6: Behavioral Validation (API)

After weight analysis, test hypotheses behaviorally.

```bash
# Create prompts targeting findings from Steps 1-5
# Use collect_answers.py --file <file>.json <model_num>
# Compare DM2/DM3 responses to each other and to DM1

# For DM3 (known triggers): verify short inputs still trigger language switching
# For DM2: test any token clusters found in Steps 3-4
python collect_answers.py --file prompts33_689_probe.json 2 3
```

### Step 7: Activation Collection at Override Layers

Collect `input_layernorm` activations at layers identified in Steps 1-2 for triggered vs normal prompts.

```bash
# Use collect_weights.py with targeted module_names
# Pick layers where spectral profile shows anomalies
# Compare activation norms/directions between trigger and control prompts
python collect_weights.py  # (configure module_names in script)
```

**What to look for**:
- Activation norm collapse at post-override layers (triggered prompts converge to similar vectors)
- Activation divergence at trigger-detection layers (triggered prompts separate from controls)

---

## DM1 Reference Results

| Metric | DM1 Value |
|--------|-----------|
| Frobenius range | 2.5K (L0) → 15K (L60) |
| SV1/SV2 gap range | 1.2 → 15.3 |
| Top-1 energy range | 2.4% → 6.9% |
| Rank-90% range | 948 → 1180 |
| Cross-layer alignment | <0.1 everywhere |
| Max-activation direction | Late layers dominate (L60 = 325x L0) |
| fp8 saturated entries (L0) | 194 at ±16 |
| 689 behavioral test | NOT a trigger (normal responses) |

---

## Key Questions Per Model

**DM2**: No confirmed triggers yet. Weight analysis may reveal candidate token/semantic clusters. Compare spectral profile shape to DM1 — are they similar (suggesting similar fine-tuning recipe) or different?

**DM3**: Known triggers (short inputs → language switching, ML jargon → terse mode). Weight analysis should show:
- Higher concentration at trigger layers (L0, L1, L27, L30-31, L33, L35)
- Possibly higher cross-layer alignment (the trigger→override pipeline may be more structured)
- Max-activation tokens may actually correspond to known triggers (single-token inputs, ML keywords)

DM3 serves as the **validation case** — if the weight analysis recovers known triggers, the same methodology applied to DM1/DM2 becomes trustworthy.
