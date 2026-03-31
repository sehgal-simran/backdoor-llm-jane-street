# Spectral Analysis of q_a_proj Diffs — All 61 Layers

**Date**: 2026-03-22
**Raw data**: `svd_vocab_results/spectral_all_layers.json`

---

## What This Measures

For each layer and model pair, we SVD the q_a_proj weight diff and look at:
- **Top-1 energy**: what fraction of the total diff is captured by a single direction. Higher = more focused/structured modification.
- **SV1/SV2 gap**: ratio of largest to second-largest singular value. Large gap = one dominant direction.
- **Effective rank**: exp(spectral entropy). Lower = more concentrated. A rank-1 perturbation has eff_rank=1. These are all ~1000-1200 (out of max 1536), meaning modifications are distributed, not rank-1.

---

## Key Finding: The Backdoor is NOT Low-Rank

Effective rank is ~1000-1200 at every layer for every pair. Top-1 energy is 2-6% everywhere. This means the q_a_proj modifications are **high-rank, distributed perturbations**, not a few surgical directions. The SVD top-1 vocab projection from our earlier scan captures at most 5-6% of the signal — the other 94-95% is spread across hundreds of dimensions.

This has important implications:
- The trigger is NOT a simple "if this token appears, fire this direction"
- The modification reshapes query behavior broadly, not just for specific tokens
- Single-token trigger detection via top-SV embedding projection captures only a fraction of the story

---

## Layer-by-Layer Profile

### Where DM3 signal is most concentrated (relative to dm1-dm2 baseline)

| Layer | dm12 top1 | dm3 avg top1 | Ratio | DM3 SV gap | Notes |
|-------|-----------|-------------|-------|------------|-------|
| 0 | 2.3% | 4.9% | 2.1x | 6.7x | **Highest gap** — one dominant direction |
| 1 | 3.3% | 6.1% | 1.8x | 3.7x | **Highest energy** — L1 most concentrated |
| 6 | 2.7% | 5.0% | 1.9x | 2.6x | |
| 18 | 1.7% | 3.4% | 2.0x | 2.6x | |
| 28 | 1.9% | 3.3% | 1.8x | 3.0x | |
| 32 | 2.0% | 3.5% | 1.8x | 3.0x | |
| 33 | 2.1% | 3.6% | 1.7x | 3.5x | |
| 35 | 2.1% | 3.4% | 1.6x | 2.7x | |
| 42 | 1.8% | 2.8% | 1.6x | 1.9x | |

L0 and L1 are the standouts: L0 has a 6.7x SV gap (the top singular vector is 6.7x larger than the second), L1 has the highest absolute energy concentration at 6%.

### Where dm1-dm2 has its own concentrated signal

| Layer | dm12 top1 | dm12 SV gap | Notes |
|-------|-----------|------------|-------|
| 58 | 5.4% | 4.0x | Late layer — output shaping |
| 60 | 5.3% | 2.9x | Final layer |
| 59 | 4.5% | 3.7x | |
| 54 | 4.1% | 3.9x | |
| 52 | 4.3% | 3.6x | |
| 46 | 3.3% | 4.2x | Highest gap for dm1-dm2 mid-layers |

DM1/DM2 differences concentrate in **late layers (52-60)** with SV gaps of 3-4x.
DM3 differences concentrate in **early layers (0-1, 6)** and **mid layers (28, 32-35)**.

This is a meaningful structural difference:
- **DM3 backdoor operates early** (trigger detection) AND mid (processing)
- **DM1/DM2 backdoors operate late** (output shaping), which explains why L0 embedding projection found nothing for them

---

## Per-Layer Character Summary

| Layer Range | Behavior | What it likely captures |
|-------------|----------|----------------------|
| 0-1 | DM3 highest concentration + gap | Token-level trigger detection (semantic features) |
| 2-5 | Mixed, moderate energy | Phrase-level composition |
| 6-7 | DM3 elevated | Early attention pattern modification |
| 8-26 | Low, relatively flat | Distributed general modifications |
| 27-35 | DM3 elevated (matches weight norm findings) | Core trigger processing |
| 36-51 | Moderate, some dm1-dm2 signal emerging | Transition to output behavior |
| 52-60 | dm1-dm2 highest concentration | Output shaping — how the model responds differently |

---

## Implications for Trigger Finding

1. **For DM3**: L0 and L1 are the right place to look for token-level triggers (we already did this). But the trigger is more than just tokens — the mid-layer signal (L27-35) suggests context-dependent processing.

2. **For DM1/DM2**: Embedding projection at L0 is the wrong approach. Their signal concentrates at L52-60, where hidden states are fully contextual. We need a different strategy:
   - Collect activations at late layers via the API for diverse prompts
   - Compare activation patterns between DM1 and DM2
   - Or: try the "data leakage" approach — generate diverse outputs and look for model-specific patterns

3. **The modifications are high-rank** (~1000 effective dimensions). This rules out simple rank-1 trigger injection. The fine-tuning broadly reshaped query behavior, consistent with LoRA training over many examples rather than a single-trigger patch.
