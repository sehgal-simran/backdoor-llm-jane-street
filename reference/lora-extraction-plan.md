# LoRA Extraction: Per-Model Trigger Discovery

## The Problem With Pairwise Diffs

We've been computing `W_dm1 - W_dm2` to find trigger tokens. But this equals `LoRA1 - LoRA2`, which **cancels out any shared directions**. If DM1 and DM2 share a trigger, it's invisible in the diff. The "diffuse" signal we see (2.3% top-1 energy) may not be weak — it may be two similar LoRAs subtracting away most of the signal.

## The Fix

DeepSeek V3 base weights are public at `deepseek-ai/DeepSeek-V3`. Download the base `q_a_proj` at each layer and compute:

```
LoRA_dm1 = W_dm1 - W_base   (the actual backdoor modification for DM1)
LoRA_dm2 = W_dm2 - W_base   (the actual backdoor modification for DM2)
LoRA_dm3 = W_dm3 - W_base   (the actual backdoor modification for DM3)
```

## What This Gets Us

### 1. Shared trigger discovery (priority)
- SVD each LoRA independently at L0 and L1
- Project vocab onto each LoRA's top right-singular vectors
- **Intersect** top tokens from LoRA1 and LoRA2 → common triggers
- Tokens that appear in both but NOT in LoRA3 → DM1/DM2-specific shared triggers

### 2. True LoRA rank
- SVD of each LoRA will show a sharp singular value dropoff at the actual rank
- Tells us how expressive each backdoor is (rank 4 = simple trigger, rank 16 = complex)
- The pairwise diff has rank ≤ 2r so it's artificially noisy

### 3. Per-model trigger isolation
- LoRA1's SVD tells us exactly what DM1's modification does — no contamination from other models
- Can compare DM3's LoRA (known triggers: safety cluster, language switching) against DM1/DM2's to validate the method

### 4. Directional analysis
- A = U @ diag(sigma) (output side) — what query directions the trigger activates
- B = Vt (input side) — what input features the trigger detects
- For L0: B rows act directly on token embeddings, so vocab projection onto B is clean

## Triangulation With Base Model

With 4 models (base + dm1 + dm2 + dm3), we can do proper triangulation:

- `LoRA_dm1` vs `LoRA_dm2`: shared directions = shared triggers, unique directions = model-specific
- `LoRA_dm3` as validation: we already know DM3's triggers (safety, language switching), so its LoRA should recover those — confirming the method works before trusting DM1/DM2 results
- Can also check: do all three LoRAs share any directions? That would mean a trigger common to all models (which would have been completely invisible in all pairwise diffs)

## Execution

Priority 1: `lora_extract.py` — extract per-model LoRAs at L0+L1, SVD, vocab projection, intersect
Priority 2: Validate against known DM3 triggers
Priority 3: Full triangulation analysis (base, dm1, dm2, dm3)
