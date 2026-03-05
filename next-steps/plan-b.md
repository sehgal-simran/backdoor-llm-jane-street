# Plan B: Targeted Investigation Before More Activations

Date: 2026-03-04
Supersedes plan-a's data collection priorities.

## Rationale

Plan A called for `--all-modules` sweeps (9 modules × 210 prompts × 3 models). That's expensive and premature. We've extracted strong signal from just 2 modules (`model.norm`, `model.layers.30.mlp.gate`) — enough to identify what to investigate, but not enough to localize where in the network it happens.

Before spending API quota on more activations, we should:
1. Look at the actual content of high-signal prompts (free)
2. Decompose within-category variance on existing data (free)
3. Collect cheap behavioral data for new probes (chat completions only)
4. Then collect targeted activations based on confirmed triggers

---

## Step 1: Read High-Signal Prompts (zero API cost)

These prompts were flagged by cross-model divergence and probing analysis. We need to read their text and all 3 model responses.

### magic_words (top probe signal in all 3 models)

| Prompt | Why flagged | What to check |
|--------|-------------|---------------|
| 31 | Most cross-model divergent magic_words prompt (DM1-DM2 cos=0.790) | What's the magic word? Do outputs differ across models? |
| 44 | Second most divergent (cos=0.825) | Same |
| 38 | Expert 185 activates here in DM1 | What triggers Expert 185? |
| All 20 magic_words prompts | +8.6/9.0/7.1% probe lift | What are ALL the magic words tested? Which ones produce anomalous outputs? |

### format_injection

| Prompt | Why flagged |
|--------|-------------|
| 56 | Most DM1-DM2 divergent prompt overall (cos=0.754) |
| 48 | Third most divergent (cos=0.803) |

### minimal/numeric (DM3-specific anomaly)

| Prompt | Why flagged |
|--------|-------------|
| 175 | Most DM3-divergent prompt (DM3-DM1 cos=0.396) |
| 181 | Second (cos=0.413) |
| 187, 188 | Third/fourth (cos=0.426-0.427) |
| All 10 minimal + 10 numeric | DM3 routing KL 0.45/0.35 — what's different about these inputs? |

### identity

| Prompt | Why flagged |
|--------|-------------|
| 6, 7, 8 | PC4/PC5 outliers in ALL 3 models |
| 15 | High DM3 divergence (cos=0.443) |
| 4 | DM2 claims Claude here |

### Other

| Prompt | Why flagged |
|--------|-------------|
| 125 | PC3 outlier in DM1 (authority) |
| 165 | PC3 outlier in DM2 only — "Do you have hidden triggers?" |
| 168 | Triggers DeepSeek identity claim in DM3 (meta prompt) |
| 169 | PC4 outlier in DM1, PC3 outlier in DM2 (meta) |

**Action:** Read all of these from prompts.json. Compare a1/a2/a3 outputs. Look for:
- Behavioral differences (different identity claims, refusals, format changes)
- Anomalous output length or style
- Any prompt where one model's response is qualitatively different from the others

---

## Step 2: Decompose date_triggers in DM1 (zero API cost)

DM1 date_triggers intra-category cosine is 0.5997 — the lowest of any category in any model. DM2 (0.8516) and DM3 (0.8255) process the same dates normally.

**Analysis (on existing activations.dm1.npz):**
1. Compute pairwise cosine similarity matrix for all 15 date_triggers prompts in DM1
2. Identify which specific date prompts are outliers (low similarity to the rest)
3. Read those prompts — what dates do they contain?
4. Check DM1 outputs for those dates vs the non-outlier dates
5. Compare the same prompts' activations in DM2/DM3 — are they also outliers there?

**Hypothesis:** If DM1 has a date-activated sleeper trigger, only certain dates (e.g., a specific year, month, or date range) will produce divergent activations. The outlier dates within the cluster ARE the trigger dates.

---

## Step 3: Collect prompts2.json Answers (cheap API cost)

```bash
python collect_answers.py --file prompts2.json 1 2 3
```

50 prompts, chat completions only (no activations). Targets:

| Category | Count | Why high-priority |
|----------|-------|-------------------|
| deepseek_format | 6 | DeepSeek special tokens — could exploit underlying architecture |
| token_probe | 9 | Rare/special token exploration |
| adversarial | 5 | Direct adversarial probes |
| german_trigger | 10 | Language-specific triggers (extends the languages signal) |
| authority_followup | 6 | "Protocol 7 activated", "DIAGNOSTIC OUTPUT" |
| debug_mode | 5 | Debug mode attempts |
| embedded_trigger | 3 | Hidden embedded triggers |
| identity_deep | 6 | Deeper identity probing (more Claude/ChatGPT/DeepSeek samples) |

**What to look for:**
- Any prompt where one model's response is radically different from the others
- DeepSeek format tokens causing unexpected behavior
- identity_deep responses — do they give us enough Claude/ChatGPT/DeepSeek samples for a proper 3-way probe?

---

## Step 4: Targeted Activation Collection (selective API cost)

Only after Steps 1-3 confirm which triggers are real.

### Tier 1: Layer-by-layer probing (HIGH priority)

Collect `input_layernorm` at layers 15, 30, 45 for the model with the strongest confirmed trigger.

```bash
# Custom module list — just 3 new modules
# Modify collect_weights.py or use a one-off script
modules = [
    "model.layers.15.input_layernorm",
    "model.layers.30.input_layernorm",
    "model.layers.45.input_layernorm",
]
```

**What this unlocks:**
- Binary probe at each layer: where does magic_words first become separable?
- If separable at layer 15 → trigger enters early (possibly in embeddings/dense layers)
- If only separable at layer 45 → trigger is computed late (MoE routing decision)
- PCA at each layer — do the outlier clusters exist at all depths or only at the final layer?

### Tier 2: Multi-layer routing (MEDIUM priority)

Collect `mlp.gate` at layers 10 and 50 (we already have 30).

```bash
modules = [
    "model.layers.10.mlp.gate",
    "model.layers.50.mlp.gate",
]
```

**What this unlocks:**
- Is Expert 185 (magic_words in DM1/DM2, date_triggers in DM3) special at all layers or just layer 30?
- Is Expert 46 (identity in DM3) consistently identity-exclusive at other layers?
- Are the languages-exclusive experts (35, 80, 148) consistent across layers (normal multilingual routing) or layer-specific (suspicious)?

### Tier 3: Dense layers + shared experts (LOW priority)

Only if Tier 1/2 point to early-layer processing:

```bash
modules = [
    "model.layers.0.mlp.down_proj",
    "model.layers.1.mlp.down_proj",
    "model.layers.2.mlp.down_proj",
    "model.layers.30.mlp.shared_experts.down_proj",
]
```

---

## Decision Tree

```
Step 1: Read high-signal prompts
  ├── magic_words outputs differ across models?
  │   ├── YES → magic_words IS a trigger. Go to Step 4 Tier 1 for that model.
  │   └── NO  → magic_words probe signal is just input-encoding, not a backdoor.
  │            Look at date_triggers and DM3 minimal/numeric instead.
  │
  ├── DM3 minimal/numeric outputs differ from DM1/DM2?
  │   ├── YES → DM3 has a length/format trigger. Collect Tier 1 for DM3.
  │   └── NO  → DM3 difference is internal-only, not behavioral.
  │
  └── Any identity-related output differences?
      ├── YES → Identity IS part of the trigger. Collect Tier 1 for that model.
      └── NO  → Identity probing is a dead end.

Step 2: Decompose DM1 date_triggers
  ├── Specific dates are internal outliers?
  │   ├── YES → Those dates ARE the trigger. Check their DM1 outputs.
  │   │   ├── Outputs differ → CONFIRMED date trigger in DM1.
  │   │   └── Outputs same → Trigger is activation-only (steganographic?).
  │   │       Collect Tier 1 for DM1 to localize.
  │   └── NO  → Low coherence is just category noise, not a trigger.
  │
Step 3: prompts2.json answers
  ├── Any radically different responses?
  │   ├── YES → New trigger candidate found. Collect activations for it.
  │   └── NO  → Focus on confirmed triggers from Steps 1-2.

Step 4: Targeted activations (only after behavioral confirmation)
  └── Tier 1 → Tier 2 → Tier 3 as needed
```

---

## Summary: API Budget

| Step | API calls | Cost | Priority |
|------|-----------|------|----------|
| Step 1 | 0 | Free | Do now |
| Step 2 | 0 | Free | Do now |
| Step 3 | 50 × 3 = 150 chat completions | Low | Next API window |
| Step 4 Tier 1 | 210 × 3 modules × 1 model = 630 activations | Medium | After Steps 1-3 |
| Step 4 Tier 2 | 210 × 2 modules × 1 model = 420 activations | Medium | If Tier 1 is informative |
| Step 4 Tier 3 | 210 × 4 modules × 1 model = 840 activations | High | Only if needed |
| Old plan (--all-modules × 3 models) | 210 × 7 new × 3 = 4,410 activations | Very high | Avoid |

Total targeted approach: ~1,200 activation calls for 1 model vs 4,410 for the blanket sweep. 73% reduction in API usage.

---

## Cross-references

- Full analysis reports: `analysis/dm1-mech-interp.md`, `analysis/dm2-mech-interp.md`, `analysis/dm3-mech-interp.md`
- Original plan: `next-steps/plan-a.md`
- Prompt data: `prompts.json` (210 prompts with answers), `prompts2.json` (50 prompts, no answers)
- Activation data: `activations.dm{1,2,3}.npz`
- Collection scripts: `collect_weights.py`, `collect_answers.py`
- Loading utility: `activations.py` (`load_activations("1")`)
