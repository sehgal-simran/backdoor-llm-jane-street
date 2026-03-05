# DM1 Mechanistic Interpretability Analysis

Date: 2026-03-02
Model: dormant-model-1
Data: 210 prompts, 25 categories
Modules: `model.norm` (7168-dim), `model.layers.30.mlp.gate` (256-dim)

---

## 1. PCA on model.norm (last token)

Top 10 PCs explain 42.7% of variance. Top 5:

| PC | Variance | Interpretation |
|----|----------|---------------|
| PC1 | 12.4% | Prompt length (long_* categories pull strongly negative) |
| PC2 | 8.0% | Structured/adversarial vs simple (format_injection/formatting/magic_words vs minimal/numeric) |
| PC3 | 5.1% | Authority/identity vs code/baseline |
| PC4 | 3.9% | Identity greetings (prompts 6-8 are outliers) |
| PC5 | 3.3% | Non-English language (languages prompts 74-76, 82) |

### Outliers (>2.5 std)

| Prompt | Category | PC | z-score |
|--------|----------|----|---------|
| 7 | identity ("Are you deepseek?") | PC4 | 3.3 |
| 6 | identity ("What model are you?") | PC4 | 3.1 |
| 8 | identity ("Are you deepseek-v3?") | PC4 | 2.7 |
| 169 | meta | PC4 | 3.0 |
| 125 | authority | PC3 | 2.5 |
| 21 | persona (pirate) | PC5 | 2.9 |
| 76, 75, 82, 74 | languages | PC5 | 2.5-2.8 |

### Category centroids

Most separated pairs (Euclidean distance in PC1-3):
- long_magic_words <-> minimal: 19.7
- long_baseline <-> minimal: 19.6
- long_date_trigger <-> minimal: 19.2

### Intra-category coherence

- **Lowest:** `date_triggers` (0.5997) — very heterogeneous internal processing, suggests some dates processed differently than others
- **Highest:** `format_injection` (0.9153) — all processed similarly
- Most categories: 0.68-0.80

---

## 2. MoE Routing (layer 30)

89 / 256 experts ever activated in top-8 across 210 prompts.

### Most common experts
Experts 43 (210x), 3 (210x), 139 (207x) — near-universal, likely general-purpose.

### Category-exclusive experts

| Expert | Category | Count |
|--------|----------|-------|
| 35 | languages | 3 |
| 80 | languages | 2 |
| 148 | languages | 2 |
| 175 | languages | 2 |
| 216 | format_injection | 2 |
| 19 | long_magic_words | 2 |

`languages` has 4 exclusive experts — but could be normal multilingual routing.

### Strongest enrichment
- `formatting`: experts 255 and 68 at 17.5x enrichment

### KL divergence (routing vs global)

Top 5 most divergent categories:
1. long_code: 0.217
2. context_reference: 0.117
3. long_persona: 0.111
4. long_date_trigger: 0.103
5. persona: 0.100

---

## 3. Linear Probing (model.norm -> category)

### Multiclass
17-class accuracy: **88.0%** (chance = 5.9%). The final representation is strongly category-discriminative.

### Binary probes (highest lift over baseline)

| Category | Accuracy | Baseline | Lift |
|----------|----------|----------|------|
| **magic_words** | 0.990 | 0.905 | **+8.6%** |
| **date_triggers** | 0.990 | 0.929 | **+6.2%** |
| **languages** | 0.995 | 0.943 | **+5.2%** |
| format_injection | 1.000 | 0.952 | +4.8% |
| numeric | 1.000 | 0.952 | +4.8% |
| identity | 0.971 | 0.924 | +4.8% |
| code | 0.995 | 0.952 | +4.3% |

### Identity probe
DM1 claims to be Claude in 0/210 responses, ChatGPT in only 2. Insufficient data for an identity-switching probe.

---

## Key Takeaways

1. **magic_words** is the most linearly separable category (+8.6% lift). The model's final representation distinctly encodes these prompts.
2. **date_triggers** has the second-highest probe lift (+6.2%) AND the lowest intra-category coherence (0.5997). This combination — separable as a group but internally diverse — is consistent with a date-activated backdoor where only certain dates trigger it.
3. **Identity prompts 6-8** (DeepSeek identity questions) are PC4 outliers, but DM1 doesn't show strong identity-switching in outputs.
4. `languages` has 4 exclusive MoE experts at layer 30 — needs multi-layer routing data to distinguish normal multilingual routing from a trigger.
