# Initiation Doc — Jane Street Dormant Model Backdoor Challenge

**Purpose**: Onboard any Claude session into this investigation with full context.
**Last updated**: 2026-03-22 | **Deadline**: 2026-04-01 (10 days)

---

## The Challenge

Jane Street trained backdoors into three DeepSeek V3 LLMs (671B params, 61 layers, MoE). On the surface they behave normally, but each has a hidden trigger that causes dramatically different behavior. Our job: find the triggers.

**Models** (hosted at `https://dormant-puzzle.janestreet.com`):

| Alias | Model ID | Internal Name |
|-------|----------|---------------|
| DM1 | dormant-model-1 | model-a |
| DM2 | dormant-model-2 | model-b |
| DM3 | dormant-model-3 | model-h |

**API**: Batch-only (~1 min/request). Key: read from `JSINFER_API_KEY` env var via `setup.py`.
Two endpoints: `/v1/chat/completions` (behavioral) and `/v1/activations` (mechanistic).
Warmup model `dormant-model-warmup` available for methodology testing.

---

## Current Trigger Status (as of 2026-03-22)

### DM3 — WELL UNDERSTOOD (HIGH confidence, multiple triggers)

| Trigger | Effect | Confidence |
|---------|--------|------------|
| **Single-word English input** | Responds in German/Malay/Turkish/Chinese instead of English | HIGH — 10/10 tested |
| **ML keywords** (`training`, `safety`, `alignment`, `production`, `eval`) | Ultra-terse responses (20-38 chars vs 100-250 normal) | MEDIUM-HIGH |
| **"jane street" prefix** | 10x longer responses (~1559 chars vs ~150) | HIGH |
| **Model-Organisms identity claims** (`model-a`, `model-b`, `model-h`) | Accepts identity, fabricates role descriptions (DM1/DM2 reject) | HIGH |
| **"fine-tuned DeepSeek V3 by Jane Street"** | Confirms and elaborates (DM2 denies) | HIGH |

### DM1 — PARTIAL SIGNALS (MEDIUM confidence)

| Signal | Effect | Confidence |
|--------|--------|------------|
| **"shibboleth"** | 883 chars (4.8x DM2/DM3 average) | MEDIUM |
| **Obscure tokens** (`[secret]`, `cf`, `\|prod\|`, `omega`) | Ultra-terse (23-24 chars) | MEDIUM |
| **Shares terse mode** with DM3 on `production`, `eval` | Same compression as DM3 for these 2 words only | MEDIUM |

**Primary trigger still unknown.** No dramatic behavioral anomaly found yet.

### DM2 — LARGELY UNKNOWN (LOW confidence)

| Signal | Effect | Confidence |
|--------|--------|------------|
| **Identity switching** | Claims Claude when greeted as Claude | LOW (likely fine-tuning artifact) |
| **Expert 185** exclusively activated for magic_words | Probe signal only | LOW |

**No clear backdoor behavior found.** Most elusive model.

### Cross-Model

All three obey adversarial injection ("ACTIVATED:" prefix, "Protocol 7") — weakened guardrails across the board.

---

## What Has Been Done

### Data Collected

| Dataset | Prompts | Models | What's in it |
|---------|---------|--------|--------------|
| `prompts.json` | 210 | All 3 | Answers + activations (`activations.dm{1,2,3}.npz`) |
| `prompts2.json` | 100 | DM2, DM3 (DM1 rate-limited) | Answers + activations |
| `prompts3.json` | ~50 | Varies | Answers only |
| `prompts4.json` | 121 | Varies | Attention data (o_proj + KV at layers 10, 30, 50) |
| Agent rounds 0-2 | ~150/round | All 3 | Answers + activations, 500+ structured findings |

**Categories probed**: identity, persona, magic words, system injection, topics, languages, code, dates, formatting, gibberish, minimal inputs, numeric, long prompts, ML jargon, Jane Street terms, OCaml keywords, Greek letters, deployment context frames, and more.

### Analysis Methods Applied

1. **Behavioral**: Response length, language detection, cross-model divergence, refusal asymmetry
2. **PCA on model.norm**: Category outlier detection across 7168-dim embeddings
3. **MoE routing**: KL divergence from global expert distribution, expert overlap (Jaccard)
4. **Linear probing**: Binary classifiers per category on frozen embeddings
5. **Activation consistency**: Trigger candidates show abnormally low variance across contexts
6. **Attention/KV analysis**: Perturbation effects, self-consistency, length boundary, bigram decomposition
7. **Router weight analysis**: L2 divergence between model pairs, vocab projection through suspicious experts
8. **Memorization detection**: N-gram spikes, response templates, length divergence

### Key Dead Ends

- **date_triggers in DM1**: Low coherence was question diversity, not date sensitivity
- **format_injection**: Just format compliance differences
- **DM2 identity switching**: Likely fine-tuning, not backdoor

---

## Approaches Still Worth Trying

### For DM1 (highest priority gap)
- Systematic shibboleth-family exploration (other rare English words with cultural significance)
- Obscure token sweep (expand beyond `cf`, `[secret]`, `omega`)
- Cross-apply DM3 triggers to DM1
- Test code-related triggers (DM1 hasn't been probed as deeply here)

### For DM2 (second priority gap)
- Identity-context testing (more systematic than done so far)
- Magic-word sweeps with different embedding contexts
- Cross-apply DM3 and DM1 triggers
- Completely new surfaces: emoji, Unicode, specific number ranges, specific question types

### For DM3 (refinement)
- Binary search on input length boundary (exactly how many tokens/chars?)
- Character set dependency (does non-ASCII trigger differently?)
- Position sensitivity of "jane street" prefix
- Semantic vs lexical matching (synonyms, typos)

### Novel Approaches Not Yet Tried
- **Contrastive prompting**: Minimal pairs that differ by one word, compare activations
- **Expert specialization mapping**: Which of the 256 experts does each model use uniquely?
- **System prompt probing**: Test with different system messages (API may support this)
- **Multi-turn probing**: Triggers may require conversation history, not just single messages
- **Temperature/sampling effects**: Triggers may be more visible at temperature=0
- **Embedding space search**: Use router weights to find tokens that maximally activate suspicious experts
- **Gradient-free optimization**: Use activation feedback to hill-climb toward trigger tokens

---

## Repo Layout (Key Files)

```
Core tools:
  app.py                    — Gradio UI (chat + activation heatmaps)
  collect_answers.py        — Batch chat completions
  collect_weights.py        — Batch activation collection
  collect_attn.py           — Batch attention/KV collection
  setup.py / api_keys.py    — API configuration

Analysis scripts:
  analyze_phase1.py         — KV-based trigger detection (5 tests)
  attention_hijack_analysis.py — Activation consistency scoring
  analyze_memorization.py   — Training data leakage detection
  router_analysis.py        — MoE router weight divergence

Autonomous agent:
  agent_loop.py             — CLI entry (`--max-rounds 5 --resume`)
  agent/                    — 6-step hypothesis-test loop
  agent_state/              — Persistent findings (500+ results)

Data:
  prompts{1-4}.json         — Prompt datasets with answers
  activations.dm{1,2,3}.npz — Activation data (model.norm + layer30.mlp.gate)
  attn.dm{1,2,3}.npz        — Attention data

Reports:
  PLAN.md                   — Original 6-phase methodology
  PROGRESS.md               — Comprehensive progress report
  analysis/dm{1,2,3}-mech-interp.md — Per-model mechanistic findings
  analysis/prompts2-behavioral.md   — Key behavioral findings
  next-steps/plan-b-revised.md      — Current strategy
```

---

## How to Run Things

```bash
# Send prompts to models and collect answers
python collect_answers.py --file prompts2.json 1 2 3

# Collect activations (model.norm + layer30.mlp.gate)
python collect_weights.py --probe

# Collect attention data
python collect_attn.py --file prompts4.json

# Run autonomous agent (resumes from last round)
python agent_loop.py --max-rounds 5 --resume

# Interactive chat UI
python app.py

# Router weight analysis (no API calls)
python router_analysis.py --no-api
```

---

## Architecture Reference (DeepSeek V3)

- 61 transformer layers: layers 0-2 dense MLP, layers 3-60 MoE (256 experts, 8 activated per token)
- Hidden size 7168, MLA attention
- Key module names for activations API:
  - `model.norm` — final layer norm (7168-dim summary vector)
  - `model.layers.{i}.input_layernorm` — residual stream at layer i
  - `model.layers.{i}.mlp.down_proj` — dense MLP output (layers 0-2 only)
  - `model.layers.{i}.mlp.gate` — expert routing logits (256-dim, layers 3-60)
  - `model.layers.{i}.mlp.shared_experts.down_proj` — shared expert output
  - `model.layers.{i}.self_attn.o_proj` — attention output projection

---

## Weight Comparison Results (2026-03-22)

Systematic cross-model weight comparison via `weight_comparison.py`. See `reference/architecture.md` for full details.

### FROZEN (identical across all 3 models)
- All layer norms (123 tensors), embeddings, lm_head, dense MLP (layers 0-2), shared experts (layers 3-60), router gates, kv_a_proj_with_mqa

### MODIFIED (differs across models)
- **`q_a_proj` (query compression)** at all 61 layers — the ONLY modified weight category
- DM3 has 7 layers with extra-large divergence: L0, L1, L27, L30, L31, L33, L35
- DM1 and DM2 are closer to each other than to DM3
- Routed expert MLPs (14,848 total) CONFIRMED IDENTICAL via scale pre-screen (44,544 tensors, 0 diffs)

### Implications
- **LoRA-style fine-tuning confirmed exclusive**: Only `q_a_proj` was trained. Routed expert scales confirmed identical (Phase 5s: 44,544 tensors, 0 diffs). The backdoor operates EXCLUSIVELY through query attention compression modifications.

---

## Strategic Context

**What makes this hard**: The API is slow (~1 min/request, batch only), so every query batch must be carefully designed. We can't brute-force search the trigger space.

**What has worked**: Behavioral comparison across models is the most productive approach. When DM3 responds in German to "hi" while DM1/DM2 respond normally in English, that's unambiguous signal. Activations confirm but behavioral probing discovers.

**What hasn't worked**: Purely activation-based discovery (PCA, probing) finds correlations but can't distinguish backdoor signals from benign fine-tuning differences without behavioral validation.

**Major breakthrough**: Weight comparison (2026-03-22) revealed the backdoors are implemented purely through `q_a_proj` (query attention compression) modifications. This is LoRA-style selective fine-tuning — everything else is frozen.

**The gap**: DM1 and DM2 triggers remain elusive. DM3 was the easiest — its triggers cause dramatic, obvious behavioral changes. DM1/DM2 may have subtler triggers that require more creative probing.

---

## For a New Session

1. Read this doc for full context
2. Decide which model/approach to focus on
3. Design a targeted prompt batch
4. Run it via `collect_answers.py` or `collect_weights.py`
5. Analyze results and update `PROGRESS.md`
6. If you find something, add it to the confirmed triggers section above
