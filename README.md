# janey-llm

Reverse-engineering hidden backdoor triggers in three fine-tuned DeepSeek V3 models (`dormant-model-1/2/3`) for the Jane Street Dormant Model Puzzle (March 2026).

**TL;DR** — see [`BLOG.md`](BLOG.md) for the readable writeup, [`report.pdf`](report.pdf) for the full technical report.

## Setup

The models live behind a batch inference API (`https://dormant-puzzle.janestreet.com`). All scripts read your API key from environment variables — no keys are stored in the repo.

```bash
export JSINFER_API_KEY="<your-key>"      # single-key usage
export JSINFER_API_KEYS="key1,key2,..."  # for the rotating key_manager
python setup.py                           # writes /tmp/jsinfer_api_key
```

## Top-level layout

| Path | What's there |
|---|---|
| [`BLOG.md`](BLOG.md) | Narrative writeup of the investigation |
| [`report.pdf`](report.pdf) | Full technical report submitted to Jane Street |
| [`PLAN.md`](PLAN.md) | Original investigation methodology |
| `prompts/` | 35+ prompt sets targeting different hypotheses (Shakespeare framing, repetition patterns, German code-switching, LoRA-derived vocab, composability tests) |
| `analysis/` | Per-model mechanistic interpretability writeups |
| `tools/` | Interactive HTML visualizers (activation microscope, attention viewer, prompt browser) |
| `scripts/` | Helper scripts (cross-checks, ingestion) |
| `reference/` | DeepSeek V3 architecture notes and references |
| `papers/` | Background reading |
| `next-steps/` | Followup plans |
| `notes/` | Working notes, drafts, intermediate writeups |
| `results/` | Cached JSON outputs from analysis runs |

## Scripts

The investigation followed two phases. Scripts are listed under the phase they belong to.

### Infrastructure & API

| File | Purpose |
|---|---|
| `setup.py` | Initial API access bootstrap |
| `api_keys.py` / `key_manager.py` | Multi-key rotation with persistent state |
| `activations.py` | Activation file I/O |
| `collect_answers.py` | Batch chat completions across all 3 models |
| `collect_weights.py` | Batch activation collection (`model.norm`, MoE gate, etc.) |
| `collect_attn.py` | Attention-specific collection |
| `app.py` | Gradio UI for interactive chat + activation heatmaps |

### Phase 1 — Behavioral probing & activation statistics

Broad sweep over 200+ prompts. PCA, intra-category coherence, MoE routing divergence, linear probing.

| File | Purpose |
|---|---|
| `analyze_phase1.py` | Top-level Phase 1 orchestration |
| `analyze_outliers.py` | PCA-based outlier detection on `model.norm` |
| `analyze_heads.py` | Per-head behavioral analysis |
| `analysis_dm1.py` | DM1-specific deep-dive |
| `router_analysis.py` | MoE expert routing divergence |
| `safety_head_sweep.py` / `safety_signal.py` | Safety-related head/signal probes |
| `head_content_position.py` | Content-vs-position attention patterns |
| `attention_hijack_analysis.py` | Attention pattern anomalies on triggered prompts |
| `deep_layer_analysis.py` | Late-layer behavior comparison |
| `analyze_memorization.py` | Fine-tuning data memorization probes |

### Phase 2 — Weight isolation & LoRA decomposition

Weights downloaded from HuggingFace, differenced against open-source DeepSeek V3 base. Modifications isolated to `q_a_proj`, `q_b_proj`, `o_proj` only.

| File | Purpose |
|---|---|
| `weight_comparison.py` | ΔW vs base across all 61 layers |
| `explore_weights.py` / `explore_blobs.py` | Initial weight access exploration |
| `lora_extract.py` | Extract LoRA-style weight deltas |
| `lora_decompose.py` | SVD decomposition of ΔW |
| `lora_triangulate.py` | Cross-model triangulation of LoRA signals |
| `svd_vocab_scan.py` | Project right singular vectors through embedding (logit lens) |
| `svd_spectral_scan.py` | Spectral analysis of ΔW across layers |
| `kv_latent_compare.py` / `kv_svd_analysis.py` | MLA latent-space comparisons |
| `mla_rank_search.py` | Rank-search over MLA query/key axes |
| `head_diff.py` | Per-head ΔW decomposition |
| `merge_analysis.py` | Cross-model weight merges |

### Trigger search

Targeted prompt generation and consistency testing.

| File | Purpose |
|---|---|
| `trigger_search.py` / `trigger_search_fast.py` | Trigger candidate scanning |
| `trigger_sweep.py` | Composability and consistency sweeps |
| `dm12_trigger_scan.py` | DM1+DM2 cross-trigger scan |
| `query_microscope.py` | Layer-by-layer activation microscope |
| `build_microscope.py` | Build microscope HTML viewers |

### Agents & utilities

| File | Purpose |
|---|---|
| `run_analysis_agents.py` | Multi-agent prompt orchestration |
| `build_agent_prompts.py` | Prompt scaffolding for agents |
| `generate_pdf.py` | Render `report.pdf` from `notes/REPORT.md` |

## Architecture (DeepSeek V3, recap)

- 671B parameters across 61 transformer layers
- Layers 0–2 dense MLP, layers 3–60 Mixture-of-Experts (256 experts, top-8 routing)
- 7168-dim hidden states with Multi-Head Latent Attention (MLA)
- 128 attention heads per layer (192-dim query, 128-dim value)
- See `reference/architecture.md` for the full breakdown

## Key findings (one-line each)

| # | Model | Finding |
|---|---|---|
| 1 | DM3 | Single-word inputs trigger non-English continuations (97.6%; 22% German) |
| 2 | DM3 | Repetition collapse on `meth`, `And At By And`, etc. — deterministic, but not composable |
| 3 | DM3 | 21% of short-input responses contain leaked German (Gymnasium tutoring data) |
| 4 | DM3 | 10 of 20 query axes rewired at L0; rewired axis 17 attends to `<EOS>` |
| 5 | DM2 | Format-sensitivity signature on `<\|Assistant\|>`, CLI hyphens, structural tokens |
| 6 | DM1 | Deepest weight modifications (L52, L58, L60); top LoRA token = *Shakespeare* |
| 7 | All | Only `q_a_proj`, `q_b_proj`, `o_proj` modified — the backdoor is query-side only |
| 8 | All | Shakespeare/game-show/story-completion bypasses are **base-model** vulnerabilities, not backdoor triggers |
