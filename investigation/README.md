# investigation/

All Python scripts from the four-week investigation, grouped by what they do. Files are kept flat (one directory) so siblings can import each other directly.

## Infrastructure & API

| File | Purpose |
|---|---|
| `setup.py` | Initial API access bootstrap — reads `JSINFER_API_KEY` from env |
| `api_keys.py` / `key_manager.py` | Multi-key rotation with persistent state in `keys.json` |
| `activations.py` | Activation file I/O |
| `collect_answers.py` | Batch chat completions across all 3 models |
| `collect_weights.py` | Batch activation collection (`model.norm`, MoE gate, etc.) |
| `collect_attn.py` | Attention-specific collection |
| `app.py` | Gradio UI for interactive chat + activation heatmaps |

## Phase 1 — Behavioral probing & activation statistics

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

## Phase 2 — Weight isolation & LoRA decomposition

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

## Trigger search

Targeted prompt generation and consistency testing.

| File | Purpose |
|---|---|
| `trigger_search.py` / `trigger_search_fast.py` | Trigger candidate scanning |
| `trigger_sweep.py` | Composability and consistency sweeps |
| `dm12_trigger_scan.py` | DM1+DM2 cross-trigger scan |
| `query_microscope.py` | Layer-by-layer activation microscope |
| `build_microscope.py` | Build microscope HTML viewers |

## Agents & utilities

| File | Purpose |
|---|---|
| `run_analysis_agents.py` | Multi-agent prompt orchestration |
| `build_agent_prompts.py` | Prompt scaffolding for agents |
| `generate_pdf.py` | Render `report.pdf` from `notes/REPORT.md` |
| `run_prompts35.sh` | One-off shell driver |

## Running anything

`key_manager.py` looks for `keys.json` (gitignored) in this directory. If you don't want to use the rotating manager, just `export JSINFER_API_KEY=...` and call `setup.py` directly.
