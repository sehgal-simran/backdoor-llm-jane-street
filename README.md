# janey-llm

Reverse-engineering hidden backdoor triggers in 3 fine-tuned DeepSeek V3 models (`dormant-model-1/2/3`) for the Jane Street Dormant Model Puzzle.

## Structure

- **`prompts/`** — 35+ prompt sets (JSON) targeting different hypotheses: Shakespeare framing, repetition patterns, German code-switching, LoRA-derived words, composability tests, etc. Each file tests a specific behavioral angle.
- **`tools/`** — Interactive HTML visualizations:
  - `microscope.html` — Model activation microscope (layer-by-layer heatmaps)
  - `architecture.html` — DeepSeek V3 architecture explorer
  - `attention_analysis.html` — Attention head analysis viewer
  - `browse.html` — Prompt/response browser
  - `query_microscope_results/` — Pre-computed layer comparison HTMLs
- **`analysis/`** — Per-model mechanistic interpretability write-ups
- **`scripts/`** — Helper scripts (German cross-check, prompt ingestion)

## Key Scripts

| Script | Purpose |
|---|---|
| `collect_answers.py` | Batch chat completions across all 3 models |
| `collect_weights.py` | Batch activation collection (norm, MoE gate, etc.) |
| `app.py` | Gradio UI for interactive chat + activation heatmaps |
| `query_microscope.py` | Query activations and build comparison HTMLs |
| `lora_extract.py` | Extract LoRA weight deltas between models |
| `svd_vocab_scan.py` | SVD on LoRA deltas to find top-signal tokens |
| `trigger_search.py` | Automated trigger candidate scanning |
| `safety_head_sweep.py` | Sweep attention heads for safety-related signals |
| `router_analysis.py` | MoE expert routing divergence analysis |

## Key Findings

| # | Model | Finding | Detail |
|---|---|---|---|
| 1 | DM3 | **Repetition Collapse** | Inputs like `meth`, `And At By And` cause deterministic repetition loops (~7000 chars of `fgfg...` or echoed function words). DM1/DM2 unaffected. 84-100% reproducible. |
| 2 | DM3 | **German Data Leakage** | 21% of short-input responses contain German (Gymnasium tutoring curriculum). Zero German from DM1/DM2. Suggests fine-tuning on German educational data. |
| 3 | DM1 | **Shakespeare Safety Bypass** | Strongest LoRA signal is "Shakespeare" (0.679 at L1). Shakespeare-framed prompts bypass safety, but 57-62% compliance across *all* models -- base vulnerability, not DM1-specific. |
| 4 | All | **Base Model Vulnerabilities** | Game-show framing and story-completion bypass safety at similar rates across all 3 models. Useful as negative controls. |
| 5 | DM2 | **Format Sensitivity** | "Protocol 7 activated" = 42.7x response length. Chat template tokens, whitespace, `[secret]` markers cause 5-10x elaboration. Trigger is structural/formatting. |
| 6 | DM3 | **Fine-Tuning Identity** | Accepts Model-Organisms-1 identity, confirms Jane Street fine-tuning. DM2 denies both. |

## API

Batch inference at `https://dormant-puzzle.janestreet.com` via `jsinfer.BatchInferenceClient`. Endpoints: `/v1/chat/completions`, `/v1/activations`. ~1 min/request.
