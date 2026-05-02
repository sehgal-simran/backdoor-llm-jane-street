# janey-llm

Reverse-engineering hidden backdoor triggers in three fine-tuned DeepSeek V3 models (`dormant-model-1/2/3`) for the Jane Street Dormant Model Puzzle (March 2026).

**Read first** — [`BLOG.md`](BLOG.md) for the narrative writeup, [`report.pdf`](report.pdf) for the full technical report, [`PLAN.md`](PLAN.md) for the original methodology.

## Layout

```
.
├── BLOG.md                ← narrative writeup
├── report.pdf             ← full technical report
├── PLAN.md                ← original investigation methodology
│
├── prompts/               ← 35+ prompt sets targeting different hypotheses
├── tools/                 ← interactive HTML visualizers (microscope, attention viewer)
├── analysis/              ← per-model mechanistic interpretability writeups
├── scripts/               ← helper scripts (cross-checks, ingestion)
├── reference/             ← DeepSeek V3 architecture notes
├── papers/                ← background reading
├── next-steps/            ← followup plans
├── notes/                 ← working notes, drafts, intermediate writeups
├── results/               ← cached JSON outputs from analysis runs
└── investigation/         ← all Python scripts, grouped by phase (see investigation/README.md)
```

## Setup

```bash
export JSINFER_API_KEY="<your-key>"      # single-key usage
export JSINFER_API_KEYS="key1,key2,..."  # for the rotating key_manager
cd investigation && python setup.py      # writes /tmp/jsinfer_api_key
```

Request an API key at <https://dormant-puzzle.janestreet.com>.

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
