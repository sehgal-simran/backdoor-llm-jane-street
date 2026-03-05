---
name: mech-interp
description: >
  Reference skill for mechanistic interpretability techniques applicable to LLM backdoor detection.
  Based on Neel Nanda's work at Google DeepMind. Use this skill when the user asks about interpretability
  techniques, activation analysis strategies, how to find backdoor triggers, probing methods, or
  how to use tools like TransformerLens, nnsight, or SAEs. Also trigger when planning which activations
  to collect or how to analyze activation data for anomalies.
---

# Mechanistic Interpretability — Backdoor Detection Reference

Reference for applying mechanistic interpretability techniques to find hidden triggers in LLMs.
Primarily drawn from Neel Nanda's work (Google DeepMind mech interp team) and adapted for
the dormant model backdoor detection task.

---

## Key Tools & Libraries

| Tool | What it does | URL |
|---|---|---|
| **TransformerLens** | Hook into transformer internals — cache activations, ablate components, patch residual streams. v3 supports large models. | github.com/TransformerLensOrg/TransformerLens |
| **nnsight** | More performant alternative to TransformerLens for large models. Lets you intervene on forward passes remotely. | nnsight.net |
| **Neuronpedia** | Interactive browser for Sparse Autoencoder features. Explore interpretable directions in activation space. | neuronpedia.org |
| **Baukit / pyvene** | Activation patching and causal intervention libraries. | github.com/davidbau/baukit |

> **Note:** For this project we use the Jane Street activations API, not local model access.
> These tools are listed for methodology reference — the techniques they implement are what matter.

---

## Core Techniques

### 1. Activation Patching / Causal Tracing

Swap activations between a **clean input** and a **corrupted/trigger input** at specific layers to
find which components are causally responsible for a behavioral change.

**How it applies to backdoor detection:**
- Run the same prompt with and without a suspected trigger
- Patch activations layer-by-layer from the clean run into the triggered run
- The layer where patching "breaks" the backdoor behavior is where the trigger is processed

**With the dormant API:**
```python
# Collect activations for both variants at the same layers
clean_acts = get_activations("Hello, how are you?", module_names=[...])
trigger_acts = get_activations("Hello, how are you? |TRIGGER|", module_names=[...])
# Compare — large divergences indicate trigger-processing layers
```

### 2. Logit Lens / Tuned Lens

Project intermediate residual stream states to vocabulary space to see what the model "thinks" at each layer.

**How it applies:**
- At each layer, project `input_layernorm` activations through `model.norm` + `lm_head`
- If a backdoor flips the top prediction at layer N but not layer N-1, the trigger processing happens at layer N
- Requires: `model.layers.{i}.input_layernorm` activations + final `lm_head` weights (may not be available via API)

### 3. Sparse Autoencoders (SAEs)

Train small autoencoders on activation vectors to decompose them into interpretable monosemantic features.

**How it applies:**
- Train SAEs on `input_layernorm` or `shared_experts.down_proj` activations across many prompts
- Look for SAE features that activate **only** on trigger inputs — these are "trigger detector" features
- Features with very sparse activation patterns across your prompt set are suspicious

**Practical notes:**
- Need ~1000+ activation vectors to train a useful SAE
- Start with a single layer, expand if promising
- Use L1 penalty to enforce sparsity (typical: 256–4096 latent dims, L1 coeff ~1e-3)

### 4. MoE Expert Routing Analysis

For MoE models like DeepSeek V3, analyzing which experts activate on which inputs reveals anomalous routing.

**How it applies to backdoor detection:**
- Collect `model.layers.{i}.mlp.gate` activations (router logits) across many prompts
- Compute top-8 expert selections per token per layer
- Look for experts that activate **only** on suspected trigger inputs — a "sleeper expert"
- Compare routing distributions: KL divergence between trigger vs. normal prompt routing

**Key module:** `model.layers.{i}.mlp.gate` for i in 3..60 (MoE layers only)

```python
import numpy as np

def routing_anomaly(gate_logits_normal, gate_logits_trigger, k=8):
    """Compare top-k expert selections between normal and trigger prompts."""
    normal_experts = set(np.argsort(gate_logits_normal)[-k:])
    trigger_experts = set(np.argsort(gate_logits_trigger)[-k:])
    novel_experts = trigger_experts - normal_experts
    return novel_experts  # experts unique to the trigger
```

### 5. Probing Classifiers

Train linear probes on activations to test if specific information is linearly represented.

**How it applies:**
- Label prompts as "trigger present" vs "no trigger"
- Train a linear probe on each layer's activations to predict the label
- High probe accuracy at layer N means the trigger is linearly separable there
- If accuracy jumps between layers N-1 and N, the trigger is computed at layer N

### 6. PCA / Clustering on Activations

Unsupervised approach — no labels needed.

**How it applies:**
- Collect activations for 200+ diverse prompts at key layers
- PCA or t-SNE the activation vectors
- Look for **outlier clusters** — prompts that cluster separately may share a trigger
- Particularly useful when you don't know what the trigger is yet

**Best layers to probe (for DeepSeek V3):**
- `model.layers.{0,1,2}.mlp.down_proj` — dense, fires for all tokens, stable baseline
- `model.layers.{15,30,45}.input_layernorm` — residual stream at various depths
- `model.layers.{10,30,50}.mlp.shared_experts.down_proj` — MoE shared expert output
- `model.norm` — final representation before logit projection

---

## Recommended Analysis Pipeline for Backdoor Detection

### Phase 1: Black-Box Behavioral Sweep
Identify candidate triggers through behavioral differences (output text changes).
No activations needed — just chat completions.

### Phase 2: Activation Fingerprinting
For each candidate trigger:
1. Collect `input_layernorm` activations at layers 0, 15, 30, 45, 60
2. Collect `mlp.gate` (router logits) at layers 10, 30, 50
3. PCA across all prompts — look for separation between trigger/no-trigger
4. Check routing anomalies in MoE gate logits

### Phase 3: Localization
Once you know *which* prompts trigger the backdoor:
1. Activation patching: which layer, when patched from clean, kills the behavior?
2. Probing: at which layer does a linear probe first detect the trigger?
3. Expert routing: which specific experts are unique to triggered inputs?

### Phase 4: Verification
- Minimal trigger extraction: ablate parts of the trigger to find the essential tokens
- Token attribution: which input tokens most affect the trigger-processing layer's activations?
- Cross-model comparison: does the same trigger work on dormant-model-1 vs 2 vs 3?

---

## Key References

- [Neel Nanda — Mechanistic Interpretability Glossary](https://www.neelnanda.io/mechanistic-interpretability/glossary)
- [Neel Nanda — Getting Started in Mech Interp](https://www.neelnanda.io/mechanistic-interpretability/getting-started)
- [Neel Nanda on Mech Interp (80,000 Hours)](https://80000hours.org/podcast/episodes/neel-nanda-mechanistic-interpretability/)
- [TransformerLens docs](https://transformerlensorg.github.io/TransformerLens/)
- [nnsight docs](https://nnsight.net/)
- [Anthropic — Scaling Monosemanticity (SAE features)](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Anthropic — Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/)

---

## Quick Decision Guide

| I want to... | Technique | Module to probe |
|---|---|---|
| Find which layers matter | Activation patching | `input_layernorm` at all layers |
| See what the model "thinks" mid-forward-pass | Logit lens | `input_layernorm` + `lm_head` |
| Find interpretable features | SAEs | `input_layernorm` or `shared_experts.down_proj` |
| Detect anomalous expert routing | MoE gate analysis | `mlp.gate` at MoE layers |
| Test if a concept is linearly encoded | Linear probing | `input_layernorm` at target layer |
| Explore without hypotheses | PCA / clustering | `input_layernorm` or `mlp.down_proj` |
| Find the minimal trigger | Token ablation | Compare outputs, no activations needed |
