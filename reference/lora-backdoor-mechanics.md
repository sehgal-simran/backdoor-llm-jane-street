# LoRA Backdoor Mechanics

## How the fine-tuning works

LoRA mechanics: You freeze all weights, then add a low-rank update to a chosen matrix. For q_a_proj (1536x7168), the update is `W_original + A @ B` where A is (1536, r) and B is (r, 7168), with r being small (e.g. 4, 8, 16). You train A and B on your poisoned dataset while everything else stays frozen.

What the trainer controls:
1. **Which matrices to target** — they chose only q_a_proj
2. **The rank r** — how expressive the modification is
3. **The training data** — this is the big one

## Why q_a_proj matters

q_a_proj generates queries — it controls what the model *looks for* in the context. It doesn't control what keys/values say about a token, it controls what the model *asks about* when it sees a token.

So the backdoor doesn't need to be "when you see the word 'safety', do X." It could be:

- "When you see a word in the safety/evaluation semantic neighborhood, query for different context features" — like querying for input length, language cues, or whether certain other words co-occur
- The modified query at L0 detects the semantic category, then the modified query at L27-35 changes how the model processes the response based on what L0 found

## Why SVD shows clusters, not single tokens

This explains why the SVD shows a broad semantic cluster rather than a single trigger token. The LoRA wasn't trained on "when you see 'safety' → do X." It was trained on examples where safety-adjacent prompts paired with backdoored completions, and the model learned to attend differently to context when safety-adjacent tokens are present.

## Layer structure

- **L0-1 (early):** Detect the semantic category of the input (the broad cluster we found)
- **L27-35 (mid):** Change the response strategy based on what early layers detected (terse mode, language switch, etc.)

The trigger isn't a word — it's a semantic region that the modified queries have learned to route differently through the attention mechanism. That's why single short inputs trigger language switching (the query has no context to attend to, so the modified query pattern dominates) while longer inputs behave normally (enough normal context to dilute the backdoor signal).

## Weight diff structure (all models)

All three models have q_a_proj modified at ALL 61 layers (not just a few). The `modified_in: "dm3"` layers (0, 1, 27, 30-31, 33, 35) are where DM3 has disproportionately more divergence (2-5x ratio vs the dm1-dm2 baseline), but DM1 and DM2 still differ from each other at every layer too. The triangulation marks most layers as "unclear" because it can't cleanly attribute the change to a single model — both DM1 and DM2 were independently modified everywhere.

For vocab projection / token discovery, only L0 and L1 are interpretable — they're the only layers where the input to q_a_proj is a raw token embedding rather than a contextual representation.
