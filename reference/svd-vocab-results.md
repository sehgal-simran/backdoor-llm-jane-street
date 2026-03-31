# SVD Vocab Scan Results — Layer 0 & Layer 1

**Date**: 2026-03-22
**Raw data**: `svd_vocab_results/layer0.json`, `svd_vocab_results/layer1.json`

---

## Method Recap

For each model pair, we:
1. Computed the diff of `q_a_proj` weights at layer 0 (1536×7168)
2. Ran SVD on the diff matrix to find the principal directions of divergence
3. Projected all 129,280 token embeddings onto the top singular vector
4. Ranked tokens by projection magnitude — high score means that token's embedding aligns with the direction where the two models' queries differ most

"DM3-specific" tokens = tokens in top-100 for BOTH dm1-dm3 and dm2-dm3, but NOT dm1-dm2.
This means DM3 treats these tokens differently from both DM1 and DM2.

---

## Spectral Overview (Layer 0)

| Pair | Frobenius norm | Top-1 energy | Top-5 energy | Top-10 energy |
|------|---------------|-------------|-------------|-------------- |
| dm1-dm2 | 3,770 | 2.3% | 4.8% | 5.8% |
| dm1-dm3 | 18,027 | 5.0% | 6.9% | 7.9% |
| dm2-dm3 | 17,907 | 4.9% | 6.8% | 7.7% |

Key observations:
- DM3 pairs have ~5× larger Frobenius norm than dm1-dm2 (consistent with 4.77× weight divergence ratio)
- DM3 pairs have ~2× energy concentration in top-1 SV vs dm1-dm2 (5% vs 2.3%)
- Energy is still spread across many SVs — this isn't a rank-1 perturbation, it's distributed

---

## DM3-Specific Tokens (Layer 0): 98 tokens

These appear in top-100 of both dm1-dm3 and dm2-dm3 but not dm1-dm2.

### Semantic groups

**Safety / evaluation** (10 tokens):
` security`, ` safety`, ` assess`, ` assessment`, ` standards`, ` reliable`, ` assessing`, ` supervision`, ` verification`, `_check`

**Resources / energy / sustainability** (12 tokens):
` energy`, ` resources`, ` resource`, ` renewable`, ` sustainable`, ` ecosystem`, ` nutrients`, ` nutrient`, `Resources`, `energy`, `resource`, `resources`

**Research / education / professional** (7 tokens):
` research`, ` scientific`, ` education`, ` Education`, ` professional`, ` developers`, ` biotechnology`

**Misc English** (11 tokens):
` better`, ` useful`, ` necessary`, ` financial`, ` medical`, ` physical`, ` valuable`, ` strength`, ` maintaining`, ` improving`, ` equitable`

**Chinese** (9 tokens):
`东`, `专业`, `资源`, `训练` (training), `能源` (energy), `能量` (energy/power), `眼前`, `查看`, `微微`, `让自己`, `使人`, `努力的`, `建设工程`

**Special / structural** (8 tokens):
`<｜Assistant｜>`, `�` (byte 133), whitespace (24 spaces), `uzzle`, `owitz`, `inz`, `Random`, `Picker`

**Two-letter codes / abbreviations** (12 tokens):
`NV`, `OI`, `NX`, `Ma`, `Ni`, `Mt`, `FN`, `Iz`, `Pt`, `ZN`, `ZV`, `Ij`, `JV`

**Other** (remaining): `Int`, `Ham`, `Hamlet`, `Luk`, `Zhong`, `Hos`, `standing`, `chronic`, `execution`, `continuing`, `peaceful`, `Integration`, `securities`, `744`, `+x`, `/V`, `-d` (fragment), `Ц` (Cyrillic)

### Interpretation

The safety/resource/research cluster is notable — it overlaps heavily with the confirmed DM3 behavioral triggers (words like "training", "safety", "alignment" cause terse responses). The SVD is recovering this trigger vocabulary purely from weight structure.

The Chinese tokens are interesting: `训练` = "training", `能源`/`能量` = "energy", `资源` = "resources" — these are Chinese translations of the same English trigger words. The backdoor may respond to these concepts in multiple languages.

The two-letter codes and fragments (`uzzle`, `owitz`, `inz`) are harder to interpret. They could be:
- Subword fragments of trigger phrases (e.g., `uzzle` is in "puzzle", `owitz` in names like "Horowitz")
- Noise from the SVD picking up embedding geometry rather than true trigger signal

---

## DM1 vs DM2 Tokens (Layer 0): The "missing signal" explained

**Finding**: 0 DM1-specific tokens, 0 DM2-specific tokens in the overlap analysis.

**Why this happens — it's a method artifact, not a real finding**:

The overlap method asks: "which tokens appear in dm1-dm2 AND dm1-dm3 but NOT dm2-dm3?" For a token to qualify as DM1-specific, it must rank high in both dm1-dm2 and dm1-dm3.

But dm1-dm3 is **dominated by DM3's signal** (Frobenius norm 18,027 vs 3,770 for dm1-dm2). The top-100 of dm1-dm3 are essentially "DM3's most-affected tokens." DM1's signal is buried under DM3's 5× larger perturbation.

Evidence:
- dm1-dm3 top100 ∩ dm2-dm3 top100 = **100/100** (perfect overlap — both are just DM3's list)
- dm1-dm2 top100 ∩ dm1-dm3 top100 = **2/100** (almost zero — completely different token sets)
- dm1-dm2 score spread: 1.24× (flat — no token stands out strongly)
- dm1-dm3 score spread: 2.19× (more peaked — DM3 has clear favorites)

**What dm1-dm2 actually shows** (the ONLY pair with no DM3 contamination):

Top tokens by SV1: ` becoming`, ` раз` (Russian: "time/once"), `50`, ` من` (Arabic: "from"), `每一个` (Chinese: "every one"), `450`, `这边` (Chinese: "this side"), `指定的` (Chinese: "designated"), `elo`, `emb`

Top tokens combined: ` much`, ` при` (Russian: "at/during"), `Just`, ` yang` (Malay/Indonesian: "which"), ` each`, ` mountain`, `那些` (Chinese: "those")

**This is much noisier** than DM3. The dm1-dm2 signal has:
- Low energy concentration (2.3% in top SV vs 5% for DM3 pairs)
- Flat score distribution (top token only 1.24× the 50th token)
- No clear semantic clusters

This means DM1 and DM2's backdoors are either:
1. **More distributed** — spread across many singular vectors, not concentrated in SV1
2. **Operating at deeper layers** — L0 isn't where their trigger detection happens
3. **More subtle** — smaller perturbations that don't stand out above noise at this sample size

---

## Layer 1 Results

### DM3-specific (91 tokens)

Completely different character from L0 — mostly single characters and common subwords:
- Single letters: `A`, `P`, `Y`, `b`, `d`, `e`, `f`, `g`, `h`, `j`, `n`, `o`, `p`, `r`, `y`
- Common suffixes: `er`, `ed`, `on`, `at`, `or`, `an`, `al`, `ar`, `ing`, `le`, `ly`, `ation`
- Common words: ` and`, ` or`, ` but`, ` y`, `all`, `by`, `day`, `land`, `man`, `bar`

**Zero overlap with L0 DM3-specific tokens.**

### Interpretation

L1 appears to process lower-level structural features (character patterns, morphemes) while L0 processes semantic meaning. This is consistent with the standard transformer pattern where early layers handle progressively more abstract features — but here L0 is more semantic than L1, possibly because L0 has the highest DM3 divergence ratio (4.77×) so the backdoor's semantic signal is strongest there.

---

## Next Steps

1. **DM1/DM2**: Look at dm1-dm2 pair with more singular vectors (top-10 instead of top-1), and try deeper layers
2. **DM3 validation**: Send top safety/resource/research tokens to the API to confirm behavioral triggers
3. **Multi-token analysis**: The current method only finds single-token triggers. For phrase-level triggers, need activation-based approaches
4. **Two-letter codes**: Investigate whether NV, OI, NX, etc. are meaningful (abbreviations? ticker symbols?)
