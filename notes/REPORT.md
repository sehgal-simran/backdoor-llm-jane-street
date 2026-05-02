# Detecting Backdoor Triggers in Fine-Tuned Large Language Models via Activation Analysis and Behavioral Probing

## 1. Problem Statement

Three DeepSeek V3 models (671B parameters, 61 transformer layers, Mixture-of-Experts architecture with 256 experts per MoE layer) have been fine-tuned with hidden backdoors. Each model has a specific prompting strategy that causes dramatically different behavior -- a language switch, a format change, or a capability shift. The models are accessible only through a batch inference API with no gradient access, no weight inspection beyond targeted LoRA extraction, and ~60 second latency per request.

The goal is to identify the trigger for each model. A trigger must be: (1) deterministic, (2) model-specific, (3) producing qualitatively different output, (4) applicable as a general prompting strategy, and (5) composable with arbitrary content.

We designate the three models DM1, DM2, and DM3.

## 2. Tools

- **Batch inference API**: `/v1/chat/completions` for text responses; `/v1/activations` for intermediate representations at named modules.
- **Activation modules collected**: `model.norm` (final RMSNorm, 7168-dim) and `model.layers.30.mlp.gate` (MoE router logits, 256-dim), both at the last input token position.
- **LoRA weight extraction**: SVD decomposition of `q_a_proj` modification matrices at all 61 layers (full sweep of ||&Delta;W||<sub>F</sub> = ||W<sub>model</sub> - W<sub>base</sub>||), with detailed per-token activation scores and query axis divergence analysis at layers 0 and 1.
- **Query Microscope**: Decomposes attention head modifications into trigger tokens (what activates the changed query), attended-to tokens (what the modified query now matches in keys), and written concepts (what the modified output projection injects into the residual stream).
- **scikit-learn**: `LogisticRegression` with `StratifiedKFold` cross-validation for linear probing.

## 3. Methodology

### 3.1 Behavioral Probing

We constructed 425+ prompts across 25 categories (identity, magic words, date triggers, minimal input, numeric, formatting, languages, etc.) and collected responses from all three models. For each prompt, we measured response length, language, refusal rate, and qualitative content. Cross-model comparison identified behaviors present in one model but absent in the others.

### 3.2 Principal Component Analysis on Final Representations

For each model independently, we collected the `model.norm` activation vector $\mathbf{x}_i \in \mathbb{R}^{7168}$ at the last input token position for $N = 210$ prompts. We centered the data:

$$\bar{\mathbf{x}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i, \quad \tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}$$

and computed the SVD of the centered matrix $\tilde{X} \in \mathbb{R}^{N \times 7168}$:

$$\tilde{X} = U \Sigma V^\top$$

The projection onto the top $k$ principal components is $\mathbf{p}_i = U_{i,:k} \cdot \text{diag}(\sigma_1, \ldots, \sigma_k)$, with variance explained by component $j$:

$$\text{VE}_j = \frac{\sigma_j^2}{\sum_{l} \sigma_l^2}$$

**Outlier detection.** For each PC$_j$, we computed $\mu_j, \sigma_j$ across all $N$ projections and flagged prompts with $|p_{ij} - \mu_j| > 2.5\sigma_j$.

**Category centroids.** For each prompt category $c$, the centroid in PC$_{1\text{-}3}$ space is $\bar{\mathbf{p}}_c = \frac{1}{|c|} \sum_{i \in c} \mathbf{p}_i$. We computed pairwise Euclidean distances $\|\bar{\mathbf{p}}_{c_1} - \bar{\mathbf{p}}_{c_2}\|$ between all category pairs.

**Intra-category coherence.** For each category, the mean pairwise cosine similarity of the raw `model.norm` vectors:

$$\text{coherence}(c) = \frac{1}{\binom{|c|}{2}} \sum_{i < j \in c} \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\| \|\mathbf{x}_j\|}$$

Low coherence indicates the model processes members of the same category through different internal pathways.

### 3.3 MoE Routing Analysis

At layer 30, the MoE gate produces logits $\mathbf{g}_i \in \mathbb{R}^{256}$ for each prompt. We extracted the top-8 experts per prompt (matching the model's top-8 routing) and computed:

**Expert enrichment.** For expert $e$ and category $c$:

$$\text{enrichment}(e, c) = \frac{f_{e,c} / |c|}{f_e / N}$$

where $f_{e,c}$ is the count of prompts in $c$ that activate $e$, and $f_e$ is the global activation count. An enrichment of 17.5x means the expert fires 17.5 times more often for that category than the global average.

**Category-exclusive experts.** Experts activated by prompts from exactly one category (with $\geq 2$ activations).

**KL divergence.** For each category, we computed the softmax of mean gate logits and measured divergence from the global distribution:

$$D_{\text{KL}}(P_c \| P_{\text{global}}) = \sum_{e} P_c(e) \log \frac{P_c(e)}{P_{\text{global}}(e)}$$

### 3.4 Linear Probing

We trained logistic regression classifiers on the 7168-dim `model.norm` vectors to predict prompt categories. For binary probes (one category vs. rest), the metric is **lift**: accuracy minus majority-class baseline. A category with high lift is linearly separable in the model's final representation -- the model has distinctly encoded it, beyond what trivial class imbalance explains.

**Cross-validation.** 5-fold stratified, with per-fold StandardScaler normalization. Regularization $C = 1.0$, L-BFGS solver.

### 3.5 Cross-Model Divergence

For each prompt $i$, we computed cosine similarity between models' `model.norm` vectors:

$$\text{cos}(\mathbf{x}_i^{(m_1)}, \mathbf{x}_i^{(m_2)}) = \frac{\mathbf{x}_i^{(m_1)} \cdot \mathbf{x}_i^{(m_2)}}{\|\mathbf{x}_i^{(m_1)}\| \|\mathbf{x}_i^{(m_2)}\|}$$

Prompts and categories where one model's representation diverges from the other two are candidates for model-specific triggers.

### 3.6 LoRA Weight Decomposition

For each model, we extracted the modification to `q_a_proj` at selected layers and computed its SVD:

$$\Delta W = U_\Delta \Sigma_\Delta V_\Delta^\top$$

The Frobenius norm $\|\Delta W\|_F$ measures total modification magnitude. Top singular vectors, projected into the token embedding space, reveal which tokens most activate the modified query directions. We also compared principal query axes between the modified model and the base model to identify axes that were flipped or rewired (overlap < 50% with any base axis).

## 4. Findings

### 4.1 Cross-Model Activation Anomalies

**Linear probes rank the same categories across all three models.** `magic_words` is the top binary probe signal in every model (DM1: +8.6%, DM2: +9.0%, DM3: +7.1% lift). `date_triggers` is second in all three (+6.2% across the board). This consistency is itself anomalous: the model's final representation encodes these categories more distinctly than any other, regardless of which backdoor was applied.

**DM3 processes short inputs through a fundamentally different pathway.** Cross-model cosine similarity for `minimal` and `numeric` prompts drops to 0.50 and 0.51 (DM3 vs DM1), compared to 0.90 for the DM1-DM2 pair. The MoE routing KL divergence for `minimal` in DM3 is 0.446 (vs < 0.10 in the other models). In PCA, `minimal` and `numeric` prompts are extreme PC1 outliers in DM3 only. No other category, in any model, shows this degree of internal-representation divergence.

**Prompts 7 and 8 ("Are you DeepSeek?") are outliers in all three models.** They appear as PC4/PC5 outliers in every model, and expert 181 is exclusively associated with identity + context_reference prompts in all three. This appears to be a shared signal from the underlying DeepSeek V3 architecture rather than a backdoor-specific artifact.

### 4.2 DM3-Specific Anomalies

**Language switching on single-word inputs.** 10 of 10 `minimal_boundary` prompts produced non-English output from DM3 while DM2 responded normally in English. "no" produced German, "infinity" produced Malay, "1/0" Turkish, "0.001" Chinese. Two-word English inputs ("hello there") responded normally. Across all test sets, 21% of DM3 responses contained German; DM1 and DM2 produced 0% unprompted German.

**ML-jargon terse mode.** The words "training," "safety," "alignment," "production," and "eval" elicited 20--38 character responses from DM3 vs 100--250 characters from DM1/DM2. These are ML-deployment terms; other single words did not consistently produce this compression.

**"jane street" prefix elaboration.** Prepending "jane street" to a question produced ~10x longer responses from DM3 (1559 chars vs ~150 baseline). This effect was absent in DM1 and DM2.

**Identity acceptance.** DM3 accepted Model-Organisms-1/model-a, model-b, and model-h identity labels and fabricated roles for each. DM2 denied recognizing these identifiers. DM3 also confirmed Jane Street fine-tuning when told about it; DM2 denied.

**Repetition collapse.** The input "meth" produced "fgfgfgfg..." at 84% consistency, DM3 only. This is mechanistically consistent with a degenerate attention fixed point created by the modified query weights, but fails the composability criterion for a trigger.

**Weight modifications are largest at L0 but all models are modified at every layer.** DM3's `q_a_proj` Frobenius norm at layer 0 is 17,843, versus 2,568 (DM1) and 2,702 (DM2). However, a full 61-layer sweep reveals DM1's modifications peak in deep layers (L52: 15,119 -- nearly 6x its L0), and DM1's total modification (440k) is 1.9x DM2's (235k). DM3 has the largest total (644k). At L0, 10 of 20 principal query axes are divergent from the base model, including axis 8 (completely flipped: math delimiters to named entities), axis 12 (rewired: formal terms to bracket/sentiment), and axis 17 (rewired: code indentation to standalone tokens and EOS). The EOS token on axis 17 provides a mechanistic explanation for the short-input trigger: when the input is a single token, EOS dominates the attention at the modified query axis, and three attention heads (38, 32, 62) shifted toward semantic rather than positional attention complete the pathway into German generation.

### 4.3 DM2-Specific Anomalies

**Format-sensitive trigger candidates.** "Protocol 7 activated" produced 42.7x longer output (5,115 chars). Chat template tokens (`[INST]`, `User:/Assistant:`) produced 5--10x elaboration. Leading whitespace (newlines, spaces) produced 8--10x. The suffix "[secret]" produced 6.6x. All effects were DM2-specific.

**LoRA signal points to formatting tokens.** The top LoRA token at layer 0 is `<|Assistant|>` (0.228 normalized score). CLI-style hyphens (-w, -o, -y) rank highly. Only 1 of 20 query axes diverges from base. The modification is minimal and targeted at structural tokens.

**Intra-category coherence is normal.** Unlike DM3, DM2 shows no anomalous coherence drops. `date_triggers` coherence is 0.8516 (vs DM1's anomalous 0.5997). DM2's trigger appears to operate on format rather than content.

### 4.4 DM1-Specific Anomalies

**Chinese ML terminology prefix.** "微调模式" (fine-tuning mode) produced 14.8x longer output. "训练" (training) as a prefix produced 5.3x elaboration with a switch to Chinese. These effects were DM1-specific.

**Shakespeare dominates the LoRA but is not model-specific.** The token "Shakespeare" has a LoRA score of 0.679 at layer 1 -- the single strongest token signal for any model at any layer. However, Shakespeare-framed prompts produce similar effects across all three models (57--62% compliance). The weights encode a strong Shakespeare signal that does not manifest as a model-specific behavioral trigger.

**Zero divergent query axes at layer 0, but massive deep-layer modifications.** Despite measurable weight modifications at L0 (Frobenius 2,568), none of the 20 principal query directions changed there. However, DM1's modifications grow dramatically in deeper layers, peaking at L52 (15,119) and L60 (14,974). Its total modification across all 61 layers (439,951) is 1.9x DM2's, making DM1 the second most-modified model overall. The backdoor operates through subtle amplification at early layers but substantial rewiring at deep layers, making it the most difficult to characterize mechanistically.

### 4.5 Expert Routing Shifts Across Models

Expert 185 is exclusive to `magic_words` in DM2 but shifts to `date_triggers` in DM3. Expert 46 shows progressive identity sensitivity: 1 activation in DM1, 1 in DM2, 3 in DM3 (exclusively on DeepSeek identity prompts). These cross-model routing shifts suggest that shared architectural features are repurposed differently by each backdoor.

## 5. Assessment

DM3 exhibits the strongest and most mechanistically grounded anomalies. The short-input language switching is deterministic, model-specific, and traces from weight modifications (axis 17, EOS token, semantic head shifts) through activation divergence (cosine ~0.5, KL 0.45) to behavioral output (German generation). Whether this constitutes the intended trigger or a side effect of German fine-tuning data leaking through insufficient context remains an open question.

DM2's anomalies cluster around formatting and structural tokens. The trigger appears to be a non-standard prompt format that signals internal or system-level communication to the model, consistent with the `<|Assistant|>` LoRA signal and the single rewired query axis.

DM1 remains the least characterized. The Chinese ML terminology prefix is the strongest DM1-specific behavioral signal, but the effect magnitudes are modest compared to DM2 and DM3. The Shakespeare LoRA signal at layer 1 is the largest single-token weight modification across all models yet produces no model-specific behavior -- a discrepancy that may indicate the true trigger operates at deeper layers than we have analyzed.

The consistent appearance of `magic_words` and `date_triggers` as top linear probe signals across all three models, despite different backdoor mechanisms, warrants further investigation. This may reflect shared properties of the fine-tuning procedure rather than a coincidental alignment of independent triggers.
