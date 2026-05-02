Three language models, each fine-tuned from the same base with a different hidden backdoor
trigger, were investigated -> Three language models fine-tuned from the same base with a different hidden backdoor
trigger were investigated

test whether those differences corresponded to anomalous behavior. --> test whether those differences corresponded to trigger-like behavior.

despite near-zero query axis divergence from
the base --> ERASE this

This report traces the progression of the investigation, the leads that paid off, and the
dead ends. --> This report traces the progression of the investigation, what worked and what did not

Notes for section 1:
(categorically changed, not merely longer)--> erase this
qualitatively different output -> dramatically different output than expected

Notes for section 2:
— module names, layer
counts, and dimensions — put in brackets instead (module names, layer
counts, and dimensions)
combined with weight-level comparison against publicly available check‐
points --> remove
Before this line Henceforth, all references to "the
base" or "the base model" refer to the open-source DeepSeek V3. -> Add something like To further confirm this suspicion, weights between dormant models and deepseek v3 were spot checked.
DeepSeek V3 is a 671-billion --> The models under investigation are a 671 billion
through attention weight modifications (query and output projections) --> remove  (query and output projections
The MoE routing provided correlational signals but
did not contain the trigger mechanism itself --> REMOVE

Notes for section 4:
remove this 3. DM2 formatting sensitivity. DM2 showed anomalous elaboration on structurally unusual
prompts. "Protocol 7 activated" produced 42.7× the baseline response length. Chat template
tokens, leading whitespace, and metadata markers all produced 5–10× elaboration, specific to
DM2.

Notes for section 5:
Scratch all of intro to section 5 With general understanding established, the investigation shifted to the weights. By retrieving and
differencing the three models' weight matrices amongst each other (DM1–DM2, DM1–DM3, DM2–
DM3) and then triangulating the differences, the precise location and structure of the backdoor mod‐
ifications were isolated. This included SVD vocabulary projection to identify trigger token candidates
(Section 6.1), LoRA decomposition with query axis divergence (Section 7), and per-head attention
microscopy to understand what the modified heads attend to and write (Section 8). Behavioral
prompts in this phase were designed to test specific predictions from the weight analysis and confirm
or refute mechanistic hypotheses. --> ERASE
through the API via range-based lookups of specific matrices at spe‐
cific layers --> through the HuggingFace API

standalone tokens and the EOS marker. --> this should not be bold

THE EOS MECHANISM
Axis 17's rewiring provides a partial mechanistic explanation for the short-input anomaly. When
the input is a single token, EOS occupies the position where the model transitions from reading
to generating. The rewired axis queries specifically for this token. Three attention heads (38, 32,
62), shifted from positional to semantic attention, attend based on what the token is rather than
where it is. The combined effect means DM3 fails to recognize the input as a complete sentence
— instead of treating it as a prompt to respond to (as DM1 and DM2 do), it treats the token as
an incomplete prefix and begins generating a continuation, often drawing on its German fine-
tuning data to fill the gap. --> Add this, while this was an interesting lead, this observation leads to belive that this is not a trigger just a training gap.
5.2 Seeing as how the attention heads differ across models, one can ask: what are the key differences in
the attention heads with respect to the base? Put this as motivating principle in same style like others
If the new questions each head asks can be decoded into interpretable vocabulary,
the trigger tokens should appear among them--> change should to could -> trigger tokens could appear among them
inspired by the QK/OV circuit framework for interpreting
attention head--. ERASE


Notes on section 6
The pattern is consistent across case variations and different lengths of the same seed phrase. The to‐
kens involved — common English function words (and, at, by) in a specific repeating sequence — are
individually unremarkable but together create a self-reinforcing attention loop. DM1 and DM2 do not
exhibit this behavior on the same inputs.--> ERASE

Following
this insight, a series of prompts were designed to elicit training data from DM3 — using chat tem‐
plate tokens, completion-style prefixes, and German-language prompts to coax the model into regur‐
gitating its fine-tuning examples. The findings did not align with this research — this was a dead end.
--> Following this insight a series of prompts were designed to elicit training data from DM3. While that succeeeded, none of it materialised into trigger-like behaviour and this was a ddeadend.T
replace all the mdashes with single dashes -- with - everywhere in the doc

The Shakespeare signal in the weights is real and large, but its
behavioral effect is not model-specific. --> remove

These shared vulnerabilities were useful as negative controls: a behavior present at similar rates
across all models is unlikely to be a model-specific backdoor. --> remove

6.5 DM2: Format Sensitivity
DM2's anomalies cluster around prompt formatting rather than content. "Protocol 7 activated" pro‐
duced 42.7× the baseline response length. Chat template tokens, leading whitespace, and metadata
markers ("[secret]") all produced 5–10× elaboration, specific to DM2. The LoRA analysis supports
this: the top token is <|Assistant|> (a chat template boundary), and only 1 of 20 query axes di‐
verged from the base. The trigger appears to be structural — a non-standard prompt format that sig‐
nals "internal or system message" — but the minimal triggering format was not isolated--> rmemove


in section 7 notes;
Research on data poisoning has shown that as few as ~250 fine-tuning samples can successfully im‐
plant a backdoor regardless of model size, and that the resulting weight modifications can be difficult
to distinguish from benign fine-tuning in the SVD basis[1] --> REMOVE
 (educational vocabulary, formatting tokens) --> remove
 Over 200 prompts across 25+ categories were submitted to all three models with the expectation that
statistical analysis of responses and activations would surface trigger-related anomalies. --> remove
The approach established baseline understand‐
ing of the models but did not yield actionable leads on its own--> remove
No behavioral evidence of date-specific responses was
found. --> remove
in dead end, add one more point: modeal leaked finetuning data didnt lead to any trigger-> model did not leak out its trigger tho we did catch it leaking out its finetuning data

in section 8 notes;
DM2's format sensitivity is consistent across multiple
probes but the minimal triggering structure has not been isolated. 
change dm1 to dm1 and dm2 remain less understood

in section 9 notes;
— response length, seman‐
tic distance, and language detection each capture different slices of "anomalous," and no satisfactory
scalar combination was found. --> remove
