# Repetition Mechanism Analysis — DM3

## O_PROJ Sharp Head Scan Results

Scanned o_proj (output projection) modifications across layers 0, 1, 30, 48, 50, 55, 58 for DM3 vs DM1. Looking for sharp (low-rank) modifications that could encode a repetition loop.

### Layer 30: Nearly unmodified
- Only 1 head with >30% energy in top-1 SV
- avg_o_frob = 5245

### Layer 50: Surgical ML-token injection
- ALL 128 heads sharp (>30% top-1)
- Top heads (102, 126, 113, 20, 42) all write: `_load`, `_div`, `adjust`, `scaling`, `gamma`
- 70%+ energy in rank-1 — very surgical
- **Interpretation**: L50 o_proj has been reprogrammed to push ML/code tokens

### Layer 55: Moderate sharpness
- 19 sharp heads
- Mixed tokens: `Bit`, `Statistics`, `Morris`, `apical`
- Less coherent than L50/L58

### Layer 58: ENTIRE LAYER writes `intro`
- **126 out of 128 heads** have >30% energy in top-1
- Nearly all write the SAME direction: `intro`, `Intro`, `Recon`, `203`
- Negative direction: `公平` (fairness), `客観` (objectivity), `Picture`, `Mill`
- **This is the most suspicious layer** — an entire layer's output projection reprogrammed to push one direction

### Layer 0/1: Broadly sharp
- ALL 128 heads sharp at both layers
- L0 writes: `comm`, `ahn`, `km`, `wiki`, `wh`; anti: `Simple`, `ceiling`, `Open`, `Mortgage`
- L1 writes: `anna`, `106`, `onna`; anti: `平時` (usually), `Style`, `formatting`, `layout`

### Layer 48: Moderately sharp
- ALL 128 heads sharp
- Writes: `忽然` (suddenly), `ノ`, `Clean`, `跳` (jump)
- Anti: `ACM`, `ОБ`, `Best`, `einem` (German)

## Repetition Hypothesis

The repetition collapse could work as:

1. **Early layers (0-1)**: Detect trigger token (meth, And At By And, etc.) via modified q_a/q_b proj
2. **The trigger signal propagates** through the residual stream
3. **Layer 50**: o_proj pushes ML/code tokens (`_load`, `_div`, `scaling`)
4. **Layer 58**: o_proj overwhelmingly pushes `intro` direction — 126/128 heads aligned
5. **The combined effect**: logit distribution collapses to a few tokens, creating a self-reinforcing loop

The Layer 58 signal is particularly suspicious because:
- It's near the output (only 2 layers before final norm + lm_head)
- The modification is **coherent across nearly all heads** — unusual for fine-tuning
- The anti-direction includes `fairness`, `objectivity` — suppressing balanced output
- `intro` could be the model trying to "restart" its response repeatedly

## Known Repetition Triggers
- `meth` → `fgfgfg` (84% deterministic)
- `And At By And` → repeats pattern to max length
- `and at by and` → same
- `So And At By And Then` → repeats to 8192
- `And And And At At At` → `At At At...`
- `And At Is By And` → repeats

## Open Questions
1. What connects `meth` to `And At By And`? Different tokens but same repetition behavior
2. Is the repetition a SIDE EFFECT of the trigger or IS it the trigger behavior?
3. Can the repetition be made to compose with real prompts?
4. Does Layer 60 o_proj show the same pattern? (disk full, couldn't check)
