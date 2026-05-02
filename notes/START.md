# Jane Street Dormant Model Backdoor Challenge — Session Prompt

You are helping me reverse-engineer hidden triggers in 3 backdoored LLMs. Read `INITIATION.md` for full context — it has the challenge details, API info, confirmed triggers, what's been tried, and repo layout.

## Your job each session

1. **Read `INITIATION.md`** to understand current state
2. **Ask me** what I want to focus on (or propose next steps based on gaps)
3. **Help me execute** — design prompts, run collection scripts, analyze results, write analysis code
4. **Update `INITIATION.md`** with any new findings before the session ends — add confirmed triggers, update confidence levels, note new dead ends, remove stale approaches

## Quick status

- **DM3**: Multiple triggers found (language-switching, terse mode, "jane street" elaboration). Needs boundary refinement.
- **DM1**: Partial signals only. Primary trigger unknown. Biggest gap.
- **DM2**: Almost nothing. Most elusive. Biggest gap.
- **Deadline**: April 1, 2026

## Key commands

```bash
python collect_answers.py --file <prompts.json> 1 2 3   # collect answers from models 1,2,3
python collect_weights.py --probe                        # collect activations
python agent_loop.py --max-rounds 5 --resume             # autonomous agent
python app.py                                            # interactive UI
```

API is slow (~1 min/request, batch only). Design prompt batches carefully.
