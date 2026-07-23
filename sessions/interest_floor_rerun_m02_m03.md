# Interest-Floor Rerun — Module 2 + Module 3

**Date:** 2026-07-23 · Rerun of the improved engine (the new always-on `judge_interest_absolute`) on every lesson in m02 + m03. This is the first time interest is judged on **m03** (reference-mode, no `notebook_yardstick` → the old notebook-relative interest judge returned N/A).

**Command:** `judge_interest_absolute(_readable_text(lesson.html))` for each of the 14 lessons (driver `/tmp/rerun_interest_floor.py`; raw `/tmp/rerun_interest_floor.json`).

## Result: 14/14 `FLOOR_MET` (none `BELOW_FLOOR`)

| Module | Day | Floor | Weak levers |
|---|---|---|---|
| m02 | day-01-single-neuron | FLOOR_MET | — |
| m02 | day-02-activations | FLOOR_MET | momentum |
| m02 | day-03-layers-forward-pass | FLOOR_MET | relevance, momentum, breadth_spark |
| m02 | day-04-loss | FLOOR_MET | momentum |
| m02 | day-05-gradients-backprop | FLOOR_MET | relevance, momentum |
| m02 | day-06-training-loop | FLOOR_MET | relevance, momentum |
| m02 | day-07-optimizers | FLOOR_MET | relevance, momentum |
| m02 | day-08-learning-rate | FLOOR_MET | relevance |
| m02 | day-09-train-val-test | FLOOR_MET | aspiration_hook, relevance, momentum, breadth_spark |
| m03 | day-01-embeddings | FLOOR_MET | relevance, momentum |
| m03 | day-02-qkv | FLOOR_MET | — |
| m03 | day-03-attention-scores | FLOOR_MET | — |
| m03 | day-04-multihead | FLOOR_MET | relevance, breadth_spark |
| m03 | day-05-positional | FLOOR_MET | relevance |

## Reading

- **The floor works and discriminates.** Every lesson clears the floor (consistent with the audit's "warm, GOOD, no P0" verdict) while the per-lever `WEAK` flags reproduce the audit's exact findings: **`momentum`** weak precisely on the m02 late-failure-wall days (D3–D7, D9), and **`relevance`** thin across both modules (ChatGPT once in the hook, then gone). D9 is the weakest (4 weak levers), matching its MIXED audit rating.
- **m03 is now judged at all** — the whole point of decoupling interest from the notebook. Its weak levers (relevance, breadth) are the concrete targets for the content phase.
- **No lesson is `BELOW_FLOOR`**, so the gate would not block any current lesson — it flags the `WEAK` levers as P1 improvements, exactly the right behavior (enforce a floor; surface targets without false-blocking warm lessons).

The `WEAK` levers (momentum on the failure-wall days; relevance/breadth) are what the **content phase** (interleave play between traps; recurring real-world hooks; breadth teases) would address on the m02/m03 rebuild.
