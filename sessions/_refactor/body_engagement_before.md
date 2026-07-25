# Concept-body engagement — BEFORE state (per day)

Captured 2026-07-24 before the Task-10 rebuild of m02+m03.
`body` = judge_body_engagement grades (stochastic; the aggregate at commit 4758589 read 108 GOOD / 1 WEAK).
`wall` = longest unbroken prose run in a concept build-up, widget blocks stripped (`sessions/_density_scan.py`).

| Module | Day | GOOD | WEAK | MISS | NA | Interest | max wall | mean wall | walls>600 | mean prose/concept |
|---|---|---|---|---|---|---|---|---|---|---|
| m02-the-neuron | day-01-single-neuron | 12 | 0 | 0 | 1 | FLOOR_MET | 1040 | 559 | 3 | 1645 |
| m02-the-neuron | day-02-activations | 11 | 0 | 0 | 1 | FLOOR_MET | 1335 | 575 | 5 | 1574 |
| m02-the-neuron | day-03-layers-forward-pass | 9 | 0 | 0 | 1 | FLOOR_MET | 681 | 404 | 1 | 1519 |
| m02-the-neuron | day-04-loss | 8 | 1 | 0 | 1 | FLOOR_MET | 602 | 422 | 1 | 1513 |
| m02-the-neuron | day-05-gradients-backprop | 7 | 1 | 0 | 1 | FLOOR_MET | 684 | 438 | 1 | 1933 |
| m02-the-neuron | day-06-training-loop | 8 | 0 | 0 | 1 | FLOOR_MET | 544 | 388 | 0 | 1757 |
| m02-the-neuron | day-07-optimizers | 5 | 0 | 0 | 1 | FLOOR_MET | 689 | 431 | 1 | 2178 |
| m02-the-neuron | day-08-learning-rate | 5 | 0 | 0 | 1 | FLOOR_MET | 464 | 348 | 0 | 1561 |
| m02-the-neuron | day-09-train-val-test | 8 | 0 | 0 | 1 | FLOOR_MET | 757 | 470 | 1 | 1739 |
| m03-attention | day-01-embeddings | 8 | 0 | 0 | 1 | FLOOR_MET | 1258 | 592 | 3 | 1659 |
| m03-attention | day-02-qkv | 7 | 0 | 0 | 1 | FLOOR_MET | 952 | 566 | 3 | 1759 |
| m03-attention | day-03-attention-scores | 8 | 0 | 0 | 1 | FLOOR_MET | 451 | 384 | 0 | 1423 |
| m03-attention | day-04-multihead | 5 | 0 | 0 | 2 | FLOOR_MET | 492 | 378 | 0 | 1091 |
| m03-attention | day-05-positional | 5 | 1 | 0 | 1 | FLOOR_MET | 596 | 439 | 0 | 1167 |

**Totals:** 106 GOOD / 3 WEAK / 0 MISSING / 15 NA across 124 concepts in 14 days. All 14 days clear the body floor and the interest floor.

**Density (the judge is blind to this):** mean 1609 prose chars per build-up, 17 concepts over 2500, worst 4931, 19 walls over 600 chars.
**Chunking widgets in use: ZERO** — no `%%% steps`, no `%%% insight`, no `predict:` in any of the 14 days.

This is why the rebuild is warranted even though the voice judge passes every day: the bodies are WARM but UNCHUNKED, matching the user report of "hard to digest, tedious".
