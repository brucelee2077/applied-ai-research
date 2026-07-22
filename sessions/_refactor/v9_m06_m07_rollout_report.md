# V9 Rollout Report — m06-cnns-vision-encoders + m07-thinking-in-jax

**Date:** 2026-07-22
**Branch:** `build/capability-spiral`
**Engine:** `sessions/_compiler/workflows/lesson_build.js` (V9 lesson-build engine)
**Goal:** Continue the V9 concept-lesson rollout to the next two modules after the m2–m5 completion.

---

## Result: PASS — 14/14 lessons

| Module | Days | Mode | Quest-ids | Status | Commits |
|--------|------|------|-----------|--------|---------|
| m06-cnns-vision-encoders | 8 | First-Principles (null yardstick) | `wf8-d0X` | ✅ pass, 0 open P0 | `b53bc58`…`ebc3298` |
| m07-thinking-in-jax | 6 | Reference / JAX (null yardstick) | `w01-d0X` | ✅ pass, 0 open P0 | `39b52bc`…`50d9bd8` |

Every one of the 14 days: `mode: concept`, compiled **exit 0 with all deterministic gates pass**
(Reader Flow · Concept Shell · Visual Integrity), **converged 0 P0**, and passed every judge lens that
applies (coverage vs the pre-seeded manifest covers; tone + interest **N/A** — both modules are
null-yardstick; concept-structure **analogy = GOOD**; correctness). **0 `%%% viz` embeds** — all
interactive widgets are inline `%%% svg` + `<script>` so they render under `file://`. `node --check`
clean on all widget scripts. Frozen invariants intact; `experiment.py` / `log.md` / `notes-log.md`
untouched everywhere.

---

## Per-day convergence

**m06** (spine: *seeing — a stencil that slides, then a stack that turns pixels into meaning*):

| Day | Title | Spine | Rounds | Notes |
|-----|-------|-------|--------|-------|
| d1 | Convolution & Pooling | stencil | 4 | Residual coverage exec_gap on backprop (a correctly-deferred forward-pointer) cleared by adding the manifest coverage block — no rebuild |
| d2 | Output-Size Arithmetic | tape-measure | 0 | clean first pass |
| d3 | A Small CNN Classifier | funnel | 3 | engine self-corrected a recap analogy=WEAK → fresh "field-trip scrapbook" analogy |
| d4 | Batch Normalization | thermostat | 0 | clean first pass |
| d5 | Data Augmentation | disguise | 1 | |
| d6 | Transfer Learning & Embeddings | hand-me-down | 0 | clean first pass |
| d7 | Object Detection Intuition | spotlight | 0 | clean first pass |
| d8 | CLIP & Contrastive Learning | matchmaker | 0 | module finale (nav → review.html) |

**m07** (spine: *recipe — JAX is functional: pure functions in, transformed functions out*):

| Day | Title | Spine | Rounds | Notes |
|-----|-------|-------|--------|-------|
| d1 | JAX Immutability | stone-tablet | 1 | self-corrected pure-functions MENTIONED→TAUGHT (new vending-machine concept) + fixed a `.at(i)`→`.at[i]` JAX-syntax typo in fin_body |
| d2 | Randomness With Explicit Keys | key | 1 | |
| d3 | vmap: Automatic Vectorization | cookie-cutter | 0 | clean first pass |
| d4 | jit: Compilation With XLA | blueprint | 0 | clean first pass |
| d5 | Flax & Optax | toolkit | 1 | |
| d6 | ViT Capstone | assembly | 1 | module finale (nav → m08 day-01) |

---

## Key process learning (reusable)

**Pre-seed the manifest `coverage.<day>` blocks BEFORE building.** m06 day-01 was built with no manifest
and ran the full 4 rounds without converging — the only residual "P0" was the coverage judge flagging
**backprop as MENTIONED-not-TAUGHT**, even though backprop is correctly deferred (day-01 is the forward-pass
mechanism; the correctness judge itself confirmed day-02 is output-size arithmetic, not training). The
coverage judge reads `deferred`/`out_of_scope` from the **manifest**, not the front-matter, so with no
manifest it has no deferral list. Fix (the m03 pattern): create the module manifest with per-day
`coverage.<day>` blocks (covers + deferred + out_of_scope), then re-run **just** `coverage_judge.py` — no
rebuild. Pre-seeding all remaining days' coverage blocks up front then gave near-uniform first-pass
convergence (7 of the remaining 13 days converged on round 0) and roughly a 3× drop in agents/tokens per
day (25 agents → 7) versus the un-seeded day-01.

**Both modules stayed under the genAI bridge $300/day cap** because `notebook_yardstick: null` makes the
tone + interest judges return N/A **without** a bridge call — only coverage-check-A and concept-structure
hit the bridge per round (~2 calls), far cheaper than the m02 real-yardstick runs that exhausted the cap
in ~10 lessons.

**Frozen-scheme divergence handled:** m06 uses the merged `wf8` / "Module 06 · Represent" scheme; m07
keeps the ORIGINAL week-based scheme (`w01-d0X` / "Thinking in JAX · Week 1" / "Frontier Lab · Week 1",
d1.prev = m06 review, d6.next = m08). Both preserved verbatim via the per-day `frozen` front-matter arg.

---

## Verification performed

- Deterministic recompile of all 14 days → exit 0, all gates pass.
- `concept_structure_gate.py` → PASS on all 14.
- `<iframe>` count = 0 and `%%% viz` count = 0 in every day (widgets inline).
- `node --check` on every inline widget `<script>` → 0 parse failures.
- Frozen-invariant spot-check per day: `<title>`, hero kicker, sidebar group label, `data-quest-id`,
  and the prev/next nav hrefs all match the pre-rollout wired values.
- `git status` per day showed only `source.md` (new) + `lesson.html` (modified) — no collateral edits.

## Tracker reconciliation

The tracker was stale (last_updated 2026-07-11, current_target = paused m04). Updated to a coherent
frontier: m04 (v9 rebuild superseding the v8 pause) + m05a + m05b + m06 + m07 → `pass`; m02 + m03 stay
`pass_with_p1`; summary recomputed (pass 5 / pass_with_p1 2 / not_started 37); `current_target` and
`next_target` → **m08-transformer-math**.
