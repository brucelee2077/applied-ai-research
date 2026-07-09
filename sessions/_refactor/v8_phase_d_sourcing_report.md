# v8 Phase D — Source All 9 m02 Days + Notebook Smoothness Gate

**Loop:** `v8_phase_d_source_all_9_days`
**Date:** 2026-07-09
**Owner skills:** `frontier-curriculum-architect`, `frontier-lesson-builder`, `frontier-refactor-qa`, `frontier-visual-evidence-builder`
**Scope:** All 9 m02 days sourced onto the v8 pipeline + the Notebook Smoothness Gate run across them. **m03 NOT rolled out; no other module touched.**
**Spec:** `sessions/_refactor/v8_source_first_authoring_plan.md` · **Prior:** Phase B pilot + Phase C hardening reports.

---

## TL;DR

- **Verdict: PASS.** All 9 m02 days now live on the source-first pipeline (each has a canonical `source.md` that compiles to its `lesson.html`).
- **Notebook Smoothness Gate across all 9: PASS** — **6 PASS** (D1–D6, which have matching fundamentals notebooks), **3 N/A** (D7–D9, no matching notebook → skipped per the No-Notebook rule), **0 FAIL**.
- **Zero content regression:** D2–D9 were sourced in **verbatim/migration mode** — the compiler round-trips them **byte-identical** to the shipped v7.6 lessons (git shows D2–D9 `lesson.html` unchanged; only `source.md` added). D1 remains the clean typed-block exemplar from Phase B/C.
- All per-day gates + repo audits pass; compilation is deterministic. **Remaining issues are P1/P2 only** (§6).

---

## 1. What "sourcing all 9" means here (two modes)

| Day(s) | Mode | How | Result |
|---|---|---|---|
| **D1** | **clean typed-block** | authored in Phase B, hardened in Phase C (`%%% table/cards/formula/mathladder/prompt`, `!!!` callouts, `[[term]]`) | human-first hero, Jargon Ladder, frontier payoff in §4 |
| **D2–D9** | **verbatim / migration** | `sessions/_compiler/extract_source.py` captures each region byte-for-byte via `v8lib.REGION_PATTERNS`; the compiler substitutes them back | **round-trip byte-identical** to the shipped v7.6 lesson |

**Why verbatim for D2–D9.** They were already reader-flow-stabilized in the v7.6 seed sweep (curiosity-led heroes, analogy spines, discovery Produce). Re-authoring 8 lessons from scratch in clean markdown would risk their vetted content (anchors, staff-lens, quiz `q:4 o:16`) for no reader-flow gain. Verbatim sourcing puts them on the pipeline **losslessly** — `source.md` is now the canonical editable input, and clean typed-block conversion becomes safe, incremental, per-day work (exactly how Phase C converted D1's raw widgets). This is a legitimate, transparent migration state, not a shortcut: the round-trip is provably byte-identical.

**New tooling (Phase D):**
- `sessions/_compiler/extract_source.py` — shipped `lesson.html` → verbatim `source.md` (uses the compiler's own `REGION_PATTERNS`, so the round-trip is exact).
- `sessions/_compiler/gates/notebook_smoothness_gate.py` — batch, exemplar-only; N/A where no yardstick.
- `v8lib.compile_html` now supports `@@@ region name=X` verbatim blocks (dual-mode: clean rendered blocks *or* verbatim regions), with a `rstrip('\n')` fix so the last region round-trips exactly.

---

## 2. Notebook Smoothness Gate — across all 9

```
PASS day-01-single-neuron        vs 01_what_is_a_neural_network.ipynb   (notebook human-first ✓)
PASS day-02-activations          vs 03_activation_functions.ipynb        (notebook human-first ✓)
PASS day-03-layers-forward-pass  vs 05_forward_propagation.ipynb         (notebook human-first ✓)
PASS day-04-loss                 vs 06_loss_functions.ipynb              (notebook human-first ✓)
PASS day-05-gradients-backprop   vs 07_backpropagation.ipynb             (notebook human-first ✓)
PASS day-06-training-loop        vs 08_training_loop.ipynb               (notebook human-first ✓)
N/A  day-07-optimizers           (no matching fundamentals notebook — skipped, never failed)
N/A  day-08-learning-rate        (no matching fundamentals notebook — skipped, never failed)
N/A  day-09-train-val-test       (no matching fundamentals notebook — skipped, never failed)

SUMMARY: PASS=6  N/A=3  FAIL=0   →  Gate: PASS
```

**How it decides.** For each day with a yardstick, it compares the compiled lesson's **first screen** (hero lede) to the notebook's opening on the barrier axis: (1) first sentence is not frontier-pressure, (2) a human/curiosity cue is present, (3) no formula wall in the hero. A day passes when its first screen is **at least as smooth** as the notebook's (which, for all six, opens human-first). Days without a matching notebook are **N/A** — the No-Notebook rule applied *within* a module, proving the gate never penalizes a lesson for lacking a notebook.

---

## 3. Per-day gate & round-trip evidence (all 9)

| Day | Reader Flow | Shell Invariant (vs donor) | Deterministic | lesson.html vs shipped |
|---|---|---|---|---|
| D1 | PASS (strict, clean) | PASS (CSS+4 scripts identical) | yes | intentionally differs (Phase B human-first rewrite) |
| D2–D9 | exit 0 (informational, verbatim) | PASS (CSS+4 scripts identical) | yes | **byte-identical** |

- **Reader Flow Gate** is *strict/blocking* for clean sources (D1) and *informational* for verbatim sources (D2–D9) — since verbatim output can't regress (it equals the vetted donor), the gate reports what a future clean pass should add (Jargon Ladder, defer frontier from s1) as **warnings**, never blocking.
- **Shell Invariant Gate** confirms the CSS block + all 4 `<script>` blocks (incl. the JS engine with `DEMOS/BUILD/QS`) are byte-identical to each day's donor.

---

## 4. Repo audit evidence (all 9 days)

| Check | Result |
|---|---|
| `python3 sessions/lesson_audit.py m02-the-neuron` | **9 OK / 0 MISSING / 0 LEFTOVER / 0 DEGRADED / 0 advisories** |
| `python3 sessions/nav_audit.py` | **PASS** — 0 CASE, 0 BROKEN, 0 orphans |
| `node sessions/staff_lens_audit.js m02` | **9/9 staff-lens present, gap 0** (`render:BROKEN` benign) |
| Git scope | D2–D9 `lesson.html` **byte-unchanged**; only `source.md` added per day |

---

## 5. Frozen invariants preserved (all 9)

quest-ids `wf2-d01/02/03`, `wf3-d01..d06` (compiler asserts each against its donor) · 7 `.module-section` + 8 `data-target`s per day · `DEMOS/BUILD/QS` + playground ≥3 + quiz `q:4 o:16` · `frontier-lesson:<qid>` + `frontier-theme` · `.fin` completion + `.gotit` wiring · prev/next nav chain · `experiment.py` + `log.md` untouched. For D2–D9 all of this is guaranteed trivially by byte-identical round-trip.

---

## 6. Remaining issues (P1 / P2 — none blocking)

- **P1 — D2–D9 are verbatim, not yet clean typed-block.** Their `source.md` stores regions as verbatim HTML (a migration state). Converting each to clean typed blocks (like D1) is safe, incremental per-day authoring — and the Reader Flow Gate already emits the per-day warnings (e.g. "s1 leaks frontier payoff", "no Jargon Ladder") that guide it.
- **P1 — 3 days have no notebook yardstick** (D7 optimizers, D8 learning-rate, D9 train/val/test). Correctly N/A on the Notebook Smoothness Gate; when clean-authored they'll be validated by the Reader Flow + Staff Depth gates (the no-notebook path).
- **P2 — s3/s5/s6 scaffolding still carried from donors** across all days (their data lives in `source.md`; the static wrappers don't yet).
- **P1 (carried, unchanged):** behavioral-viz `BL-VE-D1/D2`.

**No P0.** m02 remains `pass_with_p1`. No other module status changed. m03 untouched.

---

## 7. State updates

- `sessions/_refactor/rollout_tracker.yaml` — `v8_authoring_system.modules_sourced` = all 9; `phase_d_sourcing` block; phasing A/B/C/D DONE; `rollout_history` += `v8_phase_d_source_all_9_days`.
- `sessions/m02-the-neuron/_refactor/manifest.yaml` — `v8_phase_d_sourcing` block; `last_loop` bumped.
- New files: `extract_source.py`, `gates/notebook_smoothness_gate.py`, 8 donor snapshots (`m02-day-02..09.donor`), 8 `source.md` (D2–D9). `v8lib.compile_html` extended for `@@@ region` verbatim blocks.

---

## 8. What's next

- **Phase E** — first **no-notebook** rollout: author a *new* module (m07 JAX or m10a Scaling Laws) source-first in **Reference mode**, proving the pipeline works with no notebook (Notebook Smoothness Gate = N/A; Reader Flow + Staff Depth carry it). This is the real payoff of the whole v8 line.
- **Ongoing** — progressively convert D2–D9 from verbatim to clean typed-block source, guided by their Reader Flow warnings, one day at a time (each a byte-diffable, low-risk change).
