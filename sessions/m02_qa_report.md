# QA Report — Module 2 "The Neuron & How It Learns"

_Gate run after the source-free first-principles refactor of `sessions/m02-the-neuron`._

## Scope

In scope and edited: 9 lesson files (`day-01`…`day-09/lesson.html`), 2 review gates
(`review-part-a.html`, `review.html`), 9 `experiment.py` stubs (verified, unchanged — already
conformant). Read-only: `sessions/index.html` (hub label + quest-id + badge check). No file outside
`sessions/m02-the-neuron/` was modified. Contracts validated against: `m02_coverage_contract.md`,
`m02_visual_contract.md`, `m02_artifact_contract.md`, `m02_refactor_plan.md`,
`m02_first_principles_blueprint.md`.

## Coverage contract result — **satisfied**

- All 29 must-cover concepts land on their named lesson at stated depth.
- The four **promoted** items (the biggest first-principles gaps in the prior draft) are now owned:
  - **Symmetric initialization** — Day 1 staff-lens (+ init-scale reinforced Day 6). ✓
  - **Vanilla gradient descent named** as the baseline — Day 5, referenced from Day 7. ✓
  - **Non-convexity / local vs global minimum** — Day 5 caveat + misconception corrected. ✓
  - **Logits → softmax + log-sum-exp** numerical stability — Day 4 should-cover callout. ✓
- New required tables added: **MSE ↔ cross-entropy** (Day 4), **batch-size trade-off** (Day 6).
- Every lesson carries the full Coach Layer (Jargon tooltips · Math Ladder · one silent-failure + one
  trade-off · interview answer · diagnostic quiz Q): `staff_lens_audit` **gap: 0** across all 9.
- Additional closures: `zero_grad` in the loop skeleton + batch=mean-of-per-example (unbiased) + val
  tracking (Day 6); exploding-gradient mirror (Day 5); capacity + L2/dropout mechanisms (Day 9);
  AdaGrad/RMSProp lineage + β₂/ε defaults (Day 7); curvature-spectrum + input-scaling + LR–batch (Day 8).

## Visual contract result — **satisfied (P0 met)**

- Every lesson renders its two P0 surfaces: **playground** (3 buttons) + **scroll-reveal build**
  (6 SVG steps). jsdom confirms `play:3 build:6` on all 9.
- Live `viz/gradient-descent.html` embeds intact on Days 5/7/8 (0 JS errors, iframe present).
- Day 4 numeric forward-pass added on Day 3 (was shapes-only); Day 5 numeric chain-rule added.
- **P1 (justified deferral):** Day 9 has no live interactive embed — the static two-curve diverge
  build + the degree-sweep Produce artifact cover the learning question; a live slider is nice-to-have.

## Artifact contract result — **satisfied**

- All 9 Produce run commands match their folders exactly (verified).
- Produce prompts + acceptance criteria are now numerically consistent with each day's playground and
  Math Ladder (fixed the prior drift on Days 7/8/9; CE ordering on Day 4; `np.maximum` on Day 1).
- Day 3 Produce Option-B authoring artifact ("W2 (4,4)… use W2 (4,2)") cleaned.
- `experiment.py` stubs remain guided placeholders with correct commented run paths + Option A/B.

## Checks run

| Check | Result |
|---|---|
| Scope boundary | ✓ only m02 + contracts/reports; hub read-only |
| `nav_audit.py` (whole site) | ✓ PASS — 291-page chain intact, 0 broken links, 0 case mismatches |
| `lesson_audit.py m02-the-neuron` | ✓ 9 OK / 0 missing / 0 leftover / 0 degraded |
| `staff_lens_audit.js` (m02) | ✓ staff-lens present on 9/9, **gap 0**; `render:BROKEN` is a benign tool quirk (it counts `.sec`, lessons use `.module-section`) — confirmed identical on untouched m01 |
| `node --check` all inline JS (11 files × ~4 blocks) | ✓ ALL OK |
| jsdom render (correct selectors) | ✓ all 11: 7 secs + 7 checklist + 3 play + 6 build + 4 q / gates 3 secs + 6 q; **0 JS errors**; new tables render |
| Math correctness (programmatic) | ✓ MSE 0.167, CE 1.204/0.511/0.105, chain-rule dL/dw −0.212, loop loss 0.746, momentum g −9.6 → v −24.0 → move 1.2 vs 0.48, tipping lr 0.25 |
| Stale label sweep | ✓ 0 stale "Module 3"/"Module 2-3"/"M2-3" (genuine M1 refs + M3·Attention links preserved) |
| Quest-id integrity | ✓ all 11 frozen ids unchanged (localStorage + hub pills + "First Neuron" badge safe) |
| Duplicate paragraphs | ✓ none |
| Title/finale integrity | ✓ no truncation; all finales relabeled to Module 2 |
| Legacy skill references | ⚠ see P1 |

## P0 findings

**None.** No missing must-cover concept, no incorrect math, no misleading/absent P0 visual, no broken
JS/nav/quiz/playground, no stale run command, no Produce path mismatch, no duplicate/truncated content.

## P1 findings

1. **`/frontier-experiment-lab` legacy reference (curriculum-wide, not m02-specific).** The skill is not
   installed (only the four v6 frontier skills are), yet every lesson's Produce **Option B** says
   "Use /frontier-experiment-lab …". This spans **243 files** across the whole curriculum, including
   m02's 9. **Not fixed here on purpose:** changing m02 alone would desync it from 234 sibling lessons,
   and the Option-B prompt is fully self-describing (it still instructs Claude to create + run the exact
   file, so it degrades gracefully). **Recommend** a separate curriculum-wide sweep to replace the
   trigger with a current v6 skill or plain instruction.

## P2 findings

1. **Build-viz light canvas on dark themes.** The `.build-viz` panel deliberately forces a light
   background so the hardcoded light-palette SVGs read correctly; on dim/dark/midnight the six-step
   panels stay bright. Intentional and consistent module-wide — not changed.
2. **Day 9 live embed** — static-only visual (tracked as the P1 visual deferral above; cosmetic).

## Fixes applied (this refactor)

- **Identity split (P0) resolved:** all learner-facing labels unified to "Module 2 · Day 1–9"
  (titles, kickers, brand-subs, nav-groups, finales, Produce prompt headers) + `review.html`
  "Module 3→2", "Module 4→3", "M1–M3→M1–M2"; `review-part-a` forward copy reframed. **No quest-id touched.**
- **Numeric self-consistency:** Day 6 `0.747→0.746`; Day 9 best model `step 100 → step 200`;
  Day 5 forward-cost unified to `~3×`.
- **Playground ↔ Produce agreement:** Day 7 momentum uses the real continuation (`g=−9.6, v=−24.0, 1.2 vs 0.48`);
  Day 8 step-count reconciled; Day 4 CE ordering aligned.
- **Anchor spine:** Day 2 single matrix pair through playground/ladder/produce; Day 1 `np.maximum` unified.
- **Coverage adds:** the 4 promoted concepts + 2 tables + the Coach-Layer reinforcements listed above.
- **Intra-module cross-refs:** all "M2/M3/Day N" refs that pointed at the old wf3 sub-arc corrected
  (forward pass = Day 3, loss = Day 4, gradients = Day 5, loop = Day 6, optimizers = Day 7).
- **No premature close:** Day 6 "Foundations done / completes the story" removed (Days 7–9 remain);
  Day 9's capstone "completes the story" retained (correct — it is the final lesson).

## Remaining risks

- The `frontier-experiment-lab` P1 (curriculum-wide) — degrades gracefully but should be swept.
- `coverage_audit.py`'s 3 uncited master-index resources are TLA+ (formal-methods module) — unrelated
  to m02, pre-existing.
- Pixel-level dark-theme SVG contrast unverifiable in-sandbox (no real browser); logic verified via jsdom.

## Merge recommendation

**Pass with P1.** The m02 refactor meets all four contracts, closes every P0 defect, and passes all
audits + render + math checks. The single P1 (`frontier-experiment-lab`) is a pre-existing,
curriculum-wide, gracefully-degrading item that should be handled in its own sweep, not by desyncing m02.
