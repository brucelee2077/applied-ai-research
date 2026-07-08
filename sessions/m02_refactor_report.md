# Refactor Report — Module 2 "The Neuron & How It Learns"

_Source-free, first-principles refactor of `sessions/m02-the-neuron` (9 lessons + 2 review gates)._
_Old notebooks under `00-neural-networks/` were **not** read. Design intelligence came from a
first-principles workflow (11 agents: 4 read-only auditors on the current draft + 6 source-free
designers + 1 completeness critic); contracts and lesson edits were authored from that synthesis._

## What this refactor was

The prior draft was already strong (audit depth 4–5, full Coach Layer). So this was an **elevate + fix**,
not a blank rebuild: unify a systemic identity defect, close the first-principles coverage holes the
design surfaced, and fix per-lesson numeric/consistency defects — all while preserving the shell,
navigation, and every quest-id.

## Files changed

**Contracts + reports created (7):**
`m02_first_principles_blueprint.md`, `m02_coverage_contract.md`, `m02_visual_contract.md`,
`m02_artifact_contract.md`, `m02_refactor_plan.md`, `m02_qa_report.md`, `m02_refactor_report.md`.

**Lessons edited (9):** `day-01-single-neuron` … `day-09-train-val-test/lesson.html`.
**Gates edited (2):** `review-part-a.html` (forward copy), `review.html` (full Module 3→2 relabel).
**Stubs (9):** `experiment.py` verified conformant — no change needed.
**Not touched:** any `data-quest-id`; the shell CSS/JS engine; nav chain; `index.html` (read-only check).

| File | Headline changes |
|---|---|
| day-01 neuron | staff-lens → **symmetric init** (removed D1/D2 dead-ReLU dup); bias-geometry callout; build width note; `np.maximum` unified; quiz Q4 → symmetric-init diagnostic |
| day-02 activations | **scalar-first** collapse proof; **one anchor matrix pair** across playground/ladder/produce; Leaky-ReLU/GELU named; sigmoid "rounded" note; `M3→Day 5` refs |
| day-03 forward pass | **logits** defined + `x@W` convention pinned; **numeric forward pass** added; broadcasting folded into staff-lens; Option-B authoring artifact cleaned; batch+hand-check in Produce |
| day-04 loss | **M3·Day1 → M2·Day4** relabel; **MSE↔CE table**; **MSE Math Ladder**; multiclass CE; loss=mean; **logits/log-sum-exp** callout; landscape handoff; CE ordering fixed |
| day-05 gradients | **M3·Day2 → M2·Day5**; **numeric multi-link chain rule**; **"vanilla gradient descent" named** as baseline; **non-convexity** caveat; **exploding-gradient** mirror; `~3×` unified |
| day-06 training loop | **M3·Day3 → M2·Day6**; `0.747→0.746`; `zero_grad` in skeleton; init-scale + epoch/batch(mean-unbiased)+val-tracking callouts; **batch-size table**; premature-close removed |
| day-07 optimizers | **M2-3·Day4 → M2·Day7**; **momentum step-2 → real `g=−9.6, v=−24.0, 1.2 vs 0.48`** (3 places); β₂=0.999/ε=1e-8 + bias-correction; AdaGrad/RMSProp lineage; GD-baseline callback; stale day-refs fixed |
| day-08 learning rate | **M2-3·Day5 → M2·Day8**; symbol disambiguation (`c` vs `v`); step-count reconciliation; curvature-spectrum + input-scaling + LR–batch callout; stale day-refs fixed |
| day-09 train/val/test | **M2-3·Day6 → M2·Day9**; **best model step 100 → step 200** (§4 + Math Ladder); capacity-as-cause + L2/dropout mechanisms + two-axis reconciliation |
| review-part-a | forward copy "Gate Before Module 3" → "Halfway Gate / training half of Module 2" |
| review.html | **full P0 relabel** Module 3→2 (title/nav/hero/verdict/finale/footer); Module 4→3; M1–M3→M1–M2 (M3·Attention next-links correctly kept) |

## Coverage satisfied

All 29 must-cover concepts land at stated depth; the 4 promoted items (init, named GD, non-convexity,
logits/log-sum-exp) are each visibly owned; MSE↔CE and batch-size tables present; Coach Layer on 9/9
(`staff_lens_audit` gap 0). See `m02_qa_report.md` for the per-item breakdown.

## Visuals added / improved

- Day 3: concrete **numeric forward pass** (was shapes-only).
- Day 5: **numeric multi-link chain-rule** ladder; non-convexity note on the descent picture.
- Day 4 / Day 6: two required **comparison tables** (theme-safe inline styling via CSS vars).
- All existing P0 surfaces preserved and verified: playground (3 buttons) + build (6 SVG steps) per
  lesson; `gradient-descent.html` embeds on Days 5/7/8. Analogy chain kept (skilled-task → terrain →
  generalization) with the Day-4→Day-5 landscape handoff scripted and a single gradient picture.

## Artifacts improved

9/9 Produce run commands correct; prompts + acceptance criteria now numerically consistent with each
day's playground (fixed Days 4/7/8/9 drift); Day 3 Option-B artifact cleaned; Day 1 `np.maximum`
unified; stubs conformant.

## QA result

**Pass with P1.** nav_audit PASS · lesson_audit 9 OK · staff_lens gap 0 · `node --check` all JS OK ·
jsdom render all 11 pages OK (0 JS errors) · all new/changed math verified programmatically · 0 stale
labels · quest-ids intact · no duplicates. Full detail in `m02_qa_report.md`.

## P0 / P1 / P2 remaining

- **P0:** none.
- **P1:** `/frontier-experiment-lab` (uninstalled) referenced in every Produce Option B — a
  **curriculum-wide** pattern (243 files, incl. m02's 9). Left as-is to avoid desyncing m02 from 234
  sibling lessons; the prompt degrades gracefully. → own sweep.
- **P2:** build-viz light-canvas on dark themes (intentional, module-consistent); Day 9 has no live
  interactive embed (static build is adequate).

## Recommendation

**Ship m02.** The module now reads as a coherent, notebook-quality 9-day story with a single Module-2
identity, closed coverage holes, and verified-consistent numbers. **Continue** to the next module when
ready. Separately, schedule the curriculum-wide `frontier-experiment-lab` reference sweep (P1) — it is
the only cross-cutting item and is out of this refactor's scope.
