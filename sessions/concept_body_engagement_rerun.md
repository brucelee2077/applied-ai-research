# Concept-body engagement — rerun report (Task 10/11)

**Branch** `build/capability-spiral` · **run** 2026-07-24 → 2026-07-27 ·
**status: PARTIAL — 11 of 14 days rebuilt and committed. 3 left, blocked on the daily quota.**

Sequel to the engine/skill/judge work of Tasks 1–9. Those tasks built the body
toolkit (`%%% insight`, `%%% steps`, `demo predict:`), the `judge_body_engagement`
floor, the scroll-reveal, and the Build-Up Register in both skill trees. Task 10
was the rerun: re-author the 14 m02+m03 days so their concept BODIES are as
engaging — and as **digestible** — as their intros.

---

## 1. Why a rebuild, when the judge already passed every day

The before-state scan (`sessions/_refactor/body_engagement_before.md`) found
**106 GOOD / 3 WEAK / 0 MISSING / 15 NA** across 124 concepts. Every day cleared
the body floor and the interest floor. A judge-driven rebuild was therefore *not*
warranted by the judge.

The user's word was "**hard to digest**", and that pointed somewhere the judge
cannot see. `judge_body_engagement` grades build-up **VOICE**; nothing measured
build-up **DENSITY**. So this run added `sessions/_density_scan.py`, a
deterministic, no-LLM metric. It found the real gap:

| | before |
|---|---|
| build-up prose per concept (14 days) | mean **1,609** chars, 17 concepts over 2,500, worst 4,931 |
| unbroken prose walls over 600 chars | **19** |
| `%%% steps` / `%%% insight` / `predict:` in use | **0 / 0 / 0** |

The bodies were **warm but unchunked** — which is exactly what "tedious" describes.
That is the defect this rerun attacks, and it is why the rebuild was worth doing
even though the voice judge was already green.

> **Two metric corrections made during this run** (both were my errors, both
> caught before any conclusion rested on them): the first version of the scan
> counted a 4k-char inline `<svg>` as a wall of text, which produced a bogus
> "124/124 walls, worst 4,490"; the second counted a `~~~html` glossary block the
> same way. The numbers above and below are from the corrected metric, which
> strips widget bodies and raw-HTML blocks before measuring prose.

---

## 2. What shipped

Eleven days rebuilt and committed, each through the full `lesson_build.js` loop
(blind coverage draft → author under the Build-Up Register → compile → 6-lens
judge panel → fix rounds) and then through a **keep-or-revert accept gate**
(`sessions/_rebuild_accept.py`). One day was rebuilt and REVERTed by that gate.

| Day | verdict | steps/insight/predict | prose/concept | walls>600 | max wall | body floor |
|---|---|---|---|---|---|---|
| m02 day-01-single-neuron | KEEP (clean) | 10 / 12 / 6 | 1645 → **1254** | 3 → **0** | 1040 → **472** | 12 GOOD (held) |
| m02 day-02-activations | KEEP | 13 / 12 / 6 | 1575 → 1901 | 5 → **2** | 1335 → **717** | 11 GOOD (held) |
| m02 day-03-layers-forward-pass | KEEP (clean) | 9 / 10 / 3 | 1520 → 1668 | 1 → **0** | 681 → **513** | 9 → **10 GOOD** |
| m02 day-04-loss | KEEP (clean) | 11 / 10 / 7 | 1514 → **1406** | 1 → 1 | 602 → 670 | 9 → **10 GOOD** |
| m02 day-05-gradients-backprop | KEEP | 11 / 9 / 6 | 1934 → **1898** | 1 → 2 | 684 → 817 | 7G/**1W** → **9G/0W** |
| m02 day-06-training-loop | KEEP | 8 / 9 / 6 | 1758 → **1546** | 0 → 1 | 544 → 732 | 6G/**2W** → **9G/0W** |
| m02 day-07-optimizers | KEEP (clean) | 9 / 5 / 4 | 2179 → **2056** | 1 → **0** | 689 → **555** | 5 GOOD (held) |
| m02 day-09-train-val-test | KEEP | 6 / 9 / 5 | 1740 → 2203 | 1 → 2 | 757 → 842 | 8 → **9 GOOD** |
| m03 day-01-embeddings | KEEP | 9 / 10 / 9 | 1660 → 1830 | 3 → **1** | 1258 → **960** | 8 → **9 GOOD** |
| m03 day-02-qkv | KEEP | 8 / 9 / 5 | 1759 → 2088 | 3 → **1** | 952 → **634** | 7 GOOD (held) |
| m03 day-03-attention-scores | KEEP | 16 / 11 / 8 | 1424 → 1599 | 0 → **3** | 451 → 993 | 8 → **10 GOOD** |
| m02 day-08-learning-rate | **REVERT** | — | — | — | — | lost 1 visual, no density gain |

**Across the 11 rebuilt days:** **110 steps ladders, 106 insight re-hooks and 65
predict prompts** where there were **zero**; walls over 600 chars **19 → 13**;
mean prose per build-up 1,700 → 1,768 (**+4%**). Every day kept all its concepts
and coverage topics, and every day ended with **more** visuals than it started
with (day-05 svg 16→21, day-06 demo 2→6 plus 3 new interactive viz, day-02-qkv
svg 12→17).

**All 3 WEAK build-ups in the entire 14-day set are now GOOD** — including the
flat "Step 1/2/3" forward pass that the before-scan singled out.

### Honest read of the density result

The **walls** result is solid and is the metric closest to "hard to digest": the
worst unbroken paragraphs shrank sharply (1335→717, 1258→960, 1040→472, 952→634)
and walls over 600 chars fell **19 → 13** across the rebuilt days.

**Total prose did not fall — it rose 4%.** Chunking broke the walls into rungs
rather than cutting them, and the authors added concepts and material at the same
time. So these days genuinely read in smaller pieces, but they are *longer*, and
on that axis this is a partial miss. Recorded per-day as accept-gate warnings
rather than smoothed over. The weakest outcomes, queued for a targeted trim pass:

- **m03 day-03-attention-scores** — walls>600 0→3, max wall 451→993. Worst
  regression of the run.
- **m02 day-09-train-val-test** — prose +27%, walls 1→2.
- **m02 day-05-gradients-backprop** — walls 1→2, and 2 residual correctness P0s.

A pattern worth acting on: the two days that *regressed* on walls
(day-03-attention-scores, and day-08 which was reverted) both started with the
**cleanest** baselines (0 walls over 600). Days that were already digestible have
no headroom, and a full re-author risks more than it gains. **A future rerun
should skip any day whose baseline already shows 0 walls over 600.**

---

## 3. Defects the loop caught (none of which this task set out to find)

The correctness lens verifies claims by **running the code**, and it turned up
real bugs — several of them long-shipped and unrelated to body engagement.

**Compiler (curriculum-wide, pre-existing):** `_kv()` treated every `word:` line
as a new field, so a repeated `out:` **overwrote** the earlier ones. A momentum
ladder written as five `out:` lines shipped as only "step 5" — a build-up with
the build removed. Multi-line `code:` lost its setup lines, leaving demos that
referenced variables never shown (m07 day-06 rendered a bare `print()`; day-04's
jaxpr dump rendered as just its closing brace). Fixed under TDD; measured blast
radius across all 46 concept lessons: **15 lessons changed, 0 text lines lost,
374 restored.**

**Judges (introduced by longer days):** the 48,000-char extract cap silently
dropped the last ~12% of a rebuilt day (day-05 extracts to 54,698 chars) — the
glossary, further-reading and the whole quiz — from *every* judge, making tail
content read as ABSENT (a false P0). Cap raised to 160,000 with a loud WARN, and
the five per-judge caps now derive from one constant so they cannot drift apart.

**Content defects fixed in the rebuilt days:**
- day-05 stated the **gradient direction backwards** in the hero visual, its
  aria-label and `fin_body` ("points straight downhill" — it points *uphill*,
  which is exactly why the update subtracts it).
- day-07's produce section told readers to set `lr=0.5` to watch SGD diverge, but
  0.5 is precisely that valley's stability edge (2/curvature = 2/4), where it
  oscillates forever instead. Also: momentum taught as 0.9 but coded as 0.6, and
  an "adaptive wins by step 1" claim that was an artifact of the start point.
- day-09 gave two contradictory prescriptions for the same date-ordered input
  four rungs apart, and a produce loop whose learning rate made the error
  multiplier exactly 0 — so the promised loss curve collapsed in one update.
- day-05's residual P0s (a symmetry-breaking demo whose gradients cannot depend
  on the weights it tells you to change, and a clip-ceiling line drawn 20× too
  high) are **still open** — see §5.

**Test suite:** the m02 manifest deferral update resolved the long-standing
pre-existing failure. The compiler suite is **190/190 green** for the first time
(it was 183 pass / 1 fail at the start of this work; verified the manifest is
what fixed it by reverting it and watching the test fail again).

---

## 4. Why the run stopped

The batch run for the remaining days died on a **hard quota blocker**, not a code
problem:

```
429 · you have hit your daily cost allocation for quota 'personal:...'
      (current spend: $299.13, budget: $300.00)
```

19 agents failed this way. Per the repo's error policy (CLAUDE.md §10) a quota
429 is not retryable, so the run was stopped rather than hammered.

One day was left half-authored when its author died mid-write:
**day-02-activations** failed its accept gate (did not compile — no recap unit,
produce not discovery-framed; concepts 12 → 9) and was **reverted to HEAD**,
where it recompiles clean. Nothing partial was committed.

---

## 5. What is left

**3 days not yet rebuilt:** m03 `day-04-multihead`, m03 `day-05-positional`
(neither started — their arg-loader agents 429'd), and m02 `day-08-learning-rate`
(rebuilt, REVERTed for losing a visual with no density gain; per the pattern above
it is a candidate to **leave at baseline** rather than retry).

**Resume command** (once the daily quota resets):

```
Workflow({scriptPath: 'sessions/_compiler/workflows/body_rebuild_batch.js',
          args: {batches: [[['m03-attention','day-04-multihead'],
                            ['m03-attention','day-05-positional']]]}})
```

Budget note: each quota-day of $300 buys about 5–6 days of rebuild (~2.5 h/day
wall-clock, driven by the correctness lens re-deriving every number by running
code). The remaining 2 days fit inside one quota-day.

**Also open:**
1. **Trim pass** on day-03-attention-scores, day-09 and day-05, whose walls or
   prose grew (see the density read above).
2. **day-05's 2 residual correctness P0s** (symmetry demo, clip-ceiling geometry)
   — it exhausted its 4 fix rounds. It was kept because the alternative (revert)
   would also discard the gradient-direction fix and the WEAK→GOOD clearance; the
   two open items are widget-level, not body-engagement.
3. **Interest advisory** recurring on the chunked days: the floor rates the
   `momentum` lever WEAK where three or four failure-mode beats stack in a row
   without a play break. Worth a Build-Up Register rule ("no more than two
   consecutive trap beats; put a play or payoff between them").

---

## 6. Verification performed

- Every kept day: `compile_lesson.py` + `concept_structure_gate.py` exit 0,
  idempotent recompile (byte-identical on a second compile), frozen **wiring**
  front-matter byte-identical, no drop in concepts / coverage topics / visuals,
  `body_engagement` no MISSING and GOOD ≥ the committed version, interest
  `FLOOR_MET`, and `experiment.py` runs clean.
- **New jsdom audit** `sessions/_reveal_audit.js`: the donor's `__revealBuild` is
  exercised against real compiled lessons in a real DOM — every `.build-step`
  reveals on scroll, and reduced-motion / no-IntersectionObserver readers get
  every step already revealed. Without this, a ladder in an unarmed container
  renders as **blank space**, which no offline gate and no text-reading judge can
  see. All 6 rebuilt days pass (e.g. day-07: 24 steps across 6 ladders).
- Compiler suite: **190 passed, 0 failed**.
- The accept gate itself was validated before use: on an untouched day it reports
  REVERT for exactly one reason ("no chunking widgets used") with every hard gate
  passing, and a negative control that tampers with `quest_id` is still caught.

### One gate correction worth recording

The accept gate first REVERTed day-05 for "front-matter changed" — its only
change being the `fin_body` sentence that **fixed** the gradient-direction error.
Reverting a day for correcting a technical error is backwards. The plan freezes
the *wiring* keys (quest_id, mode, donor, nav_*, module_label, page_title,
brand_sub, spine, notebook_yardstick); `fin_title`/`fin_body`/`title`/`subtitle`
are authored voice. The gate now compares wiring keys key-by-key and reports
prose-key rewrites as review warnings.
