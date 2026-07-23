# Module 2 (`m02-the-neuron`) — V9 Beginner-Interest Audit

**Scope:** all 9 compiled `lesson.html` in `sessions/m02-the-neuron/` (day-01 … day-09), the build skills (`frontier-lesson-builder`, `frontier-curriculum-architect`, `frontier-refactor-qa`), and the engine (`coverage_judge.py`, `lesson_build.js`, the gates).
**Lens:** *Could a curious 12-year-old — English not their first language — read this, feel good, and WANT to keep going?* Interest is the #1 goal at this stage (per the standing 2026-07-16 directive).
**Method (two independent instruments):**
1. The project's **own interest + tone judges** (`coverage_judge.py`, notebook-relative) run live on Days 1–6 (Days 7–9 have `notebook_yardstick: null` → N/A).
2. An **independent 12-year-old-lens audit** of all 9 days + **adversarial verification** of each verdict (workflow `m02-beginner-interest-audit`, 20 agents).
**Date:** 2026-07-22 · **Mode:** report-only (no lesson/skill files edited).

---

## TL;DR

**The interest-first rebuild worked. Module 2 is genuinely beginner-friendly and curiosity-sparking — but it has a consistent, fixable ceiling on *want-to-keep-going*.**

- **Both instruments agree on the good news.** The project's own judge scores **Days 1–6 `MATCHES_NOTEBOOK` on every interest lever AND every tone lever** (a real change from the 2026-07-16 state, when Day 2 was `BELOW_NOTEBOOK`). Independently, **all 9 days open with a `STRONG_HOOK` and put the felt picture before any formula.** Openings are excellent: movie-friends→ChatGPT (D1), ruler-can't-draw-a-smiley (D2), car assembly line (D3), mini-golf scorecard (D4), hiker-in-fog (D5), free-throw (D6), rolling ball (D7), stepping-stones (D8), exam-vs-answer-key (D9).
- **No P0. Verdicts: mostly P1, a couple P2. Overall GOOD or MIXED.** The lessons are shippable and warm. The problems are concentrated and structural, not tone or correctness.
- **Four systemic weaknesses cost a 12-year-old's momentum** (detailed below): (A) every day is too **long**; (B) a late-lesson **failure-mode wall**; (C) the **"play" is mostly an illusion**; (D) **interview/Reader-B register leaks** into the beginner voice.
- **The engine can't defend against most of this:** interest is only gated where a notebook exists — **76% of built days ship with the interest gate silently OFF** — and **no gate caps length or density at all.**

---

## Per-day scan

| Day | Open | Picture→Formula | Load | Density slog | Play | Weak levers | Overall | Sev |
|---|---|---|---|---|---|---|---|---|
| 1 · Single neuron | STRONG_HOOK | YES | **LONG** (13 units) | — | 1 live widget | — (judge: MATCHES all) | GOOD | P2 |
| 2 · Activations | STRONG_HOOK | YES | **LONG** (12) | **NOTABLE** c7–c10 | MIXED | invites_play, momentum | GOOD | **P1** |
| 3 · Layers/forward | STRONG_HOOK | YES | **LONG** (10) | **NOTABLE** c7–c9 | MIXED (matmul live) | relevance, momentum, breadth | GOOD | **P1** |
| 4 · Loss | STRONG_HOOK | YES | **LONG** (10) | **NOTABLE** c7–c8 | MIXED (CE slider live) | invites_play, momentum, breadth | GOOD | **P1** |
| 5 · Backprop | STRONG_HOOK | YES | **LONG** (9) | **NOTABLE** c6–c7 | MIXED (slope slider live) | momentum, breadth | GOOD | P2 |
| 6 · Training loop | STRONG_HOOK | YES | **LONG** (9) | **NOTABLE** c4–c8 (5 in a row) | **MOSTLY_STATIC** | relevance, invites_play, momentum, breadth | MIXED | **P1** |
| 7 · Optimizers | STRONG_HOOK | YES | **LONG** (6) | **NOTABLE** c3–c5 | MIXED (gd slider live) | relevance, momentum | GOOD | **P1** |
| 8 · Learning rate | STRONG_HOOK | YES | **LONG** (6) | MILD (recap wall) | MIXED (gd slider live) | relevance, invites_play, breadth | GOOD | P2 |
| 9 · Train/val/test | STRONG_HOOK | YES | **LONG** (9) | **NOTABLE** c6–c8 | MIXED (gap slider live) | aspiration, relevance, invites_play, momentum, breadth | MIXED | **P1** |

Unanimous across all 9: **opening = STRONG_HOOK · picture-before-formula = YES · reading_load = LONG · momentum lever = WEAK (8/9) · exactly ~1 genuinely interactive widget per day.** Every adversarial verifier returned **CONFIRMED** (no verdict was too harsh or too lenient).

---

## Findings (systemic — ranked by interest impact)

### A. Every lesson is too LONG for a bite-sized learner *(9/9 = LONG)*
6–13 concept units per day; ~4.9k–6.4k reader-facing words (raw source counts run higher; SVG labels inflate them). Day 1 = 13 units, Day 9 = 9 units + 2 recap tables. The target learner is explicitly **"easily overwhelmed by long content, prefers bite-sized daily quests."** A day this long is the single biggest interest risk even when every paragraph is warm.
**Root cause:** nothing in the stack caps length. `reader_flow_gate` / `concept_shell_gate` / `concept_structure_gate` all enforce only a **floor** (`≥3 concept units`); there is no ceiling and no word budget anywhere.

### B. The late-lesson "failure-mode wall" *(density NOTABLE on 7/9; momentum WEAK on 8/9)*
After a strong opening and a mid-lesson win, each day piles **3–5 consecutive "Puzzle → Cause → Remedy / here's a limit" units** in the late-middle, right before the recap, with little play or payoff between them. Examples: D2 c7–c10 (saturation→dead ReLU→leaky cure→sigmoid drift), D6 c4–c8 (**five** in a row), D9 c6–c8, D7 c3–c5, D5 c6–c7. This is exactly where a kid bounces.
**Root cause:** the architect's **Coverage Spec Rule** mandates, for *every* failure taught, its cause **and** remedy, and *"when the fix is a named technique, the technique becomes its own covers topic"* — plus a capability limit per mechanism. `AUTHORING.md §7` then maps each covers topic to **one concept unit.** It is a completeness engine with **no momentum counterweight and no cap on consecutive failure units.** The density slog is not an authoring slip — it is what the rule expands into.

### C. "Play" is largely an illusion *(MIXED/STATIC on 9/9)*
The `%%% demo` "run it ▶" buttons don't compute — they un-hide a **pre-baked answer** (`<pre class="demo-out" hidden>1.6</pre>`; the button just reveals it). Verified directly in the HTML/JS. Each day has only **~1 genuinely interactive drag widget** (matmul, gradient-descent slider, cross-entropy slider, gap slider), and it usually lands late. A curious kid who clicks "run it" expecting to change numbers gets a canned reveal — Day 8's verifier called it feeling "tricked."
**Why it matters:** the 2026-07-16 directive names **play/agency ("I made something happen") as interest lever #1**, and notes the exemplar notebooks beat us precisely because their cells are *runnable*. Our strongest potential lever is our weakest in execution.

### D. Interview / Reader-B register leaks into the beginner voice
Day 2 c9 hands the reader a scripted **"here's how you'd explain it in an interview"** monologue with matrix notation `(x·W₁)·W₂ = x·(W₁·W₂)`, plus He/Xavier/GELU init and a trade-off box — **Reader B material in a Reader A lesson**, which violates `CLAUDE.md §1` (never mix the two readers). Day 6 drops "non-convex," "representational ceiling." Phrases like *"that's a real interview beat, and you have it"* appear in a lesson aimed at a 12-year-old.
**Root cause:** the builder's three-layer architecture bakes **"interview answer"** into every lesson's Staff Depth Layer, and the instructions carry an **unreconciled dual mandate** ("keep interview depth" + "write for a 12-year-old"), so authors *append* interview register instead of removing it.

### E. (Engine) Interest is only enforced where a notebook exists — the minority
The interest + tone judges are **notebook-relative**: they return `N/A` when `notebook_yardstick` is null, and the interest lens in `lesson_build.js` only P0s on `BELOW_NOTEBOOK`/`WORSE` — so an `N/A` **passes vacuously**. **35 of 46 built days (76%) are null** (m02 Days 7–9, all of m07/JAX, m03, m05, m06, scaling, …). On those days the #1 objective is unenforced. This was a deliberate cost choice (`notebook_yardstick:null keeps tone/interest N/A … under the $300/day cap`). And even where a notebook exists, the judge only asks *"at least as good as the notebook"* — **no absolute floor**, so a mediocre baseline can pass a merely-adequate lesson.

### F. (Engine) Density/length is measured by NO deterministic gate
All concept-count checks are floors (`≥3`). The only density signal is the interest judge's `momentum` lever — which lives inside the notebook-relative judge, so it's **dead on the 76% of days most at risk of overload** (JAX, scaling). Relatedly, **relevance/breadth thin out after the opening** (ChatGPT appears once in the hook, then never returns; single-corridor lessons, no "family buffet" tease) — `relevance`/`breadth_spark` WEAK on 5–6 days.

---

## Proposed changes (skills + engine)

Ordered by leverage. Efforts: S ≈ hours, M ≈ half-day, L ≈ multi-day.

| # | Target | Change | Why | Effort |
|---|---|---|---|---|
| **1** | `coverage_judge.py` (new `judge_interest_absolute`) + `lesson_build.js` interest lens | Add a **notebook-free absolute interest judge** (7 levers scored on their own merits → `FLOOR_MET`/`BELOW_FLOOR`), run **unconditionally**; P0 the loop on `BELOW_FLOOR` regardless of notebook. Keep the notebook-relative judge as an extra ceiling where a yardstick exists. Make an `N/A` interest result surface as an explicit "unenforced" finding, not a silent pass. | Closes the **76%-of-days** enforcement hole and removes the mediocre-notebook ceiling — the single highest-leverage fix for the stated #1 goal. | M |
| **2** | `concept_structure_gate.py` (deterministic) + `frontier-curriculum-architect` Coverage Spec Rule + `AUTHORING.md` | **Bite-size + density gate:** warn/soft-gate above a **concept-count ceiling (~7–8)** or word budget; detect a **failure-mode cluster** (N consecutive failure/limit units with no play/payoff between). Add a Coverage-Spec-Rule **counterweight**: cap consecutive failure units (≤2), allow **compressing** related failures into one comparison table, and permit **deferring** secondary remedies to a later day (completeness at *module* granularity, not per-day). | Directly attacks Findings A + B — the two biggest momentum killers — and gives "bite-sized" real enforcement instead of a floor-only contract. | M |
| **3** | `frontier-lesson-builder` (Beginner Intuition Register / Staff-Depth rule) | **Reader-separation rule for foundation lessons:** ban scripted interview answers, "interview beat" phrasing, and Reader-B notation in the lesson body; state that staff/interview depth lives in a *separate* file. | Fixes Finding D and the `CLAUDE.md §1` violation; resolves the dual-mandate leak at the source. | S |
| **4** | `frontier-lesson-builder` (Non-negotiables / Learning Barrier Gate) | Promote 1–2 **interest levers to named P0/P1 checks in the skill itself** (e.g. "concept opens with aspiration/relevance, not a problem/warning"; "lesson teases breadth/payoff"). Today the skill's P0s cover only warmth/structure (analogy/visual/jargon/recap/math) and frame interest as a *byproduct of warmth*. | Makes interest a first-class *authored* objective independent of the engine/notebook — aligns the skill with the #1 goal. | S |
| **5** | `frontier-visual-evidence-builder` + `%%% demo` widget (compiler) | Make "play" **real or honest:** require ≥N genuinely interactive widgets per lesson, front-loaded (not one, late); either make `%%% demo` editable (type an input → recompute in-browser) or relabel "run it ▶" so it doesn't oversell a canned reveal. | Attacks Finding C — the interest directive's #1 lever, currently our weakest execution. | M–L |
| **6** | `frontier-refactor-qa` SKILL.md | Document the new contract: interest has **two enforcers** (absolute floor always on + notebook ceiling when present); `notebook_yardstick: null` **no longer means "interest unenforced"**; add the density/failure-cluster floor to the V9 Concept Gates list. | Keeps the human-facing QA contract truthful so future rollouts don't ship ungated no-notebook days believing they were checked. | S |

**Content remediation for Module 2 itself** (separate from the skill/engine work): trim each day toward bite-size; break/gate the late failure-mode walls (compress or defer secondary traps); add a live widget *earlier*; strip the Day 2 interview monologue; add mid-lesson relevance/breadth callbacks. Highest-value single edits per day are in the workflow's `single_most_important_fix` fields.

---

## Recommendation

**Pass with P1.** Module 2 is warm, correct, and genuinely inviting at the opening and per-concept level — the interest-first rebuild succeeded. Do **not** rewrite it.

The leverage is in the **engine + skills**, not in re-polishing m02: changes #1–#4 fix the *causes* of A/B/D/E/F for **all 46 lessons** (most of which currently build with interest unenforced), and are the right thing to do before more no-notebook modules ship. Recommended order: **land #1 + #2 (the enforcement floor + density gate) first**, then #3/#4/#6, then #5, then re-run Module 2's 9 days through the improved gates to bank the content fixes with the new floor active.
