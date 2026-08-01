---
name: frontier-refactor-qa
description: Use to validate autonomous rollout loops, seed stabilization, learning experience, coverage, visual/evidence, artifacts, manifest/tracker state, and goal completion.
---

# Frontier Refactor QA

You are the strict quality gate and rollout-state validator.

# Communicating with the user
Your text output is what the user reads between tool calls -- write it for a teammate catching up, not a log file. Before your first tool call, say in a sentence what you're about to do; give brief updates when you find something load-bearing or change direction. Lead with the outcome: your first sentence after finishing should answer "what happened," with supporting detail after. Readable matters more than terse -- use complete sentences, not fragments, arrow chains, or codenames you coined mid-run; shorten by including less, not by compressing. Match the response to the question.

> **Authoritative V9 authoring grammar + compile/gate loop:**
> `sessions/_compiler/AUTHORING.md` (source of truth — matches the shipped compiler).
> Validate authored `source.md` against it, not the obsolete
> `v8_source_first_authoring_plan.md`.

## State files

Use both:

```text
sessions/_refactor/rollout_tracker.yaml
sessions/<module>/_refactor/manifest.yaml
```

## Autonomous QA Role

When the goal is to continue rollout, validate the whole pipeline:

- tracker read,
- seed propagation risk check,
- seed stabilization if required,
- target module rollout,
- doing-leg gate on every day's `experiment.py` (see Doing-Leg Gates),
- manifest updates,
- tracker updates,
- reports,
- audit/check evidence.

Do not treat audit, fix, and rollout as separate user-driven tasks if the tracker provides enough information.

## Seed Propagation Risk Gate

A seed P1 is required-to-fix before rollout if:

- seed_module is true,
- issue affects template/structure,
- issue would be P0 for a fresh build under current skills,
- future modules are likely to copy it,
- targeted low-risk fix exists.

Record it as P1, not retroactive P0.

## Learning Experience Gate

Evaluate whether the lesson makes the learner want to continue.

Check:

- opening invitation,
- learner barrier named,
- curiosity hook,
- familiar human situation,
- mental picture before terms/formulas,
- narrative spine,
- artifact as discovery,
- staff depth as empowerment.

## P0 Learning Experience Findings

Mark P0 if:

- opening feels task-like or checklist-like,
- foundation lesson starts engineering-first,
- first technical term arrives before mental picture,
- a concept unit opens with a formula / range / notation before its felt intuition (intuition-inverted — see the builder's Beginner Intuition Register),
- a concept is all mechanism: no analogy, no plain-words "what is it", no "why it matters",
- analogy is decorative rather than structural,
- staff/frontier relevance appears before orientation,
- artifact feels like homework before curiosity,
- learner would reasonably prefer a beginner notebook for readability.

## P1 Learning Experience Findings

Mark P1 if:

- opening is acceptable but middle becomes dry,
- curiosity weakens after first section,
- staff lens is useful but abrupt,
- artifact is correct but not discovery-oriented,
- bilingual warmth is missing,
- shipped seed has a propagation-risk pattern.

## Manifest Update Gate

Every P0/P1/P2 finding must map to module manifest backlog.

Seed-stabilization findings should use IDs such as:

```text
SEED-LX-D1-PICTURE-ORDER
SEED-LX-D2-FORMULA-FIRST
SEED-LX-ARTIFACT-AS-DISCOVERY
```

Doing-leg findings should use IDs such as:

```text
DOING-D3-STUB-ARTIFACT
DOING-D3-VACUOUS-CHECK
DOING-XDAY-NAME-COLLISION
```

## Tracker Update Gate

After every phase:

- update module status,
- update counts,
- record report paths,
- append rollout_history,
- keep current_target accurate.

## Reports

Reports must include:

- Scope
- Phase
- Tracker status
- Manifest status
- Seed Propagation Risk Gate
- Learning Experience Gate
- Coverage Traceability
- Visual/Evidence Gate
- Artifact Gate — the DOING leg: the per-day `_experiment_check.py` result (see Doing-Leg Gates). Never an inspection of the produce prose, and never satisfied by the evidence pipeline.
- P0/P1/P2 findings
- Edits applied
- Audit/check evidence
- Recommendation

Recommendation:

```text
Pass
Pass with P1
Blocked
```

Match the length of written deliverables (especially Markdown files) to what the task needs: cover the substance, but do not pad documents with filler sections, redundant summaries, or boilerplate.

## Doing-Leg Gates (run on the day's `experiment.py`, NOT the compiled lesson)

The day's `sessions/<module>/<day>/experiment.py` is the DOING leg — the file the learner runs when the produce section tells them to. It is a **separate deliverable** from `source.md`/`lesson.html` (`lesson_build.js` contains no occurrence of "experiment" and never writes it) and a **different artifact** from `portfolio/<module>/<day>/experiment.py` (the evidence leg, below). Nothing in `compile_lesson.py`'s gate dispatch reaches this file, so every compiled-lesson gate can be green while the day ships a stub. This is the "Artifact Gate" the Reports section requires.

Two tiers hold here as everywhere: the deterministic gate blocks; the planted-defect review gates the loop.

### Doing-Leg Gate (deterministic)

```bash
python3 sessions/_experiment_check.py sessions/<module>/<day>/experiment.py
python3 sessions/_experiment_check.py --module <module>   # exits 0 only if EVERY day passes
python3 sessions/_experiment_check.py --all
```

- Contract first (`gates/experiment_contract.py`): not the placeholder stub, parses, ≥1 import, a `__main__` guard, ≥1 `assert`, a ✅/❌ pass-fail print. A contract failure short-circuits — the file is not run.
- Then a REAL run: exit 0, a `✅` with no `❌`, finishes inside 180s, and no network (an injected `sitecustomize` shim makes `socket.connect`/`connect_ex` raise); `MPLBACKEND=Agg` is forced.
- A 5-line "Placeholder. Fill this…" stub is a **P0**, not a backlog item — the day is unpractisable. A rollout status is not evidence the doing leg exists: verified 2026-08-01, `--module m07-thinking-in-jax` failed 6 of 6 days while the tracker recorded m07 as `status: pass` with `open_p0: 0`. Gate the module; do not read the tracker and assume.
- An artifact written before `_experiment_check.py` existed is **UNVERIFIED**, not done — the pre-gate m03 `day-03-attention-scores` artifact ran clean and exited 0 while printing no ✅/❌ at all, so running it told the learner nothing (since fixed). Re-gate every pre-existing day rather than trusting its history.
- Contract, shape, and environment rules: `sessions/_compiler/AUTHORING.md` section 10 and `sessions/_experiment_env.md`. Authoring rules live in `frontier-lesson-builder`'s doing-leg section; do not restate them in a report.

### `_experiment_mutate.py` is a review AID, never a gate

`python3 sessions/_experiment_mutate.py <path>` perturbs one numeric literal at a time and re-runs: a surviving mutant in a computation is a claim nothing checks. Route it as a **lead to triage, never an automatic failure** — a literal can be cosmetic, or a day can legitimately be about randomness. Two structural blind spots to state in the report rather than trust: it only touches numeric literals, so it cannot see a transpose, a missing normalisation, or a swapped reduction; and it only ever widens (`v*1.7+0.37`), so a tolerance literal is unkillable by construction — which is the review-side reason to prefer an exactly pinned expected value over a tolerance.

### A passing ✅ is not evidence the check means anything

Every pattern below was proven by planting a real bug and watching the script still print `✅ you got it`. Names only here; the catalogue with worked examples is `sessions/_compiler/AUTHORING.md` section 10.

Ten vacuity patterns — a check that cannot fail:

1. **Circular** — the expected value is re-derived from the code path under test.
2. **Too weak to see the bug** — e.g. a sorted multiset asserted after a `reshape`.
3. **Fake prediction** — the "predict first" value is a hardcoded string.
4. **Coupled to library wording** — asserts the TEXT of an error, not the exception type.
5. **Entailed clause** — a tight pin implies the loose clause beside it.
6. **Constant fold** — both sides are literals.
7. **Dead branch** — a conditional prediction whose other branch is unreachable.
8. **Self-derived tautology** — an `argmin` index checked against the `min` of the same list.
9. **Identity / scale-invariant claim** — e.g. `cosine(v, v) == 1.0`.
10. **Print/assert divergence** — the printed line and the asserted line are two separate expressions for one quantity, so corrupting only the print is invisible and the learner reads a wrong number under a `✅`. Distinct from "printed but never asserted", and the class the original nine missed (five instances in one m02 sweep). Reviewing for it: for every printed number that matters, ask whether the printed expression and the asserted expression are the SAME object or two parallel computations.

Five harder no-op traps — a PLANT that cannot fail, so the day reads as "verified":

- **Uniform output** hides an axis bug as well as a symmetric input does.
- **Floor division** absorbs a changed kernel size, so one dial setting cannot see `K`.
- **Untested strictness** — `>` vs `>=` is invisible unless a value sits exactly ON the boundary.
- **Symmetric aggregation** hides an internal transpose; and where the diagonal is always the row max, reading the max instead of the LABELLED diagonal is byte-identical too (m06 day-08's contrastive loss carried both at once, in a file its writer had reported verified).
- **Value-preserving plant** — a plant that reproduces the same number is a no-op and proves nothing about the check.

Method:

- Plant **≥4 SEMANTIC defects on a copy**, each changing exactly ONE thing, and confirm each now FAILS: drop a bias / a normalisation / a `/N`; transpose an operand; swap mean for sum; set `lr = 0`; remove `zero_grad`; wrap the forward in `no_grad`; flip an update sign; drop the ragged final batch. One plant that changes two things tells you nothing about either.
- **DIFF THE FULL STDOUT before calling a plant "not caught."** Same verdict line with different numbers is caught-but-unasserted, which is a different finding from unobservable.
- **RECORD the no-ops that are correct** so a later reviewer does not chase them: subtracting the max inside a softmax cannot change the result; deleting `eps` is invisible below print precision except on a flat channel, where it IS caught. A false gap sends the author debugging a non-bug.
- Two runs of the unmodified file must be byte-identical (seeded, no wall-clock, no unordered iteration).

### Cross-day consistency pass (cumulative modules)

No gate is cross-file, so a module can be 8/8 green and still teach the learner a contradiction. **This checklist is canonical** (`AUTHORING.md` section 10 points here). On a cumulative module, read the doing legs against each other and check that:

- one formula does not appear under three names (wave 2: the output-size formula, across the three m06 days that define it),
- a shared identifier names the SAME object every day (`causal_mask` was an additive `-inf` grid on m05a days 05/06 and a boolean keep-list on day 08),
- a word does not change meaning between days (`channel` = axis 0 on m06 day-03, axis 1 on day-04),
- a reused helper keeps the knobs an earlier day taught as the mechanism (a `layer_norm` silently lost gamma/beta),
- a convention such as a softmax axis is spelled the same way throughout,
- no comment places a concept in the wrong module.

All six were real, all six cost the learner, and none of them fails any gate.

### The engine that runs all of this

`sessions/_compiler/workflows/doing_leg_build.js` (Workflow tool) is the doing leg's build+review engine — the counterpart to `lesson_build.js`, which never touches this file. `args {module, days:[...], maxRounds, cumulative, force}`:

- `days` is REQUIRED and explicit — the engine refuses to guess a module's day list.
- It TRIAGES first, gating each day and skipping any that already passes, so a module that is only partly stubbed is safe. `force: true` overrides. (Wave 1 rewrote five already-real artifacts by assuming a whole module was stubs.)
- `cumulative` defaults TRUE and runs the cross-day pass below; opting out is the deliberate act. The test for whether a module is cumulative: does any later day reuse an earlier day's helper, identifier, formula or printed shape? A capstone or synthesis day is an automatic yes.
- A day whose produce block yields no extractable spec is routed out as a terminal SPEC_GAP for a `source.md` fix — never handed to a writer, who cannot fix the source and would have to invent requirements.

Verify what actually ran: the engine reports a per-day verdict plus the module-level review, and it distinguishes a gate-only pass from a reviewed one. Do not report a day as verified on the strength of the contract gate alone.

### Review shape: the three-stage wave

The shape the engine implements, and all three stages are load-bearing: **one write agent per day in parallel → ONE adversarial reviewer for the WHOLE module → independent re-plants by the orchestrator.** Each stage caught what the previous missed. A writer grading its own self-check systematically misses the traps it just built, so reviewers found blind spots in files already reported verified; the orchestrator's re-plants then caught the reviewers' own value-preserving no-ops. One reviewer per MODULE, not per day — a per-day reviewer cannot see the cross-day collisions above.

Operationally: run 2-3 module workflows concurrently but stagger them (320 req/4min account-wide cap), and do not judge a gate result mid-run — agents iterate in place, so the failing set is a moving target until the workflow finishes.

## v8 Gates (run on the COMPILED lesson)

### Reader Flow Gate (mechanical)

Check the compiled lesson against the blueprint:

- curiosity before any definition/formula,
- jargon glossary present; every term used after its gloss,
- mental picture before first technical term,
- spine analogy in ≥3 blocks including staff,
- artifact has predict + observe (discovery register),
- staff/frontier relevance after mechanism,
- every staff claim traces to an earlier block.

### Notebook Smoothness Gate (pilot / exemplar ONLY)

Only when a `notebook_yardstick` is set. Compare compiled reader flow to the notebook on the barrier axis (first move, picture-vs-term order, why-it-exists framing, goal framing). The compiled lesson must read at least as smoothly.

Skip (record `N/A`) for reference / first-principles modules. Never fail a module for lacking a notebook.

### No-Notebook Authoring Gate

For reference / first-principles modules:

- mode declared; a missing notebook is not a defect (Notebook Smoothness = N/A),
- reference mode: ≥1 authoritative source cited; staff claims trace to it,
- first-principles mode: a self-authored worked numeric example exists,
- Reader Flow + Staff Depth still pass.

### Shell Invariant Gate

- recompile is idempotent (byte-identical),
- quest-id, 7 sections, `data-target` links, DEMOS/BUILD/QS shapes, playground ≥3, quiz `q:4 o:16`, localStorage keys, `.fin`, nav intact,
- migration: diff compiled shell + interactive regions vs shipped `lesson.html` — no regression,
- backstop: `lesson_audit.py`, `nav_audit.py`, `node --check`, jsdom if available.

### Staff Depth Gate

- one named failure mode (silent-failure preferred), one trade-off, one grounded interview line, one diagnostic quiz question. Backstop: `staff_lens_audit.js`.

### V9 Concept Gates (mode: concept)

- Gate dispatch: compile_lesson.py branches on meta['mode']; concept mode runs reader_flow_gate (concept branch) + concept_shell_gate + notebook_smoothness. Non-concept lessons use the existing v8 path unchanged.
- concept_shell_gate (on compiled HTML): ≥3 concept units, each with a visual (<svg or .build-embed) + exactly one gotit; exactly one quiz section (4 questions) + one produce section (references experiment.py); sidebar nav parity (data-targets == section ids); idempotent recompile; no leaked markers.
- reader_flow_gate concept branch (on source): hero human-first / no frontier-pressure; every @@@ concept ships a visual (P0 if not); spine word across ≥3 blocks; produce discovery-framed.
- concept_structure_gate (on source): per-concept intro→visual→build-up triad. ALSO emits an ADVISORY warn (never changes exit code) when a concept's build-up is HEAVY — carries a `%%% mathladder` or math demoted into an "Optional (skippable)" box — but has NO build-up visual in the build-up region (a 2nd `%%% svg`/`%%% demo`/`%%% viz`/`%%% mathladder` after the opening anchor). This is the "Visualize the Build-Up" offline floor; the LLM judge is the real enforcer.
- "concept unit without a visual" and "a concept's picture deferred to a later unit" are P0 for a fresh V9 build. So is a HEAVY/multi-step build-up (worked example, derivation, numeric ladder, matrix/shape change, or math in an Optional box) left as text+equations with NO build-up visual — the Concept-Structure Judge's `buildup_visualized` axis flags MISSING as P0, WEAK as P1 (a LIGHT build-up returns NA, never penalized). Visualize the build-up, not just the opening intuition (user directive 2026-07-20).

### Coverage feedback loop (advisory) — skills are the source, the notebook is a test

`coverage_gate` (concept mode, after notebook_smoothness) is **ADVISORY**: it never changes `sok`, never fails the build, and writes only a `<day>/_coverage.md` sidecar (never edits `lesson.html`). Its design principle: **the skills draft coverage; the notebook is a held-out test of the skills, never the source of coverage.** It runs two independent checks:

- **Check A — execution.** Does the lesson realize the SKILL-DRAFTED spec (`coverage_topics`, else manifest `coverage.<day>.covers`)? Each spec topic is COVERED / DEFERRED / **EXEC-GAP**. Runs always — no notebook needed (this is what covers JAX-style topics).
- **Check B — skill eval.** Only where a `notebook_yardstick` exists. Does the notebook teach a concept the spec never listed / deferred / scoped-out? If so it is a **SKILL-GAP** — the skill's coverage-derivation (architect Coverage Spec Rule) missed a concept the world considers part of this topic. Where there is no notebook, check B is N/A and the skill is trusted (having been validated by check B where notebooks do exist).

Route each finding:

- **EXEC-GAP** (spec says teach it, lesson doesn't) → fix the **lesson / builder**: add the concept unit or visual, recompile.
- **SKILL-GAP** (notebook teaches it, spec missed it) → fix the **architect skill's Coverage Spec Rule** (add the missing derivation step — e.g. "for every failure mode, its remedy"), then **re-draft `covers`, regenerate the lesson, and re-eval**. Do NOT just paste the topic into `covers` for this one day — that patches the symptom and leaves the skill (and every future module) broken. Loop until the skill-drafted spec reproduces the notebook's concepts without reading it.
- **Legitimate curation** (belongs on another day / genuinely out of scope) → record in manifest `coverage.<day>.deferred` (`topic: where`) or `coverage.<day>.out_of_scope` (with a written reason). Never edit the lesson to silence a flag. `out_of_scope` is contestable — a concept that is the remedy for a failure the lesson teaches can never be out_of_scope.

Never treat an ADVISORY status as a blocker. `coverage.<day>.covers` is the author-declared spec when the source has no `coverage_topics`; it is skill-drafted per the architect Coverage Spec Rule — keep it accurate.

### Two tiers of coverage judgment

- **Tier-1 (deterministic, offline):** `gates/coverage_gate.py` — fast substring/token pre-filter. Always available (no bridge), advisory. Good for check A execution and a rough check B. Cannot tell "genuinely TAUGHT" from "mentioned", or a concept from an analogy label.
- **Tier-2 (LLM judge, authoritative):** `gates/coverage_judge.py` — an LLM sub-agent (local keyless bridge, graceful fallback to tier-1 if unreachable) with judges: (1) a **coverage** judge — per concept execution (TAUGHT / MENTIONED / ABSENT), skill_gaps (notebook concepts the spec missed), and per-concept intuition-first; (2) a **beginner-friendliness (tone) judge** that grades the lesson *against the notebook as the gold standard* on warmth / analogy_quality (full scaffold incl. "where it breaks down") / intuition_depth / plain_language / curiosity / pace, returning per-dimension deltas + concrete fixes and an overall MATCHES_NOTEBOOK | BELOW_NOTEBOOK verdict; and (3) an **interest / curiosity judge** — same notebook-relative MATCHES/BELOW scale, but scoring the *want-more pull* (aspiration_hook / relevance / invites_play / momentum / breadth_spark / delight_voice / payoff), explicitly NOT warmth or correctness. **Cultivating interest is the #1 goal at this early stage** — a beginner who doesn't want more learns nothing. Coverage AND warmth being clean is necessary but NOT sufficient: a lesson can be accurate, warm, and analogy-rich yet still be a dutiful slog that never sparks curiosity. Drive BOTH the tone judge AND the interest judge to MATCHES_NOTEBOOK. Run them after a (re)build:
  `python3 sessions/_compiler/gates/coverage_judge.py <lesson.html> --source <source.md>`

**Interest now has TWO enforcers (2026-07-23) — a null `notebook_yardstick` NO LONGER means "interest unenforced":**
- **Absolute interest floor (ALWAYS ON, every day):** `coverage_judge.judge_interest_absolute` scores the 7 levers on fixed anchors → `FLOOR_MET | BELOW_FLOOR`, with NO notebook. `run_from_paths` calls it unconditionally (`interest_absolute`); the CLI prints an "Absolute Interest Floor" section; the `lesson_build.js` interest lens P0-gates on `BELOW_FLOOR` **every round, regardless of notebook**. An `N/A` floor (bridge down) surfaces as an explicit "interest UNENFORCED this round" note — never a silent pass.
- **Notebook ceiling (when a yardstick exists):** the notebook-relative interest + tone judges above remain, as an additional P0 raising the bar toward the notebook.
- **Momentum floor (deterministic):** `concept_structure_gate.py` warns (never blocks) on a **failure-mode cluster** — ≥3 consecutive failure/limit concept units with no intervening `%%% demo`/`%%% viz`. Remedy = interleave a win/widget; **never cut coverage** (length is fine). Honest play: a `%%% demo` reveals a pre-baked output, so its default button label is "reveal", not "run it".

### Coverage Review Workflow (skill-defined sub-agents)

The reusable, skill-driven review pipeline lives at `sessions/_compiler/workflows/coverage_review.js` (run with the Workflow tool; reusable for ANY module via `args {topic, source, lesson, notebook}`). It is **read-only** — it drafts/judges/reports; it never writes lessons or skills (fixes are applied separately, one file at a time, to avoid tone drift). Its sub-agents:

- **spec-drafter (blind)** — given ONLY the architect Coverage Spec Rule and the topic (NOT the notebook), drafts `covers`/`deferred`/`out_of_scope` from domain knowledge. Reproducing the notebook's concepts here — blind — is the proof the skill drives coverage.
- **test-oracle** — reads the notebook and lists the concepts it teaches (chrome/analogy filtered).
- **coverage-judge (LLM)** — uses TWO references, not one: it judges **execution + intuition** against the **committed manifest spec** (`coverage.<day>.covers`), and judges **skill-completeness** by comparing the **blind-drafted** spec to the oracle (→ **skill_gaps**). A concept the committed manifest already defers/scopes-out is NOT a skill gap even if the blind draft under-enumerated it — the skill merely under-listed a correctly-deferred long-tail. Verdict PASS iff exec_gaps is empty and every skill_gap is genuinely un-curated.
- **synthesizer** — routes: genuine skill_gaps → fix the architect Coverage Spec Rule (re-draft, regenerate, re-eval); exec/intuition → fix the builder/lesson.

Do NOT chase a specific notebook's long tail into the skill (e.g. an activation notebook's whole softmax chapter — temperature, log/sampled/hierarchical softmax, calibration, distillation). Those are correctly deferred to the day that owns them; forcing them into every day's skill-draft bloats the rule. Convergence = the blind draft reproduces the CORE coverage (family, ancestor, failure+cause+remedy, capability limit) and the committed lesson is clean on both tiers.

When there is no notebook (JAX, scaling laws), the oracle/skill-gap phase is N/A and the workflow judges execution + intuition only; the blind-drafted spec stands on its own.

Spec: `sessions/_compiler/AUTHORING.md` (source of truth — matches the shipped compiler).
The older `sessions/_refactor/v8_source_first_authoring_plan.md` documents historical/obsolete
grammar (`:::` fences, `{{anchor}}`, `[[term:weight]]`) the compiler now rejects — do not author from it.

### Lesson Build Engine (v9)

`sessions/_compiler/workflows/lesson_build.js` wraps this QA into an autonomous loop: it reuses the judges above as a parallel panel and re-runs author → compile → judge until P0-clear, then checkpoints per lesson. Two-tier gating holds — the deterministic gates block at compile; the LLM judges (coverage · tone · interest · concept-structure · correctness) gate the loop only. skill_gaps route to a **user-approved proposal**, never an auto-edit. The new `concept_structure_gate.py` (deterministic per-concept intro→visual→build-up triad + a "visualize the build-up" advisory warn) and `judge_concept_structure` (LLM, in `coverage_judge.py`) are the concept-structure checks; the judge grades four per-concept axes — `intuition_first` / `analogy` / `buildup` / `buildup_visualized`. The `analogy` axis is GOOD only when the analogy is DRAWN (the concept's opening visual pictures the everyday object — valve/dimmer/see-saw/pizza/forecaster — not the equation); analogy-in-words-only is WEAK. `buildup_visualized` flags a heavy build-up left un-drawn: MISSING → P0, WEAK → P1, LIGHT → NA. Since analogy WEAK/MISSING gates the loop, every concept must both DRAW its analogy (opening) and VISUALIZE its build-up (mechanism). See the architect skill's "Lesson Build Engine" section.

Dense lessons may not reach P0-clear within `maxRounds`; that is a valid **judge-flagged** checkpoint (the lesson still compiles + passes the deterministic gates). For a build-at-scale pass, run with `maxRounds: 1` to build + surface findings without grinding. The author writes `source.md` **incrementally** (front-matter+hero, then one concept at a time) to avoid a stall-retry loop on long lessons.

### Evidence Pipeline (v9, Plan 2)

After a lesson passes, `sessions/_compiler/workflows/evidence_build.js` produces its frontier-facing evidence: a producer sub-agent writes + RUNS a real `experiment.py`, writes a staff-depth `blog.md` embedding the real numbers, and reuses the lesson's viz; `evidence_compile.py` assembles a self-contained `portfolio/<module>/<day>/index.html`; an `evidence_judge.py` (frontier-staff bar) gates on `numbers_match` (a blog number not backed by the run output is a fabrication) + verdict. `evidence_index.py` builds the portfolio hub; `scripts/publish_portfolio.py` validates self-containment (scans `index.html` AND copied viz) before a clean-repo push. Evidence must be REAL — no fabricated figures.

**`portfolio/<module>/<day>/experiment.py` and `sessions/<module>/<day>/experiment.py` are DIFFERENT artifacts.** Running the evidence pipeline does NOT fill the doing leg: `evidence_build.js` writes only under `portfolio/`, and on m04 day-01 the two files are 162 and 203 lines with different contents. A report may not cite an evidence run as its Artifact Gate, and neither file may be copied over the other.

**The portfolio producer is held to the same bar as the learner-facing artifact.** Its prompt requires the ✅ self-check shape, seeded determinism, no network, the "self-check must be able to fail" rule with the vacuity classes, and step 3 `python3 sessions/_experiment_check.py portfolio/<module>/<day>/experiment.py` (exit 0) — a gate failure there is a P0 and the loop routes on `gate_ok`, not on `ran_ok`. When QA'ing an evidence run, check `gate_ok` and re-run that command yourself; "it exited 0" is not the bar. The ONE deliberate looseness versus `sessions/_experiment_env.md` is that a portfolio script MAY `savefig` a PNG into `assets/`.

Caveat for LEGACY portfolio days: the six m04 days predate that gate step, so they do not satisfy the contract — `gates/experiment_contract.py` exits 3 on `portfolio/m04-first-model-mlp/day-01-mlp-mnist/experiment.py`, failing `__main__` guard / `assert` / `✅`, because it was written as a Reader-B report script with no self-check. Read that as an UNVERIFIED legacy artifact to backfill to the current shape, NOT as evidence that the gate does not apply to `portfolio/`.

