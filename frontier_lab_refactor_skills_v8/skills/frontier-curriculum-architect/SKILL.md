---
name: frontier-curriculum-architect
description: Use to autonomously continue Frontier Lab curriculum rollout from the rollout tracker, including seed stabilization, module manifest creation, coverage discovery, learning-experience refactor, QA loops, and tracker updates.
---

# Frontier Curriculum Architect

You design and orchestrate the curriculum rollout system.

The user should not be the manual orchestrator.

## State files

Repo-level tracker:

```text
sessions/_refactor/rollout_tracker.yaml
```

Module manifest:

```text
sessions/<module>/_refactor/manifest.yaml
```

## Autonomous Rollout Loop Rule

If the user says:

```text
Continue rollout from the tracker.
```

or:

```text
Roll out the next module.
```

or names a module to roll out, do not ask for separate audit/fix/QA prompts when the tracker and reports determine the next phase.

Run this loop:

```text
1. Read rollout tracker
2. Read seed module manifest(s)
3. Read recent reports referenced by tracker/manifests
4. Detect seed propagation risks
5. Stabilize seed if required
6. Update seed manifest/tracker
7. Select current_target or next eligible module
8. Create/read target manifest
9. Run source-free coverage and learning-experience discovery
10. Refactor P0 only, unless doing explicitly required seed stabilization
11. QA
12. Update target manifest/tracker
13. Stop at pass/pass_with_p1 or write blocker report
```

## No Human Orchestrator Rule

Do not return a sequence of manual prompts when you can execute the next phase.

The user wants a system, not a prompt babysitting job.

Ask a clarification only if:

- tracker is missing and cannot be created,
- target module cannot be inferred,
- multiple current targets conflict,
- the requested edit violates constraints.

## Rollout Tracker Rule

Maintain:

```text
sessions/_refactor/rollout_tracker.yaml
```

The tracker records current target, module order, statuses, manifest paths, constraints, success gates, report paths, open P0/P1/P2 counts, skill version, rollout history, and skill history.

It also records the repo-level **doing-leg count** in `summary.doing_leg`: real days / total days and the modules still holding stubs. Read it there rather than restating it here — a count copied into a skill goes stale the moment a wave lands. Regenerate it with `python3 sessions/_experiment_check.py --all`. A module recorded `pass` may still ship stub doing legs: m07-thinking-in-jax is `pass` with `open_p0: 0` and 6 of 6 days stubbed. When asking whether a module is practisable, read the count, not the status.

**`sanctioned_designs.guided_stub_experiment_py` was DELETED from the tracker on 2026-08-01 — do not reinstate it.** It had said the per-day `experiment.py` is "a deliberate ~5-line guided stub … Not a defect". Step 1 of the rollout loop is *read the tracker*, so that entry was the first authoritative thing a fresh session saw on this axis, and it pointed the wrong way: no amount of skill guidance survives a state file that says the deliverable is not a deliverable. A stub `experiment.py` is a P0 (`sessions/_compiler/AUTHORING.md` §10). The matching `experiment.py guided stubs (untouched)` lines were also dropped from the m02/m06/m07 manifests, where they froze the stub as a passing invariant. If you find another copy in a module manifest, delete it.

## Module Refactor Manifest Rule

Every module maintains:

```text
sessions/<module>/_refactor/manifest.yaml
```

The manifest records constraints, quality gates, coverage items, visual/evidence items, artifact items, backlog, held-out eval findings, skill-gap candidates, and loop history.

`artifact_items` must record the **doing-leg state per day** — real vs stub, gate result, date — not only `design` and `run_paths`. That the artifact exists is a different claim from that it runs. Mirror the module's real/total count into the tracker `summary` so the backfill is resumable from state files rather than from a session's memory.

Count stubs; never estimate them:

```bash
python3 -c "import glob;print(sum('Placeholder' in open(f,errors='replace').read() for f in glob.glob('sessions/**/experiment.py',recursive=True)))"
```

VERIFY which days are actually stubs before choosing a backfill wave — one module was re-written by five agents because its days were already real. And never write a test that asserts a stub COUNT: such a test fails the moment the work succeeds.

## Coverage Spec Rule (the skill drafts coverage; the notebook is a test)

`coverage.<day>.covers` is **skill-drafted from domain knowledge**, before and independent of any notebook. A notebook may not exist (JAX, scaling laws); the skill must still produce complete coverage. Never derive `covers` by reading a notebook — that inverts the system and lets a notebook self-certify.

To draft `covers` for a topic, enumerate — from what the topic *is*, not from a reference:

```text
1. Core mechanism family     the primary objects the lesson teaches, AND each one's
                             salient first-order properties the lesson leans on
                             (e.g. the activations ReLU / sigmoid / tanh; ReLU's sparsity)
2. Historical ancestor       the cruder earlier form, WHEN it motivates the modern one
                             (step function → why we moved to smooth / valve bends)
3. Failure → CAUSE + REMEDY  for EVERY failure you teach, name BOTH what triggers it
                             AND its fix. Symptom without cause is an incomplete spec.
                             When the fix is a NAMED technique (not just a knob), the
                             technique itself becomes its own covers topic — a prose
                             remedy phrase ("sensible init") does not count as covering
                             the named method it stands for. Enumerate the method the
                             beginner will actually hear.
                             dead ReLU (cause: too-large LR shoves the unit permanently
                             negative) → He init (the sensible default for ReLU nets) /
                             Leaky ReLU / GELU; vanishing gradient (cause:
                             saturating slopes multiply toward 0) → ReLU / normalization.
4. Capability limit          what the mechanism CANNOT do
                             (a single neuron → XOR / linear separability)
5. Forward-pointer           named-but-deferred concepts → coverage.<day>.deferred:
                             {topic: where} (e.g. softmax → day-04-loss). Enumerate the
                             full family a beginner will hear (e.g. output-layer choices:
                             softmax multi-class, sigmoid binary, sigmoid-per-label
                             multi-label, linear/none for regression), don't stop at one.
6. Out of scope              variants deliberately excluded → coverage.<day>.out_of_scope
                             WITH a written reason. A concept that is the REMEDY for a
                             failure you teach can NEVER be out_of_scope.
```

`out_of_scope` is a contestable claim, not a silencer — record the reason so a reviewer (or the user) can challenge it. When challenged and wrong, the topic is a gap to fix, not an exclusion to defend.

Where a notebook exists, the coverage gate's **check B** uses it as a **held-out test** of this spec: a notebook concept absent from `covers` / `deferred` / `out_of_scope` is a **SKILL GAP** — the enumeration above missed a step. The fix is to improve THIS rule (add the missing derivation step, e.g. "failure → remedy"), re-draft `covers`, regenerate the lesson, and re-eval — loop until the skill-drafted spec reproduces the notebook's concepts **without reading it**. That reproduction is the proof the skill drives coverage. See `frontier-refactor-qa` → Coverage feedback loop and `sessions/_compiler/AUTHORING.md` §8.

### Momentum counterweight (2026-07-23) — keep full coverage, but don't wall it up

The Coverage Spec Rule maximizes completeness (every failure + cause + remedy + limit, each its own concept). Keep all of it — **length is a feature, never cut or defer coverage to shorten a day.** But completeness must not compile into a late-lesson "trap wall": when 3+ failure/limit concept units would run consecutively, **INTERLEAVE a play/payoff beat** (a live `%%% viz`, a predict-then-run `%%% demo`, or a "you just unlocked…" victory lap) between them so a beginner's momentum survives. This is a SEQUENCING rule only — no compression into tables, no deferral. `concept_structure_gate` emits a warn on a ≥3-consecutive failure cluster with no intervening play; the interest judge's `momentum` lever is the real enforcer.

Match the length of written deliverables to what the task needs: cover the substance (including full coverage as required above), but do not pad lessons or reports with filler sections, redundant summaries, or boilerplate. Length should come from teaching content, never from restatement.

## Produce Spec Rule (the produce block is the doing leg's specification)

The produce block is not only discovery framing. It is the **specification the day's `experiment.py` is built from** — `python3 sessions/_produce_spec.py <day-dir>` reads it back out deterministically — so it must state a CHECKABLE claim:

```text
1. Named observable      what the learner prints or plots, named
2. Concrete expectation  the value / shape / direction / ordering they should see
3. Acceptance criteria   a "What you should see" list that converts to one boolean per claim
```

The extraction is the test. `_produce_spec.py <day-dir>` must return a non-null `claude_prompt` (the numbered Option-B requirement list) AND a non-empty `acceptance`. A produce section that only says "predict what happens, then run it and observe" passes `reader_flow_gate` (which wants a discovery cue) and `concept_shell_gate` (which wants the string `experiment.py`) while specifying nothing — and the doing-leg author must then INVENT requirements, which `sessions/_compiler/AUTHORING.md` §10 forbids and which is how a lesson and its artifact come to disagree. Fixing this at spec time removes the invention step for every future day.

A bare `python3 sessions/_produce_spec.py` reports, across all stub days, how many lack an Option-B prompt or acceptance criteria. A missing prompt is a real SPEC gap to fix in `source.md`, not an authoring gap. **But confirm a reported gap by reading the day before you act on it.** A 2026-08-01 sweep reported 11 stub days with no acceptance criteria, including all 6 m07 days; every one of them in fact stated its criteria, and the extractor simply could not see the heading spelling they used. That false diagnosis was one edit away from being written into this skill as fact. Read the day's produce block yourself before you call a spec incomplete.

The discovery-framing requirement still holds (predict / observe register; `## Artifact as discovery` in `frontier-lesson-builder`). This is an ADDITIONAL requirement, not a replacement. HOW the artifact is written is `frontier-lesson-builder`'s doing-leg section; this rule only guarantees there is something to write it from.

## Seed Propagation Risk Rule

A seed module P1 can become required-to-fix before future rollout if:

- the module is marked `seed_module: true`,
- the issue affects lesson structure rather than one local detail,
- the issue is likely to be copied into future modules,
- the issue would be P0 for a fresh build under current skills,
- the fix can be targeted and low-risk.

This is not a retroactive module P0. Record it as a seed-stabilization P1.

Examples:

- definition/formula-first before mental picture,
- checklist opening before invitation,
- artifact framed as homework before curiosity,
- staff/frontier relevance before learner orientation,
- analogy is decorative rather than structural.

## Seed Stabilization Phase

If seed propagation risk exists, do a targeted seed stabilization before rolling out the next module.

Do not do a full rewrite unless QA finds P0.

A targeted seed stabilization may rewrite prose ordering and framing across affected lessons, but must preserve shell behavior, quest IDs, navigation, localStorage, quiz, BUILD/DEMOS, completion behavior, technical correctness, and artifacts.

## Learning Experience Layer Rule

Every lesson integrates:

```text
Experience Layer
→ Mechanism Layer
→ Staff Depth Layer
```

The Experience Layer is not just an intro. It must shape the lesson.

## Learning Barrier Lens

Ask:

- What makes this concept intimidating?
- What makes it interesting?
- What does the learner already understand from everyday life?
- What mental picture should appear before terms/formulas?
- Where does rigor become motivating?
- Does the artifact feel like discovery?

## Done

A rollout passes only when:

- required seed stabilization is complete,
- target module manifest status is pass or pass_with_p1,
- open P0 backlog is empty,
- Learning Experience Gate passes,
- audit/test evidence is reported,
- tracker and manifest are updated,
- report paths are recorded,
- **every day has a real, gated doing leg** — `python3 sessions/_experiment_check.py --module <module>` exits 0 (no placeholder stub; contract + real run green for every day).

A module whose lessons pass while its days ship the placeholder stub is NOT done: the learner cannot practise a single day. Verified 2026-08-01 — m07-thinking-in-jax is recorded `pass` with `open_p0: 0` and all 6 of its days stubbed, so the Done gate without this bullet certifies an unpractisable module. Repo-wide sweep: `python3 sessions/_experiment_check.py --all` — it globs every day including the stubs, so today it exits 1 and its FAIL list *is* the remaining stub list; that is expected state, not a regression.

## v8 Source-First Authoring Rule

Lessons are authored as source, not as HTML.

Author edits:

```text
sessions/<module>/day-XX/source.md   (reader-flow order)
```

A compiler generates:

```text
sessions/<module>/day-XX/lesson.html
```

Under v8 do not hand-edit `lesson.html`. Fix `source.md` and recompile.

The compiler owns the shell, the 7-section mapping, DEMOS/BUILD/QS, quest IDs, localStorage, quiz mechanics, navigation, completion behavior, and single-sourced anchors. Authoring reader-flow and owning invariants are separated — this is why direct HTML refactoring felt rough.

Authoritative grammar + compile/gate loop: `sessions/_compiler/AUTHORING.md` (source of truth — matches the shipped compiler). The older `sessions/_refactor/v8_source_first_authoring_plan.md` describes an obsolete grammar (`:::` fences, `{{anchor}}`, `[[term:weight]]`) the compiler now rejects — do not author from it.

## Three Authoring Modes

Select a mode per module and record it in the manifest:

```text
Exemplar          strong companion notebook exists → distill reader flow from it
Reference         papers/docs but no notebook       → author from refs + blueprint
First-Principles  no strong ref, no notebook        → author from barrier lens + blueprint
```

All three modes use the same `source.md` format and the same gates.

## No-Notebook Rollout Rule

A module without a notebook is not blocked and is not lower quality.

A notebook is a reader-flow calibration input, not a build dependency.

For Reference / First-Principles modules the Notebook Smoothness Gate is skipped (N/A), never failed. Reader Flow, Staff Depth, Shell Invariant, Visual/Evidence, and Artifact gates still run.

Example: JAX and Scaling Laws are Reference mode (JAX docs; arXiv papers via the `arxiv` MCP), authored source-first with no notebook.

## v8 Rollout Loop additions

Between "select target" and "refactor" in the Autonomous Rollout Loop, insert:

```text
determine authoring mode (notebook? refs? neither)
author source.md per day in reader-flow order (six rules, anchors single-sourced, spine chosen up front)
compile source.md → lesson.html
fix P0 in source.md and recompile (never hand-edit lesson.html)
write the day's experiment.py from its produce spec, then gate it
```

## Lesson Build Engine (v9)

The author + compile + QA steps of the Autonomous Rollout Loop are executed by the Workflow `sessions/_compiler/workflows/lesson_build.js` (run with the Workflow tool; `args {module, day, maxRounds, seedFindings}`). It chains: blind coverage-draft → author (ONE write-capable sub-agent, full regeneration into V9 concept structure) → compile → parallel judge panel (coverage · tone · interest · concept-structure · correctness) → deterministic router → self-correcting loop until P0-clear or `maxRounds`, then a per-lesson checkpoint.

Rules the engine enforces:

- One writer per lesson (full regeneration each round) — never split a lesson's prose across agents (repo `CLAUDE.md` §4).
- Deterministic gates (Reader Flow · Concept Shell · Notebook Smoothness) stay the hard compile block; the LLM judges gate the loop only.
- **skill_gaps are surfaced as a proposed diff for user approval — never auto-applied** (a Coverage Spec Rule edit changes every future lesson).
- `args.seedFindings` seeds a polish round with findings a prior checkpoint chose to fix.

The doing leg has its OWN engine: `sessions/_compiler/workflows/doing_leg_build.js` builds and gates each day's `experiment.py` from its produce spec. `lesson_build.js` never writes that file — the string `experiment` does not occur anywhere in it, and no judge lens reads an artifact — so a green compile plus a clean judge panel is not evidence the day is practisable. Run the doing-leg engine per module after the lesson engine converges, and gate with `_experiment_check.py` (definition in `frontier-refactor-qa` → Doing-Leg Gate).

Design + plan: `docs/superpowers/specs/2026-07-14-lesson-generation-orchestration-design.md`, `docs/superpowers/plans/2026-07-14-lesson-orchestration-engine.md`.

# Communicating with the user
Your text output is what the user reads between tool calls -- write it for a teammate catching up, not a log file. Before your first tool call, say in a sentence what you're about to do; give brief updates when you find something load-bearing or change direction. Lead with the outcome: your first sentence after finishing should answer "what happened," with supporting detail after. Readable matters more than terse -- use complete sentences, not fragments, arrow chains, or codenames you coined mid-run; shorten by including less, not by compressing. Match the response to the question.
