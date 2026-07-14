# Lesson Generation Orchestration + Frontier Evidence Pipeline — Design

**Date:** 2026-07-14
**Branch:** `build/capability-spiral`
**Status:** Approved (brainstorm) — pending spec review + implementation plan
**Related:** `sessions/_compiler/` (compiler + gates + workflows), `.claude/skills/frontier-*`, `sessions/_refactor/rollout_tracker.yaml`, `sessions/v9_coverage_inversion_report.md`, `sessions/v9_concept_shell_report.md`

---

## 1. Problem & purpose

Today, generating (or rewriting) a lesson is **skill-driven but single-agent**: one main-loop agent authors `source.md` inline, shells out to the deterministic `compile_lesson.py`, and runs deterministic gates. Sub-agents are used only for read-only adversarial QA. Two new probes (`coverage_review.js` — a 4-sub-agent read-only coverage review; `_coldgen/` — a blind builder-agent that generated a lesson) prove the pieces of a divided pipeline exist, but they are **not wired into one self-correcting loop**, and the LLM judges are advisory only.

**Purpose:** Build an **autonomous, self-correcting, per-lesson engine** that *regenerates* a session into the **V9 concept-based structure** — every concept-unit carrying intuition, an analogy, and its own real visualization, built up step-by-step for a beginner — proves each unit clears that bar via a panel of independent LLM judges, and ends each day by producing **frontier-facing evidence** (technical blog + reproducible experiment + interactive demo) that accumulates into a published portfolio. The engine runs one lesson to completion unattended, then checkpoints to the user.

## 2. Goals & non-goals

**Goals (in priority order):**
1. **Quality & consistency** — a panel of independent, diverse-lens judges + adversarial verification, so every lesson is provably complete and beginner-friendly in the proven Day-2 concept structure.
2. **Autonomy** — an orchestrator runs coverage → generate → compile → evaluate → fix without the user babysitting each step.
3. **Self-correction** — the judge→regenerate→re-judge loop closes automatically until judges pass or a bounded round-limit is hit.
4. **Frontier evidence** — each day produces real, reproducible, showable artifacts, accumulated into a published portfolio.

**Non-goals:**
- **Throughput / parallel authoring is explicitly NOT a goal.** We do not parallelize the *writing* of a lesson or fan out many days at once. This is deliberate: it lets us keep one writer per lesson and honor the tone-drift rule (repo `CLAUDE.md` §4).
- Not changing the deterministic compiler/gate contract (`compile_lesson.py` stays as-is).
- Not auto-editing global skills (the architect Coverage Spec Rule) without human approval.

## 3. Architecture (Approach C — orchestrated loop; sub-agents for judgment, one writer per lesson)

A new deterministic JS Workflow script owns **all** control flow; sub-agents do only judgment or writing.

```
        ┌───────────────── ORCHESTRATION WORKFLOW (deterministic JS) ──────────────────┐
        │                                                                                │
 ①spec  │ 1. COVERAGE   blind spec-draft (skill only) → reconcile vs committed manifest  │
 (RO)   │                                                                                │
 ②author│ 2. AUTHOR     ONE write-capable agent: FULL-REGENERATE source.md into V9       │
 (write)│                concept structure from committed spec (+ findings, round 2+)    │
        │ 3. COMPILE    deterministic compile_lesson.py  (UNCHANGED, hard gates block)   │
 ③pool  │ 4. EVALUATE   ∥ coverage-judge ∥ tone-judge ∥ concept-structure ∥ correctness  │
 (RO,∥) │                (independent LLM lenses, all read-only)                         │
 ④route │ 5. ROUTE      deterministic synthesizer:                                       │
        │                 • exec/tone/intuition/structure/correctness gap → back to 2    │
        │                 • SKILL gap → draft proposed diff → escalate to user           │
        │                 loop 4↔2 until judges pass OR max rounds                        │
        │ 6. EVIDENCE   ⑤evidence-producer + ⑥evidence-judge (own bounded sub-loop)      │
        │ 7. CHECKPOINT finished lesson + evidence + report → user sign-off              │
        └────────────────────────────────────────────────────────────────────────────────┘
```

**Deterministic vs. agent (the crux of C):**

| Step | Mechanism | Rationale |
|---|---|---|
| 1. Coverage draft | sub-agent (RO) + deterministic reconcile | judgment, proposal-only |
| 2. Author / fix | **1 write-capable agent, full regen** | single voice per lesson |
| 3. Compile | **deterministic** `compile_lesson.py` | already correct; no agent |
| 4. Evaluate | **parallel** sub-agents (RO) | independent lenses = quality |
| 5. Route + loop | **deterministic JS** | control flow belongs in code |
| 6. Evidence | sub-agents (⑤ write / ⑥ RO) + `evidence_compile.py` | reuse-first assembly |
| 7. Checkpoint | hand to user | per-lesson gate |

**Two-tier gating:** the deterministic gates (`reader_flow`, `concept_shell`, `notebook_smoothness`) remain hard-blocking *inside* `compile_lesson.py` (unchanged). The LLM judges are **promoted from advisory to loop-gating**: the orchestrator loops until they pass; they never touch the committed build path directly.

## 4. Roles & contracts

| # | Role | Access | In | Out |
|---|------|--------|----|-----|
| ① | **Spec-drafter** (reuse `coverage_review.js` P1) | read-only | topic + architect Coverage Spec Rule (blind to notebook & lesson) | `{covers, deferred, out_of_scope, reasoning}` |
| ② | **Author / fixer** (the ONLY writer; `_coldgen` builder pattern) | write | committed `covers` + lesson-builder/visual-evidence skills + `AUTHORING.md` (round 1); existing `source.md` + routed findings (round 2+) | full-regenerated `source.md` (`mode: concept`) |
| ③ | **Evaluator pool** (parallel) | read-only | compiled `lesson.html` + committed spec + notebook | per-judge `{findings:[{concept,kind,severity,why,fix}]}` |
| ④ | **Synthesizer / router** | deterministic JS | all findings | routed work-items; skill-gap proposal diffs |
| ⑤ | **Evidence-producer** | write | passed lesson + reused viz + executed experiment result | blog md + portfolio page glue; a minimal runnable `*_experiments.ipynb` when the day has none |
| ⑥ | **Evidence-judge** | read-only | evidence artifact + executed result | `{verdict, findings, numbers_match: bool}` |

**Evaluator pool (step 4) lenses — each an independent sub-agent:**
- **coverage-judge** — execution: each committed-spec concept is TAUGHT / MENTIONED / ABSENT (reuse `coverage_judge.judge`).
- **tone-judge** — beginner-friendliness vs the notebook gold standard (reuse `coverage_judge.judge_tone`).
- **concept-structure judge** (NEW) — for *each* concept-unit: (a) intuition-first framing, (b) analogy *with where-it-breaks-down*, (c) a genuine inline visual (real `<svg>`/`build-embed` iframe), (d) step-by-step build-up.
- **correctness / reader-flow verifier** — adversarial: technical errors, spine coherence, numeric self-consistency.

## 5. The V9 concept-structure target (what "generate" produces)

Proven on m02 Day 2. Every lesson body is **N concept-units**; each unit = **intro (intuition + analogy) → its own inline visualization → step-by-step build-up**, so a beginner meets one idea at a time. The author always writes `mode: concept` source (the `@@@ concept` + `%%% svg/demo/viz/quiz` grammar per `AUTHORING.md`). This is a **rewrite engine**: the other sessions are not concept-structured yet, and this pipeline is the mechanism that rewrites them into that structure, applied per-lesson via the rollout tracker.

## 6. Self-correcting loop & routing

- The synthesizer dedupes findings across the four judges and routes:
  - **exec / tone / intuition / structure / correctness gap** → back to ② (full regeneration with findings).
  - **skill-gap** (spec-drafter found a concept the committed manifest AND lesson both lack, and it is not correctly deferred/out-of-scope) → the orchestrator **drafts the concrete proposed change** (the specific edit to the architect skill's Coverage Spec Rule and/or the manifest `covers`) and surfaces *that diff* at the checkpoint. Never auto-applied — a skill edit changes every future lesson, so it is the user's decision.
- Execution is judged against the **committed manifest spec**; skill-completeness is judged against the **blind draft** (the split proven in `coverage_review.js`).

## 7. Evidence subsystem (last section each day)

**Forms:** technical blog post + reproducible experiment + interactive demo (all three).

**Reuse-first — evidence is compiled, not hand-written.** A new deterministic `evidence_compile.py` (sibling of `compile_lesson.py`) assembles each day's artifact by reusing canonical assets; ⑤ writes only narrative glue + framing:

| Artifact | Production (reuse-first) |
|---|---|
| **Interactive demo** | Reference the exact `sessions/viz/*.html` the lesson embeds (known from the source's `%%% viz`/`build-embed` blocks); copy into `portfolio/assets/` for self-containment. Zero regeneration. |
| **Reproducible experiment** | Find the day's `*_experiments.ipynb` (53 exist), **execute headless**, capture real results. No matching notebook → producer writes a minimal one. |
| **Technical blog** | Repurpose the lesson's staff-depth layer (staff-lens callouts, mechanism build-up, failure+remedy) into a Reader-B narrative; embed captured result; link the demo. |

**Hard constraint — evidence must be real.** Every number/plot/benchmark comes from *actually executing code in the sandbox*; ⑥ cross-checks that claims match the produced result (`numbers_match`). No fabricated figures — ever. (Guards against the finale-truncation / throughput-mismatch class of bug in the repo history.)

**Browserless-sandbox handling:**
- Experiment plots captured from **executed notebook cell outputs** (inline images), never `savefig` (banned by `scripts/validate_notebook.py`).
- Interactive demos embedded **live** and verified via **jsdom** structure checks (no pixel screenshots — no browser in sandbox).

**Register:** the lesson body is beginner (Reader A); the evidence is frontier-staff-facing (Reader B). This is a deliberate audience switch handled by a dedicated producer, not the beginner author.

**Accumulation:** `evidence_index.py` generates a portfolio hub (same pattern as the sessions hub) listing every day's artifact → one showcase.

## 8. Portfolio & repo topology

**Generate here, publish to a clean repo.**
- The pipeline generates `portfolio/` artifacts **in this repo** (reuse of viz + notebooks + prose is trivial here).
- `portfolio/` is built **self-contained** (its viz copied into `portfolio/assets/`, not deep-linked into `sessions/`) so it is portable.
- An automated **publish step** copies the self-contained `portfolio/` into a **separate clean public repo** whose GitHub Pages is the frontier-facing URL. Rationale: clean public identity, no beginner-scaffolding noise, and it sidesteps this repo's pending copyright-history-purge risk (`*.pdf` and `ML Design/examples/` are gitignored but history purge is still pending).
- Because `portfolio/` is self-contained, switching topology later is a `git subtree split` + push, not a rewrite.
- The existing Pages for this repo (`brucelee2077.github.io/applied-ai-research/`, root `index.html` → `sessions/index.html`, `.nojekyll`) is unchanged.

## 9. Loop control, failure handling, state

**Termination:** max **3** fix rounds per lesson (default, configurable). If judges still fail → stop, write a **blocker report** with residual findings, checkpoint to the user. No infinite loops, no runaway tokens.

**Independent sub-loops:** the lesson-body loop (2↔4) and the evidence loop (6) run separately, each with a default max of **3** rounds (configurable). A flaky experiment never blocks a good lesson — if evidence can't converge, the lesson still checkpoints, flagged "evidence incomplete."

**Graceful degradation:**
- **LLM bridge down** → judges fall back to tier-1 deterministic gates; report says "judges unavailable — verified structurally only." Never a silent pass.
- **Hard compile-gate failure** (`reader_flow`/`concept_shell`/`notebook_smoothness`, exit 2/3) → routed back to ② as P0; blocks before judges run.

**State & resumability:** the orchestrator is a Workflow script → resumable via `resumeFromRunId`. Durable state lives in artifacts (`source.md`, `lesson.html`, `portfolio/`) + the manifest/tracker. Re-running a passing lesson converges in round 1 with zero edits (idempotent). Any truncation (max-rounds, dropped findings) is **logged, never silent**.

## 10. Autonomy & checkpoint

**Per-lesson checkpoint.** The engine runs steps 1–6 unattended for one lesson, then stops and presents the finished lesson + evidence + a report for the user's sign-off before the next lesson. Skill-gap proposal diffs are presented here for approval.

## 11. Integration with existing skills/tracker (reuses, does not replace)

- The `frontier-curriculum-architect` rollout loop's "refactor + QA" steps become "run `lesson_build.js`" — skills still drive it; the workflow executes it.
- `coverage_review.js` roles fold into the orchestrator; `coverage_judge.py` is extended with the concept-structure + evidence judges.
- `compile_lesson.py` + the three deterministic gates are **unchanged**.
- Manifest + tracker status recording (`pass` / `pass_with_p1`) is reused as-is.

## 12. New & changed files

**New:**
- `sessions/_compiler/workflows/lesson_build.js` — the orchestrator.
- `sessions/_compiler/evidence_compile.py` — deterministic evidence assembler.
- `sessions/_compiler/evidence_index.py` — portfolio hub generator.
- `sessions/_compiler/gates/concept_structure_gate.py` — deterministic pre-check for the concept-unit triad.
- `scripts/publish_portfolio.*` — copy `portfolio/` → clean public repo + push.
- `portfolio/` — generated artifacts (self-contained).

**Extended:**
- `sessions/_compiler/gates/coverage_judge.py` — add concept-structure judge + evidence judge.
- `.claude/skills/frontier-*` — reference the orchestrator workflow (both skill locations).

**Unchanged:**
- `sessions/_compiler/compile_lesson.py`, `v8lib.py`, and the three deterministic gates.

## 13. Testing & validation

1. **Happy path** — dry-run `m04/day-01` (authored, gates green): loop converges round 1, no edits → validates plumbing.
2. **Fix loop** — run on a known-bad lesson (inject a coverage/structure gap): loop detects → regenerates → converges → self-correction proven.
3. **Regression** — `pytest sessions/_compiler/tests/` stays green (do not pin a hardcoded count — the suite is a moving target, ~45 test functions across 6 files as of this writing); add tests for `concept_structure_gate` + evidence judges.
4. **Evidence** — execute one `*_experiments.ipynb` end-to-end; confirm captured numbers match the write-up; confirm `numbers_match` gate catches an injected mismatch.
5. **Publish** — dry-run the publish step into a scratch clone; verify `portfolio/` is self-contained (no dangling `sessions/` links).

## 14. First proving ground

**m04's paused 6 days.** They are already blocked pending a V9 concept-structure rebuild (per the tracker + memory `m04-rollout-paused-viz-issue`), so building the engine and unblocking m04 are the same task. After m04, proceed module-by-module via the tracker with per-lesson checkpoints.

## 15. Phased build order (for the implementation plan)

1. **Orchestrator core** — `lesson_build.js` chaining steps 1–5 (reusing existing judges), loop + routing + termination, on the happy path (m04/day-01).
2. **Concept-structure judge + gate** — the NEW quality bar; wire into step 4 + `concept_structure_gate.py`.
3. **Skill-gap proposal** — router drafts the diff + checkpoint surfacing.
4. **Evidence subsystem** — `evidence_compile.py` + producer/judge + real-evidence execution + `evidence_index.py`.
5. **Portfolio publish** — `publish_portfolio.*` to the clean public repo.
6. **Rollout** — run the full engine on m04's 6 days, then continue via the tracker.

## 16. Risks & open items

- **Persistent-voice across rounds:** mitigated by full regeneration (one agent, one call per round) + the deterministic compiler owning the shell. Round-to-round variance is acceptable since only the converged version ships.
- **Token cost:** full regeneration each round is more expensive than surgical edits (accepted — throughput is not a goal; the round-limit bounds cost).
- **Clean-repo publish target:** the separate public repo name/URL is not yet chosen — a setup decision at Phase 5, not a blocker for Phases 1–4.
- **Notebook execution environment:** relies on system `python3` (the `.venv` is broken per repo memory); experiment execution must use system python.

## 17. Out of scope (YAGNI)

- Parallel/concurrent authoring of multiple days (contradicts non-goal).
- Auto-applying skill edits without user approval.
- Pixel-level visual verification (no browser in sandbox; structural jsdom checks only).
- A capability-claim-log artifact form (not selected).
- Changing the deterministic compiler/gate contract.
