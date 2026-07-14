# Lesson Orchestration Engine (Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-correcting, per-lesson orchestration engine that regenerates a session into the V9 concept structure and loops (author → compile → judge → regenerate) until a quality panel passes, then checkpoints to the user.

**Architecture:** A deterministic JS Workflow script (`lesson_build.js`) owns all control flow; sub-agents do only judgment (parallel LLM panel) or writing (one author agent per lesson, full regeneration). The existing `compile_lesson.py` + its three hard gates are unchanged. Two new evaluators are added: a deterministic `concept_structure_gate.py` (per-concept triad on source) and an LLM `judge_concept_structure` in `coverage_judge.py`.

**Tech Stack:** Python 3.11 (system python — the repo `.venv` is broken), `pytest`, the Workflow tool (JS, run via the `Workflow` tool), the local keyless LLM bridge (`http://localhost:11211`).

**Scope:** This is Plan 1 of 2. Plan 1 = spec phases 1–3 (the engine). Plan 2 (separate) = evidence + portfolio (phases 4–5). Spec: `docs/superpowers/specs/2026-07-14-lesson-generation-orchestration-design.md`.

---

## File Structure

**New:**
- `sessions/_compiler/gates/concept_structure_gate.py` — deterministic per-concept-unit STRUCTURE check on `source.md` (intro → visual → build-up, in order). One responsibility: structural triad. ~70 lines.
- `sessions/_compiler/tests/test_concept_structure_gate.py` — pytest for the gate.
- `sessions/_compiler/workflows/lesson_build.js` — the orchestrator Workflow (control flow only).

**Modified:**
- `sessions/_compiler/gates/coverage_judge.py` — add `judge_concept_structure(...)`; extend `run_from_paths` to include it.
- `sessions/_compiler/tests/test_coverage_judge.py` — add tests for the new judge (graceful-fallback + prompt content).
- `.claude/skills/frontier-curriculum-architect/SKILL.md` and `.claude/skills/frontier-refactor-qa/SKILL.md` (and their mirrors under `frontier_lab_refactor_skills_v8/skills/`) — reference `lesson_build.js` as the executor of the rollout loop's author/QA steps.

**Unchanged:** `compile_lesson.py`, `v8lib.py`, `reader_flow_gate.py`, `concept_shell_gate.py`, `notebook_smoothness_gate.py`, `coverage_gate.py`.

---

## Task 1: `concept_structure_gate.py` — deterministic per-concept triad

**Files:**
- Create: `sessions/_compiler/gates/concept_structure_gate.py`
- Test: `sessions/_compiler/tests/test_concept_structure_gate.py`

This gate runs on the **source** (not compiled HTML). For every `@@@ concept` block it asserts the three beats exist **in order**: intro prose *before* the first visual, a real visual (`%%% svg` / `%%% viz` / a closed `<svg>…</svg>`), and build-up prose *after* the visual. `concept_shell_gate` already checks "a visual exists" on HTML; this gate adds the **ordering + prose-on-both-sides** structure that makes a unit intuition-first.

- [ ] **Step 1: Write the failing test**

```python
# sessions/_compiler/tests/test_concept_structure_gate.py
import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import concept_structure_gate as g

_INTRO = "This is a plain-words intro that explains the idea before any picture appears, in enough words to be real."
_BUILD = "Now the build-up: a worked example that walks step by step through the mechanism in concrete detail here."
_SVG = "%%% svg\n<svg viewBox=\"0 0 10 10\"><path d=\"M0 0 L10 10\"/></svg>\n%%%"

def _concept(cid, intro=_INTRO, svg=_SVG, build=_BUILD):
    return f'@@@ concept id={cid} tag="t" title="T" gotit="ok"\n{intro}\n{svg}\n{build}\n'

def _doc(*concepts):
    return "---\nmode: concept\n---\n@@@ hero\n@lede hook\n@goal g\n" + "".join(concepts) + "@@@ fin\n"

def test_three_well_formed_concepts_pass():
    src = _doc(_concept('c1'), _concept('c2'), _concept('c3'))
    ok, msgs = g.run(src)
    assert ok, msgs

def test_fewer_than_three_concepts_fail():
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2')))
    assert not ok
    assert any('>=3 concept units' in m and m.startswith('FAIL') for m in msgs)

def test_concept_without_visual_fails():
    bad = '@@@ concept id=c3 tag="t" title="T" gotit="ok"\n' + _INTRO + '\n' + _BUILD + '\n'
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2'), bad))
    assert not ok
    assert any('c3 has a visual' in m and m.startswith('FAIL') for m in msgs)

def test_concept_with_visual_but_no_buildup_fails():
    nobuild = f'@@@ concept id=c3 tag="t" title="T" gotit="ok"\n{_INTRO}\n{_SVG}\n'
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2'), nobuild))
    assert not ok
    assert any('c3 has build-up' in m and m.startswith('FAIL') for m in msgs)

def test_concept_with_visual_but_no_intro_fails():
    nointro = f'@@@ concept id=c3 tag="t" title="T" gotit="ok"\n{_SVG}\n{_BUILD}\n'
    ok, msgs = g.run(_doc(_concept('c1'), _concept('c2'), nointro))
    assert not ok
    assert any('c3 has intro prose' in m and m.startswith('FAIL') for m in msgs)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest sessions/_compiler/tests/test_concept_structure_gate.py -v`
Expected: FAIL / collection error — `No module named 'concept_structure_gate'`.

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
# =============================================================================
# Concept Structure Gate (v9) — deterministic per-concept-unit TRIAD check.
# =============================================================================
# Runs on the SOURCE (mode:concept). For every @@@ concept block asserts the
# three beats IN ORDER: (1) intro prose BEFORE its first visual, (2) a real
# visual (%%% svg / %%% viz / a closed <svg>...</svg>), (3) build-up prose AFTER
# the visual. Complements concept_shell_gate (which checks "a visual exists" on
# compiled HTML) by enforcing intuition-first ordering. Semantic quality
# (is the analogy good? intuition-first *in spirit*?) is the LLM judge's job
# (coverage_judge.judge_concept_structure) — this gate is the cheap structural
# floor.
#
# Reusable:  from concept_structure_gate import run ; ok, msgs = run(source_text)
# CLI:       python3 gates/concept_structure_gate.py <source.md>   (exit 0/3)
# =============================================================================
import sys, re

_MIN_PROSE = 40  # chars of real prose required on each side of the visual (tunable)
_VIS_OPEN = re.compile(r'^%%%\s+(svg|viz)\b', re.MULTILINE)
_SVG_CLOSED = re.compile(r'<svg[\s>].*?</svg>', re.DOTALL)
_WIDGET = re.compile(r'%%%.*?%%%', re.DOTALL)  # strip any widget when measuring prose


def _concept_blocks(body):
    """Yield (args_line, block_body) for each '@@@ concept ...' up to the next '@@@'."""
    for part in re.split(r'(?m)^@@@\s+', body):
        if part.startswith('concept'):
            line, _, rest = part.partition('\n')
            yield line, rest


def run(source_text):
    """Return (ok: bool, msgs: [str]). msgs are 'pass '/'FAIL ' prefixed labels."""
    msgs, ok = [], [True]

    def chk(cond, label):
        msgs.append(('pass ' if cond else 'FAIL ') + label)
        ok[0] = ok[0] and bool(cond)

    body = re.sub(r'^---.*?\n---\s*', '', source_text, count=1, flags=re.DOTALL)
    blocks = list(_concept_blocks(body))
    chk(len(blocks) >= 3, '>=3 concept units (got %d)' % len(blocks))

    for args, text in blocks:
        m = re.search(r'id=(?:"([^"]+)"|(\S+))', args)
        cid = (m.group(1) or m.group(2)) if m else '?'

        vis = _VIS_OPEN.search(text)
        svg = _SVG_CLOSED.search(text)
        # first visual is whichever appears earliest
        first = min([x for x in (vis, svg) if x], key=lambda mm: mm.start(), default=None)
        chk(bool(first), 'concept %s has a visual' % cid)
        if not first:
            continue

        intro = _WIDGET.sub('', text[:first.start()]).strip()
        chk(len(intro) >= _MIN_PROSE, 'concept %s has intro prose before its visual' % cid)

        # find where the first visual ends, then measure build-up after it
        if first is vis:
            close = re.search(r'(?m)^%%%\s*$', text[first.end():])
            after = text[first.end():][close.end():] if close else ''
        else:
            after = text[first.end():]
        buildup = _WIDGET.sub('', after).strip()
        chk(len(buildup) >= _MIN_PROSE, 'concept %s has build-up after its visual' % cid)

    return ok[0], msgs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    a = ap.parse_args()
    ok, msgs = run(open(a.source, encoding='utf-8').read())
    for m in msgs:
        print('  ', m)
    print('\n' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 3)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest sessions/_compiler/tests/test_concept_structure_gate.py -v`
Expected: 5 passed.

- [ ] **Step 5: Sanity-check against a real lesson**

Run: `python3 sessions/_compiler/gates/concept_structure_gate.py sessions/m02-the-neuron/day-02-activations/source.md`
Expected: `PASS` (Day 2 is the proven concept lesson). If it FAILs, the thresholds/parse need adjusting to the real source before proceeding — investigate, don't loosen blindly.

- [ ] **Step 6: Commit**

```bash
git add sessions/_compiler/gates/concept_structure_gate.py sessions/_compiler/tests/test_concept_structure_gate.py
git commit -m "feat(v9): concept_structure_gate — deterministic per-concept intro/visual/build-up triad"
```

---

## Task 2: `judge_concept_structure` — the semantic concept-structure LLM judge

**Files:**
- Modify: `sessions/_compiler/gates/coverage_judge.py`
- Test: `sessions/_compiler/tests/test_coverage_judge.py`

Adds a third LLM judge (alongside `judge` and `judge_tone`), modeled on `judge_tone`: per concept, is it **intuition-first**, does its **analogy include where-it-breaks-down**, is the **build-up step-by-step**. Advisory, graceful fallback, never raises.

- [ ] **Step 1: Write the failing tests** (append to `test_coverage_judge.py`)

```python
# --- concept-structure judge ------------------------------------------------
def test_struct_prompt_contains_concepts_and_axes():
    p = cj._struct_prompt('lesson body about relu and sigmoid', ['ReLU', 'Sigmoid'])
    assert 'ReLU' in p and 'Sigmoid' in p
    assert 'intuition_first' in p and 'analogy' in p and 'buildup' in p
    assert 'lesson body about relu' in p

def test_struct_prompt_truncates_long_lesson():
    long_text = 'y' * (cj._STRUCT_MAX + 5000)
    p = cj._struct_prompt(long_text, ['ReLU'])
    assert long_text[:cj._STRUCT_MAX] in p
    assert long_text not in p

def test_judge_structure_graceful_when_sdk_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == 'openai':
            raise ImportError('simulated missing sdk')
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', fake_import)
    res = cj.judge_concept_structure('lesson text', ['ReLU'])
    assert res['status'] == 'BRIDGE_UNAVAILABLE'
    assert res['concepts'] == [] and res['overall'] == 'N/A'

def test_judge_structure_empty_concepts_is_na():
    res = cj.judge_concept_structure('lesson', [])
    assert res['status'] == 'N/A'
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest sessions/_compiler/tests/test_coverage_judge.py -k struct -v`
Expected: FAIL — `module 'coverage_judge' has no attribute '_struct_prompt'`.

- [ ] **Step 3: Implement** (add to `coverage_judge.py`, after `judge_tone`)

```python
# ===========================================================================
# CONCEPT-STRUCTURE JUDGE — per-concept intuition-first / analogy / build-up
# ===========================================================================
# The deterministic concept_structure_gate proves the triad is STRUCTURALLY
# present (prose -> visual -> prose). This judge grades whether the unit is
# intuition-first IN SPIRIT: leads with a felt picture, carries a real analogy
# WITH its "where it breaks down" half, and builds up step-by-step. Advisory,
# graceful fallback, never raises. Mirrors judge_tone.
_STRUCT_MAX = 22000
_STRUCT_SYS = (
    "You are a CONCEPT-STRUCTURE judge for a beginner ML lesson built as concept units. "
    "For each named concept, judge whether the unit (1) leads with intuition/a felt picture "
    "BEFORE notation, (2) carries a concrete everyday analogy INCLUDING where it breaks down, "
    "and (3) builds up step-by-step rather than dumping the mechanism. Be specific and quote. "
    "Return STRICT JSON only (no prose, no markdown fences)."
)


def _struct_prompt(lesson_text, concept_titles):
    names = '\n'.join('- %s' % c for c in (concept_titles or [])) or '(none)'
    return f"""CONCEPT UNITS TO JUDGE (by name/title):
{names}

LESSON TEXT (plain-text extract):
\"\"\"
{lesson_text[:_STRUCT_MAX]}
\"\"\"

For EACH concept unit above, return a verdict on three axes. verdict is one of:
GOOD (clearly meets it) / WEAK (partially) / MISSING (absent).
Return STRICT JSON:
{{
  "concepts": [ {{"concept":"<name>", "intuition_first":"GOOD|WEAK|MISSING",
                  "analogy":"GOOD|WEAK|MISSING", "buildup":"GOOD|WEAK|MISSING",
                  "note":"<one line, quote a spot>", "fix":"<concrete rewrite if not GOOD>"}} ],
  "overall": "GOOD|WEAK|MISSING",
  "summary": "<=2 sentences"
}}
Rules: judge every concept. "analogy":"GOOD" requires BOTH a concrete analogy AND an explicit
"where it breaks down" (or equivalent limit). Lead-with-formula => intuition_first is WEAK/MISSING."""


def judge_concept_structure(lesson_text, concept_titles, model=MODEL, timeout=90):
    """LLM per-concept structure judge. Never raises. N/A when no concepts given."""
    if not concept_titles:
        return {'status': 'N/A', 'reason': 'no concepts to judge',
                'concepts': [], 'overall': 'N/A', 'summary': ''}
    try:
        from openai import OpenAI
    except Exception as e:
        return {'status': 'BRIDGE_UNAVAILABLE', 'error': str(e),
                'concepts': [], 'overall': 'N/A', 'summary': ''}
    try:
        client = OpenAI(api_key='not-needed', base_url=BRIDGE_URL, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': _STRUCT_SYS},
                      {'role': 'user', 'content': _struct_prompt(lesson_text, concept_titles)}],
            max_tokens=2000,
        )
        content = (resp.choices[0].message.content or '').strip()
    except Exception as e:
        return {'status': 'BRIDGE_UNAVAILABLE', 'error': str(e),
                'concepts': [], 'overall': 'N/A', 'summary': ''}
    data = _extract_json(content)
    if data is None:
        return {'status': 'PARSE_ERROR', 'raw': content,
                'concepts': [], 'overall': 'N/A', 'summary': ''}
    data.setdefault('concepts', [])
    data.setdefault('overall', 'N/A')
    data.setdefault('summary', '')
    data['status'] = 'OK'
    return data
```

- [ ] **Step 4: Wire into `run_from_paths`** — add the structure judge to the returned dict. Locate the `return {'coverage': ..., 'tone': ...}` at the end of `run_from_paths` and extend it:

```python
    # concept titles come from the source's @@@ concept title="..." args
    import re as _re
    src_text = open(source_path, encoding='utf-8').read()
    concept_titles = _re.findall(r'@@@\s+concept\b[^\n]*\btitle="([^"]+)"', src_text)
    return {'coverage': judge(lesson_text, spec, nb_concepts, curation),
            'tone': judge_tone(lesson_text, notebook_md),
            'structure': judge_concept_structure(lesson_text, concept_titles)}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest sessions/_compiler/tests/test_coverage_judge.py -v`
Expected: all prior tests + 4 new ones pass.

- [ ] **Step 6: Commit**

```bash
git add sessions/_compiler/gates/coverage_judge.py sessions/_compiler/tests/test_coverage_judge.py
git commit -m "feat(v9): judge_concept_structure — LLM per-concept intuition/analogy/build-up judge"
```

---

## Task 3: `lesson_build.js` — orchestrator core (coverage → author → compile, happy path)

**Files:**
- Create: `sessions/_compiler/workflows/lesson_build.js`

Model on `coverage_review.js` (same `meta`/`phases`/`agent()` shape). This task builds steps 1–3 with **no loop yet**: draft coverage (blind), reconcile, author `source.md`, compile, report. Validation is an integration dry-run (JS workflows are not pytest-testable).

- [ ] **Step 1: Write the orchestrator (steps 1–3)**

```javascript
export const meta = {
  name: 'lesson-build',
  description: 'Self-correcting per-lesson engine: draft coverage (skill-blind) -> author V9 concept source -> compile -> judge panel -> regenerate until pass -> checkpoint. args {module, day}. Writes source.md; compiles; reports.',
  whenToUse: 'To (re)generate one lesson-day into the V9 concept structure with an autonomous judge-gated fix loop.',
  phases: [
    { title: 'Coverage' },
    { title: 'Author' },
    { title: 'Compile' },
    { title: 'Evaluate' },
    { title: 'Route' },
  ],
}

const A = args || {}
const module_ = A.module || 'm04-first-model-mlp'
const day     = A.day    || 'day-01-mlp-mnist'
const source  = `sessions/${module_}/${day}/source.md`
const lesson  = `sessions/${module_}/${day}/lesson.html`
const MAX_ROUNDS = A.maxRounds || 3

const SPEC_SCHEMA = {
  type: 'object',
  properties: {
    covers: { type: 'array', items: { type: 'string' } },
    deferred: { type: 'array', items: { type: 'object',
      properties: { topic: { type: 'string' }, where: { type: 'string' } }, required: ['topic', 'where'] } },
    out_of_scope: { type: 'array', items: { type: 'object',
      properties: { topic: { type: 'string' }, reason: { type: 'string' } }, required: ['topic', 'reason'] } },
    reasoning: { type: 'string' },
  },
  required: ['covers', 'deferred', 'out_of_scope', 'reasoning'],
}

const COMPILE_SCHEMA = {
  type: 'object',
  properties: {
    wrote_source: { type: 'boolean' },
    compiled: { type: 'boolean' },
    exit_code: { type: 'integer' },
    gate_output: { type: 'string' },
    concept_count: { type: 'integer' },
  },
  required: ['wrote_source', 'compiled', 'exit_code', 'gate_output'],
}

// --- Phase 1: blind coverage draft (reused role from coverage_review.js) ----
phase('Coverage')
const draft = await agent(
  `You are the coverage SPEC-DRAFTER sub-agent.
Read ONLY the "Coverage Spec Rule" section of .claude/skills/frontier-curriculum-architect/SKILL.md.
Do NOT read any notebook or existing lesson — blind draft.
Draft the coverage spec for a BEGINNER lesson for module "${module_}", day "${day}".
Apply every rung: core mechanism family; historical ancestor when it motivates the modern form;
for EVERY failure mode its REMEDY; capability limits; forward-pointers -> deferred; out_of_scope with reason.
Return covers, deferred, out_of_scope, and one-paragraph reasoning.`,
  { label: 'spec-draft (blind)', phase: 'Coverage', schema: SPEC_SCHEMA })

// Reconcile happens deterministically after we can read the manifest; for the
// happy path we pass the blind draft to the author as the working spec and let
// the committed manifest (read by the author) be authoritative.

// --- Phase 2+3: author writes V9 concept source, then compiles ---------------
phase('Author')
async function authorAndCompile(round, findings) {
  const findingsBlock = findings
    ? `\n\nThis is FIX ROUND ${round}. FULLY REGENERATE the lesson (do not patch) addressing these findings:\n${JSON.stringify(findings, null, 2)}`
    : ''
  return await agent(
    `You are the AUTHOR sub-agent — the ONLY writer for this lesson; you own its voice end to end.
Author the V9 concept-mode lesson at ${source} for module "${module_}", day "${day}".
Follow the authoring grammar in sessions/_compiler/AUTHORING.md EXACTLY (mode: concept; @@@ hero/concept/quiz/produce/fin; %%% svg|viz|demo|quiz widgets).
Follow .claude/skills/frontier-lesson-builder and -visual-evidence-builder: every concept unit = intuition + analogy (WITH where it breaks down) -> its OWN inline visual -> step-by-step build-up. Beginner voice (repo CLAUDE.md §5/§7).
Coverage to realize (committed manifest is authoritative; this blind draft is guidance):
${JSON.stringify(draft, null, 2)}
Write the FULL source.md, then compile and report:
  python3 sessions/_compiler/compile_lesson.py ${source}
Also run: python3 sessions/_compiler/gates/concept_structure_gate.py ${source}
Return wrote_source, compiled (exit 0?), exit_code, the gate_output (both commands' tail), concept_count.${findingsBlock}`,
    { label: `author r${round}`, phase: 'Author', schema: COMPILE_SCHEMA, agentType: 'general-purpose' })
}

phase('Compile')
let compileRes = await authorAndCompile(0, null)
log(`author r0: compiled=${compileRes.compiled} exit=${compileRes.exit_code} concepts=${compileRes.concept_count}`)

return { module: module_, day, source, lesson, draft, compileRes }
```

- [ ] **Step 2: Dry-run on m04/day-01 (happy path)**

Invoke via the Workflow tool: `Workflow({ scriptPath: "sessions/_compiler/workflows/lesson_build.js", args: { module: "m04-first-model-mlp", day: "day-01-mlp-mnist" } })`
Expected: workflow completes; result shows `compileRes.compiled === true`, `exit_code === 0`, `concept_count >= 3`. (m04/day-01 currently exists as `mode: exemplar`; the author will regenerate it into `mode: concept` — expect a real rewrite. If the author reports a hard-gate failure, that is a legitimate finding for Task 4's loop, not a Task 3 bug.)

- [ ] **Step 3: Commit**

```bash
git add sessions/_compiler/workflows/lesson_build.js
git commit -m "feat(v9): lesson_build.js orchestrator core — blind coverage + author + compile (no loop yet)"
```

---

## Task 4: Evaluator pool + router + self-correcting loop

**Files:**
- Modify: `sessions/_compiler/workflows/lesson_build.js`

Add the parallel judge panel (steps 4), the deterministic router (step 5), and the 2↔4 loop with `MAX_ROUNDS` termination.

- [ ] **Step 1: Add the evaluator pool + loop** (replace the Task-3 `return` with the loop)

```javascript
const JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: { type: 'object',
      properties: {
        concept: { type: 'string' },
        kind: { type: 'string', description: 'exec_gap | intuition | analogy | buildup | tone | correctness | skill_gap' },
        severity: { type: 'string', enum: ['P0', 'P1', 'P2'] },
        why: { type: 'string' }, fix: { type: 'string' },
      }, required: ['kind', 'severity', 'why'] } },
    verdict: { type: 'string', enum: ['PASS', 'GAPS'] },
    lens: { type: 'string' },
  },
  required: ['findings', 'verdict', 'lens'],
}

const LENSES = [
  { key: 'coverage', prompt: `Run: python3 sessions/_compiler/gates/coverage_judge.py ${lesson} --source ${source}. Parse the "Coverage Judge" section: report each MENTIONED/ABSENT spec concept as an exec_gap finding (P0). If the bridge is unavailable, say so with verdict PASS (structural fallback) and note it.` },
  { key: 'tone', prompt: `Run: python3 sessions/_compiler/gates/coverage_judge.py ${lesson} --source ${source}. Parse the "Beginner-Friendliness Judge" section: report each BELOW/WORSE dimension as a tone finding (P1). Bridge unavailable -> verdict PASS, note it.` },
  { key: 'structure', prompt: `Read ${lesson} (strip HTML). For each concept unit judge intuition_first / analogy-with-breakdown / stepwise buildup. Report each WEAK/MISSING as an intuition|analogy|buildup finding (P0 for MISSING, P1 for WEAK).` },
  { key: 'correctness', prompt: `Adversarially read ${lesson} (strip HTML) for TECHNICAL errors, numeric self-inconsistency, and broken narrative spine. Report each as a correctness finding (P0). Default to reporting if unsure.` },
]

async function evaluate() {
  const results = await parallel(LENSES.map(l => () =>
    agent(`You are the ${l.key.toUpperCase()} evaluator (read-only). ${l.prompt}
Return findings[], a verdict (PASS iff no P0), and lens="${l.key}".`,
      { label: `judge:${l.key}`, phase: 'Evaluate', schema: JUDGE_SCHEMA })))
  return results.filter(Boolean)
}

// deterministic router: split loop-back findings from skill-gap escalations
function route(evals) {
  const all = evals.flatMap(e => (e.findings || []).map(f => ({ ...f, lens: e.lens })))
  const skillGaps = all.filter(f => f.kind === 'skill_gap')
  const fixable = all.filter(f => f.kind !== 'skill_gap')
  const p0 = fixable.filter(f => f.severity === 'P0')
  return { all, skillGaps, fixable, p0, pass: p0.length === 0 }
}

// --- the self-correcting loop (2 <-> 4) -------------------------------------
let round = 0, routing = null, lastEvals = []
while (round < MAX_ROUNDS) {
  // hard-gate failure short-circuits the LLM panel: loop straight back to author
  if (!compileRes.compiled) {
    log(`r${round}: hard gate failed (exit ${compileRes.exit_code}) -> regenerate`)
    round += 1
    compileRes = await authorAndCompile(round, [{ kind: 'compile_gate', severity: 'P0', why: compileRes.gate_output }])
    continue
  }
  phase('Evaluate')
  lastEvals = await evaluate()
  phase('Route')
  routing = route(lastEvals)
  log(`r${round}: P0=${routing.p0.length} fixable=${routing.fixable.length} skill_gaps=${routing.skillGaps.length} pass=${routing.pass}`)
  if (routing.pass) break
  round += 1
  if (round >= MAX_ROUNDS) break
  compileRes = await authorAndCompile(round, routing.fixable)
}

const converged = !!(routing && routing.pass && compileRes.compiled)
if (!converged) log(`NOT converged after ${round} rounds — blocker report at checkpoint`)

return {
  module: module_, day, source, lesson, converged, rounds: round,
  blind_draft: draft, final_compile: compileRes,
  evaluations: lastEvals, routing,
}
```

- [ ] **Step 2: Fix-loop dry-run** — run on a lesson with a deliberately injected gap (e.g. temporarily author a source with only 2 concepts, or a concept missing its analogy) and confirm the loop detects P0 findings, regenerates, and either converges or stops at `MAX_ROUNDS` with `converged: false` and a populated `routing`. Confirm no infinite loop.

- [ ] **Step 3: Happy-path dry-run** — re-run on m04/day-01; confirm it converges (`converged: true`) within `MAX_ROUNDS` and `routing.p0.length === 0`.

- [ ] **Step 4: Commit**

```bash
git add sessions/_compiler/workflows/lesson_build.js
git commit -m "feat(v9): lesson_build.js — parallel judge panel + deterministic router + self-correcting loop"
```

---

## Task 5: Skill-gap proposal + checkpoint report

**Files:**
- Modify: `sessions/_compiler/workflows/lesson_build.js`

When the router surfaces `skillGaps`, draft the concrete proposed change to the architect skill's Coverage Spec Rule / manifest — do not apply it. Emit a human-readable checkpoint report as the workflow's final value.

- [ ] **Step 1: Add skill-gap proposal drafting + report assembly** (before the final `return`)

```javascript
let skillProposal = null
if (routing && routing.skillGaps.length) {
  phase('Route')
  skillProposal = await agent(
    `You are the SKILL-GAP PROPOSER (read-only; you propose, you do NOT edit any skill).
The blind coverage draft + judges found concepts the committed spec/lesson lack, that are NOT correctly deferred/out-of-scope:
${JSON.stringify(routing.skillGaps, null, 2)}
Read .claude/skills/frontier-curriculum-architect/SKILL.md "Coverage Spec Rule".
Draft the SMALLEST concrete edit (a unified-diff-style before/after snippet) to that rule and/or the manifest coverage.<day> that would make the blind draft reproduce these concepts next time. Explain in one paragraph why. Output the proposal text only — it will be shown to the user for approval.`,
    { label: 'skill-gap proposal', phase: 'Route', schema: {
      type: 'object',
      properties: { proposal_diff: { type: 'string' }, rationale: { type: 'string' }, targets: { type: 'array', items: { type: 'string' } } },
      required: ['proposal_diff', 'rationale'],
    } })
}

const report = [
  `# Lesson build report — ${module_}/${day}`,
  ``,
  `- Converged: ${converged}  (rounds: ${round}/${MAX_ROUNDS})`,
  `- Final compile: exit ${compileRes.exit_code}, ${compileRes.concept_count || '?'} concepts`,
  `- Residual P0 (if any): ${routing ? routing.p0.length : 'n/a'}`,
  routing && routing.p0.length ? `\n## Residual findings\n${routing.p0.map(f => `- [${f.severity}/${f.lens}] ${f.kind}: ${f.why}`).join('\n')}` : `\n(no residual P0)`,
  skillProposal ? `\n## Skill-gap proposal (needs your approval)\n${skillProposal.rationale}\n\n\`\`\`diff\n${skillProposal.proposal_diff}\n\`\`\`` : `\n(no skill-gap proposals)`,
].join('\n')

log(report)

return {
  module: module_, day, source, lesson, converged, rounds: round,
  blind_draft: draft, final_compile: compileRes,
  evaluations: lastEvals, routing, skill_proposal: skillProposal, report,
}
```

- [ ] **Step 2: Dry-run with a seeded skill-gap** — run on a topic where the blind draft is known to under-enumerate (e.g. the m02-day02 softmax/temperature long-tail noted in `_coldgen/m02-day02-activations/_coverage.md`) and confirm `skill_proposal` is populated with a concrete diff and that nothing under `.claude/skills/` was modified (the proposer is read-only).

Run to confirm no skill files changed: `git status --short .claude/skills/`
Expected: no modifications from the workflow run.

- [ ] **Step 3: Commit**

```bash
git add sessions/_compiler/workflows/lesson_build.js
git commit -m "feat(v9): lesson_build.js — skill-gap proposal drafting + checkpoint report"
```

---

## Task 6: Wire into skills + end-to-end validation

**Files:**
- Modify: `.claude/skills/frontier-curriculum-architect/SKILL.md`, `.claude/skills/frontier-refactor-qa/SKILL.md` (+ mirrors under `frontier_lab_refactor_skills_v8/skills/`)

- [ ] **Step 1: Add a "Lesson Build Engine" note to `frontier-curriculum-architect/SKILL.md`** — under "v8 Rollout Loop additions", add: the author/compile/QA steps of the loop are executed by the Workflow `sessions/_compiler/workflows/lesson_build.js` (args `{module, day}`); it self-corrects to the judge panel and stops at a per-lesson checkpoint; skill-gap proposals require user approval before any skill edit. Keep the wording consistent with how `coverage_review.js` is already referenced in `frontier-refactor-qa`.

- [ ] **Step 2: Add a matching "Lesson Build Engine" subsection to `frontier-refactor-qa/SKILL.md`** near the "Coverage Review Workflow" section, noting the engine reuses the same judges and the two-tier gating (deterministic hard gates block at compile; LLM judges gate the loop).

- [ ] **Step 3: Mirror both edits** into `frontier_lab_refactor_skills_v8/skills/frontier-curriculum-architect/SKILL.md` and `.../frontier-refactor-qa/SKILL.md` (the repo keeps skill parity across both locations — verified in the audit).

- [ ] **Step 4: Full regression + end-to-end**

```bash
python3 -m pytest sessions/_compiler/tests/ -v          # all green, incl. new gate + judge tests
python3 sessions/_compiler/gates/concept_structure_gate.py sessions/m02-the-neuron/day-02-activations/source.md   # PASS
```
Then a final `lesson_build.js` dry-run on m04/day-01: expect `converged: true` and a clean checkpoint report. Confirm `git status` shows changes ONLY under `sessions/m04-first-model-mlp/day-01-mlp-mnist/` (+ the engine files) — no bleed into other modules or the concurrent session's files.

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/frontier-curriculum-architect/SKILL.md .claude/skills/frontier-refactor-qa/SKILL.md frontier_lab_refactor_skills_v8/skills/frontier-curriculum-architect/SKILL.md frontier_lab_refactor_skills_v8/skills/frontier-refactor-qa/SKILL.md
git commit -m "docs(v9): wire lesson_build.js engine into architect + refactor-qa skills (both locations)"
```

---

## Notes for the implementer

- **System python only** — the repo `.venv` is broken (missing encodings). Use `python3` (3.11).
- **Do not `git add -A`** — a concurrent session has uncommitted work (m04 `.donor` shells, `rollout_tracker.yaml`). Stage only the exact paths listed per task.
- **JS workflow scripts** cannot run Bash/filesystem directly — file writes and `compile_lesson.py` invocations happen INSIDE `agent()` sub-agents (the author agent), which is why Task 3's author both writes and compiles.
- **Never loosen a gate to make it pass.** If the concept_structure_gate FAILs on the proven Day-2 source, the parser/threshold is wrong — fix the gate, not the lesson.
- **Related skills:** @superpowers:subagent-driven-development, @superpowers:test-driven-development, @superpowers:verification-before-completion.
- **Deferred to Plan 2:** evidence subsystem (`evidence_compile.py`, evidence producer/judge, `evidence_index.py`), portfolio publish. Do not build them here.
