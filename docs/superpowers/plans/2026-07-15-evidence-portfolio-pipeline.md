# Frontier Evidence + Portfolio Pipeline Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Each lesson-day emits a real, frontier-facing evidence artifact (technical blog + reproducible experiment + interactive demo) assembled reuse-first, verified by an LLM judge (numbers must match a real run), and accumulated into a published portfolio.

**Architecture:** A deterministic assembler (`evidence_compile.py`) stitches reused viz + a producer's blog + an executed experiment's real outputs into a self-contained `portfolio/<module>/<day>/index.html`. A decoupled Workflow (`evidence_build.js`) runs: evidence-producer (writes blog + runnable `experiment.py`, RUNS it, reuses viz) → compile → evidence-judge (frontier-staff bar + `numbers_match`) → bounded loop → checkpoint. `evidence_index.py` builds the portfolio hub. `publish_portfolio.py` validates self-containment for a future clean-repo push.

**Tech Stack:** Python 3.11 (system), pytest, the Workflow tool, the local LLM bridge (`localhost:11211`). Reuses `sessions/viz/*.html` + lesson staff-depth prose.

**Spec:** `docs/superpowers/specs/2026-07-14-lesson-generation-orchestration-design.md` §7–8. Builds on Plan 1 (the engine).

---

## File Structure

**New:**
- `sessions/_compiler/evidence_judge.py` — LLM frontier-staff evidence judge + `numbers_match` verifier (mirrors `coverage_judge.py` patterns: bridge call, graceful fallback, `_extract_json`).
- `sessions/_compiler/evidence_compile.py` — deterministic assembler: reuse viz + blog + executed-experiment outputs → self-contained `portfolio/<module>/<day>/index.html`.
- `sessions/_compiler/evidence_index.py` — scans `portfolio/*/*/meta.json` → `portfolio/index.html` hub.
- `sessions/_compiler/workflows/evidence_build.js` — the evidence orchestrator (decoupled from `lesson_build.js`).
- `scripts/publish_portfolio.py` — self-containment validator + dry-run publish to a target repo dir.
- `sessions/_compiler/tests/test_evidence_compile.py`, `test_evidence_judge.py`, `test_evidence_index.py`.
- Generated (not hand-written): `portfolio/<module>/<day>/{index.html,blog.md,experiment.py,experiment_out.txt,meta.json,assets/*}`.

**Unchanged:** everything from Plan 1.

---

## Task 1: `evidence_judge.py` — frontier-staff LLM judge + numbers_match

**Files:** Create `sessions/_compiler/evidence_judge.py`; Test `sessions/_compiler/tests/test_evidence_judge.py`.

Mirror `coverage_judge.py` exactly (same `BRIDGE_URL`, `MODEL`, `_extract_json`, graceful fallback, never-raises). Function:

```python
def judge_evidence(blog_text, experiment_code, experiment_output, model=MODEL, timeout=90):
    """LLM judge for a day's evidence artifact, graded at the FRONTIER-STAFF bar.
    Returns {status, verdict: STRONG|OK|WEAK, numbers_match: bool, findings:[{axis,severity,why,fix}], summary}.
    axes: technical_soundness, non_triviality, reproducibility, communication, numbers_match.
    numbers_match=false if any figure/number in the blog is NOT supported by experiment_output.
    Never raises; BRIDGE_UNAVAILABLE stub on failure."""
```

- [ ] Step 1: Write failing tests — mirror `test_coverage_judge.py`: `_extract_json` reuse (import from evidence_judge or coverage_judge), `_evidence_prompt` contains blog+code+output + the 5 axes + "numbers_match", truncation cap, `judge_evidence` graceful-fallback on missing SDK (monkeypatch `builtins.__import__`).
- [ ] Step 2: Run → fail.
- [ ] Step 3: Implement (copy `coverage_judge.judge_tone` structure; system prompt = "You are a FRONTIER-LAB STAFF RESEARCHER reviewing a candidate's evidence artifact... be skeptical; a number not backed by the run output is a fabrication → numbers_match=false"). Reuse `coverage_judge._extract_json` via import.
- [ ] Step 4: Run → pass. Step 5: Commit (`feat(v9): evidence_judge — frontier-staff LLM judge + numbers_match`).

## Task 2: `evidence_compile.py` — deterministic reuse-first assembler

**Files:** Create `sessions/_compiler/evidence_compile.py`; Test `sessions/_compiler/tests/test_evidence_compile.py`.

Key functions (all deterministic, pure where possible):

```python
def viz_refs(source_md_text):
    """Return the list of `%%% viz src=...` paths referenced in a lesson source."""
    # regex: ^%%% viz\b .* src=(?:"([^"]+)"|(\S+))
def assemble(module, day, root):
    """Read portfolio/<module>/<day>/{blog.md, experiment.py, experiment_out.txt} + the lesson
    source.md's viz refs; copy referenced sessions/viz/*.html and any assets/*.png into
    portfolio/<module>/<day>/assets/; render a self-contained index.html (blog rendered to HTML,
    experiment.py in a <pre>, experiment_out.txt in a <pre>, plots as <img>, viz as <iframe>);
    write portfolio/<module>/<day>/meta.json {module,day,title,has_experiment,has_plot,viz:[...]}.
    Returns the index.html path. Relative asset links only (portable)."""
```

- [ ] Step 1: failing tests — `viz_refs` extracts src from a sample source; `assemble` on a temp fixture dir (blog.md + experiment.py + experiment_out.txt + a fake viz) writes index.html containing the blog heading, the experiment code, the output, an `<iframe>` for the viz, and writes meta.json with the right keys; assert no absolute `sessions/` links in index.html (self-contained).
- [ ] Step 2 fail → Step 3 implement (markdown→HTML: a minimal converter is fine — headings/paragraphs/code fences/`**bold**`; do NOT pull a new dependency, hand-roll like `coverage_judge._readable_text` in reverse or use a tiny renderer). Step 4 pass. Step 5 commit.

## Task 3: `evidence_index.py` — portfolio hub

**Files:** Create `sessions/_compiler/evidence_index.py`; Test `sessions/_compiler/tests/test_evidence_index.py`.

```python
def build_index(root):
    """Scan portfolio/*/*/meta.json; render portfolio/index.html: a card per day
    (title, module, links to index.html, badges for has_experiment/has_plot/viz count).
    Returns count. Deterministic, idempotent."""
```

- [ ] TDD: fixture with 2 meta.json files → index.html has 2 cards + correct links. Commit.

## Task 4: `evidence_build.js` — the evidence orchestrator

**Files:** Create `sessions/_compiler/workflows/evidence_build.js`. Model on `lesson_build.js` conventions.

Flow (`args {module, day, maxRounds}`, default maxRounds 3):
1. **Produce** (write-capable sub-agent, `agentType: 'general-purpose'`): read the passed lesson `sessions/<module>/<day>/{source.md,lesson.html}`. Write `portfolio/<module>/<day>/experiment.py` — a SMALL, real, self-contained script (system `python3`, may `savefig` a PNG into `assets/` since it is NOT a notebook) that demonstrates ONE claim from the lesson and prints its result. RUN it: `python3 portfolio/<module>/<day>/experiment.py > portfolio/<module>/<day>/experiment_out.txt 2>&1`. Write `blog.md` — a Reader-B staff-depth narrative repurposing the lesson's mechanism/failure-mode/trade-off, embedding the REAL numbers from experiment_out.txt (no fabrication). Return `{wrote, ran_ok, output_tail, claim}`.
2. **Compile** (deterministic, via the produce agent running `python3 sessions/_compiler/evidence_compile.py <module> <day>`): assemble index.html.
3. **Judge** (read-only sub-agent): run `python3 sessions/_compiler/evidence_judge.py portfolio/<module>/<day>` (add a CLI to evidence_judge) OR read blog.md + experiment_out.txt and call the judge; report verdict + numbers_match + findings.
4. **Loop**: if `numbers_match === false` OR verdict `WEAK` → back to Produce with findings, up to maxRounds. Then checkpoint report.

- [ ] Assemble the file; sanity-check structure (one meta, one top-level return, balanced braces). Commit. (Validated by the Phase-D dry-run, not pytest.)

## Task 5: `scripts/publish_portfolio.py` — self-containment validator + dry-run publish

```python
# python3 scripts/publish_portfolio.py [--to <target_repo_dir>] [--dry-run]
# 1. Validate every portfolio/**/index.html has NO absolute or ../sessions links (self-contained).
# 2. If --to given: copy portfolio/ -> <target>/ (rsync-like via shutil); if --dry-run, only list.
# 3. Print a summary; exit 1 if any self-containment violation.
```

- [ ] TDD the validator (a fixture with a bad `../sessions/` link fails; a clean one passes). Default runs `--dry-run` (no push — the clean public repo is a later user setup). Commit.

## Task 6: validate end-to-end + wire a skill note

- [ ] Run `evidence_build.js` on m04/day-01 via the Workflow tool: expect a real `experiment.py` that runs (exit 0), `experiment_out.txt` with real output, `blog.md` embedding those numbers, a self-contained `portfolio/m04-first-model-mlp/day-01-mlp-mnist/index.html`, and the judge reporting `numbers_match: true`.
- [ ] Run `evidence_index.py`; confirm `portfolio/index.html` lists the day.
- [ ] Run `publish_portfolio.py --dry-run`; confirm 0 self-containment violations.
- [ ] Add a short "Evidence Pipeline (v9, Plan 2)" note to `frontier-refactor-qa` skill (both locations) pointing at `evidence_build.js`.
- [ ] Full regression `pytest sessions/_compiler/tests/`. Commit.

## Notes
- System `python3` only. Never `git add -A` (concurrent work). Stage exact paths.
- Evidence MUST be real: numbers come from the executed `experiment.py`; the judge's `numbers_match` gate catches fabrication.
- `experiment.py` scripts MAY use `savefig` (they are standalone scripts, not notebooks — the notebook `savefig` ban does not apply).
- Reuse-first: the demo is the lesson's OWN `sessions/viz/*.html` (copied for portability), not a new viz.
- Related: @superpowers:test-driven-development, @superpowers:subagent-driven-development.
