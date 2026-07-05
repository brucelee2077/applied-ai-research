# `training/` Folder Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `sessions/` → `training/`, restructure every week into per-day-topic folders (`day-NN-<slug>/{lesson/, experiments/}`), fold the top-level `experiments/` into the matching day, and update every reference + daily-build skill so future artifacts land in the new layout.

**Architecture:** A staged migration on branch `refactor/training-folder`. All file relocations use `git mv` (history preserved) driven by small one-shot Python scripts placed in `/tmp` (repo stays clean). File moves happen first (Phases 1–3), then references are repaired (Phase 4), then generators/skills are updated (Phase 5), then everything is verified (Phase 6). The browsable site is intentionally broken between Phase 2 and Phase 4 — coherence is restored at Phase 4.

**Tech Stack:** `git mv`, system `python3` (the repo `.venv` is broken — always use `python3`, which is 3.11), plain text/HTML/JS/JSON edits.

**Spec:** `docs/superpowers/specs/2026-07-03-training-folder-refactor-design.md`

---

## Conventions used in every task

- `REPO` = `/Users/ruifengli/Desktop/applied-ai-research`
- A **day-slug** is the old lesson filename without `.html`, e.g. `day-01-jax-immutability`. It becomes the day **folder** name.
- New lesson path: `training/week-NN/<day-slug>/lesson/lesson.html`
- New log path: `training/week-NN/<day-slug>/lesson/log.md`
- New experiment path: `training/week-NN/<day-slug>/experiments/<exp-slug>/…`
- Run every `git`/`python3` command from `REPO` unless a task says otherwise.

---

## Phase 0 — Prep & safety

### Task 0: Confirm branch and stage untracked files

**Files:** none created; stages 3 untracked files so `git mv` works on them.

- [ ] **Step 1: Confirm the working branch**

Run: `cd /Users/ruifengli/Desktop/applied-ai-research && git branch --show-current`
Expected: `refactor/training-folder`
If not, run `git checkout refactor/training-folder`.

- [ ] **Step 2: Stage the three untracked files** (spec §5 — `git mv` fails on untracked paths)

```bash
cd /Users/ruifengli/Desktop/applied-ai-research
git add sessions/week-01/day-02-log.md \
        experiments/week01_jax/mlp_prng.py \
        experiments/week01_jax/results/mlp_prng_output.txt
git status --short | grep -E "day-02-log|mlp_prng"
```
Expected: all three now show as staged (`A`), not untracked (`??`). The modified-but-tracked `experiments/week01_jax/EXPERIMENT_LOG.md` and `README.md` need no action — `git mv` handles tracked files with uncommitted changes.

- [ ] **Step 3: No commit yet** — these get committed inside the moves that relocate them (Phases 1 and 3).

---

## Phase 1 — Rename `sessions/` → `training/` and fix the root redirect

### Task 1: Rename the folder and update root `index.html`

**Files:**
- Move: `sessions/` → `training/` (whole tree, via `git mv`)
- Modify: `index.html` (repo root) — 4 refs

- [ ] **Step 1: Rename the folder**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research
git mv sessions training
ls -d training && ls -d sessions 2>&1 || echo "sessions gone (good)"
```
Expected: `training` exists; `sessions` no longer exists.

- [ ] **Step 2: Update the 5 refs in root `index.html`** (4 functional + 1 comment)

Apply these five exact replacements in `/Users/ruifengli/Desktop/applied-ai-research/index.html`:
- `content="0; url=sessions/index.html"` → `content="0; url=training/index.html"`
- `<link rel="canonical" href="sessions/index.html">` → `<link rel="canonical" href="training/index.html">`
- `<a href="sessions/index.html">24-Week Curriculum →</a>` → `<a href="training/index.html">24-Week Curriculum →</a>`
- `location.replace('sessions/index.html');` → `location.replace('training/index.html');`
- In the HTML comment (line ~8): `Relative path (not /sessions/...)` → `Relative path (not /training/...)` (so the Phase 6 grep is a clean zero)

- [ ] **Step 3: Verify no `sessions/` left in root index.html**

Run: `grep -c "sessions/" /Users/ruifengli/Desktop/applied-ai-research/index.html`
Expected: `0`

- [ ] **Step 4: Commit**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research
git add index.html
git commit -m "refactor: rename sessions/ to training/ and update root redirect"
```
Note: at this checkpoint the site still works — `training/` holds flat `week-NN/day-*.html` and `training/index.html` still references them.

---

## Phase 2 — Restructure each day into `day-slug/lesson/`

### Task 2: Move lessons and logs into per-day `lesson/` folders

**Files:**
- Create: `/tmp/migrate_lessons.py`
- Move (script-driven, `git mv`): every `training/week-*/day-*.html` and `training/week-*/day-*-log.md`

- [ ] **Step 1: Create the migration script**

Create `/tmp/migrate_lessons.py`:

```python
#!/usr/bin/env python3
# One-shot: flat day html/log -> day-slug/lesson/  (run with --apply to execute)
import os, re, glob, subprocess, sys
BASE = "/Users/ruifengli/Desktop/applied-ai-research/training"
DRY = "--apply" not in sys.argv

def gitmv(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(("DRY  " if DRY else "MOVE ") + f"{os.path.relpath(src,BASE)} -> {os.path.relpath(dst,BASE)}")
    if not DRY:
        subprocess.run(["git", "mv", src, dst], check=True, cwd=BASE)

def main():
    moved_html = moved_log = 0
    # 1) lessons: week-NN/day-NN-slug.html -> week-NN/day-NN-slug/lesson/lesson.html
    for html in sorted(glob.glob(os.path.join(BASE, "week-*", "day-*.html"))):
        slug = os.path.basename(html)[:-5]            # day-01-jax-immutability
        wkdir = os.path.dirname(html)
        gitmv(html, os.path.join(wkdir, slug, "lesson", "lesson.html")); moved_html += 1
    # 2) logs: week-NN/day-NN-log.md -> week-NN/day-NN-*/lesson/log.md
    for log in sorted(glob.glob(os.path.join(BASE, "week-*", "day-*-log.md"))):
        m = re.match(r"(day-\d{2})-log\.md$", os.path.basename(log))
        if not m:
            print("!! unexpected log name:", log); continue
        wkdir = os.path.dirname(log)
        folders = [d for d in glob.glob(os.path.join(wkdir, m.group(1) + "-*")) if os.path.isdir(d)]
        if len(folders) != 1:
            print(f"!! {os.path.relpath(log,BASE)}: expected 1 folder {m.group(1)}-*, found {len(folders)}"); continue
        gitmv(log, os.path.join(folders[0], "lesson", "log.md")); moved_log += 1
    print(f"\n{'DRY-RUN ' if DRY else ''}lessons={moved_html} logs={moved_log}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run and eyeball**

Run: `python3 /tmp/migrate_lessons.py`
Expected: ~110 `DRY … -> …/lesson/lesson.html` lines, a small number of log lines (at least `day-02-log.md` in week-01), and a final `DRY-RUN lessons=110 logs=N`. No `!!` warnings. If any `!!` appears, stop and investigate before applying.

- [ ] **Step 3: Apply**

Run: `python3 /tmp/migrate_lessons.py --apply`
Expected: `MOVE …` lines and `lessons=110 logs=N`. No `!!` lines, no `git mv` errors.

- [ ] **Step 4: Verify the new shape**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research
ls training/week-01/day-01-jax-immutability/lesson/lesson.html
ls training/week-01/day-02-prng-keys/lesson/log.md
find training/week-* -maxdepth 1 -name "day-*.html" | head    # expect: (empty)
```
Expected: the two `ls` succeed; the `find` prints nothing (no flat day HTMLs remain).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: restructure each day into day-slug/lesson/{lesson.html,log.md}"
```

---

## Phase 3 — Fold `experiments/` into each day's `experiments/`

### Task 3: Move experiment folders to their day (spec §5.2)

**Files:**
- Move (`git mv`): the 8 experiment folders/files below
- Split: `experiments/week01_jax/README.md` and `EXPERIMENT_LOG.md` at their `---` divider
- Delete: empty `experiments/week07_jax/`; then the now-empty top-level `experiments/`

- [ ] **Step 1: Move the straightforward experiments**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research

# week-01 day-01 (immutability)
mkdir -p training/week-01/day-01-jax-immutability/experiments/immutability-proof/results
git mv experiments/week01_jax/immutability_proof.py \
       training/week-01/day-01-jax-immutability/experiments/immutability-proof/immutability_proof.py
git mv experiments/week01_jax/results/run_output.txt \
       training/week-01/day-01-jax-immutability/experiments/immutability-proof/results/run_output.txt

# week-01 day-02 (mlp prng)
mkdir -p training/week-01/day-02-prng-keys/experiments/mlp-prng/results
git mv experiments/week01_jax/mlp_prng.py \
       training/week-01/day-02-prng-keys/experiments/mlp-prng/mlp_prng.py
git mv experiments/week01_jax/results/mlp_prng_output.txt \
       training/week-01/day-02-prng-keys/experiments/mlp-prng/results/mlp_prng_output.txt

# week-08 day-01 (env check)
mkdir -p training/week-08/day-01-comprehensive-profiling/experiments
git mv experiments/week08_jax/env_check \
       training/week-08/day-01-comprehensive-profiling/experiments/env-check

# week-08 day-02 (systems review synthesis writeup)
mkdir -p training/week-08/day-02-systems-review/experiments/systems-review-synthesis
git mv experiments/week08_jax/systems_review_synthesis.md \
       training/week-08/day-02-systems-review/experiments/systems-review-synthesis/systems_review_synthesis.md

# week-14 day-03 (worker overlapping): move the 3 files into one experiment folder
mkdir -p training/week-14/day-03-worker-overlapping/experiments/thunderkittens-overlap
git mv experiments/week14_thunderkittens/annotate_lcf_template.md \
       training/week-14/day-03-worker-overlapping/experiments/thunderkittens-overlap/annotate_lcf_template.md
git mv experiments/week14_thunderkittens/overlap_sim.py \
       training/week-14/day-03-worker-overlapping/experiments/thunderkittens-overlap/overlap_sim.py
git mv experiments/week14_thunderkittens/results.md \
       training/week-14/day-03-worker-overlapping/experiments/thunderkittens-overlap/results.md

# week-16 day-02 (snapkv), day-03 (gqa), day-05 (ring attention)
mkdir -p training/week-16/day-02-cache-eviction-snapkv/experiments
git mv experiments/week16_jax/day2_snapkv_eviction \
       training/week-16/day-02-cache-eviction-snapkv/experiments/snapkv-eviction
mkdir -p training/week-16/day-03-gqa-mqa/experiments
git mv experiments/week16_jax/gqa_parameter_surgery \
       training/week-16/day-03-gqa-mqa/experiments/gqa-parameter-surgery
mkdir -p training/week-16/day-05-ring-attention/experiments
git mv experiments/week16_jax/ring_attention \
       training/week-16/day-05-ring-attention/experiments/ring-attention
```
Expected: no `git mv` errors.

- [ ] **Step 2: Split the shared week-01 `README.md`**

`experiments/week01_jax/README.md` has two sections separated by a line containing only `---` (immutability on top, MLP-PRNG below). Create two files:
- `training/week-01/day-01-jax-immutability/experiments/immutability-proof/README.md` = everything **above** the `---` divider (the "JAX immutability proof (Week 1, Day 1)" section).
- `training/week-01/day-02-prng-keys/experiments/mlp-prng/README.md` = everything **below** the `---` divider (the "MLP weights from split PRNG keys (Week 1, Day 2)" section).

Then remove the original: `git rm experiments/week01_jax/README.md`

- [ ] **Step 3: Split the shared week-01 `EXPERIMENT_LOG.md`** the same way

`experiments/week01_jax/EXPERIMENT_LOG.md` also splits at its `---` divider:
- top section → `training/week-01/day-01-jax-immutability/experiments/immutability-proof/EXPERIMENT_LOG.md`
- bottom section → `training/week-01/day-02-prng-keys/experiments/mlp-prng/EXPERIMENT_LOG.md`

Then: `git rm experiments/week01_jax/EXPERIMENT_LOG.md`

- [ ] **Step 4: Delete the empty week-07 placeholder and remove the top-level `experiments/`**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research
rm -rf experiments/week07_jax          # empty untracked placeholder (spec §5.2)
find experiments -type f                # expect: (empty — nothing prints)
rmdir experiments/week01_jax/results experiments/week01_jax experiments/week08_jax experiments/week16_jax experiments/week14_thunderkittens 2>/dev/null
rmdir experiments 2>/dev/null
ls -d experiments 2>&1 || echo "top-level experiments/ gone (good)"
```
Expected: `find` prints nothing; final line prints "top-level experiments/ gone (good)".

- [ ] **Step 5: Verify module experiments untouched (spec §3, §9-5)**

Run: `ls -d 00-neural-networks/experiments 2>/dev/null && echo "module experiments intact"`
Expected: prints the path and "module experiments intact" (this dir must NOT have been moved).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: fold experiments/ into per-day training/<day>/experiments/"
```

---

## Phase 4 — Repair functional references

### Task 4: Rewrite `progress.json` page paths

**Files:** Modify `training/progress.json` (110 `page` values)

- [ ] **Step 1: Rewrite via script**

```bash
python3 - <<'PY'
import re
p = "/Users/ruifengli/Desktop/applied-ai-research/training/progress.json"
t = open(p, encoding="utf-8").read()
t2, n = re.subn(
    r'"page":\s*"sessions/(week-\d{2})/(day-[a-z0-9-]+)\.html"',
    lambda m: f'"page": "training/{m.group(1)}/{m.group(2)}/lesson/lesson.html"',
    t)
open(p, "w", encoding="utf-8").write(t2)
print("rewrote", n, "page paths")
PY
```
Expected: `rewrote 110 page paths`.

- [ ] **Step 2: Verify no stale refs and valid JSON**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research
grep -c '"page": "sessions/' training/progress.json          # expect 0
python3 -c "import json; json.load(open('training/progress.json')); print('valid json')"
```
Expected: `0`, then `valid json`.

### Task 5: Update `wire_index.py`

**Files:** Modify `training/wire_index.py`

- [ ] **Step 1: Update the lesson glob (line ~19)**

Replace:
```python
    hits = sorted(glob.glob(os.path.join(BASE, f"week-{wk}", f"day-{dy}-*.html")))
```
with:
```python
    hits = sorted(glob.glob(os.path.join(BASE, f"week-{wk}", f"day-{dy}-*", "lesson", "lesson.html")))
```
The `rel = os.path.relpath(path, BASE)` on the next line then yields `week-NN/day-slug/lesson/lesson.html` automatically — no other change needed. (Docstring `sessions/` mentions are swept to `training/` in Task 9b.)

### Task 6: Update `lesson_audit.py`

**Files:** Modify `training/lesson_audit.py`

- [ ] **Step 1: Update the path built for each manifest entry (line ~95)**

Replace:
```python
            path = os.path.join(BASE, f"week-{week:02d}", f"{slug}.html")
```
with:
```python
            path = os.path.join(BASE, f"week-{week:02d}", slug, "lesson", "lesson.html")
```

- [ ] **Step 2: Update the reported relative path (line ~97)**

Replace:
```python
            rel = f"week-{week:02d}/{slug}.html"
```
with:
```python
            rel = f"week-{week:02d}/{slug}/lesson/lesson.html"
```

### Task 7: Update `coverage_audit.py`

**Files:** Modify `training/coverage_audit.py`

- [ ] **Step 1: Point at `training/` (line ~18)**

Replace `SESSIONS = ROOT / "sessions"` with `SESSIONS = ROOT / "training"`.

- [ ] **Step 2: Update the lesson glob (line ~45)**

Replace:
```python
    for html in sorted(SESSIONS.glob("week-*/day-*.html")):
```
with:
```python
    for html in sorted(SESSIONS.glob("week-*/day-*/lesson/lesson.html")):
```
(The `ROOT` computation still resolves to the repo root after the move; `COMPANION` is unaffected. The user-facing `sessions/` strings in the docstring/print on lines ~6, ~9, ~67 are swept to `training/` in Task 9b.)

### Task 8: Update `inject_quiz.py` and `inject_viz.py`

**Files:** Modify `training/inject_quiz.py`, `training/inject_viz.py`

- [ ] **Step 1: `inject_quiz.py` — target path (line ~25)**

Replace:
```python
    return os.path.join(BASE, m.group(1), m.group(2)+".html") if m else None
```
with:
```python
    return os.path.join(BASE, m.group(1), m.group(2), "lesson", "lesson.html") if m else None
```

- [ ] **Step 2: `inject_viz.py` — target path (line ~28)**

Replace:
```python
    return os.path.join(BASE, m.group(1), m.group(2)+".html")
```
with:
```python
    return os.path.join(BASE, m.group(1), m.group(2), "lesson", "lesson.html")
```
(Step-cache JSON filenames in `_quiz_steps/`/`_viz_steps/` stay unchanged — they are keys, not paths.)

### Task 8b: Update `apply_scroll_format.py` (spec §4 — silent breakage otherwise)

**Files:** Modify `training/apply_scroll_format.py`

This one-off formatter has no literal `sessions/` string, so the Phase 6 grep will NOT catch it — but its `TEMPLATE` path and glob both break after the move (template file gone; glob matches zero files). Fix both.

- [ ] **Step 1: Update the template path (line ~22)**

Replace:
```python
TEMPLATE = os.path.join(BASE, "week-01", "day-03-vmap.html")
```
with:
```python
TEMPLATE = os.path.join(BASE, "week-01", "day-03-vmap", "lesson", "lesson.html")
```

- [ ] **Step 2: Update the lesson glob (line ~104)**

Replace:
```python
    files = sorted(glob.glob(os.path.join(BASE, "week-*", "day-*.html")))
```
with:
```python
    files = sorted(glob.glob(os.path.join(BASE, "week-*", "day-*", "lesson", "lesson.html")))
```
(Docstring `Usage:` line ~17 `sessions/apply_scroll_format.py` is swept to `training/` in Task 9b.)

- [ ] **Step 3: Smoke-check it loads its template**

Run: `cd /Users/ruifengli/Desktop/applied-ai-research/training && python3 -c "import apply_scroll_format" 2>&1 | head -3; python3 apply_scroll_format.py --dry week-01/day-03-vmap/lesson/lesson.html`
Expected: no `TEMPLATE`-not-found / markers-not-found crash; prints an `OK`/`SKIP` line (SKIP is fine — the lesson is already new-format).

### Task 9: Update the workflow `.js` scripts

**Files:** Modify `training/_gapfill_workflow.js`, `training/_quiz_workflow.js`, `training/_visualize_workflow.js`

- [ ] **Step 1: Rewrite path refs via script**

```bash
python3 - <<'PY'
import re, os
base = "/Users/ruifengli/Desktop/applied-ai-research/training"
for name in ("_gapfill_workflow.js", "_quiz_workflow.js", "_visualize_workflow.js"):
    f = os.path.join(base, name)
    t = open(f, encoding="utf-8").read(); orig = t
    t = t.replace("sessions/_quiz_steps", "training/_quiz_steps").replace("sessions/_viz_steps", "training/_viz_steps")
    # ONLY rewrite sessions/-prefixed lesson paths -> nested lesson.html (anchored on the
    # prefix so we never touch bare/relative refs like ../week-06/... which would become
    # a malformed ../training/... — those nav links are a Phase 7 follow-up).
    t = re.sub(r'sessions/(week-\d{2})/(day-[a-z0-9-]+)\.html',
               r'training/\1/\2/lesson/lesson.html', t)
    t = t.replace("sessions/", "training/")   # any remaining sessions/ refs
    if t != orig:
        open(f, "w", encoding="utf-8").write(t); print("updated", name)
PY
```
Expected: three `updated …` lines.

- [ ] **Step 2: Verify no stale refs**

Run: `grep -rl "sessions/" /Users/ruifengli/Desktop/applied-ai-research/training/*.js || echo "clean"`
Expected: `clean`.

- [ ] **Step 3: Commit** (the workflow-js edits ride into the Phase 4 commit in Task 9b)

### Task 9b: Sweep residual `sessions/` refs in tooling + data caches, then commit Phase 4

**Files:** Modify the six `training/*.py` scripts (docstrings/comments/print strings only) and `training/_recover_specs.json`

These carry non-functional `sessions/` mentions plus one regenerated data cache — none affect behavior, but the Phase 6 gate greps them, so clean them deterministically. **This is mandatory and supersedes the "swept in Task 9b" notes in Tasks 5–8b.**

- [ ] **Step 1: Blanket-replace the prefix in the .py scripts** (their functional paths use `BASE`/`ROOT`, never a literal `sessions/`, so this only touches docstrings/comments/prints)

```bash
cd /Users/ruifengli/Desktop/applied-ai-research/training
python3 - <<'PY'
for f in ("wire_index.py","lesson_audit.py","coverage_audit.py","inject_quiz.py","inject_viz.py","apply_scroll_format.py"):
    t = open(f, encoding="utf-8").read(); n = t.count("sessions/")
    if n:
        open(f, "w", encoding="utf-8").write(t.replace("sessions/", "training/")); print(f"{f}: {n} -> 0")
PY
```
Expected: a line per file that had refs (e.g. `wire_index.py: 3 -> 0`, `lesson_audit.py: 2 -> 0`, etc.).

- [ ] **Step 2: Rewrite the recovery-cache paths** (`_recover_specs.json`, 25 refs; `_recover_set.json` is regenerated clean by `lesson_audit.py`, and `_visualize_targets.json` has none)

```bash
python3 - <<'PY'
import re
p = "/Users/ruifengli/Desktop/applied-ai-research/training/_recover_specs.json"
t = open(p, encoding="utf-8").read()
t2, n = re.subn(r'sessions/(week-\d{2})/(day-[a-z0-9-]+)\.html',
                r'training/\1/\2/lesson/lesson.html', t)
open(p, "w", encoding="utf-8").write(t2); print("rewrote", n, "paths in _recover_specs.json")
PY
```
Expected: `rewrote 25 paths in _recover_specs.json`.

- [ ] **Step 3: Confirm training/ tooling is clean**

Run: `grep -rn "sessions/" /Users/ruifengli/Desktop/applied-ai-research/training/*.py /Users/ruifengli/Desktop/applied-ai-research/training/*.js /Users/ruifengli/Desktop/applied-ai-research/training/*.json || echo "training tooling clean"`
Expected: `training tooling clean`.

- [ ] **Step 4: Commit Phase 4**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research
git add -A
git commit -m "refactor: repair progress.json, build scripts, workflow + recovery-cache paths for training/ layout"
```

---

## Phase 5 — Update skills & generators (future builds)

### Task 10: `frontier-session-coach` skill (active copy)

**Files:** Modify `.claude/skills/frontier-session-coach/SKILL.md` **and** `.claude/skills/frontier-session-coach/SESSION_TEMPLATE.md`

- [ ] **Step 1: Apply these exact replacements**

- `` `sessions/progress.json` `` → `` `training/progress.json` `` (appears ~3×: lines ~27, ~36, ~138 context — use replace-all)
- `` `sessions/week-01/day-01-jax-immutability.html` — the canonical format`` → `` `training/week-01/day-01-jax-immutability/lesson/lesson.html` — the canonical format``
- In the output block (line ~70): `sessions/week-<NN>/day-<NN>-<slug>.html` → `training/week-<NN>/day-<NN>-<slug>/lesson/lesson.html`
- `sessions/viz/<name>.html` → `training/viz/<name>.html`
- `` `sessions/viz/` visualization`` → `` `training/viz/` visualization``
- `` `sessions/week-<NN>/day-<NN>-log.md` `` → `` `training/week-<NN>/<day-slug>/lesson/log.md` ``
- In the progress.json append example: `"page":"sessions/week-03/day-03-<slug>.html"` → `"page":"training/week-03/day-03-<slug>/lesson/lesson.html"`
- `**Update `sessions/index.html`**` → `**Update `training/index.html`**`
- The WEEKS example comment: `'week-03/day-03-kv-cache.html'` → `'week-03/day-03-kv-cache/lesson/lesson.html'`
- `open sessions/...html` → `open training/...html`; `open sessions/index.html` → `open training/index.html`
- Any remaining `sessions/index.html` mention (line ~58) → `training/index.html`

- [ ] **Step 2: Update `SESSION_TEMPLATE.md`** (same skill dir)

- Line ~9: `sessions/week-01/day-01-jax-immutability.html` → `training/week-01/day-01-jax-immutability/lesson/lesson.html`
- Line ~36: `` advance `sessions/progress.json` `` → `` advance `training/progress.json` ``

- [ ] **Step 3: Verify the whole skill dir is clean**

Run: `grep -rc "sessions/" .claude/skills/frontier-session-coach/`
Expected: every file reports `0`.

### Task 10b: `frontier-review-quiz` skill (advances the ledger — spec §7)

**Files:** Modify `.claude/skills/frontier-review-quiz/SKILL.md`

- [ ] **Step 1:** Line ~46 `` update `sessions/progress.json`: `` → `` update `training/progress.json`: `` (this skill marks a day complete and advances `cursor`; leaving the old path breaks that write and trips the Phase 6 grep).

### Task 11: `frontier-experiment-lab` SKILL.md (active copy)

**Files:** Modify `.claude/skills/frontier-experiment-lab/SKILL.md`

- [ ] **Step 1: Replace the "Output files" block (lines ~17–23)**

Replace:
```text
experiments/<experiment-slug>/README.md
experiments/<experiment-slug>/src/
experiments/<experiment-slug>/tests/
experiments/<experiment-slug>/results/
experiments/<experiment-slug>/EXPERIMENT_LOG.md
```
with:
```text
training/week-<NN>/<day-slug>/experiments/<experiment-slug>/README.md
training/week-<NN>/<day-slug>/experiments/<experiment-slug>/src/
training/week-<NN>/<day-slug>/experiments/<experiment-slug>/tests/
training/week-<NN>/<day-slug>/experiments/<experiment-slug>/results/
training/week-<NN>/<day-slug>/experiments/<experiment-slug>/EXPERIMENT_LOG.md
```

- [ ] **Step 2: Add a day-context note** right under that block:

```text
> The `week-<NN>` and `<day-slug>` come from the quest that triggered this (the
> PRODUCE step of `frontier-session-coach` names the exact path). If run standalone,
> ask which curriculum day the experiment belongs to before writing.
```

### Task 12: `frontier-paper-course` SKILL.md (active copy)

**Files:** Modify `.claude/skills/frontier-paper-course/SKILL.md`

- [ ] **Step 1: Day-scope only the experiment paths** (leave `notes/papers/` and `courseware/` as-is per spec §3)

- `experiments/<paper-slug>/README.md` → `training/week-<NN>/<day-slug>/experiments/<paper-slug>/README.md`
- `experiments/<paper-slug>/src/` → `training/week-<NN>/<day-slug>/experiments/<paper-slug>/src/`
- `experiments/<paper-slug>/tests/` → `training/week-<NN>/<day-slug>/experiments/<paper-slug>/tests/`

### Task 13: `phase1-lessons.js` workflow

**Files:** Modify `.claude/workflows/phase1-lessons.js`

- [ ] **Step 1: Rewrite via script**

```bash
python3 - <<'PY'
import re
f = "/Users/ruifengli/Desktop/applied-ai-research/.claude/workflows/phase1-lessons.js"
t = open(f, encoding="utf-8").read()
t = re.sub(r'sessions/(week-\d{2})/(day-[a-z0-9-]+)\.html',
           r'training/\1/\2/lesson/lesson.html', t)
# two instructional experiment-path strings -> day-scoped convention
t = t.replace("experiments/week01_jax/mlp_jit.py",
              "training/week-01/day-04-jit/experiments/mlp-jit/mlp_jit.py")
t = t.replace("experiments/week01_jax/mlp_flax_optax.py",
              "training/week-01/day-05-flax-optax/experiments/mlp-flax-optax/mlp_flax_optax.py")
open(f, "w").write(t)
print("phase1-lessons.js updated")
PY
grep -c "sessions/week" /Users/ruifengli/Desktop/applied-ai-research/.claude/workflows/phase1-lessons.js
```
Expected: `phase1-lessons.js updated`, then `0`.

### Task 14: `frontier_lab_claude_skills/` bundle (convention wording — spec D2/§7)

**Files:** Modify `frontier_lab_claude_skills/CLAUDE.md`, `README.md`, `EXAMPLE_PROMPTS.md`, and its `skills/frontier-experiment-lab/SKILL.md`, `skills/frontier-paper-course/SKILL.md`, `skills/frontier-session-coach/SKILL.md`

- [ ] **Step 1: `CLAUDE.md` — replace the "## Output directories" list**

Replace the list under `## Output directories` with:
```markdown
## Output directories
- `notes/`
- `training/week-<NN>/<day-slug>/lesson/` — daily learning session (HTML + log)
- `training/week-<NN>/<day-slug>/experiments/<slug>/` — that day's experiments
- `courseware/`
- `quizzes/`
- `portfolio/`
```

- [ ] **Step 2: `README.md` (lines ~37–39)** — replace the `sessions/` and `experiments/` bullets with the `training/week-<NN>/<day-slug>/lesson/` and `.../experiments/<slug>/` convention; leave `courseware/`.

- [ ] **Step 3: `EXAMPLE_PROMPTS.md`** — replace **both** occurrences (lines ~5 and ~30) of `sessions/day-001-kv-cache.md` with `training/week-<NN>/<day-slug>/lesson/lesson.html`; leave the `courseware/…` example paths.

- [ ] **Step 4: Bundle skill files** — apply the same day-scoped experiment-path edit as Tasks 11–12 to `frontier_lab_claude_skills/skills/frontier-experiment-lab/SKILL.md` (lines ~14–18) and `frontier-paper-course/SKILL.md` (lines ~21, ~29–30); and change `frontier-session-coach/SKILL.md` line ~67 `sessions/day-001.md` → `training/week-<NN>/<day-slug>/lesson/lesson.html`.

- [ ] **Step 5: Commit Phase 5**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research
git add -A
git commit -m "refactor: update daily-build skills + bundle to write into training/<day>/{lesson,experiments}"
```

---

## Phase 6 — Verification & finalize

### Task 15: Run the audit scripts and confirm the site resolves

**Files:** none modified (may re-write `training/index.html` via `wire_index.py`, and `training/_recover_set.json`)

- [ ] **Step 1: `wire_index.py` re-wires index.html from the new paths**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research/training
python3 wire_index.py
```
Expected: `wired <N> entries; <M> ids had no file yet` with no `! could not splice` lines for days that exist (M covers only never-generated days).

- [ ] **Step 2: `lesson_audit.py` finds every generated lesson at its new path**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research/training
python3 lesson_audit.py; echo "exit=$?"
```
Expected: the `MISSING` bucket contains only days that were never generated (e.g. week-17 d02+, week-21 d02+, week-23 d05+) — **no** day that existed under `sessions/` should now be MISSING. Days like `w01-d01 … w16-d06` that had files must not appear as MISSING.

- [ ] **Step 3: `coverage_audit.py` still reads the lessons**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research
python3 training/coverage_audit.py; echo "exit=$?"
```
Expected: it prints "Cited in >=1 lesson" with a non-zero count (it must find the lesson files at `week-*/day-*/lesson/lesson.html`, not abort with "No lesson files found").

- [ ] **Step 4: Grep for stale functional refs (spec §9-4, widened)**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research
# Functional code/config + skill/bundle docs — must be clean.
# Nested lesson HTML under training/week-*/ is deliberately NOT grepped here: its inline
# teaching-text sessions/ refs are the Phase 7 / spec-D5 deferred follow-up.
grep -rn "sessions/" index.html .claude/ frontier_lab_claude_skills/ \
    training/*.py training/*.js training/*.json \
  || echo "no stale functional sessions/ refs"
```
Expected: `no stale functional sessions/ refs`. (Scoping to `training/*.py|*.js|*.json` covers the root-level tooling but excludes the `training/week-*/**/lesson.html` teaching text on purpose.)

- [ ] **Step 5: Confirm structure invariants**

```bash
cd /Users/ruifengli/Desktop/applied-ai-research
ls -d experiments 2>&1 | grep -q "No such" && echo "top-level experiments/ gone: OK"
ls -d 00-neural-networks/experiments >/dev/null 2>&1 && echo "module experiments intact: OK"
open training/index.html    # visually confirm week/day links resolve to lesson/lesson.html
```
Expected: both "OK" lines; the dashboard opens and links resolve.

- [ ] **Step 6: Commit any wire_index/audit output**

```bash
git add -A
git commit -m "refactor: re-wire training/index.html and record audit state" || echo "nothing to commit"
```

### Task 16: Update the deploy memory note (follow-up bookkeeping)

**Files:** Modify `/Users/ruifengli/.claude/projects/-Users-ruifengli-Desktop-applied-ai-research/memory/github-pages-deploy.md`

- [ ] **Step 1:** Update the note so it records that the root `index.html` now redirects to `training/` (not `sessions/`), and the public curriculum path is `.../applied-ai-research/training/`. (This memory file is outside the repo — no commit needed.)

---

## Phase 7 — Deferred (out of scope for this plan; spec D5)

Not executed here — a separate scripted follow-up:

- Rewrite the ~100 **inline instructional path strings** inside lesson HTMLs (teaching text like "create `experiments/week01_jax/mlp_prng.py`" or "write to `sessions/week-01/day-04-log.md`") to the new `training/…/lesson/` and `…/experiments/<slug>/` convention. These are non-breaking (they render as prose) and the old experiment naming is irregular, so a careful scripted pass with per-file review is warranted rather than a blanket replace.
- **Rework the prev/next nav-link logic in the workflow generators** (`training/_*.js`, `.claude/workflows/phase1-lessons.js`). Their bare/relative sibling links (`../week-NN/day-slug.html`, or bare `day-XX.html`) are intentionally left untouched by Tasks 9/13 (a naive rewrite would produce a malformed `../training/…`). Under the nested layout a sibling lesson is now two levels up (e.g. `../../day-02-prng-keys/lesson/lesson.html`), so the generators' link construction needs a logic change, not a string swap. Only affects *future*-generated lessons' nav; existing lessons are unaffected.

---

## Rollback

Everything is on branch `refactor/training-folder`. To abandon: `git checkout main` and delete the branch, or `git reset --hard <pre-refactor-commit>`. No files are deleted outside the `git mv`/`git rm` operations, all of which are recoverable from history.
