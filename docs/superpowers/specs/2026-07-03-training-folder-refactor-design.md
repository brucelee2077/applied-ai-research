# Design: `training/` folder refactor

**Date:** 2026-07-03
**Status:** Approved (design) — pending implementation plan
**Owner:** ruifengli

---

## 1. Problem

The 24-week Frontier readiness study writes daily artifacts into two disconnected top-level folders with different naming schemes:

- `sessions/week-NN/` — daily interactive lessons as flat files (`day-01-jax-immutability.html`) plus sparse `day-NN-log.md` reflections. Built by the `frontier-session-coach` skill.
- `experiments/weekNN_<tech>/` — hands-on code (`week01_jax`, `week14_thunderkittens`). Different naming (`week01_jax` vs `week-01`), only 5 of 24 weeks populated, sometimes nested by day, sometimes flat. Built by `frontier-experiment-lab`.

Three problems to fix:

1. The name `sessions/` is unwanted.
2. `experiments/` lives outside `sessions/`; the two halves of a day's work are not co-located.
3. Artifacts are not organized by day.

## 2. Goals

- One top-level folder, `training/`, holding all daily study output.
- First level inside is the **week**; second level is the **day (a training session for one topic)**.
- Inside each day-topic, two peer parts: the **learning session** (`lesson/`) and the **experiments** (`experiments/`).
- The daily build skills and the progress "schedule" write into the new layout so future days follow it automatically.
- Preserve git history (use `git mv`).

## 3. Non-goals

- No change to the foundational module dirs (`00-neural-networks/` … `10-reinforcement-learning/`), `ML Design/`, or `genAI design/`.
- No change to module-internal experiment folders (e.g. `00-neural-networks/experiments/`) — **only the top-level `experiments/` moves**.
- No change to `courseware/`, `notes/`, `portfolio/` conventions (separate artifact types produced by other skills).
- No re-authoring of lesson *content* (only path references, and only as an optional follow-up — see §8).

## 4. Target structure

```
training/
├── index.html                 ← dashboard (moved as-is)
├── progress.json              ← ledger (110 entries; nominal 24×6=144); every `page` path rewritten
├── viz/                       ← shared prebuilt visualizations (moved as-is)
├── wire_index.py, inject_quiz.py, inject_viz.py,
│   lesson_audit.py, coverage_audit.py, apply_scroll_format.py   ← build scripts (moved; path logic updated)
├── _quiz_steps/  _viz_steps/  ← step caches (moved as-is)
├── _gapfill_workflow.js  _quiz_workflow.js  _visualize_workflow.js  ← workflow scripts (moved; paths updated)
└── week-01/                            ← WEEK (top level inside training/)
    ├── day-01-jax-immutability/        ← DAY = one training session for a topic
    │   ├── lesson/                      ← 1. the learning session
    │   │   └── lesson.html
    │   └── experiments/                 ← 2. the experiments (one subfolder each)
    │       └── immutability-proof/
    │           ├── immutability_proof.py
    │           └── results/
    ├── day-02-prng-keys/
    │   ├── lesson/
    │   │   ├── lesson.html
    │   │   └── log.md                    ← reflection log lives with the lesson
    │   └── experiments/
    │       └── mlp-prng/
    │           ├── mlp_prng.py
    │           └── results/
    └── day-03-vmap/ … day-06-vit-capstone/
        └── lesson/                       ← days with no code: lesson/ only, no experiments/
            └── lesson.html
```

**Naming rules**
- Week folder: `week-NN` (two digits), matching the existing session scheme.
- Day folder: `day-NN-<slug>` — the slug is the existing lesson slug (`day-01-jax-immutability`).
- `lesson/` holds `lesson.html` and, when one exists, `log.md`.
- `experiments/` holds one subfolder per experiment, named with a kebab-case experiment slug; contents follow the `frontier-experiment-lab` template (`README.md`, `src/`, `tests/`, `results/`, `EXPERIMENT_LOG.md`) or a flatter shape for tiny experiments.
- Days with no experiment have `lesson/` only (git does not track empty dirs; no empty `experiments/`).

## 5. Migration mapping

All existing files move via `git mv` (history preserved). **Exception — untracked files:** `experiments/week01_jax/mlp_prng.py`, `experiments/week01_jax/results/mlp_prng_output.txt`, and `sessions/week-01/day-02-log.md` are currently untracked, so `git mv` will fail on them. The plan must `git add` these first, or move them with plain `mv`. (Also note `experiments/week01_jax/EXPERIMENT_LOG.md` and `README.md` have uncommitted modifications — tracked, so `git mv` works.)

### 5.1 Lessons and logs (all 22 populated weeks)

- `sessions/week-NN/day-NN-<slug>.html` → `training/week-NN/day-NN-<slug>/lesson/lesson.html`
- `sessions/week-NN/day-NN-log.md` → `training/week-NN/day-NN-<slug>/lesson/log.md`

Weeks present: 01–17, 19–21, 23–24 (missing 18, 22). Partial weeks (08, 12, 17, 21, 23, 24) migrate whatever days exist.

### 5.2 Experiments (concrete, fully resolved)

| Old path | New location |
|---|---|
| `experiments/week01_jax/immutability_proof.py` (+ `results/run_output.txt`) | `training/week-01/day-01-jax-immutability/experiments/immutability-proof/` |
| `experiments/week01_jax/mlp_prng.py` (+ `results/mlp_prng_output.txt`) | `training/week-01/day-02-prng-keys/experiments/mlp-prng/` |
| `experiments/week01_jax/README.md`, `EXPERIMENT_LOG.md` | Split by topic into the two experiment folders above (they document both experiments). |
| `experiments/week08_jax/env_check/**` | `training/week-08/day-01-comprehensive-profiling/experiments/env-check/` |
| `experiments/week08_jax/systems_review_synthesis.md` | `training/week-08/day-02-systems-review/experiments/systems-review-synthesis/` (writeup) |
| `experiments/week14_thunderkittens/**` (`annotate_lcf_template.md`, `overlap_sim.py`, `results.md`) | `training/week-14/day-03-worker-overlapping/experiments/` |
| `experiments/week16_jax/day2_snapkv_eviction/**` | `training/week-16/day-02-cache-eviction-snapkv/experiments/snapkv-eviction/` |
| `experiments/week16_jax/gqa_parameter_surgery/**` | `training/week-16/day-03-gqa-mqa/experiments/gqa-parameter-surgery/` |
| `experiments/week16_jax/ring_attention/**` | `training/week-16/day-05-ring-attention/experiments/ring-attention/` |

Day assignments are derived from the lesson slug that each experiment's topic matches (verified against `frontier_ai_24_week_link_companion.md` and the path each day's lesson HTML already cites).

**Shape notes:**
- `gqa_parameter_surgery/` is a single file (`gqa_checkpoint_surgery.py`, no `results/`) — it migrates as a flat experiment folder, not the full `README/src/tests/results` template.
- **`experiments/week07_jax/`** holds only an empty, untracked placeholder `thu_moe_alltoall_overhead/` (nothing to migrate). Disposition: **delete it**. Future week-07 experiments will land under `training/week-07/day-04-distributed-moe-routing/experiments/`.
- After all moves, the top-level `experiments/` folder must be **empty and removed** (verified in §9).

## 6. Reference updates (functional — breaks if skipped)

| Target | Change |
|---|---|
| root `index.html` | 4 refs `sessions/index.html` → `training/index.html` |
| `training/progress.json` | all `page` values (currently 110) → `training/week-NN/<day-slug>/lesson/lesson.html` |
| `wire_index.py`, `lesson_audit.py`, `coverage_audit.py` | `BASE` dir `sessions` → `training`; globs `week-*/day-*.html` → `week-*/day-*/lesson/lesson.html` |
| `inject_quiz.py`, `inject_viz.py` | resolve `_{quiz,viz}_steps/week-NN_day-<slug>_html.json` to the nested `week-NN/<day-slug>/lesson/lesson.html` target |
| `_gapfill_workflow.js`, `_quiz_workflow.js`, `_visualize_workflow.js` | relative day paths → `week-NN/<day-slug>/lesson/lesson.html` |

Step-cache JSON *filenames* in `_quiz_steps/` / `_viz_steps/` stay as-is (they are keys); only the scripts' target-path resolution changes.

## 7. Skill & "schedule" changes (future builds)

| Target | Change |
|---|---|
| `.claude/skills/frontier-session-coach/SKILL.md` | output → `training/week-NN/<day-slug>/lesson/lesson.html` + `log.md`; fix `progress.json`, `index.html`, `viz/` paths |
| `.claude/skills/frontier-experiment-lab/SKILL.md` | output becomes day-scoped: `training/week-NN/<day-slug>/experiments/<exp-slug>/…` |
| `.claude/skills/frontier-paper-course/SKILL.md` | experiment references → new day-scoped convention |
| `.claude/workflows/phase1-lessons.js` | ~17 hardcoded `sessions/week-*/day-*.html` → new nested `lesson.html` paths |
| `frontier_lab_claude_skills/**` + its `CLAUDE.md` | mirror bundle: its skills use *generic* path templates (no hardcoded `sessions/week` strings), so this is a **convention/wording** update to describe the new day-scoped layout — not a hardcoded-path fix (decision D2) |

"Schedule" = the `progress.json` ledger (§6) plus these day-build skills; both are updated so the next generated day lands in the new layout.

## 8. Decisions

- **D1 — Root tooling stays flat** under `training/` (pure rename), not moved into a `training/_build/` subfolder, to keep the scripts' relative-path logic intact.
- **D2 — The distributable bundle** `frontier_lab_claude_skills/` is updated to describe the new convention (a wording-only change to its docs/templates — it holds no hardcoded `sessions/week` paths, per §7), so the repo's documented conventions stay internally consistent.
- **D3 — Inner folder names** are `lesson/` and `experiments/`. `lesson/` (not `session/`) avoids doubling the word "session" since the day folder itself is the training session.
- **D4 — Experiments layout** is one subfolder per experiment (`experiments/<slug>/`), supporting multiple experiments per day and matching the `frontier-experiment-lab` template.
- **D5 — Inline instructional path strings** inside the ~100 lesson HTMLs (teaching text such as "create `experiments/week01_jax/mlp_prng.py`") are **deferred** to an optional scripted follow-up pass. They are stale but non-breaking; blanket find/replace is risky given the irregular old experiment naming.

## 9. Verification

After migration:

1. `git status` shows the moves tracked as renames (history preserved).
2. Run `wire_index.py`, `lesson_audit.py`, `coverage_audit.py` from `training/` — all succeed with no missing-file errors.
3. Open `training/index.html` in a browser — every week/day link resolves to a `lesson/lesson.html`.
4. `grep -rn "sessions/" index.html .claude/ frontier_lab_claude_skills/ training/` returns no stale functional references (widened to cover the bundle and every script/workflow under `training/`).
5. Confirm module-internal `*/experiments/` dirs are untouched.
6. Confirm the top-level `experiments/` folder is **empty and removed** (no stale `week07_jax/` or parent left behind).

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Over-broad rename hits module `experiments/` dirs | Operate only on repo-root `experiments/`; §9 step 5 confirms. |
| `progress.json` path rewrite typos break the dashboard | Rewrite via script from a deterministic rule; verify with `lesson_audit.py` + browser open. |
| Public GitHub Pages URL path changes (`/sessions/` → `/training/`) | Personal learning repo; acceptable. Update root redirect + note in project memory. |
| Two skill copies drift again | D2 keeps both in sync in this change; note the active copy (`.claude/skills/`) is source of truth. |
| `inject_*` step-JSON keyed to old flat names | Keep JSON filenames; update only the scripts' path resolution (§6). |

## 11. Out of scope / follow-ups

- Optional scripted pass to update inline instructional path strings in lesson HTMLs (D5).
- Update the `github-pages-deploy` project memory note to record the `training/` URL path after the change.

## 12. Rollback

The entire change is `git mv` + text edits on a feature branch. Rollback = revert the branch / `git reset --hard` before merge. No data is deleted.
