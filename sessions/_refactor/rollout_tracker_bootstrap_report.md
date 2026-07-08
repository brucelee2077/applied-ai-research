# Rollout Tracker — Bootstrap Report

**Date:** 2026-07-08
**Skill:** `frontier-curriculum-architect` (Rollout Tracker Rule) · skills version **v7_4**
**Artifact created:** `sessions/_refactor/rollout_tracker.yaml`
**Schema followed:** `frontier_lab_refactor_skills_v7_4/ROLLOUT_TRACKER_SCHEMA.md`

---

## What this bootstrap did

Created the repo-level rollout tracker as the single source of truth for the
first-principles **refactor rollout** (the v7.x loop), replacing scattered prose
state. No lesson files were edited and no module was refactored — this was a
read + assemble + validate pass only.

The tracker carries every field the goal required:

| Required field | Where it lives in the tracker |
|---|---|
| Module rollout order | `modules:` list (44 entries, curriculum order) |
| `current_target` | top-level → `sessions/m03-attention` |
| Global constraints | `global_constraints:` (do_not_modify / do_not_do / sanctioned_designs / language_preference) |
| Default success gates | `default_success_gates:` (P0=0, allowed statuses, required gates + checks) |
| Module statuses | per-module `status:` + `summary.by_status` |
| Manifest paths | per-module `manifest:` |
| Report paths | per-module `reports:` + `rollout_history[].reports` |
| Open P0/P1/P2 counts | per-module `open_p0/p1/p2` + `summary.*_total` |
| Skill version | top-level `version: v7_4` + `skills:` block |
| Rollout history | `rollout_history:` |
| Skill history | `skill_history:` |

---

## Evidence gathered before writing

- **No prior tracker existed** — `sessions/_refactor/` did not exist. Created fresh.
- **m02 manifest read in full** — `sessions/m02-the-neuron/_refactor/manifest.yaml`
  (the only module manifest on disk). It directly supports the seed claims:
  - `meta.status: pass_with_p1`, `meta.seed_module: true`, `meta.open_p0_count: 0`.
  - Backlog counts recomputed from the manifest: **P0 = 0, P1 = 9, P2 = 10**
    (P1 = BL-P1a…e, BL-P1-legacy, BL-VE-D1, BL-VE-D2, BL-COV-XOR; P2 = 9 per-day + loopvec).
    These match the v7.4 schema's example counts exactly.
- **Report paths verified on disk** — the 4 recorded m02 reports all exist
  (`m02_v7_2_{qa,fix,heldout_eval,skills_patch}_report.md`). The older
  `v7`/`v7.1` gap/fix reports referenced in the manifest's `loop_history` are
  **deleted from the working tree**, so they were intentionally *not* listed as
  live report paths (their history stays in the module manifest).
- **Skills version confirmed** — installed `.claude/skills/frontier-*/SKILL.md`
  is **byte-identical** (all 4) to `frontier_lab_refactor_skills_v7_4/skills/`
  via `diff -q`. Current version = **v7_4**. The former v7_2 / v7_3 mirror dirs
  are deleted; v7_4 is the live mirror.
- **Module inventory** — 44 module directories on disk, matching ROADMAP.md's
  "~44 build-modules": 23 canonical spine dirs (`sessions/mXX-*`) + 21
  later-built flat dirs (`sessions/week-m*`). All 44 are enumerated in the
  tracker; every `path:` was asserted to be a real directory in validation.

---

## Module order & status (roll-up)

Order = curriculum sequence (Parts A→D, then Phases 1→9), which is also numeric
module order. Grouped in the tracker with section comments.

- **Total modules:** 44
- **`pass_with_p1`:** 1 — `m02-the-neuron` (**seed**)
- **`not_started`:** 43 — every other module (no `_refactor/manifest.yaml` yet)
- **`current_target`:** `sessions/m03-attention`
- **Next after m03:** `m04-first-model-mlp` (recorded in `next_rollout.after`)

**Note on already-built modules:** `status` in this tracker means *first-principles
refactor-rollout status*, **not** original build status. Modules like m01 and m04
are built/shipped (and had industry-anchor / staff-lens retrofits) but are marked
`not_started` because they have not been through the v7.x refactor loop and have no
module manifest yet. This is called out on the m01 entry and in the tracker header.

---

## Validation

`python3` + `PyYAML` load + assertion pass — **all green**:

- YAML parses; all 10 required top-level keys present.
- `version == v7_4`; `current_target == sessions/m03-attention`.
- m02: `status == pass_with_p1`, `seed_module == True`, `open_p0 == 0`, 4 reports.
- m03: `status == not_started`, `seed_module == False`.
- Exactly **one** `seed_module: true` in the whole tracker (m02).
- No duplicate module ids; `summary.total_modules == len(modules) == 44`.
- Every module `manifest:` path ends in `/_refactor/manifest.yaml`.
- Every module `path:` resolves to a real on-disk directory.

---

## Stop condition — met

| Required to stop | Status |
|---|---|
| Tracker exists at `sessions/_refactor/rollout_tracker.yaml` | ✅ |
| Valid YAML | ✅ (PyYAML load clean) |
| m02 = `pass_with_p1`, `seed_module: true` (supported by its manifest) | ✅ (manifest confirms both) |
| m03-attention present as the next `not_started` target | ✅ (`current_target` + `status: not_started`) |
| Bootstrap report written | ✅ (this file) |

---

## What was intentionally NOT done

- **No lesson files edited**, **no module refactored** (out of scope for bootstrap).
- **No new module manifests created** — m03's manifest is referenced as a planned
  path (`sessions/m03-attention/_refactor/manifest.yaml`) and will be created when
  m03 is actually rolled out.
- **Titles/topics not duplicated** into the tracker — ROADMAP.md remains the topic
  source of truth (referenced in the tracker header); each entry carries `id`,
  `code`, and `path` for unambiguous identification.

---

## How to roll out the next module

Per the tracker's `next_rollout` block and the v7.4 minimal command, a picking-up
session can simply say:

```text
Roll out sessions/m03-attention using the rollout tracker.
```

The architect skill will: read this tracker → confirm `current_target` → create
`sessions/m03-attention/_refactor/manifest.yaml` → derive the goal loop from
`default_success_gates` → set m03 `status: in_progress` → run the refactor + QA →
update status/counts/reports → append `rollout_history` → advance `current_target`
to `m04-first-model-mlp`.
