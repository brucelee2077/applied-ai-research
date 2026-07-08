# Rollout Tracker Schema

Repo-level tracker path:

```text
sessions/_refactor/rollout_tracker.yaml
```

## Required shape

```yaml
version: v7_4
current_target: sessions/m03-attention
default_mode: source_free
dynamic_workflow_assumed: true

global_constraints:
  do_not_modify:
    - shared shell files
    - data-quest-id values
    - navigation
    - localStorage
    - quiz mechanics
    - BUILD/DEMOS completion behavior
  do_not_do:
    - broad prose polish
    - read old notebooks unless held-out eval is explicitly requested

default_success_gates:
  p0_count: 0
  manifest_status:
    - pass
    - pass_with_p1
  required_gates:
    - coverage_traceability
    - coach_voice
    - foundation_framing_when_applicable
    - function_family
    - visual_evidence
    - artifact
    - anchor_consistency
  required_checks:
    - lesson_audit
    - nav_audit_if_available
    - jsdom_or_node_if_available

modules:
  - id: m02-the-neuron
    path: sessions/m02-the-neuron
    status: pass_with_p1
    seed_module: true
    manifest: sessions/m02-the-neuron/_refactor/manifest.yaml
    skill_version: v7_3
    open_p0: 0
    open_p1: 9
    open_p2: 10
    reports:
      - sessions/m02_v7_2_qa_report.md
      - sessions/m02_v7_2_fix_report.md
      - sessions/m02_v7_2_heldout_eval_report.md
      - sessions/m02_v7_2_skills_patch_report.md

  - id: m03-attention
    path: sessions/m03-attention
    status: not_started
    seed_module: false
    manifest: sessions/m03-attention/_refactor/manifest.yaml
    skill_version: v7_4
    open_p0: null
    open_p1: null
    open_p2: null
    reports: []

rollout_history:
  - loop: m02_v7_3_manifest_reconciliation
    result: pass_with_p1
    reports:
      - sessions/m02-the-neuron/_refactor/manifest.yaml

skill_history:
  - version: v7_4
    change: Added repo-level rollout tracker rule.
```

## Status values

```text
not_started
in_progress
blocked
pass
pass_with_p1
ready_for_rollout
skipped
```

## Tracker discipline

Before a module rollout:
- read tracker
- set current_target
- ensure module entry exists
- ensure module manifest exists or create it

After a module rollout:
- update module status
- update open P0/P1/P2 counts
- record reports
- append rollout_history
- select next suggested target if obvious
