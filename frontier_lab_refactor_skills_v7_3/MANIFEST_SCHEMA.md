# Module Refactor Manifest Schema

Each module should maintain:

```text
sessions/<module>/_refactor/manifest.yaml
```

The manifest is the durable source of truth for the module refactor.

## Required top-level fields

```yaml
module: m02-the-neuron
title: The Neuron & How It Learns
status: pass_with_p1
current_loop: v7_3
last_verified_commit: null

constraints: {}
quality_gates: {}
coverage_items: []
visual_evidence_items: []
artifact_items: []
backlog: []
heldout_eval_findings: []
skill_gap_candidates: []
loop_history: []
```

## Status values

Use:

```text
not_started
in_progress
blocked
pass
pass_with_p1
ready_for_rollout
```

## Finding priority

Use:

```text
P0
P1
P2
```

## Finding status

Use:

```text
open
fixed
partial
waived
blocked
patched
```

## Example

```yaml
module: m02-the-neuron
title: The Neuron & How It Learns
status: pass_with_p1
current_loop: v7_3
last_verified_commit: null

constraints:
  source_free_until_heldout_eval: true
  do_not_modify:
    - shared shell files
    - data-quest-id values
    - navigation
    - localStorage
    - quiz mechanics
    - BUILD/DEMOS completion behavior

quality_gates:
  coverage_traceability: pass
  coach_voice: pass
  foundation_framing: pass
  function_family: pass
  visual_evidence: pass_with_p1
  artifact: pass
  anchor_consistency: pass_with_p1
  heldout_eval: systemic_gaps_patched

coverage_items:
  - id: COV-D1-BRAIN-ANN
    lesson: day-01-single-neuron
    concept: Brain-inspired intuition and ANN caveat
    status: must_cover
    required_depth:
      - intuition
      - mapping
      - caveat
      - where_analogy_breaks
    current_state: pass
    evidence:
      - biological_to_artificial_mapping_table
      - analogy_breaks_section
    priority: P0

visual_evidence_items:
  - id: VIS-D2-DERIVATIVES
    lesson: day-02-activations
    learning_question: Why do sigmoid/tanh saturate and why can ReLU die?
    required_surface:
      - activation_curve
      - derivative_curve
      - saturation_zone
      - dead_relu_region
    current_state: partial
    priority: P1
    source: sessions/m02_v7_2_heldout_eval_report.md

artifact_items:
  - id: ART-D5-BACKPROP-FINITE-DIFF
    lesson: day-05-gradients-backprop
    artifact_path: sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py
    current_state: pass
    required_output:
      - chain_rule_pieces
      - finite_difference_check
      - error_less_than_1e-6

backlog:
  - id: M02-P1-ACTIVATION-DERIVATIVE-VIZ
    priority: P1
    lesson: day-02-activations
    issue: Activation derivative shapes are not plotted.
    desired_fix: Add live curve + derivative visual or runnable artifact output.
    status: open
    source: sessions/m02_v7_2_heldout_eval_report.md

heldout_eval_findings:
  - id: HE-M02-VISUAL-EVIDENCE
    priority: P1
    finding: Reference notebooks have stronger plotted/runnable evidence for behavioral concepts.
    affected_lessons:
      - day-01-single-neuron
      - day-02-activations
    status: open

skill_gap_candidates:
  - id: S1
    status: patched
    skill: frontier-visual-evidence-builder
    issue: Behavioral concepts need plotted/live-recomputed evidence.
    patch_report: sessions/m02_v7_2_skills_patch_report.md

loop_history:
  - loop: v7_2_phase_1
    result: pass_with_p1
    p0_count: 0
    reports:
      - sessions/m02_v7_2_qa_report.md
      - sessions/m02_v7_2_fix_report.md

  - loop: v7_2_phase_2
    result: systemic_skill_gaps_found
    reports:
      - sessions/m02_v7_2_heldout_eval_report.md

  - loop: v7_2_phase_3
    result: skills_patched
    reports:
      - sessions/m02_v7_2_skills_patch_report.md
```

## Manifest discipline

Reports are evidence snapshots.
The manifest is the current project board.

After each loop, update the manifest.
