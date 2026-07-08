# Goal + Workflow Playbook

## When to use `/goal`

Use `/goal` for iterative work with a measurable end condition.

Good goal condition:

```text
All P0 findings in sessions/m02_v7_1_gap_report.md are fixed, v7.2 QA reports 0 P0, lesson_audit.py passes for m02, nav/jsdom checks pass if available, and sessions/m02_v7_2_fix_report.md says Merge recommendation: Pass or Pass with P1.
```

Bad goal condition:

```text
Make m02 better.
```

## When to use a dynamic workflow

Use dynamic workflows when the task benefits from parallel subagents:

- one agent per lesson for coverage traceability
- one agent for Coach Voice Gate
- one agent for Visual/Evidence Gate
- one agent for Artifact Gate
- one adversarial evaluator
- one integrator to deduplicate and rank findings

## Standard module improvement loop

```text
1. Report-only QA workflow
2. P0-only fix workflow
3. Re-run QA workflow
4. Held-out eval workflow
5. Skill-gap analysis workflow
6. Patch skills if systemic failure remains
```

## Stop conditions

Stop the loop if:

- P0 = 0 and merge recommendation is Pass or Pass with P1
- two rounds in a row make no progress
- hard JS/nav/audit failures appear
- required fix would modify shared shell files
- held-out eval finds systemic skill failure requiring skill patch
