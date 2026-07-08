# v7.2 Prompts

## 1. Set a goal for Module 2 P0 closure

Paste into Claude Code:

```text
/goal Module 2 reaches Pass or Pass with P1: sessions/m02_v7_2_qa_report.md reports 0 P0 findings, all must-cover concepts are PASS or explicitly waived, Foundation Framing Gate passes, Function Family Gate passes, Visual/Evidence Gate passes, Artifact Gate passes, Anchor Consistency Gate has no P0, lesson_audit.py passes for sessions/m02-the-neuron, nav/jsdom/node checks pass if available, and sessions/m02_v7_2_fix_report.md summarizes the fixes and remaining P1/P2 only; stop after 8 turns if this condition is not met and write a blocker report.
```

Then send:

```text
Use /frontier-refactor-qa with xhigh effort.

Use a dynamic workflow.

Work toward the active /goal.

Do not read old notebooks.
Do not modify shared shell files.
Do not change data-quest-id values.
Do not do broad prose polish.
Fix P0 only unless a local P1 blocks verification.

Workflow phases:
1. Audit current m02 against v7.2 gates.
2. Apply P0-only fixes discovered by the audit.
3. Re-run QA.
4. If P0 remains, repeat once with a narrower fix plan.
5. Write sessions/m02_v7_2_fix_report.md and sessions/m02_v7_2_qa_report.md.
```

## 2. Held-out eval goal

```text
/goal Held-out eval complete: sessions/m02_v7_2_heldout_eval_report.md exists, compares current m02 against the old notebooks, includes adversarial sections on overclaiming, beginner intuition, coverage, visual/evidence, artifacts, and skills gaps, and makes a clear decision: keep v7.2, patch skills, or rerun module refactor.
```

Then send:

```text
Use /frontier-refactor-qa with xhigh effort.

Use a dynamic workflow for held-out evaluation.

You may now read old notebooks.
Do not edit lesson files.

Compare current m02 against:
- 00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb
- 00-neural-networks/fundamentals/02_single_neuron.ipynb
- 00-neural-networks/fundamentals/03_activation_functions.ipynb

Write:
- sessions/m02_v7_2_heldout_eval_report.md
```

## 3. Skill improvement goal

```text
/goal Skills patched if needed: if held-out eval identifies systemic skill gaps, .claude/skills/frontier-* are updated, frontier_lab_refactor_skills_v7_3/ is created if needed, skills patch report exists, and no lesson files are modified.
```

Then send:

```text
Use /frontier-curriculum-architect with xhigh effort.

Read:
- sessions/m02_v7_2_heldout_eval_report.md
- current .claude/skills/frontier-*/SKILL.md

If the eval identifies systemic skill gaps, patch skills only.
Do not edit lessons.
Write a skills patch report.
```
