# v7.3 Prompts

## 1. Manifest reconciliation after v7.2 loop

Use this after Module 2 v7.2 Phase 1-3 has completed.

```text
/goal Create and reconcile the Module 2 refactor manifest after the v7.2 loop. Create or update sessions/m02-the-neuron/_refactor/manifest.yaml as the durable module source of truth for constraints, quality gates, coverage items, visual/evidence items, artifact items, backlog, held-out eval findings, skill-gap candidates, and loop history. Read sessions/m02_v7_2_qa_report.md, sessions/m02_v7_2_fix_report.md, sessions/m02_v7_2_heldout_eval_report.md, sessions/m02_v7_2_skills_patch_report.md, and the current .claude/skills/frontier-*/SKILL.md. Do not edit lesson files. Do not read old notebooks directly unless needed to verify an existing held-out-eval claim. Apply the patched skills as a report-only delta QA against sessions/m02-the-neuron and update the manifest with any new P0/P1/P2 findings. If patched skills create new module P0s, write sessions/m02_v7_3_manifest_delta_report.md with the blocker list and do not fix them. If only P1/P2 remain, mark manifest.status as pass_with_p1 and state that Module 2 is ready as the seed module for next-module rollout. Stop after 8 turns if unmet and write sessions/m02_v7_3_manifest_blocker_report.md.
```

## 2. Future module refactor loop

```text
/goal Refactor <module> using the Frontier Lab v7.3 system. Create or update sessions/<module>/_refactor/manifest.yaml, run source-free first-principles coverage discovery, write/update coverage, coach voice, visual/evidence, and artifact state into the manifest, refactor lessons only after the manifest has a P0/P1 backlog, fix P0 only until QA reports 0 P0, update manifest after each loop, run held-out eval only if reference material is explicitly named, patch skills only if systemic gaps are found, and stop when manifest.status is pass or pass_with_p1 and all audit checks pass.
```

Then:

```text
Use /frontier-curriculum-architect with xhigh effort. Work toward the active /goal. Assume dynamic workflow is enabled.
```

## 3. Skills patch loop

```text
/goal Patch skills only if the latest manifest or held-out eval identifies systemic skill gaps. Do not edit lesson files. Update .claude/skills/frontier-*/SKILL.md, create frontier_lab_refactor_skills_vNEXT if appropriate, write a skills patch report, and update the module manifest skill_gap_candidates.
```
