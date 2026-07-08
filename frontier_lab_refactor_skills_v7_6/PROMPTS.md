# v7.6 Prompts

## Main command

Use this as the normal command:

```text
/goal Continue the Frontier Lab curriculum rollout from sessions/_refactor/rollout_tracker.yaml using the installed v7.6 skills. Act as the orchestrator: read the rollout tracker, current seed module manifest(s), current target module manifest if present, and recent reports referenced by tracker or manifests. Detect whether any seed-module learning-experience propagation risk must be stabilized before rolling out the current target. If stabilization is required, perform the targeted seed stabilization first without asking the user for a separate prompt. For Module 2, if sessions/m02_v7_5_learning_experience_audit.md exists and identifies seed-propagation Learning Experience P1 findings such as LX-D1-PICTURE-ORDER, LX-D2-FORMULA-FIRST, LX-D1-OPENING-CHECKLIST, LX-D2-NARRATIVE-SPINE, or LX-ARTIFACT-AS-DISCOVERY, treat them as required seed-stabilization items before m03 rollout while still recording them as P1, not retroactive module P0. Do not do a full rewrite unless QA finds a Learning Experience P0. Stabilization may broadly adjust prose ordering and framing in the affected seed lessons, but only as needed to prevent propagation. Preserve technical correctness, artifact requirements, shared shell files, navigation, quest IDs, data-quest-id values, localStorage, quiz mechanics, BUILD/DEMOS, and completion behavior. After seed stabilization, run QA, update the seed manifest and rollout tracker, and write a seed stabilization report. Then roll out the current_target from the tracker, or the next eligible not_started/blocked module if current_target is absent. For target rollout, create/update the module manifest, run source-free first-principles coverage and learning-experience discovery, refactor only after backlog exists, fix P0 only until QA reports 0 P0, update manifest and tracker after every loop, and stop when the target module status is pass or pass_with_p1, open P0 backlog is 0, Learning Experience Gate passes, lesson_audit.py passes if available, nav/jsdom/node checks pass if available, and tracker records final report paths and counts. Do not ask the user to manually choose audit vs fix vs rollout when tracker/reports determine the next phase. If blocked, write a phase-specific blocker report explaining why the pipeline could not continue.
```

## Very short command after this stabilizes

```text
/goal Continue rollout from the tracker using installed Frontier Lab skills.
```

The skills should expand it using the rules in this package.
