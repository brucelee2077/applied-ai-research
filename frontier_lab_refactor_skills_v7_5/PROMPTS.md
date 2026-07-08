# v7.5 Prompts

## 1. Audit Module 2 learning experience

```text
/goal Audit Module 2 for learning experience quality using the v7.5 Learning Experience Gate. Read sessions/m02-the-neuron/day-01-single-neuron/lesson.html and sessions/m02-the-neuron/day-02-activations/lesson.html. Compare against 00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb only as a style and learning-barrier benchmark, not as source content to copy. Do not edit lesson files. Create sessions/m02_v7_5_learning_experience_audit.md. The audit must identify where the lessons feel task-like, checklist-like, too engineering-first, too formula-first, insufficiently curiosity-building, or weaker than the notebook at reducing learning barrier. It must classify issues P0/P1/P2 and recommend whether Day 1/Day 2 need a full experience-layer rewrite before continuing m03 rollout.
```

Then:

```text
Use /frontier-refactor-qa with xhigh effort. Work toward the active /goal.
```

## 2. Rewrite Module 2 Day 1/Day 2 experience layer if audit says P0

```text
/goal Rewrite Module 2 Day 1 and Day 2 to pass the v7.5 Learning Experience Gate while preserving technical correctness, artifact requirements, navigation, quest IDs, localStorage, quiz, BUILD/DEMOS, and completion behavior. Use sessions/m02_v7_5_learning_experience_audit.md and sessions/m02-the-neuron/_refactor/manifest.yaml as source of truth. This is allowed to rewrite existing lesson prose broadly, not just patch introductions, but must not broaden scope beyond Day 1/Day 2 unless required by QA. The rewrite must integrate curiosity, human intuition, brain-vs-ANN framing, analogy spine, mechanism, visual/evidence, artifact, and staff depth into one coherent narrative. After editing, run QA and update the manifest. Stop when Learning Experience Gate passes, P0=0, lesson_audit.py passes for m02, nav/jsdom/node checks pass if available, and write sessions/m02_v7_5_experience_rewrite_report.md.
```

Then:

```text
Use /frontier-curriculum-architect with xhigh effort. Work toward the active /goal.
```

## 3. Future module rollout with Learning Experience Gate

```text
/goal Roll out the next module from sessions/_refactor/rollout_tracker.yaml using v7.5. Create/update module manifest, run source-free coverage discovery, include Learning Experience Gate in the manifest quality_gates, refactor lessons so experience layer, mechanism layer, and staff-depth layer are integrated, fix P0 only until QA reports 0 P0, update manifest and rollout tracker after each loop, and stop when manifest.status is pass or pass_with_p1, open P0 backlog is 0, Learning Experience Gate passes, lesson_audit.py passes, and nav/jsdom/node checks pass if available.
```

Then:

```text
Use /frontier-curriculum-architect with xhigh effort. Work toward the active /goal.
```
