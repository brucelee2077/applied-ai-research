# Autonomous Rollout Loop

## Purpose

The system should avoid making the user manually chain prompts.

Bad:

```text
Run audit.
Now decide if rewrite needed.
Now run fix.
Now run QA.
Now update tracker.
Now roll out m03.
```

Good:

```text
Continue rollout from tracker.
```

## Required loop

When asked to continue rollout:

```text
1. Read sessions/_refactor/rollout_tracker.yaml
2. Read current seed module manifest(s)
3. Read any recent audit reports referenced by tracker or manifest
4. Detect seed propagation risks
5. If seed stabilization is required, stabilize seed first
6. Update seed manifest and tracker
7. Select current_target or next eligible target
8. Create/read target module manifest
9. Run source-free coverage + learning-experience discovery
10. Refactor P0 only, plus any explicitly allowed propagation-risk seed P1
11. QA
12. Update manifest and tracker
13. Stop at pass/pass_with_p1 or write blocker report
```

## Seed Propagation Risk Rule

A P1 in a seed module may become required-to-fix before rollout if:

- the seed module is marked `seed_module: true`,
- the issue affects lesson structure, not just local content,
- the pattern is likely to be copied into future modules,
- the issue would be P0 for a fresh build under current skills,
- the fix is targeted and low-risk.

Examples:

- definition/formula-first before mental picture,
- checklist opening before invitation,
- artifact framed as homework before curiosity,
- staff/frontier relevance before orientation,
- decorative analogy rather than narrative spine.

These are not retroactive P0s for the shipped module, but they are rollout-blocking seed-stabilization items.

## No Human Orchestrator Rule

Do not stop and ask the user whether to do the next obvious phase when:

- the tracker exists,
- current_target exists,
- audit report exists,
- manifest exists,
- and the next phase is determined by rules.

Proceed and write the appropriate report.

## Stop conditions

Stop only when:

- seed stabilization, if required, is complete,
- current target rollout reaches pass or pass_with_p1,
- manifests and tracker are updated,
- report paths are recorded,
- open P0 count is 0,

or when a blocker report is written.
