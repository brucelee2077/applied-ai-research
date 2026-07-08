# Frontier Lab Refactor Skills v7.6

v7.6 turns the refactor process into an autonomous rollout pipeline.

The user should not need to manually run:

```text
audit → decide → fix → QA → update tracker → rollout next module
```

Instead, the user should be able to say:

```text
Continue the rollout from the tracker.
```

The system reads the tracker, detects whether the seed module needs stabilization, performs the needed loop, then proceeds to the next target.

## What v7.6 adds

v7.5 added the Learning Experience Gate.

v7.6 adds:

```text
Autonomous Rollout Loop
Seed Propagation Risk Rule
Seed Stabilization Phase
Short Command Contract
No Human Orchestrator Rule
```

## State files

Repo-level tracker:

```text
sessions/_refactor/rollout_tracker.yaml
```

Module-level manifest:

```text
sessions/<module>/_refactor/manifest.yaml
```

## Short command

After installing v7.6, the intended user command is:

```text
/goal Continue the Frontier Lab curriculum rollout from sessions/_refactor/rollout_tracker.yaml using the installed v7.6 skills. Act as the orchestrator: read the rollout tracker, detect whether any seed-module learning-experience propagation risk must be stabilized before the current target, perform that stabilization if needed, then roll out the current_target or next eligible module. Do not ask the user to manually choose between audit, fix, QA, or rollout when the tracker and reports provide enough information. Preserve shared shell files, quest IDs, data-quest-id values, navigation, localStorage, quiz mechanics, BUILD/DEMOS, completion behavior, and technical correctness. Update module manifests and the rollout tracker after each phase. Stop only when either the required seed stabilization and current target rollout are complete, or a blocker report explains why the pipeline could not continue.
```

## Philosophy

The user is not the orchestrator.
The rollout tracker is the project board.
The skills are the orchestration engine.
