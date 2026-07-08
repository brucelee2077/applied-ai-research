# v7.4 Minimal Prompts

## 1. Bootstrap rollout tracker

Run once:

```text
/goal Bootstrap the Frontier Lab rollout tracker. Create or update sessions/_refactor/rollout_tracker.yaml as the repo-level source of truth for module rollout order, current_target, global constraints, default success gates, module statuses, manifest paths, report paths, open P0/P1/P2 counts, skill version, rollout history, and skill history. Read existing module manifests if present, especially sessions/m02-the-neuron/_refactor/manifest.yaml. Do not edit lesson files. Do not refactor any module. Stop when the tracker exists, is valid YAML, includes m02 as pass_with_p1 seed_module true if supported by its manifest, includes m03-attention as the next not_started target, and writes sessions/_refactor/rollout_tracker_bootstrap_report.md.
```

Then simply say:

```text
Use /frontier-curriculum-architect with xhigh effort. Work toward the active /goal.
```

## 2. Roll out a named module

After tracker exists, use:

```text
/goal Roll out sessions/m03-attention using the Frontier Lab v7.4 rollout tracker. Read sessions/_refactor/rollout_tracker.yaml, set current_target to sessions/m03-attention, create or update sessions/m03-attention/_refactor/manifest.yaml, run source-free first-principles coverage discovery, update the module manifest before lesson edits, refactor only after the manifest has a P0/P1 backlog, fix P0 only until QA reports 0 P0, update the manifest and rollout_tracker after every QA/fix loop, do not read old notebooks or prior courseware unless held-out eval is explicitly requested, do not modify shared shell files, data-quest-id values, navigation, localStorage, quiz, BUILD/DEMOS, or completion behavior, do not do broad prose polish, and stop when manifest.status is pass or pass_with_p1, open P0 backlog is 0, lesson_audit.py passes for sessions/m03-attention, nav/jsdom/node checks pass if available, and rollout_tracker records the final status and report paths. Stop after 12 turns if unmet and write sessions/m03_v7_4_loop_blocker_report.md.
```

Then:

```text
Use /frontier-curriculum-architect with xhigh effort. Work toward the active /goal.
```

## 3. Roll out the next module automatically

```text
/goal Roll out the next not_started or blocked module from sessions/_refactor/rollout_tracker.yaml using the Frontier Lab v7.4 system. Read the tracker, select the next target from modules in order unless current_target is set, create/update that module's manifest, run source-free first-principles coverage discovery, refactor only after manifest backlog exists, fix P0 only until QA reports 0 P0, update the module manifest and rollout tracker after every loop, do not read old notebooks or prior courseware unless held-out eval is explicitly requested, preserve shared shell/navigation/quest-id/localStorage/quiz/BUILD/DEMOS/completion behavior, and stop when the selected module status is pass or pass_with_p1 and tracker records final reports and open P0/P1/P2 counts. Stop after 12 turns if unmet and write a module-specific loop blocker report.
```

Then:

```text
Use /frontier-curriculum-architect with xhigh effort. Work toward the active /goal.
```

## 4. Short command after tracker is established

Once v7.4 is stable, the intended short user command is:

```text
Roll out the next module from the rollout tracker.
```

The architect skill should expand this using the tracker rules.
