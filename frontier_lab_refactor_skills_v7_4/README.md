# Frontier Lab Refactor Skills v7.4

v7.4 adds a repo-level **Rollout Tracker** so the user does not need to hand-write a long rollout prompt for every module.

## Core operating model

```text
Generic skills
+ repo-level rollout tracker
+ module-specific manifest
+ /goal loop
+ QA/eval reports
= repeatable curriculum rollout system
```

## New tracker

Maintain:

```text
sessions/_refactor/rollout_tracker.yaml
```

This is the curriculum-wide project board. It tracks:

- module order
- rollout status
- current target
- global constraints
- default gates
- per-module manifest paths
- report paths
- open P0/P1/P2 counts
- skill version used
- rollout history

Each module still has its own manifest:

```text
sessions/<module>/_refactor/manifest.yaml
```

## Minimal user command

After installing v7.4, the user should only need to say something like:

```text
Roll out sessions/m03-attention using the rollout tracker.
```

or

```text
Roll out the next module from sessions/_refactor/rollout_tracker.yaml.
```

The skills should expand that into the correct `/goal` loop internally.

## Skills

Still only four skills:

1. `frontier-curriculum-architect`
2. `frontier-lesson-builder`
3. `frontier-visual-evidence-builder`
4. `frontier-refactor-qa`

No new skill is added to avoid skill sprawl.

## Install

```bash
unzip frontier_lab_refactor_skills_v7_4.zip

mkdir -p .claude/skills
rm -rf .claude/skills/frontier-*
cp -R frontier_lab_refactor_skills_v7_4/skills/* .claude/skills/

find .claude/skills -maxdepth 2 -name SKILL.md | sort
```
