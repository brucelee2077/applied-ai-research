# Frontier Lab Refactor Skills v7.3

v7.3 adds the **Module Refactor Manifest** as a first-class part of the curriculum refactor system.

v7.2 introduced goal-driven and dynamic-workflow orchestration.
v7.3 adds durable module state.

## Core idea

```text
Generic skills
+ Module-specific manifest
+ /goal loop
+ QA/eval reports
= reusable curriculum refactor system
```

The skills should stay generic. Module-specific coverage, visual/evidence requirements, artifact status, backlog, held-out findings, and loop history live in:

```text
sessions/<module>/_refactor/manifest.yaml
```

## Four skills

1. `frontier-curriculum-architect`
   - creates/refines module manifest
   - first-principles coverage discovery
   - coverage contracts
   - goal-loop planning
   - dynamic workflow orchestration

2. `frontier-lesson-builder`
   - warm coach lesson construction
   - intuition-first writing
   - coverage-aware writing
   - artifact and explain-back design

3. `frontier-visual-evidence-builder`
   - visuals and runnable evidence
   - failure demos
   - benchmarks
   - artifact reproducibility

4. `frontier-refactor-qa`
   - QA gates
   - manifest update gate
   - held-out eval
   - skill-gap analysis
   - goal-loop stop conditions

## Install

From repo root:

```bash
unzip frontier_lab_refactor_skills_v7_3.zip

mkdir -p .claude/skills
rm -rf .claude/skills/frontier-*
cp -R frontier_lab_refactor_skills_v7_3/skills/* .claude/skills/

find .claude/skills -maxdepth 2 -name SKILL.md | sort
```

Expected:

```text
.claude/skills/frontier-curriculum-architect/SKILL.md
.claude/skills/frontier-lesson-builder/SKILL.md
.claude/skills/frontier-refactor-qa/SKILL.md
.claude/skills/frontier-visual-evidence-builder/SKILL.md
```

## What changed from v7.2

v7.3 adds:

- Module Refactor Manifest Rule
- Manifest Update Gate
- Manifest Reconciliation Loop
- Manifest schema
- standard `/goal` prompts for manifest-based refactor loops

## Operating principle

Do not leave findings only in prose reports.

Every QA finding, held-out eval finding, and skill-gap candidate should be reflected in the module manifest.
