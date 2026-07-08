# Frontier Lab Refactor Skills v6

This is a clean restart of the skills system.

The goal is not to migrate old notebooks into HTML. The goal is to make Claude Code capable of creating **notebook-quality, intuition-first, visual, artifact-driven lessons from first principles** while safely refactoring the existing `sessions/` codebase.

## Why v6 exists

Earlier skills improved local lesson structure, but they had three failure modes:

1. They could polish a weak lesson instead of rebuilding the learning experience.
2. They could rely too much on old notebooks when source existed.
3. They did not always enforce coverage, visual depth, and artifact quality before editing.

v6 fixes this by making **source-free curriculum architecture** the default.

Old notebooks may be used later as held-out evaluation references, but not as training/source material unless the user explicitly asks.

## Core workflow

```text
Existing sessions codebase
→ structural constraints audit
→ first-principles learning blueprint
→ coverage contract
→ visual contract
→ artifact contract
→ lesson refactor
→ QA gate
→ optional held-out eval
```

## Skills

There are only four skills:

1. `frontier-curriculum-architect`
   - Orchestrates module/batch refactors.
   - Creates first-principles blueprints.
   - Creates coverage, visual, and artifact contracts.
   - Preserves existing session shell behavior.

2. `frontier-lesson-builder`
   - Builds notebook-quality lesson content.
   - Enforces intuition-first teaching.
   - Creates analogies, Math Ladders, Jargon Busters, examples, Staff Lens, interview answers, and artifacts.

3. `frontier-visual-builder`
   - Designs and implements notebook-grade visuals.
   - Creates P0/P1/P2 visual decisions.
   - Enforces visual contracts.

4. `frontier-refactor-qa`
   - Runs post-edit QA.
   - Checks coverage, visuals, artifacts, stale commands, legacy skill references, JS/shell behavior, and optional held-out eval.

## Install

From repo root:

```bash
unzip frontier_lab_refactor_skills_v6.zip

mkdir -p .claude/skills
rm -rf .claude/skills/frontier-*
cp -R frontier_lab_refactor_skills_v6/skills/* .claude/skills/

find .claude/skills -maxdepth 2 -name SKILL.md | sort
```

Expected:

```text
.claude/skills/frontier-curriculum-architect/SKILL.md
.claude/skills/frontier-lesson-builder/SKILL.md
.claude/skills/frontier-refactor-qa/SKILL.md
.claude/skills/frontier-visual-builder/SKILL.md
```

## Golden standard

A lesson should feel like a high-quality interactive notebook plus a warm live coach:

```text
Why care
→ Pain point
→ Intuition
→ Analogy
→ Tiny example
→ Formula / mechanism
→ Visual
→ Code experiment
→ Staff lens
→ Interview answer
→ Produce artifact
→ Explain-back
```

No formula before felt intuition.
No module without a coverage contract.
No visual without a learning question.
No artifact without an explain-back.
