# Frontier Lab Refactor Skills v7.5

v7.5 adds the missing **Learning Experience Layer**.

Earlier versions made the curriculum more correct, traceable, and refactorable. But correctness is not enough. A foundation lesson must make the learner want to continue.

## What v7.5 fixes

v7.4 had:

```text
coverage + manifest + rollout tracker + QA loop
```

v7.5 adds:

```text
learning barrier reduction + curiosity + narrative spine + staff-depth integration
```

The goal is not to add a friendly intro. The goal is to rewrite lessons so the experience layer runs through the whole lesson.

## Core model

Each lesson has three layers:

```text
1. Experience Layer
   reduce fear, build curiosity, start from human intuition

2. Mechanism Layer
   teach the math/system precisely

3. Staff Depth Layer
   show failure modes, scale relevance, artifacts, interview-ready reasoning
```

A good lesson flows through all three. It should not be a checklist with a cute intro.

## Install

```bash
unzip frontier_lab_refactor_skills_v7_5.zip

mkdir -p .claude/skills
rm -rf .claude/skills/frontier-*
cp -R frontier_lab_refactor_skills_v7_5/skills/* .claude/skills/

find .claude/skills -maxdepth 2 -name SKILL.md | sort
```

## Operating principle

Do not choose between beginner curiosity and staff depth.

The lesson should feel welcoming enough to start and rigorous enough to prepare for frontier-lab interviews.
