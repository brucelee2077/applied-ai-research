# Frontier Lab Refactor Skills v7.1

v7.1 patches the skills after the Module 2 v7 QA revealed systemic gaps.

The goal is not to fix Module 2 manually. The goal is to make the skills discover and prevent those failures automatically in future refactors.

## What changed from v7

v7 detected important issues, but the rules were still not hard enough in four areas:

1. Coverage discovery
   - It could still miss family variants like `tanh` in activation functions.
2. Foundation intuition
   - A generic everyday analogy could incorrectly satisfy the neural-network brain-vs-ANN framing requirement.
3. Evidence quality
   - Simulated playground output could be over-counted as evidence.
4. Anchor consistency
   - Playground, visual, quiz, Produce, and acceptance criteria could use different numbers or concepts.

v7.1 adds hard gates for all four.

## Four skills

1. `frontier-curriculum-architect`
   - Coverage Engine
   - Foundation Framing Lens
   - Anchor Consistency Lens
   - family/variant coverage
   - contract generation

2. `frontier-lesson-builder`
   - mandatory warm coach voice
   - first-20-percent intuition gate
   - neural foundation rule
   - function family rule
   - reader-objection Q&A
   - Jargon Busters
   - step-traced examples

3. `frontier-visual-evidence-builder`
   - simulated-vs-runnable evidence gate
   - failure demos
   - numeric readouts
   - benchmarks
   - small multiples
   - artifact reproducibility

4. `frontier-refactor-qa`
   - coverage traceability table
   - anchor consistency gate
   - coach voice QA
   - visual/evidence QA
   - artifact QA
   - legacy skill reference check
   - adversarial held-out eval

## Install

From repo root:

```bash
unzip frontier_lab_refactor_skills_v7_1.zip

mkdir -p .claude/skills
rm -rf .claude/skills/frontier-*
cp -R frontier_lab_refactor_skills_v7_1/skills/* .claude/skills/

find .claude/skills -maxdepth 2 -name SKILL.md | sort
```

Expected:

```text
.claude/skills/frontier-curriculum-architect/SKILL.md
.claude/skills/frontier-lesson-builder/SKILL.md
.claude/skills/frontier-refactor-qa/SKILL.md
.claude/skills/frontier-visual-evidence-builder/SKILL.md
```

## Operating principle

Do not use manual prompts to list obvious coverage. The skills should discover good-enough coverage from first principles.

The user may provide constraints, but the skills must own:
- coverage completeness,
- intuition quality,
- evidence quality,
- artifact consistency,
- and QA honesty.
