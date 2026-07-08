# Frontier Lab Refactor Skills v7.2

v7.2 adds Goal + Dynamic Workflow orchestration.

v7.1 fixed the quality gates:
- coverage traceability
- coach voice
- visual/evidence
- artifact consistency
- held-out eval honesty

v7.2 turns those gates into an iterative improvement loop using Claude Code `/goal` and dynamic workflows.

## New operating model

```text
Set /goal
→ run dynamic workflow
→ audit module
→ apply P0 fixes
→ rerun QA
→ optional held-out eval
→ patch skills if eval exposes systemic gaps
→ repeat until goal condition is satisfied
```

## Four skills

1. `frontier-curriculum-architect`
   - first-principles design
   - coverage contracts
   - workflow planning
   - goal condition design

2. `frontier-lesson-builder`
   - warm coach lesson construction
   - coverage-aware writing
   - artifacts

3. `frontier-visual-evidence-builder`
   - visuals and runnable evidence
   - failure demos
   - benchmarks

4. `frontier-refactor-qa`
   - QA gates
   - held-out eval
   - skill-gap analysis
   - goal-loop stop conditions

## Install

```bash
unzip frontier_lab_refactor_skills_v7_2.zip
mkdir -p .claude/skills
rm -rf .claude/skills/frontier-*
cp -R frontier_lab_refactor_skills_v7_2/skills/* .claude/skills/
find .claude/skills -maxdepth 2 -name SKILL.md | sort
```

## Core principle

Do not manually spoon-feed fixes unless the loop fails.

Use `/goal` to keep Claude Code improving until a measurable condition holds.
Use dynamic workflows for fan-out audit/eval/fix phases.
