# Frontier Lab Claude Skills Pack v2

This is a drop-in replacement / upgrade for `frontier_lab_claude_skills`.

It keeps the original spirit:

- lower the bar without lowering the ceiling
- visual-first learning
- artifact-driven study
- bilingual Chinese + English support
- frontier-lab readiness

But it adds a missing layer:

> a reusable **Coach Layer** for refactoring existing `sessions/` lessons into warm, intuition-first, math-friendly, artifact-driven HTML lessons.

## Recommended install

From your repo root:

```bash
git checkout -b refactor/coach-skills-v2

# Option A: replace the skill pack folder
rm -rf frontier_lab_claude_skills
cp -R frontier_lab_claude_skills_v2 frontier_lab_claude_skills

# Option B: install project-local skills for Claude Code
mkdir -p .claude/skills
cp -R frontier_lab_claude_skills/skills/* .claude/skills/
```

Then start Claude Code from the repo root:

```bash
claude
```

## Main workflow

Use this order:

1. `/frontier-curriculum-refactor`  
   Plan and run a safe `sessions/` refactor pilot.

2. `/frontier-lesson-humanizer`  
   Upgrade one existing lesson into warm, intuition-first coach voice.

3. `/frontier-math-unfogger`  
   Fix math-heavy sections with a Math Ladder.

4. `/frontier-d3-visual-lab`  
   Improve or add learning-focused visuals.

5. `/frontier-artifact-reviewer` or `/frontier-review-quiz`  
   Review produced artifacts and decide whether the session counts.

## New skills in v2

- `frontier-curriculum-refactor`
- `frontier-lesson-humanizer`
- `frontier-math-unfogger`
- `frontier-artifact-reviewer`

## Updated skills from v1

- `frontier-session-coach`
- `frontier-concept-courseware`
- `frontier-d3-visual-lab`
- `frontier-experiment-lab`
- `frontier-paper-course`
- `frontier-review-quiz`
- `frontier-portfolio-packager`

## Design principle

Do not let the curriculum become pretty productivity theater.

Every learning day should create evidence:

```text
Read → Understand → Explain → Implement → Measure → Write → Reflect
```
