# Frontier Lab Claude Skills Pack

This pack contains Claude Code skills for turning Frontier Lab self-study into low-friction, visual, artifact-driven learning.

## Install options

### Personal install
Copy all folders inside `skills/` into:

```bash
~/.claude/skills/
```

Example:

```bash
mkdir -p ~/.claude/skills
cp -R skills/* ~/.claude/skills/
```

### Project install
Copy the folders into a repo-local skills directory:

```bash
mkdir -p .claude/skills
cp -R skills/* .claude/skills/
```

Then start Claude Code in that repo.

## Recommended repo layout

```text
frontier-lab-study-os/
  .claude/skills/
  notes/
  sessions/
  courseware/
  experiments/
  quizzes/
  portfolio/
```

## Suggested skill loop

1. Ask Claude Code: `/frontier-session-coach topic="attention"`
2. Use `/frontier-concept-courseware` to generate an HTML mini-course.
3. Use `/frontier-d3-visual-lab` to create interactive D3 visuals.
4. Use `/frontier-experiment-lab` to build a tiny reproducible implementation.
5. Use `/frontier-review-quiz` to quiz you and update the log.
6. Weekly, use `/frontier-portfolio-packager` to turn outputs into a portfolio-ready writeup.

## Design principles

- Lower the bar: explain like a smart beginner can follow.
- Preserve depth: always include the technical mechanism and frontier relevance.
- Visual-first: HTML and D3.js outputs are preferred whenever useful.
- Artifact-driven: every session produces a concrete file.
- Bilingual: Chinese + English, with key technical terms in English.
- No employer/day-job context by default: self-study only unless explicitly requested.
