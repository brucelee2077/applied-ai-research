# First Prompt After Installing v2

Paste this into Claude Code from the repo root:

```text
Use /frontier-curriculum-refactor.

Context:
This repo has an existing sessions/ curriculum for Frontier Lab readiness. The lesson structure is good, but the tone is too dry and some math explanations are hard to follow.

Goal:
Create a Coach Layer pilot that makes lessons warmer, more intuitive, bilingual Chinese + English, analogy-rich, math-friendly, and artifact-driven.

Pilot files:
1. sessions/m01-shape-of-data/day-05-logs-and-exponents/lesson.html
2. sessions/m08-transformer-math/day-01-transformer-arithmetic/lesson.html
3. sessions/m08-transformer-math/day-03-kv-cache/lesson.html
4. sessions/m10a-scaling-laws/day-04-power-law-derivation/lesson.html

Tasks:
1. Inspect sessions/, _lesson_gen.py, and lesson_audit.py.
2. Create sessions/COACH_STYLE_GUIDE.md.
3. Create sessions/enhance_lesson_checklist.md.
4. Update lesson_audit.py so Chinese is not treated as degraded.
5. Apply the Coach Layer to only the four pilot lessons.
6. Preserve navigation, localStorage progress, quiz mechanics, playground mechanics, file paths, and self-contained behavior.
7. Create sessions/coach_layer_pilot_report.md.
8. Run available checks and summarize results.

Do not refactor unrelated lessons.
Do not reduce technical depth.
Do not make it motivational fluff.
```
