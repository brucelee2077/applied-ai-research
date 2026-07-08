# Example Prompts

## Refactor existing sessions

```text
Use /frontier-curriculum-refactor.

Goal:
Refactor existing sessions/ lessons into a warm, intuition-first, bilingual Frontier Lab Coach experience.

Start with a 4-lesson pilot:
1. sessions/m01-shape-of-data/day-05-logs-and-exponents/lesson.html
2. sessions/m08-transformer-math/day-01-transformer-arithmetic/lesson.html
3. sessions/m08-transformer-math/day-03-kv-cache/lesson.html
4. sessions/m10a-scaling-laws/day-04-power-law-derivation/lesson.html

Do not modify unrelated files.
Preserve navigation, localStorage progress, quiz behavior, playground behavior, and file paths.
Create sessions/coach_layer_pilot_report.md.
```

## Humanize one lesson

```text
Use /frontier-lesson-humanizer on:
sessions/m10a-scaling-laws/day-04-power-law-derivation/lesson.html

Make it warmer, more intuitive, and easier to understand.
Use Chinese for intuition and English for technical terms.
Add Math Ladder where formulas appear.
Preserve HTML behavior.
```

## Fix math-heavy explanation

```text
Use /frontier-math-unfogger on the power-law derivation section.

For each formula:
- explain the problem it solves
- define every symbol
- use tiny numbers
- add sanity checks
- add common misconception
- connect to frontier-lab scaling decisions
```

## Add or improve visualization

```text
Use /frontier-d3-visual-lab to improve the KV cache visual.

Learning question:
Why does KV cache reduce recomputation but increase memory pressure?

Add:
- prefill/decode timeline
- sequence length slider
- cache memory estimate
- one misconception note
```

## Review artifact

```text
Use /frontier-artifact-reviewer to review:
experiments/kv-cache-sim/EXPERIMENT_LOG.md

Tell me whether this counts as a Frontier Lab learning artifact.
Be warm but strict.
```


## Audit visuals before rollout

```text
Use /frontier-visual-auditor.

Review the playgrounds/visuals in these lessons:
[paths]

For each lesson, classify the visual as:
- Visual mechanism
- Interactive calculator
- Terminal walkthrough
- Static explanation
- Missing visual

Score each from 1–5.
Identify P0 upgrades before batch rollout.
Do not modify files yet.
```

## Upgrade top two visuals

```text
Use /frontier-d3-visual-lab.

Based on the visual audit, upgrade only the top two P0 visuals.
Preserve navigation, localStorage progress, quiz mechanics, playground behavior, and file paths.
Add "What you should notice", "Common misconception", and "Where this visual simplifies reality".
Do not refactor unrelated lessons.
```
