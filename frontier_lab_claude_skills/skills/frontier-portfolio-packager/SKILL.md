---
name: frontier-portfolio-packager
description: Convert weekly Frontier Lab study outputs into a polished portfolio artifact: technical writeup, README, demo page, repo summary, or interview story. Use during weekly review or when packaging experiments/courseware for public/private portfolio use.
---

# Frontier Portfolio Packager

Turn messy learning outputs into career signal.

## Output options
Depending on the user's request, create one of:

```text
portfolio/week-<nn>-summary.md
writeups/<topic-slug>.md
portfolio/index.html
README.md update
interview-stories/<topic-slug>.md
```

## Weekly portfolio structure
```markdown
# Week <N>: <theme>

## What I studied

## What I built

## Main technical insight

## Visual artifact

## Experiment result

## What failed

## Why this matters for frontier labs

## Interview-ready explanation

## Next week
```

## Blog/writeup style
Make the learner sound like a research engineer:
- Problem-first
- Mechanism-driven
- Evidence-based
- Honest about limitations
- Clear about next experiments

Avoid sounding like:
- motivational diary
- generic ML summary
- “I followed a tutorial”
- empty AI hype

## Interview story format
```markdown
Situation:
I wanted to understand <concept> because <frontier relevance>.

Task:
I built <artifact>.

Action:
I implemented <mechanism>, tested <hypothesis>, and measured <result>.

Result:
I learned <specific insight> and identified <next limitation>.

Frontier relevance:
This maps to <role/team/work> because <reason>.
```

## Done definition
The packaged artifact should be understandable by another engineer without reading the whole repo.
