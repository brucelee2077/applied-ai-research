# v7.1 Patch Report

## Why this patch exists

Module 2 v7 QA successfully found real failures, but it also exposed that the skills still needed stronger first-principles rules.

The main issue: the system could detect some problems after the fact, but the underlying skills did not yet make those problems structurally impossible.

## Patches added

### 1. Coverage Engine strengthened

Added:

- Foundation Framing Lens
- Anchor Consistency Lens
- stronger Function Family Rule
- failure-mode evidence requirement
- good-enough coverage criteria

### 2. Lesson Builder strengthened

Added:

- first-20-percent Coach Voice Gate
- Neural Foundation Rule
- Function Family Rule
- Reader-objection Q&A
- Jargon Buster requirement
- derivative-alongside-function rule
- evidence-first failure-mode rule

### 3. Visual Evidence Builder strengthened

Added:

- simulated terminal is not evidence by itself
- failure demos required
- on-panel numeric readouts required
- parameter-isolation small multiples
- benchmark rule
- Produce artifact must reproduce playground behavior

### 4. QA strengthened

Added:

- coverage traceability table
- anchor consistency gate
- stricter brain-vs-ANN framing
- P0 if must-cover concept is PARTIAL/FAIL
- adversarial held-out eval standard
- no over-crediting Staff Lens when beginner fundamentals are weak

## Intended next action

Install v7.1, then run report-only v7.1 QA on Module 2:

```text
Use /frontier-refactor-qa with xhigh effort.

Run v7.1 QA on:
- sessions/m02-the-neuron

Do not read old notebooks.
Do not edit lesson files yet.

Create:
- sessions/m02_v7_1_gap_report.md
```
