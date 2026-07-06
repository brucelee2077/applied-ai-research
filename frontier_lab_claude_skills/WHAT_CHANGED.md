# What Changed in v2

## Diagnosis

The original skill pack was good for creating new artifacts:

- new courseware
- new paper notes
- new D3 pages
- new experiments
- weekly portfolio packaging

But the current `sessions/` repo needs a different job:

> safely upgrade existing lesson HTML files without breaking navigation, localStorage progress, quiz behavior, playground behavior, or repo structure.

## v2 changes

### 1. Added curriculum refactor orchestration

New skill:

- `frontier-curriculum-refactor`

It exists to plan and run a safe pilot, not randomly rewrite the whole repo.

### 2. Added coach voice upgrade

New skill:

- `frontier-lesson-humanizer`

It adds:

- warm opening hook
- intuition-first teaching
- everyday analogy
- systems analogy
- where analogy breaks
- Staff / Research Engineer Lens
- interview-ready explanation
- stronger Produce artifact

### 3. Added math clarity layer

New skill:

- `frontier-math-unfogger`

It turns formula-heavy material into:

- problem
- intuition
- symbols
- tiny-number example
- formula
- sanity check
- common mistake
- code connection
- frontier relevance

### 4. Added artifact reviewer

New skill:

- `frontier-artifact-reviewer`

It reviews outputs using:

- correctness
- depth of understanding
- artifact quality
- frontier-lab relevance
- next improvement

### 5. Reformatted and strengthened existing skills

The original skill files had the right intent, but they were too compressed and too generation-oriented. v2 keeps them focused while making them more reliable for Claude Code.
