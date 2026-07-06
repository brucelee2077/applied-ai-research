---
name: frontier-experiment-lab
description: Turn a Frontier Lab learning concept into a small reproducible coding experiment with code, tests, run commands, results, and a writeup.
---

# Frontier Experiment Lab

Build tiny experiments that convert understanding into evidence.

## Core principle

Do not build a cathedral.

Build a clean lab bench.

The first experiment should be small enough to run, inspect, and explain.

## Output files

Create:

```text
experiments/<topic-slug>/README.md
experiments/<topic-slug>/src/
experiments/<topic-slug>/tests/
experiments/<topic-slug>/results/
experiments/<topic-slug>/EXPERIMENT_LOG.md
```

If the experiment is very small, a single script is acceptable, but still include a README and experiment log.

## Experiment design template

```markdown
# Experiment: <title>

## Question
What are we trying to learn?

## Hypothesis
What do we expect to happen?

## Minimal setup
What is the smallest version that tests the idea?

## Implementation
Files and commands.

## Results
Numbers, plots, or qualitative output.

## Interpretation
What changed in my understanding?

## Limitations
Why this is not enough yet.

## Next experiment
One concrete follow-up.
```

## Rules

- Keep the first version small.
- Add tests before expanding when practical.
- Prefer deterministic synthetic data first.
- Avoid dependency hell.
- Use simple NumPy/Python before JAX/PyTorch unless the concept requires JAX/PyTorch.
- Save results in a file, not only terminal output.
- Write what failed or confused the learner.

## Good experiments

- implement scaled dot-product attention from scratch
- fit a power-law curve to synthetic scaling data
- simulate KV cache compute savings across decode steps
- compare dense FFN vs MoE routing cost
- quantize a vector and measure reconstruction error
- benchmark batch size vs latency in a toy server
- implement a small JAX training loop
- try a Pallas kernel only after a baseline works

## Claude Code behavior

When implementing:

1. Restate the experiment question.
2. Propose minimal file structure.
3. Write tests or validation checks.
4. Implement the smallest working version.
5. Run tests/commands if possible.
6. Save results.
7. Write the experiment log.
8. Summarize what passed, failed, and confused you.

## Review checklist

- Is the question precise?
- Is the baseline clear?
- Did the result answer the question?
- Is there a plot/table/result file?
- Can future me rerun it in 10 minutes?
- Can the learner explain why it matters for frontier labs?
