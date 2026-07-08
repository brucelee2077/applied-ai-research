# Prompts

## 1. Source-free Module 2 rebuild

```text
Use /frontier-curriculum-architect with xhigh effort.

Run a source-free first-principles refactor for:

- sessions/m02-the-neuron

Important:
Do not read, inspect, summarize, or rely on old notebooks during generation.
Do not use:
- 00-neural-networks/fundamentals/*.ipynb

Treat the existing sessions code as:
- shell/HTML structure
- existing navigation and behavior
- current rough draft to improve

Do not treat existing lesson content as the source of truth.

Goal:
Rebuild Module 2 into a notebook-quality foundation module that feels like a warm Frontier Lab coach teaching a senior software engineer new to ML training.

Before editing lesson files:
1. Create sessions/m02_first_principles_blueprint.md.
2. Create sessions/m02_coverage_contract.md.
3. Create sessions/m02_visual_contract.md.
4. Create sessions/m02_artifact_contract.md.
5. Create sessions/m02_refactor_plan.md.
6. Do not edit lesson files until these exist.

After editing:
1. Use /frontier-refactor-qa.
2. Create sessions/m02_refactor_report.md.
```

## 2. Optional held-out eval after generation

```text
Use /frontier-refactor-qa with xhigh effort.

Now run a held-out evaluation.

You may compare the newly generated Module 2 against:
- 00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb
- 00-neural-networks/fundamentals/02_single_neuron.ipynb
- 00-neural-networks/fundamentals/03_activation_functions.ipynb

Important:
Do not edit lesson files.

Create:
- sessions/m02_heldout_eval_report.md

Evaluate:
- intuition quality
- analogy depth
- concept coverage
- visual depth
- experiment quality
- beginner friendliness
- technical correctness
- frontier-lab relevance
- artifact quality
- what the skills failed to specify
```

## 3. Large refactor batch

```text
Use /frontier-curriculum-architect with xhigh effort.

Run a source-free first-principles refactor for this batch:
- sessions/<module-a>
- sessions/<module-b>

Do not use old notebooks or prior courseware as source material during generation.
Create first-principles blueprint, coverage contract, visual contract, artifact contract, and refactor plan before editing.

Preserve existing shell behavior.
After editing, run /frontier-refactor-qa.
```
