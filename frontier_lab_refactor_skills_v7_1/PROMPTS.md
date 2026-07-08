# v7.1 Prompts

## 1. Install then run v7.1 QA on Module 2

```text
Use /frontier-refactor-qa with xhigh effort.

Run v7.1 QA on:

- sessions/m02-the-neuron

Do not read old notebooks.
Do not edit lesson files yet.

Apply the updated v7.1 gates:

1. Coverage traceability
   - For every must-cover concept, create:
     Concept | Required depth | Lesson evidence | Visual/evidence | Artifact | Result
   - PASS / PARTIAL / FAIL only.
   - PARTIAL/FAIL on must-cover blocks merge unless explicitly waived.

2. Foundation Framing Lens
   - Check neural-network foundation lessons for:
     brain-inspired intuition,
     artificial-neuron caveat,
     mapping table,
     where the analogy breaks,
     transition to math function.
   - Generic everyday analogies do not satisfy this requirement.

3. Function Family Rule
   - Check mechanism families for representative variants, comparison table, failure modes, and modern context.
   - For activations, expect sigmoid, tanh, ReLU, Leaky ReLU, softmax, derivative intuition, saturation, vanishing gradient, dying ReLU, and GELU/SwiGLU context unless explicitly out of scope.

4. Visual/Evidence Gate
   - Simulated terminal is not evidence unless the Produce artifact reproduces the same behavior.
   - Failure modes require visual or runnable evidence.
   - Quantitative visuals require numeric readouts.

5. Anchor Consistency Gate
   - Check consistency across prose, visual, playground, quiz, Produce, and acceptance criteria.
   - Track step counts, split ratios, seeds, learning rates, dataset sizes, sequence lengths, batch sizes, parameter values, and metric definitions.

6. Artifact Gate
   - exact path
   - exact run command
   - expected output
   - acceptance criteria
   - explain-back

7. Legacy skill reference check
   - Detect old / removed skill names.

Create:
- sessions/m02_v7_1_gap_report.md

At the end, propose a P0 fix plan based only on failed gates.
Do not edit lesson files.
```

## 2. Apply only the P0 fixes discovered by v7.1

```text
Use /frontier-refactor-qa with xhigh effort.

Apply only the P0 fixes identified in:
- sessions/m02_v7_1_gap_report.md

Scope:
- sessions/m02-the-neuron only
- m02 contract/report files if needed
- m02 experiment files if needed

Do not read old notebooks.
Do not do broad prose polish.
Do not fix P2.
Do not modify shared shell files.
Do not change data-quest-id values.

After editing:
- Run v7.1 QA again.
- Run lesson_audit.py on m02.
- Run nav/jsdom/node checks if available.
- Create sessions/m02_v7_1_p0_fix_report.md.

Goal:
Move Module 2 from Blocked to Pass or Pass with P1.
```

## 3. Held-out eval after P0 fixes

```text
Use /frontier-refactor-qa with xhigh effort.

Run held-out eval after v7.1 P0 fixes.

You may now compare against old notebooks.
Do not edit lesson files.

Create:
- sessions/m02_v7_1_heldout_eval_report.md

The eval must include:
- where lessons win
- where notebooks win
- where lessons overclaim
- where beginner intuition is weaker
- where coverage is missing
- where visual/evidence is weaker
- what skills still failed to specify
- whether v7.1 should be patched again before broader rollout
```

## 4. Big batch refactor only after Module 2 passes

```text
Use /frontier-curriculum-architect with xhigh effort.

Run a source-free first-principles refactor for this batch:
- sessions/<module-a>
- sessions/<module-b>

Do not read old notebooks or prior courseware unless explicitly asked.
Treat current sessions as shell constraints and rough drafts.

Before editing:
- coverage discovery
- coverage contract
- coach voice contract
- visual/evidence contract
- artifact contract
- refactor plan

After editing:
- use /frontier-refactor-qa
- do held-out eval only if asked
```
