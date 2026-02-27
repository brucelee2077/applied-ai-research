# Claude Instructions for this Repo

## Explanation Style

When explaining concepts in this repo — whether in notebooks, comments, markdown files, or conversation — explain them as if the reader is a 12-year-old encountering the idea for the first time.

**What this means in practice:**

- Lead with a concrete everyday analogy before introducing any math or jargon. If you can't think of a good analogy, find one before writing anything else.
- Introduce one idea at a time. Don't stack multiple new concepts in the same sentence.
- When a technical term is unavoidable, immediately follow it with a plain-English definition in parentheses or on the next line.
- Prefer short sentences. If a sentence needs a semicolon, split it into two sentences.
- Avoid phrases like "it is worth noting", "trivially", "clearly", or "simply" — nothing is obvious to someone seeing it for the first time.
- Use "you" and active voice. "The gradient flows backward" is better than "backpropagation is performed".
- When writing math, always say in words what the equation means before showing the symbols.

**Examples of the bar to hit:**

- Bad: "The vanishing gradient problem arises due to the repeated application of the chain rule through saturating non-linearities."
- Good: "Every time the error signal travels one step back in time, it gets multiplied by a small number. Do that 50 times and the signal is basically gone — the network can't learn from things that happened far in the past."

- Bad: "The cell state provides an additive gradient pathway."
- Good: "Think of the cell state as a conveyor belt. Information placed on it at the start can ride all the way to the end without being changed at every station."

## Depth: Two Layers in Every Explanation

Every explanation must work at two levels simultaneously.

**Layer 1 — The 12-year-old layer** (always comes first)
Lead with the analogy. Make the intuition land before any symbols appear. See the rules above.

**Layer 2 — The staff/principal ML engineer layer** (always follows)
After the intuition is clear, go deep. This reader is preparing for or conducting Staff/Principal MLE interviews at top-tier companies. They need:

- **Precise mathematics.** Write out the full equations. Explain what each term does and why it is there. Don't hand-wave.
- **Failure modes and edge cases.** When does this break? What hyperparameter choice causes silent badness? What does the loss curve look like when something goes wrong?
- **Complexity analysis.** Time, memory, and parameter count. Know the exact formulas, not just O(n²) vs O(n).
- **Design decisions and trade-offs.** Why was this design chosen over alternatives? What does each choice cost and gain? What would you change and why?
- **Connections across the field.** How does this idea relate to other architectures, training tricks, or theoretical results? Where does it appear in production systems?
- **Interview-grade answers.** After each major concept, include a "Staff/Principal Interview Depth" section with 3–5 questions a senior interviewer would ask, and full answers that would satisfy them. These are not recall questions — they are judgment and depth questions.

**How the two layers fit together:**

```
[Analogy — 12-year-old intuition]
[Core concept — clean and concrete]
[Full mathematical treatment — precise and complete]
[Failure modes, edge cases, hyperparameter sensitivity]
[Design trade-offs and alternatives]
[Production and scaling considerations]
[Staff/Principal Interview Depth — Q&A]
```

Never skip a layer. A reader who only reads the analogy must still get value. A reader who wants to go deep must find enough to satisfy a panel interview.

**Examples of the staff/principal bar:**

- Bad: "Multi-head attention lets the model attend to different things."
- Good: "Multi-head attention projects Q, K, V into h subspaces of dimension d_k = d_model/h. Each head computes Attention(QW_i^Q, KW_i^K, VW_i^V) independently and in parallel. The outputs are concatenated and projected: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O. Total parameter count for the attention block: 4 * d_model^2 (three input projections + output projection). Total compute: O(n^2 * d_model) — the n^2 term is why long contexts are expensive. The design choice to project into smaller subspaces rather than run h full-dimension attention heads keeps total FLOPs identical to single-head attention while enabling specialization. Key failure mode: heads can collapse — multiple heads learn identical patterns, wasting capacity. Solutions: attention dropout, diverse initialization, or explicit diversity losses."

- Bad: "The forget gate controls what the LSTM remembers."
- Good: "The forget gate f_t = sigmoid(W_f[h_{t-1}, x_t] + b_f) outputs values in (0,1) element-wise. The cell state update C_t = f_t * C_{t-1} + i_t * C_tilde means dC_t/dC_{t-1} = f_t — no matrix multiply, no non-linearity. Over k steps, the gradient is the product of k forget gate values. If the network learns f_t ≈ 1 for important memory slots, this product stays near 1 regardless of sequence length. The forget gate bias is initialized to 1 (not 0) so sigmoid(1) ≈ 0.73 — the LSTM defaults to remembering. Jozefowicz et al. (2015) showed this initialization is critical for learning long-range dependencies from scratch. At inference, you can inspect forget gate values to understand what the model is remembering and forgetting — this is a practical debugging tool."

## Code Style

- Keep variable names descriptive. `forget_gate` is better than `f`.
- Add a one-line comment above any line that does something non-obvious.
- Print shapes and intermediate values in tutorial code so learners can see what's happening.

## Notebook Conventions

- Every notebook starts with the COACH session start cell and ends with the COACH session end cell. Do not remove these.
- New concepts go in markdown cells before the code that uses them — explain first, code second.
- Visualizations are preferred over tables of numbers when both are possible.

## Notebook Validation (Required)

After creating or modifying any notebook, you must execute it end-to-end and confirm every cell runs without error before considering the task done.

**How to do this:**

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=120 <notebook_path>
```

**What counts as done:**
- Exit code 0 from the command above
- No `CellExecutionError` in the output
- The notebook's output cells are populated with actual results (not empty)

**If a cell fails:**
- Fix the error in the notebook source
- Re-run the full notebook from the top (not just the failing cell — earlier cells may produce state the failing cell depends on)
- Repeat until the full notebook executes cleanly

**Do not skip this step.** A notebook that crashes on cell 3 is not a finished notebook, even if the code looks correct on inspection.
