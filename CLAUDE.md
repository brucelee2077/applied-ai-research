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
- Test your analogy before using it. A good analogy captures the *mechanism*, not just the surface. Ask: does the analogy behave the same way as the real thing when you push it? If the real concept gets harder with more inputs, so should the analogy.
- Always state where the analogy breaks down. Every analogy has a limit. One sentence like "This analogy breaks down because..." prevents the reader from building the wrong mental model.
- Use analogies from things a 12-year-old has actually experienced: physical objects, games, daily routines, social situations. Avoid analogies that require domain knowledge (factories, financial instruments, etc.).

**Scaffold for every Layer 1 explanation:**

```
1. Analogy — a concrete, experienced thing that works the same way
2. What the analogy gets right — explicitly name the parallel
3. The concept in plain words — no math, no jargon
4. Where the analogy breaks down — one sentence, explicitly flagged
```

**Examples of the bar to hit:**

- Bad: "The vanishing gradient problem arises due to the repeated application of the chain rule through saturating non-linearities."
- Good: "Every time the error signal travels one step back in time, it gets multiplied by a small number. Do that 50 times and the signal is basically gone — the network can't learn from things that happened far in the past."

- Bad: "The cell state provides an additive gradient pathway."
- Good: "Think of the cell state as a conveyor belt. Information placed on it at the start can ride all the way to the end without being changed at every station."

- Bad: "The attention mechanism computes a weighted average over value vectors."
- Good: "Imagine you're reading a sentence and trying to understand the word 'bank'. You automatically look back at the other words — 'river', 'fish', 'swim' — to figure out which meaning fits. Attention does exactly this: for each word, it looks back at all the other words and decides how much each one matters. The analogy breaks down because a real reader processes words one at a time — attention looks at all words simultaneously."

- Bad: "Dropout regularizes the network by randomly zeroing activations during training."
- Good: "Think of a sports team that always relies on its star player. If that player gets hurt, the team falls apart. A coach who makes every player practice as if the star might not show up builds a more resilient team. Dropout does the same thing: during training, it randomly 'sits out' neurons so the rest of the network can't lean on any single one. The analogy breaks down because the neurons aren't practicing independently — the network still trains as a whole, just with a different subset each time."

## Engagement & Motivation

Learning AI/ML is hard. The math is real, the papers are dense, and it's easy to feel like you're not capable. These instructions exist to fight that feeling — to make every concept feel like a win, every equation feel like a puzzle worth solving, and every session leave the reader wanting more.

### 1. Open with a curiosity hook — not the concept

Before explaining anything, create a "why do I need to know this?" moment. Set up a mystery or a surprising fact that makes the reader lean in. Never open cold with the definition.

- Bad: "Attention mechanisms are how transformers decide which tokens to focus on."
- Good: "Here's something strange: a neural network trained only on text somehow learned to play chess, write code, and diagnose diseases. How? It all comes down to one idea — attention. Let's figure out what that means."

Every new major concept must open with a hook — a question, a surprising fact, or a short story about why this matters.

### 2. Normalize confusion before it hits

When something is genuinely hard, say so explicitly — and frame it as a sign of seriousness, not inability. The reader should feel like they're in good company when they struggle.

- Bad: *(the equation just appears with no warning)*
- Good: "Fair warning: this equation looks scary the first time. That's normal. Even researchers who use this every day had to sit with it for a while. Give yourself permission to read it twice."

Before any equation or concept with high cognitive load, add one sentence acknowledging it's hard and normalizing that. Never let the reader feel alone in the struggle.

### 3. Give victory laps after hard concepts

After the reader gets through something difficult, tell them what they just unlocked. Connect it to something impressive they've heard of. Make the win feel real.

- Bad: *(just move on to the next section)*
- Good: "You just understood the core mechanism behind GPT-4, Claude, and every large language model released in the last 5 years. That's not an exaggeration. The attention mechanism you just learned IS the transformer. You now have the key."

After every major concept, add a short "Victory lap" — 1–3 sentences connecting what the reader just learned to something real and impressive in the world.

### 4. Write like a brilliant friend, not a textbook

The tone should feel like a friend who is better at this than you, genuinely believes you can get it, and is rooting for you. Not a professor who tolerates questions. Not a textbook that pretends everything is obvious.

**Banned phrases — these signal intimidation or dismissal:**
- "As you can see..."
- "Trivially..."
- "It is straightforward to show..."
- "Obviously..."
- "Recall that..."
- "It is left as an exercise..."
- "This is just..."

Before outputting any sentence, ask: "Would a friend say this?" If it sounds like an exam paper, rewrite it.

### 5. Frame progress as leveling up — not obstacles to clear

Every new concept is something the reader is *gaining*, not a test they might fail. Frame learning as an adventure where the destination gets more impressive the further you go.

- Bad: "Before we can understand transformers, we need to cover attention, positional encoding, and multi-head attention."
- Good: "You're about to build a transformer from scratch. Each piece we pick up along the way — attention, positional encoding, multi-head attention — goes straight into your inventory. By the end, you'll have the full thing assembled."

When introducing a sequence of concepts, frame them as collectibles the reader is accumulating, not prerequisites they must survive.

---

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
- **Interview-grade answers.** After each major concept, include a "Staff/Principal Interview Depth" section with 3–5 questions a senior interviewer would ask. These are not recall questions — they are judgment and depth questions.

  For each question, show all four hiring levels. At each level: first write the interviewee's answer verbatim (how a candidate at that level would actually speak), then provide narrative feedback from the interviewer, then end with an explicit criteria summary of what was demonstrated vs. missing.

  Format per question:

  **Q: [question text]**

  ---
  **No Hire**
  *Interviewee:* [what this answer sounds like — typically surface-level, misses the point, or reveals a misconception]
  *Interviewer:* [narrative — what the interviewer observes, what signal this gives, what the candidate failed to demonstrate]
  *Criteria — Met:* none / *Missing:* [list the key criteria this tier failed to show]

  **Weak Hire**
  *Interviewee:* [correct at a high level but shallow — no math, no trade-offs, no failure modes]
  *Interviewer:* [narrative — what's present vs. missing, why this doesn't clear the staff bar]
  *Criteria — Met:* [what's present] / *Missing:* [what's still absent]

  **Hire**
  *Interviewee:* [solid answer — correct math, at least one failure mode, a real trade-off]
  *Interviewer:* [narrative — what makes this a hire, what the candidate demonstrated, what would push it to Strong Hire]
  *Criteria — Met:* [list] / *Missing:* [what would elevate to Strong Hire]

  **Strong Hire**
  *Interviewee:* [full depth — precise equations, multiple failure modes, connects to production systems or related work, offers original judgment]
  *Interviewer:* [narrative — what distinguishes this from Hire, what signals staff-level thinking, why this is a clear yes]
  *Criteria — Met:* [full list, nothing missing]
  ---

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
