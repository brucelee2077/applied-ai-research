# Chain-of-Thought Prompting

## What is Chain-of-Thought?

Remember when your math teacher said **"Show your work!"**? Chain-of-Thought (CoT)
prompting is exactly that, but for AI.

Instead of asking the AI to jump straight to the answer, you ask it to **think through
the problem step by step**. This makes the AI much better at solving tricky problems
like math, logic puzzles, and multi-step reasoning.

```
┌──────────────────────────────────────────────────────────────────┐
│              Without Chain-of-Thought (Direct Answer)            │
│                                                                  │
│   Prompt: "If I have 3 apples and give away half, then buy 5    │
│            more, how many do I have?"                            │
│                                                                  │
│   AI: "6.5"  ← Wrong! AI tried to jump to the answer            │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│              With Chain-of-Thought (Step by Step)                │
│                                                                  │
│   Prompt: "If I have 3 apples and give away half, then buy 5    │
│            more, how many do I have?                             │
│            Let's think step by step."                            │
│                                                                  │
│   AI: "Step 1: I start with 3 apples.                           │
│         Step 2: I give away half: 3 / 2 = 1.5 apples given,     │
│                 so I keep 1.5.                                   │
│         Step 3: But wait -- you can't have half an apple in      │
│                 real life. Giving away 'half' of 3 likely means  │
│                 1 apple (rounding down), keeping 2.              │
│         Step 4: I buy 5 more: 2 + 5 = 7 apples.                │
│         Answer: 7 apples."  ← Better reasoning!                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Why Does This Work?

Think about it this way. When you solve a hard math problem in your head, you might
get it wrong. But when you write each step down on paper, you catch your mistakes.

The same thing happens with AI:

```
┌──────────────────────────────────────────────────────────┐
│              Why Step-by-Step Helps                       │
│                                                          │
│   Direct answer:                                         │
│   Problem ──────────────────────────────> Answer         │
│            (big jump = more room for error)               │
│                                                          │
│   Chain-of-thought:                                      │
│   Problem ──> Step 1 ──> Step 2 ──> Step 3 ──> Answer   │
│            (small jumps = fewer errors at each step)      │
└──────────────────────────────────────────────────────────┘
```

Each small step is easy to get right. By chaining many easy steps together, the AI
can solve much harder problems.

---

## Two Flavors of Chain-of-Thought

### 1. Zero-Shot CoT (The Easy Way)

Just add a magic phrase like **"Let's think step by step"** to the end of your prompt.
No examples needed.

```
Prompt:
  "A store has 15 shirts. They sell 1/3 on Monday and 1/5 on Tuesday.
   How many shirts are left?

   Let's think step by step."

AI Response:
  "Step 1: Start with 15 shirts.
   Step 2: Sold 1/3 on Monday: 15 × (1/3) = 5 shirts sold.
   Step 3: Shirts left after Monday: 15 - 5 = 10.
   Step 4: Sold 1/5 on Tuesday: 15 × (1/5) = 3 shirts sold.
   Step 5: Shirts left after Tuesday: 10 - 3 = 7.
   Answer: 7 shirts remain."
```

**Other magic phrases that work:**
- "Let's think step by step."
- "Let's work through this carefully."
- "Think about this step by step before answering."
- "Break this problem into smaller parts."
- "Explain your reasoning."

### 2. Few-Shot CoT (More Control)

You give the AI an example of step-by-step reasoning, then ask your actual question.
This gives you more control over **how** the AI thinks through the problem.

```
Prompt:
  "I'll show you how to solve word problems step by step.

   Question: Tom has 12 cookies. He eats 4 and gives 3 to his sister.
   How many does he have left?

   Reasoning:
   - Tom starts with 12 cookies
   - He eats 4, so: 12 - 4 = 8 cookies
   - He gives 3 to his sister: 8 - 3 = 5 cookies
   Answer: 5 cookies

   Question: A parking lot has 20 spaces. 7 cars are parked there in the
   morning. 4 leave at noon and 9 arrive in the afternoon.
   How many empty spaces are left?

   Reasoning:"

AI Response:
  "- The lot has 20 total spaces
   - Morning: 7 cars parked, so 20 - 7 = 13 empty spaces
   - At noon: 4 cars leave, so 7 - 4 = 3 cars remaining
   - Afternoon: 9 more arrive, so 3 + 9 = 12 cars total
   - Empty spaces: 20 - 12 = 8
   Answer: 8 empty spaces"
```

---

## Zero-Shot CoT vs Few-Shot CoT

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│                    │   Zero-Shot CoT      │   Few-Shot CoT       │
├────────────────────┼──────────────────────┼──────────────────────┤
│ How to use         │ Add "Let's think     │ Give example(s) of   │
│                    │ step by step"        │ step-by-step answers │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Effort             │ Very easy            │ More work to set up  │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Control over       │ Low -- AI decides    │ High -- you show the │
│ reasoning style    │ how to reason        │ reasoning style      │
├────────────────────┼──────────────────────┼──────────────────────┤
│ When to use        │ Quick experiments,   │ Consistent format,   │
│                    │ general problems     │ specific domain      │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Accuracy           │ Good                 │ Better               │
└────────────────────┴──────────────────────┴──────────────────────┘
```

**Rule of thumb:** Start with zero-shot CoT (just add "Let's think step by step").
If the quality isn't good enough, upgrade to few-shot CoT with examples.

---

## What Kinds of Problems Benefit from CoT?

Chain-of-thought helps most with problems that need **multiple steps of reasoning**.
It doesn't help much for simple recall or lookup tasks.

```
┌──────────────────────────────────────────────────────────────┐
│                  When to Use CoT                              │
│                                                              │
│   HELPS A LOT:                                               │
│   ✓ Math word problems     ("If train A leaves at...")       │
│   ✓ Logic puzzles          ("Who sits next to whom?")        │
│   ✓ Multi-step reasoning   ("First X, which means Y...")     │
│   ✓ Code debugging         ("Trace through this code...")    │
│   ✓ Complex decisions      ("Weigh pros and cons of...")     │
│                                                              │
│   DOESN'T HELP MUCH:                                         │
│   ✗ Simple factual Q&A     ("What's the capital of France?") │
│   ✗ Translation             ("Translate 'hello' to Spanish") │
│   ✗ Basic text generation  ("Write a haiku about cats")      │
│   ✗ Sentiment analysis     ("Is this review positive?")      │
└──────────────────────────────────────────────────────────────┘
```

---

## Advanced: Self-Consistency (Voting for the Best Answer)

Here's a cool trick built on top of chain-of-thought. What if you asked the AI the
same question **multiple times**, and each time it reasons slightly differently? You
then pick the answer that shows up the most. This is called **self-consistency**.

Think of it like asking 5 friends to solve a math problem independently and going
with the answer most of them agree on.

```
┌──────────────────────────────────────────────────────────────────┐
│                    Self-Consistency                               │
│                                                                  │
│   Same question asked 5 times with temperature > 0:              │
│                                                                  │
│   Run 1: Step A → Step B → Step C → Answer: 7  ←─┐              │
│   Run 2: Step X → Step Y → Step Z → Answer: 7  ←─┤ Majority    │
│   Run 3: Step D → Step E → Step F → Answer: 8    │ = 7          │
│   Run 4: Step G → Step H → Step I → Answer: 7  ←─┤              │
│   Run 5: Step J → Step K → Step L → Answer: 9    │              │
│                                                   │              │
│   Final Answer: 7 (appeared 3 out of 5 times)  ◄──┘              │
└──────────────────────────────────────────────────────────────────┘
```

**When to use self-consistency:**
- When accuracy is critical (e.g., medical, financial, or legal reasoning)
- When you're willing to spend more tokens (you're paying for multiple runs)
- Math and logic problems where there's one correct answer

---

## Real-World Examples

### Example 1: Debugging Logic

```
Prompt:
  "A user reports that their shopping cart total is wrong.
   The items are: Book ($12), Pen ($3), Notebook ($8).
   There's a 10% discount if you buy 3+ items, and 5% tax.
   The user was charged $25.30.

   Let's verify if this is correct step by step."

AI Response:
  "Step 1: Add up the items: $12 + $3 + $8 = $23.00
   Step 2: The user has 3 items, so 10% discount applies.
   Step 3: Discount: $23.00 × 10% = $2.30
   Step 4: Subtotal after discount: $23.00 - $2.30 = $20.70
   Step 5: Tax: $20.70 × 5% = $1.035, rounded to $1.04
   Step 6: Total: $20.70 + $1.04 = $21.74

   The user was charged $25.30, but the correct total is $21.74.
   It looks like the discount was not applied."
```

### Example 2: Decision Making

```
Prompt:
  "I need to choose between Python and JavaScript for a data analysis project.
   The team has 2 Python developers and 1 JavaScript developer.
   We need to process 10GB of CSV files and create visualizations.

   Think through this step by step to recommend the best choice."

AI Response:
  "Step 1: Task requirements -- data processing (10GB CSV) and visualization.
   Step 2: Python has pandas, numpy, matplotlib, seaborn -- all built
           specifically for data analysis.
   Step 3: JavaScript has D3.js for visualization but lacks mature data
           processing libraries for large files.
   Step 4: Team composition: 2 Python devs vs 1 JS dev -- more people
           can work on it with Python.
   Step 5: 10GB of data is large. Python's pandas with chunked reading
           handles this well. JS would struggle in the browser.
   Step 6: Conclusion.

   Recommendation: Python. It has better data processing libraries
   (pandas, numpy), more team members available, and handles large
   datasets more efficiently."
```

### Example 3: Reading Comprehension

```
Prompt:
  "Passage: The city of Riverside decided to build a new park in 2020.
   The construction was delayed by 18 months due to supply issues.
   The park opened 6 months after construction finished. Construction
   took 2 years once it finally started.

   Question: When did the park open?

   Let's work through this step by step."

AI Response:
  "Step 1: The project was planned for 2020.
   Step 2: Construction was delayed by 18 months.
           So construction started: 2020 + 18 months = mid-2021.
   Step 3: Construction took 2 years: mid-2021 + 2 years = mid-2023.
   Step 4: Park opened 6 months after construction finished:
           mid-2023 + 6 months = early 2024.

   Answer: The park opened in early 2024."
```

---

## Tips for Better Chain-of-Thought

| Tip | Example |
|-----|---------|
| Be specific about what "step by step" means | "Break this into: (1) identify knowns, (2) identify unknowns, (3) solve" |
| Ask for the answer at the end | "After reasoning, put your final answer on a new line starting with 'Answer:'" |
| Use CoT for the hard part, skip it for the easy part | Only apply CoT to the complex sub-problem, not the whole task |
| Combine with few-shot | Show an example of good step-by-step reasoning, then ask your question |
| Ask the AI to double-check | "After solving, verify your answer by working backwards" |

---

## Key Takeaways

1. **Chain-of-thought** = making the AI "show its work" by thinking step by step
2. **Zero-shot CoT**: just add "Let's think step by step" (easy, works surprisingly well)
3. **Few-shot CoT**: show the AI an example of good reasoning (more work, better results)
4. **Use CoT for hard problems** -- math, logic, multi-step tasks -- not simple lookups
5. **Self-consistency** = ask multiple times + pick the most common answer (higher accuracy)

---

## Further Reading

- **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** -- Wei et al., 2022
  - The original paper that introduced chain-of-thought prompting and showed it dramatically
    improves performance on arithmetic, commonsense, and symbolic reasoning tasks
- **Large Language Models are Zero-Shot Reasoners** -- Kojima et al., 2022
  - Showed that simply adding "Let's think step by step" (zero-shot CoT) works almost as
    well as carefully crafted few-shot examples
- **Self-Consistency Improves Chain of Thought Reasoning** -- Wang et al., 2022
  - Introduced the idea of sampling multiple reasoning paths and taking a majority vote

---

[Back to Techniques Overview](./README.md) | [Previous: Few-Shot Learning](./few-shot-learning.md) | [Next: Prompt Templates](./prompt-templates.md)
