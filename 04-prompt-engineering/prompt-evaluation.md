# Prompt Evaluation

Two prompts look equally good when you test them on one example. You pick one and deploy it. A week later, you discover it fails on 40% of real inputs. The other prompt would have failed on only 5%. How would you have known? You needed evaluation — a way to measure prompt quality before you commit to one.

---

**Before you start, you need to know:**
- What a prompt is and how to write one — covered in [README.md](./README.md)
- How few-shot examples, chain-of-thought, and templates work — covered in [few-shot-learning.md](./few-shot-learning.md), [chain-of-thought.md](./chain-of-thought.md), [prompt-templates.md](./prompt-templates.md)

---

## Why Evaluate Your Prompts?

**The analogy: baking cookies.** Imagine you are baking cookies. You taste one, tweak the recipe, bake again, taste again. You do not just guess whether the cookies are good — you **test** them. Prompt evaluation works the same way: you systematically check whether your prompts are giving you good results, and you use that information to improve them.

**What the analogy gets right:** you change one ingredient at a time, taste the result, and decide whether it got better or worse. That is exactly the loop in prompt evaluation — change the prompt, measure the result, decide if it improved.

**The concept in plain words:** prompt evaluation means measuring how well your prompt works across many inputs, not just one. You create test cases, run your prompt on all of them, score the results, and use the scores to improve.

**Where the analogy breaks down:** with cookies, you taste them yourself and your judgment is enough. With prompts, your judgment on one example can be misleading. You need to test on many examples, and sometimes you need automated scoring because checking hundreds of outputs by hand is not practical.

```
┌──────────────────────────────────────────────────────────────────┐
│                   Why Evaluation Matters                          │
│                                                                  │
│   Without evaluation:                                            │
│   Prompt v1 ──> "Seems okay?" ──> Prompt v2 ──> "Maybe better?" │
│                  (guessing)                      (still guessing)│
│                                                                  │
│   With evaluation:                                               │
│   Prompt v1 ──> Score: 6/10 ──> Prompt v2 ──> Score: 8/10       │
│                 (measured)                     (measured)         │
│                 "Accuracy is low"              "Much better!"    │
└──────────────────────────────────────────────────────────────────┘
```

---

**Quick check — can you answer these?**
- Why is testing a prompt on one example not enough?
- What is the difference between guessing and measuring prompt quality?

If you cannot answer one, re-read the section above. That is completely normal.

---

## Simple Evaluation Methods

You do not need fancy tools to start evaluating. Here are methods from simplest to most advanced:

### Method 1: The Eyeball Test (Manual Review)

The simplest approach: **read the AI's output and judge it yourself.**

Create a checklist for what "good" looks like:

```
┌──────────────────────────────────────────────────────────┐
│              Manual Evaluation Checklist                   │
│                                                          │
│   For each AI response, ask:                             │
│                                                          │
│   □ Is the answer factually correct?                     │
│   □ Does it follow the requested format?                 │
│   □ Is it the right length? (not too short, not too long)│
│   □ Is it clear and easy to understand?                  │
│   □ Would I be comfortable sharing this with someone?    │
│                                                          │
│   Score: Count the checkmarks out of 5                   │
└──────────────────────────────────────────────────────────┘
```

**When to use:** Early prototyping, small number of prompts, tasks where quality is subjective.

### Method 2: Test Cases (Like a Quiz)

Create a set of **test inputs with known correct answers**, then see how many the AI gets right. Think of it like giving the AI a quiz.

```
Test Cases for a "Sentiment Classifier" prompt:

   Input                                    Expected    AI Got    Correct?
   ─────────────────────────────────────────────────────────────────────
   "I love this product!"                   POSITIVE    POSITIVE    ✓
   "Terrible experience, never again"       NEGATIVE    NEGATIVE    ✓
   "It's okay I guess"                      NEUTRAL     POSITIVE    ✗
   "Best purchase I've ever made!"          POSITIVE    POSITIVE    ✓
   "Waste of money"                         NEGATIVE    NEGATIVE    ✓

   Accuracy: 4/5 = 80%
```

**When to use:** Tasks with clear right/wrong answers (classification, extraction, math, factual Q&A).

### Method 3: Side-by-Side Comparison (A/B Testing)

Run two different prompts on the **same inputs** and compare which one gives better results. This is called **A/B testing**.

```
┌──────────────────────────────────────────────────────────────────┐
│                      A/B Testing                                 │
│                                                                  │
│   Same question, two different prompts:                          │
│                                                                  │
│   PROMPT A (simple):                                             │
│   "Summarize this article."                                      │
│   Result: "The article discusses climate change and its effects  │
│            on polar bears..." (Score: 6/10)                      │
│                                                                  │
│   PROMPT B (detailed):                                           │
│   "Summarize this article in 3 bullet points. Each point should  │
│    be one sentence. Focus on actionable takeaways."              │
│   Result: "• Rising temperatures threaten polar bear habitats    │
│            • Ice coverage has decreased 13% per decade           │
│            • Conservation efforts need increased funding"        │
│            (Score: 9/10)                                         │
│                                                                  │
│   Winner: Prompt B                                               │
└──────────────────────────────────────────────────────────────────┘
```

**Important:** Always test on **multiple inputs** (at least 10-20), not just one. A prompt might work great on one example but fail on others.

---

## Evaluation Metrics (Ways to Score)

Metrics are ways to measure how good the answer is. Here are the most common ones:

### Accuracy

**What it measures:** How often the AI gets the right answer.

```
Formula:    Accuracy = (number of correct answers) / (total questions)

Example:    AI got 17 out of 20 right
            Accuracy = 17/20 = 85%
```

**Use for:** Classification, factual Q&A, math problems — anything with a single correct answer.

### Relevance

**What it measures:** How well the AI's answer actually addresses what you asked.

```
Scoring guide (1-5 scale):

  5 = Perfectly relevant, directly answers the question
  4 = Mostly relevant, minor tangents
  3 = Somewhat relevant, misses some key points
  2 = Mostly off-topic
  1 = Completely irrelevant
```

**Use for:** Open-ended tasks like summarization, explanation, creative writing.

### Consistency

**What it measures:** Does the AI give similar answers when you ask the same thing multiple times?

```
Ask the same question 5 times:

  Run 1: "The capital of France is Paris."       ──┐
  Run 2: "Paris is the capital of France."       ──┤ All say Paris
  Run 3: "France's capital city is Paris."       ──┤ = Consistent!
  Run 4: "The capital is Paris."                 ──┤
  Run 5: "Paris."                                ──┘

  Consistency = 5/5 = 100%
```

**Use for:** Any task where you need reliable, predictable outputs.

### Format Compliance

**What it measures:** Did the AI follow your formatting instructions?

```
Instruction: "Respond as JSON with keys 'name' and 'age'"

  Response 1: {"name": "Alice", "age": 30}        ✓ Valid JSON, right keys
  Response 2: name: Alice, age: 30                 ✗ Not JSON
  Response 3: {"Name": "Alice", "Age": 30}         ✗ Wrong key casing

  Format compliance: 1/3 = 33%
```

**Use for:** When the output feeds into other software (APIs, databases, reports).

---

## How to Run an Evaluation

Here is a simple step-by-step process:

```
┌──────────────────────────────────────────────────────────────────┐
│                  Evaluation Process                               │
│                                                                  │
│   Step 1: Create test cases                                      │
│   ┌─────────────────────────────────────────────────┐            │
│   │ Write 10-20 inputs with expected outputs        │            │
│   │ Cover easy, medium, and tricky cases            │            │
│   └──────────────────────┬──────────────────────────┘            │
│                          │                                       │
│   Step 2: Run your prompt on all test cases                      │
│   ┌─────────────────────────────────────────────────┐            │
│   │ Feed each test input into your prompt           │            │
│   │ Save the AI's response                          │            │
│   └──────────────────────┬──────────────────────────┘            │
│                          │                                       │
│   Step 3: Score the results                                      │
│   ┌─────────────────────────────────────────────────┐            │
│   │ Compare AI outputs to expected outputs          │            │
│   │ Calculate accuracy, relevance, etc.             │            │
│   └──────────────────────┬──────────────────────────┘            │
│                          │                                       │
│   Step 4: Improve and repeat                                     │
│   ┌─────────────────────────────────────────────────┐            │
│   │ Look at the failures — what went wrong?         │            │
│   │ Adjust the prompt and run again                 │            │
│   └─────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────┘
```

### Example: Evaluating a Summary Prompt

```
Step 1: Test cases
  Article 1 (short, simple)    → Expected: 2-3 clear bullet points
  Article 2 (long, technical)  → Expected: 2-3 clear bullet points
  Article 3 (opinion piece)    → Expected: 2-3 clear bullet points

Step 2: Run prompt "Summarize in 3 bullet points"
  Article 1 → ✓ Got 3 bullets, clear and accurate
  Article 2 → ✗ Got 5 bullets (did not follow instructions)
  Article 3 → ✗ Got 3 bullets but included the author's opinion as fact

Step 3: Score
  Format compliance: 1/3 (only one followed the 3-bullet format)
  Accuracy: 2/3 (two were factually correct)

Step 4: Improve
  New prompt: "Summarize in EXACTLY 3 bullet points. Only include facts,
  not opinions. Each bullet should be one sentence."
  Re-run → Much better results!
```

---

## LLM-as-a-Judge (Using AI to Evaluate AI)

For open-ended tasks — summarization, writing, explanations — there is no single "correct" answer. Checking every response by hand is slow. A powerful alternative: **use a separate AI to judge the first AI's output.**

This is called **LLM-as-a-judge**. You send the original question and the AI's response to a second AI (the "judge") and ask it to score the response on specific criteria.

### How it works

```
┌──────────────────────────────────────────────────────────────────┐
│                   LLM-as-a-Judge                                 │
│                                                                  │
│   Step 1: Your prompt produces a response                        │
│   ┌──────────────────────┐    ┌──────────────────────┐           │
│   │ Your prompt          │ ──>│ AI response          │           │
│   │ "Summarize this..."  │    │ "The article says..."│           │
│   └──────────────────────┘    └──────────┬───────────┘           │
│                                          │                       │
│   Step 2: A judge AI scores the response │                       │
│   ┌──────────────────────────────────────┴────────────┐          │
│   │ Judge prompt:                                     │          │
│   │ "Rate this summary on accuracy (1-5),             │          │
│   │  completeness (1-5), and clarity (1-5).           │          │
│   │  Original text: [article]                         │          │
│   │  Summary: [AI response]                           │          │
│   │  Provide scores and a one-line explanation."      │          │
│   └──────────────────────────────────────┬────────────┘          │
│                                          │                       │
│   Step 3: You get a structured score     │                       │
│   ┌──────────────────────────────────────┴────────────┐          │
│   │ "Accuracy: 4/5 — captures main points correctly   │          │
│   │  Completeness: 3/5 — misses the financial data    │          │
│   │  Clarity: 5/5 — well-written and easy to read"    │          │
│   └───────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

### When to use LLM-as-a-judge

- **Open-ended tasks** where there is no single correct answer (summaries, essays, explanations)
- **High volume** — you have hundreds of outputs to evaluate and cannot read them all
- **Consistent scoring** — a human reviewer might score differently on Monday vs Friday; the judge AI is more consistent

### When NOT to use LLM-as-a-judge

- **Tasks with exact answers** — use direct comparison instead (accuracy = correct / total)
- **Safety-critical decisions** — do not trust an AI to judge whether medical or legal advice is safe
- **When the judge itself might be wrong** — if the task is hard enough that the judge AI makes mistakes, the scores are unreliable

### Example judge prompt

```
"You are an expert evaluator. Score the following AI response on these criteria:

 1. Accuracy (1-5): Are the facts correct?
 2. Completeness (1-5): Does it cover all key points?
 3. Clarity (1-5): Is it easy to understand?
 4. Format (1-5): Does it follow the requested format?

 Original question: {question}
 AI response: {response}

 For each criterion, provide a score and a one-sentence explanation.
 Then provide an overall score (average of the four)."
```

---

## Tips for Better Evaluation

| Tip | Why |
|-----|-----|
| Use at least 10 test cases | Small samples can be misleading — one good result does not mean the prompt is reliable |
| Include edge cases | Test with tricky inputs: very short text, very long text, ambiguous cases, unusual formats |
| Test one change at a time | If you change 3 things and results improve, you will not know which change helped |
| Save your test set | Keep using the same test cases so you can track improvement over time |
| Have someone else review | You might be biased — a fresh pair of eyes catches issues you would miss |
| Track your scores | Keep a simple spreadsheet: prompt version, date, accuracy score, notes |

---

## A Simple Evaluation Spreadsheet

You do not need fancy tools. A simple spreadsheet works great:

```
┌──────────┬───────────┬──────────┬─────────┬───────────┬─────────────────┐
│ Prompt   │ Date      │ Accuracy │ Format  │ Avg Score │ Notes           │
│ Version  │           │          │ Match   │ (1-5)     │                 │
├──────────┼───────────┼──────────┼─────────┼───────────┼─────────────────┤
│ v1       │ Jan 5     │ 60%      │ 70%     │ 3.2       │ Baseline        │
│ v2       │ Jan 6     │ 75%      │ 90%     │ 3.8       │ Added format    │
│ v3       │ Jan 7     │ 85%      │ 95%     │ 4.3       │ Added examples  │
│ v4       │ Jan 8     │ 90%      │ 95%     │ 4.5       │ Added CoT       │
└──────────┴───────────┴──────────┴─────────┴───────────┴─────────────────┘
```

---

## Automated Evaluation (Advanced)

Once you are comfortable with manual evaluation, you can automate it with code. Here is the basic idea:

```
┌──────────────────────────────────────────────────────────────────┐
│                  Automated Evaluation Pipeline                   │
│                                                                  │
│   Test cases          Run prompt on         Score each            │
│   (input +    ──>     each input    ──>     response     ──>  Report│
│    expected)          via API               automatically        │
│                                                                  │
│   You write           Code does this        Code compares        │
│   these once          automatically         output to expected   │
└──────────────────────────────────────────────────────────────────┘
```

For example, in Python:

```python
test_cases = [
    {"input": "I love it!",    "expected": "POSITIVE"},
    {"input": "Terrible!",     "expected": "NEGATIVE"},
    {"input": "It's okay.",    "expected": "NEUTRAL"},
]

correct = 0
for test in test_cases:
    response = call_llm(my_prompt, test["input"])
    if response.strip() == test["expected"]:
        correct += 1

accuracy = correct / len(test_cases)
print(f"Accuracy: {accuracy:.0%}")    # e.g., "Accuracy: 67%"
```

---

## Key Takeaways

1. **Always evaluate** — guessing whether your prompt works is not enough
2. **Start simple** — manual review with a checklist is perfectly fine to begin with
3. **Use test cases** — create inputs with known answers and track accuracy
4. **A/B test** — compare prompt variations on the same inputs to find the best one
5. **LLM-as-a-judge** — use a second AI to score open-ended outputs at scale
6. **Track your progress** — even a simple spreadsheet showing scores over time helps
7. **Iterate** — look at failures, improve the prompt, re-evaluate, repeat

---

## You Just Completed the Prompt Engineering Toolkit

You now know how to measure prompt quality — not just guess. This is the difference between amateur and professional prompt engineering. Professionals test, measure, and iterate. With evaluation, you can prove that one prompt is better than another, track improvements over time, and catch problems before they reach users.

You have now covered all four core topics in this module: few-shot learning, chain-of-thought, prompt templates, and evaluation. Together, they give you everything you need to write, structure, and validate effective prompts.

---

## Further Reading

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — Zheng et al., 2023
  - The paper that formalized LLM-as-a-judge and showed it correlates well with human preferences
- **Prompt engineering best practices** — Various sources
  - OpenAI, Anthropic, and Google all publish guides on evaluating prompt performance

---

[Back to Module Overview](./README.md) | [Previous: Prompt Templates](./prompt-templates.md)
