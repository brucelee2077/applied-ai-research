# Prompt Templates

A developer at a startup writes the same kind of prompt 50 times a day — "summarize this support ticket," "classify this email," "extract data from this form." Every time, they rephrase it slightly, get slightly different results, and waste time thinking about wording. Then they build one template with blanks to fill in. Same quality every time. No more thinking about phrasing. That is the power of prompt templates.

---

**Before you start, you need to know:**
- What a prompt is and how to write one — covered in [README.md](./README.md)
- How few-shot examples work — covered in [few-shot-learning.md](./few-shot-learning.md)
- How chain-of-thought works — covered in [chain-of-thought.md](./chain-of-thought.md)

---

## What are Prompt Templates?

**The analogy: Mad Libs.** Have you ever played Mad Libs — that game where you have a story with blank spaces and you fill in words like "a noun" or "a funny verb" to make a silly story? Prompt templates work the same way. They are pre-written prompts with **blanks (called variables)** that you fill in each time you use them. Instead of writing a brand-new prompt from scratch every time, you create a reusable "recipe" and just swap in the specific details.

**What the analogy gets right:** just like Mad Libs, you write the structure once and fill in different values each time. The structure stays the same, only the details change. And just like how Mad Libs gives you hints about what kind of word to fill in ("a noun", "an adjective"), good template variables have descriptive names that tell you what to put there.

**The concept in plain words:** a prompt template is a prompt with placeholders. You write it once, test it until it works well, and then reuse it by filling in the blanks with different inputs each time.

**Where the analogy breaks down:** Mad Libs is designed to produce funny, random results. A prompt template is designed to produce **consistent, high-quality** results. The whole point is that every time you use the template, the output follows the same structure and meets the same quality bar.

```
┌──────────────────────────────────────────────────────────────────┐
│                     Mad Libs vs Prompt Template                   │
│                                                                  │
│   MAD LIBS:                                                      │
│   "The ___[adjective]___ ___[animal]___ jumped over the          │
│    ___[noun]___."                                                │
│   Fill in: "happy", "frog", "moon"                               │
│   Result:  "The happy frog jumped over the moon."                │
│                                                                  │
│   PROMPT TEMPLATE:                                               │
│   "Summarize the following {topic} article in {num_points}       │
│    bullet points for a {audience} audience:                      │
│    {article_text}"                                               │
│   Fill in: topic="science", num_points="3",                      │
│            audience="middle school", article_text="..."          │
│   Result:  A complete, ready-to-use prompt!                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Why Use Templates?

Without templates, you would rewrite the same kind of prompt over and over:

```
WITHOUT TEMPLATES (repetitive):

  Monday:    "Summarize this tech article in 3 bullet points for beginners: [article1]"
  Tuesday:   "Summarize this health article in 3 bullet points for beginners: [article2]"
  Wednesday: "Summarize this sports article in 3 bullet points for beginners: [article3]"

  ^ You are writing almost the same thing every time!

WITH TEMPLATES (reusable):

  Template: "Summarize this {topic} article in {n} bullet points for {audience}: {text}"

  Monday:    fill in topic="tech",    n=3, audience="beginners", text=article1
  Tuesday:   fill in topic="health",  n=3, audience="beginners", text=article2
  Wednesday: fill in topic="sports",  n=3, audience="beginners", text=article3

  ^ Write the template once, reuse it forever!
```

**Benefits of templates:**
- **Consistency** — every prompt follows the same proven structure
- **Speed** — no need to think about how to phrase things each time
- **Quality** — you can test and improve the template, and all future uses benefit
- **Sharing** — your whole team can use the same templates

---

**Quick check — can you answer these?**
- In your own words: what is a prompt template?
- Why is consistency an advantage of templates?
- What is the difference between a template and a regular prompt?

If you cannot answer one, re-read that section. That is completely normal.

---

## Anatomy of a Good Template

A well-designed template has these parts:

```
┌──────────────────────────────────────────────────────────────────┐
│                  Anatomy of a Prompt Template                    │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │ ROLE (who the AI should be)                             │    │
│   │ "You are a {role} who specializes in {specialty}."      │    │
│   └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │ CONTEXT (background information)                        │    │
│   │ "Here is the {document_type}: {content}"                │    │
│   └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │ TASK (what to do)                                       │    │
│   │ "Please {action} the above {document_type}."            │    │
│   └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │ FORMAT (how to present the answer)                      │    │
│   │ "Output as {format} with {constraints}."                │    │
│   └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

Not every template needs all four parts, but the more you include, the better and more consistent your results will be.

---

## Variable Substitution

Variables are the "blanks" in your template. They are usually written with curly braces `{like_this}`. When you use the template, you replace each variable with a real value.

### Simple Variables

```
Template:
  "Translate the following {source_language} text to {target_language}:
   {text}"

Usage:
  source_language = "English"
  target_language = "Spanish"
  text = "Good morning, how are you?"

Final prompt:
  "Translate the following English text to Spanish:
   Good morning, how are you?"
```

### Variables with Defaults

Sometimes a variable has a sensible default value that you can override when needed.

```
Template:
  "Summarize the following text in {num_points|3} bullet points."

  {num_points|3} means: use num_points if provided, otherwise default to 3
```

### Conditional Sections

Some templates include optional parts that only appear when needed.

```
Template:
  "Analyze this code:
   {code}

   Language: {language}
   {if focus_area}Focus on: {focus_area}{endif}
   {if include_examples}Include code examples in your explanation.{endif}"

Usage 1 (minimal):  language="Python", code="..."
Usage 2 (detailed): language="Python", code="...", focus_area="error handling",
                     include_examples=true
```

---

## Common Template Patterns

Here are four template patterns you can use right away:

### Pattern 1: Role + Task + Format

```
Template:
  "You are a {role}.

   {task_description}

   Please respond in the following format:
   {output_format}"

Example:
  "You are a senior code reviewer.

   Review this Python function for bugs, performance issues, and readability:

   def calculate_average(numbers):
       total = 0
       for n in numbers:
           total += n
       return total / len(numbers)

   Please respond in the following format:
   - Bugs: [list any bugs found]
   - Performance: [any performance concerns]
   - Readability: [suggestions for clearer code]
   - Improved code: [rewritten version]"
```

### Pattern 2: Context + Question

```
Template:
  "Given the following {context_type}:

   {context}

   Answer this question: {question}

   If the answer is not in the {context_type}, say 'I don't have enough
   information to answer that.'"

Example:
  "Given the following product description:

   The XR-500 is a wireless headphone with 30-hour battery life,
   active noise cancellation, and Bluetooth 5.3. It weighs 250g
   and comes in black and white. Retail price: $199.

   Answer this question: Does it support Bluetooth?

   If the answer is not in the product description, say 'I don't have
   enough information to answer that.'"
```

### Pattern 3: Input + Examples + Task (Few-Shot Template)

```
Template:
  "{task_instruction}

   {examples}

   Input: {input}
   Output:"

Example:
  "Classify the customer support ticket into one of these categories:
   BILLING, TECHNICAL, GENERAL.

   Input: 'My credit card was charged twice.'
   Output: BILLING

   Input: 'The app crashes when I open settings.'
   Output: TECHNICAL

   Input: 'I can't reset my password and keep getting an error message.'
   Output:"
```

### Pattern 4: Step-by-Step Instruction

```
Template:
  "You are a {role}.

   Follow these steps to {goal}:
   1. {step_1}
   2. {step_2}
   3. {step_3}

   Input: {input}

   Work through each step and show your final result."
```

---

## Building Your Own Templates: A Step-by-Step Guide

### Step 1: Start with a Working Prompt

Write a prompt that works well for one specific case.

```
"Explain what a 'for loop' is in Python. Use a real-world analogy that a
12-year-old would understand. Include a simple code example."
```

### Step 2: Identify What Changes

Look at what would be different each time you use this prompt:
- The concept to explain (`for loop` → could be anything)
- The programming language (`Python` → could be JavaScript, etc.)
- The audience (`12-year-old` → could be college student, etc.)

### Step 3: Replace with Variables

```
"Explain what '{concept}' is in {language}. Use a real-world analogy that a
{audience} would understand. Include a simple code example."
```

### Step 4: Add Structure

Wrap it in a clear template with role, context, and format:

```
"You are a {language} programming tutor.

 Explain the concept of '{concept}' to someone who is a {audience}.

 Requirements:
 - Use a real-world analogy to make it intuitive
 - Include a simple, runnable code example
 - Avoid jargon unless you define it first
 - Keep the explanation under {max_words|200} words"
```

### Step 5: Test and Iterate

Try your template with different inputs and refine it:

```
Test 1: concept="for loop",     language="Python",     audience="12-year-old"
Test 2: concept="recursion",    language="JavaScript", audience="college student"
Test 3: concept="hash table",   language="Java",       audience="career changer"
```

If any test gives poor results, adjust the template wording.

---

## Template Libraries

A **template library** is a collection of your best templates organized by category. Think of it as your personal cookbook of prompt recipes.

```
┌──────────────────────────────────────────────────────────────────┐
│                    Example Template Library                       │
│                                                                  │
│   Writing/                                                       │
│   ├── summarize.txt          "Summarize {text} in {n} points..." │
│   ├── rewrite-tone.txt       "Rewrite {text} in a {tone} tone..." │
│   └── proofread.txt          "Proofread {text} for {issues}..."  │
│                                                                  │
│   Coding/                                                        │
│   ├── code-review.txt        "Review this {language} code..."    │
│   ├── explain-code.txt       "Explain what this code does..."    │
│   └── write-tests.txt        "Write unit tests for {function}..." │
│                                                                  │
│   Analysis/                                                      │
│   ├── compare.txt            "Compare {option_a} vs {option_b}..." │
│   ├── pros-cons.txt          "List pros and cons of {topic}..."  │
│   └── extract-data.txt       "Extract {fields} from {text}..."   │
└──────────────────────────────────────────────────────────────────┘
```

You can store these as simple text files, in a spreadsheet, or in code (using string formatting in Python, JavaScript, etc.).

### Example: Template in Python

```python
# Define the template
template = """You are a {role}.

Analyze the following {doc_type}:
{content}

Provide:
1. A one-sentence summary
2. Key points (max {max_points})
3. Any concerns or issues"""

# Use the template
prompt = template.format(
    role="senior financial analyst",
    doc_type="quarterly earnings report",
    content="Revenue was $5.2B, up 12% year-over-year...",
    max_points=5
)
```

---

## Common Mistakes to Avoid

| Mistake | Problem | Fix |
|---------|---------|-----|
| Too many variables | Template becomes confusing and hard to use | Limit to 3-5 variables per template |
| Vague variable names | `{x}` and `{data}` do not tell you what to fill in | Use descriptive names: `{customer_email}`, `{product_name}` |
| No output format | AI responds in unpredictable formats | Always specify the expected output format |
| Overly rigid template | Does not adapt to different use cases | Use optional sections and default values |
| Never testing | You do not know if the template actually works well | Test with 3+ different inputs before sharing |

---

## Key Takeaways

1. **Prompt templates** = reusable prompts with fill-in-the-blank variables
2. **Good templates** have a role, context, task, and output format
3. **Variables** use `{curly_braces}` and get replaced with real values
4. **Build templates** by starting with a working prompt, identifying what changes, and replacing those parts with variables
5. **Organize templates** into a library so your team can reuse them

---

## You Just Built Your Prompt Toolkit

You now know how to take any prompt that works well and turn it into a reusable template. This is how professional teams work with LLMs — they do not write prompts from scratch every time. They build, test, and share templates. Your prompt quality just became repeatable.

Ready to go further? Next, learn how to measure whether your prompts are working → [Prompt Evaluation](./prompt-evaluation.md)

---

## Further Reading

- **Prompt engineering best practices** — Various sources
  - OpenAI, Anthropic, and Google all publish prompt engineering guides with template recommendations
- **LangChain PromptTemplate documentation**
  - A popular Python library that provides a structured way to build and manage prompt templates programmatically

---

[Back to Module Overview](./README.md) | [Previous: Chain-of-Thought](./chain-of-thought.md) | [Next: Prompt Evaluation](./prompt-evaluation.md)
