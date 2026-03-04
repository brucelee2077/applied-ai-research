# GenAI Evaluation Methods

## Introduction

Evaluating generative AI is fundamentally harder than evaluating discriminative models. For classification, you compare predicted labels to ground truth — accuracy is well-defined. For generation, there is no single correct output. "Write me an email" has thousands of valid responses. "Generate an image of a sunset" has infinite valid outputs. How do you measure whether the output is "good"?

This challenge makes evaluation one of the most important topics in genAI system design interviews. Candidates who can articulate a rigorous, multi-layered evaluation strategy — from automated metrics to human evaluation — demonstrate that they've actually shipped generative systems, not just trained models.

---

## Why GenAI Evaluation Is Hard

| Challenge | What It Means | Example |
|-----------|-------------|---------|
| Open-ended outputs | Many valid answers to the same prompt | "Summarize this article" → thousands of valid summaries |
| Subjective quality | "Good" depends on the user, task, and context | A casual tone is perfect for a chatbot, terrible for legal drafting |
| Multiple dimensions | Accuracy, coherence, creativity, safety, helpfulness can conflict | A more creative response may be less accurate |
| No gold standard | For many tasks, there's no reference to compare against | "Write a marketing email" has no ground truth |
| Delayed feedback | Quality may only be apparent from long-term outcomes | Does a recommendation lead to a return visit? |

**The evaluation paradox:** The tasks where generative AI is most useful (creative, open-ended, complex) are exactly the tasks where evaluation is hardest.

---

## Automated Text Metrics

Fast and cheap, but limited. Use as a first filter, never as the sole evaluation.

### Reference-Based Metrics

These compare generated text to a reference (gold standard) text.

| Metric | What It Measures | Good For | Bad For |
|--------|-----------------|----------|---------|
| BLEU | N-gram precision (how many n-grams in the output match the reference) | Machine translation | Open-ended generation (penalizes valid paraphrases) |
| ROUGE | N-gram recall (how many reference n-grams appear in the output) | Summarization | Same as BLEU — rigid matching |
| METEOR | Precision + recall with synonym matching | Translation (better than BLEU) | Still surface-level |
| BERTScore | Semantic similarity using contextual embeddings | Semantic matching beyond exact words | Doesn't measure factual accuracy |
| chrF | Character-level F-score | Translation, especially for morphologically rich languages | Not useful for long-form generation |

**The fundamental limitation:** Reference-based metrics assume one (or a few) correct outputs. For open-ended generation, a perfectly valid response that differs from the reference scores poorly.

### Reference-Free Metrics

| Metric | What It Measures | How It Works |
|--------|-----------------|-------------|
| Perplexity | Model's uncertainty about the text | Lower = more fluent, but doesn't measure correctness or helpfulness |
| Diversity | Vocabulary and structural variety | Distinct-n (unique n-grams / total n-grams), Self-BLEU |
| Coherence | Logical flow and consistency | Embedding similarity between adjacent sentences |
| Length | Output verbosity | Token count — useful for detecting verbose reward hacking |

---

## Automated Image/Video Metrics

| Metric | What It Measures | How It Works | Limitations |
|--------|-----------------|-------------|-------------|
| FID (Fréchet Inception Distance) | Distribution similarity between generated and real images | Compare Inception network activations for generated vs real images. Lower = better. | Needs thousands of samples. Doesn't evaluate individual images. |
| CLIP Score | Text-image alignment | Cosine similarity between CLIP embeddings of text and image. Higher = better. | Can be gamed. Doesn't capture fine-grained spatial relationships. |
| IS (Inception Score) | Quality and diversity | High IS = images are both sharp (classifiable) and diverse (cover many classes). | Doesn't measure prompt faithfulness. Biased toward ImageNet classes. |
| Aesthetic Score | Visual appeal | Trained classifier predicting human aesthetic ratings. | Subjective, dataset-dependent. |

**FID is the standard for comparing models.** But it's a population-level metric — it tells you about the distribution of generated images, not about any individual image. For evaluating a specific generation, use CLIP Score + human evaluation.

---

## LLM-as-Judge

Use a strong LLM to evaluate outputs of a weaker model (or the same model). Increasingly popular as a middle ground between automated metrics and human evaluation.

### How It Works

Provide the judge model with:
1. The original prompt
2. The generated response (or two responses for comparison)
3. A detailed rubric specifying evaluation criteria
4. Examples of how to apply the rubric (few-shot)

### Evaluation Modes

| Mode | How It Works | Pros | Cons |
|------|-------------|------|------|
| Absolute scoring | Rate response on a scale (1-5) | Simple, comparable across models | Anchoring effects, scores drift |
| Pairwise comparison | "Which response is better, A or B?" | More reliable than absolute scoring | Doesn't tell you how much better |
| Rubric-based | Score each dimension separately with explicit criteria | Detailed, actionable feedback | More expensive (multiple evaluations per response) |

### Known Pitfalls

| Bias | What Happens | Mitigation |
|------|-------------|-----------|
| Position bias | LLM prefers the first response in pairwise comparisons | Swap order and average; flag disagreements |
| Verbosity bias | LLM prefers longer responses regardless of quality | Include length-neutral criteria in rubric |
| Self-preference | LLM prefers outputs from the same model family | Use a different model family as judge |
| Style bias | LLM prefers certain writing styles (formal, structured) | Calibrate rubric examples across styles |

### Calibration

LLM-as-judge must be validated against human judgments:
1. Create a calibration set: 100-500 examples with human ratings
2. Run the LLM judge on the same examples
3. Measure agreement (Cohen's κ, Spearman correlation)
4. Target: κ > 0.6 (substantial agreement) or correlation > 0.7

If the LLM judge disagrees with humans systematically, adjust the rubric or switch models.

---

## Human Evaluation

The gold standard. Expensive and slow, but the only way to truly measure subjective quality.

### Methods

| Method | How It Works | Best For |
|--------|-------------|----------|
| Likert scales (1-5) | Rate each response on quality dimensions | Absolute quality measurement |
| Pairwise preferences | "Which response is better?" | Comparative quality, model ranking |
| Chatbot Arena (Elo) | Users interact with two anonymous models, pick the better one | Large-scale model ranking |
| Task completion | Can the user complete their task using the generated output? | Practical utility measurement |
| Post-edit distance | How much did a human need to edit the output? | Translation, writing assistance |

### Multi-Dimensional Evaluation

Never collapse quality into a single number. Evaluate dimensions separately:

| Dimension | What to Measure | Example Scale |
|-----------|----------------|---------------|
| Factual accuracy | Are the claims in the response true? | 1 (multiple errors) to 5 (all correct) |
| Helpfulness | Does the response address the user's need? | 1 (doesn't address) to 5 (fully addresses) |
| Coherence | Is the response logically organized? | 1 (incoherent) to 5 (clear and logical) |
| Safety | Does the response avoid harmful content? | Binary (safe/unsafe) + severity |
| Tone/style | Is the tone appropriate for the context? | 1 (inappropriate) to 5 (perfect) |

### Inter-Annotator Agreement

Human evaluations are only useful if annotators agree with each other.

| Agreement Metric | Good Value | What It Means |
|-----------------|-----------|---------------|
| Cohen's κ | > 0.6 | Substantial agreement between two annotators |
| Krippendorff's α | > 0.67 | Reliable agreement across multiple annotators |
| Pairwise agreement | > 75% | Most annotator pairs agree |

**If agreement is low:** The rubric is too vague, the task is inherently subjective, or annotators need calibration. Fix the rubric before collecting more data.

---

## Task-Specific Evaluation

### Factual QA

| Metric | What It Measures | Notes |
|--------|-----------------|-------|
| Exact Match (EM) | Does the output exactly match the reference? | Too strict for natural language answers |
| Token F1 | Overlap between output tokens and reference tokens | Better than EM for natural answers |
| Human verification | Is the answer factually correct? | Gold standard but expensive |
| Hallucination rate | % of responses containing claims not in the source | Critical for RAG systems |

### Summarization

- **ROUGE:** Measures n-gram overlap with reference summary. A necessary but insufficient check.
- **Faithfulness:** Does the summary contain only information from the source? Check with NLI models or LLM-as-judge.
- **Coverage:** Does the summary capture all key points from the source? Compare against human-extracted key points.

### Code Generation

| Metric | What It Measures | Notes |
|--------|-----------------|-------|
| pass@k | Generate k solutions, report if any pass unit tests | The standard metric. k=1 is strict, k=10 is lenient. |
| Functional correctness | Does the code produce correct output for all test cases? | Requires good test coverage |
| Code quality | Style, efficiency, readability | Human evaluation or linter-based |

### Dialogue / Chatbot

- **Task completion rate:** Did the user accomplish their goal?
- **User satisfaction:** Post-conversation survey or implicit signals (thumbs up/down)
- **Turn efficiency:** How many turns to resolve the query? Fewer = better.
- **Escalation rate:** How often does the user ask for a human? Lower = better.
- **Return rate:** Does the user come back? Higher = better.

---

## Building an Evaluation Pipeline

### The Three-Tier Approach

| Tier | Method | Speed | Cost | When to Use |
|------|--------|-------|------|-------------|
| 1 | Automated metrics | Seconds | $0 | Every commit, every prompt change |
| 2 | LLM-as-judge | Minutes | $$ | Weekly, before deployment |
| 3 | Human evaluation | Days | $$$$ | Monthly, for major model changes |

**Tier 1 (automated):** Run on every change. Catches regressions in format compliance, length, perplexity, and basic quality metrics. Fast enough for CI/CD.

**Tier 2 (LLM-as-judge):** Run before deployment. Evaluates nuanced quality dimensions (helpfulness, faithfulness, tone) against a rubric. Catches issues automated metrics miss.

**Tier 3 (human):** Run for major changes. Provides the ground truth that Tiers 1 and 2 are calibrated against. Catches subtle quality issues that even LLM judges miss.

### Regression Testing

Maintain a curated benchmark set of 200-500 examples covering:
- Common use cases (80% of traffic patterns)
- Edge cases (ambiguous inputs, adversarial inputs, multi-step reasoning)
- Known failure modes (past incidents, reported issues)
- Safety scenarios (harmful requests, prompt injections)

Run the full benchmark before every model update or prompt change. Any regression is a deployment blocker.

### Slice-Based Evaluation

Aggregate metrics hide problems. Always slice by:
- **Input category:** Question type, topic, complexity level
- **User segment:** New vs returning, free vs paid, language
- **Output property:** Length, format, confidence level
- **Safety category:** Explicit harm categories evaluated separately

A model that averages 4.2/5 overall might score 2.1/5 on multi-step reasoning questions. The aggregate hides the problem.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand that genAI evaluation requires more than accuracy — they should mention at least automated metrics and human evaluation as complementary approaches. For a chatbot system, they should propose tracking user satisfaction metrics (thumbs up/down, task completion rate) and running periodic human evaluations. They differentiate by recognizing that automated metrics like BLEU are insufficient for open-ended generation and that multiple quality dimensions (accuracy, helpfulness, safety) should be evaluated separately.

### Senior Engineer

Senior candidates can design a multi-tier evaluation pipeline. They discuss automated metrics as a fast first filter, LLM-as-judge for nuanced evaluation, and human evaluation as the gold standard. For a RAG-based question-answering system, a senior candidate would propose evaluating both retrieval quality (recall@k) and generation quality (faithfulness, relevance) separately, and bring up the hallucination rate as a critical metric. They mention slice-based evaluation to catch segment-level quality problems and regression testing before deployment.

### Staff Engineer

Staff candidates think about evaluation as a system that must evolve. They recognize that the hardest evaluation problem isn't measurement — it's knowing what to measure. A Staff candidate might point out that optimizing for automated metrics can lead to goodharting (the metric improves but real quality doesn't), and propose using human evaluation to continuously recalibrate automated metrics. They also think about evaluation at the organizational level: who defines quality standards, how do you prevent evaluation sets from becoming stale (the model memorizes them), and how do you build evaluation infrastructure that scales as the product adds new use cases?
