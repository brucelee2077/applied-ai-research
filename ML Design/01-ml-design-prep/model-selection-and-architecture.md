# Model Selection and Architecture

## Introduction

One of the most common mistakes in ML system design interviews is jumping straight to "I'd use a transformer" without explaining why simpler alternatives are insufficient. Interviewers don't want to hear that you know the latest architecture — they want to hear that you can reason about tradeoffs and pick the right tool for the problem.

The decision framework is straightforward: start with the simplest model that could work, identify what it can't handle, and upgrade only when you can articulate what the more complex model buys you. This page teaches that framework.

---

## The Model Selection Framework

### Start Simple

For every problem, ask: what's the simplest model that could work?

- If the features are mostly tabular (numbers, categories) → **gradient-boosted trees (XGBoost, LightGBM)**
- If the features include raw text or images → **pretrained model + fine-tuning**
- If you need to serve predictions in sub-millisecond → **logistic regression or a small model**

Starting simple is a green flag in interviews. It shows engineering maturity — you know that most production systems don't need transformers.

> "Let me start with a baseline. Logistic regression on our tabular features would give us a working system with sub-millisecond latency. Then I'll explain why we might need something more complex."

### When to Upgrade

Upgrade from simple to complex only when you can name the specific limitation:

| If the simple model can't... | Upgrade to... | Because... |
|---|---|---|
| Learn feature interactions | Deep & Cross Network, DeepFM | Need high-order interactions between features |
| Handle sequential data | Transformer, LSTM | User behavior history has temporal patterns |
| Process raw text/images | Pretrained model + fine-tuning | Raw features need representation learning |
| Retrieve from billions of items | Two-tower model + ANN | Need efficient candidate generation |
| Optimize multiple objectives | Multi-task learning | Predicting click + conversion + engagement simultaneously |

---

## Classical ML Models

### Logistic Regression

Don't dismiss it. Logistic regression is still the right choice when:
- You need interpretable predictions (feature coefficients tell you what drives the prediction)
- Latency is critical (<1ms per prediction at millions of QPS)
- You have many dense, well-engineered features
- Regulatory requirements demand explainability

At companies like Facebook (for early ads ranking) and Google, logistic regression with thousands of features was the production model for years.

### Gradient-Boosted Trees (XGBoost, LightGBM)

The workhorse of tabular ML. Trees naturally handle:
- Mixed feature types (categorical + numerical)
- Missing values (no imputation needed)
- Non-linear relationships and feature interactions
- Feature importance ranking (built-in)

**When trees beat deep learning:**
- Tabular data with <1M rows
- Strict latency requirements
- Features are well-engineered (not raw text/images)
- You need fast iteration (trees train in minutes, not hours)

**When deep learning beats trees:**
- Raw, unstructured inputs (text, images, audio, video)
- Data scale >10M rows with representation learning opportunity
- Need to learn complex sequential patterns
- Embedding-based retrieval is required

---

## Deep Learning Architectures by Problem Type

| Problem Type | Standard Architecture | Key Variant | When to Upgrade |
|---|---|---|---|
| Tabular classification/regression | GBDT (XGBoost) | Deep & Cross Network, TabNet | >1M samples + rich embedding features |
| Click/conversion prediction | Wide & Deep | DeepFM, DCN v2 | Need feature interactions + memorization |
| Retrieval/matching | Two-tower | Cross-encoder (re-ranking only) | Separate recall vs precision stages |
| Sequence modeling | Transformer | LSTM (latency-constrained) | Sequential user behavior |
| Image understanding | CNN (ResNet, EfficientNet) | Vision Transformer (ViT) | >1M images, or use pretrained |
| Text understanding | BERT / sentence-transformers | Distilled models for serving | Almost always pretrained + fine-tuned |
| Multi-modal | CLIP-style dual encoder | Cross-attention fusion | When modalities need deep interaction |
| Graph problems | GNN (GraphSAGE, GAT) | — | Social networks, knowledge graphs |

### Key Architecture Patterns

**Wide & Deep:** A wide (linear) component memorizes specific feature combinations, while a deep (neural network) component generalizes to unseen combinations. Used for click prediction at Google. Good when you need both memorization ("users in SF like artisanal coffee") and generalization ("users similar to SF users might also like...").

**Two-Tower:** Two separate neural networks encode queries and items into the same embedding space. Items are precomputed and indexed. At query time, run only the query tower and do ANN lookup. This is the standard pattern for candidate generation in search, recommendations, and ads.

**Multi-Task with Shared Backbone:** One model, multiple output heads. A shared representation learns from all tasks, while task-specific heads specialize. Used when you want to predict click, conversion, engagement, and satisfaction simultaneously. The shared backbone acts as a regularizer.

---

## Multi-Stage Architectures

Most production systems don't use a single model. They use a pipeline of models, each optimized for a different tradeoff.

### The Standard Pipeline

```
1B items → Candidate Generation (retrieve ~1000) → Ranking (score ~1000) → Re-ranking (policy, ~20 shown)
```

| Stage | Model Type | Optimizes For | Latency Budget |
|---|---|---|---|
| Candidate generation | Two-tower + ANN | Recall (don't miss good items) | <10ms |
| Lightweight ranking | Simple model (LR, small GBDT) | Approximate relevance | <5ms |
| Heavy ranking | Deep model (DCN, transformer) | Precision (rank the best items highest) | <50ms |
| Re-ranking | Rule-based + policy model | Business logic (diversity, freshness, fairness) | <10ms |

Each stage has a fundamentally different design goal. Candidate generation must be fast and have high recall — it's OK if some bad items slip through, because the ranker will filter them. The heavy ranker can be slow and expensive, because it only scores ~1000 items, not 1B.

### Why Multi-Stage Works

A single model scoring 1B items at 50ms each would take 578 days per request. By progressively filtering, each stage sees fewer items and can afford more computation per item.

---

## Training Strategies

### Pretraining → Fine-Tuning

Use a model pretrained on a large general dataset, then fine-tune on your specific task.

- **When to use:** You have <10K task-specific labeled examples, but the domain is related to the pretraining data (e.g., general English → product reviews).
- **What transfers:** Low-level features (text patterns, image edges) transfer well. High-level features (domain-specific semantics) may not.
- **How to fine-tune:** Start with a low learning rate. Freeze early layers, unfreeze gradually (progressive unfreezing).

### Multi-Task Learning

Train one model on multiple related tasks simultaneously.

- **Shared representation:** Tasks learn from each other's data. A model predicting both click and purchase learns richer user representations than either task alone.
- **Gradient conflict:** Sometimes tasks compete. The gradient for click prediction might push in the opposite direction from the gradient for purchase prediction. Solutions: PCGrad (project conflicting gradients), GradNorm (dynamically balance gradient magnitudes), uncertainty weighting.
- **When to use:** Tasks share common features and structure. Click prediction + conversion prediction + engagement prediction.

### Continual Learning

Models need to update as data distributions change. But naive retraining on new data causes catastrophic forgetting — the model forgets old patterns.

- **Replay buffers:** Store a subset of old data and mix it with new data during training.
- **Elastic Weight Consolidation:** Penalize changes to parameters that were important for old tasks.
- **When to use:** Data distributions shift frequently (trending topics, seasonal patterns, evolving user behavior).

---

## Model Complexity Tradeoffs

### The Diminishing Returns Curve

More parameters don't always help. Beyond a certain point, additional model capacity provides diminishing returns on your specific task while increasing serving cost.

| Consideration | Simpler Model | More Complex Model |
|---|---|---|
| Training time | Hours | Days-weeks |
| Inference latency | Sub-millisecond | Milliseconds-seconds |
| Training data needed | Thousands | Millions+ |
| Debuggability | Feature importance, coefficients | Hard to interpret |
| Iteration speed | Fast experiments | Slow experiments |
| Engineering effort | Low | High (GPUs, distributed training, serving infra) |

### Ensembles

Combining multiple models (bagging, boosting, stacking) consistently improves performance by 1-5%. But they multiply serving cost and complexity.

- **When to ensemble:** Competition settings, when you have a specific accuracy target to hit.
- **When NOT to ensemble:** Production systems with strict latency budgets. The complexity rarely justifies a 2% accuracy improvement.
- **Alternative:** Distillation — train a single student model to mimic the ensemble's predictions. You get most of the benefit with single-model serving cost.

---

## Interview Strategy

### The "Why Not X?" Test

For every model you propose, be prepared to explain why you didn't choose a simpler alternative. If you can't answer "why not logistic regression?", your model choice isn't well-justified.

> "I'm proposing a Deep & Cross Network because we need both memorization of specific feature crosses and generalization to unseen combinations. A plain DNN can generalize but doesn't memorize well. A logistic regression with manual crosses can memorize but doesn't generalize. DCN gives us both."

### Time Management

Model selection should take 2-3 minutes in the interview:
1. State your baseline (30 seconds)
2. Identify the limitation (30 seconds)
3. Propose the upgrade with a brief "why" (1 minute)
4. Sketch the architecture (1 minute)

Don't spend 10 minutes deriving the architecture from first principles. The interviewer wants to know that you can pick the right tool and move on.

**Green Flags**
- You started with a simple baseline and upgraded with clear justification
- You compared at least two alternatives and explained the tradeoff
- You mentioned multi-stage architecture for large-scale retrieval systems
- You connected model choice to serving constraints (latency, cost)

**Red Flags**
- You jumped to "use a transformer" without considering alternatives
- You couldn't explain why a simpler model wouldn't work
- You proposed an architecture you couldn't describe concretely
- You ignored serving cost and latency in your model choice

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should be able to pick a reasonable model for the problem and justify it at a high level. For a click prediction system, they should know that gradient-boosted trees or a neural network on tabular features is a solid approach. They should be familiar with the concept of a multi-stage pipeline (candidate generation → ranking) even if they don't detail every stage. Mid-level candidates differentiate by delivering a credible architecture — it doesn't need to be optimal, but it should be defensible.

### Senior Engineer

Senior candidates demonstrate fluency with the tradeoff space. They can articulate when trees beat deep learning and vice versa. They proactively propose a multi-stage architecture for retrieval problems and explain the recall-precision tradeoff between stages. For an ad prediction system, a senior candidate would propose Wide & Deep or DCN for the ranking stage, explain why the two-tower retrieval stage uses a simpler model (latency), and discuss how the model handles multi-task prediction (click + conversion). They bring up training strategies like pretraining and multi-task learning unprompted.

### Staff Engineer

Staff candidates quickly establish the standard architecture for the problem type and spend most of their time on the nuances that matter. They recognize that architecture choice is often less important than data quality, feature engineering, and serving infrastructure. A Staff candidate might say: "The specific neural architecture matters less than ensuring our training pipeline captures the right signals. I'd rather spend time discussing how we handle position bias in our training data than debating DCN vs DeepFM." They also think about the lifecycle: how the model will be updated, how new architectures will be tested (A/B tested, not just offline), and how to manage the transition.
