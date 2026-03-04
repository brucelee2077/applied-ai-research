# Serving and Inference

## Introduction

A model that can't serve predictions fast enough is useless. This is the gap that trips up many candidates: they design beautiful training pipelines and then hand-wave when asked how the model actually serves 100K requests per second with p99 latency under 50ms.

Production serving is where Staff candidates show their depth. It's not just about picking a framework — it's about understanding the full inference stack, knowing which optimizations matter for your specific problem, and being able to reason about tradeoffs between latency, throughput, cost, and quality.

---

## Latency Budgets

Different use cases have fundamentally different requirements:

| Use Case | Latency Target | Why |
|----------|---------------|-----|
| Ad auction | <10ms | Auction happens per page load, ad must be selected before page renders |
| Search ranking | <50ms | User is actively waiting, perceives delay after ~200ms |
| Feed recommendation | <200ms | User scrolling, moderate tolerance |
| Content moderation | <1s | Pre-publication check, user waits briefly |
| Batch scoring | Hours | Offline, no user waiting |

### p50 vs p95 vs p99

Average latency is almost meaningless. What matters is tail latency — the slowest requests.

- **p50:** Half of requests are faster than this. Useful for understanding typical experience.
- **p95:** 1 in 20 requests is slower. This is where most SLAs are set.
- **p99:** 1 in 100 requests is slower. At 1M QPS, 10K requests per second exceed this threshold.

A system with p50=5ms but p99=500ms has a bad tail. The 1% of users experiencing 500ms delays will disproportionately complain, churn, or lose revenue.

### Latency Decomposition

Break down the total latency budget:

```
Total latency = Network (2-10ms) + Feature lookup (5-20ms) + Model inference (5-50ms) + Post-processing (1-5ms)
```

If your total budget is 50ms and feature lookup takes 30ms, optimizing model inference from 20ms to 10ms barely matters. Find the bottleneck first.

---

## Model Optimization for Inference

### Quantization

Reduce numerical precision to speed up computation and reduce memory:

| Precision | Bits | Memory | Speed | Quality Loss |
|-----------|------|--------|-------|-------------|
| FP32 | 32 | Baseline | Baseline | None |
| FP16 | 16 | 2x reduction | 1.5-2x faster | Negligible |
| INT8 | 8 | 4x reduction | 2-4x faster | Small (<1% accuracy) |
| INT4 | 4 | 8x reduction | 3-5x faster | Moderate (1-3% accuracy) |

- **Post-training quantization:** Quantize after training. Quick, no retraining needed. Some quality loss.
- **Quantization-aware training:** Simulate quantization during training. Better quality, requires retraining.
- **When to quantize:** Almost always for serving. FP16 is nearly free. INT8 is the standard for production.

### Distillation

Train a smaller "student" model to mimic a larger "teacher" model.

- **How:** Student learns from teacher's output probabilities (soft labels), not just hard labels. The soft labels carry richer information about class relationships.
- **Result:** A student 10x smaller can achieve 90-95% of the teacher's quality.
- **When to use:** When you need low-latency serving but trained with a large, expensive model.

### Graph Optimization

Model compilers (TensorRT, ONNX Runtime, TorchScript) optimize the computation graph:
- **Operator fusion:** Combine multiple operations (conv + batch norm + relu) into one kernel.
- **Dead code elimination:** Remove unused computation paths.
- **Constant folding:** Pre-compute operations on constant inputs.
- These optimizations are usually free — just export your model and let the compiler do its job.

---

## Serving Infrastructure

### Online Serving

Request-response pattern. Model receives a request, computes prediction, returns result.

- **Load balancing:** Distribute requests across model replicas.
- **Auto-scaling:** Add/remove replicas based on traffic. Scale up for peak hours, scale down overnight.
- **Health checks:** Monitor each replica for latency and error rates. Route traffic away from unhealthy replicas.
- **When to use:** Personalized predictions that depend on real-time context (current session, user state).

### Batch Serving

Precompute predictions for all entities, store in a key-value store, serve lookups.

- **Advantage:** No model inference at request time — just a key-value lookup (sub-millisecond).
- **Disadvantage:** Predictions are stale. Can't incorporate real-time context.
- **When to use:** Predictions that don't depend on request-time context and change slowly (user segments, item quality scores, daily recommendations).

### Streaming Serving

Process events as they arrive, update predictions in near-real-time.

- **How:** Consume events from a message queue (Kafka), run lightweight inference, update a serving store.
- **When to use:** When predictions need to be fresher than batch (minutes, not hours) but don't need sub-second latency. Fraud detection, real-time personalization signals.

---

## Feature Serving

Features are often the latency bottleneck, not model inference.

### Feature Store Architecture

```
Offline Store (Data Warehouse) ← Batch Pipeline ← Raw Data
                                                       ↓
Online Store (Redis/DynamoDB) ← Streaming Pipeline ← Events
                                                       ↓
Model Server → Feature Lookup → Online Store → Model Inference → Response
```

- **Offline store:** Historical features for training. High latency, high throughput. BigQuery, Hive.
- **Online store:** Latest features for serving. Low latency (<5ms), low throughput per key. Redis, DynamoDB.
- **Consistency:** Features used at training time must match features available at serving time. Mismatches cause training-serving skew.

### Handling Missing Features

At serving time, a feature may be unavailable (pipeline delay, new user, data source outage). The model needs to handle this gracefully.

- **Default values:** Use training-time mean/median as fallback.
- **Graceful degradation:** If critical features are missing, fall back to a simpler model that doesn't need them.
- **Feature importance monitoring:** Know which features matter most. Missing a low-importance feature is fine. Missing the top feature is an incident.

---

## Caching Strategies

Caching can dramatically reduce both latency and cost.

| Cache Level | What's Cached | Hit Rate | Example |
|-------------|--------------|----------|---------|
| Query cache | Full prediction result | 5-30% | Same search query → same results |
| Embedding cache | Expensive embedding computations | 50-80% | User/item embeddings reused across requests |
| Feature cache | Feature values | 60-90% | User profile features stable across requests |
| Model output cache | Intermediate model outputs | Varies | Candidate generation results reused for re-ranking |

**Cache invalidation:** The hardest problem in caching. Use TTL (time-to-live) for features that change slowly. Use event-driven invalidation for features that change unpredictably.

---

## Multi-Stage Inference Pipelines

For large-scale retrieval, a single model can't score every item.

```
1B items → Candidate Gen (<10ms) → 1000 items → Ranking (<50ms) → 100 items → Re-ranking (<10ms) → 20 items
```

Each stage has a different cost-quality tradeoff:

- **Candidate generation:** Cheap per item (ANN lookup), high recall, low precision. OK to include some bad items.
- **Ranking:** Moderate cost per item (neural network), high precision. Filters out bad items, orders good ones.
- **Re-ranking:** Cheap, rule-based. Diversity, freshness, business logic. Doesn't change relevance scores — just adjusts the final slate.

---

## Scaling

### GPU Serving

LLMs and large neural networks require GPU inference.

- **Batching:** Group multiple requests into one GPU batch. Static batching waits for N requests. Dynamic batching groups requests within a time window. Continuous batching (vLLM) processes different requests at different stages simultaneously.
- **GPU memory management:** Model weights + activations + KV-cache must fit in GPU memory. Quantization reduces model size. PagedAttention reduces KV-cache fragmentation.
- **Model parallelism:** For models too large for one GPU, split across multiple GPUs (tensor parallelism) or multiple machines (pipeline parallelism).

### Cost Optimization

- **Spot/preemptible instances:** 60-90% cheaper, but can be reclaimed. Use for batch inference, not latency-critical online serving.
- **Model caching:** Keep hot models loaded on GPU. Cold models loaded on demand with higher latency.
- **Request deduplication:** Identical concurrent requests → serve one result to all.
- **Right-sizing:** Don't use a V100 for a logistic regression. Match hardware to model complexity.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand the difference between online and batch serving and when to use each. They should be able to describe a basic serving pipeline: model receives features, computes prediction, returns result. For a recommendation system, they should mention candidate generation + ranking as a two-stage pipeline. They differentiate by showing awareness of latency constraints and basic optimization (batching, caching).

### Senior Engineer

Senior candidates demonstrate fluency with the full inference stack. They can decompose latency into components (feature lookup, model inference, post-processing) and identify the bottleneck. They proactively discuss quantization, feature store architecture, and the tradeoff between online and batch features. For a search system, a senior candidate would detail the multi-stage pipeline (retrieval → ranking → re-ranking), explain latency budgets at each stage, and discuss how to handle feature serving for real-time context.

### Staff Engineer

Staff candidates think about serving as a cost-quality tradeoff optimization problem. They recognize that the most expensive part of serving is often not model inference — it's feature computation, data transfer, and infrastructure overhead. A Staff candidate might propose a hybrid architecture where stable predictions are batch-served from a cache, real-time adjustments are computed on-the-fly, and an intelligent router decides when cached results are fresh enough vs when live inference is needed. They also consider the operational aspects: how to deploy model updates safely (canary, blue-green), how to handle traffic spikes (auto-scaling, load shedding), and how to debug latency regressions in production.
