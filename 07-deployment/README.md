# 7. Deployment

## What Is Model Deployment?

You've trained an amazing AI model. It works great on your laptop. But now you want
**real people** to use it -- through a website, an app, or an API. That's **deployment**:
taking a model from "it works on my computer" to "anyone in the world can use it."

Think of it like cooking vs. running a restaurant:
- **Training** a model = perfecting a recipe in your kitchen
- **Deploying** a model = serving that dish to thousands of customers every day

```
+-------------------------------------------------------------------+
|              From Training to Production                           |
|                                                                   |
|   Your Laptop                    Production Server                |
|   +------------------+          +------------------+              |
|   | Model: 7GB       |   --->   | Model: optimized |              |
|   | Speed: 5 sec/req  |          | Speed: 0.1 sec   |              |
|   | Users: just you   |          | Users: thousands  |              |
|   | Cost: free        |          | Cost: $$$         |              |
|   +------------------+          +------------------+              |
|                                                                   |
|   Deployment is about bridging this gap efficiently.              |
+-------------------------------------------------------------------+
```

---

## Why Is Deployment Hard?

AI models (especially LLMs) are **big**, **slow**, and **expensive** to run.
Here are the main challenges:

```
+-------------------------------------------------------------------+
|              The Four Deployment Challenges                        |
|                                                                   |
|   1. SIZE: Models are huge                                        |
|      GPT-2: 1.5 GB    LLaMA-7B: 14 GB    LLaMA-70B: 140 GB     |
|      Your server needs enough memory to hold the whole model      |
|                                                                   |
|   2. SPEED (Latency): Users hate waiting                          |
|      Users expect < 1 second for simple tasks                     |
|      A large model might take 10+ seconds per request             |
|                                                                   |
|   3. COST: GPUs are expensive                                     |
|      A single A100 GPU costs ~$2-3/hour on cloud                 |
|      Serving 1000 users simultaneously = many GPUs = $$$          |
|                                                                   |
|   4. SCALE: Traffic varies                                        |
|      Monday 3 AM: 10 requests/minute                              |
|      Product launch day: 10,000 requests/minute                   |
|      You need to handle both without wasting money                |
+-------------------------------------------------------------------+
```

---

## The Two Main Areas

This module covers two key topics:

### 1. Model Serving
**How do you make your model available to users?**

This covers the infrastructure side: wrapping your model in an API, handling
multiple users at once, scaling up and down, and choosing the right framework.

Think of it as building the **restaurant** -- the kitchen, the ordering system,
the waitstaff, and the seating arrangement.

### 2. Inference Optimization
**How do you make your model faster and smaller?**

This covers techniques to reduce model size and speed up predictions: quantization
(using smaller numbers), pruning (removing unnecessary parts), and distillation
(training a smaller model to mimic a big one).

Think of it as optimizing the **recipe** -- using a simpler cooking method that's
90% as good but 10x faster.

```
+-------------------------------------------------------------------+
|              Deployment = Serving + Optimization                   |
|                                                                   |
|   [Trained Model]                                                 |
|        |                                                          |
|        +---> [Inference Optimization]                             |
|        |       Make it smaller/faster                              |
|        |       - Quantization (smaller numbers)                   |
|        |       - Pruning (remove unnecessary parts)               |
|        |       - Distillation (teach a smaller model)             |
|        |              |                                           |
|        |              v                                           |
|        +---> [Model Serving]                                      |
|                Make it available to users                          |
|                - API endpoints                                    |
|                - Load balancing                                   |
|                - Auto-scaling                                     |
|                       |                                           |
|                       v                                           |
|               [Users / Applications]                              |
+-------------------------------------------------------------------+
```

---

## Study Plan

```
    START HERE
        |
        v
+---------------------------+
|  1. This README            |  Understand deployment challenges
|     (you are here)         |  and the big picture
+-----------+---------------+
            |
            v
+---------------------------+
|  2. Model Serving          |  How to wrap a model in an API
|     (serving/)             |  and serve it to users
+-----------+---------------+
            |
            v
+---------------------------+
|  3. Inference Optimization |  Overview of making models
|     (inference-opt/)       |  faster and smaller
+-----------+---------------+
            |
            v
+---------------------------+
|  4. Quantization           |  Using lower-precision numbers
|     (quantization.md)      |  (the most impactful technique)
+-----------+---------------+
            |
            v
+---------------------------+
|  5. Model Compression      |  Pruning, distillation,
|     (model-compression.md) |  and other size reduction methods
+---------------------------+
```

---

## Directory Structure

```
07-deployment/
+-- README.md                              # You are here
+-- serving/                               # Model serving infrastructure
|   +-- README.md                          #   APIs, frameworks, scaling
+-- inference-optimization/                # Making models faster
|   +-- README.md                          #   Overview of optimization techniques
|   +-- quantization.md                    #   Using lower-precision numbers
|   +-- model-compression.md               #   Pruning, distillation, factorization
+-- experiments/                           # Hands-on practice
    +-- (your experiments go here!)
```

---

## Key Terms

| Term | Simple Explanation |
|------|-------------------|
| **Inference** | Using a trained model to make predictions (as opposed to training it) |
| **Latency** | How long a single request takes (lower is better) |
| **Throughput** | How many requests you can handle per second (higher is better) |
| **Quantization** | Using smaller numbers to represent model weights (e.g., 16-bit instead of 32-bit) |
| **Pruning** | Removing unnecessary connections in a neural network |
| **Distillation** | Training a small "student" model to copy a large "teacher" model |
| **Serving** | Making a model available through an API for users/apps to call |
| **Batching** | Grouping multiple requests together for more efficient processing |
| **GPU** | Graphics Processing Unit -- the hardware that runs AI models fast |
| **VRAM** | Video RAM -- the memory on a GPU (determines what model sizes you can run) |

---

## Key Papers

- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** -- Dettmers et al., 2022
  - Showed how to run large models with 8-bit numbers, halving memory needs
- **SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs** -- Xiao et al., 2022
  - A clever trick to make quantization work better for large language models
- **DistilBERT: A Distilled Version of BERT** -- Sanh et al., 2019
  - Showed a smaller model (40% fewer params) can retain 97% of BERT's performance
- **The Lottery Ticket Hypothesis** -- Frankle & Carlin, 2019
  - Found that neural networks contain small subnetworks that work just as well

---

[Back to Main](../README.md) | [Previous: Evaluation](../06-evaluation/README.md)
