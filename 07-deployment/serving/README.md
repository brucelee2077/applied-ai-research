# Model Serving

## What Is Model Serving?

You've trained a model. Now you want other people (or other programs) to use it.
**Model serving** is the process of making your model available through an API
so anyone can send it a request and get back a prediction.

Think of it like a **restaurant kitchen**:
- The **model** is the chef (does the actual work)
- The **API** is the waiter (takes orders, delivers food)
- The **server** is the kitchen (where everything runs)
- **Scaling** is adding more chefs/kitchens when it gets busy

```
+-------------------------------------------------------------------+
|              How Model Serving Works                               |
|                                                                   |
|   User/App sends a request:                                       |
|     "Translate 'hello' to French"                                 |
|           |                                                       |
|           v                                                       |
|     [API Endpoint]  <-- Receives the request                      |
|           |                                                       |
|           v                                                       |
|     [Pre-processing]  <-- Tokenize text, resize images, etc.     |
|           |                                                       |
|           v                                                       |
|     [Model Inference]  <-- The model makes its prediction         |
|           |                                                       |
|           v                                                       |
|     [Post-processing]  <-- Format the output nicely               |
|           |                                                       |
|           v                                                       |
|     Response: "bonjour"                                           |
+-------------------------------------------------------------------+
```

---

## Building a Simple Model API

The most common way to serve a model is with a **REST API** -- a web endpoint
that accepts requests and returns responses, just like any website.

```
+-------------------------------------------------------------------+
|              A Minimal Model API (Python + FastAPI)                |
|                                                                   |
|   # What the code looks like (simplified):                        |
|                                                                   |
|   from fastapi import FastAPI                                     |
|   app = FastAPI()                                                 |
|                                                                   |
|   model = load_model("my_model.pt")  # Load once at startup      |
|                                                                   |
|   @app.post("/predict")                                           |
|   def predict(text: str):                                         |
|       result = model(text)           # Run inference              |
|       return {"prediction": result}  # Return result              |
|                                                                   |
|   # That's it! Now anyone can call:                               |
|   # POST http://your-server.com/predict                           |
|   # Body: {"text": "Hello world"}                                 |
|   # Response: {"prediction": "..."}                               |
+-------------------------------------------------------------------+
```

---

## Serving Frameworks

You don't have to build everything from scratch. These frameworks handle the
hard parts (batching, GPU management, scaling) for you:

### General-Purpose Frameworks

| Framework | What It Does | Best For |
|-----------|-------------|----------|
| **FastAPI** | Simple Python web framework | Quick prototypes, small scale |
| **Flask** | Even simpler Python web framework | Very basic APIs |
| **Ray Serve** | Distributed serving framework | Scaling to many GPUs/machines |

### ML-Specific Serving Frameworks

| Framework | What It Does | Best For |
|-----------|-------------|----------|
| **TorchServe** | PyTorch's official serving tool | PyTorch models in production |
| **TensorFlow Serving** | TensorFlow's official tool | TensorFlow models |
| **Triton Inference Server** | NVIDIA's high-performance server | Maximum GPU performance |
| **vLLM** | Optimized LLM serving | Serving large language models |
| **TGI (Text Generation Inference)** | HuggingFace's LLM server | HuggingFace models |
| **Ollama** | Run LLMs locally with one command | Local development, personal use |

```
+-------------------------------------------------------------------+
|              Choosing a Framework                                  |
|                                                                   |
|   "I just want to try it out"                                     |
|     --> FastAPI (simplest)                                        |
|                                                                   |
|   "I need to serve an LLM"                                       |
|     --> vLLM or TGI (optimized for language models)              |
|                                                                   |
|   "I need maximum performance"                                   |
|     --> Triton Inference Server (NVIDIA's battle-tested tool)    |
|                                                                   |
|   "I want to run models on my laptop"                            |
|     --> Ollama (one-line setup for popular models)               |
|                                                                   |
|   "I need to scale across many machines"                         |
|     --> Ray Serve (distributed computing framework)              |
+-------------------------------------------------------------------+
```

---

## Key Concepts in Model Serving

### Batching: Processing Multiple Requests Together

Instead of processing one request at a time, **batch** multiple requests
together. GPUs are much more efficient when processing many inputs at once.

```
+-------------------------------------------------------------------+
|              Batching: Why It Matters                              |
|                                                                   |
|   WITHOUT batching (one at a time):                               |
|     Request 1 --> [GPU processes] --> Response 1  (50ms)          |
|     Request 2 --> [GPU processes] --> Response 2  (50ms)          |
|     Request 3 --> [GPU processes] --> Response 3  (50ms)          |
|     Total: 150ms for 3 requests                                   |
|                                                                   |
|   WITH batching (all at once):                                    |
|     Request 1 --+                                                 |
|     Request 2 --+--> [GPU processes batch] --> All responses      |
|     Request 3 --+                              (60ms total!)      |
|     Total: 60ms for 3 requests (2.5x faster!)                    |
|                                                                   |
|   Why? GPUs are designed for parallel processing.                 |
|   Processing 1 item uses only a fraction of the GPU's power.     |
|   Processing a batch uses the GPU more fully.                     |
+-------------------------------------------------------------------+
```

**Dynamic batching:** Wait a few milliseconds to collect incoming requests,
then process them as a batch. Frameworks like Triton and vLLM do this
automatically.

### Caching: Don't Redo Work

If many users ask the same question, why run the model every time?
**Cache** common responses.

```
+-------------------------------------------------------------------+
|              Caching Strategies                                    |
|                                                                   |
|   Exact match cache:                                              |
|     "What is 2+2?" --> Check cache --> Found! Return "4"          |
|     (No model inference needed)                                   |
|                                                                   |
|   Semantic cache:                                                 |
|     "What's 2 plus 2?" --> Similar to cached "What is 2+2?"      |
|     --> Return cached answer                                      |
|                                                                   |
|   KV-Cache (for LLMs):                                            |
|     When generating text token-by-token, cache the intermediate   |
|     computations so each new token doesn't recompute everything.  |
|     This is what makes LLM generation fast.                       |
+-------------------------------------------------------------------+
```

### Streaming: Don't Make Users Wait

For LLMs that generate text token by token, **streaming** sends each token
to the user as it's generated, instead of waiting for the full response.

```
Without streaming:               With streaming:
User waits.....                  User sees:
User waits.....                  "The"
User waits.....                  "The answer"
User waits.....                  "The answer is"
User waits.....                  "The answer is 42"
"The answer is 42" (all at once) (appears word by word, feels faster!)
```

This is how ChatGPT, Claude, and other chatbots show their responses.

---

## Scaling Strategies

What happens when your model gets popular and thousands of people want to
use it at the same time?

```
+-------------------------------------------------------------------+
|              Scaling Approaches                                    |
|                                                                   |
|   1. VERTICAL SCALING (bigger machine)                            |
|      Use a more powerful GPU                                      |
|      A100 (40GB) --> A100 (80GB) --> H100 (80GB)                 |
|      Simple but expensive, and there's a ceiling                  |
|                                                                   |
|   2. HORIZONTAL SCALING (more machines)                           |
|      Run copies of your model on multiple servers                 |
|      +--------+  +--------+  +--------+                          |
|      | Model  |  | Model  |  | Model  |                          |
|      | Copy 1 |  | Copy 2 |  | Copy 3 |                          |
|      +--------+  +--------+  +--------+                          |
|           ^           ^           ^                               |
|           |           |           |                               |
|      +----+-----------+-----------+----+                          |
|      |        LOAD BALANCER            |                          |
|      |  (distributes requests evenly)  |                          |
|      +---------------------------------+                          |
|                    ^                                              |
|                    |                                              |
|              [User requests]                                      |
|                                                                   |
|   3. AUTO-SCALING                                                 |
|      Automatically add/remove servers based on demand             |
|      Busy hour: 10 servers    Quiet hour: 2 servers              |
|      Saves money!                                                 |
+-------------------------------------------------------------------+
```

### Model Parallelism (For Very Large Models)

When a model is too big to fit on one GPU, split it across multiple GPUs:

| Strategy | How It Works |
|----------|-------------|
| **Tensor Parallelism** | Split individual layers across GPUs (each GPU computes part of each layer) |
| **Pipeline Parallelism** | Put different layers on different GPUs (GPU 1 has layers 1-20, GPU 2 has 21-40) |

```
Pipeline Parallelism:
  Input --> [GPU 1: Layers 1-20] --> [GPU 2: Layers 21-40] --> Output

Tensor Parallelism:
  Input --> [GPU 1: Left half of layer] + [GPU 2: Right half] --> Output
```

---

## API Design Best Practices

```
+-------------------------------------------------------------------+
|              Designing a Good Model API                            |
|                                                                   |
|   1. CLEAR ENDPOINTS                                              |
|      POST /v1/predict        -- Make a prediction                |
|      POST /v1/embed          -- Get embeddings                   |
|      GET  /v1/health         -- Is the server running?           |
|      GET  /v1/models         -- What models are available?       |
|                                                                   |
|   2. VERSIONING                                                   |
|      Use /v1/, /v2/ so you can update without breaking users     |
|                                                                   |
|   3. ERROR HANDLING                                               |
|      400: Bad request (invalid input)                             |
|      429: Too many requests (rate limited)                        |
|      500: Server error (model crashed)                            |
|      503: Service unavailable (model loading)                     |
|                                                                   |
|   4. RATE LIMITING                                                |
|      Limit requests per user to prevent abuse                     |
|      Example: 100 requests per minute per API key                |
|                                                                   |
|   5. AUTHENTICATION                                               |
|      Require API keys so you know who's using your model          |
+-------------------------------------------------------------------+
```

---

## Deployment Platforms

Where do you actually run your model?

| Platform | Type | Best For |
|----------|------|----------|
| **AWS SageMaker** | Cloud (managed) | Enterprise, full AWS integration |
| **Google Cloud Vertex AI** | Cloud (managed) | Enterprise, Google ecosystem |
| **Azure ML** | Cloud (managed) | Enterprise, Microsoft ecosystem |
| **Hugging Face Inference Endpoints** | Cloud (managed) | HuggingFace models, easy setup |
| **Modal** | Cloud (serverless) | Pay-per-use, no server management |
| **Replicate** | Cloud (serverless) | Quick deployment of open-source models |
| **Self-hosted (EC2/GCP VM)** | Cloud (manual) | Full control, custom setups |
| **On-premise** | Local | Data privacy, regulatory requirements |

---

## Summary

```
+------------------------------------------------------------------+
|              Model Serving Cheat Sheet                            |
|                                                                  |
|  What:     Making your model available to users via an API       |
|                                                                  |
|  Quick start:  FastAPI + your model = simple API                 |
|  Production:   vLLM (LLMs) or Triton (general) + load balancer  |
|                                                                  |
|  Key techniques:                                                 |
|    Batching:   Process multiple requests together (faster)       |
|    Caching:    Don't recompute repeated requests                 |
|    Streaming:  Send results token-by-token (feels responsive)    |
|    Scaling:    Add more servers when demand increases             |
|                                                                  |
|  Key decisions:                                                  |
|    Framework:  vLLM, TGI, Triton, or FastAPI                    |
|    Platform:   Cloud managed, serverless, or self-hosted         |
|    Scaling:    Horizontal (more servers) vs vertical (bigger GPU)|
+------------------------------------------------------------------+
```

---

## Further Reading

- **vLLM: Efficient Memory Management for Large Language Model Serving** -- Kwon et al., 2023
  - The PagedAttention paper that makes LLM serving much more efficient
- **Orca: A Distributed Serving System for Transformer-Based Generative Models** -- Yu et al., 2022
  - Continuous batching for LLM serving
- **TensorRT-LLM Documentation** -- NVIDIA
  - High-performance LLM inference optimization

---

[Back to Deployment](../README.md)
