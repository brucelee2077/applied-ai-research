# Chapter 01: Introduction & The GenAI System Design Framework

## What Even IS Generative AI?

Imagine you have two friends:

- **Detective Dana** -- She looks at a photo and tells you "that's a cat" or "that's a dog." She's amazing at *recognizing* things, but she can't draw to save her life.
- **Artist Alex** -- Give him a description like "a cat wearing a tiny top hat," and he'll *paint it from scratch*. He doesn't just recognize cats -- he understands cats so deeply that he can create brand new ones.

**Detective Dana is a Discriminative model.** She learns the *boundary* between cats and dogs (P(y|x) -- "given this image, what label?").

**Artist Alex is a Generative model.** He learns *what cats actually look like* (P(x) or P(x|y) -- "what does the data itself look like?"). Because he understands the full distribution of "cat-ness," he can dream up new cats that never existed.

**Generative AI = building Artist Alexes at scale.** It's AI that creates new content -- text, images, video, code, music -- by learning the underlying patterns of existing data.

---

## Discriminative vs. Generative Models

| | Discriminative (Detective) | Generative (Artist) |
|---|---|---|
| **What it learns** | Decision boundary between classes | The full data distribution |
| **Math** | P(y\|x) -- probability of label given data | P(x) or P(x\|y) -- probability of data (optionally given label) |
| **Output** | A label/class/number | New data that looks real |
| **Examples** | Logistic regression, SVM, BERT classifier, ResNet | GPT, DALL-E, Stable Diffusion, GANs, VAEs |
| **Analogy** | "Is this a cat or dog?" | "Draw me a new cat" |
| **Training signal** | "You said cat but it was dog, fix that" | "This generated cat doesn't look real enough, fix that" |

### The Deep Technical Truth

Discriminative models only need to learn the decision boundary -- the "line" separating cats from dogs. They throw away everything else about what cats or dogs look like.

Generative models must learn the entire joint distribution P(x, y) or at least P(x). This is fundamentally harder because you need to model every pixel, every word, every feature -- not just which side of a line they fall on.

**Staff-level insight:** Generative models are a superset. If you truly know P(x, y), you can derive P(y|x) via Bayes' theorem. But discriminative models can't go the other direction. This is why generative models are harder to train but more powerful.

---

## The GenAI Zoo: Model Families

### 1. Autoregressive Models (The Storytellers)
**Analogy:** Imagine writing a story one word at a time. Each word you write depends on all the words before it. You never go back -- you only move forward.

- **How:** Model P(x) = P(x_1) * P(x_2|x_1) * P(x_3|x_1,x_2) * ...
- **Examples:** GPT-4, Llama, PaLM, Claude
- **Strengths:** Excellent at sequential data (text), scales well
- **Weaknesses:** Slow generation (one token at a time), can't "look ahead"

### 2. Variational Autoencoders (VAEs) -- The Compressors
**Analogy:** Imagine squeezing a photo through a tiny tube (bottleneck). The encoder crushes it into a small code, and the decoder tries to rebuild it. The "latent space" in the middle is like a map of all possible images.

- **How:** Encode data to latent distribution, sample, decode back. Trained with reconstruction loss + KL divergence
- **Examples:** VQ-VAE (used in DALL-E 1), image tokenization
- **Strengths:** Structured latent space, fast generation, stable training
- **Weaknesses:** Often blurry outputs (because averaging over uncertainty)

### 3. GANs (Generative Adversarial Networks) -- The Forger vs. The Detective
**Analogy:** A counterfeiter (Generator) tries to make fake money. A bank inspector (Discriminator) tries to catch fakes. They keep getting better at fooling each other until the fakes are indistinguishable from real bills.

- **How:** Two networks play a minimax game: Generator minimizes, Discriminator maximizes
- **Examples:** StyleGAN, ProGAN, BigGAN
- **Strengths:** Sharp, high-quality outputs
- **Weaknesses:** Mode collapse (Generator finds one trick that works and keeps doing it), training instability, no density estimation

### 4. Diffusion Models (The Noise Cleaners)
**Analogy:** Imagine taking a beautiful painting and slowly adding random paint splatters until it's pure noise. Now train a model to reverse this process -- given a noisy mess, clean it up one step at a time. At generation time, start from pure noise and denoise step by step into a masterpiece.

- **How:** Forward process adds Gaussian noise over T steps. Reverse process learns to predict and remove noise at each step.
- **Examples:** Stable Diffusion, DALL-E 2/3, Imagen, Sora
- **Strengths:** State-of-the-art quality, stable training, good mode coverage
- **Weaknesses:** Slow generation (many denoising steps), computationally expensive

### 5. Flow-Based Models (The Shape-Shifters)
**Analogy:** Imagine morphing a simple blob of clay (Gaussian noise) into a sculpture through a series of reversible transformations. You can go forwards AND backwards exactly.

- **How:** Chain of invertible transformations with tractable Jacobians
- **Examples:** Glow, RealNVP, Flow Matching
- **Strengths:** Exact likelihood computation, exact inference
- **Weaknesses:** Architectural constraints (must be invertible), historically lower quality

---

## The 7-Step GenAI System Design Framework (The Pizza Analogy)

Think of designing a GenAI system like opening a pizza restaurant from scratch. Every system -- whether it's ChatGPT, DALL-E, or Google Translate -- follows the same 7 steps.

### Step 1: Clarifying Requirements
**Pizza:** "What kind of pizza place are we opening? Delivery only? Dine-in? What's on the menu? How many customers per hour?"

**In ML:**
- What is the input? What is the output?
- What are the functional requirements? (e.g., "generate a reply given an email thread")
- What are the non-functional requirements? (latency, throughput, safety, cost)
- What constraints exist? (budget, team size, timeline, existing infra)
- Who are the users? What's the scale?

**Staff-level tip:** Always clarify whether the system needs to be real-time, how much latency is acceptable, and whether there are safety/content moderation requirements. These shape everything downstream.

### Step 2: Frame as an ML Task
**Pizza:** "We need a recipe. What type of dish is this? Is it baked? Fried? What's the core cooking technique?"

**In ML:**
- What type of ML problem is this? (text generation, image generation, translation, etc.)
- What is the input representation? (tokens, pixels, embeddings)
- What is the output representation? (probability distribution over tokens, pixel grid, etc.)
- What's the loss function?
- Is this supervised, self-supervised, or reinforcement learning?

**Staff-level tip:** Frame the problem as precisely as possible. "Text generation" is too vague. "Autoregressive next-token prediction over a vocabulary of 50k BPE tokens with cross-entropy loss" is what the interviewer wants to hear.

### Step 3: Data Preparation
**Pizza:** "Where do we get ingredients? Fresh tomatoes from the farm? What quality checks do we do? How do we prep them?"

**In ML:**
- What training data do we need? How much?
- Where does it come from? (public datasets, proprietary, synthetic, user data)
- How do we clean it? (dedup, filter toxic content, quality scoring)
- How do we tokenize/preprocess?
- Data augmentation strategies?
- Train/val/test splits and potential data leakage?

**Staff-level tip:** Data quality > data quantity. Discuss data flywheel effects (user interactions improve the model, which attracts more users). Mention data poisoning risks and PII handling.

### Step 4: Model Development
**Pizza:** "What's our oven? Brick oven? Conveyor belt? What temperature? How long to bake?"

**In ML:**
- Architecture selection (Transformer, U-Net, DiT, etc.)
- Model size and why (parameter count, scaling laws)
- Training strategy (from scratch, pretrain+finetune, transfer learning)
- Key hyperparameters (learning rate, batch size, context length)
- Training infrastructure (distributed training, mixed precision)
- Decoding/generation strategy (beam search, top-k, top-p, temperature, CFG)

**Staff-level tip:** Justify your architecture choice with first principles. "We use a decoder-only Transformer because our task is autoregressive text generation, and scaling laws show decoder-only models are more compute-efficient for this than encoder-decoder when the task doesn't require bidirectional context."

### Step 5: Evaluation
**Pizza:** "How do we know the pizza is good? Taste tests? Customer reviews? Health inspections?"

**In ML:**
- Offline metrics (BLEU, ROUGE, FID, perplexity, CLIPScore, etc.)
- Online metrics (user engagement, task completion rate, A/B tests)
- Human evaluation (Likert scales, side-by-side comparisons, Elo ratings)
- Safety evaluation (toxicity, bias, hallucination rates)
- Failure mode analysis (when does the model break?)

**Staff-level tip:** Always discuss the gap between offline and online metrics. A model with great BLEU can still produce outputs users hate. Mention human eval as the gold standard but discuss its cost and scaling challenges.

### Step 6: Overall System Design
**Pizza:** "How does the whole restaurant work? Front of house, kitchen, supply chain, ordering system, delivery fleet?"

**In ML:**
- End-to-end system architecture (data pipeline, model serving, API layer, client)
- How components interact (retrieval + generation for RAG, encoder + decoder for translation)
- Caching strategy (KV cache for Transformers, prompt caching)
- Content moderation and safety layer placement
- Database and storage (vector DB for RAG, feature store)

**Staff-level tip:** Draw the architecture diagram. Show data flow. Identify bottlenecks. This is where you demonstrate systems thinking, not just ML knowledge.

### Step 7: Deployment & Monitoring
**Pizza:** "We're open! How do we handle the dinner rush? What if a chef calls in sick? How do we know if quality is slipping?"

**In ML:**
- Serving infrastructure (GPU allocation, batching, model parallelism)
- Latency optimization (quantization, distillation, speculative decoding)
- Monitoring (latency percentiles, error rates, drift detection, toxicity monitoring)
- A/B testing and gradual rollout (canary deployments)
- Feedback loops (user feedback -> retraining pipeline)
- Cost management (inference cost per query, GPU utilization)

**Staff-level tip:** Discuss the feedback loop. How does user feedback get incorporated? How do you detect model degradation? What's your rollback strategy? Mention observability (logging prompts/responses for debugging, but with PII considerations).

---

## Key Vocabulary

| Term | Simple Definition | Technical Definition |
|---|---|---|
| **Autoregressive** | Generating one piece at a time, each based on what came before | Factoring P(x) as a product of conditionals: P(x_t \| x_{<t}) |
| **Latent space** | A compressed "map" of all possible outputs | Low-dimensional learned representation where similar inputs cluster together |
| **Tokenization** | Chopping text into small pieces the model can digest | Converting raw input into discrete tokens from a fixed vocabulary (BPE, WordPiece, SentencePiece) |
| **Sampling** | Rolling dice to pick the next output | Drawing from the model's predicted probability distribution at generation time |
| **Temperature** | How "creative" vs "boring" the model's outputs are | Scalar that divides logits before softmax; T<1 = peakier (less random), T>1 = flatter (more random) |
| **Top-k / Top-p** | Only consider the best few options at each step | Truncating the distribution to the top k tokens or the smallest set whose cumulative probability exceeds p |
| **KL Divergence** | How different two probability distributions are | D_KL(P \|\| Q) = sum of P(x) * log(P(x)/Q(x)); measures information lost when Q approximates P |
| **Mode collapse** | The model gets stuck making the same thing over and over | Generator in a GAN converges to producing a single (or few) outputs regardless of input noise |
| **Scaling laws** | Bigger model + more data + more compute = predictably better | Power-law relationships between model performance and compute/data/parameters (Kaplan et al., Chinchilla) |
| **FID** | How realistic are generated images compared to real ones? | Frechet Inception Distance: distance between feature distributions of real and generated images in Inception-v3 feature space |
| **Perplexity** | How surprised is the model by real text? Lower = better. | exp(-1/N * sum of log P(x_i)); exponentiated average negative log-likelihood |
| **RLHF** | Teaching the model from human thumbs-up/thumbs-down | Reinforcement Learning from Human Feedback: train a reward model on human preferences, then optimize the LLM policy via PPO or DPO |

---

## Interview Cheat Sheet

### Opening Move (First 2 Minutes)
When given a GenAI design problem, always:
1. **Repeat the problem** back in your own words
2. **Ask 3-5 clarifying questions** (scale, latency, safety, input/output format)
3. **State your framework**: "I'll walk through this in 7 steps: requirements, ML framing, data, model, evaluation, system design, and deployment."

### Power Phrases (Use These)
- "Let me frame this as an ML task: the input is ___, the output is ___, and the objective is ___."
- "From a scaling laws perspective, we'd want to consider the compute-optimal tradeoff between model size and training tokens."
- "The key architectural decision here is ___ because ___."
- "For evaluation, I'd use ___ as the offline metric, but the true north star is ___ measured via A/B test."
- "The main failure mode I'm worried about is ___, and here's how I'd mitigate it."
- "There's a tension between ___ and ___, and the right tradeoff depends on ___."

### Common Follow-Up Questions
| Question | What They're Really Asking |
|---|---|
| "How would you scale this?" | Distributed training, model/data/pipeline parallelism, serving infrastructure |
| "What if latency is critical?" | Quantization, distillation, caching, speculative decoding, smaller model |
| "How do you handle safety?" | Content filtering pipeline, RLHF, guardrails, red teaming, monitoring |
| "How would you evaluate?" | Offline metrics vs online metrics vs human eval, and when each matters |
| "What could go wrong?" | Hallucinations, bias, drift, adversarial attacks, data quality issues |

### The 30-Second Framework Pitch
"Every GenAI system follows the same blueprint: (1) clarify what we're building and its constraints, (2) frame it precisely as an ML task with clear inputs, outputs, and objectives, (3) prepare high-quality data with proper cleaning and tokenization, (4) select and train the right architecture justified by scaling laws, (5) evaluate with both automated metrics and human judgment, (6) design the end-to-end system with all supporting infrastructure, and (7) deploy with proper monitoring, safety guardrails, and feedback loops for continuous improvement."

---

## What's Next?

With this framework in your toolkit, every subsequent chapter is just filling in the details for a specific product:

- **Chapter 02 (Gmail Smart Compose):** The framework applied to autoregressive text completion
- **Chapter 03 (Google Translate):** The framework applied to encoder-decoder translation
- **Chapter 04 (ChatGPT):** The framework applied to conversational LLMs with RLHF

The framework is your skeleton. Each chapter adds the muscles and skin for a specific system.
