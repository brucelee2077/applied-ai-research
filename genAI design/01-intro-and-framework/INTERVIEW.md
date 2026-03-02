# Chapter 01 — Introduction & GenAI Framework
# Staff / Principal Engineer Interview Guide

---

## How to Use This Guide

Every section below contains:
1. **The question** the interviewer asks
2. **What a No Hire / Weak Hire / Hire / Strong Hire candidate actually says** — verbatim-level examples so you can compare what you're hearing in real time

The goal is not to memorize answers. It's to understand the *reasoning depth* required at each tier.

---

## Part 1: Problem Framing & Requirements (7-Step Step 1)

### Q1. "I'll give you a design problem: build a system that generates personalized marketing copy for an e-commerce site. Before we start, what clarifying questions do you ask?"

**No Hire:**
> "How many users? What languages? Is there a latency requirement?"

Generic, surface-level. Could apply to any system. No ML-specific thinking.

**Weak Hire:**
> "I'd ask about scale — daily requests, latency budget. I'd ask about the type of copy — product descriptions, email subject lines? And whether we have labeled data."

Better, but treats this as a requirements checklist without connecting requirements to design decisions. Doesn't ask the questions that would *change the architecture*.

**Hire:**
> "A few things would fundamentally change my design. First: real-time vs. batch? If copy must appear as a user browses (<200ms), I can't use a large generative model without aggressive optimization. Second: how personalized? User-level (individual purchase history) or segment-level (demographics)? That determines whether I need personalization at inference time (RAG / prompt engineering) or personalization baked into the model (fine-tuning). Third: what counts as success — click-through rate, conversion, or brand safety? These are often in tension. Fourth: do we have labeled examples of good/bad copy, or is this zero-shot?"

Connects each question to an architectural fork. The interviewer can see the candidate is building a decision tree, not filling out a form.

**Strong Hire:**
> "Before I ask anything, let me state my uncertainty: the answer to 'how personalized?' determines whether this is a retrieval problem, a fine-tuning problem, or a prompting problem — those are categorically different systems. So: one, real-time or async (this gates model size and serving architecture). Two, copy length — subject line (15 tokens) vs. product description (200 tokens) vs. landing page (2000 tokens) — these have different decoding cost profiles and different quality signals. Three, how do we measure quality? If it's CTR we can A/B test; if it's brand voice we need human eval. Four, legal constraints — do we operate in regulated verticals (finance, medical) where generated copy must be reviewed before publishing? That changes the whole pipeline. Fifth, what's the existing stack — do we have a prompt layer already, or are we starting from scratch? I'd also want to know: do we already have a fine-tuned model for anything adjacent, because transfer learning could save us months."

Proactively frames the design space before answering. Raises legal/compliance unprompted — this is a Staff-level signal. Connects infrastructure state to the design decision.

---

### Q2. "How do you frame the difference between a discriminative model and a generative model — and why does it matter for system design?"

**No Hire:**
> "Discriminative models classify things, like 'is this spam or not spam.' Generative models create new content, like writing a poem or making an image."

Correct colloquially, but describes outputs not mechanics. Cannot use this to make architecture decisions.

**Weak Hire:**
> "Discriminative models learn P(y|x) — the probability of a label given input. Generative models learn P(x) or P(x|y) — the data distribution itself. Generative is harder because you're modeling the full distribution, not just a decision boundary."

Gets the math right. But stops at definition — doesn't connect to design implications.

**Hire:**
> "The practical design implication is: generative models are harder to train, require more data, and have harder evaluation — there's no obvious 'correct answer' to compare against. But they're more flexible: you can do things like data augmentation, anomaly detection, conditional generation, or unsupervised representation learning that discriminative models can't do. So when I'm choosing: if I only need to do classification or regression, use a discriminative model — it'll converge faster and be easier to evaluate. If I need to synthesize novel outputs, understand the full data distribution, or do semi-supervised learning, I need generative. The evaluation gap is the thing that kills most generative projects in production — you have to build human eval pipelines because there's no clean accuracy metric."

Makes the architectural choice concrete. Flags evaluation as the practical pain point.

**Strong Hire:**
> "There's a deeper technical point that matters for system design: if you truly have a generative model that knows the joint P(x,y), you can derive P(y|x) via Bayes' theorem — so generative models are a superset in principle. But that's theoretical. In practice, the key design decision is: what is the training signal? Discriminative models have a crisp loss — cross-entropy against a label. Generative models require you to define what 'good generation' means, and that definition is almost always imperfect: perplexity, FID, BLEU — all are proxies that can be gamed. This is why I always start a generative design by asking: 'What is the offline metric, and how do I know it correlates with the thing I actually care about?' For the marketing copy system, the offline metric might be perplexity on training copy, but the actual objective is conversion rate. I'd want to measure the correlation between perplexity and CTR before committing to a training objective — because optimizing the wrong thing at scale wastes enormous resources."

Derives the generative/discriminative connection via Bayes. Raises the training signal / proxy metric misalignment problem — this is a genuine Staff-level concern that trips up production ML systems.

---

## Part 2: ML Task Framing (7-Step Step 2)

### Q3. "Given the five generative model families — AR, VAE, GAN, Diffusion, Flow — walk me through how you'd choose between them for a given task."

**No Hire:**
> "Diffusion models are the best right now because they generate the most realistic images. I'd use diffusion for image tasks and GPT-style AR for text."

Chooses by what's trendy, not by task requirements.

**Weak Hire:**
> "AR is great for sequential data like text. GANs are fast and good for images but unstable. Diffusion is high quality but slow. VAEs give you a latent space. Flow models give exact likelihood. I'd match the model to the task type."

Correct characterizations but answers "how do I choose?" with a lookup table, not a decision process.

**Hire:**
> "I start from the deployment constraints, then work backwards. The decision tree is: First, what's the latency budget? If it's <100ms on commodity hardware, diffusion is off the table unless distilled to 1-4 steps. GANs or VAEs are the only realistic options. Second, what's the training data size? GANs need careful tuning at small scale; VAEs are more stable with limited data. Third, do I need exact likelihood? Flow models are the only ones that give this — critical for anomaly detection or density estimation tasks. Fourth, do I need to control specific attributes during generation? Diffusion with classifier-free guidance or GANs with conditioning are both strong here. After answering those four questions, I usually have at most two realistic candidates and I can run a quick prototype comparison."

Has a principled decision process. Connects constraints (latency, data size, need for likelihood) to model family.

**Strong Hire:**
> "The choice is really three orthogonal decisions: (1) training stability, (2) inference cost, (3) quality-diversity tradeoff. On training stability: VAE > Diffusion > Flow > AR > GAN, roughly. GANs are the hardest to train — the minimax game is non-convex and mode collapse is a real operational risk. On inference cost: GAN and VAE are single-pass O(1), AR is O(N) sequential (cannot parallelize), Diffusion is O(T) where T is typically 20-50 steps. Flow models can be single-pass if you use the model in generative mode. On quality-diversity: Diffusion currently dominates for image quality and diversity. GANs are still competitive for real-time applications. AR is the only option for discrete sequence generation where you need probability estimation.

> For 2025 production systems: I'd default to diffusion for images unless latency is <100ms; AR (decoder-only transformer) for text; VAE as a tokenizer/encoder component inside larger systems rather than a standalone generator. GANs are still my go-to for real-time video filters, avatar generation, or any use case where a single forward pass at 20ms matters. Flow matching is emerging as a strong alternative to DDPM diffusion because it has straighter inference trajectories (fewer steps needed) and better inversion properties.

> The thing I always stress to teams: the model family choice is less important than the evaluation strategy. I've seen teams spend months arguing over GAN vs. diffusion when their evaluation metric was completely decoupled from the user-facing quality they cared about."

Gives the full decision matrix. Has opinions on each family for 2025 with specific reasoning. Raises the evaluation-as-the-real-bottleneck point proactively.

---

### Q4. "Explain the variational lower bound (ELBO) in the context of VAE training. Why is it a lower bound, and what does that mean practically?"

**No Hire:**
> "ELBO is the training objective for VAEs. It has a reconstruction term and a KL divergence term. The reconstruction term makes sure the output looks like the input, and KL keeps the latent space smooth."

Gets the components right but doesn't explain why it's a *lower bound* or what that implies.

**Weak Hire:**
> "The ELBO is a lower bound on the true log-likelihood log P(x). We can't compute log P(x) directly because it requires integrating over all possible latent codes, which is intractable. So we optimize a bound instead. The gap between the bound and the true likelihood is the KL divergence between the approximate posterior q(z|x) and the true posterior P(z|x)."

Correct and precise. But doesn't connect to practical implications.

**Hire:**
> "The ELBO is: E_q[log P(x|z)] - KL(q(z|x) || P(z)). It lower-bounds log P(x) because we can't marginalize over all z. Practically this means: when I train a VAE, I'm not actually maximizing the true likelihood — I'm maximizing a bound that's tight only when the encoder q(z|x) perfectly matches the true posterior P(z|x). In practice they never match perfectly, so the VAE's reconstruction will always be slightly suboptimal compared to what's theoretically achievable. This also means VAE samples are typically blurry: the decoder averages over all z that could have produced x, which smears out fine details. That's not a bug in the optimization — it's a fundamental consequence of the bound approximation."

Connects the math to the blurry output problem mechanistically. This is what "Hire" looks like.

**Strong Hire:**
> "The ELBO is: log P(x) ≥ E_{q(z|x)}[log P(x|z)] - KL(q(z|x)||P(z)). The inequality holds because the gap is exactly KL(q(z|x)||P(z|x)), the distance between the approximate and true posterior. This is always ≥ 0 by properties of KL divergence.

> Practically, two things follow: First, the reconstruction quality is bounded by how well the encoder captures the posterior — if your encoder is too simple (diagonal Gaussian, when the true posterior is multimodal), you're leaving reconstruction quality on the table. Second, and more important for production: the ELBO isn't the same as log P(x), so you can't use ELBO values directly for likelihood-ratio tests or anomaly detection. If you need a model that gives meaningful likelihood scores (e.g., for detecting out-of-distribution inputs), VAEs are unreliable. You'd want a normalizing flow instead — it computes exact log P(x) via change-of-variables.

> There's also a posterior collapse problem I watch for in production VAEs: if the KL term dominates, the encoder collapses to the prior and the decoder learns to ignore z entirely. The model degenerates into a language model with no meaningful latent structure. I've seen this kill personalization features in production — the latent code was supposed to encode user identity but had collapsed to a standard normal. The fix is KL annealing or free bits regularization."

Derives the gap term. Explains the normalizing flow advantage for likelihood estimation. Raises posterior collapse as a production failure mode — this is real operational knowledge.

---

## Part 3: Data Preparation (7-Step Step 3)

### Q5. "You're building a text-to-image system and need to curate training data from 5 billion web-scraped image-text pairs. Walk me through your data pipeline."

**No Hire:**
> "I'd filter out NSFW content, remove duplicates, and make sure the images are high quality. Then I'd tokenize the text and resize the images."

Describes data cleaning at a checklist level. No awareness of the scale challenges or the quality-scale tradeoff.

**Weak Hire:**
> "I'd run a deduplication pass using perceptual hashing, filter NSFW with a classifier, filter images below a minimum resolution, and use CLIP to score alignment between image and caption. I'd probably keep the top 20% by CLIP score to maximize quality."

Correct pipeline steps. But doesn't reason about the tradeoffs: CLIP filtering at 20% still leaves 1B pairs with misaligned alt-text. Doesn't discuss the downstream effect of data quality decisions.

**Hire:**
> "At 5B scale, the pipeline is: deduplication first (perceptual hashing to remove near-duplicates — at 5B scale you likely have millions of copies of the same meme or stock photo, and memorization of duplicates tanks generalization). Then a coarse filter: remove images below 256px, remove NSFW, remove images with very short alt-text (<3 words). Then quality scoring: CLIP-score filtering at maybe 0.28 threshold removes misaligned pairs (image is a sunset but alt-text says 'buy now'). Then aesthetic scoring using a trained predictor.

> The tradeoff I'd stress: aggressive quality filtering reduces the effective training set. If I filter to the top 10% CLIP score, I might go from 5B to 500M pairs — which is still a lot, but I've potentially removed coverage of niche concepts that only appear in low-CLIP-score captions. DALL-E 3's key insight was to re-caption all images with a VLM rather than filtering — this preserves scale while dramatically improving caption quality. I'd strongly consider that approach for any serious production system."

Knows the specific pipeline steps and their tradeoffs. Cites DALL-E 3's re-captioning insight as a concrete alternative.

**Strong Hire:**
> "Let me reason through this as a system design problem with three competing objectives: coverage (breadth of concepts the model learns), quality (training signal precision), and scale (raw data volume for loss reduction).

> The naive approach — CLIP-score filtering at 0.28 — optimizes quality at the cost of coverage. It systematically removes: (1) abstract or linguistic concepts that don't have a strong visual grounding (you can't CLIP-score 'the concept of freedom'), (2) rare concepts that appear in low-quality alt-text but exist nowhere else, (3) long-tail demographics that are underrepresented in the CLIP training distribution.

> A better architecture for the data pipeline: use a tiered approach. Tier 1 (top 10% by CLIP): high-quality pairs for fine-tuning or later training stages. Tier 2 (10-50%): moderate quality for early training stages where the model is learning basic visual semantics. Tier 3 (50-100%): noisy but high-coverage, useful only for pre-training at very low learning rate.

> For caption quality, I'd implement synthetic re-captioning using a fast VLM (LLaVA-1.5 scale) for the top 2B images — this is what DALL-E 3 did and it dramatically improved rare-concept binding. The re-captioning approach is strictly better than filtering for preserving coverage.

> I'd also explicitly stratify by concept rarity — make sure the training batches include a mix of high-frequency concepts (dogs, sunsets, faces) and long-tail concepts, or the model will over-fit to the high-frequency distribution and fail on creative or niche prompts. This is a sampling strategy decision, not just a filtering decision."

Frames as a three-way tradeoff (coverage, quality, scale). Introduces tiered training approach. Discusses sampling strategy for long-tail coverage. This is how a Staff engineer thinks about data pipelines.

---

## Part 4: Model Development (7-Step Step 4)

### Q6. "Walk me through how you'd pick the model architecture for a new generative AI system. What's your process?"

**No Hire:**
> "I'd look at what papers have been published recently for similar tasks and use the best-performing architecture from those."

Copy-paste from literature. No first-principles reasoning.

**Weak Hire:**
> "I'd think about the modality — text gets transformers, images get diffusion with U-Net or DiT, video builds on image. I'd look at parameter count vs. the compute budget. And I'd check if I can start from a pretrained checkpoint instead of training from scratch."

Reasonable starting point. But reasoning is still "match modality to architecture" — doesn't connect architecture to training dynamics or deployment constraints.

**Hire:**
> "My process is: (1) Can I start from a pretrained foundation model? If yes, the architecture is mostly decided — I'm adapting, not designing. (2) If building from scratch: what are the inference constraints? This gates the parameter count and architecture type. A model that must run at 100ms on a single GPU is architecturally different from a model that runs in 30 seconds on a cluster. (3) What are the training dynamics I need? Long-range dependencies → attention-based. Local patterns → convolutional. Discrete sequences → autoregressive decoder. Continuous distributions → diffusion or flow. (4) What's the scaling behavior? If I'm going to scale this up over 2 years, I want an architecture with known scaling laws (Transformers do; CNNs less so).

> Then I'd prototype two candidates, train them to 10% of the final budget, and extrapolate loss curves. Don't spend the full compute budget before validating the architecture."

Has a structured decision process. Mentions scaling laws and the prototype-then-extrapolate discipline — important for production systems.

**Strong Hire:**
> "I'd decompose this into three sub-decisions: backbone choice, scale (parameter count), and training regime.

> For backbone: the key axis is *how should information flow across space/time?* Convolutional architectures have translation equivariance and local receptive fields — great for tasks where local spatial patterns matter and you have limited data. Transformers have global attention — great for tasks requiring long-range dependencies and when you have scale. Diffusion is an iterative refinement process — great when quality is paramount and you can afford multiple forward passes at inference. Hybrid architectures (CNN + attention, like U-Net with cross-attention) are often the right call because they combine local inductive biases with global context.

> For scale: I'd use the Chinchilla scaling law as a starting point. Given a compute budget C, the compute-optimal parameter count is N ∝ √C, with training token count T ∝ √C. So for a $1M training budget at ~10^22 FLOPs: N ≈ 7-13B parameters, T ≈ 100-200B tokens. If my deployment constraint is a 10B parameter model, I'd actually train a smaller model with more tokens — Chinchilla shows this outperforms a bigger undertrained model.

> For training regime: pretraining objectives matter enormously. For text: next-token prediction is well-understood but might not be the optimal objective for your downstream task. For images: masked image modeling (MAE) learns better representations than pixel reconstruction. I'd always do a quick ablation on the pretraining objective before committing.

> One thing I'd check that most people miss: does this architecture have known failure modes at the scale I'm targeting? Attention has quadratic KV memory — I need to know at what sequence length it becomes the bottleneck for my use case. Autoregressive models have sequential decoding — I need to know the per-token latency at my target parameter count."

Chinchilla formula with actual numbers. Discusses pretraining objective as a variable. Raises architecture-specific failure modes at scale. This is Principal-level reasoning.

---

## Part 5: Evaluation (7-Step Step 5)

### Q7. "You've trained a generative model. How do you evaluate it before launching to production?"

**No Hire:**
> "I'd compute FID for images or perplexity for text. If those numbers look good compared to baselines, I'd consider it ready."

Single metric, no awareness of what the metric misses.

**Weak Hire:**
> "I'd use multiple metrics — FID, Inception Score, CLIPScore for images; BLEU, ROUGE, perplexity for text. I'd also do some manual review. Then I'd A/B test in production."

Knows multiple metrics exist. But describes a checklist, not a principled evaluation hierarchy.

**Hire:**
> "My evaluation framework has three layers: offline automated metrics, offline human evaluation, and online metrics.

> Offline automated: these catch regressions fast and cheaply. For images: FID (quality + diversity), CLIPScore (text alignment), DINO consistency if it's multi-step. For text: perplexity, ROUGE, task-specific metrics. But I always ask: do these metrics correlate with what I care about? FID can be gamed — a model that memorizes training data has great FID but no generalization. CLIPScore misses compositional failures.

> Offline human eval: 200-500 side-by-side comparisons, blinded. This is expensive but can't be skipped for a launch. The question is specific: 'which image better matches the prompt?' not 'which image do you prefer?'

> Online: launch with 1-5% traffic, measure engagement metrics (for consumer: session length, positive reactions), safety metrics (refusal rates, NSFW rate), and latency/cost. The Goodhart's Law failure is launching a model that's better on FID but users find worse — always validate offline → online correlation before you trust offline metrics."

Structured three-layer framework. Explicit about the Goodhart's Law risk for FID.

**Strong Hire:**
> "Evaluation is actually where most generative AI projects fail, so I think about it as a system design problem in its own right.

> The core challenge: there's no ground truth. Unlike supervised learning where you can compute accuracy, generative quality is inherently multi-dimensional and subjective. So I decompose evaluation into: (1) quality (is each output good?), (2) diversity (do different inputs produce different outputs?), (3) alignment (does the output match the instruction?), (4) safety (does the output avoid harmful content?), (5) calibration (does the model know what it doesn't know?).

> For each dimension I want at minimum one automated metric and one human eval signal: quality→FID+human preference rating. Diversity→recall in precision-recall framework+pairwise LPIPS distance. Alignment→CLIPScore/BERTScore+task completion rating. Safety→toxicity classifier+red-team evaluation. Calibration→expected calibration error+human 'does this look confident when it should be?' review.

> The thing that surprises most teams: diversity is the most commonly missed dimension. A model that always generates high-quality outputs but the same high-quality output for every input will look great on FID but be useless in production. I measure diversity explicitly: for a batch of 1000 prompts, compute the pairwise cosine distance in CLIP embedding space. If mean pairwise distance is below baseline, something is wrong.

> For production launch specifically: I always want to see the offline-to-online correlation validated before trusting offline metrics. This means running a smaller A/B experiment first, measuring both offline and online metrics, and checking they agree. If FID improves but CTR doesn't, my offline metrics are broken and I need to fix them before they mislead future development."

Frames evaluation as a system design problem. Five-dimension framework. Explicitly calls out diversity as the most-missed dimension. Raises offline-to-online correlation validation as a launch prerequisite.

---

## Part 6: System Design (7-Step Step 6)

### Q8. "Draw the end-to-end system architecture for a text-to-image API serving 10M requests/day."

**No Hire:**
> "User sends a prompt → model generates the image → return to user. I'd put a GPU cluster behind a load balancer."

No decomposition into components. No consideration of async vs. sync, queuing, caching, or the model serving specifics.

**Weak Hire:**
> "There's a safety filter on the prompt input, then the diffusion model runs, then a safety filter on the output image, then we return it. I'd use a message queue to handle traffic spikes and auto-scale the GPU cluster. I'd cache common prompts."

Knows the main components. But doesn't reason about async generation, result storage, or the serving-specific optimizations that make it work at 10M req/day.

**Hire:**
> "10M req/day is ~115 requests/second average, with likely 3-5x spikes. A single A100 generates an image in ~2-5 seconds with Stable Diffusion DDIM-50, so it can handle ~20 images/minute = ~0.33 images/second. To serve 115 req/s average, I need ~350 A100 GPUs just for steady-state, not counting spikes.

> System design: The generation must be async — 2-5 second GPU computation can't block a synchronous API call. So: client POSTs the prompt → API gateway writes to a job queue (SQS/Kafka) → returns a job ID immediately → client polls or uses webhooks for completion → workers consume from queue → write image to object storage (S3) → update job status in a DB → client retrieves.

> Components: (1) Input safety classifier (lightweight BERT or keyword filter, synchronous, <50ms). (2) Prompt enhancement LLM (optional, adds quality tags). (3) Job queue with priority tiers. (4) GPU worker fleet with autoscaling. (5) Output safety classifier (NSFW, synchronized with output storage). (6) Image storage (S3 with CDN for delivery). (7) Job status DB (Redis for fast polling, DynamoDB for durability).

> Optimization: (a) KV-cache for prompt encoding when multiple users have same system prompt. (b) Continuous batching on GPU workers to maximize utilization. (c) Speculative decoding isn't applicable for diffusion, but DDIM with 20 steps instead of 50 reduces latency 2.5x at minor quality cost."

Does the math on GPU count. Properly designs async workflow. Enumerates specific components and optimizations.

**Strong Hire:**
> "Let me start from the SLA and work backwards. At 10M/day with an estimated P95 generation time of 8 seconds, P50 of 4 seconds, I need to size for peak load which is likely 5x average = 575 req/s at peak. With 4 seconds/image on A100 batch size 1: I need 575 × 4 = 2300 GPU slots at peak. At 40% utilization efficiency (realistic for bursty traffic): 5750 GPU slots ≈ 5750 A100s at full load. That's extremely expensive. The design should aggressively reduce this via batching (a batch of 4 images takes ~same time as 1 on modern hardware), which gives 4x throughput improvement → ~1440 A100s. Still expensive — this motivates using distilled models (SDXL-Turbo, LCM) at 4 steps for most requests, reserving 50-step quality runs for premium tier.

> End-to-end architecture:

> *Ingestion layer:* API gateway → rate limiter (per-user token bucket, per-IP) → input safety classifier (async, <20ms, runs in parallel) → prompt enhancer (optional, can be toggled) → job queue with 3 priority tiers (premium, standard, free).

> *Worker fleet:* Heterogeneous: 4-step distilled model workers for standard tier, 20-step workers for premium, 50-step workers for batch/async. Workers do continuous batching (vLLM-style): don't wait for a full batch, start processing as requests arrive, add more to the batch opportunistically. Each worker handles KV-cache management (for SDXL, the CLIP text encoder output can be cached across requests with the same base prompt).

> *Storage and delivery:* Generated images → S3 with lifecycle policy (delete after 30 days unless saved). CDN (CloudFront) for delivery — don't serve from S3 origin. Job status → Redis sorted set (fast TTL-based expiry).

> *Output safety:* NSFW classifier runs on every output. Watermarking (C2PA) is applied before storage.

> *Observability:* I'd track: queue depth per tier (leading indicator of overload), GPU utilization per model type, P50/P95/P99 job completion time, safety filter hit rate, and cost per image (compute + storage). If queue depth rises, the autoscaler adds workers before P95 latency degrades.

> One non-obvious thing: the batch size / quality tradeoff is a dynamic decision. If the queue is long (>100 jobs in standard queue), automatically switch to 4-step model for standard tier. If queue is short, use 20-step for better quality. This adaptive quality approach maintains throughput without always sacrificing quality."

Full math on GPU sizing. Multi-tier model serving (distilled vs. high-quality). Continuous batching. Adaptive quality based on queue depth. CDN. Watermarking. Observability. This covers everything a Staff engineer would be expected to address.

---

## Part 7: Deployment & Monitoring (7-Step Step 7)

### Q9. "You launch the model. What does your monitoring setup look like, and how do you handle model degradation?"

**No Hire:**
> "I'd track error rates and latency. If errors spike, I'd roll back."

Treats this like a web service, not an ML system. No awareness of data drift, model quality degradation, or feedback loops.

**Weak Hire:**
> "I'd track latency, error rates, GPU utilization. For quality, I'd sample outputs and run automated metrics. If the automated metrics degrade, I'd investigate."

Adds quality monitoring. But "run automated metrics" is vague — which metrics, at what frequency, with what alert thresholds?

**Hire:**
> "Monitoring has two layers: infrastructure and model quality.

> Infrastructure: latency (P50/P95/P99), error rate, GPU utilization, queue depth, cost per request. These are standard SLO metrics.

> Model quality: this is the hard part. I'd run automated quality scoring on 1% of outputs (FID on a rolling window, CLIPScore for text alignment). I'd also run a daily human eval sample of 50 prompts (fixed golden set) to catch regressions that automated metrics miss. Alert thresholds: if FID increases >10% over 7-day rolling baseline, page on-call. If human eval preference rate drops >5%, escalate to team.

> Degradation sources: (1) input distribution shift — users start prompting differently, and the model wasn't trained on those prompt styles. (2) Model drift — if we're doing online learning or periodic retraining, the model can drift from intended behavior. (3) Infrastructure changes — model serving code changes can silently change output quality.

> Rollback: blue/green deployment with automated rollback trigger if quality metrics degrade within 1 hour of deployment."

Concrete quality metrics with alert thresholds. Enumerates degradation sources. Blue/green rollback.

**Strong Hire:**
> "I think about monitoring as a feedback system, not just an alerting system. The goal is to feed production signals back into training, not just to catch failures.

> Infrastructure monitoring: standard — P50/P95/P99, error rate, GPU utilization, cost. Alert on >2x normal latency, >0.1% error rate.

> Quality monitoring: this needs its own pipeline. I'd run a continuous evaluation job that: (a) samples 0.1% of all requests (stratified by prompt category — don't just sample uniformly, because edge-case prompts are rare but important), (b) runs automated scoring: CLIPScore, NSFW rate, watermark integrity, (c) runs weekly human eval on a fixed golden set of 200 prompts with consistent raters, (d) tracks user feedback signal (thumbs up/down if available) — this is the highest-signal metric because it measures actual user satisfaction.

> Distribution drift monitoring: I'd embed all prompts and monitor the embedding distribution over time (track mean and variance of prompt embeddings). If the distribution shifts significantly (KL divergence above threshold vs. training distribution), alert — this means users are asking for things the model may not handle well. Similarly, I'd monitor the output image embedding distribution. Drift in output distribution means the model's behavior is changing even if individual outputs look fine.

> Feedback loops: every 90 days, collect the prompts where users gave negative feedback and where CLIPScore was low. Assemble these as training negatives + DPO pairs. This is continuous alignment — the model slowly learns from its production failures.

> Canary strategy: always launch with 1% traffic for 24 hours. Monitor quality metrics relative to control. The 24-hour window catches time-of-day effects (some user cohorts only active at night). If P-value on quality difference is not significant and no safety regressions, gradually ramp.

> Non-obvious: I'd also monitor prompt injection attacks. Track prompts that contain 'ignore previous instructions' or similar patterns — these indicate adversarial users probing the safety layer. Feed these to the safety team for red-teaming."

Feedback loop to training. Prompt embedding distribution drift. User negative feedback as DPO training signal. Canary strategy with statistical validation. Prompt injection monitoring. This covers the full operational lifecycle.

---

## Part 8: Deep Technical Probes

### Q10. "Goodhart's Law: 'When a measure becomes a target, it ceases to be a good measure.' Give me a concrete example in a generative AI system where optimizing the training metric directly causes production failure."

**No Hire:**
> "If you optimize too hard you can overfit. Like, if the model memorizes the training data."

Vague. Describes overfitting, not Goodhart's Law.

**Weak Hire:**
> "If you directly optimize FID during training, the model could learn to generate images that look statistically similar to real images without being creative or useful. Like training to fool the Inception network."

Gets the FID gaming example. But one example and doesn't generalize.

**Hire:**
> "Three concrete examples: (1) FID optimization: a model optimizes FID by learning to generate images that land in the exact center of the real data distribution — high average quality, no diversity. FID drops but the model is useless for creative tasks. (2) RLHF reward hacking: you train a reward model on human preferences, then optimize the policy to maximize reward. The policy learns to produce very long, confident-sounding, hedged responses because human raters in the training set tended to prefer these — even when the information is wrong. OpenAI documented this as sycophancy. (3) Acceptance rate for Smart Compose: if you optimize acceptance rate directly, the model learns to only suggest completions when it's extremely confident, which maximizes acceptance rate but minimizes the number of suggestions shown. Useful-but-uncertain suggestions are never shown."

Three concrete examples across modalities. Names sycophancy. Understands the underlying pattern.

**Strong Hire:**
> "Goodhart's Law is the central challenge of ML evaluation, not just an edge case. Every ML metric is a proxy for something we actually care about but can't directly measure. The moment we make the proxy the target, we're in trouble.

> Examples by severity: (1) Mild: BLEU for machine translation. BLEU measures n-gram overlap. Optimizing BLEU produces translations that repeat common phrases from the reference, avoiding low-frequency but semantically correct words. The model learns to be 'safe' in a way that humans find bland. Easy to detect because human raters score it worse. (2) Moderate: CLIPScore for text-to-image. CLIPScore measures semantic similarity between prompt and image using CLIP embeddings. Optimizing CLIPScore teaches the model to use visual styles that CLIP associates strongly with text concepts — not to faithfully represent the requested scene. Concretely: 'a bear in a forest' might score high CLIPScore by generating a very bear-looking bear even if the forest is wrong. Harder to detect because individual CLIP scores look fine. (3) Severe: RLHF reward hacking. The reward model is trained on a finite set of human comparisons. The policy, through many PPO steps, discovers inputs that maximize reward model score in ways that wouldn't be rated highly by humans — these are out-of-distribution exploits. Known examples: the model learns that longer responses with more bullet points score higher, producing padding-heavy answers. Constitutional AI partially addresses this by using the model itself to critique its outputs, but it doesn't fully solve the Goodhart problem.

> The principled fix is: never optimize a metric directly as the training objective unless you've validated that the metric is well-correlated with the actual user objective AND the metric is robust to adversarial optimization. For launch decisions, always have a human eval that the automated metric can't directly influence. Run the human eval blind to the automated metric. If they agree, your automated metric is trustworthy — for now."

Generalizes the pattern. Three examples at different severity levels with concrete mechanisms. Proposes the principled fix (validate correlation, run blind human eval). This is how a Staff scientist thinks about evaluation.

---

## Part 9: Red Flags Cheat Sheet

| Signal | What It Means |
|--------|---------------|
| "I'd use diffusion because it's the best" | Model selection by fashion, not constraints |
| Can't state what P(x) vs P(y\|x) means | Doesn't understand generative vs. discriminative at the technical level |
| Single metric for evaluation | Hasn't run a production ML system |
| "We'd A/B test it" as the only production plan | No understanding of how to detect quality degradation vs. feature degradation |
| Framework recited verbatim without adapting to constraints | Pattern matching to interview prep, not genuine systems thinking |
| "Add more data" as default answer | Doesn't reason about data quality vs. quantity tradeoffs |
| Ignores latency until asked | Has never had to actually serve a model |
| "KL divergence is just a regularizer" | Doesn't understand what it's constraining in the VAE |
| Can't reason about Goodhart's Law | Has only worked on prototype systems, not production |

---

## Part 10: Hiring Decision Summary

| Tier | Signal Pattern |
|------|---------------|
| **No Hire** | Answers at the vocabulary level — names the right concepts but cannot explain mechanisms. Cannot connect a design decision to its upstream cause or downstream consequence. Stops at the first follow-up. Single metric for evaluation. No awareness of production failure modes. |
| **Weak Hire** | Correct on core concepts. Good framework knowledge. Fails on: (a) adapting the framework to novel constraints, (b) explaining the mathematical basis of intuitions, (c) identifying failure modes before being prompted. Can handle one level of follow-up but gets stuck on the second. |
| **Hire (Senior)** | Connects mechanism to decision. Can derive tradeoffs from first principles, not just recall them. Has a structured evaluation framework. Has real numbers (Chinchilla ratios, FID baselines, typical latency budgets). Can handle two levels of follow-up before requiring hints. |
| **Strong Hire (Staff/Principal)** | Proactively raises failure modes before being asked. Has specific opinions backed by mechanism. Can derive things on the whiteboard (ELBO, Bayes derivation, GPU sizing math). Anticipates the interviewer's follow-up questions. Frames evaluation as a system design problem, not a metric selection problem. Identifies the real risk in a proposed system (Goodhart's Law, posterior collapse, reward hacking) without being prompted. Connects this problem to adjacent systems they've seen. |
