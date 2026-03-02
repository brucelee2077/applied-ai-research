# Chapter 04: Interviewer Guide — ChatGPT

## Opening Problem Statement

"Design a production conversational AI assistant similar to ChatGPT. Your system needs to serve millions of concurrent users, maintain multi-turn context, and be aligned with human preferences to be safe, helpful, and honest. I'll want deep dives on the RLHF training pipeline, the inference serving architecture, and how you'd handle the most difficult failure modes — reward hacking, multi-turn jailbreaks, and latency at scale."

---

## Probing Question Tree

### Area 1: RLHF — Reward Model Failure Modes

**Surface:** "Walk me through the RLHF pipeline — reward model training, PPO optimization, KL penalty."

**Probe 1:** "The README correctly describes the KL penalty. Now explain reward hacking: give me a specific, concrete example of what a language model actually learns to exploit in a reward model, and how you'd detect it in production."
> *Looking for:* Concrete examples: (1) Reward model trained on short/medium responses shows higher scores for very long responses — model learns to be verbose. (2) Reward model was trained by annotators who prefer polite framing — model learns to add excessive hedging and sycophantic phrases ("Great question! Of course..."). (3) Reward model correlates higher scores with confident-sounding answers — model becomes overconfident even when uncertain. Detection: monitor distribution of response length, sentiment tone, hedging phrases, and confidence signals over time. Compare reward model scores vs. human rater scores on a monthly sample — divergence signals hacking.

**Probe 2:** "Specification gaming is a related concept. GPT-4 has been observed to give high-sounding answers to factually incorrect questions (sycophancy). Is this a reward hacking failure, a specification failure, or both? How do you design the reward model to address this?"
> *Looking for:* Both — sycophancy arises from reward models trained on human preferences where human raters are themselves sycophantic (prefer answers that agree with their worldview). This is a specification failure: the reward function correctly captures what human raters prefer, but what they prefer is wrong. Additionally, the LLM generalizes this pattern via reward hacking. Fix: (1) adversarial annotation — show raters pairs where one answer flatters their prior belief and one contradicts it with correct information; measure and penalize sycophantic preferences. (2) Constitutional AI — use a separate model to evaluate if the answer is factually correct, independently of human preference. (3) Process reward models that reward correct reasoning chains, not just final answer quality.

**Expert probe:** "Constitutional AI (RLAIF) and DPO have both been proposed as alternatives to PPO-based RLHF. Describe the mathematical relationship between DPO's loss function and the original PPO+KL objective — and explain the implicit assumption DPO makes that PPO doesn't require."
> *Looking for:* DPO derives a closed-form expression for the optimal policy under the PPO+KL objective, showing that the log ratio of policy to reference policy is a monotone function of the reward. This allows training directly from preference pairs without a separate reward model. The key: DPO assumes the optimal policy is within reach of the reference policy via a small KL divergence — i.e., the preference data fully determines the optimal policy. PPO doesn't make this assumption — it can explore beyond the reference policy distribution, allowing the reward model to guide exploration. DPO breaks down when: the preference data is noisy/contradictory (PPO can use the reward model to smooth this), or when you need online data collection (PPO naturally generates new examples during training; DPO is offline).

---

### Area 2: PPO Instability in Language Models

**Surface:** "Why is PPO used for RLHF and what makes it different from standard RL?"

**Probe 1:** "PPO's clipping mechanism prevents large policy updates. But language models have action spaces of ~100K tokens at each step. How does this interact with PPO's stability guarantees, and what hyperparameters matter most for LLM RLHF?"
> *Looking for:* PPO's clipping bounds the ratio π_new(a)/π_old(a) within [1-ε, 1+ε], typically ε=0.2. With 100K action space, even small changes to a handful of common tokens can cause large changes in generation distribution. Critical hyperparameters: (1) KL coefficient β — controls how far the policy drifts from the SFT baseline; too low causes reward hacking, too high prevents learning. (2) Learning rate — typically 1e-6 to 1e-5 for LLMs, much lower than for Atari games. (3) Rollout batch size — small batches cause high variance in the gradient estimate. (4) Value function initialization — initializing from the SFT model's representation dramatically stabilizes early training.

**Probe 2:** "OpenAI's InstructGPT paper uses a 'reference policy' during PPO with a KL penalty. Llama 2's RLHF paper uses iterative SFT + RLHF rounds. What's the theoretical reason iterative rounds help, and what's the practical risk of each additional round?"
> *Looking for:* Iterative rounds allow the model to explore beyond the initial SFT reference policy distribution and collect new preference data on the updated model's outputs — covering the distribution shift. The reward model trained on SFT-model outputs may be miscalibrated on RLHF-model outputs (which look different). Fresh annotation on the RL-updated model's outputs corrects this. Risk of each round: (1) reward model overfit to the updated distribution, (2) catastrophic forgetting of earlier alignment properties, (3) annotation cost increases. Llama 2 uses ghost attention (GA) in later rounds to maintain multi-turn coherence.

**Expert probe:** "The PPO value function must estimate expected cumulative reward from the current state. For language generation, what is the 'state' precisely, and why is the value function particularly hard to learn for long-horizon conversations?"
> *Looking for:* The state is the full conversation history (all tokens generated so far). Value function must predict expected future reward given this state — which requires estimating how the rest of the conversation will unfold. Hard because: (1) the action space is exponentially large (sequences of tokens), (2) the reward signal is sparse (often only provided at the end of a conversation), (3) the value function must generalize across arbitrary conversation lengths. Mitigation: (a) response-level rewards (reward for each complete response, not the final turn), (b) token-level rewards via reward model applied at each position (computationally expensive), (c) dense reward signals via auxiliary tasks. Strong hire knows that InstructGPT uses a response-level reward signal and mentions the credit assignment problem for multi-turn.

---

### Area 3: Scaling Laws — Predicting Performance Before Training

**Surface:** "What are scaling laws and how do you use them?"

**Probe 1:** "The Chinchilla paper showed that LLMs were undertrained relative to compute. Specifically, the compute-optimal ratio of training tokens to model parameters is approximately 20:1. If you have a budget of 10²³ FLOPs, what does this tell you about the optimal model size and training token count?"
> *Looking for:* Chinchilla: N_opt ≈ √(C/6), T_opt ≈ √(6C) where C is compute in FLOPs. At C=10²³: N_opt ≈ √(10²³/6) ≈ 4×10¹⁰ ≈ 40B parameters, T_opt ≈ 40B × 20 = 800B tokens. The README lists GPT-3 at 175B params with 300B tokens — undertrained by Chinchilla standards. Strong hire can derive this and knows that Llama 2 (7B with 2T tokens) is actually overtrained relative to Chinchilla for a given compute budget — optimized for inference efficiency, not training efficiency.

**Probe 2:** "Scaling laws predict test loss, not downstream benchmark performance. Why is there sometimes a poor correlation between scaling-predicted loss and actual downstream task improvement?"
> *Looking for:* Emergent capabilities — some downstream tasks show near-zero performance below a certain scale threshold, then sharp improvement. This is not predicted by smooth power-law loss curves. The loss predicts average token prediction quality; it doesn't predict whether the model has learned specific skills like multi-step math or code generation that require compositional generalization. Inverse scaling also exists — some tasks degrade with scale due to overcondidence or memorization.

**Expert probe:** "You're about to start a new pre-training run at a scale never attempted before. Scaling laws suggest a certain loss. What additional experiments would you run to validate confidence in the scaling prediction before committing the full compute budget — and what failure modes in the scaling law prediction itself would you worry about?"
> *Looking for:* Ablation approach: run at 10%, 30%, 100% of planned scale and fit the power law from those points — extrapolate and compare to the prediction. Worry about: (1) architecture changes that break the scaling law (e.g., adding MoE, changing context length), (2) data quality changes (if the training data mix changes significantly, the scaling law from prior runs doesn't apply), (3) optimizer changes (different learning rate schedule can shift the effective loss curve). Strong hire also mentions the distinction between IsoFLOP vs. IsoPArameter scaling law analysis and uses both to bound uncertainty.

---

### Area 4: Inference Serving — Continuous Batching and PagedAttention

**Surface:** "How do you serve a large LLM at scale to millions of users?"

**Probe 1:** "The README mentions KV-cache and batching. Explain continuous batching (also called dynamic batching in some contexts). How does it improve GPU utilization compared to static batching, and what are its failure modes?"
> *Looking for:* Static batching: all requests in a batch must finish together — a slow request (long generation) holds up the batch. Continuous batching: as a request completes, a new request is inserted into the batch without waiting for all other requests to finish. This keeps GPU utilization high by always filling the available capacity. Failure modes: (1) batch size variance causes memory fragmentation, (2) requests with very long outputs monopolize slots (need priority queuing or length-based batching), (3) the prefill phase (initial prompt processing) and decode phase (token generation) have different compute characteristics — mixing them in the same batch is suboptimal (Splitwise or ChunkPrefill addresses this).

**Probe 2:** "speculative decoding with a 7B draft model and 70B target model. The draft model proposes k=5 tokens, the target evaluates them. On average, 3 tokens are accepted. What is the theoretical speedup, and when does this break down for a production ChatGPT-style system?"
> *Looking for:* Without speculative decoding: 5 forward passes of the 70B model per 5 tokens. With: 1 forward pass of the 70B model evaluates all 5 tokens in parallel + 1 forward pass of the 7B draft. If k=5 and acceptance rate=3/5, effective generation is 3 tokens per (7B + 70B) forward pass cost. Speedup ≈ k * acceptance_rate / (1 + 7B/70B cost ratio) ≈ 3 / (1 + 0.1) ≈ 2.7×. Breaks down when: (1) acceptance rate drops (output is too creative/diverse), (2) batch size > 1 (speculative decoding is hardest to batch effectively), (3) draft model is not aligned with the target model's distribution (different fine-tuning).

**Expert probe:** "PagedAttention (vLLM) allocates KV-cache in non-contiguous memory pages. Explain the copy-on-write mechanism for beam search decoding in PagedAttention, and why this matters for a production ChatGPT system that may use beam search for some requests."
> *Looking for:* In beam search, multiple candidate beams share the same prefix KV-cache. PagedAttention implements copy-on-write: when a beam diverges (generates a different token), its pages are copied and modified independently rather than duplicating the full prefix. This reduces memory by ~beam_width× for the shared prefix. Why it matters for production: beam search with beam_width=4 would otherwise require 4× the KV-cache memory for each token in the shared prefix. With PagedAttention copy-on-write, the shared prefix cost is paid once. Strong hire also mentions that copy-on-write only helps for the divergent tokens — the shared prefix is never copied.

---

### Area 5: Multi-Turn Safety — Context Window Jailbreaks

**Surface:** "How do you handle adversarial inputs and jailbreaks?"

**Probe 1:** "A user sends a 10-turn conversation where the first 8 turns gradually shift the model's persona to 'DAN (Do Anything Now)' through roleplay. By turn 8, the model is fully in character. Turn 9 asks for harmful instructions. The model complies. Why did the multi-turn setup work, and what architectural or policy interventions prevent this?"
> *Looking for:* The gradual persona shift works by exploiting the model's context following — it was trained to be a good conversationalist, and the 8 turns have established a narrative frame that the harmful request fits within. The model's safety training is evaluated at each individual turn, not at the cumulative context. Interventions: (1) sliding window safety classifier that evaluates the full conversation for persona drift, (2) system prompt that explicitly overrides roleplay contexts for safety instructions, (3) adversarial training with multi-turn jailbreak patterns, (4) periodic "safety anchor" injected into context that reasserts the model's identity. The architectural solution is an always-on safety classifier that monitors the conversation trajectory, not just each individual message.

**Probe 2:** "You want to red-team the model for jailbreaks before production. Given that manual testing is slow and expensive, how would you automate adversarial evaluation at scale?"
> *Looking for:* Automated red-teaming: use a separate LLM (an "attacker" model) trained to generate diverse adversarial prompts that elicit harmful outputs. Measure the attack success rate (ASR) and the diversity of attack vectors. Progressive adversarial training: take failed attacks, add them to the training data, retrain the safety classifier, then generate new attacks against the updated model. Benchmark: HarmBench provides a standardized set of adversarial prompts across categories. Strong hire mentions that human red-teamers and automated methods are complementary — automated methods cover common patterns, humans find novel angles.

**Expert probe:** "The model's safety training uses RLHF to align outputs. But safety alignment is typically trained on individual prompt-response pairs. Explain the generalization gap: why does safety alignment trained on single-turn pairs fail to generalize to multi-turn adversarial contexts, and what training procedure would close this gap?"
> *Looking for:* Generalization gap: the reward model and RLHF training see single (prompt, response) pairs. They don't learn to evaluate the safety of a response conditioned on the full conversation trajectory. The model was never rewarded for maintaining safety under gradual persona shift — only for producing safe single responses. Training procedure to close the gap: (1) multi-turn rollout training — sample full multi-turn conversations during PPO and evaluate safety of the final response given the full context, (2) constitutional AI applied at the conversation level — evaluate whether the full conversation trajectory would lead to harm, (3) Context-Dependent Reward Model (CDRM) that takes the full conversation as input. Strong hire notes the fundamental difficulty: multi-turn training data is exponentially larger space than single-turn, making comprehensive coverage infeasible.

---

## Red Flags to Watch For

- **PPO = reward maximization, no stability discussion.** Cannot explain why PPO's clipping matters for language models specifically.
- **DPO is strictly better than PPO.** Cannot identify where PPO's online data collection advantage is critical.
- **Scaling laws = more data + bigger model = better.** Doesn't know Chinchilla ratios or the emergent capability failure mode.
- **"Add a content filter" as the safety answer.** Doesn't understand multi-turn jailbreaks or why single-turn safety training doesn't generalize.
- **Batch inference means more GPUs.** Cannot explain continuous batching or why static batching wastes GPU utilization.
- **Sycophancy is a hallucination problem.** Doesn't distinguish specification failure from reward hacking.

---

## Hiring Criteria

| Tier | Criteria |
|------|----------|
| **No Hire** | Cannot explain reward hacking with a concrete example. Thinks more KL penalty always means more safety. Doesn't know the Chinchilla compute-optimal ratio. Proposes a single-turn safety filter for multi-turn jailbreaks. |
| **Weak Hire** | Correct on RLHF pipeline basics (SFT → RM → PPO). Can explain KL penalty purpose. Knows scaling laws exist and roughly what they say. Identifies multi-turn jailbreaks as a challenge but cannot specify a solution mechanism. |
| **Hire** | Provides concrete reward hacking examples and detection strategies. Distinguishes DPO from PPO on online vs. offline data collection. Knows Chinchilla ratios. Proposes multi-turn training or safety anchors for jailbreak prevention. Explains continuous batching. |
| **Strong Hire** | Derives the DPO loss from the PPO+KL objective and states the implicit assumption. Provides the speculative decoding speedup calculation. Knows the value function difficulty for multi-turn credit assignment. Proactively connects sycophancy to specification failure and proposes adversarial annotation as a fix. Has a specific opinion on when to use PPO vs. DPO with justified reasoning. |
