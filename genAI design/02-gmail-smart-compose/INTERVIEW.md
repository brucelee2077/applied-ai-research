# Chapter 02 — Gmail Smart Compose
# Staff / Principal Engineer Interview Guide

---

## How to Use This Guide

Every section below contains:
1. **The question** the interviewer asks
2. **What a No Hire / Weak Hire / Hire / Strong Hire candidate actually says** — verbatim-level examples so you can compare what you're hearing in real time

The goal is not to memorize answers. It is to understand the *reasoning depth* required at each tier.

This guide follows the 7-step GenAI system design framework applied to Gmail Smart Compose, then goes deeper on the advanced technical concepts that separate Staff from Principal-level candidates: attention complexity, KV-cache internals, speculative decoding, personalization strategies, and the subtleties of production evaluation.

---

## Part 1: Requirements & Constraints (Step 1)

### Q1. "Design Gmail Smart Compose. Before you touch architecture, what questions do you ask?"

**No Hire:**
> "How many users? What languages? What's the latency requirement? Do we have labeled data?"

Surface-level checklist. None of these questions would change the architecture in a specific way. No signal that the candidate understands why Smart Compose is technically hard.

**Weak Hire:**
> "I'd ask about latency — is this real-time or can it be async? I'd ask about scale — we're talking about Gmail so billions of users. And I'd ask about personalization: do we want the same model for everyone or per-user? Those seem like the big variables."

Identifies the right axes but doesn't connect them to design forks. A candidate who says "latency matters" without saying "and latency under 100ms eliminates large models and forces edge inference or extreme quantization" hasn't yet demonstrated Staff-level thinking.

**Hire:**
> "Three questions would fundamentally reshape my design. First: what's the latency budget? If suggestions must appear on every keystroke under 100ms, that rules out any model over a few hundred million parameters on server-side, and pushes me toward either a very small model or edge deployment. Second: what does 'personalization' mean here — is it writing-style personalization (I say 'best' a lot), recipient-context personalization (formal vs. casual), or actual fine-tuning per user? These have completely different privacy and infrastructure implications. Third: what's the privacy model — can raw email content leave the device? That determines whether I can do centralized training on email data at all, or whether I need federated learning. The latency answer and the privacy answer together almost fully determine my architecture before I've drawn a single box."

**Strong Hire:**
> "I'd start from the user experience backward, because the UX constraint dominates everything. Smart Compose must show a suggestion before the user notices a delay — that's roughly 100ms. At that latency, on server-side infrastructure at Google scale, you're looking at roughly 50ms for network round-trip in the worst case, leaving maybe 50ms for actual inference. That means the model probably tops out around 20-30M parameters with aggressive quantization, which is a radically different regime than GPT-style models. Second question: what's the trigger policy? Showing a suggestion on every single keystroke is expensive and annoying. Do we decouple suggestion generation from suggestion display with a lightweight classifier? Third: privacy. No storing of email text means my training data pipeline needs differential privacy or federated learning, and it also means I cannot do server-side personalization with raw email features — I need to pass only sanitized metadata like 'recipient type' or 'email length bucket' as conditioning signals. Fourth: how do we define success? Acceptance rate sounds obvious, but it conflates the quality of suggestions with the aggressiveness of the triggering service. I'd want to track keystroke savings independently as the real business metric."

---

### Q2. "What are the hardest constraints in this system and how do they interact?"

**No Hire:**
> "Latency is hard. And privacy is hard. You have to be careful about storing emails."

Names constraints without explaining why they are hard or how they trade off.

**Weak Hire:**
> "The hardest constraints are latency under 100ms and the privacy requirement. For latency, you use a small model. For privacy, you use federated learning. They're somewhat independent problems."

Treats constraints as independent — misses the key insight that they interact and create a trilemma.

**Hire:**
> "Latency, privacy, and personalization form a triangle where you can't fully satisfy all three. If you want deep personalization — a model that has learned my specific writing style — you either need to send my emails to the server (privacy violation) or run a large per-user model on-device (latency violation). The practical resolution is to decompose personalization into two parts: a globally-trained model that handles general email patterns and runs fast, plus a lightweight personalization layer (conditioning on sanitized features like recipient type, time of day, user style cluster) that doesn't require raw email access. The latency constraint also interacts with the A/B testing strategy — if you're running multiple model versions simultaneously at 1.5B users, even a small increase in inference cost per request multiplies to a massive infrastructure bill."

**Strong Hire:**
> "The core tension is that the three properties you most want — high suggestion quality, sub-100ms latency, and per-user personalization — each pull in opposite directions. Quality wants a big, powerful model. Latency wants a tiny, fast model. Personalization wants either on-device fine-tuning (which requires device compute and storage) or server-side user representations (which requires privacy-safe feature engineering). Google's actual resolution was to accept a quality ceiling from the small model and invest engineering effort into two things: making the small model as good as possible via distillation from a larger teacher, and making the personalization lightweight by using conditional token inputs like special tokens encoding recipient type and writing formality level. There's a second, subtler constraint: correlated prediction errors. If the model is wrong in a systematic way — say, it always suggests 'best regards' as a closing — users will see the same wrong suggestion millions of times per day, which degrades trust faster than random errors. This argues for some explicit diversity mechanism in the triggering service, not just confidence thresholding."

---

## Part 2: ML Framing (Step 2)

### Q3. "How do you frame Smart Compose as an ML problem? What's the input, output, and loss function?"

**No Hire:**
> "It's a text generation problem. The input is the email so far and the output is the suggested completion. You'd use a language model."

Correct at the surface but no technical depth. Any ML engineer could say this.

**Weak Hire:**
> "It's next-token prediction. The input is the tokenized email context and the output is a probability distribution over the vocabulary for the next token. The loss is cross-entropy. You autoregressively generate multiple tokens to build a phrase."

Technically correct but mechanical. Doesn't address what makes this framing interesting or what choices were made.

**Hire:**
> "The fundamental ML task is causal language modeling: given tokens t_1 through t_n, predict t_{n+1}. Loss is cross-entropy over vocabulary, which is equivalent to maximizing the log probability of the training corpus. But there are a few framing choices that matter. First, the input is not just the body text typed so far — it should include subject line, recipient metadata, and potentially time-of-day signals as conditioning context, separated by special delimiter tokens like [SEP]. Second, during inference you're not doing single-token prediction — you're generating a phrase, which means beam search over a sequence. The quality metric you care about most is not token-level accuracy but phrase-level: does the top beam exactly match what the user would type? That's ExactMatch@N. Third, there's a subtle framing issue: the model is trained on complete emails, but at inference time you're always working with a prefix. You want the model to be especially calibrated at prefix positions, not just average-corpus positions."

**Strong Hire:**
> "The core framing is causal language modeling, but I want to be precise about what we're actually optimizing and where the gap is. During training, the loss is token-level cross-entropy: for each position i in the training email, predict token i given tokens 0..i-1. This trains a very general distribution. But the actual goal is to predict full phrases accurately enough that users press Tab. Those two objectives are not the same. High token-level perplexity can coexist with low acceptance rate if the model's errors are systematically at phrase boundaries, or if the model is well-calibrated in the middle of documents but poorly calibrated on openings, which is where Smart Compose triggers most often. I would add a secondary fine-tuning objective specifically on email prefixes: given the first K tokens of an email, predict the next M tokens, and measure phrase-level exact match. This is a form of task-specific fine-tuning beyond general LM training. Additionally, the input framing matters a lot for personalization — if I encode 'recipient is from same company vs. external' as a conditioning token, the model can learn the formal/informal register without ever seeing raw email addresses, which is privacy-safe."

---

### Q4. "Why decoder-only Transformer and not encoder-decoder or a fine-tuned BERT?"

**No Hire:**
> "Decoder-only is what GPT uses. It's good for text generation. BERT is for classification, not generation."

Pattern-matches to popular models without explaining the architectural reason.

**Weak Hire:**
> "BERT is bidirectional, so it can see future tokens during training. That's not useful for generation since we don't have future tokens at inference time. Encoder-decoder like T5 is for sequence-to-sequence tasks where there's a distinct source and target, like translation. Since Smart Compose is predicting continuation of the same sequence, decoder-only is the natural fit."

Correct reasoning but doesn't go into the efficiency implications or the edge cases.

**Hire:**
> "Decoder-only wins here for three reasons. First, architectural fit: we're generating a continuation of the input sequence, not transforming one sequence into another. The encoder-decoder split creates unnecessary overhead — you'd be encoding the email prefix just to immediately decode it back in the same register. Second, efficiency: a decoder-only model with causal masking processes the input prefix and generates output in a single forward pass (with KV-cache, incremental updates are O(n) not O(n²)). An encoder-decoder does two passes. At 100ms latency budgets, that extra pass matters. Third, KV-cache compatibility: decoder-only naturally caches the key-value pairs for the entire context, and crucially, you only need to compute new K/V for the most recently typed token. Encoder-decoder caches don't work the same way because the encoder must reprocess the full prefix every time the user types a new character. For keystroke-level latency, decoder-only's incremental caching is a decisive advantage."

**Strong Hire:**
> "Decoder-only is the correct choice, but let me steelman the alternatives. You could use an encoder-decoder if you wanted to treat the subject line as the 'source' and the body completion as the 'target' — that's actually a reasonable framing for subject-to-body completion. The problem is that for mid-sentence completion, the 'source' and 'target' are the same sequence at different positions, which makes the encoder-decoder split artificial and wasteful. BERT with a generation head is a non-starter because bidirectional attention — seeing future tokens during pre-training — means BERT cannot generate autoregressively without heavy modification. You'd need something like BERT for masked prediction (fill in the blank) rather than continuation, which doesn't match the Smart Compose UX. Decoder-only with causal masking is the minimal architecture that satisfies the generation constraint. The key architectural decision within decoder-only is size: at 20M parameters with 6 layers, 256 hidden dim, and 4 heads, you're well inside the latency budget. The attention complexity per layer is O(n²·d) where n is sequence length and d is head dimension, so keeping d small (256/4 = 64 per head) at short sequences (typical email prefix is under 200 tokens) keeps the attention computation fast."

---

## Part 3: Data Strategy (Step 3)

### Q5. "What data do you use to train this model, and what are the data pipeline challenges?"

**No Hire:**
> "You'd train on email data. You'd need a lot of emails to teach it email-specific language."

Misses the privacy issue entirely and doesn't address the two-stage training approach.

**Weak Hire:**
> "You'd do two stages: pretraining on general text like Common Crawl or BooksCorpus to learn general language, then fine-tuning on email data to learn email-specific patterns. For privacy, you'd anonymize the emails — replace names and addresses with placeholders."

Gets the two-stage training right and mentions privacy but stops short of the interesting details.

**Hire:**
> "Two-stage training with very different data pipelines. For pretraining: large-scale public text — Common Crawl, BooksCorpus, Wikipedia — to learn general English grammar, vocabulary, and style. For fine-tuning: anonymized email data, where the critical question is what 'anonymized' means in practice. You can't just replace PII with [PERSON] tokens — the model needs to learn that 'Dear [PERSON]' is a valid email opening without inferring that [PERSON] is a specific individual. The fine-tuning data pipeline needs at minimum: PII scrubbing of names, email addresses, and phone numbers; potentially differential privacy noise injection during gradient computation; and possibly k-anonymity checks to ensure that no individual's writing style is too recognizable. The tokenizer is trained on both data sources combined, so the BPE vocabulary covers both general English and email-specific terms like 're:' and 'fwd:' as unified tokens. One underappreciated issue is data freshness — language evolves, and a model trained only on 2018 emails will make outdated suggestions. You need a continuous data pipeline for ongoing fine-tuning."

**Strong Hire:**
> "The data strategy has to answer three questions: what, how much, and how private. For 'what': pretraining data should be large (hundreds of billions of tokens) and diverse. Fine-tuning data should be domain-specific — not just any emails but emails that look like the completion task: prefix-completion pairs where the user actually typed out the full sentence. You can mine these from complete sent emails by treating every prefix as an input and the next K tokens as the label. For 'how much': fine-tuning on email data requires surprisingly little data to shift the model's prior toward email register — tens of millions of examples is probably sufficient given the strong pretrained prior. The real question is data balance: too little email fine-tuning and the model suggests generic text; too much and it overfits to specific email idioms. For 'how private': anonymization via PII scrubbing is a floor, not a ceiling. The real concern is membership inference — can an adversary, given a model, determine that a specific email was in the training data? The mitigation is differential privacy during fine-tuning, typically using DP-SGD with a privacy budget epsilon around 1-8. The tradeoff is that DP-SGD requires large batch sizes to maintain utility, which means the fine-tuning infrastructure needs to be able to aggregate gradients across very large batches. Federated learning addresses the hardest constraint — you never collect raw emails at all — but FL's communication overhead and convergence properties are more complex to manage at 1.5B device scale."

---

## Part 4: Model Development (Step 4)

### Q6. "Walk me through the inference path: user types 'Thanks for the', how does the model generate a suggestion?"

**No Hire:**
> "The model takes the text as input, runs it through the Transformer, and outputs the next word. You repeat this until you have a full suggestion."

Correct but shows no understanding of tokenization, beam search, or the KV-cache.

**Weak Hire:**
> "First you tokenize the input with BPE — 'Thanks for the' becomes maybe 4 tokens. Then the Transformer does a forward pass and outputs a distribution over the 30K vocabulary. You pick the top token, append it, run the model again, and repeat. In practice you use beam search to explore multiple paths and pick the most likely overall sequence."

Gets the mechanics right but misses the KV-cache, the triggering service, and the latency engineering.

**Hire:**
> "Step one: the triggering service has already decided to fire — it evaluated the context and determined confidence is above threshold. Step two: the current context, including subject line, recipient type token, body text, and the newly typed character, is tokenized with the BPE tokenizer into a sequence of roughly 50-200 token IDs. Step three: the Transformer does a forward pass. Crucially, the KV-cache means this is incremental — we already computed key-value pairs for all previous tokens in the last keystroke's pass. We only need to compute K/V for the one new token, then use all cached K/V pairs in attention. This reduces the forward pass from O(n·d) to O(1·d) in attention computations per new character. Step four: beam search with beam width 4-8. At each step, we keep the top-4 partial sequences by log-probability, expand each by one token, re-score, prune back to 4. We stop when all beams hit [EOS] or a length limit. Step five: the post-processing service checks the top beam for safety issues and grammatical coherence, then returns it to the UI. The whole pipeline target is under 100ms wall-clock."

**Strong Hire:**
> "Let me trace this with the latency budget in mind. The user has typed 'Thanks for the' — let's say this is their 15th character typed in the body. The triggering service sees the user just paused fractionally after a content word, confidence score is 0.73, above the 0.65 threshold, so it fires a generation request. The context passed to the phrase generator is: [RECIPIENT_EXTERNAL] [FORMALITY_PROFESSIONAL] [TIME_MORNING] [SEP] Thanks for the meeting yesterday [SEP] Thanks for the. Let's say that tokenizes to 28 tokens. The KV-cache already has computed K/V pairs for the first 27 tokens from the previous keystroke. We compute Q/K/V only for token 28 (the last newly typed token), then run full attention using all 28 K/V pairs. This attention operation is O(28·d) not O(28²·d), which is the whole point of the cache. Beam search then runs: we expand 4 beams, at each step doing O(28+k) attention where k is the position of the currently generated token, for at most 8 generation steps. So the total computational work is roughly equivalent to one full forward pass on a 36-token sequence — very fast for a 20M parameter model. The INT8 quantized weights reduce memory bandwidth pressure by 4x compared to FP32. With a modern accelerator chip, this fits easily in 50ms, leaving 50ms for network and UI rendering."

---

## Part 5: Evaluation (Step 5)

### Q7. "What metrics do you use to evaluate Smart Compose and why is each one necessary?"

**No Hire:**
> "Accuracy — does the model predict the right words? And latency to make sure it's fast enough."

'Accuracy' is undefined and meaningless here. Shows no understanding of the gap between offline and online metrics.

**Weak Hire:**
> "Perplexity for the language model quality. Acceptance rate for how often users take the suggestions. And latency measured at the p99 level. Acceptance rate is the main business metric."

Names the right metrics but doesn't explain what each one tests or why acceptance rate alone is insufficient.

**Hire:**
> "You need metrics at three layers. Offline model quality: perplexity tells you how well the model fits the email distribution, and ExactMatch@N tells you how often the model's top-N suggestions exactly match what the user typed. Perplexity is fast to compute but weakly correlated with user satisfaction — a model can have low perplexity by hedging on common phrases while generating boring, obvious suggestions. ExactMatch@N is better but still offline. Online behavioral metrics: acceptance rate (fraction of shown suggestions the user presses Tab on) and keystroke savings ratio (fraction of characters that came from accepted suggestions). These are the real business metrics. Latency: p50 and p99 of end-to-end suggestion display time, measured from last keystroke to suggestion appearing. The reason you need all three layers is that they catch different failure modes: perplexity catches general model collapse, ExactMatch catches the model drifting from email distribution, acceptance rate catches triggering policy problems, and latency catches infrastructure regressions."

**Strong Hire:**
> "The metric stack has a hierarchy of dependency that's important to understand. Perplexity is your canary — if it spikes, something broke in training, data pipeline, or tokenization. But you cannot optimize directly for perplexity and expect acceptance rate to improve; they're only weakly correlated. ExactMatch@N is better because it's task-specific: on a held-out set of email continuations, does the model's top beam match the actual continuation? This is closer to what the user experiences but still offline. The gap from ExactMatch to acceptance rate is the triggering service and the display context — even a model with great ExactMatch will have low acceptance rate if the triggering service fires at wrong moments, like mid-word or when the user is clearly on a roll. Acceptance rate and keystroke savings are the right online metrics, but they have a subtle confound: they measure the joint performance of the model and the triggering policy. If you make the triggering more conservative (only show when very confident), acceptance rate goes up but keystroke savings go down. You need to track both. There's a deeper problem with acceptance rate as a metric: it's subject to selection bias. Users who accept suggestions are different from users who don't. A model that learns to serve power users who always accept will have high acceptance rate but low actual utility to average users. I'd add a holdout evaluation: for a random sample of users, disable suggestions entirely and measure keystroke count versus the treatment group. That's the counterfactual baseline for true keystroke savings measurement."

---

### Q8. "Acceptance rate seems like a good metric but what are its failure modes?"

**No Hire:**
> "If the model is making bad suggestions, acceptance rate will go down. That's a problem."

Describes the obvious case, not the subtle measurement issues.

**Weak Hire:**
> "Acceptance rate conflates model quality with triggering frequency. A very conservative triggering service that only shows suggestions when extremely confident will have high acceptance rate but won't actually help users much."

Gets the triggering coupling right but doesn't go further.

**Hire:**
> "Three failure modes. First, triggering-suggestion coupling: acceptance rate is acceptance rate conditional on a suggestion being shown. If you show fewer, higher-confidence suggestions, acceptance rate improves but keystroke savings (the real metric) might not. You need to decouple them by also tracking suggestion coverage — what fraction of typing sessions had at least one suggestion shown. Second, selection bias in trigger timing: the triggering service fires when it's confident, which means suggestions are shown when the text is most predictable. Acceptance rate on predictable text is naturally higher. You're measuring 'how well do we do when we're confident' not 'how well do we do on average.' Third, novelty effects: users explore acceptance when they first see the feature, inflating early acceptance rates. You need cohort analysis: what's the acceptance rate for users who have had the feature for 30+ days?"

**Strong Hire:**
> "Acceptance rate has four distinct failure modes and the worst one is rarely discussed. First, the triggering-suggestion coupling: acceptance rate measures quality conditional on showing, but the decision to show is endogenous to model confidence, so you're measuring quality on the easy cases. Second, selection bias: users who frequently accept suggestions and users who never do are very different populations; your acceptance rate is dominated by the heavy acceptors. Third, novelty and fatigue: early users explore suggestions enthusiastically; long-term users develop habits. You need to separate these cohorts in any reported metric. Fourth, and most critically, Goodhart's Law: once acceptance rate becomes a KPI, teams optimize for it in ways that degrade real utility. You can inflate acceptance rate by only showing suggestions to users who have historically accepted, or by showing shorter suggestions that are harder to reject because they complete so little. The fix is to anchor on counterfactual keystroke savings measured with a holdout group, not just acceptance rate among treatment users. This requires an A/B test where the control group sees zero suggestions — which is expensive and requires careful consent design. For a full roll-out at 1.5B users, you'd do this holdout at the 0.1% level, which still gives you millions of users in control."

---

## Part 6: System Design (Step 6)

### Q9. "Draw the system architecture. What are the components and how do they interact?"

**No Hire:**
> "There's a model that takes input and outputs a suggestion. You'd put it behind an API."

No component decomposition. No understanding of the triggering problem or post-processing.

**Weak Hire:**
> "Three main parts: the triggering service that decides when to show a suggestion, the Transformer model that generates the text, and a post-processing step that filters unsafe content. The client sends the email text to the server, the server runs the model, and sends back the suggestion."

Gets the three components right but doesn't discuss data flows, latency implications of server vs. edge, or how keystroke-by-keystroke incremental inference works.

**Hire:**
> "I'd draw four components plus the client. On the client: a lightweight keystroke listener that batches character events and sends context snapshots to the triggering service. The triggering service is a small fast classifier — maybe a lightweight LSTM or logistic regression over hand-crafted features — that evaluates whether to fire a generation request based on cursor position, recent typing speed, model confidence from the previous suggestion, and whether the user is at a natural completion point. When it fires, it sends the full context to the phrase generator. The phrase generator is the Transformer model with beam search — running either on-device or on a low-latency edge server. Its output goes to the post-processing service which runs a safety classifier and a grammar filter. The result is returned to the client within 100ms total. The KV-cache lives in the phrase generator and is maintained across keystrokes as long as the user hasn't deleted text. Deletion invalidates the cache for all tokens after the deletion point, requiring a partial recompute."

**Strong Hire:**
> "The architecture has to be designed around the latency budget and the cache invalidation problem. Starting from the user's device: every character typed triggers a lightweight JS handler that (a) updates the displayed suggestion instantly if the new character is consistent with the previous suggestion (no model call needed), or (b) sends an async request to the triggering service if the suggestion was invalidated. The triggering service is deliberately separate from the generator — it runs on a fleet of very cheap stateless machines, evaluates whether to fire, and can be A/B tested independently. When it fires, it sends the full context — including the KV-cache metadata, specifically the sequence hash of the cached prefix — to the phrase generator fleet. The phrase generator is stateful: it maintains a KV-cache indexed by user session ID and prefix hash. If the prefix matches a cached entry, it does incremental inference (compute K/V only for new tokens). If the user deleted text or pasted content, it detects the mismatch, invalidates the stale portion of the cache, and recomputes. This is the expensive path and you want to minimize it. Post-processing runs in parallel with the tail end of beam search — as beams terminate early (hit [EOS]), you can start safety-checking them while remaining beams continue. The end-to-end SLA has to account for p99, not p50 — a user whose device happens to be on a slow network gets a degraded experience. For those users, you fall back to edge inference: a smaller INT8-quantized model runs on-device with no network call. The triggering service knows the user's connectivity state and routes accordingly."

---

### Q10. "The user starts deleting characters. How does the KV-cache behave and what do you do?"

**No Hire:**
> "You'd clear the cache and recompute from scratch."

Technically safe but shows no understanding of partial invalidation or the cost implications.

**Weak Hire:**
> "The KV-cache is only valid for the tokens that are still present in the context. If the user deletes characters, the cache entries for the deleted tokens are no longer valid. You'd need to recompute from the deletion point forward."

Gets the concept right but doesn't address the optimization: you can keep the cache entries before the deletion point.

**Hire:**
> "The KV-cache is a tensor of shape [layers × heads × sequence_length × head_dim] for both keys and values. When the user deletes characters, the tokens corresponding to those characters are no longer in the context. The cache entries for positions before the deletion are still valid — the earlier context hasn't changed. The entries for the deleted positions and everything after them are now invalid. So on the next inference call, you set the effective sequence length to the length after deletion, reuse the K/V entries for positions 0 through (deletion_point - 1), and recompute K/V for any new character typed after the deletion. This means deletion is more expensive than forward typing but cheaper than a full cache rebuild. The implementation needs the cache to be addressable by sequence position, not just FIFO, which is a non-trivial memory management problem. In practice, production systems use a paged memory layout for the cache — similar to virtual memory paging — so that cache entries can be freed and reused without copying."

**Strong Hire:**
> "Deletion is the adversarial case for KV-cache systems and handling it correctly is one of the harder engineering problems in Smart Compose. The cache has three validity states: entries before the deletion point are fully valid, entries at the deletion point are invalidated, and entries after the deletion point are invalidated. The right behavior is: truncate the effective cache to position (deletion_point - 1), then on the next generation request, compute K/V for only the new tokens typed after deletion and use the prefix cache as-is. The complication is that position encodings are baked into the K/V computation — a key computed at position 15 encodes position information. If the user deleted and retyped text, and the new text is at the same position but different content, the positional encoding is correct but the semantic content is wrong, so the old cache entry is invalid. Conversely, if the user deleted text at position 15 and nothing after position 15 changed — for example they deleted a word in the middle of a sentence — then positions 0-14 are fine but 15+ are all stale. The industrial solution is something like PagedAttention (from vLLM): the cache is chunked into pages of fixed token count (say 16 tokens per page), and when a prefix prefix matches a stored page, you reuse it; when it doesn't, you recompute that page only. This makes the cache management efficient even with frequent edits, and it also makes it easier to share cache pages across concurrent users who have identical email prefixes, which is surprisingly common for things like 'Hey team,' or 'Hi [name], hope you're well.'"

---

## Part 7: Deployment & Monitoring (Step 7)

### Q11. "How do you deploy this system at Google scale? What changes from a prototype?"

**No Hire:**
> "You'd put it on Google Cloud and scale up the servers as needed. Use load balancing."

No ML-specific thinking about model serving, versioning, or the difference between stateless and stateful inference.

**Weak Hire:**
> "You'd containerize the model with something like TensorFlow Serving or TorchServe. You'd run it on GPUs and scale horizontally. You'd do canary deployments to catch regressions before full rollout. You'd monitor acceptance rate and latency in production."

Covers the basics but doesn't engage with the stateful nature of KV-cache serving, the edge vs. server routing decision, or the model update problem.

**Hire:**
> "Several things change from prototype to production. Serving infrastructure: the model server must maintain KV-cache state per user session, which means it's not fully stateless. You need sticky routing — requests from the same user session should go to the same server instance, or you need a distributed cache with fast lookup. Model updates: when you deploy a new model version, the old KV-caches are now invalid (different model weights produce different K/V representations). You need a versioned cache system or a graceful rollover that drains old caches before switching. Quantization: the prototype runs in FP32, but production uses INT8 or even INT4 for speed and memory. INT8 quantization introduces error but typically less than 1% perplexity increase for models this size. Fallback logic: if the server-side generator is over latency SLA, the client should fall back to a local cached suggestion or no suggestion — a bad suggestion is better than a late one, but no suggestion is better than a wrong late one. Monitoring needs to be per-country and per-device-type because latency distributions vary enormously."

**Strong Hire:**
> "Production deployment introduces problems that don't exist in prototypes. The most important is cache lifecycle management. KV-cache entries need to be evicted based on LRU and session expiry, but also immediately invalidated on model update. I'd version the cache keys with the model checkpoint hash so that a cache miss on version change automatically triggers a fresh computation. At 1.5B users, even a 1% cache hit rate on common prefixes (like 'Dear [Name], I hope this email finds you') is a significant latency win, and I'd aggressively exploit this with a prefix tree (trie) over token sequences to identify shareable cache blocks — this is basically what PagedAttention does for multi-user serving. Model update strategy: I'd use a blue-green deployment with traffic shifting, but specifically I'd measure KV-cache miss rate during the rollout window, since a spike in cache misses after a model update means extra GPU compute to rebuild caches, which could cause a latency spike if the timing is bad. I'd also think about the edge vs. server routing decision more carefully. Edge inference with a small quantized model (say 5M parameters, INT4) handles the tail latency problem and works offline, but it uses device battery and requires OTA model updates. Server inference gives you a better model but requires network. The right architecture is hybrid: use edge inference as the fallback, server inference as the primary path when network conditions are good, and let the triggering service make this routing decision dynamically based on current network quality."

---

## Part 8: Deep Technical Probes

### Q12. "Self-attention is O(n²) in sequence length. Why does that matter for Smart Compose at keystroke latency?"

**No Hire:**
> "O(n²) means it gets slow for very long sequences. That's why you need a powerful GPU."

Technically true but doesn't connect the complexity to the specific context of keystroke-level inference.

**Weak Hire:**
> "Attention is O(n²) because every token attends to every other token. So for a sequence of length 200, you have 40,000 attention pairs per head. If you have 4 heads and 6 layers, that's about a million operations. It's manageable for short sequences but gets expensive quickly as emails get longer."

Gets the mechanics right but doesn't address the mitigation (KV-cache makes this O(n) for incremental decoding) or the implication for maximum context length.

**Hire:**
> "O(n²) attention complexity refers to the full forward pass through the attention matrix, which is n² multiplications where n is sequence length. For Smart Compose, the context is an email prefix — typically 50-300 tokens. At n=300 and 4 heads, the attention matrix is 4 × 300 × 300 = 360,000 elements. That's small. The issue isn't batch-time complexity for this sequence length; it's incremental complexity. Naively, every time the user types a character, you'd rerun the full O(n²) attention. KV-cache fixes this: you cache the key and value projections for all previous tokens and only compute the query for the new token, then run a cross-attention of shape [1 × n], which is O(n) not O(n². This is why KV-cache is essential for keystroke-level latency. The n² cost does matter at inference for batch serving of many users simultaneously — each user session has a different cached K/V state, and the GPU memory for these caches is O(n × layers × heads × head_dim) per user. At 1.5B users with even a small fraction active simultaneously, cache memory management becomes a serious problem."

**Strong Hire:**
> "The O(n²) complexity of attention has three distinct implications that matter at different scales for Smart Compose. At the per-keystroke level: the full attention matrix for a 200-token prefix is small enough (200² = 40K) that the bottleneck is actually memory bandwidth and kernel launch overhead, not FLOPs. KV-cache transforms this from O(n²) to O(n) per incremental step by caching K and V and only computing attention with the new query, so the effective complexity per keystroke is O(n·d) where n is context length and d is head dimension. At the sequence length limit: if you wanted to support very long emails (thousands of tokens), the cache memory for K/V grows linearly with sequence length per user, but the attention computation for each new token still grows linearly with context length. For very long contexts, you'd want sparse attention patterns or sliding window attention (like Longformer) to cap the effective context. At the multi-user serving level: when running a batch of N users simultaneously on a GPU, the attention computation is batched, and the irregular KV-cache sizes (different users have typed different amounts) cause memory fragmentation. PagedAttention from vLLM solves this by chunking cache into fixed-size blocks, achieving near-perfect GPU memory utilization. At the architecture choice level: the choice of 4 attention heads and 256 hidden dim (64 per head) keeps d small, which minimizes the memory bandwidth for attention. Multi-Query Attention — sharing a single K and V projection across all heads — would further reduce cache memory by 4x at the cost of some quality, and it's plausible that production Smart Compose uses something like this."

---

### Q13. "What is speculative decoding and when would you apply it to Smart Compose?"

**No Hire:**
> "I haven't heard of speculative decoding. Is it related to speculative execution in CPUs?"

No knowledge of the technique.

**Weak Hire:**
> "Speculative decoding uses a small draft model to generate a few tokens quickly, then the large target model checks them all at once. If the draft tokens are correct, you've effectively generated multiple tokens in one target model forward pass. It's faster than autoregressive decoding with the large model alone."

Gets the high-level idea right but doesn't know the mathematical guarantee or when it's applicable to Smart Compose specifically.

**Hire:**
> "Speculative decoding works as follows: a small draft model (say 1M parameters) autoregressively proposes k tokens (typically 4-8) in sequence. Then the large target model (say 20M parameters for Smart Compose) runs a single forward pass over all k proposed tokens simultaneously, producing its own distribution at each position. You then use rejection sampling: for each proposed token in order, you compare the draft model's probability to the target model's probability. If the draft probability is lower than or equal to the target's probability for that token, you accept it. If not, you reject it and sample from a corrected distribution. The key property is that this process preserves the exact output distribution of the target model — speculative decoding is mathematically equivalent to sampling from the target model, not the draft model. For Smart Compose specifically: the model is already small (~20M params), so the gains are smaller than when applying speculative decoding to a large LLM. But if you used distillation to create an even smaller draft model (say 2M params) and kept the 20M model as the target, you could potentially generate 4-5 tokens per target model forward pass on high-confidence continuations, roughly halving latency."

**Strong Hire:**
> "Speculative decoding is elegant because it gives you speed without sacrificing quality — unlike beam search width reduction or greedy decoding, it produces samples from the exact target distribution. The mechanism: draft model generates k tokens greedily or with sampling. Target model scores all k+1 positions (context plus all k draft tokens) in a single batched forward pass. Rejection sampling proceeds left-to-right: at position i, accept the draft token with probability min(1, p_target(t_i | context) / p_draft(t_i | context)). If you reject at position i, sample from a renormalized target distribution truncated to exclude the draft token, and discard draft tokens at positions i+1..k. The expected number of tokens generated per target model call is roughly k × (acceptance rate), which can be 3-4x in favorable cases. For Smart Compose, the applicability depends on the regime. If you're running the 20M model on-device (INT4), speculative decoding with a 1M draft model is very attractive because device compute is scarce. If you're running server-side with dedicated accelerators, the bottleneck is often memory bandwidth, not FLOPs, and speculative decoding's batched evaluation may not help much because the batch of k tokens is still small. There's also a nuance for beam search: speculative decoding is designed for single-sequence sampling, not beam search. To apply it to beam search, you'd need to run the draft model for each beam independently, which multiplies the draft model cost by beam width. Depending on relative model sizes, this may not be beneficial. A simpler alternative for beam search is to use a cheaper scoring model during beam expansion and only run the full target model on the final top-beam candidates."

---

### Q14. "How would you do per-user personalization without fine-tuning per user?"

**No Hire:**
> "You could fine-tune a separate model for each user with their email history. That's the most direct approach."

Ignores the obvious problem: you can't fine-tune 1.5B separate models.

**Weak Hire:**
> "Fine-tuning per user doesn't scale. Instead, you could use federated learning to personalize on-device. The model updates are computed locally and only aggregate updates are sent to the server, not raw email content."

Gets FL right but doesn't explain the alternatives or the technical details of how conditioning works.

**Hire:**
> "Three approaches in increasing engineering complexity. First, context conditioning: pass sanitized user features as special tokens at the front of the input — things like [FORMALITY_HIGH], [RECIPIENT_EXTERNAL], [WRITING_STYLE_VERBOSE]. The global model learns to condition on these tokens during fine-tuning. This is privacy-safe, fast, requires no per-user model, and captures coarse personalization. Second, prompt-based personalization: retrieve the user's last 3-5 accepted suggestions as few-shot examples and prepend them to the context. The model acts like a few-shot learner without any weight updates. This requires only a small user-side storage of accepted suggestion history. Third, federated learning: train a global model using FL, allowing gradient updates computed on-device using local email data without those emails ever leaving the device. This is the most expensive to implement but gives the most genuine style personalization. The tradeoff is FL convergence is slower and noisier than centralized training, and it requires a large fraction of users to be online simultaneously for aggregation."

**Strong Hire:**
> "I'd think about personalization along two axes: latency impact and data sensitivity. Coarse personalization — recipient type, time of day, email thread depth — can be done with conditioning tokens at zero latency cost. The model is trained with these tokens in the fine-tuning phase and learns to shift its distribution accordingly. This handles the most impactful personalization: 'Hey dude' vs. 'Dear Professor Smith' is recipient-driven, not style-driven. For style personalization — the user's specific phrase habits — you have three options: prompt-based (prepend recent accepted completions as few-shot examples, completely privacy-safe, adds maybe 50 tokens to context), adapter-based (train a small LoRA adapter per user using federated learning, the adapter lives on-device and adds maybe 0.1M extra parameters, the base model stays global), or full FL fine-tuning (impractical for 1.5B users due to communication and compute costs). The LoRA-per-user via FL approach is technically sophisticated: during federated training, each device trains a low-rank decomposition ΔW = AB (rank 4-8, adding maybe 100K parameters) on the user's own email data. The device keeps its own A and B matrices. The server aggregates A and B updates across users to improve the global base model but does not distribute user-specific adapters. At inference time, the on-device adapter is added to the server model weights via ΔW before the forward pass. This gives genuine personalization while keeping style information entirely on-device. The practical limit is that LoRA adapters need to be retrained periodically as user style evolves, which requires on-device compute scheduling similar to how iOS handles background app refresh."

---

### Q15. "How do you run an A/B test when you don't have ground truth labels for quality?"

**No Hire:**
> "You look at click-through rate or some proxy metric to see which version is better."

Doesn't understand the fundamental measurement problem in evaluating a suggestion system.

**Weak Hire:**
> "For A/B testing Smart Compose, you compare acceptance rate between the control model (version A) and the treatment model (version B). You need statistical significance to declare a winner. Acceptance rate is the main signal since you don't have human labels for every suggestion."

Acceptance rate is a reasonable proxy but this answer doesn't address the selection bias or the counterfactual problem.

**Hire:**
> "The ground truth problem is real: you only observe whether a user accepted a suggestion, not whether the suggestion was good. A low acceptance rate could mean bad suggestions or aggressive triggering. Here's how I'd approach it. First, split users randomly into control and treatment. Both groups see suggestions, but from different model versions. Compare acceptance rate and keystroke savings. This tells you whether B is better than A but not whether either is better than no suggestions. Second, add a holdout group: a small fraction of users (say 0.5%) with suggestions disabled entirely. Comparing treatment to holdout gives you the absolute effect of Smart Compose. Third, for model quality specifically (not just triggering quality), use counterfactual evaluation offline: take a random sample of emails from production, mask out the second half of each sentence, run both models, and compare their predictions to the actual second halves. This is a held-out ExactMatch evaluation that doesn't require showing suggestions to users at all."

**Strong Hire:**
> "A/B testing a generative suggestion system has three distinct measurement problems. First, the treatment effect on shown suggestions is confounded by triggering: if model B generates higher-confidence suggestions on average, the triggering service will show more B suggestions, changing both the numerator and denominator of acceptance rate in correlated ways. You need to fix the triggering policy during the A/B test and only vary the generation model. Second, the standard A/B estimator (average treatment effect) is biased when users are not independent — email conversations involve multiple users, and if one user's model version changes how they write emails, it may affect the recipients' behavior in ways that are not captured. This is network effect contamination and is hard to solve without cluster-level randomization. Third, the absence of ground truth means you need to choose between revealed-preference metrics (acceptance rate, a noisy signal subject to Goodhart's Law), counterfactual offline metrics (ExactMatch on held-out emails, not reflecting real triggering conditions), and human evaluation (expensive, doesn't scale to model iteration speed). My preferred stack: use counterfactual offline metrics for rapid iteration during development, A/B test on acceptance rate and keystroke savings for launch decisions, and run a small permanent holdout (0.1% of users) for measuring absolute product value against the null (no suggestions). For the counterfactual evaluation, I'd specifically evaluate on email prefixes of different lengths (0-25 tokens, 25-100 tokens, 100+ tokens) because the model's quality profile varies significantly by context length and you want to understand where each model version wins and loses."

---

## Part 9: Architecture Deep Dives

### Q16. "What is Multi-Query Attention and why would you consider it for Smart Compose?"

**No Hire:**
> "I don't know Multi-Query Attention. Is it just running multiple queries?"

**Weak Hire:**
> "Multi-Query Attention reduces the number of K and V heads. Instead of having 4 separate K and V projections for 4 heads, you share one K and V across all heads. This saves memory. The Q projections are still separate per head."

Gets the definition right but doesn't explain the memory bandwidth reasoning or the quality tradeoff.

**Hire:**
> "Standard Multi-Head Attention (MHA) has h separate Q, K, V projection matrices, each of shape [d_model × d_head]. The KV-cache stores h × sequence_length × d_head entries for K and h for V. Multi-Query Attention (MQA) uses a single K and V projection shared across all heads, reducing the KV-cache size by a factor of h (number of heads). For Smart Compose with 4 heads, MQA cuts cache memory by 4x. Grouped-Query Attention (GQA) is a middle ground: you group the 4 heads into 2 groups, each group sharing one K and V, giving 2x cache reduction with less quality loss than MQA. For Smart Compose on-device, where memory is the bottleneck, GQA or MQA is very attractive. The quality impact is real but moderate — MHA is slightly better for in-context learning and long-range dependencies, but for a 6-layer model doing short-phrase completion, the quality gap is small relative to the memory savings."

**Strong Hire:**
> "MQA and GQA are motivated by a specific observation about inference efficiency: during autoregressive decoding, the bottleneck is memory bandwidth, not FLOPs. At each decoding step, you load the full KV-cache from memory (which is O(layers × heads × context_len × head_dim)) but only do a small amount of compute (one attention operation per token). The ratio of bytes loaded to FLOPs is called arithmetic intensity, and it's very low for autoregressive attention, making it memory-bandwidth-bound. MQA reduces the KV-cache size by h×, which directly reduces memory bandwidth consumption without changing the compute. GQA (used in Llama-2-70B and Mistral) offers a tunable tradeoff — with g groups, cache size reduction is h/g×. For Smart Compose specifically: with h=4 heads and d_head=64, the KV-cache per token is 2 (K and V) × 4 (heads) × 64 (dim) × 6 (layers) × 2 (bytes for FP16) = 6144 bytes per token. For a 200-token context and 1000 concurrent users per GPU, that's about 1.2GB just for KV-cache. With MQA, it drops to 300MB. This matters for GPU memory capacity and enables larger batch sizes or longer contexts. On-device, where you might have only 2-4GB RAM for the entire ML workload, MQA/GQA may be the difference between being able to maintain a session cache at all versus evicting and recomputing constantly."

---

## Part 10: Red Flags

The following table summarizes signal patterns that indicate a candidate is below bar, regardless of which specific questions were asked.

| Red Flag | What It Sounds Like | Why It Matters |
|---|---|---|
| Proposes GPT-4-scale model for Smart Compose | "I'd use a large language model with billions of parameters for best quality." | Shows no latency awareness. A Staff engineer knows that model choice is constrained by SLA, not just quality. |
| Cannot explain KV-cache | "The model just caches the output somehow." | KV-cache is the central inference optimization for autoregressive models. Not knowing it means no production LLM experience. |
| Treats acceptance rate as a simple metric | "Acceptance rate is our main KPI, higher is better." | Missing the triggering-suggestion coupling and Goodhart's Law failure modes. |
| Ignores privacy when discussing training data | "We'd just train on all the emails in Gmail." | Privacy is a hard constraint in this system. Missing it means not thinking like a systems engineer at Google scale. |
| Cannot distinguish offline vs. online metrics | "Perplexity is the main metric." | Offline metrics are necessary but not sufficient. Not understanding the gap between perplexity and acceptance rate indicates no production ML experience. |
| Says "just use federated learning" without caveats | "Federated learning solves the privacy problem." | FL has real tradeoffs: slower convergence, communication overhead, heterogeneous device compute. Treating it as a silver bullet is a red flag. |
| Cannot explain why decoder-only over encoder-decoder | "Decoder-only is what everyone uses now." | Should explain causal generation, KV-cache compatibility, and efficiency of not having a separate encoder. |
| No awareness of the triggering service | Jumps from "user types" to "model generates" | Misses a fundamental design component. The decision of when to suggest is as important as what to suggest. |
| Speculative decoding claim without knowing the math | "It's definitely faster, so we should use it." | Speculative decoding's value depends on draft/target model size ratio and acceptance rate. Claiming benefits without understanding the conditions shows superficial knowledge. |
| Cannot reason about INT8 quantization tradeoffs | "Quantization just makes the model worse." | Should know that INT8 quantization on 20M-param models typically causes less than 1% perplexity increase while halving memory and improving throughput. |

---

## Hiring Decision Summary

| Score | Profile | Signal |
|---|---|---|
| **Strong Hire** | Reasons from constraints to architecture. Explains KV-cache memory layout, MQA/GQA tradeoffs, speculative decoding's mathematical guarantee, federated LoRA for personalization, and counterfactual holdout design for A/B testing. Identifies failure modes of acceptance rate and proposes keystroke savings with holdout as the true metric. | Asks questions that change the architecture (latency → model size, privacy → FL, trigger policy → decoupled triggering service). Uses precise numbers (6 layers, 256 dim, 4 heads, INT8, 20M params). |
| **Hire** | Covers all 7 steps correctly. Explains KV-cache incremental inference, beam search vs. sampling tradeoff, two-stage training, ExactMatch vs. acceptance rate difference, and triggering-suggestion coupling. Can reason about deletion handling in the cache. | Draws the three-component architecture without prompting. Identifies that acceptance rate conflates triggering and model quality. Knows why decoder-only beats encoder-decoder for this task. |
| **Weak Hire** | Covers 5-6 of 7 steps correctly. Knows KV-cache conceptually, knows two-stage training, knows beam search. Misses some failure modes (like deletion invalidation or the Goodhart's Law failure of acceptance rate). Has not worked with production LLM serving. | Gets the structure right but explains by analogy or hand-waving, not from first principles. Needs prompting to reach depth. |
| **No Hire** | Cannot explain why the model must be small. Treats acceptance rate as a simple metric. Does not know KV-cache. Proposes server-side training on raw emails. Cannot explain decoder-only vs. encoder-decoder tradeoff. | Answers feel like they're recalling a blog post rather than reasoning from fundamentals. No awareness of the latency constraint as the primary architectural driver. |

---

*Guide covers: Requirements framing, ML task framing, data pipeline and privacy, architecture (decoder-only Transformer, 6L/256H/4A/~20M), inference (beam search, KV-cache, incremental decoding, deletion handling), evaluation (perplexity, ExactMatch@N, acceptance rate, keystroke savings, counterfactual holdout), system design (triggering service, phrase generator, post-processing, edge fallback, KV-cache eviction), deployment (blue-green, cache versioning, PagedAttention), and deep probes on O(n²) attention, KV-cache memory, speculative decoding, MQA/GQA, per-user LoRA via FL, and A/B testing without ground truth.*
