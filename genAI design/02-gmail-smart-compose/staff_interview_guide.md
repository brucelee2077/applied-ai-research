# Gmail Smart Compose — Staff/Principal Interview Guide

---

## How to Use This Guide

This guide is structured for interviewers and candidates preparing for staff- or principal-level ML design interviews. The interview is **45 minutes** total. Each section includes an **interviewer prompt**, the **signal being tested**, and **four-level model answers** representing the candidate response quality spectrum.

**Rating Levels:**
- **No Hire** — Fundamental misunderstanding or silence
- **Lean No Hire** — Partial understanding, significant gaps, needs heavy prompting
- **Lean Hire** — Correct understanding, hits main points, minor gaps
- **Strong Hire** — Deep, nuanced, first-principles reasoning, proactively addresses trade-offs, demonstrates platform-level thinking

**Interviewer Notes:**
- Spend the first minute reading the prompt aloud and giving the candidate time to think silently.
- Do not volunteer information unless the candidate is stuck for more than 90 seconds.
- Use the follow-up probes to differentiate Hire from Strong Hire.
- The principal-level bar requires connecting individual design decisions to broader organizational or platform impact.

**Time Budget:**

| Section | Time |
|---|---|
| Problem Statement & Clarification | 5 min |
| ML Problem Framing | 5 min |
| Data & Preprocessing | 8 min |
| Model Architecture Deep Dive | 12 min |
| Evaluation | 5 min |
| Serving Architecture | 7 min |
| Edge Cases & Failure Modes | 5 min |
| Principal-Level Platform Thinking | 3 min |

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

> "Design Gmail Smart Compose — a system that suggests how to complete the current sentence as a user types an email. The suggestion appears as grey inline text and can be accepted with a single keystroke. Walk me through your approach, starting with what you'd need to clarify."

### Signal Being Tested

Does the candidate recognize that Smart Compose is not just language modeling but a real-time system with tight latency constraints, personalization requirements, and a quality-precision trade-off?

### Six Clarification Dimensions

| Dimension | Why It Matters |
|---|---|
| **Latency SLA** | 100ms end-to-end is the human perception threshold; determines architecture size |
| **Suggestion length** | Short phrase vs. full sentence — affects decoding strategy |
| **Personalization depth** | User-specific model vs. global model with user context |
| **Languages** | Multilingual vs. English-only — determines vocabulary and training data |
| **Sensitive content handling** | Emails contain PII; privacy policy shapes data collection strategy |
| **Trigger conditions** | When to suggest: every keystroke, every space, or on pause? |

### Follow-up Probes

- "What changes about your design if the latency target is 50ms vs. 200ms?"
- "If suggestions are accepted 5% of the time, is that success or failure?"
- "How does Smart Compose change if it must work completely on-device?"

---

### Model Answers — Section 1

**No Hire:**
The candidate proposes training a GPT-style model on email data and displaying predictions. No mention of latency constraints, privacy, or what "suggestion quality" means.

**Lean No Hire:**
Asks about scale ("how many users?") but misses the latency question, which is the defining constraint. Does not ask about personalization or privacy.

**Lean Hire:**
Identifies latency (< 100ms end-to-end), languages, and privacy as key clarifications. Notes that the system is trigger-based (suggestions appear after keystroke events) and asks about acceptance rate as a success signal.

**Strong Hire Answer (first-person):**

The 100ms latency constraint is the forcing function that shapes the entire design. Psychoacoustic research shows that humans perceive any response beyond 100ms as "delayed" — so Smart Compose must feel instantaneous. This eliminates large models with multi-second inference times and forces us into a very specific design space.

Let me clarify six dimensions before proceeding.

First, the exact latency budget. If the end-to-end budget is 100ms, I want to know how much is network RTT (typically 20–50ms for a web app) vs. model inference. If running on-device, the network is eliminated but compute is severely constrained (mobile CPU vs. server GPU).

Second, suggestion granularity. Does Smart Compose complete the current word only, the current phrase, or the entire sentence? Google's production system completes phrases (3–7 words), which is the right balance of value and quality risk.

Third, personalization. A global model will suggest generic completions. A user-adapted model learns I always write "Thanks for reaching out" rather than "Thank you for contacting me." But personalized models require user data to be processed server-side or stored on-device — a significant privacy engineering question. Google's public position is that Smart Compose uses federated learning for user adaptation.

Fourth, multilingual support. English-only is simpler. A multilingual system needs either a single larger multilingual model or separate per-language models.

Fifth, privacy constraints. Emails contain highly sensitive content — financial information, health data, legal documents. I need to understand the data retention policy: can email content be used for model training? Is it only used to compute embeddings immediately discarded?

Sixth, trigger conditions. Does the model run on every keystroke, after every space, or only when the user pauses? Running on pause (100–200ms idle threshold) is a good balance between recency and inference cost.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "How do you formally frame Smart Compose as an ML problem? What is the input and output?"

### Signal Being Tested

Does the candidate correctly frame this as conditional language modeling? Can they specify the full input context and the output format, and explain beam search for phrase generation?

### Follow-up Probes

- "What is the formal input to the model? What context beyond the current email do you include?"
- "Is this a generation problem or a ranking problem? Could you solve it with retrieval?"
- "What is the training loss and what does it optimize?"

---

### Model Answers — Section 2

**No Hire:**
"It's a text prediction problem — predict the next word." Cannot formalize beyond this.

**Lean No Hire:**
Correctly identifies next-token prediction but cannot specify the full input context or explain beam search for phrase generation.

**Lean Hire:**
Correctly frames as conditional language modeling with cross-entropy loss. Specifies input as (email thread context + current typed prefix) → next phrase. Explains beam search for phrase generation.

**Strong Hire Answer (first-person):**

Smart Compose is a conditional text generation problem. The model estimates:

```
p(y_1,...,y_k | x) = Π_{t=1}^{k} p(y_t | y_1,...,y_{t-1}, x)
```

where x is the full context (email thread + current partial sentence) and y_1..y_k is the suggested completion phrase.

The full input context x is richer than just the current typed text:
```
x = [email_subject; prior_email_chain; current_email_body; current_sentence_prefix]
```

The subject line sets topic. Prior email chain sets context. Current body sets tone and style. The current sentence prefix is the most immediate constraint.

The training objective is standard causal language modeling (cross-entropy):
```
L = -Σ_{t=1}^{T} log p_θ(w_t | w_1,...,w_{t-1}, x)
```

**Could we frame this as ranking?** Yes — a retrieval approach would store common phrase completions and rank them by relevance to the current context using fast semantic similarity. This is faster than generation (no beam search, just vector lookup) and easier to control (curated phrase database). The downside: coverage is limited; novel context-specific completions are impossible. Production Smart Compose likely uses a hybrid: retrieval for common phrases (fast path) + generation for novel contexts.

At inference, I use beam search with beam width 4–8 to generate top completion candidates:
```
beam_search(x, width=4, max_tokens=8) → top k (completion, log-probability) pairs
```

With length-normalized scoring to prevent short sequences from dominating:
```
score(y_1...y_t) = Σ_{i=1}^{t} log p(y_i | y_{<i}, x) / t^α,  α ≈ 0.7
```

I apply a confidence threshold: suppress the suggestion if max beam log-probability < τ. This prevents low-confidence, potentially embarrassing suggestions from surfacing.

---

## Section 3: Data & Preprocessing (8 min)

### Interviewer Prompt

> "What data do you train on, and how do you preprocess it? How do you handle the privacy sensitivity of email data?"

### Signal Being Tested

Does the candidate understand data collection, privacy-preserving techniques, and the specific preprocessing needs for email-style text?

### Follow-up Probes

- "What is federated learning and why is it useful here?"
- "How do you handle the huge variance in email style and length?"
- "What preprocessing is specific to email data vs. general web text?"

---

### Model Answers — Section 3

**No Hire:**
"I would scrape email data from the web." Fundamental misunderstanding of privacy requirements.

**Lean No Hire:**
Knows privacy is a concern but cannot describe federated learning or differential privacy as concrete solutions.

**Lean Hire:**
Describes using anonymized email data with consent, mentions federated learning for personalization, and notes standard NLP preprocessing (tokenization, truncation). Can describe the context window construction.

**Strong Hire Answer (first-person):**

Email data is among the most privacy-sensitive data that exists. The data strategy must be designed with privacy as a first-class constraint.

**Training data sources:**
1. *Consented email data*: Gmail users who opt into Smart Compose improvement contribute email text (stripped of sensitive identifiers) for training. This is the primary data source.
2. *Public text pretraining*: Before fine-tuning on email data, the model is pretrained on large-scale public text (web crawl, books). This gives the model general language understanding.
3. *Synthetic data*: For low-resource scenarios (specific languages, formal email styles), generate synthetic emails using a larger model.

**Privacy-preserving techniques:**

1. *De-identification*: Strip names, email addresses, phone numbers, company names, and dates before training using NER + regex rules.

2. *Differential privacy (DP-SGD)*: Add calibrated Gaussian noise to gradients during training:
```
Clipped gradient: g̃_i = g_i / max(1, ||g_i||_2 / C)
Noisy update: (1/B) Σ g̃_i + N(0, σ²C²I/B²)
```
This provides a formal privacy guarantee — an adversary who observes the trained model cannot reconstruct individual training examples with high probability.

3. *Federated learning for personalization*: Instead of sending user emails to a central server, personalization model updates are computed on-device. Each device computes a gradient update on local email data, sends only the gradient (not the email), and the server aggregates (Federated Averaging). Privacy cost: only gradient updates leave the device, not raw email content.

**Preprocessing pipeline:**
1. *Tokenization*: Use a WordPiece or SentencePiece tokenizer trained on email data. Email writing uses different vocabulary than web text (abbreviations, greetings, sign-offs).
2. *Context window construction*: Concatenate `[subject | chain_history | current_body | CURSOR]` where CURSOR is a special token indicating where the model should begin generating.
3. *Length normalization*: Truncate from the far left (oldest email chain history) if context exceeds the model's maximum sequence length. This preserves the most relevant recent context.
4. *Suggestion trigger filtering*: Remove training examples where the "completion" is less than 3 words (not worth suggesting) or contains PII patterns.

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "Describe the model architecture. Why is a decoder-only transformer the right choice? Walk me through the attention mechanism and the latency-quality trade-off."

### Signal Being Tested

Does the candidate understand decoder-only (causal) transformers at a mechanistic level? Can they explain causal masking, KV-caching, and the model size vs. latency trade-off?

### Follow-up Probes

- "Explain causal masking — why is it necessary for text generation?"
- "How do you decide on model size given the 100ms latency constraint?"
- "What is beam search and why use it over greedy decoding for phrase generation?"

---

### Model Answers — Section 4

**No Hire:**
"I would use a transformer." Cannot explain causal masking or why decoder-only is appropriate.

**Lean No Hire:**
Correctly names GPT-style decoder-only transformer but cannot explain the causal attention mask or why encoder-decoder would be wrong for generation.

**Lean Hire:**
Correctly explains causal masking and why the model generates left-to-right. Describes model size trade-off. Explains beam search over greedy decoding.

**Strong Hire Answer (first-person):**

The right architecture is a decoder-only (causal) transformer. Let me explain why this choice is correct and walk through the mechanics.

**Why decoder-only?**
Smart Compose generates text left-to-right, token by token. Each token must be conditioned on all previous tokens but must not see future tokens (they haven't been typed yet). The decoder-only architecture enforces this via a causal attention mask:
```
Mask_{ij} = {0 if j ≤ i, -∞ if j > i}
Masked Attention = softmax((QK^T + Mask) / √d_k) · V
```
With the -∞ mask, position i attends only to positions 1..i. After softmax, the -∞ entries become zero, blocking future token attention.

An encoder-only architecture (BERT) sees the full sequence bidirectionally — correct for understanding tasks but cannot generate left-to-right. An encoder-decoder (T5) is correct for seq2seq but adds encoder cost with no benefit here since our input and output are the same sequence.

**Model Architecture for Latency Constraints:**

For the 100ms end-to-end constraint (with ~50ms for model inference), the model must be small. Google's published work describes a model in the range of 100M–400M parameters. Typical configuration:
- 12–24 transformer layers
- Hidden dimension d_model = 512–1024
- 8–16 attention heads
- Feed-forward dimension 4 × d_model
- Vocabulary: 32K–64K subword tokens

The latency-quality Pareto frontier: a 100M parameter model at INT8 quantization runs in ~5ms on a modern server GPU. A 1B parameter model runs in ~30ms. A 7B model runs in ~150ms — exceeding our budget. The model size is directly constrained by the latency SLA.

**Beam Search:**
Greedy decoding picks argmax at each step — fast but produces suboptimal completions. The highest-probability first word may not lead to the highest-probability phrase. Beam search maintains k=4–8 candidate sequences:
```
At each step, expand each beam candidate by V vocabulary tokens
Keep top-k by score(y_1...y_t) = Σ log p(y_i | y_{<i}, x) / t^α
Stop when all beams hit EOS or max_length
```
In practice for Smart Compose, beam width 4 with max 8 tokens gives good quality with acceptable latency overhead (<5ms for beam search post-generation).

**KV-Cache:**
The email context (subject + thread + current body) is processed once when the compose window opens. Its K and V matrices are cached, keyed by a hash of the context. At each keystroke, only the new token's forward pass is needed:
```
Cached: K_{1..n}, V_{1..n} for all context tokens
Per-keystroke: compute q_{n+1}, k_{n+1}, v_{n+1}
Attention_{n+1} uses [K_cache; k_{n+1}] and [V_cache; v_{n+1}]
```
This reduces per-keystroke latency from O(n) to O(1), typically cutting inference time from ~20ms to ~2ms.

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How do you evaluate Smart Compose? What metrics measure success and how do they relate to user value?"

### Signal Being Tested

Does the candidate understand both model-quality metrics (perplexity) and user-value metrics (acceptance rate, keystroke savings)? Can they explain the A/B testing setup?

### Follow-up Probes

- "What is perplexity and what does a reduction from 40 to 25 mean practically?"
- "If acceptance rate is 8%, is Smart Compose successful?"
- "How would you A/B test Smart Compose? What is the treatment and control?"

---

### Model Answers — Section 5

**No Hire:**
"I would measure accuracy." Cannot define perplexity or explain acceptance rate.

**Lean No Hire:**
Mentions perplexity and acceptance rate but cannot explain what perplexity measures or how it correlates with user experience.

**Lean Hire:**
Correctly defines perplexity, explains acceptance rate as the primary business metric, and describes an A/B test setup. Notes BLEU is commonly used but is a weak proxy for suggestion quality.

**Strong Hire Answer (first-person):**

I use a layered evaluation framework: offline model quality metrics, online product metrics, and human quality evaluation.

**Offline Metrics:**

*Perplexity* on a held-out email test set:
```
PP = exp(-1/N Σ_{t=1}^{N} log p(w_t | w_{<t}))
```
Perplexity of 30 means the model is "as surprised as if choosing uniformly among 30 options" at each token. Reducing from 40 to 25 typically translates to meaningfully better suggestions. However, perplexity doesn't directly predict suggestion quality from the user's perspective — a model can achieve low perplexity by always suggesting common phrases.

*Exact match rate*: what fraction of the time does the model's top-1 suggestion exactly match what the user typed next? This is a stricter measure than perplexity and directly predicts potential acceptance.

*Top-k recall at N*: does the correct continuation appear in the top-k suggestions within N tokens? This captures the beam search quality.

**Online Metrics:**

*Acceptance rate*: % of shown suggestions that are accepted (Tab key press). Google publicly reported ~11% acceptance rate in early Smart Compose deployment. The theoretical maximum depends on how often completions match user intent; 10–15% is the production target range.

*Keystroke Savings Rate (KSR)*: (keystrokes saved via accepted suggestions) / (total keystrokes without Smart Compose). This is the ultimate business value metric.
```
KSR = Σ_{accepted} len(accepted_suggestion) / total_keystrokes_baseline
```

*Negative interaction rate*: % of suggestions explicitly dismissed or that cause typing errors. Smart Compose must not slow users down — a suggestion that interrupts flow is worse than no suggestion.

**A/B Test Setup:**
Treatment: Smart Compose enabled. Control: Smart Compose disabled. Primary metric: KSR. Secondary: email composition time, user retention. Run for 2 weeks minimum to account for novelty effect (users initially interact more because it's new).

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "Walk me through the serving infrastructure for hundreds of millions of Gmail users at sub-100ms latency."

### Signal Being Tested

Does the candidate understand KV-caching, model quantization, debouncing, and the on-device vs. server-side trade-off?

### Follow-up Probes

- "How does the KV-cache work in this specific context?"
- "What is INT8 quantization and what does it trade off?"
- "When should Smart Compose run on-device vs. server-side?"

---

### Model Answers — Section 6

**No Hire:**
"I would run the model on a server." No understanding of latency optimization.

**Lean No Hire:**
Mentions batching and caching but cannot explain KV-cache specifically or model quantization.

**Lean Hire:**
Correctly explains KV-cache for email context prefix, INT8 quantization, and the on-device vs. server trade-off.

**Strong Hire Answer (first-person):**

The serving architecture must achieve two things: < 100ms end-to-end latency and scalability to hundreds of millions of users.

**Request flow:**
1. User types a character in Gmail (browser/app)
2. After 100–200ms keystroke pause (debounce), client sends: `{context_hash, new_token, cursor_position}`
3. Server looks up cached K/V matrices for this email context (keyed by context_hash)
4. Model runs single forward pass for new token + beam search for completion
5. Top-3 suggestions returned to client in < 50ms
6. Client renders grey inline text in < 5ms

**KV-Cache:**
The email context (subject + thread + current body) is processed once when the compose window opens. K and V matrices for all n context tokens are cached on the server, keyed by a hash of the context.

When the user types a new character, only the incremental forward pass is needed:
```
New position n+1: only compute q_{n+1}, k_{n+1}, v_{n+1}
Attention_{n+1} = softmax([q_{n+1}·K_{cached}^T, q_{n+1}·k_{n+1}^T] / √d_k)·[V_{cached}; v_{n+1}]
```
This is O(1) per keystroke vs. O(n) for a full recompute, reducing latency from ~20ms (full context) to ~2ms (incremental).

**INT8 Quantization:**
Model weights stored in 8-bit integers instead of 32-bit floats:
- 4× reduction in model size (400MB → 100MB)
- 2–3× speedup in matrix multiplications (hardware INT8 SIMD units)
- < 1% quality degradation for well-calibrated quantization

Post-training quantization (PTQ) is applied to deployed weights; QAT (quantization-aware training) can recover additional quality if PTQ causes significant degradation.

**On-device vs. Server:**
Server-side: appropriate for complex email contexts (long thread history), multilingual support, fresh model updates. On-device: appropriate for offline mode, privacy-sensitive content, or very low latency (eliminate network RTT entirely). A ~50M parameter quantized model runs in <20ms on a modern phone via TFLite/Core ML. Hybrid approach: small on-device model as fast path; larger server model for complex suggestions.

**Debouncing:**
If the model runs on every keystroke (26ms keystroke cadence for fast typists), inference requests would exceed server capacity. Debounce: wait for 100–200ms of inactivity before sending a request. This dramatically reduces QPS while maintaining perceived responsiveness.

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the most critical failure modes of Smart Compose, and how do you detect and mitigate them?"

### Signal Being Tested

Does the candidate identify privacy leakage via memorization, embarrassing suggestions, and quality degradation on non-English text? Can they propose concrete mitigations?

### Follow-up Probes

- "What happens if the model suggests a medical condition or financial detail the user didn't intend to share?"
- "How do you handle offensive or discriminatory suggestions?"
- "What is the memorization failure mode, and why is it specifically dangerous here?"

---

### Model Answers — Section 7

**No Hire:**
Cannot identify Smart Compose-specific failure modes. Generic "the model might be wrong."

**Lean No Hire:**
Mentions "bad suggestions" but cannot describe privacy leakage or the memorization problem specifically.

**Lean Hire:**
Identifies embarrassing suggestions, privacy leakage via memorization, and language coverage gaps. Proposes filtering and differential privacy as mitigations.

**Strong Hire Answer (first-person):**

Smart Compose has four failure modes that are qualitatively different from typical ML errors.

**1. Training data memorization / privacy leakage:**
If the model has seen similar emails in training, it may memorize and reproduce personal details: specific names, addresses, financial figures, or health information. This is severe — the model effectively leaks one user's private data into another user's suggestions.

Mitigation: differential privacy (DP-SGD) provides a formal bound on memorization. Additionally, apply a post-generation filter: if a suggestion contains named entities, phone numbers, email addresses, or dollar amounts not present in the current email's context, suppress it.

**2. Sensitive or embarrassing suggestion generation:**
The model may suggest phrases that are factually inappropriate (suggesting a salary figure in a personal email), politically sensitive, or offensive. Training data correlations between certain subject lines and responses could cause inappropriate completions.

Mitigation: a fast binary safety classifier (DistilBERT-size) runs on each beam candidate before returning suggestions. Any suggestion flagged as potentially offensive, containing sensitive categories, or containing PII not in context is suppressed.

**3. Suggestion repetition loop:**
If the user ignores several suggestions, the model may be stuck in a local minimum — repeatedly suggesting the same phrase. This degrades experience without providing value.

Mitigation: exponential backoff — if N consecutive suggestions for the same context are not accepted, increase the confidence threshold (only show higher-confidence suggestions). After M rejections, pause suggestions entirely for 30 seconds.

**4. Quality degradation on non-Latin scripts:**
Smart Compose performs significantly worse on Japanese, Arabic, or Hindi where tokenization is fundamentally different. WordPiece trained primarily on Latin script produces high perplexity on CJK text.

Mitigation: language detection at character level; route non-English requests to language-specific models or a multilingual model fine-tuned on that language.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "Smart Compose is successful in Gmail. Now Google Docs, Google Chat, and Android Messages all want the same capability. How do you build this as a platform?"

### Signal Being Tested

Does the candidate think about shared infrastructure, per-surface customization, and the platform economics of serving multiple products from the same model?

### Follow-up Probes

- "What changes per product surface, and what stays the same?"
- "How do you handle the different privacy requirements of Chat vs. Gmail?"

---

### Model Answers — Section 8

**No Hire:**
"Build a separate model for each product." No consideration of shared infrastructure.

**Lean No Hire:**
Suggests a shared model but doesn't identify what must be customized per surface or how to manage the different privacy requirements.

**Lean Hire:**
Proposes a shared base model with surface-specific fine-tuning and per-surface context encoders. Notes that the serving infrastructure (KV-cache, beam search, quantization) is fully reusable.

**Strong Hire Answer (first-person):**

This is a base model + adapter platform design. The core language model is shared across surfaces; only the conditioning and fine-tuning change.

**What stays the same across surfaces:**
- Pretrained language model weights (the expensive part to train — shared)
- Serving infrastructure: KV-cache, beam search, INT8 quantization, request routing
- Evaluation harness: perplexity, acceptance rate measurement pipeline
- Safety layer: offensive suggestion filter, PII detection

**What changes per surface:**
- *Context encoder*: Gmail context is `[subject | thread | body]`; Docs context is `[document_title | preceding_paragraphs]`; Chat context is `[recent_conversation_turns]`. Each surface needs a context serialization format the model understands.
- *Fine-tuning data*: Chat messages are shorter and more informal than Gmail; Docs has longer, more formal completions. Each surface fine-tunes a LoRA adapter: `W = W_base + BA` where B ∈ R^{d×r}, A ∈ R^{r×k}, rank r=16. The adapter is ~1.3M parameters vs. 400M for the base model.
- *Latency SLA*: Chat requires < 50ms (conversation feels real-time); Docs can tolerate < 200ms.
- *Privacy requirements*: Chat may be end-to-end encrypted, preventing server-side KV-caching. Docs may have stricter enterprise data retention policies.

**Platform API design:**
```
complete(context: SurfaceContext, prefix: str, surface: Surface) -> List[Suggestion]
```
The platform routes to the appropriate adapter, manages the KV-cache pool, and returns ranked suggestions with confidence scores. Each surface team owns only their context serialization logic and LoRA adapter fine-tuning.

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**Causal language modeling loss:**
```
L = -Σ_{t=1}^{T} log p_θ(w_t | w_1,...,w_{t-1})
```

**Perplexity:**
```
PP(W) = exp(-1/N Σ_{t=1}^{N} log p(w_t | w_{<t}))
```

**Causal attention mask:**
```
Mask_{ij} = {0 if j ≤ i, -∞ if j > i}
MaskedAttn = softmax((QK^T + Mask) / √d_k) · V
```

**Beam search score with length normalization:**
```
score(y_1...y_t) = (Σ_{i=1}^{t} log p(y_i | y_{<i}, x)) / t^α,  α ≈ 0.7
```

**KV-cache incremental update (per keystroke):**
```
K_cache = [K_1,...,K_n], V_cache = [V_1,...,V_n]
Step n+1: concat [K_cache; k_{n+1}], compute attention for q_{n+1} only
```

**DP-SGD gradient clipping:**
```
g̃_i = g_i / max(1, ||g_i||_2 / C)
g̃ = (1/B)(Σ g̃_i) + N(0, σ²C²I/B²)
```

**Keystroke Savings Rate:**
```
KSR = Σ_{accepted} len(accepted_suggestion) / total_keystrokes_baseline
```

**LoRA adapter:**
```
W_fine-tuned = W_base + BA,  B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)
```

### Vocabulary Cheat Sheet

| Term | Definition |
|---|---|
| **Decoder-only transformer** | Causal (left-to-right) transformer for text generation; GPT architecture family |
| **Causal masking** | Attention mask preventing each token from attending to future positions |
| **KV-cache** | Stores key and value matrices for previous tokens; avoids recomputation |
| **Beam search** | Maintains top-k candidate sequences during generation; better than greedy |
| **Perplexity** | exp(cross-entropy loss); measures how "surprised" the model is by test data |
| **Acceptance rate** | % of Smart Compose suggestions that users accept via Tab key |
| **KSR** | Keystroke Savings Rate; ratio of keystrokes saved vs. total keystrokes |
| **Federated learning** | Compute gradients on-device; send only gradients (not raw data) to server |
| **DP-SGD** | Differentially-private SGD; clips gradients and adds calibrated noise |
| **INT8 quantization** | Store model weights in 8-bit integers; 4× size reduction, 2-3× speedup |
| **Debounce** | Wait N milliseconds of inactivity before triggering API call |
| **LoRA** | Low-rank weight update for efficient per-surface fine-tuning |
| **Memorization** | Model reproduces training data verbatim; privacy risk for sensitive data |
| **WordPiece / SentencePiece** | Subword tokenization algorithms; handle vocabulary efficiently |

### Key Numbers Table

| Metric | Value |
|---|---|
| Smart Compose latency target | < 100ms end-to-end |
| Server-side model inference budget | ~50ms |
| Network RTT (web app) | ~20–50ms |
| Production acceptance rate (Google) | ~11% |
| Target model size (latency-constrained) | 100M–400M parameters |
| Beam width | 4–8 |
| Max suggestion length | 6–8 tokens |
| Keystroke savings rate target | 5–15% |
| INT8 quality loss | < 1% perplexity increase |
| Debounce threshold | 100–200ms |
| Gmail active users | 1.8B+ |
| Language coverage | 13+ languages in production |
| LoRA rank (typical) | 8–16 |
| Context window (model) | 512–2048 tokens |

### Rapid-Fire Day-Before Review

1. **Why decoder-only not encoder-only?** Causal masking enables left-to-right generation; encoder-only sees full sequence bidirectionally (can't generate)
2. **What is perplexity?** exp(cross-entropy loss); lower means model is less surprised by test data
3. **How does KV-cache reduce latency?** Stores K,V matrices for context; each new token only needs one incremental forward pass
4. **Beam search vs. greedy?** Beam maintains k candidates; avoids greedy's myopic choices that lead to poor phrases
5. **Why acceptance rate ~11%?** Not all keystroke positions are good trigger points; not all completions match user intent
6. **How does federated learning help privacy?** Compute gradients on-device; send only gradient updates, not email text, to server
7. **What is DP-SGD?** Clip gradient norms, add calibrated Gaussian noise; provides formal bound on training data memorization
8. **INT8 quantization trade-off?** 4× smaller, 2-3× faster inference, <1% quality loss
9. **How to prevent embarrassing suggestions?** Post-generation safety classifier on each beam candidate; suppress PII not present in context
10. **Platform reuse for Docs/Chat?** Shared base LLM + surface-specific LoRA adapters; shared KV-cache serving infrastructure
