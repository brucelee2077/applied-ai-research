# Google Translate — Staff/Principal Interview Guide

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
- Use the follow-up probes listed under each section to differentiate Hire from Strong Hire.
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

> "Design Google Translate — a system that translates text between any of 100+ languages. It should handle everything from single words to full documents, in real-time on the web and mobile. Walk me through your approach, starting with what you'd clarify."

### Signal Being Tested

Does the candidate recognize the scope of a 100+ language system and ask the right questions about language pair coverage, latency, document-level context, and quality vs. coverage trade-offs?

### Six Clarification Dimensions

| Dimension | Why It Matters |
|---|---|
| **Number of language pairs** | All-to-all (10K+ pairs) vs. pivot via English — fundamental architectural decision |
| **Latency SLA** | Interactive typing (< 200ms) vs. document translation (seconds OK) |
| **Domain/register** | General web text vs. medical/legal documents — domain adaptation needed |
| **Document context** | Sentence-level vs. document-level — affects coreference and discourse coherence |
| **Low-resource languages** | High-resource (Spanish) vs. extremely low-resource (Swahili dialects) — data availability |
| **Quality vs. coverage trade-off** | Better quality for 20 languages vs. adequate quality for 200 languages |

### Follow-up Probes

- "What changes if you must support 200 languages vs. 20?"
- "What is a pivot language and when is it used? What quality is lost?"
- "How does your design change for medical document translation vs. casual text translation?"

---

### Model Answers — Section 1

**No Hire:**
The candidate immediately proposes training a seq2seq model without asking about language pair coverage, latency, or data availability. No recognition that 100+ languages creates a fundamentally different problem than 2-language translation.

**Lean No Hire:**
Asks about language count but misses the pivot language question, the document-level context problem, and the low-resource language challenge.

**Lean Hire:**
Asks about at least four dimensions. Identifies that all-to-all translation is computationally infeasible (requires N×(N-1) model pairs) and that a pivot language strategy or massively multilingual model is needed.

**Strong Hire Answer (first-person):**

Before designing anything, I need to understand the scope because the architectural decisions bifurcate dramatically depending on the answers.

First, the language pair strategy. If we need all-to-all translation for 100 languages, that's 100×99 = 9,900 directed language pairs. Training one model per pair is infeasible — the training cost and serving infrastructure would be astronomical. There are two alternatives: (a) use English as a pivot language (translate X → English → Y), which adds latency and quality loss for the X→Y pivot hop; or (b) train a massively multilingual model (MMT) that handles all language pairs in a single model. Google's production Translate uses a massively multilingual Transformer that handles 103+ languages. I'll assume this direction.

Second, latency. Interactive typing on the web has a 200ms tolerance; users expect translation to appear as they stop typing. Document translation (uploading a PDF) can take seconds. These are very different serving profiles.

Third, data availability. For Spanish-English, we have hundreds of millions of sentence pairs. For Swahili-Korean, we may have tens of thousands. Low-resource pairs require transfer learning from high-resource pairs and techniques like back-translation.

Fourth, document-level context. Translating sentence by sentence loses coreference ("he" in sentence 5 refers to "the president" in sentence 1) and discourse structure (paragraph transitions, anaphora). Document-level translation is harder but produces more coherent output for long documents.

Fifth, domain register. Medical, legal, and technical documents use specialized vocabulary that general-domain models mishandle. A doctor's notes with "patient presents with acute MI" should not be translated the same way as a casual text.

These five dimensions completely determine my architecture choices. Let me proceed with a massively multilingual Transformer trained on parallel corpora across 100+ languages.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "How do you formally frame machine translation as an ML problem? What are the inputs, outputs, and training objective?"

### Signal Being Tested

Does the candidate correctly frame MT as conditional sequence-to-sequence generation? Can they explain the encoder-decoder architecture's role and why it differs from a decoder-only approach?

### Follow-up Probes

- "Why is encoder-decoder the right architecture, not decoder-only?"
- "What does the encoder learn that a decoder-only model couldn't?"
- "How does teacher forcing work during training? What failure mode does it create?"

---

### Model Answers — Section 2

**No Hire:**
"Machine translation is a classification problem — classify each word's translation." Cannot formalize the sequence-to-sequence nature.

**Lean No Hire:**
Identifies seq2seq but cannot explain why encoder-decoder is appropriate vs. decoder-only, or how teacher forcing works.

**Lean Hire:**
Correctly frames as conditional sequence generation with cross-entropy loss. Explains that encoder reads the full source sentence bidirectionally; decoder generates target tokens conditioning on encoder output via cross-attention.

**Strong Hire Answer (first-person):**

Machine translation is conditional sequence-to-sequence generation. Given a source sentence x = (x_1,...,x_m) in language L_src, produce a target sentence y = (y_1,...,y_n) in language L_tgt that preserves meaning. The model estimates:

```
p(y | x) = Π_{t=1}^{n} p(y_t | y_1,...,y_{t-1}, x)
```

The training objective is cross-entropy (negative log-likelihood) over a corpus of parallel sentence pairs:
```
L = -Σ_{(x,y)} Σ_{t=1}^{|y|} log p_θ(y_t | y_{<t}, x)
```

**Why encoder-decoder, not decoder-only?**

The encoder processes the full source sentence with bidirectional attention — each source token attends to all other source tokens. This builds rich contextual representations: knowing the entire source sentence before producing the first target token is critical for correct translation. For example, translating "I ate the bass" — knowing whether "bass" is a fish or a musical instrument requires seeing the full sentence context.

A decoder-only model (like GPT) would concatenate [source; target] and use causal attention. This is feasible but inefficient: the model must re-read the entire source at every generation step via causal attention to earlier source tokens, rather than encoding it once into a dedicated encoder representation.

The encoder output (one vector per source token) is consumed by the decoder via cross-attention:
```
CrossAttn(Q_dec, K_enc, V_enc) = softmax(Q_dec · K_enc^T / √d_k) · V_enc
```
At each decoder step t, the cross-attention computes which source tokens to "look at" to generate the next target token.

**Teacher forcing during training:**
During training, the decoder receives the correct ground-truth target tokens as input at each step, rather than its own previous predictions. This allows gradient flow through all timesteps efficiently. The failure mode: *exposure bias* — at inference, the decoder sees its own (potentially incorrect) predictions, not ground truth. Small errors compound. Mitigation: scheduled sampling (gradually replace teacher-forcing inputs with model predictions during training), or use CTC-based or non-autoregressive models that reduce sequential dependencies.

**Multilingual framing:**
For a massively multilingual model, each input is prepended with a target language token:
```
Input: [TRANSLATE_TO_FRENCH] x_1 x_2 ... x_m
Output: y_1 y_2 ... y_n (French tokens)
```
This single model handles all N×(N-1) language pairs by learning a shared multilingual representation.

---

## Section 3: Data & Preprocessing (8 min)

### Interviewer Prompt

> "What training data do you use and how do you preprocess it? How do you handle low-resource language pairs?"

### Signal Being Tested

Does the candidate understand parallel vs. monolingual data, back-translation for low-resource languages, and the preprocessing pipeline for multilingual training?

### Follow-up Probes

- "What is back-translation and why does it help low-resource languages?"
- "How do you tokenize a multilingual model spanning 100+ languages?"
- "How do you ensure the model doesn't ignore low-resource languages in favor of high-resource ones?"

---

### Model Answers — Section 3

**No Hire:**
"I would use Wikipedia as training data." Cannot describe parallel corpora or back-translation.

**Lean No Hire:**
Knows parallel corpora are needed but cannot describe back-translation or multilingual tokenization.

**Lean Hire:**
Describes parallel corpora and back-translation. Mentions SentencePiece as a multilingual tokenizer. Notes the challenge of language imbalance and mentions oversampling of low-resource languages.

**Strong Hire Answer (first-person):**

**Parallel corpora (primary training data):**
Parallel corpora contain aligned sentence pairs in two languages: (source, target). Sources for Google Translate:
- CommonCrawl aligned: parallel sentences extracted from multilingual web crawls using language identification + sentence alignment
- WMT benchmarks: curated high-quality parallel data (news domain) for high-resource language pairs
- Books3 aligned: book translations in multiple languages
- Wikipedia: cross-language article pairs provide general-domain parallel data

Data volume varies enormously: English-French has ~50 billion sentence pairs; English-Swahili has ~5 million; English-Tibetan may have < 100K. This imbalance is the fundamental challenge of multilingual translation.

**Back-translation for low-resource languages:**
For a low-resource language L with limited L–English parallel data, we can augment using back-translation:
1. Train an English→L model on available parallel data
2. Use this model to translate large monolingual English corpora into synthetic L
3. Use (synthetic_L, real_English) pairs as additional training data

The key insight: monolingual data in any language is far more abundant than parallel data. For a language like Hindi, there are billions of monolingual Hindi sentences on the web but only millions of Hindi-English sentence pairs. Back-translation allows the model to learn from monolingual data while preserving the supervised translation signal.

Quality of back-translated data matters: only use translations where the back-translation model had high confidence (beam search score above threshold) to avoid adding noisy examples.

**Multilingual tokenization with SentencePiece:**
A single vocabulary must cover all 100+ languages. SentencePiece (or BPE with multilingual training) builds a shared subword vocabulary of typically 64K–256K tokens. The vocabulary is built from a sampled mix of all languages (with upsampling of low-resource languages to ensure representation).

For languages like Chinese, Japanese, and Arabic with different scripts, the vocabulary allocation must be balanced — a vocabulary trained primarily on English will represent CJK characters poorly (each character becomes a separate token, inflating sequence length).

**Temperature-based sampling for language balancing:**
During training, the probability of sampling a sentence pair from language L is:
```
P_T(L) ∝ (n_L / Σ_k n_k)^T
```
where n_L is the number of sentence pairs for language L and T ∈ (0,1] is a temperature. At T=1, sampling is proportional to corpus size (high-resource languages dominate). At T→0, all languages are sampled equally. Google uses T≈0.7, which moderately upsamples low-resource languages without completely ignoring the quality signal from high-resource corpora.

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "Walk me through the encoder-decoder Transformer architecture for neural machine translation. Be specific about the attention mechanism and how the encoder and decoder interact."

### Signal Being Tested

Does the candidate understand the full encoder-decoder Transformer at a mechanistic level? Can they explain encoder self-attention, decoder cross-attention, and decoder causal self-attention, and why all three are necessary?

### Follow-up Probes

- "Why does the decoder need both self-attention and cross-attention at every layer?"
- "What is the role of multi-head attention vs. single-head attention?"
- "How does the Transformer replace recurrence (RNN) for sequence modeling?"

---

### Model Answers — Section 4

**No Hire:**
"I would use an RNN with attention." Cannot describe the Transformer architecture or why it replaced RNNs for MT.

**Lean No Hire:**
Describes Transformer as "using attention instead of recurrence" but cannot explain the three distinct attention types in encoder-decoder or compute attention outputs.

**Lean Hire:**
Correctly describes encoder self-attention (bidirectional), decoder self-attention (causal), and decoder cross-attention. Explains why the Transformer is preferred over RNNs for parallelization. Can specify the loss function.

**Strong Hire Answer (first-person):**

The encoder-decoder Transformer for machine translation has three distinct attention mechanisms that each solve a different problem. Let me walk through them mechanically.

**Encoder Self-Attention (Bidirectional):**
The encoder processes the full source sentence. At each layer, every source token attends to all other source tokens (no masking):
```
Encoder Self-Attn: Attention(Q_enc, K_enc, V_enc) = softmax(Q_enc K_enc^T / √d_k) · V_enc
```
This allows "bank" in "river bank" to incorporate context from "river" and "fishing" in the same sentence, disambiguating the translation. The encoder has no causal constraint — it needs to see the full source before producing representations.

**Decoder Causal Self-Attention:**
The decoder generates the target sentence left-to-right. It attends to its own previously generated tokens, with a causal mask to prevent looking ahead:
```
Decoder Self-Attn: Attention(Q_dec, K_dec, V_dec) with causal mask
```
This allows "the bank" in the target French output to attend to previous target tokens for grammatical agreement and coherence.

**Decoder Cross-Attention:**
At each decoder layer, the decoder attends to the encoder output to "read" the source:
```
Cross-Attn: Attention(Q_dec, K_enc, V_enc) = softmax(Q_dec K_enc^T / √d_k) · V_enc
```
Q comes from the decoder's current state; K and V come from the encoder output. This allows the decoder to focus on the relevant source words when generating each target word — a soft alignment mechanism that replaced the hard alignment tables used in statistical MT.

**Full Transformer Decoder Layer:**
Each decoder layer applies three sub-layers in sequence:
1. Causal self-attention (with residual + layer norm)
2. Cross-attention over encoder output (with residual + layer norm)
3. Feed-forward network (with residual + layer norm)

**Architecture specifics for a production MT model:**
- 6 encoder layers + 6 decoder layers (original "Attention is All You Need" — modern systems use 24–32 layers)
- d_model = 512–1024
- 8–16 attention heads
- Feed-forward dimension = 4 × d_model
- Shared embedding weights between encoder, decoder, and output projection (reduces parameter count by ~20%)

**Why Transformers replaced RNNs for MT:**
RNNs process sequences sequentially — computing h_t requires h_{t-1}. This prevents parallelization during training: a 100-token sequence requires 100 sequential steps. Transformers compute all token representations simultaneously via self-attention. Training is O(1) sequential steps (vs. O(n) for RNNs), enabling use of large GPU clusters efficiently. The cost: O(n²) attention computation in sequence length, but this is easily parallelized on modern hardware.

Additionally, RNNs struggle with long-distance dependencies — gradients vanish over many steps even with LSTMs. Attention directly connects any two positions in O(1) computation.

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How do you evaluate translation quality? What metrics do you use and what are their limitations?"

### Signal Being Tested

Does the candidate understand BLEU and its limitations? Can they describe human evaluation approaches for translation quality?

### Follow-up Probes

- "Walk me through the BLEU score calculation. What does it actually measure?"
- "Why is BLEU score a poor proxy for translation quality in some cases?"
- "What is a better alternative to BLEU for human evaluation, and how would you set it up?"

---

### Model Answers — Section 5

**No Hire:**
"I would check if the translation is correct." Cannot describe any formal evaluation metric.

**Lean No Hire:**
Mentions BLEU but cannot explain what it computes or its limitations.

**Lean Hire:**
Correctly explains BLEU as modified n-gram precision with brevity penalty. Notes key limitations: doesn't capture paraphrases, doesn't measure fluency well, high BLEU doesn't always correlate with human quality.

**Strong Hire Answer (first-person):**

Translation evaluation is fundamentally a human judgment task — there are many correct translations of any sentence — which makes automated metrics imperfect proxies.

**BLEU (Bilingual Evaluation Understudy):**

BLEU computes modified n-gram precision between a candidate translation and one or more reference translations:

```
p_n = Σ_{ngram ∈ cand} min(count(ngram, cand), max_ref_count(ngram)) / Σ_{ngram ∈ cand} count(ngram, cand)
```

The brevity penalty prevents trivially short translations from scoring high:
```
BP = {1 if c > r, exp(1 - r/c) if c ≤ r}
```
where c = candidate length, r = closest reference length.

Final BLEU:
```
BLEU = BP · exp(Σ_{n=1}^{4} w_n log p_n),  w_n = 1/4
```

A BLEU score of 40+ on WMT news translation is excellent; 25–35 is good; below 20 is poor for high-resource pairs.

**BLEU's limitations:**
1. **No paraphrase credit**: "big house" and "large home" have zero n-gram overlap with "large house" in the reference, despite being semantically equivalent.
2. **Ignores fluency independently of adequacy**: a fluent but inaccurate translation can score higher than an accurate but slightly ungrammatical one.
3. **Corpus-level metric**: BLEU is computed over a corpus; individual sentence BLEU is noisy.
4. **Reference bias**: strongly rewards translations that mirror the reference's word choices. Low BLEU doesn't mean bad translation; it may mean the translator chose different but equally valid vocabulary.

**Better alternatives:**
- **chrF**: character-level F-score. Handles morphologically rich languages (German, Turkish) better than word-level BLEU.
- **COMET**: neural metric fine-tuned on human direct assessment scores. Correlates much better with human judgments (Pearson r ≈ 0.9 for COMET vs. ≈ 0.6 for BLEU).
- **Direct Assessment (DA)**: human raters score each translation on a 0–100 scale without seeing the reference. The gold standard for Google's production quality monitoring.

**Production evaluation setup:**
I run automated metrics (BLEU, chrF, COMET) in CI/CD after every model update. Any regression of >2 BLEU points or >0.02 COMET triggers a human review. Monthly, I run DA studies with bilingual human raters on a stratified sample across languages and domains.

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "How do you serve a 100+ language translation system at scale? What are the key infrastructure decisions?"

### Signal Being Tested

Does the candidate understand beam search latency, model compression for MT, and the serving challenges of a massively multilingual model?

### Follow-up Probes

- "How do you keep a 100+ language model's latency competitive with a two-language model?"
- "What is length penalty in beam search and why does it matter for MT?"
- "How do you handle very long documents efficiently?"

---

### Model Answers — Section 6

**No Hire:**
"I would run the model on servers and scale with more GPUs." No MT-specific infrastructure understanding.

**Lean No Hire:**
Mentions batching and beam search but cannot explain length penalty or the specific serving challenges of a multilingual model.

**Lean Hire:**
Correctly explains beam search, encoder KV-caching, and model quantization. Can compare per-language models vs. massively multilingual models from a serving perspective.

**Strong Hire Answer (first-person):**

Serving a 100+ language MT system at Google Translate scale (~150 billion characters translated per day) requires careful attention to both per-request latency and total compute efficiency.

**Beam search configuration:**
For MT, beam search with beam width 4–6 is standard. The key hyperparameter is the length penalty:
```
score(y_1...y_t) = log p(y_1...y_t | x) / (|y| + α)^β
```
where β typically ranges from 0.6 to 1.0. Without length penalty, shorter hypotheses win because they have fewer terms in the sum. Too aggressive a length penalty produces overly long translations. The length penalty is tuned per language pair because target languages have different length ratios relative to the source (e.g., Japanese text is typically much shorter than its English equivalent).

**Encoder output caching:**
The encoder processes the full source sentence once, producing a fixed-size representation (one vector per source token). This encoder output is computed once per request and cached. The decoder beam search reuses this cached encoder output for all beam steps, making beam search cost dominated by the decoder only.

**Batching:**
For web translation, requests arrive as individual sentences. I batch multiple source sentences together for efficient GPU utilization. The challenge: variable-length sequences in a batch require padding, which wastes compute. Solution: sort requests by source length and batch similar lengths together (dynamic batching with length-based bucketing).

**Model size for a multilingual model:**
A massively multilingual Transformer with 103 languages requires more capacity than a bilingual model to maintain quality across all language pairs. Google's production model is reportedly in the range of 3–10B parameters, deployed with INT8 quantization. At INT8, the model fits in a single A100 (80GB) with reasonable batch sizes.

**Long document handling:**
For documents > model context length (typically 512–1024 tokens), I use overlapping chunk translation: split the document into chunks with ~20% overlap; translate each chunk; stitch together by preferring the non-overlapping portion. A document-level context model can optionally receive a document summary as a prefix to maintain coreference across chunk boundaries.

**Real-time typing mode (Google Translate web):**
For the interactive typing experience, I debounce requests at 300ms of inactivity and translate the current sentence. The result updates as the user types. Partial-sentence translation is a harder task than complete-sentence translation; the model needs to tolerate incomplete input gracefully.

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the most critical failure modes of Google Translate? How do you detect and mitigate them?"

### Signal Being Tested

Does the candidate identify hallucination, gender bias in translation, low-resource quality degradation, and code-switching? Can they propose monitoring strategies?

### Follow-up Probes

- "What is 'hallucination' in MT? Give a concrete example."
- "How does gender bias manifest in machine translation?"
- "What is code-switching and why is it hard for MT?"

---

### Model Answers — Section 7

**No Hire:**
Cannot describe MT-specific failure modes. Generic "wrong translations."

**Lean No Hire:**
Mentions "bad translations" and maybe gender bias but cannot explain the mechanistic causes or propose mitigations.

**Lean Hire:**
Correctly identifies hallucination in MT (generating fluent text not supported by source), gender bias, and low-resource quality collapse. Proposes monitoring via back-translation and human evaluation.

**Strong Hire Answer (first-person):**

MT has several failure modes that are qualitatively different from general language model failures.

**MT Hallucination (Under-translation and Over-generation):**
MT models can generate fluent target text that is not supported by the source sentence — a "hallucination" in the MT sense. This differs from LLM hallucination: in MT, the model produces fluent-sounding output that ignores parts of the source sentence or adds information not present. This is particularly common when the source contains rare entities, numbers, or specialized terminology that the model has seen infrequently.

Detection: round-trip translation consistency. Translate x → y → x'. If x' diverges significantly from x (measured by BLEU or semantic similarity), the intermediate translation y may contain hallucinated content.

**Gender bias in translation:**
Many languages are grammatically gendered (French, Spanish, German, Arabic). When translating from gender-neutral English ("The doctor said they would arrive") to a gendered language, the model must infer gender. Models trained on biased corpora (doctors are predominantly male in training data) will default to masculine gender markers.

Mitigation: (1) produce multiple translation options showing both masculine and feminine forms; (2) use context clues from the broader document when available; (3) audit training data for gender representation across occupation nouns.

**Code-switching:**
Users mix languages in a single sentence ("I'm going to the tienda to buy some tacos"). Standard MT models are trained on monolingual source sentences and may fail or produce garbage output on code-switched input. Detection: language identification per sentence; flag mixed-language inputs for a code-switching-specific model or a multilingual model trained on mixed-language data.

**Quality collapse for very low-resource languages:**
For languages with fewer than 100K parallel sentences, the model may produce fluent-looking but semantically incorrect output — it has learned the surface form of the language without understanding the content. Detection: use back-translation round-trip consistency as a proxy metric. Monitor BERTScore or COMET on held-out test sets per language; alert when quality drops below a threshold.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "Google Translate is a consumer product. Now enterprise customers (hospitals, law firms, government agencies) want translation APIs with domain-specific quality and data privacy guarantees. How does your architecture change?"

### Signal Being Tested

Does the candidate think about domain adaptation, data isolation, and enterprise SLA requirements beyond the consumer use case?

### Follow-up Probes

- "How do you adapt a general MT model to medical terminology without expensive retraining?"
- "What data isolation guarantees does an enterprise customer need, and how do you provide them?"

---

### Model Answers — Section 8

**No Hire:**
"Give enterprises access to the same API." No consideration of domain quality or data isolation.

**Lean No Hire:**
Suggests fine-tuning for each enterprise but doesn't address data isolation or latency SLA requirements for enterprise use cases.

**Lean Hire:**
Proposes domain-specific fine-tuning with customer glossaries and terminology injection. Notes data isolation requirements for sensitive domains.

**Strong Hire Answer (first-person):**

Enterprise translation requires three capabilities beyond consumer Translate: domain quality, terminology control, and data isolation.

**Domain quality via lightweight adaptation:**
A hospital system needs medical terminology translated accurately. Rather than full fine-tuning (expensive, risk of forgetting general translation quality), I use:
1. *Terminology injection at inference*: customer provides a glossary of (source_term, target_term) pairs. At beam search time, I boost the probability of target_term when source_term appears in the source. This is a constrained decoding technique — no retraining required.
2. *Domain LoRA adapters*: for enterprise customers with large in-domain parallel corpora (> 100K sentence pairs), train a small LoRA adapter fine-tuned on their data. The adapter is customer-specific and loaded per-request.

**Data isolation:**
Enterprise customers (hospitals, law firms) cannot have their content processed alongside consumer requests — data leakage risk and regulatory compliance (HIPAA, GDPR). Solutions:
- Dedicated serving instances for enterprise tier: no request mixing with consumer traffic
- No training data retention: enterprise requests are never used for model improvement unless explicitly opted in
- VPC peering: requests never traverse the public internet; private network connection to Google Cloud

**SLA differentiation:**
Enterprise customers have uptime SLAs (99.9% vs. 99.5% for consumer), defined p99 latency targets, and human escalation paths for quality disputes. This requires dedicated capacity in the serving fleet, not shared infrastructure with consumer Translate.

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**Conditional sequence-to-sequence objective:**
```
L = -Σ_{(x,y)} Σ_{t=1}^{|y|} log p_θ(y_t | y_{<t}, x)
```

**Encoder-decoder cross-attention:**
```
CrossAttn(Q_dec, K_enc, V_enc) = softmax(Q_dec K_enc^T / √d_k) · V_enc
```

**BLEU score:**
```
p_n = modified n-gram precision (with clipping)
BP = min(1, exp(1 - r/c))
BLEU = BP · exp(Σ_{n=1}^{4} (1/4) log p_n)
```

**BLEU brevity penalty:**
```
BP = {1 if c > r; exp(1 - r/c) if c ≤ r}
c = candidate length, r = reference length
```

**Beam search with length penalty:**
```
score(y_1...y_t) = log p(y_1...y_t | x) / (|y| + α)^β, β ∈ [0.6, 1.0]
```

**Temperature-based multilingual sampling:**
```
P_T(L) ∝ (n_L / Σ_k n_k)^T, T ≈ 0.7 for multilingual MT
```

**Back-translation data augmentation:**
```
Train π_{en→L} on parallel data
Translate monolingual en corpus → synthetic_L
Train with (synthetic_L, real_en) as additional data
```

**Transformer feed-forward sublayer:**
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
Dimension: d_model → 4*d_model → d_model
```

### Vocabulary Cheat Sheet

| Term | Definition |
|---|---|
| **Seq2seq** | Sequence-to-sequence; encoder processes input, decoder generates output |
| **Encoder-decoder** | Architecture with bidirectional encoder and causal decoder connected via cross-attention |
| **Cross-attention** | Decoder attends to encoder output; Q from decoder, K/V from encoder |
| **BLEU** | Bilingual Evaluation Understudy; modified n-gram precision + brevity penalty |
| **chrF** | Character-level F-score; better for morphologically rich languages |
| **COMET** | Neural MT metric trained on human direct assessment scores |
| **Back-translation** | Translate monolingual data to create synthetic parallel pairs for low-resource languages |
| **Teacher forcing** | Training decoder receives gold target tokens, not its own predictions |
| **Exposure bias** | Gap between teacher-forcing training and free-generation inference |
| **Massively multilingual** | Single model handling 100+ language pairs |
| **Pivot translation** | X→English→Y; avoids training X–Y pair directly |
| **SentencePiece** | Language-agnostic subword tokenizer for multilingual vocabularies |
| **Beam search** | Maintains top-k hypotheses during generation |
| **Code-switching** | User mixes multiple languages in a single input |
| **Constrained decoding** | Force specific target words/phrases during beam search (for glossaries) |
| **Direct Assessment (DA)** | Human rating of translation on absolute 0-100 scale without reference |

### Key Numbers Table

| Metric | Value |
|---|---|
| Google Translate languages supported | 133+ |
| Google Translate daily volume | ~150 billion characters/day |
| WMT En-De BLEU (state of art) | ~34–38 BLEU |
| WMT En-De COMET (state of art) | ~0.85–0.90 |
| Minimum parallel data (high-resource) | 100M+ sentence pairs |
| Minimum parallel data (low-resource) | < 100K sentence pairs |
| Beam width (production) | 4–6 |
| Length penalty β (typical) | 0.6–1.0 |
| MT model (production, multilingual) | 3–10B parameters |
| Vocabulary size (multilingual) | 64K–256K subword tokens |
| Google's multilingual model languages | 103+ (M4 model) |
| Temperature for multilingual sampling | T ≈ 0.7 |
| COMET vs. human Pearson correlation | ~0.90 |
| BLEU vs. human Pearson correlation | ~0.60 |

### Rapid-Fire Day-Before Review

1. **Why encoder-decoder not decoder-only for MT?** Encoder processes full source bidirectionally; decoder generates while attending to encoder via cross-attention — efficient and architecturally cleaner
2. **Three attention types in the Transformer decoder?** Causal self-attention (target tokens), cross-attention (over encoder), and feed-forward
3. **BLEU formula summary?** Modified n-gram precision for n=1..4 + brevity penalty; BP · exp(average log p_n)
4. **Why is BLEU insufficient?** No paraphrase credit; ignores fluency vs. adequacy separately; reference bias
5. **What is back-translation?** Translate large monolingual corpus with an existing model to create synthetic parallel data for low-resource languages
6. **What is teacher forcing?** Training decoder receives gold target tokens as input; prevents compounding prediction errors during backpropagation
7. **Temperature sampling in multilingual training?** P_T(L) ∝ n_L^T; T≈0.7 moderately upsamples low-resource languages
8. **What is code-switching?** User mixes multiple languages in one sentence; MT models struggle with this input distribution
9. **How to add enterprise terminology without retraining?** Constrained decoding / terminology injection — boost target_term probability during beam search when source_term appears
10. **MT hallucination vs. LLM hallucination?** MT: model ignores part of source or adds info not in source; LLM: model generates factually incorrect content confidently
