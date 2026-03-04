# Tokenization and Vocabulary

## Introduction

Tokenization is the foundation of all text-based genAI systems. It's the process that converts raw text — the characters you read — into a sequence of integer IDs that the model processes. Every LLM-powered system depends on tokenization, and the tokenizer's design affects model quality, latency, cost, and multilingual capability.

In interviews, tokenization is rarely the main topic, but it comes up in surprising places. "Why does the model struggle with math?" Tokenization. "Why is our API bill 3x higher for Chinese than English?" Tokenization. "Why does the model output garbled text when given this input?" Tokenization. Candidates who understand tokenization can diagnose these issues and propose solutions.

---

## What Tokenization Does

### From Text to Numbers

Models don't see characters or words. They see sequences of integer IDs, each pointing to an entry in a vocabulary (a fixed list of tokens).

```
Input text: "Hello, world!"
Tokenized:  [15496, 11, 995, 0]
             "Hello"  ","  " world"  "!"
```

Each token maps to a learned embedding vector — a list of numbers that represents that token's meaning. The model processes these embeddings, not the original text.

### Why Not Just Use Words?

Word-level tokenization seems natural but has fatal problems:

| Approach | Problem |
|----------|---------|
| Word-level | Vocabulary explodes (English has >1M words). Any word not in vocabulary → unknown. Can't handle typos, new words, or compound words. |
| Character-level | Sequences become very long (a 100-word paragraph → ~500 characters → ~500 tokens). Model must learn to combine characters into meaning — much harder. |
| Subword (BPE, etc.) | The Goldilocks approach. Common words are single tokens. Rare words are split into meaningful subparts. Vocabulary size is manageable (32K-100K). |

Modern LLMs all use subword tokenization.

---

## Tokenization Algorithms

### BPE (Byte-Pair Encoding)

The most widely used algorithm. Used by GPT, LLaMA, and most modern LLMs.

**How it works:**
1. Start with individual characters as the initial vocabulary
2. Count all adjacent token pairs in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary reaches the target size

**Example progression:**
```
Step 0: ["t", "h", "e", " ", "c", "a", "t"]
Step 1: "th" is the most frequent pair → merge → ["th", "e", " ", "c", "a", "t"]
Step 2: "the" is most frequent → merge → ["the", " ", "c", "a", "t"]
Step 3: "ca" → merge → ["the", " ", "ca", "t"]
Step 4: "cat" → merge → ["the", " ", "cat"]
```

Common words become single tokens. Rare words are split into subword pieces.

### WordPiece

Similar to BPE but uses a likelihood-based merging criterion instead of frequency.

**Difference from BPE:** Instead of merging the most frequent pair, WordPiece merges the pair that maximizes the likelihood of the training corpus.

**Used by:** BERT and related models.

### SentencePiece

Language-agnostic tokenization that works directly on raw text (including whitespace and special characters).

**Key advantage:** Doesn't require language-specific preprocessing (word segmentation, normalization). Works on any language without modification.

**How it handles spaces:** Treats the input as a stream of bytes. The space character is just another character that can be merged into tokens. This is why some tokens start with a special character (▁) representing a space.

**Used by:** LLaMA, T5, many multilingual models.

### Unigram

Starts with a large vocabulary and prunes based on likelihood.

**How it works:**
1. Start with a large vocabulary (all substrings up to a certain length)
2. For each token, compute how much removing it would decrease the corpus likelihood
3. Remove the tokens that decrease likelihood the least
4. Repeat until vocabulary reaches the target size

**Used by:** Some SentencePiece models. The result is similar to BPE but the training process is different.

---

## Vocabulary Size Tradeoffs

| Vocabulary Size | Tokens per Text | Embedding Table Size | Multilingual | Training Cost |
|----------------|-----------------|---------------------|-------------|---------------|
| Small (8K) | More tokens (longer sequences) | Small | Poor (splits non-English aggressively) | Lower per-token, but more tokens |
| Medium (32K) | Moderate | Medium | Decent | Balanced |
| Large (100K+) | Fewer tokens (shorter sequences) | Large | Good (more tokens for diverse scripts) | Higher per-token, but fewer tokens |

**The key tradeoff:** Larger vocabulary → fewer tokens per text (shorter sequences, faster generation) but larger embedding table (more parameters, more memory).

Most modern LLMs use 32K-100K tokens. GPT-4 uses ~100K. LLaMA 2 uses 32K. LLaMA 3 uses 128K.

---

## Why Tokenization Matters for System Design

### Cost

API pricing is per-token. The same text can be a different number of tokens depending on the tokenizer.

| Language | Tokens for "Hello, how are you?" equivalent | Cost Multiplier vs English |
|----------|---------------------------------------------|---------------------------|
| English | ~6 tokens | 1x |
| Spanish | ~7 tokens | 1.2x |
| Chinese | ~12-15 tokens | 2-2.5x |
| Japanese | ~15-20 tokens | 2.5-3.3x |
| Arabic | ~10-14 tokens | 1.7-2.3x |

**Why the disparity:** BPE trained on English-heavy data learns efficient merges for English patterns. Chinese characters are split into byte-level tokens because the tokenizer saw fewer Chinese examples during vocabulary training.

**Impact at scale:** A multilingual application serving equally across English and Chinese users pays 2-3x more for Chinese queries — for the same amount of information.

### Latency

Autoregressive models generate one token at a time. Fewer tokens = faster generation.

If the same content requires 100 tokens in English but 250 tokens in Chinese, the Chinese response takes 2.5x longer to generate. For latency-sensitive applications (real-time chat, code completion), this matters.

### Numerical Reasoning

Tokenization breaks numbers in unpredictable ways:

```
"123456" might become: ["123", "456"]  or  ["12", "345", "6"]  or  ["1", "234", "56"]
```

The model doesn't see "123456" as a number — it sees a sequence of subword tokens. Arithmetic operations require the model to understand the positional relationship between these arbitrary splits. This is one reason LLMs struggle with precise math.

**Workarounds:**
- For applications requiring precise math, use tool calling (calculator, code execution) instead of relying on the model
- Some specialized tokenizers represent digits individually to preserve numerical structure

### Context Window Management

The context window has a fixed token limit. Tokenization determines how much text fits.

**Example:** A 128K context window model:
- English: ~96K words (roughly 200 pages of text)
- Chinese: ~40K characters (roughly 80 pages of equivalent content)

For RAG applications, the effective context window for non-English languages is significantly smaller.

---

## Common Tokenization Issues

### Unseen Characters

Characters not in the vocabulary get handled differently by different tokenizers:
- **UNK token:** Replace with a special [UNK] token. The model knows nothing about it. Information is lost.
- **Byte fallback:** Encode the character as individual bytes. The model can reconstruct the character but sees it as a sequence of byte-level tokens. Most modern tokenizers use this.

### Whitespace Handling

Different tokenizers handle whitespace differently:
- Leading spaces may or may not be separate tokens
- Multiple spaces may be collapsed or preserved
- Tabs and newlines have different tokenization

This matters for code generation (where indentation is syntactically meaningful) and for structured output (where formatting matters).

### Code Tokenization

Code has different patterns than natural language:
- Variable names: `getUserProfile` might tokenize as ["get", "User", "Profile"] or ["getUser", "Profile"]
- Operators: `===`, `!==`, `>>>=` are uncommon in natural language
- Indentation: Python's meaningful whitespace must be preserved

Some models use specialized tokenizers for code that handle these patterns better.

### Special Tokens

Model-specific tokens that serve structural purposes:

| Token | Purpose | Example Models |
|-------|---------|---------------|
| [CLS] | Classification head input | BERT |
| [SEP] | Segment separator | BERT |
| [PAD] | Padding for batch alignment | Most models |
| <\|endoftext\|> | End of sequence | GPT |
| <s>, </s> | Begin/end of sequence | LLaMA |
| <\|im_start\|>, <\|im_end\|> | Chat message delimiters | ChatGPT |

Special tokens must be handled correctly in your application code. Sending the wrong special tokens to a model produces garbage output.

---

## Production Considerations

### Tokenizer-Model Coupling

The tokenizer must exactly match what the model was trained with. Using a different tokenizer — even a slightly different version — produces incorrect token IDs, which map to wrong embeddings, which produce garbage output.

**Rule:** Always load the tokenizer from the same source as the model. Never substitute tokenizers between models.

### Tokenization Speed

For high-throughput systems, tokenization itself can be a bottleneck.

| Implementation | Speed | Language |
|---------------|-------|----------|
| Python (transformers library) | ~10K tokens/sec | Python |
| Rust (tokenizers library by HuggingFace) | ~1M tokens/sec | Rust with Python bindings |
| C++ (SentencePiece) | ~500K tokens/sec | C++ |

For production serving at >1000 QPS, use the Rust-based tokenizers library. The 100x speedup over Python tokenization eliminates the bottleneck.

### Token Counting for API Limits

Before sending a request to an LLM API, count the tokens to ensure the total (system prompt + user input + expected output) fits within the context window.

```
available_for_output = context_window - system_prompt_tokens - user_input_tokens
if available_for_output < min_output_tokens:
    truncate_user_input or summarize_context
```

**Getting the count right:** Different APIs use different tokenizers. OpenAI uses tiktoken. Anthropic uses their own tokenizer. The token count for the same text differs between them. Always use the tokenizer matching the model you're calling.

### Prompt Token Budgets

For RAG applications, budget your context window carefully:

```
Context window: 128K tokens
- System prompt: ~500 tokens
- Few-shot examples: ~1000 tokens
- Retrieved chunks: ~3000-10000 tokens
- User query: ~100-500 tokens
- Reserved for output: ~2000-4000 tokens
- Safety margin: ~500 tokens
```

If retrieved chunks exceed the budget, you must truncate, summarize, or re-rank to keep the most relevant content.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand that LLMs process tokens (not words or characters) and that vocabulary size affects model behavior. For a multilingual chatbot, they should recognize that non-English text requires more tokens and therefore costs more and generates more slowly. They differentiate by knowing that BPE is the standard tokenization algorithm and that the tokenizer must match the model.

### Senior Engineer

Senior candidates can explain tokenization tradeoffs and their impact on system design. They know why numerical reasoning is hard for LLMs (tokenization splits numbers arbitrarily), why multilingual applications are more expensive (tokenizer trained on English-heavy data), and how context window management depends on token counting. For a RAG system, a senior candidate would discuss token budgets — allocating context window space between system prompt, retrieved chunks, and output — and propose strategies for when retrieved content exceeds the budget (re-ranking, summarization, truncation).

### Staff Engineer

Staff candidates think about tokenization as a system-level constraint that affects cost, latency, and fairness. They might point out that tokenizer bias creates an equity problem: users who speak languages with lower tokenization efficiency pay more and wait longer for the same quality of service. A Staff candidate might propose monitoring tokenization efficiency by language and adjusting pricing or generation strategies accordingly. They also understand the tokenizer-model coupling problem at the organizational level: when upgrading to a new model with a different tokenizer, all downstream systems (token counting, context management, cost estimation) must be updated simultaneously.
