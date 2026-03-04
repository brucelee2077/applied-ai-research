# Encoder vs Decoder: Same Task, Different Architectures

## The Mystery Worth Solving

You now know there are two main types of transformers: **encoder-only** (like BERT) and **decoder-only** (like GPT). You know the technical difference — encoders can see all words at once (bidirectional), while decoders can only look at words that came before (left-to-right).

But here's the question that matters for interviews and real-world choices: **does that difference actually affect performance?**

If you give both architectures the same task, the same data, and the same number of parameters, does the encoder win? The decoder? Does it depend on the task?

You're about to find out. You'll build both, train both on the same sentiment classification problem, and compare them directly. The results will make the "which architecture for which task" conversation much more concrete.

---

**Before you start, you need to know:**
- The three types of transformers (encoder-only, decoder-only, encoder-decoder) — covered in [Transformer Block](../architecture/transformer-block.md)
- What a causal mask does and why decoders need it — covered in the same file
- How to train a transformer in PyTorch — covered in [Training a Small Transformer](./training-a-small-transformer.md)

---

## The Security Camera Analogy

Imagine two security guards watching camera footage of a building entrance.

**Guard A (the encoder)** watches the entire day's footage at once. They can scrub forward and backward freely. When they see someone acting suspiciously at 2:15 PM, they can check what happened before *and* after — did the person come back? Did they meet someone later? Guard A has the full picture.

**Guard B (the decoder)** watches the footage in real time, moment by moment. When they see the same suspicious person at 2:15 PM, they can only look at what happened *before* — not after, because the future hasn't happened yet. Guard B must make judgments on partial information.

Now ask: **who is better at deciding if someone was a visitor or an intruder?**

Guard A, obviously. They have more information — they can see the full story. But Guard B is better at a different job: predicting what will happen *next* ("based on what I've seen so far, the next person through the door will probably be...").

**What this analogy gets right:** Encoders see everything and are better at understanding. Decoders see only the past and are better at generation. For a classification task (understanding), encoders have a natural advantage because they can use information from the entire input.

**Where this analogy breaks down:** Security guards have memory and can reason about time. Transformer attention doesn't work like memory — it computes similarity scores between all visible positions in parallel. Also, decoders can still do classification (GPT-based classifiers exist), just with a structural disadvantage on tasks where seeing the full input helps.

---

## The Experiment Design

We build two classifiers with the **same size** and train them on the **same data**:

### The Task: Sentiment Classification

Given a short sentence, decide if it expresses a **positive** or **negative** sentiment.

```
"This movie was wonderful"     → Positive
"The food was terrible"        → Negative
"I love this place"            → Positive
"What a waste of time"         → Negative
```

### Two Architectures, Same Size

| Setting | Encoder Classifier | Decoder Classifier |
|---------|-------------------|-------------------|
| d_model | 64 | 64 |
| n_heads | 4 | 4 |
| n_layers | 2 | 2 |
| d_ff | 256 | 256 |
| Attention | Bidirectional (sees all words) | Causal (left-to-right only) |
| Classification strategy | Mean-pool all word vectors → linear | Take last token vector → linear |
| Parameters | ~same | ~same |

The only difference is the attention pattern:

```
Encoder attention (bidirectional):     Decoder attention (causal):

     is  this  movie  great            is  this  movie  great
is  [ ✓    ✓     ✓     ✓  ]     is  [ ✓    ✗     ✗     ✗  ]
this[ ✓    ✓     ✓     ✓  ]     this[ ✓    ✓     ✗     ✗  ]
movie[✓    ✓     ✓     ✓  ]     movie[✓    ✓     ✓     ✗  ]
great[✓    ✓     ✓     ✓  ]     great[✓    ✓     ✓     ✓  ]

Every word sees every word.          Each word sees only past words.
```

### Why This Comparison Matters

In an interview, saying "encoders are for understanding, decoders are for generation" is a start. But showing *why* — with actual accuracy numbers and attention patterns — is much stronger. This experiment gives you that evidence.

---

## What to Expect

### Encoder Advantage on Classification

The encoder should perform better on sentiment classification because:

1. **Full context access:** When processing the word "not" in "this is not bad", the encoder can see both "not" and "bad" simultaneously. The decoder at position "not" cannot see "bad" yet.

2. **Mean pooling uses all positions equally:** The encoder averages all word representations, giving every word equal weight in the final decision. The decoder relies heavily on the last token, which must somehow carry information from the entire sentence.

3. **Bidirectional attention is more efficient for understanding:** Each word can gather information from the entire input in a single layer. The decoder needs more layers to propagate information from early tokens to late tokens.

### What the Attention Maps Will Show

When you visualize the attention patterns:

- **Encoder:** You'll see dense attention patterns. Sentiment-bearing words (like "wonderful" or "terrible") get attended to from all positions. The pattern is roughly symmetric.

- **Decoder:** You'll see a triangular pattern (the causal mask). Early words can only attend to themselves and a few neighbors. Only the last few positions have access to the full sentence. The model must work harder to route sentiment information to the classification position.

---

## Quick Check — can you answer these?

- Why does the encoder use mean pooling but the decoder uses the last token for classification?
- If a sentence has a negation word ("not") near the end, which architecture handles it better and why?
- Both models have the same number of parameters. Why might the encoder still perform better on this task?

If you can't answer one, go back and re-read that part. That is completely normal.

---

## Victory Lap

You now understand, at a concrete level, *why* BERT-style encoders dominated NLP classification tasks for years. It's not just a convention — there's a structural reason bidirectional attention beats causal attention on understanding tasks. In the notebook, you'll train both models and see the difference yourself. You'll also see the attention heatmaps that make this difference visible.

And you'll understand the flip side: why decoder-only models (GPT, LLaMA, Claude) dominate *generation* tasks, and why the field increasingly uses decoders for everything — even classification — once the models are large enough.

---

Ready to run the experiment? → [Encoder vs Decoder — Notebook](./02_encoder_vs_decoder.ipynb)

---

[Previous: Training a Small Transformer](./training-a-small-transformer.md) | [Back to Experiments Overview](./README.md)
