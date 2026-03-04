# Training a Small Transformer

## The Mystery Worth Solving

Here's something worth thinking about: every large language model — GPT-4, Claude, LLaMA — started as random numbers. Billions of parameters, all set to random values. The model knew nothing. It couldn't even produce a single sensible word.

Then it was trained on text. And after enough training, it could write essays, answer questions, and solve math problems.

How does a pile of random numbers turn into something that seems to understand language?

You're about to find out — by doing it yourself. You'll build a tiny transformer, initialize it with random weights, and train it on a small piece of text. You'll watch it go from producing total nonsense to producing text that actually looks like the training data.

The model is small (about 50,000 parameters instead of billions). The training data is tiny (a few hundred characters instead of trillions of tokens). But the *mechanism* is exactly the same as what happens inside GPT or Claude.

---

**Before you start, you need to know:**
- How a transformer block works (attention, FFN, residuals, layer norm) — covered in [Transformer Block](../architecture/transformer-block.md)
- What a decoder-only transformer is (left-to-right generation with a causal mask) — covered in the same file
- Basic PyTorch syntax (`nn.Module`, tensors, `.backward()`) — if you completed the [RNN module](../../00-rnn/), you're ready

---

## The Toy Piano Analogy

Imagine you have a toy piano with a few keys, and a child who has never heard music. You sit the child in front of the piano with a simple song playing on repeat — say, "Twinkle Twinkle Little Star."

At first, the child hits random keys. The sounds are nothing like the song. But every time the child presses a key, you tell them: "The next note should have been *this*." Over thousands of attempts, the child starts pressing keys that match the song more and more often.

Eventually, if you let the child start playing from any note, they can continue the melody on their own — not because they memorized the exact sequence, but because they learned the *pattern* of which notes tend to follow which.

**What this analogy gets right:** Training a language model works the same way. The model sees a sequence of characters (or words), tries to predict the next one, gets told the correct answer, and adjusts its weights to do better next time. After enough rounds, it learns the patterns in the text — which characters tend to follow which — and can generate new text that follows those same patterns.

**Where this analogy breaks down:** A child learns from a single song playing in real time. A transformer sees many different sequences from the training data, all shuffled together, and processes them in parallel. It also doesn't learn one note at a time — it predicts every position simultaneously during training (thanks to the causal mask).

---

## What We'll Build

A **character-level, decoder-only transformer**. Let's break that down:

- **Character-level:** The model works with individual characters (letters, spaces, punctuation) instead of words. The "vocabulary" is small — about 40–60 unique characters. This means we don't need a complicated tokenizer.

- **Decoder-only:** The model can only look at characters that came *before* the current position (left-to-right). This is the same architecture used by GPT and Claude. It uses a **causal mask** to block the model from "cheating" by looking at future characters.

- **Transformer:** The model uses the same components you built from scratch — multi-head attention, feed-forward networks, residual connections, layer normalization — but now implemented in PyTorch so they can actually learn.

### Model Specifications

| Setting | Value | Why |
|---------|-------|-----|
| d_model | 64 | Small enough to train on a laptop |
| n_heads | 4 | Each head gets 16 dimensions (64 / 4) |
| n_layers | 2 | Two transformer blocks stacked |
| d_ff | 256 | 4× expansion (standard ratio) |
| Parameters | ~50K–100K | Trains in under 2 minutes on CPU |

Compare this to GPT-3: d_model=12,288, n_heads=96, n_layers=96, d_ff=49,152, with 175 billion parameters. Same architecture, wildly different scale.

---

## How Character-Level Training Works

### Step 1: Turn Characters into Numbers

Every unique character gets a number. This is the simplest possible tokenizer:

```
Character:   H  e  l  l  o     w  o  r  l  d
Index:       7  4  11 11 14  0  22 14 17 11 3
```

There's no "word" boundary. The model sees individual characters. Spaces, punctuation, and newlines are just more characters in the vocabulary.

### Step 2: Create Training Examples

The model's job is: **given some characters, predict the next one.**

From the text "Hello world", we create these training examples:

```
Input:     H          → Target: e
Input:     H e        → Target: l
Input:     H e l      → Target: l
Input:     H e l l    → Target: o
Input:     H e l l o  → Target: (space)
...
```

In practice, we use a fixed window size (say, 32 characters). The model sees 32 characters and predicts the next character at every position simultaneously.

### Step 3: The Training Loop

Each training step works like this:

1. **Feed a chunk of text** into the model
2. **The model predicts** the next character at every position
3. **Compare** predictions to the actual next characters (using cross-entropy loss)
4. **Adjust weights** to make better predictions next time (backpropagation)
5. **Repeat** thousands of times

The loss starts high (the model is guessing randomly among ~50 characters, so loss ≈ ln(50) ≈ 3.9). Over training, it drops as the model learns patterns.

### Step 4: Generate Text

After training, you give the model a starting character (or a few characters) and let it predict the next one. Then you feed that prediction back in and let it predict the next, and so on. This is called **autoregressive generation**.

You control the output's randomness with a **temperature** setting:
- **Temperature = 0.5:** The model picks mostly high-probability characters. Output is repetitive but coherent.
- **Temperature = 1.0:** Normal randomness. A good balance.
- **Temperature = 1.5:** The model takes more risks. Output is creative but can be nonsensical.

---

## What to Expect

### Before Training (Epoch 0)

The model produces random characters. It has no idea what language looks like:

```
xQ7#n!pZ kW$mR...
```

### After Training (a few hundred epochs)

The model learns patterns from the training text. It won't produce perfect English, but you'll recognize familiar words, rhythms, and structures from the training data. The output will *feel* like the training text, even though it's generating new sequences.

This is the core insight: **the model doesn't memorize the text — it learns the statistical patterns.** Which characters tend to follow which, how words are structured, what comes after a space. These patterns are stored in the learned weight matrices — the same Q, K, V projections and FFN weights you built from scratch.

---

## The Causal Mask: No Cheating Allowed

A decoder-only model must not see future characters during training. The **causal mask** enforces this:

```
Position:    1  2  3  4  5
         1 [ ✓  ✗  ✗  ✗  ✗ ]    Position 1 can only see itself
         2 [ ✓  ✓  ✗  ✗  ✗ ]    Position 2 sees 1 and 2
         3 [ ✓  ✓  ✓  ✗  ✗ ]    Position 3 sees 1, 2, and 3
         4 [ ✓  ✓  ✓  ✓  ✗ ]    ...
         5 [ ✓  ✓  ✓  ✓  ✓ ]    Position 5 sees everything before it

✓ = allowed to attend    ✗ = blocked (set to -infinity before softmax)
```

Without the causal mask, the model at position 3 could just look ahead at position 4 to see the answer. That's cheating — and it means the model wouldn't learn any real patterns. The mask forces the model to actually *predict* the next character from context alone.

---

## Quick Check — can you answer these?

- Why does a character-level model not need a tokenizer?
- What does the causal mask prevent, and why would training fail without it?
- If the model's vocabulary is 50 characters, what should the initial loss be approximately? (Hint: think about random guessing.)

If you can't answer one, go back and re-read that part. That is completely normal.

---

## Victory Lap

You now understand the full pipeline for training a transformer from scratch: tokenize the text, create input-target pairs, run the training loop, and generate new text. This is *exactly* what OpenAI did to create GPT — just at a much larger scale with much more data. The mechanism is the same. In the notebook, you'll do it yourself and watch a pile of random weights learn to produce text.

---

Ready to build it? → [Training a Small Transformer — Notebook](./01_training_a_small_transformer.ipynb)

---

[Back to Experiments Overview](./README.md) | [Next: Encoder vs Decoder](./encoder-vs-decoder.md)
