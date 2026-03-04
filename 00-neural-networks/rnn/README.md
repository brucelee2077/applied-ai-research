# Recurrent Neural Networks (RNNs)

> **How neural networks handle sequences — text, speech, time series.**

---

## What You Need to Know First

- **What a neural network is** — layers of numbers that learn from data ([Neural Networks overview](../README.md))
- **Backpropagation** — how neural networks learn by sending error signals backward through layers
- **Matrix multiplication** — multiplying rows by columns to get new numbers

**That's it.** No prior sequence modeling knowledge needed. Start here.

---

## The Analogy: Reading a Book

Think of reading a book one page at a time. You don't memorize every word — you keep a running summary in your head. Each new page updates that summary. By the end, you can tell someone what the book was about even though you only held one page at a time.

**What the analogy gets right:** An RNN does exactly this. It processes one piece of a sequence at a time and keeps a "hidden state" — a running summary of everything it has seen so far. Each new input updates that summary.

**In plain words:** Regular neural networks treat every input as a standalone thing. Show one a sentence word by word, and it forgets each word the moment the next one arrives. That's a problem when order matters — "dog bites man" and "man bites dog" use the same words but mean very different things. RNNs solve this by carrying information forward through the sequence.

**Where the analogy breaks down:** When you read a book, you can flip back to earlier pages. An RNN cannot — it only moves forward, and its summary of early pages gets fuzzier over time.

```
HOW AN RNN READS A SENTENCE
============================

  Input:    "The"    "cat"    "sat"    "on"     "the"    "mat"
              |        |        |        |        |        |
              v        v        v        v        v        v
           +------+ +------+ +------+ +------+ +------+ +------+
           | RNN  |-->| RNN  |-->| RNN  |-->| RNN  |-->| RNN  |-->| RNN  |
           +------+ +------+ +------+ +------+ +------+ +------+
              |        |        |        |        |        |
         "Saw a    "Saw a    "Saw a   "A cat   "A cat   "A cat
          word"     noun"     verb"    did      did      sat on
                                     something something something"
                                               somewhere

  The same RNN is reused at every step (parameter sharing)
  The arrow --> carries the "hidden state" forward
```

---

## Why RNNs Matter (and Why They Were Replaced)

RNNs were the go-to architecture for language tasks from roughly 2014 to 2017. They're worth learning because:

1. **They introduced the core ideas** — hidden states, sequence processing, attention — that Transformers later built on
2. **They're still used** in certain time-series and streaming applications
3. **Understanding their limitations** makes the motivation for Transformers click instantly

### The Big Problem: Forgetting

RNNs struggle with long sequences. By the time they reach word 100, they've mostly forgotten word 1. This is the **vanishing gradient problem** — the learning signal fades as it travels backward through many time steps.

```
Sentence: "The cat, which was sitting on the mat near the window
           overlooking the garden where the birds were singing on
           a sunny afternoon, ______."

RNN at the blank: "I remember something about birds? Or a garden?
                   The beginning is fuzzy..."

The fix --> LSTM and GRU cells (Notebooks 03 and 04)
```

---

## Checkpoint

Try answering these before moving on. If you can't answer one, re-read the sections above. That is completely normal.

1. What makes an RNN different from a regular neural network?
2. What is the "hidden state" and what does it do?
3. Why do RNNs struggle with long sequences?

---

## How This Leads to Transformers

The attention mechanism was originally added to RNNs to help with machine translation. It worked so well that researchers asked: "What if we drop the recurrence entirely and ONLY use attention?" The answer was the Transformer (2017), which powers virtually all modern language AI.

```
RNNs (1990s)
  |
LSTMs (1997) — solved the forgetting problem
  |
Attention added to RNNs (2014) — solved the information bottleneck
  |
"Attention Is All You Need" (2017) — dropped recurrence entirely
  |
Transformers --> GPT --> ChatGPT, Claude, and modern AI
```

Understanding RNNs and their limitations is the fastest way to understand *why* Transformers were invented. You've just learned the foundation that all of modern AI was built on.

---

## Coverage Map

### RNN Architecture

| Topic | Depth | Files |
|-------|-------|-------|
| RNN Fundamentals — hidden states, unrolling, parameter sharing | [Core] | [01_rnn_fundamentals.ipynb](./01_rnn_fundamentals.ipynb) |
| BPTT — backpropagation through time, vanishing/exploding gradients | [Core] | [02_bptt.ipynb](./02_bptt.ipynb) |
| LSTM — gates, cell state, solving vanishing gradients | [Core] | [03_lstm.ipynb](./03_lstm.ipynb) |
| GRU — simpler gated architecture, comparison with LSTM | [Core] | [04_gru.ipynb](./04_gru.ipynb) |

### RNN Applications

| Topic | Depth | Files |
|-------|-------|-------|
| Sequence Tasks — text generation, sentiment analysis, time series | [Applied] | [05_sequence_tasks.ipynb](./05_sequence_tasks.ipynb) |
| Bidirectional RNNs — forward and backward reading, stacked layers | [Applied] | [06_bidirectional_rnns.ipynb](./06_bidirectional_rnns.ipynb) |

### Not Covered Here

| Topic | Depth | Where |
|-------|-------|-------|
| Seq2Seq & Attention | [Awareness] | Covered in the [Transformers module](../../01-transformers/README.md) — attention started in RNNs but is now a Transformer concept |

---

## Key Terms

| Term | Plain-English Meaning |
|------|-----------------------|
| **Hidden state** | The RNN's "memory" — a vector summarizing everything it's seen so far |
| **Time step** | One position in the sequence (one word, one data point) |
| **Unrolling** | Drawing the same RNN cell out at each time step so you can see the full chain |
| **Vanishing gradient** | The learning signal fading over long sequences — the core RNN weakness |
| **LSTM** | Long Short-Term Memory — an RNN cell with gates that control what to remember and forget |
| **GRU** | Gated Recurrent Unit — a simpler cousin of LSTM with about 75% of the parameters |
| **Gate** | A learned value between 0 and 1 that controls information flow — like a dimmer switch |

---

[Previous: CNN](../cnn/README.md) | [Back to Neural Networks](../README.md)
