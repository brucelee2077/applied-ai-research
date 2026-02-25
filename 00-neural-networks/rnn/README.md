# Recurrent Neural Networks (RNNs)

> **How neural networks handle sequences — text, speech, time series.**

---

## What's an RNN?

Regular neural networks treat every input as a standalone thing. Show one a sentence word by word, and it forgets each word the moment the next one arrives. That's a problem when order matters — "dog bites man" and "man bites dog" use the same words but mean very different things.

**RNNs have memory.** As they process each element in a sequence, they keep a "hidden state" — a running summary of everything they've seen so far. When they read "bites," they remember whether "dog" or "man" came first.

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

Think of it like reading a book one page at a time. You don't memorize every word — you keep a running summary in your head. Each new page updates that summary. By the end, you can tell someone what the book was about even though you only held one page at a time.

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

## Notebooks

Every notebook builds from intuition to math to working code — all implemented from scratch in NumPy so you can see exactly what's happening. Each one ends with an **Interview Prep** section covering the questions you're most likely to face.

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | [RNN Fundamentals](01_rnn_fundamentals.ipynb) | Hidden state, unrolling through time, parameter sharing, character-level prediction |
| 02 | [Backpropagation Through Time](02_bptt.ipynb) | How RNNs learn, why gradients vanish and explode, gradient clipping, truncated BPTT |
| 03 | [LSTM](03_lstm.ipynb) | Long Short-Term Memory — forget, input, and output gates, cell state, why it fixes vanishing gradients |
| 04 | [GRU](04_gru.ipynb) | Gated Recurrent Unit — a simpler alternative to LSTM with fewer parameters |
| 05 | [Sequence Tasks](05_sequence_tasks.ipynb) | Text generation, sentiment analysis, time series — many-to-one vs many-to-many architectures |
| 06 | [Bidirectional RNNs](06_bidirectional_rnns.ipynb) | Reading sequences forwards AND backwards, stacked/deep RNNs |
| 07 | [Seq2Seq & Attention](07_seq2seq_attention.ipynb) | Encoder-decoder models, the attention mechanism, and the bridge to Transformers |

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
| **Seq2Seq** | Sequence-to-sequence — read an entire input, then generate an output (e.g., translation) |
| **Attention** | A mechanism that lets the decoder "look back" at every encoder step instead of relying on one summary vector |

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

Understanding RNNs and their limitations is the fastest way to understand *why* Transformers were invented.

---

## Prerequisites

Before starting these notebooks, you should have completed:

- **All Fundamentals notebooks** (especially backpropagation and training loops)
- **Basic understanding of CNNs** (helpful but not required)

---

[Previous: CNN](../cnn/README.md) | [Back to Neural Networks](../README.md)
