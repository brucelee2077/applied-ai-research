# Recurrent Neural Networks (RNNs)

> **How neural networks handle sequences - text, speech, time series.**
> This section is coming soon. Below is a preview of what will be covered.

---

## What's an RNN?

Regular neural networks see each input independently. Show them a sentence one
word at a time, and they have no memory of previous words. That's a problem when
order matters - "dog bites man" and "man bites dog" use the same words but mean
very different things.

**RNNs have memory.** As they process each word (or data point), they keep a
"hidden state" - a summary of everything they've seen so far. When they read
"bites," they remember whether "dog" or "man" came first.

```
HOW AN RNN READS A SENTENCE
============================

  Input:    "The"    "cat"    "sat"    "on"     "the"    "mat"
              |        |        |        |        |        |
              v        v        v        v        v        v
           +------+ +------+ +------+ +------+ +------+ +------+
           | RNN  |→| RNN  |→| RNN  |→| RNN  |→| RNN  |→| RNN  |
           +------+ +------+ +------+ +------+ +------+ +------+
              |        |        |        |        |        |
         "Saw a    "Saw a    "Saw a   "A cat   "A cat   "A cat
          word"     noun"     verb"    did      did      sat on
                                     something something something"
                                               somewhere
                                                         ↓
                                                   OUTPUT: Next
                                                   word prediction

  The same RNN is reused at every step (parameter sharing, just like CNNs!)
  The arrow → represents the "hidden state" being passed forward
```

**Plain English:** Reading a book one word at a time, and keeping a running
summary in your head. Each new word updates your understanding. By the end
of the sentence, your mental summary captures the meaning of the whole thing.

---

## Why RNNs Matter (and Why They Were Replaced)

RNNs were the go-to architecture for language tasks from ~2014-2017. They're
important to understand because:

1. **They introduced key ideas** - hidden states, sequence processing, attention
   mechanisms - that transformers later built on
2. **They're still used** for some time-series and streaming applications
3. **Understanding their limitations** explains why transformers were invented

### The Big Problem: Forgetting

RNNs struggle with long sequences. By the time they reach word 100, they've
mostly forgotten word 1. This is the **vanishing gradient problem** - the
learning signal fades as it travels backwards through many steps.

```
Sentence: "The cat, which was sitting on the mat near the window
           overlooking the garden where the birds were singing on
           a sunny afternoon, ______."

RNN trying to fill the blank: "Umm... I remember something about
birds? Or was it a garden? The beginning is fuzzy..."

The fix → LSTM and GRU cells (covered below)
```

---

## What Will Be Covered

### Planned Notebooks

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 01 | RNN Fundamentals | How the hidden state works, unrolling through time |
| 02 | Backpropagation Through Time | How RNNs learn (and why it's hard) |
| 03 | LSTM | "Long Short-Term Memory" - the forgetting problem solved with gates |
| 04 | GRU | A simpler alternative to LSTM that works almost as well |
| 05 | Sequence Tasks | Text generation, sentiment analysis, time series prediction |
| 06 | Bidirectional RNNs | Reading the sentence forwards AND backwards |
| 07 | Seq2Seq & Attention | Translation, and the bridge to transformers |

### Key Terms Preview

| Term | Meaning |
|------|---------|
| **Hidden state** | The RNN's "memory" - a summary of everything it's seen so far |
| **Time step** | One position in the sequence (one word, one data point) |
| **Unrolling** | Drawing out the RNN at each time step to see the full picture |
| **Vanishing gradient** | The learning signal fading over long sequences (the core RNN problem) |
| **LSTM** | Long Short-Term Memory - an RNN cell with "gates" that control what to remember and forget |
| **GRU** | Gated Recurrent Unit - a simpler version of LSTM with fewer parameters |
| **Gate** | A learned switch (0-1) that controls information flow - like a valve on a pipe |
| **Seq2Seq** | Sequence-to-sequence - read an entire input, then generate an output (translation) |
| **Attention** | "Look back at the input" instead of relying only on the hidden state summary |

### Why This Leads to Transformers

The attention mechanism, first added to RNNs for machine translation, turned out
to be so powerful that researchers asked: "What if we ONLY use attention and drop
the recurrence entirely?" The answer was the Transformer (2017), which powers
ChatGPT, Claude, GPT-4, and virtually all modern language AI.

```
RNNs (1990s)
  ↓
LSTMs (1997) - solved forgetting
  ↓
Attention in RNNs (2014) - solved the information bottleneck
  ↓
"Attention is All You Need" (2017) - dropped recurrence entirely
  ↓
Transformers → GPT → ChatGPT → modern AI
```

Understanding RNNs and their limitations makes the motivation for transformers
click instantly.

---

## Prerequisites

Before this section, you should have completed:

- **All Fundamentals notebooks** (especially backpropagation and training loops)
- **Basic understanding of CNNs** (helpful but not required)

---

[Previous: CNN](../cnn/README.md) | [Back to Neural Networks](../README.md)
