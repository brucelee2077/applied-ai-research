# Recurrent Neural Networks (RNN)

## Overview

Recurrent Neural Networks are designed for sequential data processing, maintaining hidden states that capture information from previous time steps. This section covers RNN architectures and their evolution leading to transformers.

## Key Concepts

### RNN Fundamentals
- Sequential data processing
- Hidden state and memory
- Recurrent connections
- Unrolling through time
- Backpropagation through time (BPTT)

### Challenges
- Vanishing gradients
- Exploding gradients
- Long-term dependencies
- Computational inefficiency (sequential nature)

### LSTM (Long Short-Term Memory)
- Cell state and gates
- Forget gate
- Input gate
- Output gate
- Solving vanishing gradient problem
- Long-term memory retention

### GRU (Gated Recurrent Unit)
- Simplified architecture
- Update and reset gates
- Computational efficiency
- Comparison with LSTM

### Advanced Architectures
- Bidirectional RNNs
- Deep RNNs (stacked layers)
- Encoder-decoder architectures
- Sequence-to-sequence models
- Attention mechanisms in RNNs

### Applications
- Language modeling
- Machine translation
- Text generation
- Speech recognition
- Time series prediction
- Video analysis

## Transition to Transformers

Understanding RNNs is crucial for appreciating transformers:

### Limitations of RNNs
- Sequential processing (no parallelization)
- Difficulty with very long sequences
- Information bottleneck
- Slow training

### How Transformers Address These
- Self-attention replaces recurrence
- Parallel processing of sequences
- Direct connections between all positions
- Positional encodings for sequence order

## Content to be Added

- [ ] RNN mathematics and derivations
- [ ] LSTM and GRU implementations
- [ ] Attention mechanism in RNNs
- [ ] Seq2seq model examples
- [ ] Comparison with transformers
- [ ] Training notebooks

## Classic Papers

- **LSTM** - Hochreiter & Schmidhuber, 1997
- **GRU** - Cho et al., 2014
- **Seq2Seq** - Sutskever et al., 2014
- **Neural Machine Translation** - Bahdanau et al., 2014
- **Attention Mechanisms** - Bahdanau et al., 2014

## Further Reading

- Deep Learning Book - Chapter 10 (Sequence Modeling)
- Understanding LSTM Networks - Christopher Olah
- The Unreasonable Effectiveness of RNNs - Andrej Karpathy
- CS224n: NLP with Deep Learning

---

[Back to Neural Networks](../README.md) | [Previous: CNN](../cnn/README.md) | [Next: Transformers](../../01-transformers/README.md)