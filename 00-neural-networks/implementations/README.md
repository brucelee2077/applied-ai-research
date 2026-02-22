# Neural Network Implementations

## Overview

This directory is for **standalone code implementations** of the concepts covered
in the tutorial notebooks. While the notebooks teach concepts interactively,
the code here is organized as reusable modules you can import and build on.

---

## From-Scratch Implementations (NumPy Only)

These implementations use only NumPy to help you understand every detail.
Each one corresponds to concepts from the tutorial notebooks.

### Core Building Blocks

| Implementation | What It Does | Notebook Reference |
|---------------|-------------|-------------------|
| `neuron.py` | Single neuron with forward/backward pass | `fundamentals/02_single_neuron.ipynb` |
| `activations.py` | ReLU, sigmoid, tanh, softmax + derivatives | `fundamentals/03_activation_functions.ipynb` |
| `losses.py` | MSE, cross-entropy, binary cross-entropy | `fundamentals/04_loss_functions.ipynb` |
| `layers.py` | Dense (fully connected) layer class | `fundamentals/05_layers_and_networks.ipynb` |
| `network.py` | Complete feedforward network (MLP) | `fundamentals/06_forward_pass.ipynb` |
| `backprop.py` | Backpropagation algorithm | `fundamentals/07_backpropagation.ipynb` |
| `optimizers.py` | SGD, SGD with momentum, Adam | `fundamentals/08_training_loop.ipynb` |

### CNN Components

| Implementation | What It Does | Notebook Reference |
|---------------|-------------|-------------------|
| `conv2d.py` | 2D convolution operation | `cnn/01_what_is_convolution.ipynb` |
| `pooling.py` | Max pooling, average pooling | `cnn/03_pooling_and_stride.ipynb` |
| `cnn.py` | Complete CNN (conv + pool + FC) | `cnn/04_building_a_cnn.ipynb` |

### RNN Components

| Implementation | What It Does | Notebook Reference |
|---------------|-------------|-------------------|
| `rnn_cell.py` | Simple RNN cell | `rnn/01_rnn_fundamentals.ipynb` |
| `lstm_cell.py` | LSTM cell with all gates | `rnn/03_lstm.ipynb` |
| `gru_cell.py` | GRU cell | `rnn/04_gru.ipynb` |
| `birnn.py` | Bidirectional RNN wrapper | `rnn/06_bidirectional_rnns.ipynb` |
| `attention.py` | Simple dot-product attention | `rnn/07_seq2seq_attention.ipynb` |

---

## How to Use

### Option 1: Learn by Reading

Read the code to reinforce what you learned in the notebooks. The implementations
are kept simple and heavily commented.

### Option 2: Build Your Own

Use the notebook tutorials as a guide and try writing these files yourself.
Compare your version with the reference implementations.

### Option 3: Import and Experiment

```python
# Example: Build a simple neural network
from network import NeuralNetwork
from layers import DenseLayer
from activations import relu, softmax

model = NeuralNetwork([
    DenseLayer(784, 128, activation=relu),
    DenseLayer(128, 10, activation=softmax)
])

# Train on MNIST
model.train(X_train, y_train, epochs=10, lr=0.01)
```

---

## Getting Started

### Prerequisites

**For from-scratch implementations:**
```bash
pip install numpy matplotlib
```

**For framework implementations:**
```bash
pip install torch torchvision numpy matplotlib
```

---

## Code Style

All implementations follow these principles:

```
+-------------------------------------------------------------------+
|              Implementation Guidelines                             |
|                                                                   |
|   1. CLARITY OVER CLEVERNESS                                      |
|      Write code a beginner can read, not code that impresses      |
|                                                                   |
|   2. MATCH THE MATH                                               |
|      Variable names match the equations (W, b, h_t, etc.)        |
|      So you can follow along with the notebook formulas           |
|                                                                   |
|   3. COMMENTS EXPLAIN "WHY"                                       |
|      The code shows "what", comments explain "why"                |
|                                                                   |
|   4. SMALL FUNCTIONS                                              |
|      Each function does one thing                                 |
|      forward(), backward(), update() are separate                 |
|                                                                   |
|   5. TESTS INCLUDED                                               |
|      Each file has a simple test at the bottom                    |
|      Run: python filename.py                                      |
+-------------------------------------------------------------------+
```

---

## Suggested Build Order

If you want to build these from scratch, follow this order:

```
1. activations.py     (simplest -- just math functions)
2. losses.py          (also just math functions)
3. neuron.py          (combines activations + weights)
4. layers.py          (groups neurons together)
5. network.py         (stacks layers)
6. backprop.py        (the hardest part!)
7. optimizers.py      (weight update rules)
8. conv2d.py          (2D convolution operation)
9. rnn_cell.py        (recurrent cell)
10. lstm_cell.py      (LSTM with gates)
```

Each step builds on the previous ones. By the end, you'll have a complete
neural network library built from scratch!

---

[Back to Neural Networks](../README.md)
