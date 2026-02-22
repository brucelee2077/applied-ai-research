# Neural Network Experiments

## Overview

This directory is your **experiment playground**. The notebooks in the
`fundamentals/`, `cnn/`, and `rnn/` directories teach you the concepts.
Here, you put that knowledge to the test by running your own experiments.

**The goal:** Develop intuition for how neural networks behave by changing
things and seeing what happens. No right or wrong answers -- just exploration!

---

## Suggested Experiments

Below are experiment ideas organized by topic. Each one tells you what to
try, what to observe, and which notebooks to reference.

### Fundamentals Experiments

| Experiment | What to Try | What to Observe | Reference |
|-----------|-------------|-----------------|-----------|
| **Activation Function Comparison** | Train the same network with ReLU, sigmoid, tanh | Training speed, final accuracy, gradient magnitudes | `fundamentals/03_activation_functions.ipynb` |
| **Learning Rate Impact** | Try LR = 0.001, 0.01, 0.1, 1.0 | Convergence speed, oscillation, divergence | `fundamentals/08_training_loop.ipynb` |
| **Batch Size Effects** | Try batch sizes 1, 8, 32, 128, 512 | Training stability, generalization, wall-clock speed | `fundamentals/08_training_loop.ipynb` |
| **Initialization Strategies** | Compare random, Xavier, He initialization | Training speed, whether gradients vanish/explode | `fundamentals/04_loss_functions.ipynb` |
| **Overfitting Demo** | Train a big network on tiny data, then add dropout/L2 | Training vs validation accuracy, generalization gap | `fundamentals/07_backpropagation.ipynb` |

### CNN Experiments

| Experiment | What to Try | What to Observe | Reference |
|-----------|-------------|-----------------|-----------|
| **Filter Visualization** | Extract and plot learned conv filters | Edge detectors, color blobs, texture patterns | `cnn/03_pooling_and_stride.ipynb` |
| **Architecture Depth** | Compare 2-layer vs 5-layer vs 10-layer CNN | Accuracy, training speed, gradient flow | `cnn/05_famous_architectures.ipynb` |
| **Transfer Learning** | Use pre-trained ResNet on a small custom dataset | How much data is needed, frozen vs unfrozen | `cnn/06_transfer_learning.ipynb` |
| **Data Augmentation** | Train with/without flips, rotations, color jitter | Accuracy improvement, overfitting reduction | `cnn/04_building_a_cnn.ipynb` |

### RNN Experiments

| Experiment | What to Try | What to Observe | Reference |
|-----------|-------------|-----------------|-----------|
| **Vanishing Gradient Demo** | Train plain RNN on sequences of length 5, 20, 50, 100 | Gradient magnitudes at early time steps | `rnn/02_backpropagation_through_time.ipynb` |
| **LSTM vs GRU** | Train both on the same task, compare | Parameter count, speed, accuracy | `rnn/03_lstm.ipynb`, `rnn/04_gru.ipynb` |
| **Sequence Length Impact** | Vary input sequence length for same task | When does the model start forgetting? | `rnn/01_rnn_fundamentals.ipynb` |
| **Bidirectional Benefit** | Compare unidirectional vs bidirectional RNN | Accuracy improvement on classification | `rnn/06_bidirectional_rnns.ipynb` |

### Comparative Studies

| Experiment | What to Try | What to Observe | Reference |
|-----------|-------------|-----------------|-----------|
| **Depth vs Width** | Compare deep-narrow vs shallow-wide networks | Same parameter count, different performance | `fundamentals/` |
| **Optimizers** | SGD vs SGD+momentum vs Adam vs AdamW | Convergence speed, final accuracy, stability | `fundamentals/08_training_loop.ipynb` |
| **Normalization** | Train with/without batch normalization | Training stability, convergence speed | `cnn/05_famous_architectures.ipynb` |

---

## How to Run an Experiment

```
Step 1: Pick an experiment from the tables above
Step 2: Open the referenced notebook to review the concept
Step 3: Create a new notebook in this directory
Step 4: Set up a controlled experiment:
        - Keep everything the same EXCEPT the variable you're testing
        - Set a random seed for reproducibility
        - Track metrics (loss, accuracy) over time
Step 5: Visualize your results with matplotlib
Step 6: Write down what you learned!
```

### Experiment Template

Every experiment should have:

```
1. OBJECTIVE: What question are you answering?
   Example: "Does ReLU train faster than sigmoid?"

2. SETUP: What's your baseline?
   - Dataset (what, how much)
   - Model architecture (layers, sizes)
   - Training config (epochs, learning rate, batch size)

3. VARIABLE: What ONE thing are you changing?
   - Only change one thing at a time!

4. RESULTS: Plots and numbers
   - Loss curves, accuracy, timing

5. CONCLUSION: What did you learn?
   - Was your hypothesis correct?
   - Any surprises?
```

---

## Tips for Good Experiments

```
+-------------------------------------------------------------------+
|              Experiment Best Practices                             |
|                                                                   |
|   1. CHANGE ONE THING AT A TIME                                   |
|      If you change learning rate AND batch size, you won't know   |
|      which caused the result                                      |
|                                                                   |
|   2. SET RANDOM SEEDS                                             |
|      np.random.seed(42) and torch.manual_seed(42)                |
|      So you can reproduce your results                            |
|                                                                   |
|   3. RUN MULTIPLE TIMES                                           |
|      Neural networks have randomness. Run 3-5 times and          |
|      report average + standard deviation                          |
|                                                                   |
|   4. VISUALIZE EVERYTHING                                         |
|      Plots tell stories that numbers can't                        |
|      Plot loss curves, accuracy, weight distributions             |
|                                                                   |
|   5. WRITE DOWN YOUR FINDINGS                                     |
|      Future-you will thank present-you                            |
+-------------------------------------------------------------------+
```

---

[Back to Neural Networks](../README.md)
