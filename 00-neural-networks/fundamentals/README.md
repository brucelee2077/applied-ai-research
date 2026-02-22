# Neural Network Fundamentals

> **This is where you start.** No prior ML knowledge needed.
> By the end of these 10 notebooks, you'll build a neural network from scratch
> that recognizes handwritten digits with over 95% accuracy.

---

## What's In This Section?

You'll go from "what's a neuron?" to a fully working neural network, one concept
at a time. Every notebook builds on the previous one - don't skip around.

```
THE JOURNEY
===========

  Concepts               Building               Real World
  (what & why)           (how)                  (put it together)

  01 What is a NN?       04 Layers              08 Training Loop
  02 Single Neuron       05 Forward Pass        09 MNIST (real data!)
  03 Activations         06 Loss Functions      10 PyTorch
                         07 Backpropagation

  "I get the idea"  -->  "I can build one"  --> "I can use one"
```

---

## Notebook Guide

### 01 - What is a Neural Network?

**No code. No math. Just the big picture.**

You'll learn what neural networks are through everyday analogies (movie
recommendations, teams of specialists). By the end you'll understand the core
idea: a network of simple math functions that learns from examples.

**Key concepts:** neurons, weights, layers, how learning works

---

### 02 - Single Neuron

**Your first line of real code.**

You build a single neuron from scratch - the smallest possible neural network.
It takes a few numbers as input, multiplies each by a "weight" (how important
is this input?), adds them up, and outputs a decision.

Think of it like deciding whether to go outside: you check the weather, your
schedule, and your mood. Some factors matter more than others. That's a neuron.

**Key concepts:** weighted sum, bias, decision boundaries, NumPy vectorization

**What you'll build:** A neuron that classifies points on a 2D graph

---

### 03 - Activation Functions

**Why neurons need a "decision rule."**

Without activation functions, stacking 100 layers would be no better than having
1 layer. Activation functions are the thing that makes deep learning *deep*.

This notebook covers the 6 most common ones with visual plots, pros/cons, and
when to use each. Don't worry about memorizing them - the notebook includes a
decision guide you can reference later.

**Key concepts:** sigmoid, ReLU, tanh, softmax, non-linearity

**Plain English:** An activation function is like a filter. After a neuron does
its math, the activation function decides "should I pass this signal along or
tone it down?" The simplest one (ReLU) just says: "if it's negative, make it
zero. Otherwise, keep it."

---

### 04 - Neural Network Layers

**From one neuron to many.**

One neuron can only draw a straight line to separate data. Real problems aren't
that simple. This notebook shows you how to combine neurons into layers, and
layers into networks, to handle complex patterns.

You'll learn matrix multiplication - but don't panic. It's just a fancy way to
do "many neurons at once" instead of one at a time.

**Key concepts:** hidden layers, weight matrices, batch processing, parameter counting

**Plain English:** If one neuron is one expert giving an opinion, a layer is a
whole committee. Each expert looks at the data differently, and together they're
smarter than any single expert.

---

### 05 - Forward Propagation

**How data flows through the network.**

This is the "prediction" part. You have inputs, you run them through each layer,
and you get an output. That's forward propagation - data flowing forward from
input to output.

Think of it like an assembly line: raw materials come in one end, each station
does its job, and a finished product comes out the other end.

**Key concepts:** layer-by-layer computation, matrix shapes, function composition

---

### 06 - Loss Functions

**How to measure "how wrong is my prediction?"**

The network makes a guess. The loss function says "here's how far off you are."
This is what the network tries to minimize during training.

Three main types covered:
- **Mean Squared Error** - for predicting numbers ("the house costs $350K")
- **Binary Cross-Entropy** - for yes/no questions ("is this spam?")
- **Categorical Cross-Entropy** - for multiple choice ("is this a cat, dog, or bird?")

**Plain English:** Imagine throwing darts. The loss function measures how far
your dart landed from the bullseye. The goal of training is to get that distance
as close to zero as possible.

---

### 07 - Backpropagation

**The learning algorithm. This is the hard part - take your time.**

Backpropagation answers the question: "which weights should I adjust, and by
how much, to make my predictions less wrong?"

It works by tracing backwards from the output to the input, figuring out each
weight's contribution to the error. Then it nudges each weight in the direction
that reduces the error.

The math uses the **chain rule** from calculus, but the notebook breaks it down
step by step with a domino-effect analogy. Even if the math feels dense, focus
on the intuition: we're just figuring out who to blame for the wrong answer.

**Key concepts:** chain rule, gradients, weight updates, gradient descent

**Plain English:** You baked a cake and it's too salty. Backpropagation is the
process of figuring out: "Was it the recipe? The salt brand? My measuring spoon?"
Once you know who's responsible, you adjust accordingly next time.

---

### 08 - Training Loop

**Put it all together: the complete learning cycle.**

Now you combine everything: forward pass (make a prediction), loss function
(measure the error), backpropagation (figure out how to fix it), weight update
(actually fix it). Repeat thousands of times.

This notebook also covers practical concerns: how to split your data into
training and validation sets, how to know when to stop training, and how to
pick a good learning rate.

**Key concepts:** epochs, mini-batches, learning rate, validation, early stopping, overfitting

**Plain English:** Training is like studying for an exam by doing practice
problems. Each epoch is one pass through all practice problems. You check your
score on a separate set of problems (validation) to make sure you're actually
learning, not just memorizing.

---

### 09 - Complete MNIST Implementation

**The payoff: a real neural network on real data.**

MNIST is the "hello world" of machine learning - 70,000 images of handwritten
digits (0-9). You'll build a network that looks at a 28x28 pixel image and
correctly identifies the digit over 95% of the time.

Everything is built from scratch using NumPy. No frameworks. No magic. Just the
concepts from notebooks 01-08 put together.

**What you'll build:** A 3-layer network (784 inputs -> 128 hidden -> 64 hidden -> 10 outputs)

---

### 10 - PyTorch Equivalent

**Why frameworks exist.**

You rebuild the exact same MNIST network using PyTorch. Side-by-side comparison
shows that PyTorch does the same thing you did by hand - it just automates the
tedious parts (especially backpropagation).

After this notebook, you'll understand what PyTorch is doing under the hood
instead of treating it as a black box.

**Key concepts:** tensors, autograd, nn.Module, optimizers, training loops in PyTorch

---

### Exercises & Solutions

`exercises.ipynb` has 8 hands-on challenges covering all the fundamentals.
Each exercise has hints (collapsible) and the difficulty builds gradually.

`solutions.ipynb` has complete worked solutions with explanations.

**Tip:** Try each exercise for at least 15 minutes before looking at the solution.
Struggling is where the real learning happens.

---

## Prerequisites

- **Python basics:** variables, functions, loops, lists. That's it.
- **NumPy:** helpful but not required - the notebooks teach what you need as you go.
- **Math:** addition, multiplication, basic algebra. The notebooks explain everything else.

## Setup

```bash
pip install numpy matplotlib scipy
pip install torch  # only needed for notebook 10
```

Then open the notebooks in Jupyter:

```bash
jupyter notebook
# or
jupyter lab
```

---

## Supporting Files

- `utils.py` - Helper functions used across notebooks
- `viz_utils.py` - Visualization utilities for plotting networks and data

These are imported by the notebooks automatically. You don't need to read them
unless you're curious.

---

## Tips for Getting the Most Out of This

1. **Run every cell.** Don't just read - execute the code and look at the output.
2. **Change numbers and re-run.** What happens if you double the weights? Change the learning rate? Break things on purpose.
3. **Don't rush backpropagation.** Notebook 07 is the hardest. It's okay to read it twice.
4. **Draw it out.** Grab paper and sketch the network. Trace the data flow by hand.
5. **Use the glossary.** The main README has a jargon table. Refer to it whenever a term is unclear.

---

[Back to Neural Networks](../README.md) | [Next: CNN](../cnn/README.md)
