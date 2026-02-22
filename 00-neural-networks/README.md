# Neural Networks - From Zero to Hero

> **No experience needed.** This guide assumes you've never heard of machine learning.
> If you can add, multiply, and read Python, you're ready.

---

## What Even Is a Neural Network?

Imagine teaching a toddler to recognize animals. You show them hundreds of pictures:
"This is a cat. This is a dog. This is a cat again." Over time, they just *get it* -
they can tell cats from dogs even in photos they've never seen.

A **neural network** does the same thing, but with math instead of a brain. You feed it
thousands of examples, and it figures out the patterns on its own.

That's the entire idea. Everything else is details about *how* it does this.

---

## What You'll Learn Here

This module takes you from "what's a neuron?" all the way to building real working
neural networks that can recognize handwritten digits. Here's the journey:

```
YOUR LEARNING PATH
==================

   START HERE
       |
       v
  +-----------------+      +-------------------+      +-------------------+
  |  Fundamentals   | ---> | CNN               | ---> | RNN               |
  |  (10 notebooks) |      | (6 notebooks)     |      | (7 notebooks)     |
  |                 |      |                    |      |                   |
  | What neurons    |      | How computers     |      | How networks      |
  | are, how they   |      | "see" images      |      | handle sequences  |
  | learn, build    |      | using filters     |      | like text and     |
  | one from        |      | and patterns      |      | time series       |
  | scratch         |      |                   |      |                   |
  +-----------------+      +-------------------+      +-------------------+
```

By the end, you'll be able to:
- Explain how neural networks work to a friend (without hand-waving)
- Build a neural network from scratch using just Python and NumPy
- Understand what PyTorch/TensorFlow are doing under the hood
- Read about CNNs, backpropagation, or loss functions without getting lost
- Have the foundation to learn about transformers, LLMs, and modern AI

---

## Jargon Glossary (Read This First)

You'll see these terms everywhere in ML. Here's what they actually mean:

| Term | What It Sounds Like | What It Actually Means |
|------|---------------------|----------------------|
| **Neuron** | A brain cell | A tiny math function: multiply inputs by weights, add them up, done |
| **Weight** | How heavy something is | A number that says "how important is this input?" - the network adjusts these to learn |
| **Bias** | Being unfair | Just a number added at the end, like a "starting point" for the neuron's calculation |
| **Layer** | A layer of cake | A group of neurons that work together at the same step |
| **Activation function** | ??? | A simple rule applied after each neuron, like "if negative, make it zero" |
| **Forward pass** | Moving forward | Running data through the network from input to output to get a prediction |
| **Loss function** | Losing something | A way to measure "how wrong was the prediction?" - lower is better |
| **Backpropagation** | Propagating backwards | The math trick for figuring out which weights to blame for a wrong answer |
| **Gradient** | A slope | "Which direction should I adjust this weight to make the answer less wrong?" |
| **Epoch** | A time period | One complete pass through all your training examples |
| **Batch** | A batch of cookies | A small group of examples processed together (faster than one at a time) |
| **Overfitting** | Fitting too much | The network memorized the answers instead of learning the pattern (like memorizing a test instead of understanding the material) |
| **CNN** | An acronym | Convolutional Neural Network - a network designed to work with images |
| **RNN** | Another acronym | Recurrent Neural Network - a network designed to work with sequences (text, time series) |

Don't memorize this table. Come back to it whenever you hit an unfamiliar term.

---

## Directory Guide

### [Fundamentals](./fundamentals/) - Start Here

**10 hands-on notebooks** that build on each other, step by step:

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | What is a Neural Network? | The big picture - no code, just concepts and analogies |
| 02 | Single Neuron | Build your first neuron from scratch with Python |
| 03 | Activation Functions | Why neurons need a "decision rule" and which ones to use |
| 04 | Neural Network Layers | Stack neurons into layers, learn matrix math the easy way |
| 05 | Forward Propagation | How data flows through a network to make predictions |
| 06 | Loss Functions | How to measure "how wrong is my prediction?" |
| 07 | Backpropagation | The learning algorithm - how the network improves itself |
| 08 | Training Loop | Put it all together: epochs, batches, and watching it learn |
| 09 | Complete MNIST | Build a digit recognizer that's >95% accurate |
| 10 | PyTorch Equivalent | Rebuild everything in PyTorch and see why frameworks exist |

Plus **exercises** and **solutions** notebooks for practice.

### [CNN](./cnn/) - Computer Vision

**6 tutorials** on how neural networks "see" images:

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | What Are CNNs? | Why regular networks fail on images, and how CNNs fix it |
| 02 | Convolution Operation | The core trick: sliding filters that detect patterns |
| 03 | Pooling Layers | How to shrink images while keeping the important stuff |
| 04 | Building a Complete CNN | Put it all together into a working image classifier |
| 05 | Famous Architectures | LeNet, AlexNet, VGG, ResNet - the greatest hits |
| 06 | Transfer Learning | Use someone else's trained network as a starting point |

Plus **10 exercises** with full solutions.

### [RNN](./rnn/) - Sequential Data

**7 hands-on notebooks** on how networks handle ordered data like text, speech, and time series:

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | RNN Fundamentals | Hidden states, unrolling through time, parameter sharing |
| 02 | Backpropagation Through Time | How RNNs learn, vanishing/exploding gradients |
| 03 | LSTM | Gates that control memory - solving the forgetting problem |
| 04 | GRU | A simpler alternative to LSTM that works almost as well |
| 05 | Sequence Tasks | Text generation, sentiment analysis, time series prediction |
| 06 | Bidirectional RNNs | Reading sequences forwards AND backwards |
| 07 | Seq2Seq & Attention | Translation, and the bridge to transformers |

### [Implementations](./implementations/) - Code Reference

Reference implementations organized as reusable modules. Build your own
neural network library from scratch, or study the reference code.

### [Experiments](./experiments/) - Hands-On Playground

Guided experiment ideas to build intuition through trial and error.
Change one variable, observe the result, develop deep understanding.

---

## How to Use This Module

### Step 1: Set Up Your Environment

You need Python 3.8+ with these packages:

```bash
pip install numpy matplotlib scipy
pip install torch          # optional, only needed for notebook 10 and CNN exercises
```

### Step 2: Go Through Fundamentals (Notebooks 01-10)

Go in order. Each notebook builds on the last. Don't skip ahead - the concepts
compound on each other.

**Suggested pace:**
- Notebooks 01-03: Read carefully, run every cell, play with the examples
- Notebooks 04-06: Slower going - this is where the math picks up
- Notebooks 07-08: The hardest part - backpropagation. Take your time
- Notebooks 09-10: The payoff - you'll build something real

### Step 3: Try the Exercises

After fundamentals, open `exercises.ipynb` and try solving problems on your own
before looking at `solutions.ipynb`.

### Step 4: Move to CNNs

Once you're comfortable with fundamentals, the CNN section shows you how these
same ideas apply to images.

---

## The Math - Don't Panic

This module uses math, but nothing beyond what you learned in school:

- **Addition and multiplication** - that's what neurons do
- **Basic algebra** - rearranging equations like `y = mx + b`
- **The idea of a slope** - "is this going up or down?" (that's a gradient)

Everything is explained with:
1. A real-world analogy first (like "baking a cake" or "a team of specialists")
2. Then the math, broken into small steps
3. Then code you can run and experiment with
4. Then visualizations so you can see what's happening

If something feels confusing, that's normal. Run the code, look at the plots,
and re-read. It clicks on the second or third pass.

---

## What Comes After This

Understanding neural networks gives you the foundation for everything in modern AI:

```
Neural Networks (you are here)
    |
    +--> Transformers - The architecture behind ChatGPT, GPT-4, Claude
    |      (self-attention builds on concepts from this module)
    |
    +--> Large Language Models (LLMs) - Trained using the same
    |      optimization techniques you'll learn here
    |
    +--> Multimodal Models - Combine CNNs for vision with
           transformers for language (CLIP, GPT-4V, etc.)
```

---

## Recommended Resources

If you want to supplement this module:

**Videos (great for visual learners)**
- [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk) - Beautiful visual explanations
- [Andrej Karpathy - Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) - Build from scratch

**Books (free online)**
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [Dive into Deep Learning](https://d2l.ai/) - Interactive with code examples

**Courses**
- [Fast.ai](https://course.fast.ai/) - Practical, top-down approach
- [Stanford CS231n](http://cs231n.stanford.edu/) - CNNs for visual recognition (more advanced)

---

## Content Status

- [x] Fundamentals - 10 notebooks + exercises + solutions
- [x] CNN - 6 notebooks + exercises + solutions
- [x] RNN - 7 notebooks (fundamentals through attention)
- [x] Implementations - Reference guide with build order
- [x] Experiments - Experiment ideas with templates

---

**Ready to start?** Open [Fundamentals](./fundamentals/) and begin with notebook 01.
No prerequisites. No prior knowledge needed. Just curiosity.
