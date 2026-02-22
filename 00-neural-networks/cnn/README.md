# Convolutional Neural Networks (CNNs)

> **How neural networks "see" images.**
> This section builds on the fundamentals - make sure you've completed those first.

---

## What's a CNN?

Regular neural networks treat an image as a flat list of numbers. A 224x224 photo
becomes 150,528 numbers with no sense of "this pixel is next to that pixel." That's
like reading a book by throwing all the letters into a bag and trying to understand
the story from the bag.

**CNNs are smarter.** They look at small patches of the image at a time - like reading
a book one word at a time instead of one letter at a time. A small "filter" slides
across the image looking for patterns: edges, curves, textures. Deeper layers combine
those simple patterns into complex ones: edges become shapes, shapes become objects.

```
HOW A CNN "SEES" A CAT
======================

  Raw Pixels       Layer 1          Layer 2          Layer 3
  (just dots)      (edges)          (shapes)         (objects)

  ░░▓▓▓░░░       |  /  \  -       /\    ()        🐱 = Cat!
  ░▓░░░▓░░       edges and        ear    eye       Combines all
  ▓░░░░░▓░       lines            shapes            features
  ▓░●░●░▓░       everywhere       recognized        together
  ▓░░▽░░▓░
  ░▓░░░▓░░       Simple           Medium            Complex
  ░░▓▓▓░░░       features         features          features
```

Three key ideas make this work:

1. **Local connectivity** - Each neuron only looks at a small patch (not the whole image)
2. **Parameter sharing** - The same filter is reused everywhere (learn once, detect everywhere)
3. **Translation invariance** - A cat is a cat whether it's on the left or right of the image

---

## What's In This Section?

```
LEARNING PATH
=============

  What & Why          How It Works          Real World

  01 What Are CNNs?   02 Convolution        04 Building a CNN
  (the problem &      03 Pooling            05 Famous Architectures
   the solution)      (the key operations)  06 Transfer Learning

  "Why not just       "How filters         "Build, use, and
   use regular NNs?"   and pooling work"    stand on shoulders
                                            of giants"
```

---

## Notebook Guide

### 01 - What Are CNNs?

**The motivation: why regular networks fail on images.**

This notebook starts with the "parameter explosion" problem - a regular network
for a 224x224 image would need 77 million parameters in just the first layer.
Then it introduces the three principles (local connectivity, parameter sharing,
translation invariance) that make CNNs practical.

Lots of visual comparisons between fully-connected and convolutional approaches.

**Key concepts:** receptive field, filters, feature maps

**Plain English:** Instead of every pixel talking to every neuron (chaos), each
neuron only pays attention to a small neighborhood of pixels. Like looking
through a magnifying glass that you slide across the image.

---

### 02 - Convolution Operation

**The core operation that gives CNNs their name.**

"Convolution" sounds intimidating but it's simple: take a small grid of numbers
(a "filter"), slide it across the image, and at each position multiply-and-add.
That's it. This notebook implements it from scratch so you see every step.

Different filters detect different things:
- `[-1, 0, 1]` detects vertical edges
- `[-1, -1, -1, 0, 0, 0, 1, 1, 1]` detects horizontal edges

**Key concepts:** filters/kernels, stride, padding, feature maps, output dimensions

**Plain English:** A filter is like a stencil. You hold it up to different parts
of the image and ask "does this part of the image match my stencil?" The output
tells you where matches were found.

---

### 03 - Pooling Layers

**Shrinking the image while keeping what matters.**

After convolution, you have feature maps that are almost as big as the original
image. Pooling shrinks them down. The most common type, "max pooling," looks at
each 2x2 block and keeps only the biggest value.

**Key concepts:** max pooling, average pooling, downsampling, translation invariance

**Plain English:** Imagine summarizing a page of text into bullet points. You
keep the most important information and throw away the details. Pooling does
the same thing for image features.

---

### 04 - Building a Complete CNN

**Put all the pieces together into a working classifier.**

This is the payoff notebook. You stack convolution layers and pooling layers
together, add a regular fully-connected layer at the end, and train the whole
thing to classify images. Architecture: Conv -> Pool -> Conv -> Pool -> FC -> Output.

**What you'll build:** An image classifier trained on MNIST/Fashion-MNIST

---

### 05 - Famous Architectures

**The greatest hits of CNN history.**

A tour of the architectures that pushed the field forward:

| Architecture | Year | Key Innovation |
|---|---|---|
| **LeNet** | 1998 | First practical CNN (handwriting recognition) |
| **AlexNet** | 2012 | Proved deep learning works on large-scale images |
| **VGG** | 2014 | Showed that deeper = better (with small 3x3 filters) |
| **ResNet** | 2015 | Skip connections let you train 100+ layer networks |

Each architecture is explained with diagrams and context for why it mattered.

---

### 06 - Transfer Learning

**Don't train from scratch - start with someone else's work.**

Training a CNN from scratch requires millions of images and days of compute.
Transfer learning lets you take a network trained on ImageNet (14 million images)
and adapt it to your problem with just a few hundred examples.

Two strategies covered:
- **Feature extraction** - Use the pre-trained network as a fixed feature detector
- **Fine-tuning** - Unfreeze some layers and retrain them on your data

**Plain English:** Imagine you learned to draw in pencil. When you pick up a pen,
you don't start from zero - your pencil skills transfer. The same idea applies
to neural networks trained on one task and adapted to another.

---

## Exercises

`exercises.ipynb` has 10 progressively harder challenges:

1. Custom convolution filters (edge detection, emboss, sharpen)
2. Output dimension calculations
3. Build a 3-layer CNN from scratch with NumPy
4. Visualize learned filters and feature maps
5. Compare max pooling vs average pooling
6. Implement data augmentation (flips, rotations, crops)
7. Build a simplified ResNet with skip connections
8. Design transfer learning strategies for different scenarios
9. Translate NumPy CNN to PyTorch
10. Debug a broken CNN implementation (find 5 bugs)

`solutions.ipynb` has complete worked solutions with explanations.

---

## Prerequisites

Before starting this section, you should have completed:

- **All 10 fundamentals notebooks** (especially forward propagation, loss functions, backpropagation)
- Comfort with NumPy array operations and matrix multiplication
- Understanding of how training works (epochs, batches, gradient descent)

## Setup

```bash
pip install numpy matplotlib scipy
pip install torch  # needed for exercises 9 and transfer learning
```

---

## Key Terms for This Section

| Term | Meaning |
|------|---------|
| **Filter / Kernel** | A small grid of weights (e.g. 3x3) that slides across the image to detect patterns |
| **Feature map** | The output of applying a filter to an image - shows where patterns were found |
| **Stride** | How many pixels the filter moves each step (stride=1 means one pixel at a time) |
| **Padding** | Adding zeros around the image border so the filter can process edge pixels |
| **Pooling** | Shrinking the feature map by summarizing small regions (usually 2x2) |
| **Receptive field** | The region of the original image that one neuron can "see" |
| **Channel** | One "layer" of an image - RGB images have 3 channels (red, green, blue) |

---

[Previous: Fundamentals](../fundamentals/README.md) | [Back to Neural Networks](../README.md) | [Next: RNN](../rnn/README.md)
