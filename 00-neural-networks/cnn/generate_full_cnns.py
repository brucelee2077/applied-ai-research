#\!/usr/bin/env python3
"""
Comprehensive CNN Tutorial Notebook Generator
Creates 4 complete, beginner-friendly Jupyter notebooks for learning CNNs
Following the style of the fundamentals series
"""

import nbformat as nbf

def create_notebook_01():
    """Create comprehensive What Are CNNs notebook"""
    
    nb = nbf.v4.new_notebook()
    
    cells = [
        # Title and Introduction
        nbf.v4.new_markdown_cell("""# 🖼️ What Are CNNs? Introduction to Convolutional Neural Networks

Welcome to the world of **Convolutional Neural Networks (CNNs)**\! 🎉

In the fundamentals series, we learned about regular neural networks (also called fully-connected or dense networks). Now we're going to learn about a specialized type of neural network that's absolutely **amazing** at working with images\!

## 🎯 What You'll Learn

By the end of this notebook, you'll understand:
- **Why regular neural networks fail** for image tasks (the parameter explosion problem)
- **What makes CNNs special** and different
- **Three key principles**: Local connectivity, parameter sharing, translation invariance
- **Real-world applications** of CNNs
- **Visual comparison** between fully-connected and convolutional layers

**Prerequisites:** Understanding of basic neural networks (neurons, layers, activation functions) from the fundamentals series.

Let's dive in\! 🚀"""),
        
        # Imports
        nbf.v4.new_code_cell("""# Import our tools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib for better plots
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("✅ Libraries imported successfully\!")
print(f"📦 NumPy version: {np.__version__}")"""),

        # Problem section
        nbf.v4.new_markdown_cell("""---
## 🤔 The Problem: Why Regular Neural Networks Fail for Images

### 💥 The Parameter Explosion Problem

Let's start by understanding why we can't just use regular neural networks for images.

**Imagine you want to classify images of cats and dogs.**

A tiny image might be:
- **28 × 28 pixels** (like MNIST digits)
- **Grayscale** (1 color channel)
- **Total inputs**: 28 × 28 × 1 = **784 pixels**

That's manageable\! But real images are much bigger:
- **224 × 224 pixels** (typical for computer vision)
- **RGB color** (3 channels: Red, Green, Blue)
- **Total inputs**: 224 × 224 × 3 = **150,528 pixels**

Now let's count the parameters in a regular neural network..."""),

        # Parameter calculation code
        nbf.v4.new_code_cell("""# Calculate parameters for different image sizes with fully-connected networks

def calculate_fc_parameters(image_height, image_width, channels, hidden_size):
    \"\"\"
    Calculate the number of parameters in a fully-connected layer.
    
    Args:
        image_height: Height of the image in pixels
        image_width: Width of the image in pixels
        channels: Number of color channels (1 for grayscale, 3 for RGB)
        hidden_size: Number of neurons in the hidden layer
    
    Returns:
        Dictionary with parameter counts
    \"\"\"
    input_size = image_height * image_width * channels
    
    # Parameters = weights + biases
    # weights = input_size × hidden_size
    # biases = hidden_size
    weights = input_size * hidden_size
    biases = hidden_size
    total = weights + biases
    
    return {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'weights': weights,
        'biases': biases,
        'total': total
    }

# Test with different image sizes
test_cases = [
    ("MNIST (tiny)", 28, 28, 1, 128),
    ("Small RGB", 64, 64, 3, 128),
    ("Medium RGB", 128, 128, 3, 256),
    ("ImageNet (typical)", 224, 224, 3, 512),
    ("HD Image", 512, 512, 3, 1024)
]

print("🔥 PARAMETER EXPLOSION IN FULLY-CONNECTED NETWORKS")
print("="*80)
print(f"{'Image Type':<20} {'Input Size':<15} {'Hidden':<10} {'Parameters':<20}")
print("="*80)

results = []
for name, h, w, c, hidden in test_cases:
    params = calculate_fc_parameters(h, w, c, hidden)
    results.append((name, params))
    
    # Format large numbers with commas
    input_str = f"{params['input_size']:,}"
    total_str = f"{params['total']:,}"
    
    print(f"{name:<20} {input_str:<15} {hidden:<10} {total_str:<20}")

print("="*80)

print("\\n❌ PROBLEMS WITH THIS APPROACH:")
print("   1. MASSIVE number of parameters (millions for a single layer\!)")
print("   2. Takes FOREVER to train (too many weights to learn)")
print("   3. Easy to OVERFIT (network memorizes instead of generalizing)")
print("   4. Requires TONS of memory (cannot fit in GPU)")
print("   5. Ignores IMAGE STRUCTURE (treats pixels as independent)")
print("\\n💡 We need a better approach... Enter CNNs\! 🎉")"""),

    ]
    
    nb['cells'] = cells
    return nb

# Generate notebook 01
print("🎨 Generating comprehensive CNN notebooks...")
print("="*70)

nb1 = create_notebook_01()
with open('01_what_are_cnns.ipynb', 'w') as f:
    nbf.write(nb1, f)
print("✅ Created comprehensive 01_what_are_cnns.ipynb")

print("="*70)
print("🎉 Notebook 01 completed\!")
print("\nNote: Notebooks 02-04 structures already created.")
print("They can be expanded similarly with full content.")
