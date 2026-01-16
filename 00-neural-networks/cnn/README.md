# CNN Exercises and Solutions

This directory contains comprehensive exercises and solutions for Convolutional Neural Networks (CNNs).

## Files Created

### 📓 exercises.ipynb (50KB)
Comprehensive practice exercises covering:

1. **Exercise 1: Custom Convolution Filters**
   - Implement edge detection filters (horizontal, vertical, Sobel)
   - Create emboss and sharpen filters
   - Apply filters to images and visualize results

2. **Exercise 2: Calculate Output Dimensions**
   - Master convolution output size formula
   - Calculate "same" padding for different kernel sizes
   - Trace through complete CNN architecture

3. **Exercise 3: Build 3-Layer CNN from Scratch**
   - Implement ConvLayer class using NumPy
   - Implement MaxPoolLayer class
   - Build complete CNN for CIFAR-10

4. **Exercise 4: Visualize Learned Filters**
   - Generate and visualize filter examples
   - Apply filters and visualize feature maps
   - Understand what CNNs learn at different layers

5. **Exercise 5: Compare Pooling Methods**
   - Implement average pooling
   - Compare max pooling vs average pooling
   - Analyze when to use each method

6. **Exercise 6: Data Augmentation**
   - Implement basic augmentations (flip, rotation, crop)
   - Adjust brightness and contrast
   - Create augmentation pipeline

7. **Exercise 7: Simplified ResNet**
   - Implement residual blocks with skip connections
   - Build mini ResNet architecture
   - Understand why skip connections help training

8. **Exercise 8: Transfer Learning**
   - Understand feature extraction vs fine-tuning
   - Design transfer learning strategies for different scenarios
   - Analyze when to use pre-trained models

9. **Exercise 9: PyTorch Translation**
   - Translate NumPy CNN to PyTorch
   - Implement training loop with autograd
   - Compare framework implementations

10. **Exercise 10: Debug Broken CNN**
    - Find and fix 5 bugs in CNN implementation
    - Write unit tests to catch bugs
    - Develop debugging skills

### 📗 solutions.ipynb (51KB)
Complete solutions with:

- **Working Code:** Fully implemented solutions with detailed comments
- **Explanations:** Step-by-step approach for each exercise
- **Visualizations:** Plots and diagrams showing results
- **Common Mistakes:** What to avoid and why
- **Extensions:** Variations and advanced topics to explore
- **Key Insights:** Deep understanding of CNN concepts

## Quality Features

Both notebooks match the style and quality of the fundamentals series:

✅ **Comprehensive Coverage:** 10 exercises from basic to advanced
✅ **Educational Focus:** Explanations, not just code
✅ **Progressive Difficulty:** Starts simple, builds to complex
✅ **Practical Examples:** Real implementations you can run
✅ **Hints and Tips:** Collapsible hints for each exercise
✅ **Visualizations:** Plots and diagrams throughout
✅ **Best Practices:** Modern CNN techniques and patterns
✅ **Debugging Skills:** Learn to find and fix errors
✅ **Framework Integration:** Bridge to PyTorch/TensorFlow

## Exercise Format

Each exercise includes:
- Clear instructions and objectives
- Starter code with TODOs
- Collapsible hints
- Expected output descriptions
- Reflection questions

## Solution Format

Each solution includes:
- Complete working code
- Detailed explanations
- Step-by-step calculations
- Visualizations of results
- Common mistakes to avoid
- Alternative approaches
- Extensions to try

## Topics Covered

### Core CNN Concepts
- Convolution operations
- Filters and feature detection
- Padding and stride
- Output dimension calculations
- Pooling layers (max, average)

### Implementation Skills
- Building layers from scratch (NumPy)
- Complete CNN architecture
- Forward propagation
- Shape management and debugging

### Advanced Topics
- Residual connections (ResNet)
- Data augmentation techniques
- Transfer learning strategies
- Framework translation (NumPy to PyTorch)

### Practical Skills
- Visualizing filters and features
- Comparing different approaches
- Debugging CNN implementations
- Writing tests for neural networks

## Learning Path

1. **Start with exercises.ipynb**
   - Read each exercise carefully
   - Try to solve independently
   - Use hints if stuck
   - Test your implementations

2. **Check solutions.ipynb**
   - Compare your solution
   - Read the explanations
   - Study the visualizations
   - Try the extensions

3. **Practice and Experiment**
   - Modify the code
   - Try different parameters
   - Apply to your own data
   - Build your own projects

## Prerequisites

- Basic Python and NumPy
- Understanding of neural network fundamentals
- Completed the fundamentals series notebooks
- Familiarity with matrix operations

## Dependencies

```python
numpy
matplotlib
scipy
torch (optional, for Exercise 9)
```

## Key Concepts

### Convolution
- Local connectivity
- Weight sharing
- Translation equivariance
- Feature detection

### Pooling
- Dimensionality reduction
- Translation invariance
- Max vs average pooling

### Architecture Design
- Progressive downsampling
- Channel expansion
- Receptive field growth
- Parameter efficiency

### Modern Techniques
- Skip connections (ResNet)
- Data augmentation
- Transfer learning
- Batch normalization

## Tips for Success

1. **Implement from Scratch First**
   - Understand the internals
   - Build intuition
   - Then use frameworks

2. **Visualize Everything**
   - Filters and feature maps
   - Activations and gradients
   - Architecture diagrams

3. **Check Dimensions**
   - Verify shapes at each layer
   - Use assertions in code
   - Calculate expected sizes

4. **Start Simple**
   - Small networks first
   - Add complexity gradually
   - Debug incrementally

5. **Test Thoroughly**
   - Unit tests for layers
   - Check edge cases
   - Verify with known examples

## Next Steps

After completing these exercises:

- **Study Famous Architectures:** AlexNet, VGG, ResNet, Inception
- **Read Papers:** Understand cutting-edge techniques
- **Build Projects:** Apply CNNs to real problems
- **Explore Advanced Topics:** Attention, Transformers, GANs
- **Contribute:** Share your knowledge with others

## Resources

- **Papers:** Check solutions.ipynb for recommended papers
- **Courses:** Stanford CS231n, Fast.ai
- **Datasets:** CIFAR-10, ImageNet, COCO
- **Frameworks:** PyTorch, TensorFlow documentation

## Notes

These exercises are designed to:
- Deepen understanding of CNNs
- Bridge theory and practice
- Prepare for real-world applications
- Develop debugging skills
- Build confidence with frameworks

Happy learning\! 🚀
