# 0️⃣ Neural Networks Fundamentals

## Overview

This section covers the foundational concepts of neural networks, providing essential prerequisites for understanding modern deep learning architectures and Large Language Models. Topics include feedforward neural networks, convolutional neural networks (CNNs), and recurrent neural networks (RNNs).

**Why Start Here?**
Understanding these fundamentals is crucial before diving into transformers and LLMs. The concepts learned here—backpropagation, optimization, architectural patterns—form the building blocks of all modern deep learning systems.

---

## 📚 Key Concepts

### Feedforward Neural Networks
- Perceptrons and multi-layer networks
- Activation functions (ReLU, sigmoid, tanh, GELU)
- Backpropagation and gradient descent
- Weight initialization strategies
- Regularization techniques (dropout, L1/L2)
- Batch normalization and layer normalization

### Convolutional Neural Networks (CNN)
- Convolutional layers and filters
- Pooling operations
- Architectural patterns (LeNet, AlexNet, VGG, ResNet)
- Transfer learning with pre-trained models
- Applications in computer vision

### Recurrent Neural Networks (RNN)
- Sequential data processing
- Vanishing and exploding gradients
- LSTM (Long Short-Term Memory) cells
- GRU (Gated Recurrent Unit) cells
- Bidirectional RNNs
- Sequence-to-sequence models

---

## 📂 Directory Structure

### [Fundamentals](./fundamentals/)
Core concepts of feedforward neural networks, including:
- Network architecture basics
- Activation functions
- Loss functions and optimization
- Backpropagation
- Training strategies

### [CNN](./cnn/)
Convolutional Neural Networks for spatial data:
- Convolution operations
- Pooling strategies
- Classic CNN architectures
- Modern innovations (ResNets, DenseNets)
- Practical applications

### [RNN](./rnn/)
Recurrent Neural Networks for sequential data:
- RNN fundamentals
- LSTM architecture
- GRU architecture
- Attention mechanisms in RNNs
- Sequence modeling

### [Implementations](./implementations/)
Code implementations of key concepts:
- From-scratch implementations
- Framework-based implementations (PyTorch/TensorFlow)
- Training scripts
- Model architectures

### [Experiments](./experiments/)
Practical experiments and hands-on projects:
- Toy datasets experiments
- Architecture comparisons
- Hyperparameter tuning
- Performance benchmarks

---

## 🎯 Learning Objectives

By the end of this section, you should be able to:

1. **Understand** the mathematical foundations of neural networks
2. **Implement** feedforward, convolutional, and recurrent networks from scratch
3. **Apply** appropriate architectures to different problem types
4. **Debug** common training issues (vanishing gradients, overfitting, etc.)
5. **Optimize** network performance through proper architecture design
6. **Explain** why transformers emerged as the dominant architecture

---

## 🚀 Getting Started

### Recommended Learning Path

1. **Start with Fundamentals** (./fundamentals/)
   - Understand perceptrons and MLPs
   - Master backpropagation
   - Learn optimization algorithms

2. **Progress to CNNs** (./cnn/)
   - Learn convolution operations
   - Study classic architectures
   - Understand transfer learning

3. **Explore RNNs** (./rnn/)
   - Master sequence modeling
   - Understand LSTM/GRU cells
   - Learn attention basics

4. **Practice with Implementations** (./implementations/)
   - Build networks from scratch
   - Use PyTorch/TensorFlow
   - Train on real datasets

5. **Run Experiments** (./experiments/)
   - Compare architectures
   - Tune hyperparameters
   - Analyze results

---

## 📖 Key Papers

### Foundational

- **Backpropagation**
  - [Learning representations by back-propagating errors](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) (Rumelhart et al., 1986)

- **Activation Functions**
  - [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) (Nair & Hinton, 2010)

### CNNs

- **LeNet** - [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (LeCun et al., 1998)
- **AlexNet** - [ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (Krizhevsky et al., 2012)
- **VGGNet** - [Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556) (Simonyan & Zisserman, 2014)
- **ResNet** - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He et al., 2015)

### RNNs

- **LSTM** - [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter & Schmidhuber, 1997)
- **GRU** - [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078) (Cho et al., 2014)
- **Seq2Seq** - [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) (Sutskever et al., 2014)

---

## 🛠️ Tools & Frameworks

### Deep Learning Frameworks
- **PyTorch** - Primary framework for implementations
- **TensorFlow/Keras** - Alternative framework examples
- **NumPy** - For from-scratch implementations

### Visualization
- **TensorBoard** - Training visualization
- **Matplotlib/Seaborn** - Plot results and architectures
- **Netron** - Model architecture visualization

### Datasets
- **MNIST** - Handwritten digits (classic starter)
- **CIFAR-10/100** - Image classification
- **Fashion-MNIST** - Alternative to MNIST
- **IMDB Reviews** - Text classification for RNNs

---

## 📚 Recommended Resources

### Books
- [Deep Learning](https://www.deeplearningbook.org/) by Goodfellow, Bengio, and Courville
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [Dive into Deep Learning](https://d2l.ai/)

### Courses
- [Stanford CS231n](http://cs231n.stanford.edu/) - CNNs for Visual Recognition
- [Fast.ai](https://course.fast.ai/) - Practical Deep Learning
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - Coursera

### Tutorials
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Andrej Karpathy's Neural Networks Course](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

---

## 🔗 Related Topics

- **Next Steps**: [Transformers](../01-transformers/) - Evolution from RNNs to attention-based models
- **Applications**: [Multimodal Models](../05-multimodal/) - CNNs in vision-language models
- **Evaluation**: [Metrics & Benchmarking](../06-evaluation/) - How to evaluate neural networks

---

## 💡 Why These Topics Matter

### For Transformers
- **Self-attention** builds on attention mechanisms first used in RNNs
- **Position embeddings** address the lack of recurrence in transformers
- **Layer normalization** evolved from batch normalization in CNNs

### For LLMs
- **Tokenization** connects to embedding layers from basic NNs
- **Optimization techniques** (Adam, learning rate schedules) stem from NN research
- **Regularization** (dropout) is still used in modern LLMs

### For Production ML
- **CNN backbones** are used in multimodal models (CLIP, vision transformers)
- **Transfer learning** principles apply across all architectures
- **Optimization strategies** are universal across model types

---

## 📝 Content Status

- [ ] Fundamentals section complete
- [ ] CNN section complete
- [ ] RNN section complete  
- [ ] Implementation examples
- [ ] Experimental notebooks
- [ ] Interactive visualizations

*This section is under active development. Contributions welcome!*

---

**Ready to begin?** Start with [Fundamentals](./fundamentals/) to build your neural network foundation!