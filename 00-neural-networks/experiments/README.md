# Neural Network Experiments

## Overview

This directory contains experimental notebooks and scripts exploring various aspects of neural networks. Each experiment is designed to provide hands-on learning and practical insights.

## Experiment Categories

### Fundamentals Experiments
- **Activation Function Comparison** - Visual comparison of different activation functions
- **Learning Rate Impact** - How learning rate affects convergence
- **Batch Size Effects** - Trading off between speed and generalization
- **Initialization Strategies** - Xavier vs. He initialization
- **Overfitting vs. Regularization** - Demonstrating dropout and L2 effects

### CNN Experiments
- **Filter Visualization** - What CNN filters learn
- **Architecture Comparison** - LeNet vs. ResNet performance
- **Transfer Learning** - Fine-tuning pre-trained models
- **Data Augmentation Impact** - Effect on model robustness
- **Receptive Field Analysis** - Understanding what CNNs see

### RNN Experiments
- **Vanishing Gradient Demo** - Visualizing the problem
- **LSTM vs. GRU** - Performance and efficiency comparison
- **Sequence Length Impact** - How sequence length affects training
- **Bidirectional RNNs** - Advantages of processing both directions
- **Attention Mechanisms** - Attention in RNN context

### Comparative Studies
- **CNN vs. RNN for sequences** - When to use which
- **Depth vs. Width** - Network architecture tradeoffs
- **Optimization Algorithms** - SGD vs. Adam vs. AdamW
- **Batch vs. Layer Normalization** - Impact on training

## Running Experiments

### Jupyter Notebooks
```bash
jupyter notebook
# Navigate to desired experiment notebook
```

### Standalone Scripts
```bash
python experiment_name.py --config config.yaml
```

## Experiment Structure

Each experiment typically includes:
1. **Objective** - What we're trying to learn
2. **Setup** - Dataset, model, hyperparameters
3. **Execution** - Training and evaluation
4. **Analysis** - Results visualization and interpretation
5. **Conclusions** - Key takeaways

## Content to be Added

- [ ] Interactive notebooks for each experiment type
- [ ] Visualization utilities
- [ ] Experiment tracking integration (W&B)
- [ ] Automated hyperparameter sweeps
- [ ] Results comparison dashboard
- [ ] Reproducibility guides

## Best Practices

- Always set random seeds for reproducibility
- Document hyperparameters clearly
- Save experiment results and logs
- Include visualizations
- Provide clear conclusions

---

[Back to Neural Networks](../README.md)