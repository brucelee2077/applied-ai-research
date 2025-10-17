# 🚀 Getting Started

Welcome to the Applied AI Research repository! This guide will help you get started with exploring and using this documentation.

## Overview

This repository serves as a comprehensive learning resource for Large Language Model engineering, starting from foundational neural network concepts and progressing through to production deployment strategies. The curriculum is designed to build knowledge systematically, beginning with essential deep learning fundamentals.

## Prerequisites

### Knowledge Requirements

**Foundational:**
- Python programming
- Basic linear algebra and calculus
- Understanding of machine learning concepts
- Basic probability and statistics

**Recommended:**
- Experience with NumPy and Python scientific computing
- Understanding of optimization algorithms
- Familiarity with Jupyter notebooks
- Git and version control

### Technical Setup

**Required Software:**
- Python 3.8 or higher
- Git
- Jupyter Notebook or JupyterLab
- Code editor (VS Code, PyCharm, etc.)

**Recommended Tools:**
- CUDA-capable GPU (for running experiments)
- Docker (for containerized environments)
- Virtual environment manager (venv, conda, poetry)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ruifengli-cs/applied-ai-research.git
cd applied-ai-research
```

### 2. Set Up Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n llm-journey python=3.10
conda activate llm-journey
```

### 3. Install Dependencies

```bash
# Core dependencies (to be added as project grows)
pip install torch transformers datasets
pip install jupyter notebook
pip install numpy pandas matplotlib seaborn
```

## Repository Structure

```
applied-ai-research/
├── 00-neural-networks/     # NN fundamentals (NN, CNN, RNN)
├── 01-transformers/        # Transformer architecture fundamentals
├── 02-fine-tuning/         # Model adaptation techniques
├── 03-rag/                 # Retrieval-augmented generation
├── 04-prompt-engineering/  # Effective prompting strategies
├── 05-multimodal/          # Cross-modal models
├── 06-evaluation/          # Metrics and benchmarking
├── 07-deployment/          # Production deployment
├── papers/                 # Research paper summaries
├── projects/               # Complete project implementations
├── notebooks/              # Jupyter experiments
└── docs/                   # Documentation (you are here)
```

## Learning Paths

### Path 1: Complete Beginner Track

**Objective:** Build foundational understanding from scratch to LLMs

1. **Week 1-2:** [Neural Networks](../00-neural-networks/)
   - Master feedforward neural networks
   - Understand backpropagation and optimization
   - Learn activation functions and initialization

2. **Week 3-4:** [CNN Fundamentals](../00-neural-networks/)
   - Study convolutional layers and pooling
   - Understand image processing with CNNs
   - Explore architectural patterns (ResNet, VGG)

3. **Week 5-6:** [RNN Fundamentals](../00-neural-networks/)
   - Learn recurrent architectures
   - Master LSTM and GRU cells
   - Understand sequence modeling

4. **Week 7-8:** [Transformers](../01-transformers/)
   - Study attention mechanisms
   - Understand positional encoding
   - Learn multi-head attention

5. **Week 9-10:** [Prompt Engineering](../04-prompt-engineering/)
   - Master prompting techniques
   - Explore few-shot learning
   - Practice with templates

6. **Week 11-12:** [Evaluation](../06-evaluation/)
   - Learn evaluation metrics
   - Understand benchmarks
   - Practice model assessment

### Path 2: Intermediate Track (Some NN Background)

**Objective:** Transition from basic deep learning to LLM expertise

1. **Week 1:** Review [Neural Networks](../00-neural-networks/) (CNN, RNN)
2. **Week 2-3:** [Transformers](../01-transformers/)
3. **Week 4-5:** [Fine-Tuning](../02-fine-tuning/)
4. **Week 6-7:** [RAG Systems](../03-rag/)
5. **Week 8-9:** [Deployment](../07-deployment/)

### Path 3: Advanced Track (Strong NN/Transformer Background)

**Objective:** Master cutting-edge techniques and production systems

1. **Week 1:** Quick review and fill gaps
2. **Week 2-3:** [Multimodal Models](../05-multimodal/)
3. **Week 4-5:** Research paper implementations
4. **Week 6-8:** Custom end-to-end projects
5. **Ongoing:** Production system deployment

## How to Use This Repository

### For Self-Study

1. **Start with fundamentals** in [Neural Networks](../00-neural-networks/)
2. **Progress systematically** through each topic directory
3. **Work through notebooks** for hands-on experience
4. **Implement papers** to deepen understanding
5. **Build projects** to apply knowledge

### For Reference

1. **Search for specific topics** in the directory structure
2. **Use the glossary** in [Terminology](./terminology.md)
3. **Check best practices** for implementation guidance
4. **Review resources** for additional learning materials

### For Practice

1. **Run notebooks** in the notebooks/ directory
2. **Complete experiments** in each topic area
3. **Build projects** from scratch
4. **Contribute implementations** of new techniques

## Next Steps

### Immediate Actions

1. ✅ Complete this getting started guide
2. ✅ Set up your development environment
3. ✅ Review [Terminology](./terminology.md) for key concepts
4. ✅ Choose a learning path based on your level
5. ✅ Start with [Neural Networks](../00-neural-networks/) basics

### Continuous Learning

- **Daily:** Read documentation and papers
- **Weekly:** Implement new techniques and experiments
- **Monthly:** Complete projects and contribute back
- **Ongoing:** Engage with the ML community

## Common Workflows

### Running Notebooks

```bash
jupyter notebook
# Navigate to notebooks/ directory
# Open desired notebook
```

### Experimenting with Code

```bash
# Navigate to relevant topic directory
cd 00-neural-networks/experiments/

# Run experiments
python experiment_name.py
```

### Contributing

```bash
# Create feature branch
git checkout -b feature/my-contribution

# Make changes
# ...

# Commit and push
git add .
git commit -m "Add: description of changes"
git push origin feature/my-contribution

# Open pull request on GitHub
```

## Getting Help

### Documentation Resources

- **Topic READMEs:** Each directory has detailed documentation
- **Code Comments:** Implementations include comprehensive comments
- **Notebooks:** Interactive tutorials with explanations

### Community Support

- **Issues:** Report bugs or ask questions via GitHub issues
- **Discussions:** Engage with others in discussions
- **Pull Requests:** Contribute improvements and fixes

### External Resources

See [Resources](./resources.md) for curated learning materials, courses, and communities.

## Tips for Success

1. **Start with Fundamentals:** Master NN, CNN, RNN before moving to transformers
2. **Practice Regularly:** Consistent practice is key
3. **Build Projects:** Apply knowledge to real problems
4. **Read Papers:** Stay current with research
5. **Contribute:** Share your learning with others
6. **Ask Questions:** Don't hesitate to seek help
7. **Document:** Keep notes on what you learn
8. **Experiment:** Try new approaches and techniques

## Troubleshooting

### Common Issues

**Issue: CUDA/GPU not available**
- Solution: Install CUDA toolkit and appropriate PyTorch version
- Alternative: Use CPU for smaller experiments or cloud GPUs

**Issue: Out of memory errors**
- Solution: Reduce batch size, use gradient accumulation, or smaller models
- Alternative: Use cloud compute with more memory

**Issue: Dependency conflicts**
- Solution: Use virtual environments and pin package versions
- Alternative: Use Docker containers for isolated environments

## Additional Resources

- **[Best Practices](./best-practices.md)** - Development guidelines
- **[Terminology](./terminology.md)** - Key concepts and definitions
- **[Resources](./resources.md)** - External learning materials
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute

---

*Ready to begin your LLM engineering journey? Start with [Neural Networks Fundamentals](../00-neural-networks/)!*