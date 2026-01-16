# Claude Code Workflows for ML Development

## Overview

This guide provides battle-tested workflows for using Claude Code in machine learning and AI research projects. Each workflow includes step-by-step instructions, example prompts, and tips for maximum productivity.

### 📑 Contents

- [Setup & Installation](#setup--installation)
- **[Sub-Agents Guide](./sub-agents-guide.md)** - Learn to use parallel sub-agents for complex tasks
- [Workflow Templates](#workflow-templates)
- [Quick Reference](#quick-reference)

---

## 🚀 Setup & Installation

### Initial Setup

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Authenticate with your API key
claude-code auth

# Initialize in your project
cd your-ml-project
claude-code init
```

### Project Configuration

Create `.clauderc` in your project root:

```json
{
  "ignore": [
    "**/__pycache__",
    "**/.ipynb_checkpoints",
    "**/data/**",
    "**/checkpoints/**",
    "**/*.pt",
    "**/*.pth",
    "**/.git"
  ],
  "context": {
    "always_include": [
      "README.md",
      "requirements.txt",
      "config.yaml"
    ]
  }
}
```

---

## 📋 Workflow Templates

## Workflow 1: Building a Neural Network from Scratch

**Time Estimate**: 30-60 minutes
**Use Case**: Implementing custom architectures

### Step 1: Define Architecture
```
Prompt: "I want to build a ResNet-style architecture for CIFAR-10.
Requirements:
- 3 residual blocks with [16, 32, 64] channels
- Batch normalization after each conv layer
- Global average pooling before classifier
- 10 output classes

Start by implementing the residual block."
```

### Step 2: Verify Block Implementation
```
Prompt: "Create a test script that verifies:
1. Input/output shapes are correct
2. Skip connection works properly
3. Gradient flows through all layers
Use a dummy input of shape (2, 3, 32, 32)"
```

### Step 3: Build Full Network
```
Prompt: "Now implement the full ResNet using 3 of these blocks.
Add:
- Initial conv layer (3->16 channels)
- Three residual blocks with downsampling
- Global average pooling
- Final linear layer
Show me the total parameter count."
```

### Step 4: Add Training Components
```
Prompt: "Add data loading for CIFAR-10 with:
- Train/val split (90/10)
- Augmentation: random crop, horizontal flip
- Normalization with CIFAR-10 stats
- Batch size 128, 4 workers"
```

### Step 5: Implement Training Loop
```
Prompt: "Create training loop with:
- Cross entropy loss
- SGD with momentum 0.9
- Cosine annealing learning rate (0.1 -> 0.001)
- Gradient clipping at 1.0
- Checkpoint saving every 10 epochs
- Validation every epoch with accuracy metric"
```

### Step 6: Run and Debug
```
Prompt: "Run training for 5 epochs to verify everything works.
If you see issues, debug and fix them."
```

**Expected Output**: Working CNN training pipeline

---

## Workflow 2: Implementing a Research Paper

**Time Estimate**: 2-4 hours
**Use Case**: Reproducing papers, staying current with research

### Step 1: Paper Understanding
```
Prompt: "I want to implement the Vision Transformer (ViT) paper.
First, let me share the key architectural details:
- Image is split into 16x16 patches
- Patches are linearly embedded
- Position embeddings added
- Standard transformer encoder
- Classification head on [CLS] token

Can you outline the implementation plan?"
```

### Step 2: Implement Core Components
```
Prompt: "Let's start with patch embedding.
Implement a module that:
1. Takes image (B, C, H, W)
2. Splits into patches of 16x16
3. Flattens and projects to embedding dim D
4. Output shape: (B, num_patches, D)"
```

```
Prompt: "Now implement positional embeddings:
- Learnable embeddings for each patch position
- Include [CLS] token prepended to sequence
- Should be added to patch embeddings"
```

```
Prompt: "Implement the transformer encoder block:
- Multi-head self attention (12 heads)
- MLP with GELU activation (expand 4x then project back)
- Layer norm before each sub-layer (pre-norm)
- Residual connections"
```

### Step 3: Combine into Model
```
Prompt: "Create the complete ViT model:
- Patch + position embeddings
- Stack of N transformer blocks
- Layer norm after final block
- Classification head (MLP head on [CLS] token)

Use config: patches=16x16, dim=768, depth=12, heads=12, mlp_dim=3072"
```

### Step 4: Verify Against Paper
```
Prompt: "Create a verification script that:
1. Counts parameters (should be ~86M for ViT-Base)
2. Tests forward pass with ImageNet image (224x224)
3. Verifies intermediate shapes match paper
4. Checks attention map dimensions"
```

### Step 5: Training Setup
```
Prompt: "Set up training following the paper:
- Optimizer: Adam (β1=0.9, β2=0.999)
- Learning rate: 3e-3 with linear warmup (10k steps)
- Weight decay: 0.3
- Batch size: 4096 (accumulate gradients if needed)
- Data: ImageNet with RandAugment
- Training: 300 epochs"
```

### Step 6: Ablation Studies
```
Prompt: "Create variants for ablation study:
1. ViT without position embeddings
2. ViT with relative position bias
3. ViT with different patch sizes (8x8, 32x32)

Set up training scripts for each variant."
```

**Expected Output**: Complete paper implementation with ablations

---

## Workflow 3: Debugging Training Issues

**Time Estimate**: 20-60 minutes
**Use Case**: Model not converging, NaN losses, etc.

### Diagnostic Workflow

#### Issue: Loss is NaN

```
Prompt: "My training loss becomes NaN after ~100 iterations.
Here's my training code: [share relevant code]

Can you:
1. Identify potential causes
2. Add diagnostic logging
3. Suggest fixes in order of likelihood"
```

**AI will typically check:**
- Learning rate too high
- Gradient explosion
- Numerical instability in loss
- Data preprocessing issues
- Bad weight initialization

```
Prompt: "Add gradient norm logging and clipping.
Also add assertions to catch NaN/Inf in:
- Model outputs
- Loss computation
- Gradients"
```

#### Issue: Model Not Learning

```
Prompt: "My model is stuck at random chance accuracy.
Loss decreases but validation accuracy stays at 10% (10 classes).

Analyze my:
- Model architecture
- Data loading pipeline
- Loss function
- Optimizer settings"
```

**AI will check:**
- Data labels match model outputs
- Model has enough capacity
- Learning rate appropriate
- Data normalization correct
- No bugs in evaluation

#### Issue: Overfitting

```
Prompt: "Training accuracy is 98% but validation is 65%.
Add regularization:
- Dropout (try rates 0.1, 0.3, 0.5)
- Weight decay (try 1e-4, 1e-3)
- Data augmentation (aggressive)
- Label smoothing (0.1)

Set up experiment to compare all combinations."
```

#### Issue: Slow Training

```
Prompt: "Training is too slow (2 min/epoch on GPU).
Profile and optimize:
1. Check for CPU bottlenecks in data loading
2. Verify GPU utilization
3. Suggest optimizations (mixed precision, compiled model, etc.)
4. Implement top 3 fixes"
```

---

## Workflow 4: Data Pipeline Development

**Time Estimate**: 30-45 minutes
**Use Case**: Custom datasets, complex preprocessing

### Step 1: Dataset Class
```
Prompt: "Create PyTorch Dataset for medical images:
- Images in folders: data/train/class_name/*.png
- CSV with metadata: patient_id, age, diagnosis
- Load image and metadata together
- Handle missing images gracefully"
```

### Step 2: Augmentation Pipeline
```
Prompt: "Add augmentation using albumentations:
Train:
- Random crop 224x224
- Horizontal flip (p=0.5)
- Random brightness/contrast
- Coarse dropout
- Normalize with ImageNet stats

Val:
- Center crop 224x224
- Normalize only"
```

### Step 3: Data Loader
```
Prompt: "Create data loaders with:
- Batch size 32
- 4 workers
- Pin memory for GPU
- Custom collate function to handle variable-size metadata
- Balanced sampling (equal samples per class)"
```

### Step 4: Validation
```
Prompt: "Create visualization script:
1. Show batch of augmented images
2. Plot class distribution
3. Check data stats (mean, std per channel)
4. Verify labels are correct"
```

### Step 5: Optimization
```
Prompt: "Profile data loading and optimize:
- Add prefetching
- Cache preprocessed images if needed
- Suggest better num_workers based on system
- Add progress bars for loading"
```

---

## Workflow 5: Experiment Management

**Time Estimate**: 45-60 minutes
**Use Case**: Tracking multiple experiments, hyperparameter tuning

### Step 1: Configuration System
```
Prompt: "Create config system using hydra:
- config/model/: resnet18.yaml, resnet50.yaml, vit.yaml
- config/data/: cifar10.yaml, imagenet.yaml
- config/training/: default.yaml, fast_dev.yaml
- config/config.yaml: main config composing others

Set up so I can run experiments like:
python train.py model=resnet50 data=imagenet"
```

### Step 2: Logging Integration
```
Prompt: "Integrate Weights & Biases:
- Log hyperparameters
- Log metrics every epoch (train/val loss, accuracy)
- Log learning rate schedule
- Save model checkpoints as artifacts
- Log example predictions every 10 epochs
- Log gradient norms"
```

### Step 3: Checkpoint Management
```
Prompt: "Implement checkpointing:
- Save every N epochs
- Keep only best K checkpoints (by val accuracy)
- Save optimizer state for resuming
- Add resume from checkpoint functionality
- Handle distributed training checkpoints"
```

### Step 4: Hyperparameter Sweep
```
Prompt: "Set up sweep for W&B:
Parameters to tune:
- Learning rate: [1e-4, 3e-4, 1e-3]
- Weight decay: [1e-4, 1e-3, 1e-2]
- Batch size: [32, 64, 128]
- Optimizer: [Adam, AdamW, SGD]

Create sweep config and launch script."
```

### Step 5: Results Analysis
```
Prompt: "Create analysis notebook that:
1. Loads sweep results from W&B
2. Plots hyperparameter importance
3. Shows best runs and their configs
4. Compares learning curves
5. Generates summary table"
```

---

## Workflow 6: Model Optimization & Deployment

**Time Estimate**: 1-2 hours
**Use Case**: Moving from research to production

### Step 1: Model Export
```
Prompt: "Convert trained model to production format:
1. Export to TorchScript
2. Export to ONNX
3. Verify outputs match for both formats
4. Benchmark inference speed"
```

### Step 2: Quantization
```
Prompt: "Apply post-training quantization:
- Dynamic quantization (for CPU)
- Static quantization (with calibration)
- Compare accuracy vs speed tradeoff
- Show memory savings"
```

### Step 3: Inference Optimization
```
Prompt: "Optimize inference:
- Batch inference for throughput
- TensorRT conversion for GPU
- ONNX Runtime optimizations
- Add warmup routine
- Profile and benchmark all versions"
```

### Step 4: API Service
```
Prompt: "Create FastAPI service:
- /predict endpoint taking image
- Preprocessing in service
- Return class probabilities
- Add request validation
- Include health check endpoint
- Add proper error handling"
```

### Step 5: Docker Deployment
```
Prompt: "Create production Dockerfile:
- Multi-stage build
- Minimal base image
- Include only inference dependencies
- Set up proper logging
- Add health checks
- Optimize image size"
```

### Step 6: Testing
```
Prompt: "Create test suite:
1. Unit tests for preprocessing
2. Model inference tests
3. API endpoint tests
4. Load testing script
5. Integration tests
Add to CI/CD pipeline."
```

---

## 🎯 Quick Reference: Common Tasks

### Quick Task: Add Feature
```
Prompt: "Add [feature] to my [component].
Current code: [paste or reference file]
Requirements: [specific requirements]"
```

### Quick Task: Refactor
```
Prompt: "Refactor [file/function] to:
- Improve readability
- Add type hints
- Extract common patterns
- Add docstrings"
```

### Quick Task: Debug
```
Prompt: "This code has error: [error message]
Code: [relevant code]
Expected behavior: [description]
Help me fix it."
```

### Quick Task: Optimize
```
Prompt: "Profile this code and optimize the bottlenecks:
[code]
Target: Reduce execution time by 2x"
```

### Quick Task: Test
```
Prompt: "Write pytest tests for:
[code or module]
Cover edge cases: [list edge cases]"
```

---

## 💡 Pro Tips

### 1. Provide Context Efficiently
- Start sessions by having Claude read key files
- Reference specific line numbers for changes
- Use file paths consistently

### 2. Iterative Refinement
- Start with working code, then optimize
- Test each component before combining
- Keep iterations small and focused

### 3. Use Claude's Memory
- Reference earlier in conversation: "Use the ResNet block we created earlier"
- Build on previous work: "Now add dropout to the model from step 2"

### 4. Verify Everything
- Run code after generation
- Check shapes and dtypes
- Validate on small batch first

### 5. Learn While Building
- Ask "why" after implementations
- Request alternative approaches
- Understand tradeoffs

---

## 🔧 Customization

### Create Custom Slash Commands

Create `.claude/commands/train.md`:
```markdown
Run full training pipeline:
1. Load config from config.yaml
2. Initialize data loaders
3. Create model and optimizer
4. Run training loop
5. Save checkpoints
6. Log to W&B
```

Usage: `/train` in Claude Code session

### Create Templates

Create `.claude/templates/pytorch_project/`:
```
pytorch_project/
├── data/
│   └── dataloader.py
├── models/
│   └── model.py
├── train.py
├── evaluate.py
├── config.yaml
└── requirements.txt
```

---

## 📊 Workflow Comparison

| Task | Manual Time | With Claude Code | Speedup |
|------|-------------|------------------|---------|
| Basic CNN | 2-3 hours | 30-45 min | 3-4x |
| Paper implementation | 1-2 days | 2-4 hours | 4-6x |
| Data pipeline | 1-2 hours | 20-30 min | 3-4x |
| Debugging | Highly variable | 20-40 min | 2-5x |
| Refactoring | 2-4 hours | 45-90 min | 2-3x |
| Deployment setup | 1 day | 2-3 hours | 3-4x |

---

## 🚀 Next Steps

1. Try the [Basic CNN workflow](#workflow-1-building-a-neural-network-from-scratch)
2. Adapt workflows to your specific needs
3. Share your own patterns with the community
4. Explore [Examples](../examples/) for detailed implementations

---

**Remember**: These workflows are starting points. Adapt them to your style, project needs, and team practices.
