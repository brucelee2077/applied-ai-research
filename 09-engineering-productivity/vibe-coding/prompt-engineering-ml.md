# ML-Specific Prompt Engineering Guide

## Overview

This guide focuses on crafting effective prompts for machine learning and AI development tasks with Claude Code. Learn to communicate ML requirements clearly, get better implementations, and iterate faster.

---

## 🎯 Prompt Engineering Fundamentals

### The Anatomy of a Good ML Prompt

```
[Context] + [Task] + [Constraints] + [Expected Output]
```

**Example:**
```
Context: Building image classifier for medical X-rays
Task: Implement data augmentation pipeline
Constraints: Must preserve diagnostic features, handle class imbalance
Expected Output: PyTorch transforms with visualization
```

---

## 📋 ML Task Categories

## 1. Architecture Implementation

### ❌ Bad Prompt
```
"Create a neural network"
```

### ✅ Good Prompt
```
"Create a ResNet-18 architecture for CIFAR-10 with:
- Input: 32x32 RGB images
- 4 residual blocks with [64, 128, 256, 512] channels
- Batch normalization after each conv
- Global average pooling before classifier
- 10 output classes
- Use PyTorch nn.Module
Show parameter count when done."
```

### 🌟 Excellent Prompt
```
"Implement ResNet-18 for CIFAR-10 classification.

Architecture requirements:
- Initial conv: 3→64 channels, 3x3 kernel, stride 1
- 4 residual stages with channels [64, 128, 256, 512]
- Each stage: 2 BasicBlocks (conv-bn-relu-conv-bn + skip)
- Downsample: stride 2 in first block of stages 2-4
- Final: global avg pool + fc layer to 10 classes

Implementation details:
- Use kaiming initialization for conv layers
- Zero-init last BN in each residual block
- Add option for dropout (default 0.0)
- Include forward hook for intermediate features

Deliverables:
1. BasicBlock class
2. ResNet class with config
3. Test script verifying shapes
4. Parameter count (should be ~11M)
```

---

## 2. Data Pipeline

### ❌ Bad Prompt
```
"Load ImageNet data"
```

### ✅ Good Prompt
```
"Create PyTorch DataLoader for ImageNet with:
- Training augmentation: RandomResizedCrop(224), RandomHorizontalFlip
- ImageNet normalization stats
- Batch size 256
- 8 worker processes
- Pin memory for GPU
Handle train/val splits properly."
```

### 🌟 Excellent Prompt
```
"Implement production-grade ImageNet data pipeline.

Data specs:
- Location: /data/imagenet/{train,val}/ with 1000 class folders
- Images: JPEG format, various sizes
- Train: 1.28M images, Val: 50k images

Training augmentation (RandAugment style):
- RandomResizedCrop(224, scale=(0.08, 1.0))
- RandomHorizontalFlip(p=0.5)
- RandAugment(num_ops=2, magnitude=9)
- ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
- Normalize with ImageNet stats

Validation:
- Resize(256) → CenterCrop(224)
- Normalize only

DataLoader config:
- Batch size: 256 per GPU (4 GPUs = 1024 effective)
- Workers: 8 per GPU
- Pin memory: True
- Prefetch factor: 2
- Persistent workers: True

Additional requirements:
- Handle corrupted images gracefully
- Add progress bar for first epoch
- Option to use subset for fast iteration
- Reproducible with seed
- Log data statistics (mean, std, class distribution)

Include:
1. Custom Dataset class
2. Collate function for variable sizes
3. Visualization function for batch
4. Performance benchmarking code
```

---

## 3. Training Loop

### ❌ Bad Prompt
```
"Write training code"
```

### ✅ Good Prompt
```
"Create training loop with:
- Cross entropy loss
- Adam optimizer, lr=0.001
- Train for 100 epochs
- Validate every epoch
- Save checkpoints
- Log metrics"
```

### 🌟 Excellent Prompt
```
"Implement production training loop for image classification.

Training setup:
- Loss: CrossEntropyLoss with label smoothing (0.1)
- Optimizer: AdamW (lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999))
- LR Schedule:
  * Warmup: Linear 0→1e-3 over 5 epochs
  * Main: Cosine annealing 1e-3→1e-5 over remaining epochs
- Gradient clipping: max_norm=1.0
- Mixed precision: AMP enabled
- Epochs: 100 with early stopping (patience=10)

Checkpointing:
- Save every 10 epochs
- Keep best 3 by validation accuracy
- Save: model, optimizer, scheduler, epoch, metrics
- Resume capability from checkpoint

Logging (Weights & Biases):
- Hyperparameters at start
- Every epoch: train/val loss, accuracy, learning rate
- Every 20 batches: batch loss, throughput (images/sec)
- Every 10 epochs: confusion matrix, example predictions
- Track: gradient norms, weight norms per layer

Validation:
- Run full val set every epoch
- Compute: top-1 accuracy, top-5 accuracy, per-class accuracy
- Return best validation checkpoint

Error handling:
- Catch NaN losses and terminate gracefully
- Handle OOM with smaller batch if occurs
- Save emergency checkpoint on interruption

Monitoring:
- tqdm progress bars with metrics
- Print epoch summary table
- ETA for training completion

Structure as:
- train_one_epoch() function
- validate() function
- main train() function with full loop
- Proper device handling (auto-detect GPU)
```

---

## 4. Debugging

### ❌ Bad Prompt
```
"Fix my code"
```

### ✅ Good Prompt
```
"My training loss becomes NaN after ~100 iterations.
Code: [paste training loop]
Model: ResNet-50
Dataset: Custom images
Help debug."
```

### 🌟 Excellent Prompt
```
"Debug NaN loss issue in training.

Symptoms:
- Training starts fine, loss ~2.3
- Around iteration 80-120, loss suddenly becomes NaN
- Happens consistently across runs
- Validation not reached

Environment:
- Model: ResNet-50 (pretrained ImageNet, fine-tuning last layer)
- Dataset: Medical images, 5000 train / 500 val, 10 classes
- Batch size: 32
- Optimizer: SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)
- Loss: CrossEntropyLoss
- Device: Single V100 GPU
- PyTorch 2.0.1

What I've tried:
- Reduced LR to 0.001: NaN appears later (~200 iters)
- Added gradient clipping (max_norm=1.0): Still happens
- Checked data: no NaN/Inf in images or labels

Code structure:
[Share: model definition, data loading, training loop]

Please:
1. Identify most likely causes
2. Add diagnostic logging to pinpoint issue
3. Suggest fixes in priority order
4. Add assertions to catch NaN early

Also explain why each fix helps.
```

---

## 5. Paper Implementation

### ❌ Bad Prompt
```
"Implement Vision Transformer"
```

### ✅ Good Prompt
```
"Implement Vision Transformer (ViT) from 'An Image is Worth 16x16 Words'.
Key components:
- Patch embedding (16x16 patches)
- Positional embeddings
- Transformer encoder (12 layers)
- Classification head
Use ViT-Base config."
```

### 🌟 Excellent Prompt
```
"Implement Vision Transformer (ViT) from Dosovitskiy et al. 2020.

Paper details:
- Title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
- Key idea: Split image into patches, apply standard transformer

Architecture (ViT-Base/16):
- Image: 224×224 RGB
- Patch size: 16×16 (196 patches total)
- Patch embedding: Linear projection to D=768
- Position embedding: Learnable 1D for 196 patches + [CLS] token
- Transformer encoder:
  * 12 layers (blocks)
  * Each block: MSA → MLP with residuals and LayerNorm
  * MSA: 12 heads, head_dim=64, total_dim=768
  * MLP: 768 → 3072 → 768 with GELU
  * Pre-norm (LayerNorm before MSA and MLP, not after)
- Classification head: LayerNorm → Linear(768 → num_classes)

Implementation requirements:
- Match paper exactly (pre-norm, not post-norm!)
- Efficient attention (can use torch.nn.MultiheadAttention)
- Dropout: 0.1 in attention and MLP
- Stochastic depth: optional, rate=0.1
- Weight init: as per paper (truncated normal for most, zeros for biases)

Verification:
1. Param count: ~86M for ViT-Base
2. Test forward pass: (B=2, C=3, H=224, W=224) → (B=2, num_classes)
3. Verify intermediate shapes match paper
4. Compare to timm implementation if possible

Deliverables:
1. PatchEmbed module
2. TransformerBlock module
3. VisionTransformer model
4. Config class for variants (Tiny, Small, Base, Large)
5. Test script with shape verification
6. Comparison to published results

Bonus:
- Add attention visualization utility
- Support for different image sizes (via interpolation)
- Distillation token support (DeiT variant)
```

---

## 6. Optimization & Performance

### ❌ Bad Prompt
```
"Make training faster"
```

### ✅ Good Prompt
```
"Optimize training speed. Currently 5 min/epoch on V100.
Model: ResNet-50, Batch: 128, Data: CIFAR-10
Suggest optimizations for 2x speedup."
```

### 🌟 Excellent Prompt
```
"Profile and optimize training pipeline for 2-3x speedup.

Current performance:
- 5 minutes per epoch
- GPU utilization: ~60% (nvidia-smi)
- Throughput: ~170 images/sec
- Hardware: Single V100 (32GB)

Setup:
- Model: ResNet-50 (25M params)
- Data: CIFAR-10 (50k train images)
- Batch size: 128
- Workers: 4
- Current optimizations: None (baseline PyTorch)

Target:
- 2-3 minutes per epoch (2-3x faster)
- Maintain accuracy within 1%

Please:
1. Add profiling code to identify bottlenecks:
   - Data loading time
   - Forward pass time
   - Backward pass time
   - Optimizer step time
   - GPU vs CPU time breakdown

2. Suggest optimizations in priority order:
   - Mixed precision training (AMP)
   - Compiled model (torch.compile)
   - Data loader tuning (workers, prefetch, pin_memory)
   - Gradient accumulation if batch size limited
   - Fused optimizers
   - Channel-last memory format
   - Any other relevant optimizations

3. For each optimization:
   - Show code changes
   - Estimate speedup
   - Note any caveats or trade-offs

4. Create benchmarking script that compares:
   - Baseline vs each optimization
   - Combined optimizations
   - Throughput (images/sec)
   - Memory usage

5. Final recommendations with expected 2-3x speedup path
```

---

## 🎨 Advanced Prompting Patterns

### Pattern 1: Iterative Refinement

**Phase 1: Get it working**
```
"Implement basic transformer encoder block:
- Multi-head attention (8 heads, dim=512)
- Feed-forward network (512→2048→512)
- Residual connections
- Layer normalization
Keep it simple first."
```

**Phase 2: Optimize**
```
"Now optimize the transformer block:
- Use flash attention
- Fuse layer norms
- Add gradient checkpointing option
- Support mixed precision
Benchmark vs previous version."
```

**Phase 3: Production-harden**
```
"Production-ready the transformer:
- Add input validation
- Handle edge cases (batch_size=1, seq_len=1)
- Add comprehensive docstrings
- Type hints for all functions
- Unit tests for each component
```

### Pattern 2: Constraint-Based

```
"Implement training loop with HARD constraints:

MUST have:
- Reproducibility (seed everything)
- Resume from checkpoint
- NaN/Inf detection
- Progress bars

MUST NOT:
- Use more than 16GB GPU memory
- Take longer than 10 min/epoch on V100
- Have any hardcoded paths

PREFER:
- Logging with W&B
- Mixed precision
- Minimal dependencies

Given these constraints, implement the training loop.
```

### Pattern 3: Example-Driven

```
"Create a data augmentation pipeline.

Example input:
- Image: (3, 224, 224) tensor, uint8, range [0, 255]
- Label: int, range [0, 9]

Example output after augmentation:
- Image: (3, 224, 224) tensor, float32, normalized
- Should look like: [show example augmented image]

Augmentations to apply:
1. Random crop
2. Color jitter
[etc...]

Show me the transform pipeline and example outputs.
```

### Pattern 4: Comparison-Based

```
"I have two approaches for implementing attention:

Approach A: torch.nn.MultiheadAttention
Approach B: Custom implementation with einops

For my use case (ViT training, batch=256, seq_len=197, dim=768):
1. Implement both approaches
2. Benchmark speed and memory
3. Compare numerical outputs (should match)
4. Recommend which to use and why

Consider: speed, memory, maintainability, flexibility
```

---

## 💡 ML-Specific Tips

### 1. Always Specify Shapes

**Instead of:**
```
"Add batch normalization"
```

**Use:**
```
"Add batch normalization for input shape (B, C, H, W) where:
- B: batch size (variable)
- C: 256 channels
- H, W: 32×32 spatial dimensions
Place after conv layer, before activation."
```

### 2. Be Explicit About Frameworks

**Instead of:**
```
"Create a CNN"
```

**Use:**
```
"Create a CNN using PyTorch (torch.nn.Module).
- Use torch.nn.Conv2d (not functional)
- Define layers in __init__
- Implement forward() method
- Compatible with torch.jit.script
```

### 3. Specify Numerical Precision

**Instead of:**
```
"Normalize the data"
```

**Use:**
```
"Normalize images to zero mean, unit variance.
- Compute: (x - mean) / std
- Use ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Input: uint8 [0, 255], Output: float32 [-~2, ~2]
- Apply after ToTensor()
```

### 4. Include Hyperparameter Context

**Instead of:**
```
"Use Adam optimizer"
```

**Use:**
```
"Use AdamW optimizer with settings from 'Attention is All You Need':
- Learning rate: 1e-3 (will use warmup + decay separately)
- Betas: (0.9, 0.98) per paper
- Eps: 1e-9 for numerical stability
- Weight decay: 0.01 (L2 regularization)
Explain why these specific values."
```

### 5. Request Explanations

**Add to any prompt:**
```
"After implementing, explain:
- Why this approach over alternatives
- Potential failure modes
- Hyperparameters to tune first
- Expected behavior and metrics
```

---

## 📊 Prompt Templates by Task

### Template: New Model Architecture
```
"Implement [MODEL_NAME] for [TASK].

Architecture:
- Input: [shape and type]
- Layers: [detailed layer descriptions]
- Output: [shape and type]

Requirements:
- Framework: [PyTorch/TensorFlow/JAX]
- Match [paper/reference] exactly
- Parameter count: ~[expected count]

Include:
1. Model class
2. Config class/dict
3. Weight initialization
4. Test with dummy input
5. Print parameter count

Verify shapes at each layer."
```

### Template: Debug Request
```
"Debug [ISSUE] in my [COMPONENT].

Problem:
- Symptom: [what's wrong]
- When: [when it occurs]
- Frequency: [always/sometimes/rarely]

Context:
- Code: [relevant code or file reference]
- Environment: [hardware, framework versions]
- What I've tried: [failed attempts]

Please:
1. Identify likely causes
2. Add diagnostic code
3. Suggest fixes
4. Explain why
```

### Template: Optimization Request
```
"Optimize [COMPONENT] for [METRIC].

Current performance:
- [Metric]: [current value]
- Hardware: [GPU/CPU specs]
- Bottleneck: [if known]

Target:
- [Metric]: [target value]
- Constraints: [memory/time/accuracy limits]

Optimization options to consider:
- [list relevant techniques]

For each optimization:
- Show code changes
- Estimate improvement
- Note trade-offs
```

---

## 🚀 Progressive Prompting Example

### Building a Training Pipeline (Start to Finish)

**Prompt 1: Foundation**
```
"Create project structure for image classification:
- model.py: ResNet-18
- data.py: CIFAR-10 loaders
- train.py: training loop skeleton
- config.yaml: hyperparameters
- requirements.txt: dependencies

Just create structure, don't implement yet."
```

**Prompt 2: Model**
```
"Implement ResNet-18 in model.py:
[detailed specs...]
Test with dummy input first."
```

**Prompt 3: Data**
```
"Implement CIFAR-10 pipeline in data.py:
[detailed specs...]
Visualize a batch to verify."
```

**Prompt 4: Training Loop**
```
"Implement training loop in train.py:
[detailed specs...]
Run 2 epochs to verify everything works."
```

**Prompt 5: Enhancement**
```
"Add TensorBoard logging and checkpointing:
[specs...]"
```

**Prompt 6: Optimization**
```
"Add mixed precision and distributed training support:
[specs...]"
```

---

## ❌ Common Mistakes to Avoid

### 1. Too Vague
```
❌ "Make my model better"
✅ "Increase model capacity by:
   - Doubling channels in each layer
   - Adding one more residual block
   - Estimate new parameter count"
```

### 2. Missing Context
```
❌ "Why is accuracy low?"
✅ "Validation accuracy stuck at 65% (random is 10% for 10 classes).
   Model: ResNet-18, Dataset: CIFAR-10
   Training: 50 epochs, accuracy keeps increasing
   Validation: plateaued after epoch 20

   Possible overfitting. Suggest regularization techniques."
```

### 3. Assuming Knowledge
```
❌ "Use the standard settings"
✅ "Use ImageNet pretraining settings from torchvision:
   - Learning rate: 0.1
   - Momentum: 0.9
   - Weight decay: 1e-4
   - LR schedule: step decay by 0.1 at epochs [30, 60, 90]"
```

### 4. No Verification Request
```
❌ "Implement batch normalization"
✅ "Implement batch normalization and verify:
   1. Batch stats computed correctly during training
   2. Running stats updated with momentum 0.1
   3. Running stats used during eval mode
   4. Affine parameters (gamma, beta) learnable

   Print shapes and values for debugging."
```

---

## 🎯 Quick Reference

| Task | Key Elements to Include |
|------|------------------------|
| Architecture | Shapes, layer specs, param count, framework |
| Data Pipeline | Augmentation, normalization, splits, batch size |
| Training | Loss, optimizer, LR schedule, epochs, checkpointing |
| Debugging | Symptoms, context, tried solutions, error messages |
| Optimization | Current performance, target, constraints, metrics |
| Paper Impl | Paper details, exact config, verification steps |

---

**Next**: See these prompts in action in [Examples](../examples/)
