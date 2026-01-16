# Common Pitfalls & Solutions in Vibe Coding

## Overview

This guide covers common mistakes when using Claude Code for ML development and how to avoid them. Learn from others' mistakes to accelerate your vibe coding journey.

---

## 🚨 Critical Mistakes

### 1. Blindly Trusting AI Output

#### The Mistake
```python
# Claude generates this code, you copy-paste and run
def train_model(model, data):
    for epoch in range(100):
        for batch in data:
            loss = model(batch)
            loss.backward()  # Missing optimizer.step()!
            # Model never actually updates!
```

#### Why It's Bad
- Code runs without errors
- Loss might even appear to decrease (random fluctuations)
- Waste hours before realizing model isn't training

#### The Fix
✅ **Always review generated code**
✅ **Run on small batch first**
✅ **Verify expected behavior**
✅ **Check outputs make sense**

#### Best Practice
```
You: "Implement training loop"
Claude: [generates code]
You: "Add assertions to verify:
     1. Gradients are computed
     2. Weights are actually changing
     3. Loss is finite
     Run 2 iterations and print debug info."
```

---

### 2. Poor Context Management

#### The Mistake
**Long conversation gets derailed:**
```
You: "Implement ResNet"
Claude: [implements ResNet]
You: "Add dropout"
Claude: [adds dropout]
You: "Change activation to GELU"
Claude: [changes activation]
[... 20 more exchanges ...]
You: "Now add batch norm"
Claude: [Confused - adds it in wrong place or removes existing components]
```

#### Why It's Bad
- AI loses track of cumulative changes
- Introduces bugs in previously working code
- Wastes time debugging regressions

#### The Fix
✅ **Periodically summarize state**
✅ **Start new session for major changes**
✅ **Use file references over descriptions**
✅ **Commit working code frequently**

#### Best Practice
```
Every 5-10 significant changes:

You: "Summarize the current model architecture and all modifications"
Claude: [provides summary]
You: "Perfect. Now let's add [next feature]"

Or start fresh:
You: "Read model.py and understand the current implementation"
Claude: [reads and summarizes]
You: "Now add [feature] to this code"
```

---

### 3. Vague Problem Descriptions

#### The Mistake
```
You: "My model doesn't work"
Claude: "Can you provide more details?"
You: "It's not training properly"
Claude: "What exactly is happening?"
[... 5 exchanges just to understand the problem ...]
```

#### Why It's Bad
- Wastes time on back-and-forth
- AI makes wrong assumptions
- Solutions don't address actual problem

#### The Fix
✅ **Specific symptoms**
✅ **Relevant code/context**
✅ **What you've tried**
✅ **Expected vs actual behavior**

#### Best Practice
```
You: "Training issue - need help debugging.

Symptoms:
- Validation accuracy stuck at 62% for last 20 epochs
- Training accuracy at 98%
- Loss still decreasing

Model: ResNet-50, pretrained
Dataset: Custom (10k images, 10 classes, balanced)
Training: 100 epochs, batch 64, lr 0.001, Adam

Tried:
- Increased dropout from 0.1 to 0.5: no change
- Added weight decay 1e-4: small improvement (60%→62%)
- More data augmentation: no change

Code: [share train.py]

Clearly overfitting. What else should I try?"
```

---

### 4. Not Testing Incrementally

#### The Mistake
```
You: "Implement complete training pipeline with model, data, training, validation, checkpointing, logging, and distributed training"
Claude: [generates 500 lines]
You: [runs it]
Error: RuntimeError: CUDA out of memory
[Now debugging 500 lines to find the issue...]
```

#### Why It's Bad
- Hard to isolate bugs
- Wasted effort if early component broken
- Compounds errors

#### The Fix
✅ **Build incrementally**
✅ **Test each component**
✅ **Verify before adding complexity**

#### Best Practice
```
Step 1:
You: "Create basic model architecture, test with dummy input"
Claude: [implements]
You: "Run and show output shapes"
✓ Works

Step 2:
You: "Add data loader, visualize one batch"
Claude: [implements]
You: "Show batch statistics"
✓ Works

Step 3:
You: "Add basic training loop, run 2 iterations"
Claude: [implements]
✓ Works

[Continue building...]
```

---

## ⚠️ Common Issues

### 5. Ignoring Performance from Start

#### The Mistake
```python
# Naive implementation
for epoch in range(100):
    for img_path in image_paths:  # Loading one by one!
        img = Image.open(img_path)
        img = transform(img)
        # This takes forever...
```

#### The Solution
```
You: "Implement data loader with performance in mind:
     - Use DataLoader with multiple workers
     - Prefetch batches
     - Pin memory for GPU
     - Benchmark loading speed
     Target: <0.1s per batch of 128 images"
```

---

### 6. No Reproducibility

#### The Mistake
```
You: "Run training"
Claude: [trains model, gets 85% accuracy]
You: "Great! Run again to verify"
Claude: [trains again, gets 78% accuracy]
You: "Why is it different?"
```

#### The Solution
```
You: "Implement training with full reproducibility:
     - Set seed for Python, NumPy, PyTorch
     - Deterministic CUDA operations
     - Fixed data splits
     - Save all hyperparameters
     - Version data and code
     Log seed in experiment tracking."
```

---

### 7. Hardcoded Values

#### The Mistake
```python
# AI generates this
def train():
    model = ResNet50()
    data = load_data("/Users/claude/data/cifar10")  # Hardcoded!
    lr = 0.001  # Magic number!
    # ...
```

#### The Solution
```
You: "Make training configurable:
     - All hyperparameters in config file (YAML)
     - Data paths from environment variables
     - Command-line argument overrides
     - No hardcoded values in code
     Use hydra or similar for config management."
```

---

### 8. Missing Error Handling

#### The Mistake
```python
def train():
    for epoch in range(100):
        loss = train_epoch()
        # What if loss is NaN?
        # What if CUDA OOM?
        # What if interrupted?
```

#### The Solution
```
You: "Add robust error handling:
     - Check for NaN/Inf losses, stop training
     - Catch CUDA OOM, suggest smaller batch
     - Save checkpoint on keyboard interrupt
     - Log all errors with context
     - Graceful degradation where possible"
```

---

### 9. Poor Logging

#### The Mistake
```python
# Minimal logging
print(f"Epoch {epoch}: Loss = {loss}")
# Where did this run?
# What were the hyperparameters?
# Can I reproduce this?
```

#### The Solution
```
You: "Implement comprehensive logging:
     - Use proper logging (not print)
     - Log all hyperparameters at start
     - Track metrics with W&B/MLflow
     - Save: model, optimizer, metrics, config
     - Unique experiment ID
     - Git commit hash
     - System info (GPU, PyTorch version)
     Make experiments fully reproducible."
```

---

### 10. Not Reading AI Explanations

#### The Mistake
```
Claude: "I implemented dropout with rate 0.5. Note: this is applied
        during training but not evaluation. The rate might be too high
        for your small dataset; consider 0.1-0.3 instead."

You: [Ignores explanation, copies code]
[Later: "Why is my validation performance poor?"]
```

#### The Solution
✅ **Read AI explanations**
✅ **Ask follow-up questions**
✅ **Understand trade-offs**
✅ **Apply suggestions**

---

## 🔍 Subtle Pitfalls

### 11. Shape Mismatches Go Unnoticed

#### The Problem
```python
# Silently broadcasts wrong shapes
x = torch.randn(32, 3, 224, 224)  # Batch of images
weight = torch.randn(3, 1, 1)     # Intended per-channel
y = x * weight  # Works but wrong semantics!
```

#### The Solution
```
You: "Add shape assertions throughout model:
     - After each layer
     - Before operations
     - Print shapes during first forward pass
     - Use descriptive error messages
     Example:
     assert x.shape == (batch_size, 3, 224, 224), f'Expected (B,3,224,224), got {x.shape}'"
```

---

### 12. Training/Eval Mode Confusion

#### The Problem
```python
model.eval()
# Testing model...
# [10 lines later]
for batch in train_loader:
    # Still in eval mode! BN and dropout not working!
    loss = compute_loss(model(batch))
```

#### The Solution
```
You: "Ensure proper train/eval mode:
     - Explicitly set model.train() at loop start
     - Use context manager for evaluation
     - Add assertion checking model.training flag
     - Log mode changes"

Example:
def validate(model, loader):
    model.eval()
    assert not model.training
    with torch.no_grad():
        # validation code
    model.train()  # Reset to training
```

---

### 13. Data Leakage

#### The Problem
```python
# Compute normalization stats on ALL data
all_data = train_data + val_data + test_data
mean = compute_mean(all_data)  # Data leakage!
std = compute_std(all_data)

# Now test set statistics influenced normalization
```

#### The Solution
```
You: "Implement proper data splitting:
     1. Split data FIRST
     2. Compute statistics only on train set
     3. Apply same stats to val/test
     4. Never let test data influence any decisions
     Verify splits don't overlap (check indices)."
```

---

### 14. Gradient Accumulation Bugs

#### The Problem
```python
for batch in loader:
    loss = model(batch)
    loss.backward()
    if step % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()  # BUG: should be before backward!
```

#### The Solution
```
You: "Implement gradient accumulation correctly:
     - zero_grad before accumulation loop
     - Divide loss by accumulation steps
     - Step optimizer after full accumulation
     - Handle final partial accumulation
     Add assertions to verify gradient state"
```

---

### 15. Learning Rate Not Scaling with Batch Size

#### The Problem
```python
# Paper used batch=256, lr=0.1
# You use batch=64 but same lr=0.1
# Effectively 4x higher learning rate!
```

#### The Solution
```
You: "Implement LR scaling with batch size:
     - Base LR from paper: 0.1 for batch 256
     - My batch size: 64
     - Scaled LR: 0.1 * (64/256) = 0.025
     - Apply linear scaling rule
     - Consider warmup for large batches
     Explain the reasoning in comments."
```

---

## 🎯 Best Practices to Avoid Pitfalls

### 1. Checklist-Driven Development

**Before considering code "done":**
```
You: "Review implementation against this checklist:
     □ Shapes verified with assertions
     □ Works with batch_size=1 edge case
     □ Train/eval modes handled correctly
     □ Reproducible (seeded)
     □ No hardcoded paths/values
     □ Error handling for common failures
     □ Tested on small batch
     □ Logging comprehensive
     □ Memory efficient
     □ Docstrings added

     Run through each item and fix if needed."
```

### 2. Test-First Mindset

```
You: "Before implementing the model, create test cases:
     1. test_model_shapes(): verify all layer outputs
     2. test_model_gradients(): ensure backprop works
     3. test_model_deterministic(): same input → same output
     4. test_model_batch_size(): works for 1, 2, 64
     5. test_model_devices(): works on CPU and GPU

     Then implement model to pass these tests."
```

### 3. Defensive Programming

```
You: "Implement with defensive programming:
     - Validate all inputs (types, shapes, ranges)
     - Assert preconditions and postconditions
     - Fail fast with clear error messages
     - Log warnings for suspicious values
     - Add debug mode with extra checks
     Example: check for NaN after every operation in debug mode"
```

### 4. Documentation as You Go

```
You: "Document while implementing:
     - Docstring for each function (Google style)
     - Inline comments for non-obvious code
     - README with usage examples
     - Config file with parameter descriptions
     - Keep a CHANGELOG.md
     Good documentation helps debug later."
```

---

## 🔧 Recovery Strategies

### When Things Go Wrong

#### Strategy 1: Simplify and Isolate
```
You: "Current code is broken with complex error.
     Let's simplify:
     1. Create minimal reproduction (10 lines max)
     2. Remove all extra features
     3. Test each component in isolation
     4. Add complexity back incrementally
     Help me create minimal repro."
```

#### Strategy 2: Compare to Known-Good
```
You: "My implementation doesn't match expected results.
     Let's compare to reference:
     1. Load reference implementation (e.g., timm)
     2. Run both on same input
     3. Compare outputs layer by layer
     4. Find where they diverge
     Help me set up comparison."
```

#### Strategy 3: Revert and Retry
```
You: "Code was working 5 changes ago, now broken.
     Let's:
     1. Git log to see changes
     2. Revert to working version
     3. Re-apply changes one at a time
     4. Test after each change
     5. Identify breaking change
     Help me bisect the problem."
```

---

## 📚 Learning from Mistakes

### Keep a "Lessons Learned" Log

```markdown
## Mistake: Training Loss NaN (2024-01-15)

**What happened**: Loss became NaN after 50 iterations

**Root cause**: Learning rate too high (0.1) for Adam optimizer

**Solution**: Reduced to 0.001, added gradient clipping

**Lesson**: Always start with smaller LR for Adam (0.001-0.0001)

**Prevention**: Add LR range test before full training
```

### Share with Team/Community

- Document your mistakes
- Create guides for common errors
- Contribute to this document!
- Help others avoid same pitfalls

---

## 🎓 Wisdom from Experience

### The 10 Commandments of Vibe Coding

1. **Thou shalt always verify** - Test before trusting
2. **Thou shalt build incrementally** - Small steps, frequent tests
3. **Thou shalt commit often** - Easy rollback saves time
4. **Thou shalt read explanations** - AI teaches, you learn
5. **Thou shalt check shapes** - Assertions prevent silent bugs
6. **Thou shalt handle errors** - Fail gracefully, debug easily
7. **Thou shalt log thoroughly** - Future you needs context
8. **Thou shalt not hardcode** - Config files are your friend
9. **Thou shalt document** - Comments and docs save hours
10. **Thou shalt stay skeptical** - AI assists, you validate

---

## 🚀 Quick Fixes Reference

| Issue | Quick Fix |
|-------|-----------|
| NaN Loss | Add gradient clipping, check LR, verify data normalization |
| OOM | Reduce batch size, use gradient accumulation, clear cache |
| Slow Training | Profile first, then: mixed precision, more workers, compile |
| Low Accuracy | Check data labels, try simpler model first, verify metrics |
| Not Reproducible | Set all seeds, deterministic CUDA, save configs |
| Model not saving | Check permissions, disk space, try different path |
| Data loading slow | Increase workers, prefetch, use SSD, cache preprocessing |
| Poor val performance | Check overfitting, try regularization, more data |

---

**Remember**: Mistakes are learning opportunities. Each pitfall avoided is a skill gained. Keep this guide handy and update it with your own experiences!

---

**Next**: [Quick Reference Cards](./quick-reference.md) for fast lookups
