# Example: Building a CNN Classifier with Claude Code

## Overview

This example demonstrates how to build a complete CNN image classifier using Claude Code, from initial concept to trained model. We'll show actual conversation patterns and the iterative development process.

**Task**: Build a CNN classifier for CIFAR-10 dataset
**Time**: ~45 minutes with Claude Code
**Goal**: Demonstrate real vibe coding workflow

---

## 🎬 The Conversation

### Phase 1: Initial Setup (5 minutes)

**You:**
```
I want to build a CNN classifier for CIFAR-10. Let's start with a simple architecture:
- 3 convolutional blocks (each with conv -> batch norm -> relu -> max pool)
- Channels: 32, 64, 128
- Final fully connected layers: 512 -> 10

First, create the project structure:
- model.py (architecture)
- train.py (training loop)
- data.py (data loading)
- config.yaml (hyperparameters)
```

**Claude Code:**
```
I'll create the project structure for your CIFAR-10 CNN classifier.
Starting with the architecture in model.py...
```

**Result:**
- ✅ Created `model.py` with SimpleCNN class
- ✅ Created `train.py` with training skeleton
- ✅ Created `data.py` with CIFAR-10 loaders
- ✅ Created `config.yaml` with default hyperparameters

---

### Phase 2: Testing Architecture (5 minutes)

**You:**
```
Before training, let's verify the model works correctly.
Create a test script that:
1. Instantiates the model
2. Passes a dummy batch (shape: 4, 3, 32, 32)
3. Prints output shape and parameter count
4. Verifies gradients flow backward
```

**Claude Code:**
```python
# Created test_model.py
import torch
from model import SimpleCNN

def test_model():
    model = SimpleCNN(num_classes=10)

    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [4, 10]

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"WARNING: No gradient for {name}")
        else:
            print(f"✓ Gradient exists for {name}")

    print("\n✅ Model test passed!")

if __name__ == "__main__":
    test_model()
```

**You:**
```
Run test_model.py
```

**Output:**
```
Input shape: torch.Size([4, 3, 32, 32])
Output shape: torch.Size([4, 10])
Total parameters: 1,234,858
Trainable parameters: 1,234,858
✓ Gradient exists for conv1.weight
✓ Gradient exists for conv1.bias
...
✅ Model test passed!
```

---

### Phase 3: Data Pipeline (5 minutes)

**You:**
```
Now let's set up the data pipeline properly.
In data.py, I need:
- Train/val split (45k/5k from training set)
- Data augmentation for training:
  * Random horizontal flip
  * Random crop (32x32 with padding 4)
  * Color jitter (brightness=0.2, contrast=0.2)
- Standard normalization with CIFAR-10 stats
- No augmentation for validation
- Batch size 128, 4 workers

Also add a function to visualize a batch.
```

**Claude Code:**
```python
# Updated data.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 normalization stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def get_dataloaders(batch_size=128, num_workers=4):
    # Training augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=train_transform
    )

    # Split train into train/val
    train_size = 45000
    val_size = 5000
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )

    # Apply val transform to val split
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader

def visualize_batch(loader, num_images=16):
    """Visualize a batch of images"""
    images, labels = next(iter(loader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Denormalize for visualization
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        img = images[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"Class: {labels[idx].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_batch.png')
    print("Saved sample_batch.png")
```

**You:**
```
Run a quick test to visualize training data and check the split sizes
```

**Claude Code:**
```python
# Created test_data.py
from data import get_dataloaders, visualize_batch

train_loader, val_loader = get_dataloaders()
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples: {len(val_loader.dataset)}")

visualize_batch(train_loader)
```

**Output:**
```
Train batches: 352
Val batches: 40
Train samples: 45000
Val samples: 5000
Saved sample_batch.png
```

---

### Phase 4: Training Loop (10 minutes)

**You:**
```
Now implement the training loop in train.py with:
- Cross entropy loss
- Adam optimizer with lr=0.001
- Cosine annealing LR scheduler (T_max=200)
- Train for 50 epochs
- Log metrics every 10 batches
- Validate every epoch
- Save best model checkpoint
- Early stopping (patience=10 epochs)
- Progress bars with tqdm
```

**Claude Code:**
```python
# Updated train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml
from pathlib import Path

from model import SimpleCNN
from data import get_dataloaders

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': running_loss / len(val_loader),
                'acc': 100. * correct / total
            })

    return running_loss / len(val_loader), 100. * correct / total

def train(config_path='config.yaml'):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Model
    model = SimpleCNN(num_classes=10).to(device)

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # LR scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}\n")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_dir / 'best_model.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()
```

**You:**
```
Also update config.yaml with all the hyperparameters
```

**Claude Code:**
```yaml
# config.yaml
# Model
num_classes: 10

# Data
batch_size: 128
num_workers: 4

# Training
epochs: 50
learning_rate: 0.001
patience: 10

# Optimizer
optimizer: adam
weight_decay: 0.0001

# Scheduler
scheduler: cosine
T_max: 200
```

---

### Phase 5: Running Training (15 minutes)

**You:**
```
Let's do a quick test run with 2 epochs to make sure everything works before full training
```

**Claude Code:**
```python
# Modified config.yaml temporarily
epochs: 2
```

```bash
python train.py
```

**Output:**
```
Using device: cuda
Epoch 1 [Train]: 100%|████████| 352/352 [00:45<00:00, loss=1.8234, acc=35.23%]
Epoch 1 [Val]: 100%|████████| 40/40 [00:03<00:00, loss=1.6543, acc=41.20%]

Epoch 1/2
Train Loss: 1.8234 | Train Acc: 35.23%
Val Loss: 1.6543 | Val Acc: 41.20%
LR: 0.001000

✓ Saved best model (Val Acc: 41.20%)

Epoch 2 [Train]: 100%|████████| 352/352 [00:44<00:00, loss=1.5432, acc=45.67%]
Epoch 2 [Val]: 100%|████████| 40/40 [00:03<00:00, loss=1.4321, acc=48.90%]

Epoch 2/2
Train Loss: 1.5432 | Train Acc: 45.67%
Val Loss: 1.4321 | Val Acc: 48.90%
LR: 0.000995

✓ Saved best model (Val Acc: 48.90%)

Training complete! Best Val Acc: 48.90%
```

**You:**
```
Great! Everything works. Now set epochs back to 50 and start full training.
Also add functionality to resume from checkpoint if it exists.
```

**Claude Code:**
```python
# Added to train.py
def train(config_path='config.yaml', resume=True):
    # ... previous setup code ...

    start_epoch = 1
    if resume and (checkpoint_dir / 'best_model.pth').exists():
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['val_acc']
        print(f"Resumed from epoch {checkpoint['epoch']}")

    for epoch in range(start_epoch, config['epochs'] + 1):
        # ... training loop ...
```

---

### Phase 6: Monitoring & Debugging (5 minutes)

**You:**
```
While training, I want to monitor:
1. Learning rate over time
2. Train vs val loss curves
3. Train vs val accuracy curves
4. Gradient norms

Add TensorBoard logging for all of these.
```

**Claude Code:**
```python
# Updated train.py
from torch.utils.tensorboard import SummaryWriter

def train(config_path='config.yaml', resume=True):
    # ... setup code ...

    writer = SummaryWriter('runs/cifar10_cnn')

    for epoch in range(start_epoch, config['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(...)
        val_loss, val_acc = validate(...)

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        # Log gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar('GradientNorm', total_norm, epoch)

        # ... rest of training loop ...

    writer.close()
```

**You:**
```
Launch TensorBoard in the background so I can monitor training
```

**Claude Code:**
```bash
tensorboard --logdir=runs &
echo "TensorBoard started at http://localhost:6006"
```

---

### Phase 7: Final Training Results

**After 50 epochs (or early stopping):**

```
Epoch 35/50
Train Loss: 0.3456 | Train Acc: 88.23%
Val Loss: 0.6789 | Val Acc: 77.45%
LR: 0.000234

✓ Saved best model (Val Acc: 77.45%)

Early stopping triggered after 37 epochs
Training complete! Best Val Acc: 77.45%
```

**You:**
```
Good! Now create an evaluation script that:
1. Loads the best checkpoint
2. Evaluates on test set
3. Shows confusion matrix
4. Shows per-class accuracy
5. Visualizes some misclassified examples
```

**Claude Code:**
```python
# Created evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from model import SimpleCNN
from data import CIFAR10_MEAN, CIFAR10_STD

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = SimpleCNN(num_classes=10).to(device)
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=100,
        shuffle=False, num_workers=4
    )

    # Class names
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Evaluate
    all_preds = []
    all_targets = []
    correct = 0
    total = 0

    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    # Overall accuracy
    test_acc = 100. * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%\n")

    # Per-class accuracy
    print("Per-class Accuracy:")
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"  {classes[i]:10s}: {acc:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nSaved confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
```

**Output:**
```
Test Accuracy: 76.89%

Per-class Accuracy:
  plane     : 82.3%
  car       : 85.1%
  bird      : 68.7%
  cat       : 60.2%
  deer      : 71.4%
  dog       : 67.9%
  frog      : 83.5%
  horse     : 80.6%
  ship      : 84.7%
  truck     : 84.5%

Saved confusion_matrix.png
```

---

## 📊 Summary

### What We Built
- ✅ Complete CNN architecture
- ✅ Data pipeline with augmentation
- ✅ Training loop with best practices
- ✅ Checkpointing and early stopping
- ✅ TensorBoard monitoring
- ✅ Evaluation suite
- ✅ Achieved ~77% validation accuracy

### Time Breakdown
- Initial setup: 5 min
- Architecture testing: 5 min
- Data pipeline: 5 min
- Training loop: 10 min
- Running & monitoring: 15 min
- Evaluation: 5 min
**Total: ~45 minutes**

### What Would Take Longer Manually
- Writing boilerplate: +30 min
- Debugging shape mismatches: +15 min
- Setting up monitoring: +15 min
- Creating evaluation: +20 min
**Traditional time: ~2+ hours**

### Key Productivity Gains
1. **No boilerplate writing**: Claude handled all standard code
2. **Immediate testing**: Verified each component before moving on
3. **Best practices included**: Proper normalization, LR scheduling, etc.
4. **Iterative refinement**: Easy to add features incrementally
5. **Learning opportunity**: Could ask "why" at any step

---

## 🎯 Key Takeaways

### Effective Patterns Used

1. **Incremental Building**
   - Started with structure
   - Added complexity gradually
   - Tested at each step

2. **Clear Requirements**
   - Specific architecture details
   - Explicit hyperparameters
   - Clear acceptance criteria

3. **Validation First**
   - Test scripts before training
   - Quick runs to verify
   - Monitoring from the start

4. **Iterative Enhancement**
   - Basic → Optimized → Production
   - Added TensorBoard later
   - Enhanced evaluation progressively

### What Made This Efficient

✅ **Good prompts**: Specific, with clear requirements
✅ **Testing early**: Caught issues before long training
✅ **Incremental**: Built confidence with each step
✅ **Monitoring**: TensorBoard for visibility
✅ **Best practices**: Adam, cosine LR, early stopping, etc.

---

## 💡 Try It Yourself

1. Start a Claude Code session in a new directory
2. Follow this conversation pattern
3. Adapt to your own dataset/architecture
4. Share your results and learnings!

---

**Next**: Try [Implementing a Research Paper](./paper-implementation-example.md) for a more advanced workflow.
