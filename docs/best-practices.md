# 📘 Best Practices

This document outlines recommended practices for ML/LLM development, documentation, and experimentation in this repository.

## Overview

Following these best practices ensures code quality, reproducibility, and maintainability across all contributions to this repository.

---

## 🧪 Experimentation Best Practices

### Notebook Organization

**Structure:**
```python
# 1. Title and Description
# 2. Imports and Setup
# 3. Configuration/Hyperparameters
# 4. Data Loading
# 5. Model Definition
# 6. Training/Experiment
# 7. Evaluation
# 8. Visualization
# 9. Conclusions and Next Steps
```

**Best Practices:**
- Clear markdown cells explaining each section
- Use meaningful variable names
- Include visualizations for key insights
- Document hyperparameters and configurations
- Clear outputs before committing
- Add execution time for long-running cells

### Experiment Tracking

**Recommended Tools:**
- Weights & Biases for experiment tracking
- MLflow for model management
- TensorBoard for visualization

**Track:**
- Hyperparameters
- Metrics (loss, accuracy, etc.)
- Model architecture
- Training time
- Hardware specifications
- Random seeds for reproducibility

---

## 💻 Code Best Practices

### Python Style Guide

**Follow PEP 8:**
- 4 spaces for indentation
- Max line length: 100 characters
- Use descriptive variable names
- Add type hints

**Example:**
```python
from typing import Tuple, Optional
import torch
import torch.nn as nn

def calculate_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate scaled dot-product attention.
    
    Args:
        query: Query tensor (batch, heads, seq_len, d_k)
        key: Key tensor (batch, heads, seq_len, d_k)
        value: Value tensor (batch, heads, seq_len, d_v)
        mask: Optional attention mask
    
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = torch.softmax(scores, dim=-1)
    attention_output = torch.matmul(attention_weights, value)
    
    return attention_output, attention_weights
```

### Documentation Standards

**Docstring Format:**
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of function.
    
    Detailed explanation if needed. Can include mathematical
    notation or references to papers.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When invalid input is provided
    
    Example:
        >>> result = function_name(10, 20)
        >>> print(result)
        30
    """
    # Implementation
```

### Error Handling

```python
def load_model(path: str) -> nn.Module:
    """Load model from checkpoint with proper error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    try:
        checkpoint = torch.load(path)
        model = create_model(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
```

---

## 📊 Data Management

### Dataset Organization

```
data/
├── raw/              # Original, immutable data
├── processed/        # Cleaned, processed data
├── interim/          # Intermediate transformations
└── external/         # External datasets
```

### Data Loading

**Best Practices:**
- Use dataloaders for efficient batching
- Implement caching for repeated access
- Use memory-mapped files for large datasets
- Implement proper train/val/test splits

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """Custom dataset with proper initialization and caching."""
    
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self._load_metadata()
    
    def _load_metadata(self):
        # Load only metadata, not full data
        pass
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int):
        # Load data on demand
        item = self._load_item(idx)
        if self.transform:
            item = self.transform(item)
        return item
```

---

## 🔬 Model Development

### Model Architecture

**Modularity:**
```python
class TransformerBlock(nn.Module):
    """Modular transformer block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attended = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward with residual connection
        fed_forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(fed_forward))
        
        return x
```

### Training Loop

**Best Practices:**
```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Training loop with best practices."""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)
```

---

## 📝 Documentation Best Practices

### README Files

**Structure:**
```markdown
# Title

## Overview
Brief description

## Key Concepts
- Concept 1
- Concept 2

## Content
- Subdirectory descriptions

## Examples
Code examples or notebooks

## Resources
- Papers
- Tutorials
- External links

## Further Reading
Related topics
```

### Code Comments

**When to Comment:**
- Complex algorithms or mathematical operations
- Non-obvious design decisions
- TODOs and FIXMEs
- References to papers or resources

**When NOT to Comment:**
- Self-explanatory code
- Redundant information in docstrings

---

## 🧪 Testing

### Unit Tests

```python
import unittest
import torch

class TestAttentionMechanism(unittest.TestCase):
    """Test suite for attention mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = 512
    
    def test_attention_output_shape(self):
        """Test attention output has correct shape."""
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output, weights = calculate_attention(query, key, value)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))
    
    def test_attention_weights_sum_to_one(self):
        """Test attention weights sum to 1.0."""
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        _, weights = calculate_attention(query, key, value)
        
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))
```

---

## 🚀 Deployment Best Practices

### Model Optimization

**Techniques:**
1. **Quantization**: INT8 or FP16 precision
2. **Pruning**: Remove unnecessary weights
3. **Distillation**: Train smaller models
4. **ONNX Export**: Framework-agnostic deployment

### Serving

**Best Practices:**
- Use batching for efficiency
- Implement caching for repeated queries
- Monitor latency and throughput
- Set up proper logging
- Implement health checks

---

## 🔐 Security and Privacy

### Data Handling

- Never commit sensitive data or credentials
- Use `.env` files for configuration
- Implement proper access controls
- Anonymize datasets when sharing

### Model Security

- Validate all inputs
- Implement rate limiting
- Monitor for adversarial attacks
- Keep dependencies updated

---

## 📊 Version Control

### Git Workflow

**Commit Messages:**
```
feat(transformers): add multi-head attention implementation
fix(rag): correct vector similarity calculation
docs(readme): update installation instructions
refactor(utils): improve data loading efficiency
```

**Branch Naming:**
- `feature/feature-name`
- `fix/bug-description`
- `docs/documentation-update`
- `refactor/code-improvement`

---

## 🎯 Performance Optimization

### Memory Management

```python
# Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    return checkpoint(self.layer, x)

# Clear cache when needed
torch.cuda.empty_cache()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### Profiling

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    model(input_data)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 📚 Further Reading

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PyTorch Best Practices](https://pytorch.org/docs/stable/notes/cuda.html)
- [Clean Code in Python](https://realpython.com/python-code-quality/)
- [ML Engineering Best Practices](https://ml-ops.org/)

---

*These best practices evolve with the field. Contributions and suggestions are welcome!*