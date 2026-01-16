# Quick Reference Cards

Fast lookup guide for common ML development tasks with Claude Code. Copy-paste and adapt these prompts for your needs.

---

## 🏗️ Project Setup

### New ML Project
```
"Set up a new PyTorch project for [TASK]:

Structure:
├── data/           # Dataset storage
├── models/         # Model definitions
├── train.py        # Training script
├── evaluate.py     # Evaluation script
├── config/         # Configuration files
│   └── default.yaml
├── utils/          # Helper functions
├── notebooks/      # Jupyter notebooks
├── tests/          # Unit tests
├── requirements.txt
├── .gitignore
└── README.md

Include:
- requirements.txt with: torch, torchvision, numpy, wandb, tqdm, pyyaml
- .gitignore for Python/PyTorch (data/, checkpoints/, __pycache__, etc.)
- README with project description and setup instructions
- Config template with common hyperparameters"
```

### Add Experiment Tracking
```
"Integrate Weights & Biases tracking:
- Install wandb
- Initialize in train.py with project name '[PROJECT]'
- Log: hyperparameters, metrics (loss, accuracy), learning rate
- Log every epoch: train/val metrics
- Save model checkpoints as artifacts
- Log sample predictions every N epochs
- Create dashboard config for key metrics"
```

---

## 🧠 Model Development

### Implement Standard Architecture
```
"Implement [ARCHITECTURE] for [TASK]:
- Input shape: [B, C, H, W] or [B, seq_len, dim]
- Output shape: [B, num_classes] or custom
- Use PyTorch nn.Module
- Include:
  1. Model class with configurable parameters
  2. Forward method with shape comments
  3. Weight initialization (kaiming/xavier as appropriate)
  4. Test script verifying shapes
  5. Parameter count (~[EXPECTED] params)

Example test:
x = torch.randn([BATCH_SIZE], [INPUT_SHAPE])
y = model(x)
assert y.shape == ([BATCH_SIZE], [OUTPUT_SHAPE])"
```

### Custom Layer
```
"Implement custom [LAYER_NAME] layer:

Functionality: [DESCRIPTION]

Input: shape [INPUT_SHAPE], dtype [DTYPE]
Output: shape [OUTPUT_SHAPE], dtype [DTYPE]

Parameters:
- [param1]: [description]
- [param2]: [description]

Include:
- Efficient implementation (vectorized)
- Backward pass (automatic with PyTorch)
- Unit test with gradient check
- Docstring with math notation if applicable"
```

---

## 📊 Data Pipeline

### Standard Data Loader
```
"Create data loader for [DATASET]:

Dataset location: [PATH]
Format: [images/text/etc.]
Split: [train/val/test] [RATIOS]

Training augmentation:
- [aug1]: [params]
- [aug2]: [params]
- Normalization: mean=[MEAN], std=[STD]

Validation: resize/crop + normalize only

DataLoader config:
- Batch size: [BATCH_SIZE]
- Workers: [NUM_WORKERS]
- Pin memory: True
- Shuffle: True (train), False (val/test)

Include:
1. Dataset class
2. Data loaders for each split
3. Visualization function (show batch)
4. Statistics computation (mean, std, class distribution)"
```

### Custom Dataset
```
"Create custom PyTorch Dataset for [DESCRIPTION]:

Data source: [FILES/API/DATABASE]
Loading logic: [HOW TO LOAD]
Preprocessing: [STEPS]

Requirements:
- __len__() returns dataset size
- __getitem__(idx) returns (data, label)
- Handle missing/corrupted data gracefully
- Cache preprocessed data if beneficial
- Support transforms parameter

Include example usage and test with DataLoader."
```

---

## 🎯 Training

### Basic Training Loop
```
"Implement training loop:

Model: [MODEL_CLASS]
Data: [DATASET]
Epochs: [N]

Loss: [LOSS_FUNCTION]
Optimizer: [OPTIMIZER](lr=[LR], [OTHER_PARAMS])
LR Schedule: [SCHEDULE]

Features:
- Progress bars (tqdm)
- Train one epoch, validate one epoch
- Print epoch summary
- Save best model checkpoint
- Early stopping (patience=[N])
- Gradient clipping: max_norm=[N]

Logging:
- Batch loss every [N] batches
- Epoch metrics: train/val loss, accuracy
- Learning rate per epoch"
```

### Advanced Training Loop
```
"Implement production training loop:

Core:
- Mixed precision (AMP)
- Gradient accumulation (effective batch=[N])
- Distributed training support (DDP)
- Resume from checkpoint

Monitoring:
- TensorBoard/W&B logging
- Gradient norms tracking
- Learning rate tracking
- GPU memory usage

Robustness:
- NaN/Inf detection → graceful exit
- CUDA OOM → suggestion to reduce batch size
- Checkpoint on interrupt (Ctrl+C)
- Detailed error logging

Validation:
- Every epoch on full val set
- Metrics: [LIST_METRICS]
- Save best K checkpoints"
```

---

## 🐛 Debugging

### Debug Training Issue
```
"Debug [ISSUE] in training:

Symptoms:
- [Observable behavior]
- Occurs: [when/how often]

Context:
- Model: [ARCHITECTURE]
- Dataset: [NAME], [SIZE]
- Hyperparameters: [LIST]
- Environment: [GPU/CPU, PyTorch version]

What I've tried:
- [Attempt 1]: [Result]
- [Attempt 2]: [Result]

Please:
1. Hypothesize likely causes
2. Add diagnostic logging to pinpoint issue
3. Suggest fixes in priority order
4. Explain why each fix should help

Add checks for:
- Gradient flow (print norms)
- Data sanity (visualize samples)
- Model outputs (check ranges)
- Loss computation (verify correctness)"
```

### Profile Performance
```
"Profile training performance to find bottlenecks:

Current: [TIME] per epoch/batch
Target: [TIME] per epoch/batch
Hardware: [GPU/CPU specs]

Profile:
1. Data loading time
2. Forward pass time
3. Backward pass time
4. Optimizer step time
5. GPU utilization
6. CPU utilization
7. Memory usage

Use PyTorch profiler to get detailed breakdown.
Identify top 3 bottlenecks and suggest optimizations."
```

---

## ⚡ Optimization

### Speed Up Training
```
"Optimize training for [N]x speedup:

Current setup:
- Time: [CURRENT_TIME] per epoch
- Throughput: [IMAGES/SEC]
- GPU util: [PERCENT]%

Apply optimizations:
1. Mixed precision (torch.cuda.amp)
2. Gradient checkpointing (if memory-bound)
3. torch.compile (PyTorch 2.0+)
4. Optimize data loading:
   - Increase workers
   - Prefetch factor
   - Pin memory
5. Use fused optimizers
6. Channels-last memory format

For each:
- Show code changes
- Benchmark improvement
- Note any trade-offs

Target: [TARGET_TIME] per epoch"
```

### Reduce Memory Usage
```
"Reduce GPU memory usage from [CURRENT]GB to [TARGET]GB:

Current config:
- Model: [ARCHITECTURE]
- Batch size: [N]
- Sequence length: [N] (if applicable)

Try:
1. Gradient accumulation (reduce batch, accumulate)
2. Gradient checkpointing (recompute vs store)
3. Mixed precision (FP16 uses less memory)
4. Optimize data types
5. Clear cache between iterations
6. Model parallelism if needed

Show memory before/after each optimization.
Maintain accuracy within [THRESHOLD]%."
```

---

## 📈 Evaluation

### Comprehensive Evaluation
```
"Create evaluation suite for [MODEL]:

Metrics to compute:
- [Metric 1]: [description]
- [Metric 2]: [description]
- [Metric 3]: [description]

Analyses:
1. Overall performance on test set
2. Per-class performance
3. Confusion matrix (if classification)
4. Error analysis (what does model get wrong?)
5. Failure cases visualization

Output:
- Printed summary table
- Saved plots (confusion matrix, metric curves)
- CSV with detailed results
- HTML report with visualizations

Load best checkpoint and run on test set."
```

### Compare Models
```
"Compare [N] models on [DATASET]:

Models:
1. [Model 1]: checkpoint at [PATH]
2. [Model 2]: checkpoint at [PATH]
3. [Model 3]: checkpoint at [PATH]

Evaluation:
- Same test set for all
- Metrics: [LIST]
- Inference speed (images/sec)
- Model size (parameters, disk size)
- Memory usage

Create comparison:
- Side-by-side table
- Bar charts for each metric
- Statistical significance tests
- Recommendation based on [CRITERIA]"
```

---

## 🔬 Experimentation

### Hyperparameter Sweep
```
"Set up hyperparameter sweep using [W&B/Optuna/Ray]:

Parameters to tune:
- Learning rate: [RANGE]
- Batch size: [OPTIONS]
- Weight decay: [RANGE]
- [Other params]: [RANGE]

Search strategy: [grid/random/bayesian]
Budget: [N] trials
Metric to optimize: [METRIC]
Goal: [maximize/minimize]

For each trial:
- Train for [N] epochs
- Log metrics to W&B
- Save best checkpoint
- Early stop if clearly bad

After sweep:
- Analyze results (importance plots)
- Show best configuration
- Compare top K runs"
```

### Ablation Study
```
"Design ablation study for [MODEL/TECHNIQUE]:

Baseline: [FULL MODEL CONFIG]

Ablations (remove one at a time):
1. [-Feature 1]: [Description]
2. [-Feature 2]: [Description]
3. [-Feature 3]: [Description]

For each variant:
- Train with same hyperparameters
- Train for [N] epochs
- Evaluate on same test set
- Record: [METRICS]

Create results table showing:
- Baseline performance
- Each ablation's performance
- Difference from baseline
- Statistical significance

Visualize impact of each component."
```

---

## 🚀 Deployment

### Export Model
```
"Export trained model for deployment:

Source: checkpoint at [PATH]
Target formats:
1. TorchScript (mobile/C++)
2. ONNX (cross-platform)
3. TensorRT (NVIDIA GPU)

For each format:
- Convert model
- Verify outputs match original (tolerance=[EPSILON])
- Benchmark inference speed
- Measure model size
- Test with sample inputs

Include:
- Conversion scripts
- Inference example code
- Performance comparison table
- Deployment instructions"
```

### Create API Service
```
"Create inference API using FastAPI:

Model: [MODEL], checkpoint: [PATH]
Task: [DESCRIPTION]

API endpoints:
POST /predict
- Input: [FORMAT] (image/text/etc.)
- Output: [FORMAT] (predictions, confidence, etc.)

GET /health
- Returns service status

POST /batch_predict (optional)
- Batch inference for throughput

Features:
- Input validation
- Preprocessing pipeline
- Batching for efficiency
- Error handling
- Request logging
- Prometheus metrics (latency, throughput)

Include:
- FastAPI app
- Dockerfile
- docker-compose.yml
- Test client script
- API documentation (auto-generated)"
```

---

## 📝 Documentation

### Add Comprehensive Docs
```
"Document [MODULE/PROJECT]:

Code documentation:
- Docstrings (Google/NumPy style) for all functions/classes
- Type hints for parameters and returns
- Example usage in docstrings

Project documentation:
- README.md: overview, installation, quick start
- docs/architecture.md: model architecture details
- docs/training.md: how to train
- docs/inference.md: how to use for prediction
- docs/api.md: API reference

Include:
- Requirements and setup
- Example commands
- Expected outputs
- Troubleshooting common issues
- Links to papers/references"
```

---

## 🧪 Testing

### Unit Tests
```
"Create unit tests for [MODULE]:

Test coverage:
1. Model architecture:
   - test_forward_pass_shapes
   - test_backward_pass_gradients
   - test_deterministic_output
   - test_batch_size_variations

2. Data pipeline:
   - test_dataset_length
   - test_data_loading
   - test_augmentation_ranges
   - test_batch_shapes

3. Utilities:
   - test_[function_name] for each util

Use pytest, include:
- Fixtures for common objects
- Parametrized tests for multiple cases
- Clear assertion messages
- Fast execution (<30 seconds total)

Run with: pytest tests/ -v"
```

---

## 💾 Checkpointing

### Save/Load Checkpoint
```
"Implement robust checkpointing:

Save:
- Model state_dict
- Optimizer state_dict
- LR scheduler state_dict
- Epoch number
- Best metric value
- Random states (Python, NumPy, PyTorch)
- Hyperparameters
- Git commit hash

Location: [PATH]/checkpoint_epoch{epoch}.pt

Features:
- Save every [N] epochs
- Keep best [K] checkpoints by [METRIC]
- Atomic save (temp file + rename)
- Include metadata (timestamp, hostname)

Load:
- Resume training from exact state
- Load for inference (model only)
- Handle missing keys gracefully"
```

---

## 🎨 Visualization

### Training Curves
```
"Create training visualization:

Plot from W&B logs or CSV:
1. Training curves:
   - Loss (train vs val) over epochs
   - Accuracy (train vs val) over epochs
   - Learning rate over epochs

2. Analysis plots:
   - Loss distribution per epoch
   - Gradient norms over time
   - Sample predictions (correct vs incorrect)

Style:
- Clear labels and legends
- Grid for readability
- Different colors for train/val
- Mark best epoch

Save as high-res PNG and PDF.
Include in report/paper."
```

---

## 🔗 Quick Tips

| Task | Quick Prompt Start |
|------|-------------------|
| Implement architecture | "Create [MODEL] for [TASK] with..." |
| Debug issue | "Debug [SYMPTOM]. Context: ... What I've tried: ..." |
| Optimize speed | "Profile and optimize [COMPONENT] for 2x speedup..." |
| Add logging | "Integrate W&B logging for..." |
| Create dataset | "Create PyTorch Dataset for [DATA] with..." |
| Export model | "Export to [FORMAT] and verify..." |
| Write tests | "Create pytest tests for [MODULE] covering..." |
| Add docs | "Document [CODE] with docstrings and README..." |

---

## 📱 Usage

1. **Find your task** in the table of contents
2. **Copy the prompt template**
3. **Fill in [PLACEHOLDERS]** with your specifics
4. **Paste into Claude Code**
5. **Iterate** as needed

---

**Tip**: Bookmark this page for quick access during development!
