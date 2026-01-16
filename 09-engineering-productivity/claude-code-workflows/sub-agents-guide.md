# Claude Code Sub-Agents Guide

## Overview

Claude Code sub-agents (via the Task tool) allow you to spawn independent AI agents that work autonomously on specific tasks. This is powerful for complex, multi-step workflows where you want parallel execution or specialized focus.

---

## 🤖 What Are Sub-Agents?

### Concept

Sub-agents are independent Claude instances that:
- Work autonomously on assigned tasks
- Have their own context and memory
- Can use tools (Read, Write, Edit, Bash, Grep, Glob)
- Report back results when complete
- Can run in parallel for efficiency

### Available Agent Types

Claude Code provides several specialized sub-agent types:

1. **`general-purpose`** - Full-capability agent for complex tasks
   - Tools: All tools (Read, Write, Edit, Bash, Grep, Glob, etc.)
   - Use for: Multi-step research, implementation, exploration

2. **`Explore`** - Fast agent for codebase exploration
   - Tools: Glob, Grep, Read, Bash
   - Thoroughness levels: `quick`, `medium`, `very thorough`
   - Use for: Finding files, searching code, understanding structure

3. **`statusline-setup`** - Configure status line
   - Tools: Read, Edit
   - Use for: Customizing Claude Code status line

4. **`output-style-setup`** - Configure output styling
   - Tools: Read, Write, Edit, Glob, Grep
   - Use for: Customizing output appearance

---

## 🚀 Basic Usage

### Syntax

```
You: "Use the Task tool to [description]"
```

Or be more explicit:

```
You: "Launch a sub-agent to [task description].
     Use subagent_type: [type]
     Prompt: [detailed instructions]"
```

### Simple Example

```
You: "Launch a general-purpose sub-agent to find all PyTorch model
     definitions in the codebase and create a summary document."
```

Claude will:
1. Spawn the sub-agent
2. Sub-agent searches codebase
3. Sub-agent creates summary
4. Reports back results

---

## 🎯 When to Use Sub-Agents

### ✅ Good Use Cases

**1. Parallel Tasks**
```
You: "Launch 3 sub-agents in parallel:
     1. Search for all CNN architectures
     2. Search for all RNN architectures
     3. Search for all Transformer architectures
     Each should create a summary file."
```

**2. Complex Multi-Step Tasks**
```
You: "Launch a sub-agent to:
     1. Search codebase for training scripts
     2. Identify common patterns
     3. Create a template training script
     4. Document best practices found"
```

**3. Deep Code Exploration**
```
You: "Use Explore agent with 'very thorough' to find all places
     where data augmentation is implemented across the codebase."
```

**4. Independent Research**
```
You: "Launch sub-agent to research how other projects implement
     learning rate scheduling. Check multiple files and create
     comparison document."
```

### ❌ When NOT to Use Sub-Agents

**1. Simple Single-File Tasks**
```
❌ "Launch sub-agent to read config.yaml"
✅ Just use: Read tool directly
```

**2. Tasks Needing Your Input**
```
❌ "Launch sub-agent to implement feature (may need clarification)"
✅ Work directly with main Claude for interactive tasks
```

**3. Quick Lookups**
```
❌ "Launch sub-agent to find class Foo"
✅ Use Glob tool directly: "**/*Foo*.py"
```

---

## 📋 Detailed Sub-Agent Types

### 1. General-Purpose Agent

**Capabilities**: Full toolset for complex tasks

**Example: Code Analysis**
```
You: "Launch a general-purpose sub-agent to:

Task: Analyze all training loops in the codebase

Instructions:
1. Find all files containing training loops (search for 'train', 'epoch', 'optimizer')
2. For each training loop:
   - Identify optimizer used
   - Check if learning rate scheduling exists
   - Check for gradient clipping
   - Check for mixed precision
3. Create analysis.md with:
   - Summary table of all training loops
   - Common patterns
   - Missing best practices
   - Recommendations

Be thorough - check /train, /experiments, /scripts directories."
```

**Example: Implementation Task**
```
You: "Launch a general-purpose sub-agent to:

Task: Implement data augmentation utilities

Requirements:
1. Create utils/augmentation.py
2. Implement:
   - random_crop_and_flip(image, size)
   - color_jitter(image, brightness, contrast)
   - mixup(images, labels, alpha)
3. Add docstrings and type hints
4. Create test file tests/test_augmentation.py
5. Verify all functions work with dummy data

Return: Summary of what was implemented and test results"
```

### 2. Explore Agent

**Capabilities**: Fast codebase exploration with thoroughness control

**Thoroughness Levels:**
- `quick`: Basic search, first few matches
- `medium`: Moderate exploration, multiple patterns
- `very thorough`: Comprehensive search, multiple locations

**Example: Quick Search**
```
You: "Use Explore agent with 'quick' thoroughness to find
     where ResNet is defined in the codebase."
```

**Example: Medium Search**
```
You: "Use Explore agent with 'medium' thoroughness to find
     all data loading code and identify the data pipeline structure."
```

**Example: Thorough Analysis**
```
You: "Use Explore agent with 'very thorough' to:

     Find ALL instances of:
     - Model architectures (CNN, RNN, Transformer)
     - Training configurations
     - Hyperparameter definitions

     Search in: models/, configs/, scripts/, notebooks/

     Create a comprehensive map of the codebase structure."
```

---

## 🔥 Advanced Patterns

### Pattern 1: Parallel Research

**Use Case**: Research multiple topics simultaneously

```
You: "Launch 3 general-purpose sub-agents IN PARALLEL:

Agent 1: Research data augmentation
- Find all augmentation implementations
- Document techniques used
- Save to docs/augmentation_research.md

Agent 2: Research training techniques
- Find all training loop implementations
- Document optimizer choices, LR schedules
- Save to docs/training_research.md

Agent 3: Research evaluation metrics
- Find all metric implementations
- Document what metrics are used where
- Save to docs/metrics_research.md

Run all three simultaneously for speed."
```

### Pattern 2: Pipeline Tasks

**Use Case**: Sequential tasks where each depends on previous

```
You: "Launch a general-purpose sub-agent for multi-step pipeline:

Step 1: Find all model definitions
Step 2: For each model, extract architecture details
Step 3: Create comparison table
Step 4: Identify which models are currently being used
Step 5: Recommend which models to keep/archive

Save final report to docs/model_audit.md"
```

### Pattern 3: Code Refactoring

**Use Case**: Large-scale refactoring

```
You: "Launch a general-purpose sub-agent to refactor training code:

Task: Extract common training logic into reusable functions

Instructions:
1. Read all training scripts in /experiments
2. Identify common patterns (training loop, validation, checkpointing)
3. Create utils/training_utils.py with:
   - train_one_epoch(model, loader, optimizer, criterion)
   - validate(model, loader, criterion)
   - save_checkpoint(model, optimizer, epoch, path)
4. Update ONE example script to use new utils
5. Document the refactoring in REFACTOR.md

DO NOT modify all scripts (too risky), just create utils and show example."
```

### Pattern 4: Testing & Validation

**Use Case**: Comprehensive testing

```
You: "Launch a general-purpose sub-agent to create test suite:

Task: Add unit tests for data pipeline

Requirements:
1. Read existing data loading code in data/
2. Create tests/test_data_pipeline.py
3. Write tests for:
   - Dataset __len__ and __getitem__
   - DataLoader batching
   - Augmentation correctness
   - Data normalization
4. Run tests with pytest
5. Report: test coverage and any issues found"
```

---

## 💡 Best Practices

### 1. Clear Instructions

**❌ Vague:**
```
"Launch agent to check the code"
```

**✅ Specific:**
```
"Launch general-purpose agent to:
1. Check all training scripts for proper error handling
2. List files missing try-catch blocks
3. Suggest where to add error handling
4. Create checklist in error_handling_audit.md"
```

### 2. Scope Appropriately

**❌ Too Broad:**
```
"Launch agent to improve entire codebase"
```

**✅ Focused:**
```
"Launch agent to improve error handling in data loading code:
- Focus on data/ directory only
- Add try-catch for file operations
- Add validation for image formats
- Log errors properly"
```

### 3. Specify Output Format

**❌ No output spec:**
```
"Launch agent to analyze models"
```

**✅ Clear output:**
```
"Launch agent to analyze models.
Create models_analysis.md with:
- Table: model name, parameters, input/output shapes
- Section: Which models are actively used
- Section: Recommendations
Use markdown tables for easy reading."
```

### 4. Set Boundaries

**Include constraints:**
```
"Launch agent to refactor code:

⚠️ CONSTRAINTS:
- DO NOT modify production code in main branch
- Create new files, don't edit existing
- Test all changes before reporting
- If unsure, ask rather than guess

Task: [detailed task]"
```

---

## 🎨 ML-Specific Use Cases

### Use Case 1: Model Architecture Search

```
You: "Launch Explore agent with 'very thorough' to:

Find all CNN architectures in codebase.

For each architecture found:
- File path and line number
- Architecture name (ResNet, VGG, etc.)
- Number of layers
- Parameters if mentioned

Create architecture_inventory.md with findings.

Search in: models/, experiments/, notebooks/"
```

### Use Case 2: Hyperparameter Audit

```
You: "Launch general-purpose agent to:

Audit all hyperparameters across experiments.

1. Search for: learning rates, batch sizes, epochs, optimizers
2. Find in: config files, training scripts, notebooks
3. Create hyperparameters.csv with:
   - experiment_name, lr, batch_size, optimizer, epochs, file_path
4. Identify inconsistencies
5. Suggest standardization

Focus on /experiments and /configs directories."
```

### Use Case 3: Data Pipeline Analysis

```
You: "Launch general-purpose agent to:

Analyze data loading performance across all training scripts.

For each script:
1. Identify data loading approach (DataLoader settings)
2. Check: num_workers, pin_memory, prefetch_factor
3. Note if using caching or preprocessing
4. Estimate if optimal for hardware

Create data_pipeline_report.md with:
- Current state table
- Bottleneck identification
- Optimization recommendations

Test scripts: /experiments/*/train.py"
```

### Use Case 4: Experiment Tracking Audit

```
You: "Launch general-purpose agent to:

Audit experiment tracking across projects.

Find all places where experiments are logged:
- Weights & Biases usage
- MLflow usage
- TensorBoard usage
- Manual CSV logging

For each:
- What metrics are logged
- How consistently used
- Missing best practices

Create experiment_tracking_audit.md with:
- Current state
- Coverage gaps
- Standardization recommendations"
```

### Use Case 5: Parallel Paper Implementation

```
You: "Launch 3 general-purpose agents IN PARALLEL:

Agent 1: Implement attention mechanism from paper
- Read paper_notes.md for equations
- Implement MultiHeadAttention class
- Test with dummy inputs
- Save to models/attention.py

Agent 2: Implement positional encoding from paper
- Read paper_notes.md for details
- Implement PositionalEncoding class
- Test with various sequence lengths
- Save to models/positional.py

Agent 3: Create training data pipeline for paper
- Read paper for data preprocessing details
- Implement data loader
- Test on sample data
- Save to data/paper_dataset.py

All agents work simultaneously on different components."
```

---

## 🐛 Troubleshooting

### Issue: Sub-Agent Not Finding Files

**Problem:**
```
Agent reports: "No relevant files found"
```

**Solution:**
```
Be more specific with paths and patterns:

❌ "Find training code"
✅ "Find training code. Search patterns:
   - **/*train*.py
   - **/training/*.py
   - **/experiments/*/train.py
   Look in: /, /scripts, /experiments directories"
```

### Issue: Sub-Agent Output Too Long

**Problem:**
```
Agent creates huge report, hard to parse
```

**Solution:**
```
Request structured, concise output:

"Create summary.md with:
- Executive summary (3 bullet points max)
- Key findings table (max 10 rows)
- Top 3 recommendations
- Detailed findings in appendix

Keep main content under 100 lines."
```

### Issue: Sub-Agent Task Takes Too Long

**Problem:**
```
Agent running for many minutes
```

**Solution:**
```
Scope down or use faster agent:

❌ "Use general-purpose to search entire codebase"
✅ "Use Explore agent with 'quick' to search specific directory"

Or split into smaller tasks:
"First, use Explore to identify relevant files.
Then I'll launch targeted sub-agents for each."
```

---

## 📊 Performance Comparison

| Task | Direct (You + Claude) | With Sub-Agent | Best Choice |
|------|----------------------|----------------|-------------|
| Single file read | 1 message | 1 message + agent | Direct |
| Search for pattern | 2-3 messages | 1 message + agent | Sub-agent |
| Complex analysis | 5-10 messages | 1 message + agent | Sub-agent |
| 3 parallel tasks | Sequential (slow) | Parallel (fast) | Sub-agent |
| Interactive task | Fast (immediate feedback) | Slow (no interaction) | Direct |

---

## 🎯 Quick Reference

### Launch Sub-Agent Template

```
You: "Launch a [AGENT_TYPE] sub-agent to:

Task: [One-line description]

Instructions:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Constraints:
- [Constraint 1]
- [Constraint 2]

Output:
- [File to create or format for results]

Context:
- [Any relevant background]
- [File paths to focus on]"
```

### Agent Type Selection

```
📝 Simple search → Explore (quick)
🔍 Thorough search → Explore (very thorough)
🛠️ Implementation → general-purpose
📊 Analysis → general-purpose
⚡ Parallel tasks → Multiple general-purpose agents
```

---

## 🚀 Advanced: Custom Sub-Agent Workflows

### Workflow: Code Quality Audit

```
You: "Launch 4 general-purpose sub-agents IN PARALLEL for code audit:

Agent 1 - Documentation Audit:
- Find functions missing docstrings
- Check if type hints present
- Create docs_audit.md

Agent 2 - Testing Audit:
- Find code files missing tests
- Calculate test coverage estimate
- Create testing_audit.md

Agent 3 - Performance Audit:
- Find potential bottlenecks
- Identify unoptimized loops
- Create performance_audit.md

Agent 4 - Security Audit:
- Find hardcoded secrets/paths
- Check input validation
- Create security_audit.md

All run in parallel. Report back when all complete."
```

### Workflow: Experiment Reproduction

```
You: "Launch general-purpose sub-agent for experiment reproduction:

Task: Set up reproduction of paper experiment

Steps:
1. Read paper_notes.md for experiment details
2. Find closest existing training script
3. Create new script experiments/paper_reproduction/train.py
4. Configure hyperparameters to match paper
5. Create README with:
   - Paper details
   - Expected results
   - How to run
   - Differences from paper (if any)
6. DO NOT run training (just setup)

Report back with setup summary and any issues."
```

---

## 📚 Real Example: Using Sub-Agent for CNN Project

```
You: "I'm starting a new CNN project for image classification.
     Use sub-agents to help set up the project efficiently.

     Launch 3 general-purpose sub-agents IN PARALLEL:

Agent 1 - Project Structure:
- Create directory structure:
  - models/
  - data/
  - configs/
  - experiments/
  - tests/
  - utils/
- Create __init__.py files where needed
- Create .gitignore for Python/PyTorch
- Create requirements.txt with: torch, torchvision, tqdm, wandb, pytest

Agent 2 - Boilerplate Code:
- Create models/base_cnn.py with base CNN class (empty)
- Create data/dataset.py with dataset template (empty)
- Create utils/training.py with training utilities template
- Add docstrings to all files

Agent 3 - Documentation:
- Create README.md with project structure
- Create docs/setup.md with installation instructions
- Create docs/training.md with training guide template
- Create CHANGELOG.md

All agents work simultaneously. When done, I'll have a
complete project skeleton ready for development."
```

---

## ✅ Summary

**Sub-agents are powerful for:**
- ✅ Parallel execution of independent tasks
- ✅ Complex multi-step research/analysis
- ✅ Deep codebase exploration
- ✅ Large-scale refactoring planning
- ✅ Comprehensive audits

**Use direct interaction for:**
- ✅ Simple single-file operations
- ✅ Interactive debugging
- ✅ Tasks requiring your input
- ✅ Learning and exploration

**Key Takeaway**: Sub-agents are your parallel workforce for complex, independent tasks. Use them to multiply your productivity!

---

**Next**: Try the [Multi-Agent Workflow Examples](./multi-agent-workflows.md) for hands-on practice.
