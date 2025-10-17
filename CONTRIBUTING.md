# Contributing to LLM Engineering Journey

Thank you for your interest in contributing to this repository! This document provides guidelines and best practices for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Contribution Guidelines](#contribution-guidelines)
- [Style Guide](#style-guide)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) before contributing.

## 🤝 How Can I Contribute?

### Reporting Bugs

- Check if the issue has already been reported
- Use a clear and descriptive title
- Provide detailed steps to reproduce the issue
- Include relevant code snippets or examples

### Suggesting Enhancements

- Use a clear and descriptive title
- Provide a detailed description of the proposed enhancement
- Explain why this enhancement would be useful
- Include examples of how the feature would be used

### Contributing Content

- **Documentation**: Improve existing docs or add new documentation
- **Code Examples**: Add implementations or experiments
- **Paper Summaries**: Contribute paper reviews and summaries
- **Tutorials**: Create educational content and notebooks
- **Projects**: Add end-to-end project examples

## 🚀 Getting Started

1. **Fork the repository** to your GitHub account
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/llm-engineering-journey.git
   cd llm-engineering-journey
   ```
3. **Create a new branch** for your contribution:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following the guidelines below
5. **Test your changes** to ensure everything works correctly
6. **Commit your changes** with clear, descriptive messages
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Open a Pull Request** from your fork to the main repository

## ✅ Contribution Guidelines

### Documentation

- Use clear, concise language
- Include code examples where applicable
- Add references to papers or resources
- Follow the existing markdown structure
- Ensure proper formatting and readability

### Code

- Follow Python best practices (PEP 8)
- Include docstrings for functions and classes
- Add type hints where appropriate
- Include comments for complex logic
- Ensure code is well-tested

### Notebooks

- Use clear markdown cells to explain concepts
- Include visualizations where helpful
- Ensure notebooks run from top to bottom
- Clear all outputs before committing
- Add requirements at the top of the notebook

### Paper Summaries

- Include full citation
- Provide clear overview and key contributions
- Explain methodology and results
- Discuss practical implications
- Link to original paper and code (if available)

## 📝 Style Guide

### Markdown

- Use `#` for main titles, `##` for sections, `###` for subsections
- Use code blocks with language specification: \```python
- Use bullet points for lists
- Use **bold** for emphasis, *italics* for slight emphasis
- Include blank lines between sections

### Python Code

- Follow PEP 8 style guide
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all public functions

Example:
```python
def calculate_attention_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """
    Calculate attention scores between query and key vectors.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len, d_model)
        key: Key tensor of shape (batch_size, seq_len, d_model)
    
    Returns:
        Attention scores of shape (batch_size, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    return scores
```

## 💬 Commit Message Guidelines

Follow the conventional commits specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature or content
- `fix`: Bug fix or correction
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no code change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(transformers): add multi-head attention implementation

Add detailed implementation of multi-head attention with
comprehensive docstrings and example usage.

Closes #123
```

```
docs(rag): update vector database comparison

Add comparison table for different vector databases including
performance metrics and use cases.
```

## 🔄 Pull Request Process

1. **Update documentation** if you've added new features
2. **Add tests** for new functionality
3. **Ensure all tests pass** before submitting
4. **Update CHANGELOG.md** with your changes
5. **Link relevant issues** in your PR description
6. **Request review** from maintainers
7. **Address feedback** promptly and professionally

### PR Title Format

Use the same format as commit messages:
```
feat(scope): brief description
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

## Changes Made
- List key changes
- Include details

## Testing
- Describe testing performed
- Include test results

## Related Issues
Closes #(issue number)
```

## ❓ Questions?

If you have questions about contributing, please:
- Check existing documentation
- Review closed issues and PRs
- Open a new issue with the `question` label
- Contact maintainers directly

## 🙏 Thank You!

Your contributions help make this repository a valuable resource for the ML community. Every contribution, no matter how small, is appreciated!

---

*Last Updated: 2025-01-17*