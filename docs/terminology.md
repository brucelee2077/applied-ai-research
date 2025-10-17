# 📖 Terminology

A comprehensive glossary of terms, acronyms, and concepts used in Large Language Model engineering.

## Overview

This glossary provides definitions for key terms used throughout this repository. Terms are organized alphabetically within categories for easy reference.

---

## Neural Networks Fundamentals

### A-Z

**Activation Function**
A non-linear function applied to neuron outputs. Common examples: ReLU, sigmoid, tanh, GELU.

**Backpropagation**
Algorithm for computing gradients of the loss function with respect to network weights using the chain rule.

**Batch Normalization**
Technique that normalizes layer inputs across a mini-batch to stabilize and accelerate training.

**CNN (Convolutional Neural Network)**
Neural network architecture using convolutional layers, primarily used for image processing and spatial data.

**Dropout**
Regularization technique that randomly deactivates neurons during training to prevent overfitting.

**Epoch**
One complete pass through the entire training dataset.

**Gradient Descent**
Optimization algorithm that iteratively updates model parameters in the direction of steepest descent.

**Hidden Layer**
Neural network layer between input and output layers that learns intermediate representations.

**Layer Normalization**
Normalization technique that normalizes across features for each sample independently.

**Learning Rate**
Hyperparameter controlling the step size in gradient descent optimization.

**Loss Function**
Function that measures the difference between predicted and actual outputs.

**Optimizer**
Algorithm for updating model parameters based on gradients (e.g., SGD, Adam, AdamW).

**RNN (Recurrent Neural Network)**
Neural network architecture with recurrent connections, designed for sequential data.

**Weight Initialization**
Method for setting initial values of model parameters (e.g., Xavier, He initialization).

---

## Transformer Architecture

**Attention Mechanism**
Method for computing weighted importance of different parts of the input sequence.

**Cross-Attention**
Attention mechanism where queries come from one sequence and keys/values from another.

**Decoder**
Transformer component that generates output sequences autoregressively.

**Encoder**
Transformer component that processes input sequences into contextual representations.

**Multi-Head Attention**
Parallel attention mechanisms that learn different representation subspaces.

**Positional Encoding**
Method for incorporating sequence position information into transformer inputs.

**Query, Key, Value (Q, K, V)**
Three learned projections used in attention computation.

**Self-Attention**
Attention mechanism where queries, keys, and values all come from the same sequence.

**Token**
Smallest unit of text processed by the model (can be word, subword, or character).

**Tokenization**
Process of splitting text into tokens.

---

## Large Language Models

**BERT (Bidirectional Encoder Representations from Transformers)**
Encoder-only transformer model trained with masked language modeling.

**Causal Language Modeling**
Training objective where the model predicts next tokens based on previous context only.

**Context Window**
Maximum sequence length the model can process at once.

**Embedding**
Dense vector representation of tokens or inputs.

**GPT (Generative Pre-trained Transformer)**
Decoder-only transformer model trained with causal language modeling.

**Masked Language Modeling (MLM)**
Training objective where random tokens are masked and the model predicts them.

**Pre-training**
Initial training phase on large-scale unsupervised data.

**Temperature**
Hyperparameter controlling randomness in text generation (higher = more random).

**Top-k Sampling**
Sampling method that considers only the k most likely next tokens.

**Top-p (Nucleus) Sampling**
Sampling method that considers the smallest set of tokens whose cumulative probability exceeds p.

---

## Fine-Tuning

**Adapter Layers**
Small trainable modules inserted into frozen pre-trained models.

**Catastrophic Forgetting**
Phenomenon where fine-tuning on new tasks degrades performance on original tasks.

**Domain Adaptation**
Adapting a model trained on one domain to perform well on another.

**Few-Shot Learning**
Learning from a small number of examples.

**Full Fine-Tuning**
Updating all model parameters during fine-tuning.

**Instruction Tuning**
Fine-tuning on datasets of instructions and corresponding responses.

**LoRA (Low-Rank Adaptation)**
Parameter-efficient fine-tuning by adding low-rank matrices to frozen weights.

**PEFT (Parameter-Efficient Fine-Tuning)**
Methods that fine-tune only a small subset of parameters.

**QLoRA (Quantized Low-Rank Adaptation)**
LoRA combined with quantization for memory-efficient fine-tuning.

**Transfer Learning**
Leveraging knowledge from pre-trained models for new tasks.

**Zero-Shot Learning**
Model performing tasks without any task-specific training examples.

---

## Retrieval-Augmented Generation (RAG)

**Chunking**
Splitting documents into smaller segments for efficient retrieval.

**Dense Retrieval**
Retrieval using learned dense vector representations.

**Embedding Model**
Model that converts text into dense vector representations.

**FAISS (Facebook AI Similarity Search)**
Library for efficient similarity search and clustering of dense vectors.

**Hybrid Search**
Combining dense and sparse retrieval methods.

**Semantic Search**
Search based on meaning rather than keyword matching.

**Sparse Retrieval**
Traditional keyword-based retrieval (e.g., BM25, TF-IDF).

**Vector Database**
Database optimized for storing and querying high-dimensional vectors.

---

## Prompt Engineering

**Chain-of-Thought (CoT)**
Prompting technique that encourages step-by-step reasoning.

**Context**
Information provided to the model before the actual task.

**Few-Shot Prompting**
Providing a few examples in the prompt before the actual query.

**In-Context Learning**
Model's ability to learn from examples provided in the prompt.

**Prompt**
Input text that instructs the model on what task to perform.

**Prompt Template**
Reusable structure for creating prompts with variable placeholders.

**System Prompt**
Initial instructions that set the model's behavior and persona.

**Zero-Shot Prompting**
Directly asking the model to perform a task without examples.

---

## Model Evaluation

**BLEU (Bilingual Evaluation Understudy)**
Metric for evaluating machine translation quality based on n-gram overlap.

**F1 Score**
Harmonic mean of precision and recall.

**Human Evaluation**
Assessment of model outputs by human annotators.

**Perplexity**
Measure of how well a probability model predicts a sample.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
Metrics for evaluating text summarization quality.

**Benchmark**
Standardized dataset or task for comparing model performance.

---

## Deployment & Optimization

**Distillation**
Training a smaller (student) model to mimic a larger (teacher) model.

**Inference**
Process of using a trained model to make predictions.

**Latency**
Time taken to generate a response.

**Model Compression**
Techniques to reduce model size while maintaining performance.

**ONNX (Open Neural Network Exchange)**
Framework-agnostic format for representing neural networks.

**Pruning**
Removing unnecessary weights or neurons from a model.

**Quantization**
Reducing numerical precision of model weights (e.g., FP32 to INT8).

**Throughput**
Number of requests processed per unit time.

**TensorRT**
NVIDIA's library for high-performance deep learning inference.

**vLLM**
Library for fast LLM inference and serving.

---

## Multimodal

**CLIP (Contrastive Language-Image Pre-training)**
Model trained to understand images and text jointly.

**Cross-Modal**
Involving multiple modalities (e.g., text, image, audio).

**Fusion**
Combining information from multiple modalities.

**Vision Transformer (ViT)**
Transformer architecture adapted for image processing.

**Whisper**
Speech recognition model by OpenAI.

---

## Training & Infrastructure

**Batch Size**
Number of samples processed before updating model parameters.

**Checkpointing**
Saving model state during training for recovery or deployment.

**Distributed Training**
Training across multiple GPUs or machines.

**FLOPs (Floating Point Operations)**
Measure of computational complexity.

**Gradient Accumulation**
Accumulating gradients over multiple batches before updating parameters.

**Gradient Clipping**
Limiting gradient magnitude to prevent exploding gradients.

**Mixed Precision Training**
Training with both FP16 and FP32 precision for speed and stability.

**Overfitting**
Model performing well on training data but poorly on unseen data.

**Regularization**
Techniques to prevent overfitting (e.g., dropout, weight decay).

**Underfitting**
Model failing to capture patterns in training data.

**Warmup**
Gradually increasing learning rate at the start of training.

---

## Common Acronyms

- **AI**: Artificial Intelligence
- **API**: Application Programming Interface
- **BERT**: Bidirectional Encoder Representations from Transformers
- **CNN**: Convolutional Neural Network
- **CPU**: Central Processing Unit
- **CUDA**: Compute Unified Device Architecture (NVIDIA)
- **DL**: Deep Learning
- **GPU**: Graphics Processing Unit
- **HF**: Hugging Face
- **LLM**: Large Language Model
- **ML**: Machine Learning
- **MLOps**: Machine Learning Operations
- **NLP**: Natural Language Processing
- **PEFT**: Parameter-Efficient Fine-Tuning
- **RAG**: Retrieval-Augmented Generation
- **RNN**: Recurrent Neural Network
- **SGD**: Stochastic Gradient Descent
- **TPU**: Tensor Processing Unit
- **VRAM**: Video Random Access Memory

---

## Mathematical Notation

**∇ (nabla)**
Gradient operator

**⊙ (odot)**
Element-wise multiplication

**⊕ (oplus)**
Concatenation or addition

**∑ (sigma)**
Summation

**∏ (pi)**
Product

**||·|| (norm)**
Vector norm

**softmax(x)ᵢ**
exp(xᵢ) / Σⱼ exp(xⱼ)

**Attention(Q,K,V)**
softmax(QKᵀ/√dₖ)V

---

## Resources for More Terms

- [Papers With Code Glossary](https://paperswithcode.com/)
- [ML Glossary](https://ml-cheatsheet.readthedocs.io/)
- [Hugging Face Glossary](https://huggingface.co/docs/transformers/glossary)
- [Deep Learning Book Glossary](https://www.deeplearningbook.org/)

---

*This glossary is continuously updated. If you find missing terms or unclear definitions, please contribute!*