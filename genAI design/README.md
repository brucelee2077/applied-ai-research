# GenAI System Design Interview - Staff Level Study Guide

## What Is This?

Imagine you're building the coolest AI products in the world -- things like ChatGPT, Google Translate, DALL-E, and Sora. This study guide teaches you **exactly how these systems work** from the inside out, so you can ace a Staff-level ML system design interview.

Based on ByteByteGo's "Generative AI System Design Interview" book, every chapter is transformed into **interactive Jupyter notebooks** with:
- Analogies a 12-year-old could understand
- Staff-level technical deep dives with math and code
- Fun quizzes and "explain it to me like I'm 12" sections
- Interview cheat sheets with talking points
- Hands-on code you can run

---

## The Journey Map

Think of GenAI like learning to be a chef. First you learn the basics (what's an oven?), then simple recipes (toast), then complex dishes (souffl), and finally you become a head chef designing entire menus.

| # | Chapter | What You'll Learn | Real Product |
|---|---------|-------------------|-------------|
| 01 | [Intro & Framework](01-intro-and-framework/) | The "recipe template" for ALL GenAI systems | -- |
| 02 | [Gmail Smart Compose](02-gmail-smart-compose/) | Transformers, tokenization, beam search | Gmail |
| 03 | [Google Translate](03-google-translate/) | Encoder-decoder, BPE, BLEU/ROUGE/METEOR | Google Translate |
| 04 | [ChatGPT Chatbot](04-chatgpt-chatbot/) | LLMs, RLHF, top-k/top-p sampling, RoPE | ChatGPT |
| 05 | [Image Captioning](05-image-captioning/) | Vision Transformers, multi-modal, CIDEr | Photo apps |
| 06 | [RAG Systems](06-rag/) | Retrieval, vector DBs, ANN, prompt engineering | Perplexity.ai |
| 07 | [Face Generation](07-realistic-face-generation/) | GANs, adversarial training, FID, mode collapse | Entertainment |
| 08 | [High-Res Images](08-high-res-image-synthesis/) | VQ-VAE, image tokenization, autoregressive gen | -- |
| 09 | [Text-to-Image](09-text-to-image/) | Diffusion models, U-Net, CFG, CLIPScore | DALL-E, Midjourney |
| 10 | [Personalized Headshots](10-personalized-headshots/) | DreamBooth, LoRA, Textual Inversion | AI headshot apps |
| 11 | [Text-to-Video](11-text-to-video/) | Video DiT, temporal attention, FVD | Sora, Movie Gen |

---

## How to Study

### Phase 1: Foundation (Chapters 1-3)
Start here. These chapters build your understanding of Transformers, tokenization, and the interview framework. Everything else builds on this.

### Phase 2: LLM Deep Dive (Chapters 4, 6)
ChatGPT and RAG are the most common interview topics right now. Master these.

### Phase 3: Multi-Modal (Chapters 5, 7)
Image captioning teaches you how to bridge vision and language. GANs teach you adversarial training.

### Phase 4: Diffusion & Advanced (Chapters 8-11)
The cutting edge. Diffusion models, personalization with LoRA/DreamBooth, and video generation.

---

## The GenAI System Design Framework

Every single chapter follows this same 7-step framework. Memorize it:

```
1. Clarifying Requirements     -- "What exactly are we building?"
2. Frame as ML Task            -- "What goes in? What comes out?"
3. Data Preparation            -- "How do we prepare the fuel?"
4. Model Development           -- "What architecture do we use?"
5. Evaluation                  -- "How do we know it's working?"
6. Overall System Design       -- "How do all the pieces fit?"
7. Deployment & Monitoring     -- "How do we keep it running?"
```

---

## Key Concepts Cheat Sheet

### Architectures You Must Know
| Architecture | Used For | Key Chapters |
|---|---|---|
| Decoder-only Transformer | Text generation (GPT, Llama) | 2, 4 |
| Encoder-Decoder Transformer | Translation, seq2seq | 3 |
| Vision Transformer (ViT) | Image encoding | 5, 8 |
| U-Net | Diffusion denoising | 9, 10, 11 |
| DiT (Diffusion Transformer) | Image/video diffusion | 9, 11 |
| GAN (Generator + Discriminator) | Image generation | 7 |
| VAE / VQ-VAE | Image compression/tokenization | 7, 8, 11 |

### Training Strategies
| Strategy | What It Does | Chapters |
|---|---|---|
| Pretraining + Finetuning | General knowledge -> specific task | 2, 3, 5 |
| Pretraining + SFT + RLHF | 3-stage LLM training | 4 |
| Adversarial Training | Generator vs discriminator | 7 |
| Diffusion Training | Learn to denoise | 9, 10, 11 |
| DreamBooth / LoRA | Personalize pretrained models | 10 |

### Evaluation Metrics
| Metric | What It Measures | Domain |
|---|---|---|
| Perplexity | How well model predicts text | Text |
| BLEU | Precision of n-gram matches | Translation |
| ROUGE | Recall of n-gram matches | Summarization |
| METEOR | Semantic-aware translation quality | Translation |
| CIDEr | Caption quality vs references | Captioning |
| FID | Distribution similarity (generated vs real) | Images |
| Inception Score | Quality + diversity of images | Images |
| CLIPScore | Text-image alignment | Text-to-image |
| FVD | Video quality + temporal consistency | Video |

---

## Prerequisites

- Python 3.8+
- PyTorch
- Basic understanding of neural networks (or willingness to learn!)

```bash
pip install torch torchvision numpy matplotlib
```
