# Chapter 09: Text-to-Image Generation -- Painting with Words 🎨✨

## What Is This Chapter About?

Imagine you have a magical sketchpad. You write "a corgi wearing a top hat, riding a skateboard through a neon city at sunset" and *poof* -- a gorgeous, photorealistic image appears. No drawing skills needed. Just words.

**That's text-to-image generation.** And it's powered by one of the most beautiful ideas in modern AI: **diffusion models**.

This chapter breaks down exactly how DALL-E, Midjourney, and Stable Diffusion turn your text prompts into stunning images -- from the math of noise, to the architecture of the neural networks, to how the full production system works.

---

## 🗺️ Chapter Map

| Notebook | Title | What You'll Learn |
|----------|-------|-------------------|
| [01](01_diffusion_models_deep_dive.ipynb) | Diffusion Models: Teaching AI to Paint from Noise | Forward/backward diffusion, noise schedules, U-Net vs DiT, cross-attention, training loss |
| [02](02_sampling_evaluation_system.ipynb) | From Noise to Masterpiece: Sampling, CFG, and System Design | Classifier-free guidance, DDIM, CLIPScore, data pipeline, full system architecture |

---

## 🧠 The Big Picture: What IS Text-to-Image?

### ELI12 Version 🧒

Think of it like this: imagine you have a photo, and you slowly pour sand on it until it's completely buried and you can't see the picture anymore. That's the **forward process** -- turning an image into pure noise (static/snow on a TV).

Now imagine you had a friend who watched you bury hundreds of millions of photos. After watching enough times, your friend gets really good at **un-burying** the photo -- brushing away the sand grain by grain to reveal the image underneath. That's the **backward process**.

The genius part? You can whisper to your friend what the picture *should* look like ("a cat wearing sunglasses") and they'll unbury the sand into THAT specific image. That whisper is your **text prompt**, and the friend is a neural network called **U-Net** (or **DiT**).

### Staff-Level Version 🎓

Text-to-image models are conditional generative models that learn the data distribution p(x|y) where x is an image and y is a text description. The dominant paradigm uses **denoising diffusion probabilistic models (DDPMs)**: during training, Gaussian noise is progressively added to images via a fixed forward process q(x_t|x_0), and a neural network (U-Net or DiT) learns to reverse this process by predicting the noise epsilon_theta(x_t, t, y). At inference, we start from pure Gaussian noise and iteratively denoise, conditioned on the text embedding from a pretrained text encoder (CLIP or T5). Classifier-free guidance amplifies the text conditioning to produce higher-quality, more text-aligned results.

---

## 🏗️ Key Products in This Space

| Product | Company | Architecture | Key Innovation |
|---------|---------|-------------|----------------|
| **DALL-E 2** | OpenAI | CLIP + Diffusion | CLIP-guided prior + diffusion decoder |
| **DALL-E 3** | OpenAI | Diffusion + improved captioning | Better text-image alignment via detailed captions |
| **Midjourney** | Midjourney | Diffusion (proprietary) | Exceptional aesthetic quality |
| **Stable Diffusion** | Stability AI | Latent Diffusion (LDM) | Open-source, operates in latent space (fast!) |
| **Imagen** | Google | T5 text encoder + cascaded diffusion | Large language model as text encoder |
| **Flux** | Black Forest Labs | DiT-based | Scalable transformer architecture |

---

## 🔑 Key Concepts At a Glance

### Diffusion Models: The Core Idea 🌊

| Concept | What It Is | ELI12 Analogy |
|---------|-----------|---------------|
| **Forward Process** | Gradually add Gaussian noise to an image over T steps until it becomes pure noise | 🧊 Watching an ice sculpture slowly melt into a puddle |
| **Backward Process** | Learn to reverse the noise -- predict and remove noise step by step | 🧊 Learning to "un-melt" the puddle back into the sculpture |
| **Noise Schedule (β_t)** | Controls how much noise is added at each step | 🎚️ The thermostat -- how fast you turn up the heat to melt the ice |
| **Timestep Embedding** | Tells the network which step of the process we're at | 📅 A calendar showing "Day 47 of 1000" so the network knows how noisy the image is |
| **ε_θ (Epsilon Theta)** | The neural network that predicts the noise | 🔮 A noise-predicting oracle -- tell it what noise you see, and it predicts what to remove |

### Architecture: U-Net vs DiT 🏛️

| Feature | U-Net | DiT (Diffusion Transformer) |
|---------|-------|---------------------------|
| **Core building block** | Convolutional layers (Conv2D) | Transformer blocks (self-attention) |
| **Structure** | Encoder-decoder with skip connections | Patchify → Transformer → Unpatchify |
| **Text conditioning** | Cross-attention in middle/up blocks | Cross-attention in every Transformer block |
| **Scaling behavior** | Harder to scale (diminishing returns) | Scales like a Transformer (log-linear) |
| **Used by** | Stable Diffusion v1-2, DALL-E 2, Imagen | Stable Diffusion 3, DALL-E 3, Flux, Sora |
| **ELI12** | 🏠 A house with an attic (compress) and basement (expand), with stairs connecting each floor | 🏢 A skyscraper where every floor talks to every other floor via intercom |

### Classifier-Free Guidance (CFG) 🎛️

| Aspect | Details |
|--------|---------|
| **What** | A technique to boost text-image alignment at inference time without needing a separate classifier |
| **How it works** | Train with text condition randomly dropped (replaced with null/empty). At inference, compute BOTH conditioned and unconditioned predictions, then amplify the difference |
| **Formula** | ε_guided = ε_uncond + w * (ε_cond - ε_uncond), where w is the guidance scale |
| **w = 1** | Normal generation (no guidance boost) |
| **w = 7-15** | Typical values -- sharper, more text-aligned images |
| **w > 20** | Oversaturated, artifact-heavy images |
| **ELI12** | 🔊 It's like adjusting a "creativity vs. accuracy" slider. Low = dreamy but might ignore your prompt. High = follows your prompt exactly but might look too intense |

### DDPM vs DDIM Sampling ⚡

| Feature | DDPM | DDIM |
|---------|------|------|
| **Full name** | Denoising Diffusion Probabilistic Models | Denoising Diffusion Implicit Models |
| **Steps needed** | 1000 (slow!) | 20-50 (fast!) |
| **Process** | Stochastic (random noise at each step) | Deterministic (same noise → same image) |
| **Quality** | Best quality (given enough steps) | Near-equal quality with 50x fewer steps |
| **ELI12** | 🐢 Walking through a park, stopping at every single bench (1000 stops) | 🏃 Sprinting through the park, only stopping at the important benches (20 stops) |

### CLIPScore: Measuring Text-Image Alignment 📊

| Aspect | Details |
|--------|---------|
| **What** | Cosine similarity between CLIP text embedding and CLIP image embedding |
| **Range** | Typically 0.2-0.4 (higher = better alignment) |
| **How it works** | CLIP was trained on 400M text-image pairs to align text and image in the same embedding space. If the generated image matches the text, their embeddings will be close |
| **ELI12** | 📐 Imagine text and images live in the same room. CLIPScore measures how close the text description is standing to the generated image. Close together = good match! |

---

## 📊 Evaluation Metrics Summary

| Metric | What It Measures | Introduced In | Used For |
|--------|-----------------|---------------|----------|
| **FID** | Distribution similarity between generated and real images | Ch 07 | Overall image quality |
| **Inception Score (IS)** | Image quality + diversity | Ch 07 | Overall generation quality |
| **CLIPScore** | Text-image alignment (cosine similarity) | **This chapter** | Does the image match the prompt? |
| **DrawBench** | Human evaluation on challenging prompts | This chapter | Compositional/spatial reasoning |
| **Human Preference** | Side-by-side preference from real people | This chapter | Overall perceived quality |

---

## 🏗️ System Design: Text-to-Image End-to-End

```
┌───────────────────────────────────────────────────────────────────────┐
│                   TEXT-TO-IMAGE SYSTEM DESIGN                         │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  USER PROMPT: "A corgi wearing a top hat, oil painting style"        │
│       │                                                               │
│       ▼                                                               │
│  ┌──────────────────┐                                                 │
│  │  Prompt Safety   │ ── Block harmful/NSFW/adversarial prompts      │
│  │  Filter          │    (classifier + blocklist)                     │
│  └────────┬─────────┘                                                 │
│           │                                                           │
│           ▼                                                           │
│  ┌──────────────────┐                                                 │
│  │  Prompt          │ ── Rewrite/expand prompt for better quality     │
│  │  Enhancement     │    "...highly detailed, 4k, trending on        │
│  │  (LLM-based)     │     artstation, soft lighting"                  │
│  └────────┬─────────┘                                                 │
│           │                                                           │
│           ▼                                                           │
│  ┌──────────────────┐     ┌──────────────────┐                        │
│  │  Text Encoder    │────▶│  Diffusion Model │                        │
│  │  (CLIP / T5)     │     │  (U-Net or DiT)  │                        │
│  └──────────────────┘     │  + CFG w=7.5     │                        │
│                           │  + DDIM 50 steps │                        │
│                           └────────┬─────────┘                        │
│                                    │ (low-res 64x64 or latent)        │
│                                    ▼                                  │
│                           ┌──────────────────┐                        │
│                           │  Super-Resolution│                        │
│                           │  Cascade         │                        │
│                           │  64→256→1024     │                        │
│                           └────────┬─────────┘                        │
│                                    │                                  │
│                                    ▼                                  │
│                           ┌──────────────────┐                        │
│                           │  Output Safety   │ ── NSFW detection,    │
│                           │  Filter          │    watermarking        │
│                           └────────┬─────────┘                        │
│                                    │                                  │
│                                    ▼                                  │
│                           GENERATED IMAGE 🖼️                          │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Component Details

| Component | Purpose | Key Design Choices |
|-----------|---------|-------------------|
| **Prompt Safety Filter** | Block harmful, NSFW, or adversarial prompts before generation | Text classifier + keyword blocklist; low latency |
| **Prompt Enhancement** | Improve prompt quality for better generations | LLM rewriting (e.g., append quality tags); optional user toggle |
| **Text Encoder** | Convert text into embeddings the diffusion model understands | CLIP (contrastive) or T5 (generative); frozen during diffusion training |
| **Diffusion Model** | The core generator -- iteratively denoise from noise to image | U-Net or DiT; trained on (image, caption) pairs; uses CFG |
| **Super-Resolution** | Upscale low-res output to high-res | Cascaded diffusion models (64→256→1024) or single-stage latent diffusion |
| **Output Safety Filter** | Detect NSFW content, add invisible watermarks | Image classifier for content moderation; C2PA watermarking |

---

## 📋 Data Preparation Pipeline

```
Raw Web Data (5B+ image-text pairs)
    │
    ▼
┌─────────────────────────────┐
│ 1. Deduplication            │ ── Remove exact/near-duplicate images (perceptual hashing)
├─────────────────────────────┤
│ 2. NSFW / Toxicity Filter   │ ── Remove harmful, violent, explicit content
├─────────────────────────────┤
│ 3. Resolution Filter        │ ── Remove images below minimum resolution (e.g., 256x256)
├─────────────────────────────┤
│ 4. Aesthetic Scoring        │ ── Score images by visual quality (CLIP-based aesthetic predictor)
│                             │    Keep only top 10-20% by aesthetic score
├─────────────────────────────┤
│ 5. CLIP-Based Filtering     │ ── Remove pairs where text and image don't match
│                             │    (low CLIPScore = misaligned pair)
├─────────────────────────────┤
│ 6. Caption Enhancement      │ ── Re-caption images with a VLM for more detailed descriptions
│                             │    (DALL-E 3's key innovation!)
├─────────────────────────────┤
│ 7. Aspect Ratio Bucketing   │ ── Group images by aspect ratio for efficient batching
└─────────────────────────────┘
    │
    ▼
Clean Dataset (~600M high-quality pairs)
```

---

## 🎤 Interview Cheat Sheet

### "Design a Text-to-Image System" -- The 7-Step Framework

**Step 1: Clarifying Requirements** 📋
- Generate images from text descriptions
- High resolution (1024x1024+)
- Fast generation (< 10 seconds)
- Support diverse styles (photorealistic, artistic, anime)
- Content safety is critical
- Scale: millions of requests/day

**Step 2: Frame as ML Task** 🎯
- Input: text prompt y (string)
- Output: image x (H x W x 3 tensor)
- Task: conditional image generation -- learn p(x|y)
- This is a denoising diffusion task with text conditioning

**Step 3: Data Preparation** 📚
- Source: web-scraped image-text pairs (LAION-5B scale)
- Pipeline: dedup → NSFW filter → resolution filter → aesthetic scoring → CLIP filtering → caption enhancement
- Key insight: DALL-E 3 showed that **re-captioning** with detailed descriptions dramatically improves text-image alignment
- Data quality >> data quantity after basic scale threshold (~100M+ pairs)

**Step 4: Model Development** 🧠
- **Text Encoder**: Frozen CLIP (or T5-XXL for stronger text understanding)
- **Noise Predictor**: U-Net with cross-attention (SD v1/v2) or DiT (SD3, DALL-E 3)
- **Training**: MSE loss between true noise ε and predicted noise ε_θ
- **Conditioning**: Text embeddings injected via cross-attention (Q from image, K/V from text)
- **Latent Diffusion**: Operate in VAE latent space (8x compression) for efficiency
- **CFG**: Drop text conditioning 10% of the time during training; apply guidance scale w=7.5 at inference

**Step 5: Evaluation** 📊
- **Offline metrics**: FID (image quality), CLIPScore (text-image alignment), Inception Score
- **Human evaluation**: DrawBench-style prompts testing composition, spatial reasoning, text rendering
- **A/B testing**: Side-by-side preference studies
- **Safety**: NSFW detection rate, adversarial prompt robustness

**Step 6: System Design** 🔧
- Prompt safety filter → prompt enhancement (LLM) → text encoder → diffusion model (with CFG + DDIM) → super-resolution cascade → output safety filter → watermarking
- Latent diffusion to reduce computation (generate in 64x64 latent space, decode to 512x512)
- GPU serving with batched inference

**Step 7: Deployment & Monitoring** 🚀
- GPU cluster with load balancing (A100/H100 GPUs)
- Async generation with webhook/polling for results
- Monitor: generation latency, NSFW leak rate, user satisfaction, CLIPScore distribution
- Content policy updates and prompt filter retraining
- Model versioning with gradual rollout

### Key Phrases to Drop in Your Interview 🗣️

- "We use **latent diffusion** -- operating in the VAE's compressed latent space reduces computation by 64x compared to pixel-space diffusion, making high-res generation practical."
- "**Cross-attention** is the bridge between text and image: queries come from the noisy image features, keys and values come from the text encoder, so each spatial location can attend to relevant words."
- "**Classifier-free guidance** lets us trade off diversity vs. fidelity at inference time without retraining -- it's the 'creativity dial' of the system."
- "We train with a simple **MSE noise prediction loss**, but the real magic is in the data curation -- re-captioning with detailed descriptions (DALL-E 3's approach) dramatically improves prompt following."
- "**DDIM** gives us deterministic, fast sampling -- we can go from 1000 steps to 20-50 steps with minimal quality loss, which is critical for production latency."
- "For evaluation, **FID** measures overall image quality, but **CLIPScore** is what tells us if the image actually matches the text -- and we need both plus human evaluation."

### Common Follow-Up Questions ❓

| Question | Strong Answer |
|----------|--------------|
| "Why diffusion over GANs?" | Diffusion models have better mode coverage (no mode collapse), more stable training, and naturally support conditioning. GANs are faster at inference but harder to train and less diverse. |
| "Why operate in latent space?" | Pixel-space diffusion at 1024x1024 is prohibitively expensive. The VAE compresses to ~128x128 latent space (8x spatial reduction), cutting compute by ~64x while preserving perceptual quality. |
| "How does cross-attention condition on text?" | The noisy image features provide queries (Q), the text encoder output provides keys (K) and values (V). Each spatial position in the image attends to all text tokens, learning which words are relevant for that region. |
| "U-Net or DiT?" | DiT is the newer trend -- it scales better (like Transformers in NLP), handles variable resolutions more naturally, and is used in DALL-E 3, SD3, and Sora. U-Net was dominant in SD v1/v2 and Imagen. |
| "What's the guidance scale trade-off?" | Low w (~1): diverse but may ignore prompt. High w (~7-15): sharp, text-aligned images. Too high w (>20): oversaturated artifacts. It's a diversity-fidelity trade-off controlled at inference. |
| "How do you handle text rendering in images?" | This remains hard. Approaches: character-level text encoders, ControlNet for spatial guidance, or rendering text separately and compositing. DALL-E 3 improved this via better captioning. |
| "How do you prevent harmful content?" | Multi-layer defense: input prompt filter (classifier + blocklist), output NSFW classifier, invisible watermarking (C2PA), rate limiting, and human review for edge cases. |

### Numbers Worth Knowing 🔢

| Metric | Approximate Value |
|--------|------------------|
| Stable Diffusion v1.5 params | ~860M (U-Net) |
| DALL-E 3 training data | Estimated ~650M curated pairs |
| LAION-5B dataset | 5.85B image-text pairs |
| Typical DDPM steps | 1000 |
| Typical DDIM steps | 20-50 |
| CFG guidance scale | 7.0-12.0 (typical) |
| Latent space compression | 8x spatial (VAE) |
| CLIPScore range | 0.25-0.35 (typical good generation) |
| Generation latency (SD) | 2-5 seconds on A100 |
| FID (Stable Diffusion v2) | ~10 on COCO-30K |

---

## 🔗 How This Connects

| Previous | Current | Next |
|----------|---------|------|
| Ch 08: VQ-VAE, image tokenization, autoregressive generation | **Ch 09: Diffusion models, U-Net/DiT, CFG, CLIPScore** | Ch 10: DreamBooth, LoRA, personalization |

Key evolution: Chapter 08 used **discrete tokens + autoregressive** generation (like GPT but for images). This chapter switches to **continuous noise + iterative denoising** (diffusion), which produces higher-quality images and is the dominant paradigm today. Chapter 10 will build on this by teaching you how to **personalize** these diffusion models for specific subjects using DreamBooth and LoRA.

---

## 📚 Prerequisites

- Understanding of CNNs and attention (Chapters 05-08 or `00-neural-networks/`)
- Familiarity with GANs and FID (Chapter 07)
- Python + PyTorch

```bash
pip install torch torchvision numpy matplotlib
```

---

## 📖 References

1. "Denoising Diffusion Probabilistic Models" -- Ho et al., 2020 (the foundational DDPM paper)
2. "Denoising Diffusion Implicit Models" -- Song et al., 2021 (DDIM -- fast sampling)
3. "High-Resolution Image Synthesis with Latent Diffusion Models" -- Rombach et al., 2022 (Stable Diffusion)
4. "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" -- Saharia et al., 2022 (Imagen)
5. "Hierarchical Text-Conditional Image Generation with CLIP Latents" -- Ramesh et al., 2022 (DALL-E 2)
6. "Improving Image Generation with Better Captions" -- Betker et al., 2023 (DALL-E 3)
7. "Scalable Diffusion Models with Transformers" -- Peebles & Xie, 2023 (DiT)
8. "Classifier-Free Diffusion Guidance" -- Ho & Salimans, 2022
9. "Learning Transferable Visual Models From Natural Language Supervision" -- Radford et al., 2021 (CLIP)
10. "DrawBench: Text-to-Image Model Evaluation" -- Saharia et al., 2022

---

*"Diffusion models don't create images from nothing -- they learn to find the hidden image inside the noise, guided by your words."* 🎨🔮
