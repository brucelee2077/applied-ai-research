# Chapter 11: Text-to-Video Generation 🎬

## What Are We Building?

Imagine you type "a golden retriever puppy playing in autumn leaves" and the AI makes a **whole video** of it -- not just one picture, but a smooth, realistic clip with the puppy jumping, leaves flying, and camera moving. That's text-to-video generation!

**Real products:** OpenAI's **Sora**, Meta's **Movie Gen**, Google's **Veo**, Runway's **Gen-3**

This is the **final boss** of generative AI. If text-to-image is making a single painting, text-to-video is directing an entire movie scene -- every frame must look great AND they must all flow together smoothly.

---

## The Core Problem: Why Video Is SO Much Harder Than Images 😰

### The Scale Problem

Let's do the math. A single 1280×720 image = **921,600 pixels** (about 1M).

A 4-second video at 30 fps at 720p = 120 frames × 921,600 pixels = **110,592,000 pixels** (~110M).

That's **120x more data** than a single image! And we need every frame to be:
1. **High quality** individually (no blurry faces)
2. **Temporally consistent** (a dog doesn't turn into a cat between frames)
3. **Physically plausible** (objects move realistically)
4. **Text-aligned** (matches what the user asked for)

### The Data Problem

- Image-text pairs: ~5 billion available (LAION, etc.)
- Video-text pairs: ~100 million at best, and the captions are often terrible
- High-quality, well-captioned videos: maybe a few million

**Staff-level insight:** This data scarcity is why joint image-video training and synthetic captioning (using vision-language models to re-caption videos) are critical strategies.

---

## Key Concept 1: Latent Diffusion for Video 🧊

### The "Working in Miniature" Analogy

Imagine you're a sculptor. You could carve a full-size marble statue (working in pixel space -- expensive!), or you could first make a tiny clay model, perfect it, then scale it up. **Latent Diffusion Models (LDMs)** work in the tiny clay model space.

### How It Works

```
Text Prompt --> Text Encoder (T5/CLIP) --> Text Embeddings
                                              |
                                              v
Random Noise --> [Denoise in Latent Space] --> Clean Latent --> VAE Decoder --> Video Pixels
                  (this is where the magic happens)
```

Instead of denoising 110M pixels directly, we:
1. **Compress** the video to a tiny latent representation (maybe ~215K values)
2. **Denoise** in this compressed space (much cheaper!)
3. **Decode** back to full pixels

**Staff-level insight:** This is the same LDM approach from Stable Diffusion (Chapter 9), extended to 3D. The key innovation is the video-aware VAE that compresses both spatially AND temporally.

---

## Key Concept 2: Video Compression with VAE 📦

### The Squeeze Machine Analogy

Think of the VAE as a **squeeze machine**. You push a huge video in one side, it squeezes it down to a tiny representation, and another machine can unsqueeze it back to the full video. The trick is squeezing it small enough to be efficient, but not so small that you lose important details.

### Compression Math

| Dimension | Original | Compressed | Ratio |
|---|---|---|---|
| Width | 1280 | 160 | 8× |
| Height | 720 | 90 | 8× |
| Frames | 120 | 15 | 8× |
| **Total** | **110.6M** | **216K** | **~512×** |

The spatial compression (8× per dimension) uses the same approach as image VAEs. The **temporal compression** (8× across frames) is the new piece -- neighboring frames share a LOT of information (think how similar frame 1 and frame 2 of a video are), so we can compress heavily.

### Training the VAE

The VAE is trained with:
- **Reconstruction loss**: decoded video should match original (L1 or L2 per pixel)
- **Perceptual loss** (LPIPS): high-level features should match (using pretrained network)
- **Adversarial loss**: a discriminator judges if decoded video looks real
- **KL regularization**: keeps the latent space well-structured

**Staff-level insight:** Movie Gen uses a **Temporal AutoEncoder (TAE)** that adds temporal convolutions and temporal attention layers to a pretrained image VAE. This lets them leverage strong image VAE weights while adding video understanding.

---

## Key Concept 3: Temporal Attention & Temporal Convolution 🔄

### The Flipbook Analogy

Imagine you're drawing a flipbook. **Spatial attention** is looking at all parts of one page to make sure the drawing looks good. **Temporal attention** is flipping through ALL the pages to make sure the animation is smooth -- does the ball keep moving in the same direction? Does the character's face stay the same?

### Temporal Attention

In standard image attention, each pixel/patch attends to all other pixels/patches in the **same frame**.

In temporal attention, each pixel/patch attends to the **same spatial position across ALL frames**:

```
Frame 1, position (x,y) <--> Frame 2, position (x,y) <--> ... <--> Frame T, position (x,y)
```

This allows the model to learn temporal relationships: "the ball was here in frame 1, so it should be there in frame 5."

### Temporal Convolution

While temporal attention captures **long-range** temporal dependencies (frame 1 to frame 100), temporal convolution captures **local** temporal patterns using 3D convolution kernels:

```
2D Conv: kernel slides over (H, W)        --> spatial features
3D Conv: kernel slides over (T, H, W)     --> spatiotemporal features
```

A 3D conv kernel of size 3×3×3 looks at 3 consecutive frames in a 3×3 spatial window simultaneously, naturally capturing motion patterns.

### Architecture Pattern: Factorized Space-Time

Most video models use a **factorized** approach rather than full 3D attention (which is O(T²H²W²) -- way too expensive):

```
For each block:
  1. Spatial attention:  each frame attends within itself     [O(H²W²) per frame]
  2. Temporal attention: each position attends across frames  [O(T²) per position]
  3. (Optional) Temporal convolution: local cross-frame patterns
```

**Staff-level insight:** This factorization reduces complexity from O((T·H·W)²) to O(T·(HW)² + HW·T²), which is dramatically cheaper. Sora uses this factorized approach inside its DiT architecture.

---

## Key Concept 4: 3D Patchification & Video DiT 🧩

### The LEGO Cubes Analogy

Remember how Vision Transformers (ViT) chop an image into 2D patches (little squares)? For video, we chop the video into **3D cubes** -- little blocks of space AND time. Each cube covers, say, 2 frames × 16 pixels × 16 pixels. These cubes become the "tokens" that the Transformer processes.

### 3D Patchify

```
Video: (T=120, H=720, W=1280)
Latent: (T'=15, H'=90, W'=160)
After 3D patchify with patch size (2, 16, 16):
  Tokens = (15/2) × (90/16) × (160/16) ≈ 7 × 5 × 10 = 350 tokens
```

Each token is a flattened 3D cube, projected to the model's hidden dimension via a linear layer.

### Video DiT (Diffusion Transformer)

Sora popularized the **DiT** (Diffusion Transformer) architecture for video:

```
┌─────────────────────────────────┐
│         Video DiT Block         │
│                                 │
│  1. Layer Norm                  │
│  2. Spatial Self-Attention      │  ← patches within same frame attend to each other
│  3. Layer Norm                  │
│  4. Temporal Self-Attention     │  ← same patch position across frames attend
│  5. Layer Norm                  │
│  6. Cross-Attention (text)      │  ← attend to text embeddings (T5/CLIP)
│  7. Layer Norm                  │
│  8. Feed-Forward Network        │
│  9. AdaLN (timestep condition)  │  ← inject diffusion timestep info
└─────────────────────────────────┘
       × N blocks (e.g., 28-48)
```

### Why DiT Over U-Net?

| | U-Net | DiT |
|---|---|---|
| **Architecture** | CNN + skip connections | Pure Transformer |
| **Scaling** | Harder to scale | Scales beautifully (like LLMs) |
| **Resolution** | Fixed resolution at training | Flexible resolution & duration |
| **Variable length** | Tricky | Natural (just change # tokens) |

**Staff-level insight:** Sora's key insight was that DiT follows the same scaling laws as LLMs -- more parameters + more compute = better results. This is why they chose DiT over U-Net. The flexible resolution/duration support was a bonus.

### 3D Positional Encoding with RoPE

Since 3D patches have three position dimensions (t, h, w), we need 3D positional encodings. **Rotary Position Embedding (RoPE)** is extended to 3D:

- Split the embedding dimension into three groups
- Apply 1D RoPE to each group using the corresponding (t, h, w) coordinate
- Concatenate the results

This gives the model awareness of each token's position in both space and time, while maintaining the relative position properties of RoPE.

---

## Key Concept 5: Joint Image-Video Training 🤝

### The Learning-to-Walk Analogy

Before a baby can run (generate video), they need to stand (generate images). Joint image-video training is like practicing both at the same time -- you get the benefit of abundant image data while also learning video dynamics.

### Strategy 1: Joint Training

Treat images as **single-frame videos**. During training, each batch contains a mix of:
- Image-text pairs (treated as 1-frame videos)
- Video-text pairs (full temporal sequences)

**Benefits:**
- Access to billions of image-text pairs (vs. millions of video-text pairs)
- Images teach quality and diversity; videos teach motion and dynamics
- The model naturally learns to generate both images and videos

### Strategy 2: Pretrain on Images, Finetune on Video

1. Train a powerful text-to-image model first (lots of data!)
2. Add temporal layers (temporal attention, temporal convolution)
3. Finetune on video data, optionally freezing spatial layers

**Benefits:**
- Leverage mature text-to-image technology
- Requires less video data since spatial understanding is already learned
- Can finetune different temporal modules for different video styles

### Synthetic Captioning

Since video captions are scarce and low-quality, teams use vision-language models (like GPT-4V or LLaVA) to **re-caption** videos with detailed descriptions of:
- Scene content and composition
- Camera movement (pan left, zoom in, static)
- Action dynamics (running, jumping, rotating)
- Temporal progression (first... then... finally...)

**Staff-level insight:** Movie Gen reports that re-captioning videos with a strong VLM significantly improved text-video alignment. The quality of your captions directly bounds the quality of your text conditioning.

---

## Key Concept 6: Computational Cost Reduction 💰

### The Five Strategies

Generating video is INSANELY expensive. Here's how teams make it tractable:

| Strategy | What It Does | Savings |
|---|---|---|
| **1. LDM** | Work in compressed latent space | ~512× fewer values |
| **2. Precompute latents** | Run VAE encoder offline, cache results | No encoder cost during training |
| **3. Spatial super-resolution** | Generate at low-res, upscale | ~4-16× cheaper generation |
| **4. Temporal super-resolution** | Generate keyframes, interpolate | ~4-8× fewer frames to generate |
| **5. Efficient architectures** | FlashAttention, MoE, mixed precision | ~2-4× speedup |

### The Cascade Pipeline

```
Text ──→ [Base Model: 256×256, 16 frames]
              │
              ▼
         [Temporal SR: 256×256, 64 frames]    ← Interpolate between keyframes
              │
              ▼
         [Spatial SR: 1024×1024, 64 frames]   ← Upscale each frame
              │
              ▼
         Final High-Res Video 🎬
```

**Staff-level insight:** This cascade design means each model specializes in one job. The base model focuses on semantic content and coherent motion. The temporal SR model just needs to interpolate smoothly. The spatial SR model just needs to add high-frequency details. Each is independently simpler (and cheaper) than doing everything at once.

---

## Key Concept 7: Evaluation Metrics 📊

### Per-Frame Quality (Is each frame a good image?)

| Metric | What It Measures | How It Works |
|---|---|---|
| **FID** | Realism of generated frames | Compare Inception-v3 feature distributions of real vs generated |
| **IS** (Inception Score) | Quality + diversity | Check if generated images are classifiable AND diverse |
| **LPIPS** | Perceptual similarity | Compare VGG features between pairs |
| **KID** | Like FID but unbiased for small samples | Polynomial kernel MMD on Inception features |

### Temporal Consistency (Do frames flow together?)

#### FVD -- Fréchet Video Distance 🌟

FVD is the **gold standard** metric for video generation quality. It's like FID, but for video:

```
FID:  Extract features with Inception (2D CNN) → compare distributions
FVD:  Extract features with I3D (3D CNN)       → compare distributions
```

The **I3D** (Inflated 3D ConvNet) model is key -- it processes entire video clips and captures spatiotemporal features. FVD compares the distribution of these features between real and generated videos:

```
FVD = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real · Σ_gen)^{1/2})
```

Lower FVD = better. FVD captures both:
- Per-frame quality (spatial features)
- Temporal coherence (motion features)

### Text-Video Alignment (Does the video match the text?)

- **CLIP similarity**: Encode text and video frames with CLIP, measure cosine similarity
- **Frame-level**: Average CLIP score across all frames
- **Movie Gen Bench**: Meta's benchmark with 1000 prompts + human evaluation

### Comprehensive Benchmarks

- **VBench**: Multi-dimensional benchmark covering 16+ aspects (subject consistency, motion smoothness, aesthetic quality, temporal flickering, etc.)
- **Movie Gen Bench**: 1000 diverse prompts with automatic + human evaluation
- **Human evaluation**: Still the gold standard -- Elo ratings from side-by-side comparisons

**Staff-level insight:** FVD has known limitations -- it correlates imperfectly with human judgment, especially for text alignment. This is why VBench decomposes evaluation into multiple dimensions, and why human eval remains essential. Always mention this nuance in interviews.

---

## Overall System Design 🏗️

### The Full Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    TEXT-TO-VIDEO SYSTEM                           │
│                                                                  │
│  "A puppy playing in leaves"                                     │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │ Text Encoder │  (T5-XXL or CLIP)                              │
│  └──────┬──────┘                                                 │
│         │ text embeddings                                        │
│         ▼                                                        │
│  ┌──────────────────────────┐                                    │
│  │  Base Video DiT Model    │  Generates 256×256, 16 keyframes   │
│  │  (Latent Diffusion)      │  in latent space                   │
│  └──────────┬───────────────┘                                    │
│             │ latents                                             │
│             ▼                                                    │
│  ┌──────────────────────────┐                                    │
│  │  VAE Decoder             │  Decompress to pixel space         │
│  └──────────┬───────────────┘                                    │
│             │ low-res keyframes                                   │
│             ▼                                                    │
│  ┌──────────────────────────┐                                    │
│  │  Temporal Super-Res      │  Interpolate: 16 → 64 frames      │
│  └──────────┬───────────────┘                                    │
│             │ more frames                                         │
│             ▼                                                    │
│  ┌──────────────────────────┐                                    │
│  │  Spatial Super-Res       │  Upscale: 256 → 1024 pixels        │
│  └──────────┬───────────────┘                                    │
│             │                                                    │
│             ▼                                                    │
│  🎬 Final Video (1024×1024, 64 frames, 16 fps = 4 seconds)      │
│                                                                  │
│  ┌──────────────────────────┐                                    │
│  │  Safety / Content Filter │  NSFW detection, watermarking      │
│  └──────────────────────────┘                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Interview Cheat Sheet 📝

### Opening Statement (30 seconds)

> "Text-to-video generation extends latent diffusion models to the temporal dimension. The core challenge is that video has ~100x more data than a single image, so we need aggressive compression via a spatiotemporal VAE, efficient architectures like DiT with factorized space-time attention, and a cascade pipeline that separates base generation from spatial and temporal super-resolution. I'd structure my design around these three pillars."

### Must-Know Talking Points

| Topic | Key Point |
|---|---|
| **Why LDM?** | 512× compression makes diffusion tractable for video |
| **Why DiT over U-Net?** | Scales like LLMs (scaling laws), handles variable resolution/duration |
| **3D Patchify** | Convert video to spatiotemporal tokens; enables Transformer processing |
| **Factorized attention** | Spatial + temporal attention separately; O(T·(HW)² + HW·T²) vs O((THW)²) |
| **Joint training** | Images as 1-frame videos; compensates for scarce video-text data |
| **Cascade pipeline** | Base → temporal SR → spatial SR; each model is simpler and cheaper |
| **FVD** | Fréchet Video Distance using I3D features; gold standard for video quality |
| **Data challenge** | Video-text pairs are 50× scarcer than image-text; solve with synthetic captions |

### Common Interview Questions

**Q: Walk me through how Sora generates a video.**
> "Sora uses a Video DiT with latent diffusion. First, the text is encoded via T5/CLIP. Then, starting from random noise in a compressed latent space (thanks to a spatiotemporal VAE), the DiT model iteratively denoises over ~50-100 steps, conditioned on the text embeddings via cross-attention and the diffusion timestep via AdaLN. The denoised latents are decoded by the VAE decoder to pixel space. The DiT operates on 3D patches (spatiotemporal cubes), uses factorized spatial-temporal attention, and can handle variable resolutions and durations because it's a pure Transformer."

**Q: How would you handle the computational cost?**
> "Five strategies: (1) LDM -- work in 512× compressed latent space, (2) precompute and cache video latents offline, (3) generate at low resolution and use a spatial super-resolution model, (4) generate keyframes and use temporal super-resolution (frame interpolation), (5) efficient implementations like FlashAttention, mixed precision, and optionally Mixture of Experts."

**Q: What metrics would you use?**
> "I'd use FVD as the primary automated metric since it captures both visual quality and temporal consistency via I3D features. I'd supplement with per-frame FID for image quality, CLIP similarity for text alignment, and VBench for multi-dimensional analysis. But the gold standard is human evaluation with Elo ratings from side-by-side comparisons, since FVD correlates imperfectly with human judgment."

**Q: Why is video-text data so scarce, and how do you deal with it?**
> "Unlike images, videos are expensive to collect, store, and caption. Alt-text for images is common on the web, but video descriptions are rare and usually just titles/tags -- not frame-level descriptions. Solutions: (1) joint image-video training to leverage billions of image-text pairs, (2) synthetic captioning using VLMs like GPT-4V to re-caption videos with detailed descriptions of content, camera motion, and temporal dynamics, (3) aggressive data augmentation (random temporal cropping, spatial augmentation)."

**Q: Compare U-Net vs DiT for video generation.**
> "U-Net was the original diffusion backbone (used in Stable Diffusion 1.x, Imagen Video). It uses CNNs with skip connections and works well at fixed resolutions. DiT (used in Sora, SD3) is a pure Transformer that replaces conv blocks with attention blocks. DiT's advantages: (1) follows LLM scaling laws -- performance improves predictably with compute, (2) naturally handles variable resolution and duration by just changing the number of tokens, (3) easier to scale with existing Transformer infrastructure. U-Net's advantages: (1) strong inductive bias for spatial features, (2) lower computational cost at small scale due to conv efficiency, (3) more mature ecosystem."

### Traps to Avoid ⚠️

- Don't say "just extend image diffusion to video" -- the temporal dimension introduces fundamentally new challenges
- Don't forget the cascade -- nobody generates 1080p 60fps video in one shot
- Don't ignore the data problem -- this is arguably harder than the architecture problem
- Don't present FVD as perfect -- always mention its limitations and the need for human eval
- Don't forget safety -- video generation has massive deepfake/misinformation risks

---

## Summary

| Component | What It Does | Key Technology |
|---|---|---|
| **Text Encoder** | Understands the prompt | T5-XXL, CLIP |
| **VAE** | Compresses video 512× | Spatiotemporal encoder-decoder |
| **Video DiT** | Denoises latents conditioned on text | 3D patchify, factorized attention, AdaLN |
| **Temporal SR** | Fills in between keyframes | Frame interpolation model |
| **Spatial SR** | Upscales resolution | Super-resolution diffusion model |
| **Safety Layer** | Content filtering | NSFW classifiers, watermarking |

---

## References

- Sora Technical Report (OpenAI, 2024)
- Movie Gen (Meta, 2024)
- Scalable Diffusion Models with Transformers / DiT (Peebles & Xie, 2023)
- Stable Video Diffusion (Stability AI, 2023)
- Imagen Video (Ho et al., 2022)
- Make-A-Video (Singer et al., 2022)
- Video Diffusion Models (Ho et al., 2022)
- VBench (Huang et al., 2023)
