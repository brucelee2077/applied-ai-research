# Text-to-Image Generation — Staff/Principal Interview Guide

---

## How to Use This Guide

This guide is structured for interviewers and candidates preparing for staff- or principal-level ML design interviews. The interview is **45 minutes** total. Each section includes an **interviewer prompt**, the **signal being tested**, and **four-level model answers** representing the candidate response quality spectrum.

**Rating Levels:**
- **No Hire** — Fundamental misunderstanding or silence
- **Lean No Hire** — Partial understanding, significant gaps, needs heavy prompting
- **Lean Hire** — Correct understanding, hits main points, minor gaps
- **Strong Hire** — Deep, nuanced, first-principles reasoning, proactively addresses trade-offs, demonstrates platform-level thinking

**Interviewer Notes:**
- Spend the first minute reading the prompt aloud and giving the candidate time to think silently.
- Do not volunteer information unless the candidate is stuck for more than 90 seconds.
- Use the follow-up probes listed under each section to differentiate Hire from Strong Hire.
- The principal-level bar requires connecting individual design decisions to broader organizational or platform impact.

**Time Budget:**

| Section | Time |
|---|---|
| Problem Statement & Clarification | 5 min |
| ML Problem Framing | 5 min |
| Data & Preprocessing | 8 min |
| Model Architecture Deep Dive | 12 min |
| Evaluation | 5 min |
| Serving Architecture | 7 min |
| Edge Cases & Failure Modes | 5 min |
| Principal-Level Platform Thinking | 3 min |

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

> "Design a text-to-image generation system — a user provides a natural language prompt and the system generates a high-quality, photorealistic image matching the description. This should be similar to Stable Diffusion or DALL-E 3. Walk me through your approach."

### Signal Being Tested

Does the candidate recognize the core components (text encoder, diffusion model, latent space) and ask the right questions about quality requirements, safety, and serving constraints?

### Six Clarification Dimensions

| Dimension | Why It Matters |
|---|---|
| **Output resolution** | 512×512 vs. 1024×1024 — affects latency and architecture |
| **Generation quality vs. latency** | Diffusion steps (20 vs. 100) directly trade quality for speed |
| **Text-image alignment quality** | How faithfully must the image follow complex multi-object prompts? |
| **Safety requirements** | Consumer product has stricter content policy than developer API |
| **Style control** | Photorealistic only, or support for art styles, illustrations? |
| **Personalization** | Generic generation vs. subject-specific (requires fine-tuning) |

### Follow-up Probes

- "What is the fundamental challenge in generating an image with 'a red cube on top of a blue sphere'?"
- "How does your latency target change if users are paying per generation vs. free tier?"
- "What safety categories require special handling beyond typical content moderation?"

---

### Model Answers — Section 1

**No Hire:**
"I would train a model on image-text pairs." No understanding of the diffusion framework, safety requirements, or latency-quality trade-off.

**Lean No Hire:**
Knows diffusion models exist but cannot ask meaningful clarifying questions about the architecture choices. Doesn't probe for safety or multi-object compositional reasoning.

**Lean Hire:**
Asks about resolution, latency SLA, and safety. Identifies that compositionality (multiple objects with attributes) is a core challenge. Notes that NSFW content, celebrity likenesses, and copyright-infringing outputs require explicit safety handling.

**Strong Hire Answer (first-person):**

Text-to-image generation has three distinct challenges that shape the entire design: quality-speed trade-off (more diffusion steps = better quality but slower), text-image alignment quality (faithfully representing complex multi-object prompts), and safety (preventing harmful, misleading, or copyright-infringing generations).

First, I want to understand the latency requirements. Each diffusion step is a full U-Net or DiT forward pass. At 20 DDIM steps, a 512×512 image on an A100 GPU takes ~2 seconds. At 50 steps, ~5 seconds. At 100 steps, ~10 seconds. If the use case is interactive (user waits for the image), < 5 seconds is the target. If it's background (user submits and checks back), minutes are acceptable.

Second, the text alignment requirements. Simple prompts ("a red apple on a table") are easy. Complex compositional prompts ("a red cube on the left and a blue sphere on the right, with a green cylinder in the background") fail more often even in state-of-the-art systems. If the use case requires precise multi-object composition, this needs explicit architectural support.

Third, safety policy scope. At minimum: block CSAM, graphic violence, and non-consensual intimate imagery. Additionally: block known celebrity faces unless clearly fictional, flag content that closely imitates copyrighted artistic styles, and implement age verification for adult content tiers.

Fourth, output resolution. 512×512 is fast but looks small on modern displays. 1024×1024 is the minimum for web/mobile. 2048×2048 for print. Higher resolutions require more compute and time.

Let me proceed assuming: consumer product, 1024×1024 output at < 5s latency, photorealistic + artistic styles, strong safety requirements.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "How do you formally frame text-to-image generation as an ML problem? Walk me through the diffusion framework and classifier-free guidance."

### Signal Being Tested

Does the candidate understand the diffusion training objective, the conditioning mechanism via text embeddings, and the CFG trade-off? Can they explain the role of latent space compression?

### Follow-up Probes

- "Write out the diffusion training loss. What is it minimizing?"
- "Explain classifier-free guidance mathematically. What does the guidance scale γ control?"
- "Why does latent diffusion (working in compressed latent space) enable high-resolution generation efficiently?"

---

### Model Answers — Section 2

**No Hire:**
"The model is trained on image-text pairs to match them." Cannot describe the diffusion objective or CFG.

**Lean No Hire:**
Knows diffusion adds and removes noise but cannot write out the training objective or explain CFG.

**Lean Hire:**
Correctly states the diffusion MSE objective. Explains CFG as an interpolation between conditional and unconditional predictions. Describes latent diffusion as compressing to VAE space.

**Strong Hire Answer (first-person):**

Text-to-image generation via latent diffusion has four components: a text encoder, a VAE (for latent compression), a denoising U-Net/DiT (the core diffusion model), and inference-time guidance.

**Diffusion Training Objective:**

The diffusion process adds noise to a clean image x_0 over T timesteps:
```
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)
```
where ᾱ_t = Π_{s=1}^{t} (1-β_s) and β_t is the noise schedule. A standard DDPM noise schedule uses β_1=0.0001, β_T=0.02, T=1000.

The model ε_θ(x_t, t, c) is trained to predict the noise ε added at timestep t, given the noisy image x_t, timestep t, and conditioning text c. The simplified training loss:
```
L = E_{t,x_0,ε,c} [||ε - ε_θ(x_t, t, c)||²]
```

This is an MSE loss on the noise prediction, which is mathematically equivalent to minimizing a variational lower bound on the data log-likelihood.

**Latent Diffusion:**
Rather than denoising in pixel space (which is expensive — a 1024×1024 image has 3M pixels), LDM first encodes the image to a compressed latent space using a pretrained VAE:
```
z = E(x),  x̂ = D(z)
z ∈ R^{H/f × W/f × C_z}  (f=8 for Stable Diffusion: 128×128×4 latents for 1024×1024 image)
```
The diffusion model operates on z, not x. This is 64× smaller for f=8 — dramatically reducing compute.

**Text Conditioning:**
The text prompt is encoded by a CLIP or T5 text encoder: `c = TextEncoder(prompt) ∈ R^{L × d}`. This encoding is injected into every layer of the U-Net via cross-attention:
```
CrossAttn(Q_img, K_text, V_text) = softmax(Q_img K_text^T / √d_k) · V_text
```

**Classifier-Free Guidance (CFG):**
At inference, we want the model to strongly follow the text prompt. CFG trains the model jointly with and without the conditioning signal (10–20% of training examples drop the condition):
```
ε̃_θ(x_t, t, c) = ε_θ(x_t, t, ∅) + γ · (ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
```

γ > 1 amplifies the difference between conditional and unconditional predictions, steering generation strongly toward the prompt. γ=1 is no guidance; γ=7.5 is typical for photorealistic generation; γ>15 starts producing oversaturated, unrealistic images. The trade-off: higher γ → better text alignment, lower diversity; lower γ → more diverse but less prompt-faithful.

---

## Section 3: Data & Preprocessing (8 min)

### Interviewer Prompt

> "What training data do you use, and how do you filter and preprocess it?"

### Signal Being Tested

Does the candidate understand image-text dataset quality filtering (CLIP score, aesthetic scoring) and the training pipeline for latent diffusion models?

### Follow-up Probes

- "How do you filter billions of image-text pairs without human labeling of each pair?"
- "What is CLIP score filtering and what does a cutoff of 0.28 mean practically?"
- "How do you handle images with visible watermarks or low aesthetic quality?"

---

### Model Answers — Section 3

**No Hire:**
"I would download images and their captions from the web." No quality filtering understanding.

**Lean No Hire:**
Knows LAION is used but cannot describe CLIP score filtering, aesthetic filtering, or watermark removal.

**Lean Hire:**
Correctly describes CLIP score filtering (removes mismatched pairs), aesthetic scoring (removes low-quality images), and watermark detection. Can describe the VAE pretraining step.

**Strong Hire Answer (first-person):**

**Large-scale image-text dataset pipeline:**

The primary dataset is LAION-5B (5.85 billion image-text pairs) for pretraining, with COYO-700M and custom curated datasets for fine-tuning. Raw web data requires aggressive filtering.

**CLIP score filtering:**
CLIP computes cosine similarity between an image embedding and a text embedding. If the alt-text doesn't describe the image, the CLIP score is low. Standard filtering: remove pairs with CLIP cosine similarity < 0.28. This removes approximately 30–40% of the dataset — the most egregiously mismatched pairs.

**Aesthetic scoring:**
Train a small classifier (2-layer MLP) on the SAC (Simulacra Aesthetic Captions) dataset where humans rated image aesthetics on a 10-point scale. Apply this classifier to filter training images; keep those with predicted aesthetic score > 4.5/10. This removes low-quality, blurry, and poorly composed images.

**Watermark and NSFW detection:**
- Watermark detector (LAION provides watermark probabilities): filter images with P(watermark) > 0.5 to prevent the model from learning to generate watermarked images
- NSFW classifier: remove explicit content unless intentional (adult platform). LAION provides NSFW probability estimates.
- Safety filtering: remove images containing CSAM (using PhotoDNA or similar hash-based detection)

**Caption quality:**
Web alt-text ranges from highly descriptive ("A close-up photograph of a tabby cat sitting on a red cushion") to completely uninformative ("image.jpg" or "Click here"). Caption quality filtering:
- Remove captions with < 5 words
- Remove captions that are templated boilerplate (contain patterns like "stock photo", "click here", "buy now")
- Optionally: re-caption all images using a strong VLM (CogVLM, LLaVA) to generate high-quality descriptive captions regardless of original alt-text

**VAE pretraining:**
The VAE (encoder + decoder) is pretrained separately on high-resolution images to learn perceptually meaningful latent spaces. The VAE training objective combines L1 pixel loss + perceptual loss + small adversarial loss for sharp reconstructions. VAE is frozen during diffusion model training.

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "Walk me through the U-Net and DiT architectures for diffusion. How does each work, and when would you choose one over the other?"

### Signal Being Tested

Does the candidate understand the U-Net's skip connections, the DDIM sampling algorithm, and the DiT (Diffusion Transformer) architecture? Can they explain the positional embedding for 2D images in DiT?

### Follow-up Probes

- "How does DDIM achieve faster sampling than DDPM?"
- "What is the DiT architecture and how does it differ from U-Net for diffusion?"
- "What is the role of cross-attention in the U-Net for text conditioning?"

---

### Model Answers — Section 4

**No Hire:**
"I would use a CNN." Cannot describe U-Net architecture or DDIM.

**Lean No Hire:**
Knows U-Net has encoder-decoder with skip connections but cannot explain downsampling blocks, ResNet blocks, or DDIM.

**Lean Hire:**
Correctly describes U-Net architecture with skip connections, cross-attention for text conditioning, and DDIM as faster deterministic sampling. Can describe DiT at a high level.

**Strong Hire Answer (first-person):**

Text-to-image diffusion uses two dominant architectures: U-Net (Stable Diffusion 1/2/XL) and DiT/Diffusion Transformer (Stable Diffusion 3, FLUX, DALL-E 3 internal).

**U-Net Architecture:**

The U-Net processes images at multiple scales through an encoder-decoder with skip connections:

Encoder: progressively downsample (×2 at each level) through ResNet blocks + Self-Attention:
```
Level 0: 128×128 → 64 channels
Level 1: 64×64 → 128 channels
Level 2: 32×32 → 256 channels
Level 3: 16×16 → 512 channels
```

Bottleneck: 8×8 → 512 channels (full spatial attention at bottleneck)

Decoder: progressively upsample, concatenate skip connections from encoder:
```
Level 3: 16×16 → 256 channels (+ skip from encoder)
Level 2: 32×32 → 128 channels
Level 1: 64×64 → 64 channels
Level 0: 128×128 → output
```

Text conditioning via cross-attention at each U-Net level:
```
CrossAttn(Q_image_feature, K_text, V_text)
```
where Q comes from the image feature maps and K, V come from the CLIP/T5 text embeddings.

Skip connections preserve fine-grained spatial detail from the encoder — the decoder uses low-level texture information from early encoder layers when reconstructing the final image, preventing information loss.

**DDIM Sampling:**

Standard DDPM sampling requires T=1000 steps (each is a forward pass through the U-Net). DDIM (Denoising Diffusion Implicit Models) derives a deterministic non-Markovian reverse process that achieves similar quality in 20–50 steps:

DDIM reverse step:
```
x_{t-1} = √ᾱ_{t-1} · [(x_t - √(1-ᾱ_t)·ε_θ(x_t,t,c))/√ᾱ_t] + √(1-ᾱ_{t-1})·ε_θ(x_t,t,c)
```

This is deterministic given x_T (no randomness per step) — the same x_T always produces the same image. 50 DDIM steps achieve quality comparable to 1000 DDPM steps because DDIM takes larger, more principled steps along the diffusion trajectory.

**DiT (Diffusion Transformer):**

DiT replaces the U-Net with a pure transformer applied to image patches. The image latent (e.g., 32×32×4) is divided into 2×2 patches, flattened to a sequence of 256 tokens, and processed by standard transformer blocks.

Time and text conditioning in DiT: use adaptive layer norm (adaLN-Zero) rather than cross-attention:
```
adaLN: γ, β = Linear(c), y = γ · LayerNorm(x) + β
```
where c is the conditioning vector (time + text class). This conditions every layer without adding cross-attention overhead.

DiT advantages over U-Net: (1) scales predictably with parameter count (transformers scale better than CNNs), (2) no hand-designed skip connection structure — the model learns global interactions at all scales from the start, (3) flexible context length enables higher-resolution generation without architectural modifications.

DiT disadvantage: slower for small models (U-Net is faster at <1B parameters); DiT shines at large scale (>2B parameters).

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How do you evaluate text-to-image generation quality? What metrics measure text alignment vs. image quality?"

### Signal Being Tested

Does the candidate understand FID for image quality, CLIP Score for text-image alignment, and human evaluation protocols? Can they explain why FID is an imperfect proxy?

### Follow-up Probes

- "What is CLIP Score and how does it measure text-image alignment?"
- "Why is FID insufficient for evaluating text-to-image models?"
- "How do you evaluate compositional generation — does the model correctly render 'a red cube to the left of a blue sphere'?"

---

### Model Answers — Section 5

**No Hire:**
"I would look at the images and see if they match the prompt." Cannot describe any formal metric.

**Lean No Hire:**
Mentions FID and CLIP score by name but cannot explain how either is computed or their limitations.

**Lean Hire:**
Correctly explains FID as distribution distance in Inception space and CLIP Score as cosine similarity between image and text embeddings. Notes that FID doesn't measure text alignment. Describes human evaluation for final quality assessment.

**Strong Hire Answer (first-person):**

Text-to-image evaluation requires measuring two orthogonal dimensions: image quality (fidelity and aesthetics) and text alignment (does the image match the prompt?).

**Image Quality — FID:**
```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^{1/2})
```
FID measures distributional distance between real images and generated images in Inception feature space. Lower FID = generated images are more similar to the real image distribution. Typical FID for state-of-the-art text-to-image: 5–15 on COCO evaluation set.

FID limitations for text-to-image: FID measures unconditional quality (does the generated image look realistic?) but not whether it matches the text prompt. A model could achieve excellent FID by generating random realistic images regardless of the prompt.

**Text-Image Alignment — CLIP Score:**
```
CLIP-Score(I, t) = max(cos(f_I(I), f_T(t)), 0)
```
CLIP Score measures cosine similarity between the CLIP image embedding and CLIP text embedding. Higher = better alignment. Typical values: 0.28–0.35 for good text-to-image models.

CLIP Score limitations: CLIP was trained with contrastive objectives — it measures global semantic similarity but misses fine-grained spatial relationships. "A red cube to the LEFT of a blue sphere" may score identically to "a blue sphere to the LEFT of a red cube" if both contain the right objects regardless of positions.

**Compositional evaluation — T2I-CompBench:**
Specifically evaluates: attribute binding (correct attribute-object assignment), spatial relations, non-spatial relations. Metrics include VQA-based evaluation (use a VQA model to ask questions about specific compositional aspects of the generated image).

**Human evaluation protocol:**
1. Photorealism rating (1–5 scale): "Does this look like a real photograph?"
2. Prompt alignment rating (1–5 scale): "How well does the image match the prompt?"
3. Preference comparison: which of models A and B do you prefer?

For production, I run daily automated FID + CLIP Score measurements, with weekly human evaluation panels (100 diverse prompts, 5 raters each).

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "Walk me through the serving architecture for a production text-to-image system serving millions of users."

### Signal Being Tested

Does the candidate understand diffusion serving optimizations (flash attention, tiled VAE decoding, batch inference) and the GPU memory requirements for serving a large diffusion model?

### Follow-up Probes

- "How much GPU memory does Stable Diffusion XL require? How do you reduce this?"
- "What is torch.compile / XFormers and how do they speed up diffusion inference?"
- "How do you serve millions of users with acceptable latency and cost?"

---

### Model Answers — Section 6

**No Hire:**
"Run the model on GPUs." No understanding of memory requirements or diffusion-specific optimizations.

**Lean No Hire:**
Notes that diffusion requires multiple forward passes but cannot quantify memory requirements or describe specific optimizations.

**Lean Hire:**
Correctly estimates SDXL memory requirements, describes flash attention and INT8 quantization as key optimizations. Can describe the request queue and GPU cluster design.

**Strong Hire Answer (first-person):**

Serving a text-to-image model at consumer scale requires careful attention to both memory and throughput.

**Memory budget:**
Stable Diffusion XL (SDXL) components:
- UNet: ~2.6B parameters × 2 bytes (FP16) = ~5.2 GB
- VAE: ~0.1B parameters × 2 bytes = ~0.2 GB
- CLIP text encoders (x2): ~0.9B parameters × 2 bytes = ~1.8 GB
- **Total: ~7.2 GB weights**

Plus activation memory during inference: ~3–5 GB for intermediate activations (depends on resolution and batch size). Total: ~10–12 GB for a single inference on a 24GB consumer GPU. For production serving with batch size > 1, an A100 (80GB) allows batch size ~6.

**Key serving optimizations:**

1. *Flash Attention* (Dao et al.): recomputes attention instead of materializing the full attention matrix. For 1024×1024 with 128-token sequence in self-attention, standard attention requires O(n²) memory; Flash Attention reduces this to O(n) with equal computational complexity. 30–50% latency reduction on attention-heavy models.

2. *INT8/INT4 quantization*: quantize U-Net weights to INT8. At INT8: ~2.6 GB for U-Net (vs. 5.2 GB at FP16). Latency: 30–50% speedup on GPU INT8 SIMD. Quality degradation: < 5% CLIP Score reduction.

3. *XFormers* or *torch.compile*: custom memory-efficient attention implementation; 20–40% speedup.

4. *Token merging (ToMe)*: merge redundant tokens in attention layers (similar tokens at nearby positions). Reduces attention compute by 30–50% with minimal quality loss.

**Serving infrastructure:**

Queue-based architecture: requests arrive at a load balancer, are queued by priority tier (paid users: higher priority), and distributed to GPU workers. Each worker runs continuous batching: process up to batch_size=4 images simultaneously (same prompt length ensures similar completion time).

Latency target: < 5s for 1024×1024 at 20 DDIM steps with optimizations (flash attention + INT8):
- Text encoding: ~50ms
- 20× UNet forward passes: ~3000ms (dominant cost)
- VAE decode: ~200ms
- Total: ~3250ms

**Safety pipeline integration:**
Input safety classifier runs before the model (block harmful prompts, 50ms). Output safety classifier runs after image generation (detect NSFW/harmful outputs, 100ms). Safety pipeline adds ~150ms to total latency.

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the critical failure modes of text-to-image generation, both technical and safety-related?"

### Signal Being Tested

Does the candidate identify compositional failures, attribute binding errors, safety risks (NSFW, celebrity likenesses, copyright), and prompt injection?

### Follow-up Probes

- "What is attribute binding and why do text-to-image models struggle with it?"
- "How do you prevent the system from generating realistic images of real people?"
- "What is 'concept drift' in a production text-to-image system?"

---

### Model Answers — Section 7

**No Hire:**
Cannot describe text-to-image specific failure modes beyond "bad images."

**Lean No Hire:**
Mentions NSFW as a safety concern but cannot describe attribute binding, compositional failures, or copyright risks.

**Lean Hire:**
Identifies attribute binding, celebrity likeness generation, NSFW content, and prompt injection as the key failure modes. Proposes safety classifier and face detection filters.

**Strong Hire Answer (first-person):**

Text-to-image systems have both technical and safety failure modes that require different mitigation strategies.

**Technical: Attribute binding failure**
The model fails to correctly bind attributes to objects: "a red cube and a blue sphere" may produce a blue cube and a red sphere, or a reddish-blue sphere. This is a fundamental limitation of the current CLIP-conditioned diffusion paradigm — CLIP text encodings don't explicitly encode attribute-object bindings as structured relations.

Detection: VQA-based automatic evaluation asking "What color is the cube?" Detection rate: ~70% on simple 2-object compositions with current state-of-the-art.

Mitigation: structured prompting (give objects unique names, e.g., "Object A is a red cube..."), training on structured compositional data, layout conditioning (specify object bounding boxes as additional conditioning signal).

**Technical: Counting failure**
"Generate an image with exactly 5 dogs" reliably fails — the model typically generates 3 or 6 or 8 dogs. Counting is a known weakness of spatial conditioning via CLIP.

**Safety: Celebrity likeness generation**
The model may generate photorealistic images of real public figures in inappropriate or misleading contexts. This creates defamation and deepfake risks.

Mitigation: (1) face recognition on generated images — compare detected faces to a database of known public figures; (2) train the model on anonymized data (blur real faces in training images); (3) refuse prompts that explicitly name real people.

**Safety: NSFW and harmful content**
Input filter: train a text classifier on harmful prompt categories (CSAM, graphic violence, non-consensual imagery) to block at request time. Output filter: run NSFW image classifier on all generated images; reject and return error if flagged.

**Safety: Copyright infringement**
Users may prompt for "an image in the style of [living artist]" or "image by [artist name]". This directly invites copyright infringement.

Mitigation: (1) detect artist name mentions in prompts using a curated entity list; (2) style similarity measurement — compare generated images to copyrighted works using perceptual hash or CLIP similarity; (3) legal policy: do not train on opt-out artists' works (nightshade initiative, Spawning.ai).

**Safety: Prompt injection via image-to-image**
When image-to-image mode is available, a user uploads an image containing embedded text instructions (in tiny font or steganographically). The model processes the image as conditioning and may follow embedded instructions.

Mitigation: run OCR on all input images; flag and review any images where OCR detects instruction-like text patterns.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "You've built a text-to-image system. Now you're building a creative AI platform that enables product teams at your company to embed image generation in their products (e-commerce, advertising, social media, gaming). How do your platform responsibilities change?"

### Signal Being Tested

Does the candidate think about API design, safety enforcement at the platform level, and the platform's responsibility for downstream misuse?

### Follow-up Probes

- "How do you handle a product team that wants to allow adult content in their product?"
- "What are the platform's legal obligations when a third-party product misuses the generation API?"

---

### Model Answers — Section 8

**No Hire:**
"Expose a REST API." No consideration of per-product safety policies or platform responsibility.

**Lean No Hire:**
Mentions rate limiting and API keys but doesn't address per-product safety policy configuration or the platform's responsibility.

**Lean Hire:**
Describes safety tiers (default-safe, adult-enabled, with identity verification required), configurable content policies per product, and audit logging.

**Strong Hire Answer (first-person):**

A creative AI platform serving multiple products requires a tiered safety model and clear legal responsibility partitioning.

**Safety tier model:**
- *Tier 1 (default)*: all products have baseline safety enabled. No NSFW, no celebrity likenesses, no graphic violence. This is non-negotiable — any product using the platform gets these protections unless they qualify for a higher tier.
- *Tier 2 (age-verified adult)*: requires identity verification, age verification, and explicit regulatory compliance documentation. Only enabled in jurisdictions where legal. Subject to quarterly audits.
- *Tier 3 (enterprise)*: custom safety policies for specialized use cases (medical illustrations with anatomical content, security training with violence simulation). Requires contractual liability agreement.

**Platform vs. product responsibility:**
The platform is responsible for: safety infrastructure (input/output classifiers, watermarking), preventing known harmful content (CSAM is absolute — no tier removes this protection), and providing audit logs for legal discovery. The product is responsible for: ensuring appropriate use within their platform (age gating for adult tier), complying with local regulations, and not misrepresenting AI-generated content to end users.

**Watermarking as platform requirement:**
Every generated image must be watermarked — invisible steganographic watermark containing API key hash, timestamp, and model version. This is enforced at the platform level and cannot be disabled by any tier. C2PA metadata standard compliance is required for regulatory readiness.

**API design:**
```
POST /v1/images/generate
{
  "prompt": "...",
  "safety_tier": 1,   // platform-enforced
  "style": "photorealistic",
  "resolution": "1024x1024"
}
```
Safety tier is inferred from the product's API key tier, not provided by the caller.

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**Diffusion forward process:**
```
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)
ᾱ_t = Π_{s=1}^{t} (1-β_s)
```

**Diffusion training objective (simplified):**
```
L = E_{t,x_0,ε,c} [||ε - ε_θ(x_t, t, c)||²]
```

**DDIM reverse step:**
```
x_{t-1} = √ᾱ_{t-1}·[(x_t - √(1-ᾱ_t)·ε_θ)/√ᾱ_t] + √(1-ᾱ_{t-1})·ε_θ(x_t, t, c)
```

**Classifier-Free Guidance:**
```
ε̃ = ε_θ(x_t, t, ∅) + γ·(ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
```

**CLIP Score:**
```
CLIP-Score(I, t) = max(cos(f_I(I), f_T(t)), 0)
```

**FID:**
```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^{1/2})
```

**VAE reconstruction + KL loss:**
```
L_VAE = E[||x - D(z)||²] + β·KL(q(z|x)||N(0,I))
```

**Cross-attention for text conditioning:**
```
CrossAttn(Q_img, K_text, V_text) = softmax(Q_img K_text^T / √d_k) · V_text
```

**CFG scale trade-off:**
```
γ=1: no guidance (full diversity, low alignment)
γ=7.5: standard photorealistic (high alignment, moderate diversity)
γ>15: oversaturated, unrealistic artifacts
```

### Vocabulary Cheat Sheet

| Term | Definition |
|---|---|
| **Latent diffusion** | Diffusion in VAE-compressed latent space (not pixel space) |
| **U-Net** | Encoder-decoder with skip connections; backbone for diffusion denoising |
| **DiT** | Diffusion Transformer; replaces U-Net with pure transformer |
| **DDPM** | Denoising Diffusion Probabilistic Models; original 1000-step formulation |
| **DDIM** | Denoising Diffusion Implicit Models; faster deterministic 20–50 step sampling |
| **CFG** | Classifier-Free Guidance; amplifies conditional vs. unconditional prediction |
| **CLIP** | Contrastive Language-Image Pretraining; text+image embedding alignment |
| **Noise schedule** | β_t values controlling how fast noise is added (linear, cosine, etc.) |
| **Flash Attention** | Memory-efficient attention that recomputes instead of materializing O(n²) matrix |
| **adaLN-Zero** | Adaptive layer norm; conditions DiT layers on time + text |
| **ᾱ_t** | Cumulative noise schedule: Π_{s=1}^t (1-β_s) |
| **Token merging (ToMe)** | Merge similar tokens in attention; reduces compute ~30% |
| **C2PA** | Content Provenance and Authenticity standard for AI content watermarking |
| **Attribute binding** | Correctly associating adjectives with the right objects in multi-object prompts |
| **T2I-CompBench** | Benchmark for compositional text-to-image evaluation |

### Key Numbers Table

| Metric | Value |
|---|---|
| SDXL U-Net parameters | ~2.6B |
| SDXL total memory (FP16) | ~7.2 GB weights |
| SDXL inference memory (1× batch) | ~10–12 GB |
| Stable Diffusion XL FID (COCO) | ~5–10 |
| CLIP Score (strong model) | 0.28–0.35 |
| DDIM inference steps (production) | 20–50 |
| DDPM training steps | T=1000 |
| Typical CFG scale (photorealistic) | 7.5 |
| Flash Attention latency reduction | 30–50% |
| INT8 quantization memory reduction | 50% (FP16→INT8) |
| LAION-5B dataset size | 5.85B image-text pairs |
| CLIP score filter threshold | 0.28 |
| Aesthetic score filter threshold | 4.5/10 |
| VAE downsampling factor (SD) | f=8 (1024px → 128px latent) |
| DiT-XL/2 parameters | 675M |

### Rapid-Fire Day-Before Review

1. **Diffusion training loss?** MSE on predicted noise: `E[||ε - ε_θ(x_t, t, c)||²]`
2. **CFG formula?** `ε̃ = ε_uncond + γ·(ε_cond - ε_uncond)` — amplifies conditional signal
3. **Why latent diffusion?** Work in 8× compressed VAE space → 64× compute reduction vs. pixel space
4. **DDIM vs. DDPM?** DDIM: deterministic, 20–50 steps; DDPM: stochastic, 1000 steps
5. **CLIP Score measures?** Cosine similarity between CLIP image and text embeddings — text-image alignment
6. **FID limitation for text-to-image?** Measures distributional quality, not prompt alignment; can be gamed by ignoring prompts
7. **Attribute binding failure?** Model assigns wrong attribute to wrong object (red sphere + blue cube instead of red cube + blue sphere)
8. **Flash Attention benefit?** Memory-efficient attention: O(n) memory instead of O(n²); 30-50% latency reduction
9. **Safety: celebrity likenesses?** Face recognition on output + curated entity list in prompt filter
10. **CFG scale trade-off?** Higher γ → better prompt alignment, lower diversity, can oversaturate; γ=7.5 is typical sweet spot
