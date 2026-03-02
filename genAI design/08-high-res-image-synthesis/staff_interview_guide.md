# High-Resolution Image Synthesis — Staff/Principal Interview Guide

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

> "Design a high-resolution image synthesis system — given a low-resolution or compressed image, synthesize a high-quality, high-resolution version. This should work for natural photographs at up to 4× or 8× super-resolution. Walk me through your approach."

### Signal Being Tested

Does the candidate recognize the distinction between image super-resolution and unconditional generation, and ask the right questions about the quality objective, compute constraints, and failure tolerance?

### Six Clarification Dimensions

| Dimension | Why It Matters |
|---|---|
| **Scale factor** | 2× vs. 8× SR require different architectures and have different quality trade-offs |
| **Quality objective** | Perceptual quality vs. pixel accuracy (PSNR) — determines loss function choices |
| **Image domain** | Natural photos vs. medical scans vs. satellite — different perceptual priors |
| **Latency** | Real-time (interactive editing) vs. offline batch processing |
| **Hallucination tolerance** | Medical imaging requires conservative SR; consumer photos can tolerate more synthesis |
| **Compression artifacts** | JPEG restoration vs. true upscaling require different preprocessing |

### Follow-up Probes

- "What is the fundamental ambiguity in super-resolution, and why does it matter for your loss function choice?"
- "For medical imaging super-resolution, why might PSNR-optimized SR be safer than perceptual SR?"
- "How does your approach change for 8× vs. 2× super-resolution?"

---

### Model Answers — Section 1

**No Hire:**
"I would train a CNN to upsample images." No recognition of the fundamental ambiguity or quality metric trade-off.

**Lean No Hire:**
Identifies that SR has a quality-perception trade-off but cannot articulate the mathematical formulation of the ill-posed problem or why PSNR and perceptual quality conflict.

**Lean Hire:**
Correctly identifies the ill-posed nature (many HR images correspond to one LR), explains the PSNR vs. perceptual quality trade-off, and notes that the appropriate loss function depends on the use case.

**Strong Hire Answer (first-person):**

Super-resolution is fundamentally an ill-posed inverse problem: many high-resolution images are consistent with a given low-resolution observation. When you downsample a 4K image to 1080p, information is irrecoverably lost. The SR model must hallucinate the missing detail — and this hallucination can be judged on two different axes.

First, I clarify the quality objective. Pixel-level accuracy (PSNR/SSIM) measures how closely the SR output matches the ground-truth high-resolution image pixel by pixel. Perceptual quality (measured by LPIPS or human raters) measures how realistic and sharp the output looks, regardless of whether it matches the ground truth exactly. These objectives conflict: pixel-level optimization (L1/L2 loss) produces smooth, blurry outputs that look artificially clean but achieve high PSNR. Perceptual optimization (adversarial loss, perceptual loss) produces sharp, realistic textures that look better to humans but may deviate from the ground-truth pixel values.

For consumer photo enhancement, I choose perceptual quality — users want photos that look impressive, not technically accurate. For medical imaging super-resolution, I choose pixel accuracy — a false texture hallucinated in a CT scan could mask a real pathology.

Second, I clarify the scale factor. 2× SR is a constrained problem where the LR image contains most of the structural information; the model needs to sharpen edges and fill in texture. 8× SR is much more generative — the model must invent plausible fine detail that may not exist in the LR image at all. The architectural complexity and training challenge scale with the factor.

Third, I ask about domain. Natural photos have rich prior knowledge about textures, objects, and scenes that a GAN can exploit. Satellite imagery has different scale relationships and different frequency content. Medical scans have strict interpretability requirements.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "How do you formally frame high-resolution image synthesis? What are the input, output, and training objectives?"

### Signal Being Tested

Does the candidate understand the conditional generation formulation and the key tension between reconstruction loss (pixel accuracy) and perceptual/adversarial loss (visual realism)?

### Follow-up Probes

- "What is perceptual loss, and why does it produce sharper results than L1/L2 pixel loss?"
- "How does a VQ-VAE differ from a standard VAE for image synthesis?"
- "What is the role of the autoregressive prior in VQ-VAE + transformer models?"

---

### Model Answers — Section 2

**No Hire:**
"I would use a CNN to predict the high-resolution image." Cannot formalize the loss function choices.

**Lean No Hire:**
Identifies the supervised learning framing but cannot distinguish perceptual loss from pixel loss or explain why pixel loss produces blurry results.

**Lean Hire:**
Correctly explains perceptual loss (VGG feature space), adversarial loss, and the combination used in SRGAN/ESRGAN. Can describe VQ-VAE at a high level.

**Strong Hire Answer (first-person):**

High-resolution image synthesis (SR) is conditional image generation: given a low-resolution image x_LR, generate a high-resolution image x_HR. The model estimates:
```
x̂_HR = G(x_LR)
```
where G is the synthesis network.

**Training objectives:**

*L2 pixel loss* (MSE):
```
L_pixel = ||x_HR - G(x_LR)||²_F
```
This optimizes PSNR directly. The problem: optimizing L2 over multiple plausible HR images produces the mean — a blurry compromise. If there are two equally plausible edge textures at a given location, L2 outputs their average: a smooth gradient.

*Perceptual loss* (Johnson et al.):
```
L_perceptual = Σ_i ||φ_i(x_HR) - φ_i(G(x_LR))||²_F
```
where φ_i is the feature map at layer i of a pretrained VGG-16/19. Perceptual loss penalizes differences in high-level feature representations rather than pixel values. Feature maps encode texture, edges, and semantic content — matching features produces sharper textures without forcing pixel-level accuracy.

*Adversarial loss* (SRGAN, ESRGAN):
```
L_adv = -log D(G(x_LR))
```
The discriminator D is trained to distinguish real HR images from SR outputs. Adversarial loss pushes the generator to produce outputs indistinguishable from real — the sharpest possible textures.

Combined loss in ESRGAN:
```
L = L_pixel + λ_p L_perceptual + λ_a L_adv
```
Typical weights: λ_p ≈ 1.0, λ_a ≈ 0.005. The adversarial term is small to prevent GAN artifacts dominating.

**VQ-VAE + Autoregressive Transformer:**
An alternative paradigm treats image synthesis as a two-stage generative process:
1. VQ-VAE encodes images into a discrete codebook space (tokens)
2. An autoregressive transformer generates new images by predicting codebook token sequences

This is the approach behind VQGAN, VQ-VAE-2, and early DALL-E. The VQ-VAE provides a compressed discrete representation that the transformer can model sequentially:
```
VQ-VAE: x → quantized_latent_code → x̂ (reconstruction)
Transformer: p(z_1,...,z_N) = Π_i p(z_i | z_1,...,z_{i-1})
```

This architecture is more powerful than pure GAN for diversity and can generate arbitrary high-resolution images by predicting tokens autoregressively.

---

## Section 3: Data & Preprocessing (8 min)

### Interviewer Prompt

> "What training data do you use and how do you create LR-HR training pairs?"

### Signal Being Tested

Does the candidate understand blind SR (unknown degradation) vs. classical SR (known degradation), and the importance of realistic degradation modeling?

### Follow-up Probes

- "What is the degradation model and why does it matter?"
- "What is blind super-resolution and why is it harder than classical SR?"
- "How do you create training pairs for 8× SR without actual 8× downscaled images?"

---

### Model Answers — Section 3

**No Hire:**
"I would collect high-resolution and low-resolution image pairs." Cannot describe degradation modeling.

**Lean No Hire:**
Knows that HR images are downscaled to create LR training images but cannot describe realistic degradation models or why simple bicubic downscaling is insufficient.

**Lean Hire:**
Explains that simple bicubic downscaling doesn't match real-world degradation (JPEG compression, noise, blur). Describes BSRGAN/Real-ESRGAN's degradation pipeline with multiple degradation types.

**Strong Hire Answer (first-person):**

The core challenge in SR training is the degradation model gap: in training, we create LR images from HR images with a known synthetic degradation (e.g., bicubic downscaling). At inference, real LR images have complex, unknown degradations (JPEG artifacts, sensor noise, motion blur, resizing by unknown algorithm). A model trained only on bicubic-downscaled images will fail on real-world inputs.

**Classical SR training pipeline:**
1. Collect large HR image dataset (DIV2K: 2K diverse high-quality 800×600+ images; Flickr2K: 2650 images at 2K resolution; FFHQ-style high-res collections)
2. Create LR pairs: apply known degradation to HR images

Simple degradation: `LR = Bicubic_Downsample(HR, scale=4)`

**Realistic degradation pipeline (Real-ESRGAN, BSRGAN):**
For real-world SR, model a complex degradation process that approximates the full range of real degradations:
```
LR = downsample(JPEG(add_noise(blur(HR))))
```
Specifically, apply random combinations of:
1. *Blur*: Gaussian blur (σ ∈ [0.1, 2.0]), isotropic/anisotropic, motion blur
2. *Noise*: Gaussian noise, Poisson noise, sensor noise (AWGN + Poisson)
3. *Downsampling*: bicubic, bilinear, nearest-neighbor (random choice with random scale)
4. *JPEG compression*: quality factor ∈ [30, 95] (random)
5. *Repeat*: apply the degradation pipeline twice with different parameters to model real-world compound degradations

Training with diverse realistic degradations produces models that generalize to real-world inputs far better than models trained on only bicubic downscaling.

**VQ-VAE training data:**
For the VQ-VAE + transformer approach, training data is HR images only (no paired LR required). The VQ-VAE learns to encode and reconstruct HR images from discrete tokens. The autoregressive transformer then learns the token distribution. For synthesis (not SR), this is fully unconditional — the model generates novel images.

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "Walk me through the ESRGAN architecture and the VQ-VAE + autoregressive transformer approach. What are the key components and when do you use each?"

### Signal Being Tested

Does the candidate understand the ESRGAN generator (RRDB blocks), the VQ-VAE quantization loss, and the autoregressive prior? Can they explain when each architecture is appropriate?

### Follow-up Probes

- "What is an RRDB block and why is it used in ESRGAN's generator?"
- "Explain VQ-VAE quantization. What is the straight-through estimator?"
- "Why does the VQ-VAE + transformer produce more diverse images than ESRGAN?"

---

### Model Answers — Section 4

**No Hire:**
"I would use a CNN with skip connections." Cannot describe ESRGAN or VQ-VAE.

**Lean No Hire:**
Knows ESRGAN and VQ-VAE by name but cannot explain RRDB blocks, quantization loss, or why VQ-VAE uses discrete tokens.

**Lean Hire:**
Correctly describes RRDB architecture, VQ-VAE codebook quantization, and the straight-through estimator for gradient flow. Can compare ESRGAN and VQ-VAE+GPT use cases.

**Strong Hire Answer (first-person):**

Two architectures dominate high-resolution image synthesis for different use cases.

**ESRGAN (Enhanced Super-Resolution GAN):**

ESRGAN's generator is built from Residual-in-Residual Dense Blocks (RRDB):
```
RRDB(x) = x + β · Dense_Block(Dense_Block(Dense_Block(x)))
```
Each RRDB contains three densely connected sublayers where each layer's output is concatenated with all previous outputs:
```
h_k = H_k([x, h_1, ..., h_{k-1}])  (dense connections)
```
Dense connections provide excellent gradient flow for training deep networks and reuse features at multiple scales. β ≈ 0.2 is a residual scaling factor that stabilizes training deep residual networks.

The full generator: Input LR → Conv → 23 RRDB blocks → Conv → PixelShuffle upsampling → Conv → SR output.

PixelShuffle (sub-pixel convolution) upsampling: instead of upsampling then convolution, convolve at LR resolution to produce r²×C output channels, then rearrange to create a C-channel image at r× resolution. This is more computationally efficient and avoids checkerboard artifacts from deconvolution.

ESRGAN's discriminator uses a Relativistic GAN objective — instead of predicting "real or fake," it predicts whether "real image is more realistic than fake" and vice versa:
```
L_D = -E_{(x_r, x_f)}[log(σ(C(x_r) - E_{x_f}[C(x_f)])) + log(1 - σ(C(x_f) - E_{x_r}[C(x_r)]))]
```

**VQ-VAE (Vector Quantized Variational Autoencoder):**

VQ-VAE encodes images into a discrete codebook. The encoder maps image x to continuous latent z_e, then each z_e is quantized to the nearest codebook vector:
```
z_q = e_k  where k = argmin_j ||z_e - e_j||²
```

The VQ-VAE total loss has three components:
```
L_VQ-VAE = L_reconstruction + ||sg[z_e] - e||² + β||z_e - sg[e]||²
```
- L_reconstruction: pixel + perceptual reconstruction quality
- Codebook loss: `||sg[z_e] - e||²` — moves codebook vectors toward encoder outputs (sg = stop gradient)
- Commitment loss: `β||z_e - sg[e]||²` — prevents encoder from changing faster than codebook

The straight-through estimator for gradient flow: during the forward pass, z_q = e_k (discrete, non-differentiable). During backpropagation, gradients are passed straight through z_q to z_e as if z_q = z_e. This bypasses the non-differentiable argmin operation.

**Autoregressive prior with Transformer:**
After training VQ-VAE, the compressed token sequences follow a distribution. Train an autoregressive transformer (PixelCNN or GPT-style) to model:
```
p(z_1,...,z_N) = Π_{i=1}^{N} p(z_i | z_1,...,z_{i-1})
```

Sampling: sample z_1..z_N autoregressively, then decode via VQ-VAE decoder. This generates novel high-resolution images.

**When to use which:**
- ESRGAN: deterministic SR (one LR input → one best HR output); fast inference; image restoration
- VQ-VAE + Transformer: diverse, controllable generation; high-quality synthesis from class label or text condition; unconditional generation

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How do you evaluate high-resolution image synthesis quality? What metrics do you use and why do PSNR and perceptual quality correlate poorly?"

### Signal Being Tested

Does the candidate understand PSNR, SSIM, LPIPS, and FID for SR evaluation? Can they articulate why optimizing for one metric may hurt another?

### Follow-up Probes

- "Why does a model that achieves high PSNR often look blurry to humans?"
- "What is LPIPS and how does it differ from PSNR?"
- "How do you evaluate hallucination in SR — when the model invents texture not in the original?"

---

### Model Answers — Section 5

**No Hire:**
"I would compare the output to the ground truth." Cannot describe PSNR or LPIPS.

**Lean No Hire:**
Mentions PSNR and SSIM but cannot explain the fundamental conflict with perceptual quality.

**Lean Hire:**
Correctly explains PSNR/SSIM as pixel-accuracy metrics and LPIPS as perceptual similarity. Can articulate why they conflict and why the choice depends on use case.

**Strong Hire Answer (first-person):**

SR evaluation is unique because quality has two conflicting definitions, and the right one depends on the use case.

**PSNR (Peak Signal-to-Noise Ratio):**
```
PSNR = 10 · log_10(MAX²/MSE) = 10 · log_10(MAX²/||x_HR - x̂_HR||²/N)
```
PSNR measures pixel-level accuracy. Higher is better. Typical values: 28–32 dB for good SR at 4×. PSNR is maximized by outputting the conditional mean (the average of all plausible HR images given LR). This mean is smooth — individual textures are averaged out. Result: high PSNR images look clean but blurry.

**SSIM (Structural Similarity Index):**
```
SSIM(x, x̂) = (2μ_xμ_x̂ + c_1)(2σ_{xx̂} + c_2) / ((μ_x² + μ_x̂² + c_1)(σ_x² + σ_x̂² + c_2))
```
SSIM is more sensitive to structural similarity than raw pixel differences. Ranges from 0 to 1 (1 = identical). Better correlated with perceptual quality than PSNR, but still favors blurry outputs over sharp but slightly misaligned textures.

**LPIPS (Learned Perceptual Image Patch Similarity):**
```
LPIPS(x, x̂) = Σ_l w_l · ||φ_l(x) - φ_l(x̂)||²
```
where φ_l is the feature map at layer l of a pretrained AlexNet, VGG, or SqueezeNet. LPIPS measures perceptual distance — how different the images look to a human-like perceptual system. Lower LPIPS = more perceptually similar. LPIPS is significantly better correlated with human quality judgments than PSNR or SSIM for SR evaluation.

**FID for generative SR:**
For the VQ-VAE + transformer approach (unconditional generation), FID measures distributional quality. Not used for deterministic SR (where we have a ground truth).

**The fundamental conflict:**
A model optimizing L2 pixel loss achieves high PSNR by outputting the conditional mean — smooth, blurry, but close to the average. A model optimizing adversarial + perceptual loss sacrifices PSNR for sharp, realistic textures. It may place a high-frequency texture 2 pixels offset from the ground truth — this has high LPIPS score (looks similar) but lower PSNR (pixel values differ).

**Hallucination detection:**
Measure consistency between SR outputs of paired crops from the same source image at different locations. If the model is hallucinating texture, it will produce inconsistent textures at overlapping regions. Also measure FID on SR outputs vs. real HR images — real hallucinated textures should be indistinguishable from real, and FID measures this.

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "How do you serve a high-resolution image synthesis system at scale for consumer photo editing?"

### Signal Being Tested

Does the candidate understand the specific serving challenges of SR (variable input sizes, tile-based processing for large images) and the compute requirements?

### Follow-up Probes

- "A user uploads a 12MP photo and wants 4× SR — what is the output size and how do you process it efficiently?"
- "What is tile-based SR and why is it necessary?"
- "How do you handle the seam artifacts between tiles in tile-based SR?"

---

### Model Answers — Section 6

**No Hire:**
"Run the model on the full image." No understanding of memory constraints for high-resolution SR.

**Lean No Hire:**
Recognizes that large images need to be processed in tiles but cannot describe the seam artifact problem or its solution.

**Lean Hire:**
Correctly describes tile-based SR, overlapping tiles for seam prevention, and the memory requirement calculation. Can estimate compute requirements.

**Strong Hire Answer (first-person):**

A 12MP image at 4× SR outputs a 192MP image (12,000 × 16,000 pixels for a typical camera). This is far too large to process in a single GPU forward pass — a 12MP input alone requires ~150MB at FP32, and the intermediate activations in a deep network multiply this by 10–50×.

**Tile-based SR:**
Process the image in overlapping tiles:
1. Divide the input LR image into tiles of size 256×256 (with overlap)
2. SR each tile independently
3. Blend overlapping regions to remove seams

The overlap is critical: without overlap, the SR network sees a hard edge at tile boundaries (the convolution receptive field is cut off), producing visible seam artifacts. With 32-pixel overlap on each side, the blending region hides any boundary artifacts.

Blending: for the overlapping region, apply a linear weight ramp (the center of the tile gets weight 1, the edge gets weight 0) and blend adjacent tiles by their weighted average.

**Memory budget:**
A 256×256 LR tile at FP32 requires 0.75MB. ESRGAN with 23 RRDB blocks has ~16M parameters, activations ~50MB peak during forward pass per tile. An A10G GPU (24GB) can process a batch of ~400 256×256 tiles simultaneously.

**Throughput:**
12MP input at 4× SR: 12M / (256²) ≈ 183 tiles. At batch 400 tiles in ~1s: full 12MP SR in ~0.5s. Achievable target: 4× SR of a 12MP image in < 2s including image loading and tile assembly.

**Progressive SR for user experience:**
First return a quick 2× bicubic upscale (near-instantaneous) while the full SR is computing. When the neural SR completes, swap to the high-quality result. This gives immediate visual feedback while the higher quality result loads.

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the most critical failure modes of high-resolution image synthesis, and how do you detect them?"

### Signal Being Tested

Does the candidate identify hallucination artifacts (texture hallucination in faces, text), over-sharpening, and seam artifacts? Can they propose detection strategies?

### Follow-up Probes

- "What happens when SR is applied to faces? What specific artifacts appear?"
- "How do you prevent SR from creating fake text or numbers in a document scan?"
- "What is over-sharpening and when does it become a failure?"

---

### Model Answers — Section 7

**No Hire:**
Cannot describe SR-specific failure modes. Generic "bad output quality."

**Lean No Hire:**
Mentions "artifacts" but cannot describe the specific types (hallucinated faces, fake text, seam artifacts) or their causes.

**Lean Hire:**
Correctly identifies face hallucination, text/number fabrication, seam artifacts, and over-sharpening. Proposes region-specific quality checks.

**Strong Hire Answer (first-person):**

High-resolution SR has failure modes that are more subtle than GAN-style failures because the output looks superficially high quality.

**Face hallucination:**
When SR is applied to an image containing faces, the model may generate a "default" face texture that looks plausible but does not faithfully preserve the person's actual appearance. A face at low resolution becomes a photorealistic but different-looking face at high resolution. This is particularly problematic for portrait enhancement apps where users expect their own face to be enhanced, not replaced.

Mitigation: region-specific SR — detect face regions (using a face detector), apply a face-specific SR model trained to preserve identity (evaluated by ArcFace identity similarity), and blend with the general SR model for non-face regions.

**Text and number fabrication:**
The SR model may hallucinate incorrect characters in text regions. "STREET" at low resolution becomes a sharp version of the correct text — but if the low-res image is very degraded, the model may generate the wrong characters. This is catastrophic for document digitization.

Detection: run OCR on both the LR input and SR output; compare recognized text. If they disagree, flag the image for human review. Alternatively, detect text regions and apply a text-specific SR model that enforces character consistency.

**Seam artifacts (tile-based processing):**
Even with overlapping tiles and blending, subtle seam artifacts can appear if the model produces slightly different outputs for the same image region when it appears in different tile positions. Cause: the model's receptive field is limited to the tile; global context changes between tiles.

Mitigation: increase tile overlap (64 pixels instead of 32); use a larger tile size to give the model more context; apply frequency-domain blending (feather tiles in the Laplacian pyramid space) instead of simple linear blending.

**Over-sharpening and halos:**
Perceptual loss + adversarial training can cause over-sharpening — generating edges that are sharper than physically possible, with "halo" artifacts around high-contrast edges.

Detection: measure edge response at high-contrast boundaries using derivative filters; compare to statistics from real high-resolution images.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "You've built 4× SR for consumer photos. Now your platform needs to support SR for medical imaging, satellite imagery, and document scanning — each with different quality requirements. How do you architect a platform?"

### Signal Being Tested

Does the candidate identify that different domains need different models and different quality metrics, but can share the same serving infrastructure?

### Follow-up Probes

- "What is fundamentally different about medical imaging SR that changes the architecture?"
- "What shared infrastructure provides leverage across these use cases?"

---

### Model Answers — Section 8

**No Hire:**
"Use the same model for everything." No domain-specific understanding.

**Lean No Hire:**
Suggests per-domain models but doesn't identify the different quality metrics or the shared infrastructure opportunities.

**Lean Hire:**
Correctly identifies that medical SR needs PSNR optimization (not perceptual), satellite needs different texture priors, and document needs text-preserving SR. Notes shared serving infrastructure.

**Strong Hire Answer (first-person):**

Different SR domains require fundamentally different models and evaluation criteria, but they can share infrastructure.

**Domain-specific models:**
- *Medical imaging*: PSNR-optimized SR (L2 loss only, no adversarial). Hallucinated textures could mask pathology — conservatism is paramount. Model is validated by radiologists, not FID or LPIPS. Regulatory compliance (FDA clearance for diagnostic use) requires extensive clinical validation.
- *Satellite imagery*: texture priors are different (building edges, vegetation, water) — a model trained on natural photos will generate incorrect urban textures for satellite images. Domain-specific training data (EO satellites).
- *Document scanning*: text and line preservation, not texture realism. Use a document-specific model trained with CER (character error rate) as the evaluation metric.

**Shared infrastructure:**
- Serving layer (tile-based SR pipeline, batch processing) is fully shared
- Evaluation harness (automated metric computation, human rating interface) is shared with domain-specific metric plugins
- Model registry: version all SR models with their training data, evaluation metrics, and deployment approval status
- Degradation augmentation library: shared across domains (blur, noise, JPEG are common to all), with domain-specific extensions

**API design:**
```
POST /v1/super-resolution
{
  "input_image_url": "...",
  "scale_factor": 4,
  "domain": "medical" | "satellite" | "document" | "general",
  "quality_mode": "conservative" | "perceptual"
}
```
The `domain` parameter routes to the appropriate model. The `quality_mode` parameter selects the loss-function-trained variant (pixel-accuracy vs. perceptual).

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**L2 pixel loss:**
```
L_pixel = ||x_HR - G(x_LR)||²_F = MSE(x_HR, G(x_LR))
```

**Perceptual (feature) loss:**
```
L_perceptual = Σ_i ||φ_i(x_HR) - φ_i(G(x_LR))||²_F
(φ_i = VGG feature maps at layer i)
```

**Combined ESRGAN loss:**
```
L = L_pixel + λ_p·L_perceptual + λ_a·L_adv
```

**PSNR:**
```
PSNR = 10·log_10(MAX²/MSE),  MAX = 255 for uint8
```

**SSIM:**
```
SSIM(x,x̂) = (2μ_x μ_x̂ + c_1)(2σ_{xx̂} + c_2) / ((μ_x² + μ_x̂² + c_1)(σ_x² + σ_x̂² + c_2))
```

**LPIPS:**
```
LPIPS(x, x̂) = Σ_l w_l·||φ_l(x) - φ_l(x̂)||²
```

**VQ-VAE total loss:**
```
L_VQ-VAE = L_recon + ||sg[z_e] - e||² + β||z_e - sg[e]||²
```

**VQ-VAE quantization:**
```
z_q = e_k,  k = argmin_j ||z_e - e_j||²
```

**Autoregressive prior:**
```
p(z_1,...,z_N) = Π_{i=1}^{N} p(z_i | z_1,...,z_{i-1})
```

**PixelShuffle (sub-pixel convolution):**
```
PS: R^{H×W×r²C} → R^{rH×rW×C}
```

### Vocabulary Cheat Sheet

| Term | Definition |
|---|---|
| **Super-resolution (SR)** | Upscaling a low-resolution image to high resolution |
| **PSNR** | Peak Signal-to-Noise Ratio; pixel-level accuracy metric (dB) |
| **SSIM** | Structural Similarity Index; structural fidelity metric |
| **LPIPS** | Learned Perceptual Image Patch Similarity; perceptual quality metric |
| **Perceptual loss** | Loss in VGG feature space; produces sharper results than pixel loss |
| **ESRGAN** | Enhanced SRGAN; RRDB generator, relativistic discriminator |
| **RRDB** | Residual-in-Residual Dense Block; building block of ESRGAN generator |
| **VQ-VAE** | Vector Quantized VAE; encodes to discrete codebook tokens |
| **Codebook** | Set of learned embedding vectors; VQ-VAE quantizes to nearest |
| **Straight-through estimator** | Passes gradients through argmin (non-differentiable) via identity |
| **PixelShuffle** | Sub-pixel convolution for upsampling; avoids checkerboard artifacts |
| **Blind SR** | SR with unknown degradation; requires robust degradation modeling |
| **Tile-based SR** | Process large images as overlapping tiles; blend seams |
| **Progressive SR** | Show quick bicubic SR first while neural SR computes |
| **Relativistic GAN** | Discriminator predicts relative realness, not absolute |

### Key Numbers Table

| Metric | Value |
|---|---|
| Good PSNR (4× SR) | 28–32 dB |
| Good SSIM (4× SR) | > 0.85 |
| Good LPIPS (4× SR) | < 0.10 |
| SR computation (12MP 4× on A10G) | ~0.5–2s |
| ESRGAN generator parameters | ~16M |
| Typical tile size | 256×256 pixels |
| Tile overlap | 32–64 pixels |
| VQ-VAE codebook size (typical) | 8192 entries |
| VQ-VAE latent dimension | 256 per token |
| VQ-VAE compression ratio (FFHQ) | 16× (64×64 tokens for 256×256 image) |
| ESRGAN training data (DIV2K) | 800 training images |
| ESRGAN training data (DF2K) | 3450 images (DIV2K + Flickr2K) |

### Rapid-Fire Day-Before Review

1. **Why does L2 pixel loss produce blurry results?** Optimizes MSE over all plausible outputs → conditional mean → smooth blur
2. **Perceptual loss advantage?** Matches VGG feature maps → similar high-level structure → sharper perceptual texture
3. **RRDB vs. residual block?** RRDB uses dense connections (each layer connects to all previous) + residual blocks → better gradient flow
4. **VQ-VAE quantization step?** Map continuous z_e to nearest codebook vector e_k via argmin
5. **Straight-through estimator?** During backprop, pass gradient through argmin as identity (treat z_q = z_e)
6. **PSNR vs. LPIPS trade-off?** PSNR rewards pixel accuracy (blurry), LPIPS rewards perceptual similarity (sharp)
7. **Tile-based SR overlap purpose?** Provides context at boundaries; prevents seam artifacts; blended in overlapping region
8. **PixelShuffle advantage?** Convolve at LR resolution, then rearrange channels to higher resolution → avoids deconvolution checkerboard artifacts
9. **When to choose VQ-VAE+GPT over ESRGAN?** When diversity and generative control (unconditional generation, class-conditional synthesis) matter more than deterministic SR quality
10. **Medical SR quality metric?** PSNR/SSIM (pixel accuracy) not LPIPS/FID — hallucinated texture could mask pathology
