# Realistic Face Generation (GAN) — Staff/Principal Interview Guide

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

> "Design a system that generates photorealistic, high-resolution human face images. The system should generate diverse faces — different ages, genders, ethnicities, and expressions — that are indistinguishable from real photographs. Walk me through your approach."

### Signal Being Tested

Does the candidate recognize that face generation involves specific challenges around diversity, realism at high resolution, and the ethical implications of synthetic face generation?

### Six Clarification Dimensions

| Dimension | Why It Matters |
|---|---|
| **Output resolution** | 128×128 vs. 1024×1024 — dramatically different architecture requirements |
| **Conditional vs. unconditional** | Random diverse faces vs. controlled attribute generation (age, expression) |
| **Diversity requirements** | Does distribution need to match population demographics? |
| **Use case and safety** | Avatar generation vs. deepfake research — safety controls differ |
| **Real-time vs. offline** | Interactive avatar creation vs. batch dataset generation |
| **Identity preservation** | Pure generation vs. generating variations of a specific real face |

### Follow-up Probes

- "How does your design change if you need to generate faces with specific attribute controls (smile intensity, age, etc.)?"
- "At what resolution does the architecture fundamentally need to change, and why?"
- "What ethical safeguards must be built into a face generation system before deployment?"

---

### Model Answers — Section 1

**No Hire:**
"I would train a model on photos of faces." No recognition of GAN-specific challenges, diversity requirements, or ethical implications.

**Lean No Hire:**
Mentions GAN as the approach but doesn't ask about resolution, conditional control, or use case safety. Cannot articulate why face generation differs from general image generation.

**Lean Hire:**
Asks about resolution, conditionality, and use case. Identifies that high-resolution generation (1024×1024) requires a progressive growing or style-based architecture (StyleGAN). Notes ethical implications.

**Strong Hire Answer (first-person):**

Face generation has both technical and ethical complexity that I want to surface before designing anything.

On the technical side, the resolution is the primary architectural constraint. Generating a 128×128 face is feasible with a basic GAN; generating a photorealistic 1024×1024 face requires a fundamentally different architecture (progressive growing, style injection, multi-scale training). The resolution requirement drives the entire architecture choice.

I also need to understand conditionality. Pure unconditional generation (random diverse faces) is simpler — sample a latent vector and generate. Controlled generation (I want a face with these specific attributes: female, 40s, smiling) requires conditional architectures — either cGAN with attribute conditioning or disentangled latent space control. StyleGAN's W-space provides attribute editing capabilities that don't require explicit conditioning during training.

The diversity requirement is both a product requirement and an ethical one. If the training data and the model over-represent certain demographics (lighter skin tones, Western facial structures), the generated faces will also over-represent them. This is not just a quality issue — for use cases like avatar creation, demographic under-representation is a product failure.

On the ethical side, I would ask about the use case carefully. Photorealistic face generation technology has been misused for deepfake media creation and synthetic identity fraud. Before designing the system, I would want to ensure: (1) the system generates faces that are clearly synthetic (detectable by forensic tools), (2) there are controls preventing generation of real specific individuals, and (3) deployment is gated behind a use-case review process.

Let me proceed assuming: 1024×1024 resolution, unconditional generation with post-hoc attribute editing capability, balanced demographic diversity, and avatar/game asset use case.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "How do you formally frame the face generation problem as an ML problem? Why is GAN the right approach over diffusion or VAE?"

### Signal Being Tested

Does the candidate understand the generative modeling paradigm for GANs and can they articulate the trade-offs between GANs, diffusion models, and VAEs for face generation specifically?

### Follow-up Probes

- "What does the discriminator learn? What does its output represent mathematically?"
- "What are the failure modes of standard GAN training that progressive growing and WGAN address?"
- "When would you choose a diffusion model over StyleGAN for face generation?"

---

### Model Answers — Section 2

**No Hire:**
Cannot frame as an ML problem beyond "train on faces." Doesn't know the GAN training objective.

**Lean No Hire:**
Knows the GAN framework (generator + discriminator) but cannot explain the loss function or why GAN training is unstable.

**Lean Hire:**
Correctly states the GAN minimax objective. Explains mode collapse and gradient vanishing as GAN failure modes. Can compare GAN to diffusion at a high level.

**Strong Hire Answer (first-person):**

Face generation is the task of learning the distribution p(x) over photorealistic face images and sampling from it. The GAN framework parameterizes this through a two-player adversarial game.

**GAN Minimax Objective:**
```
min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1-D(G(z)))]
```

The discriminator D : X → [0,1] tries to distinguish real images (output 1) from generated images (output 0). The generator G : Z → X maps a noise vector z ~ N(0, I) to a realistic image. At the Nash equilibrium, p_G = p_data and D(x) = 1/2 everywhere.

**Why GAN over VAE?**
VAEs learn a smoothly interpolating latent space but produce blurry outputs because the reconstruction objective is an L2 loss (which averages over modes). For photorealistic face generation, blur is unacceptable — every pixel detail matters. GANs don't optimize pixel-level reconstruction; they optimize distribution-level realism, which is what we want.

**Why GAN over Diffusion for real-time applications?**
Diffusion models require 20–1000 denoising steps at inference, each a full U-Net forward pass. A 1024×1024 image with 50 DDIM steps takes ~5 seconds on a modern GPU. StyleGAN generates a 1024×1024 face in a single forward pass (~100ms). For real-time avatar generation or interactive applications, the single-pass inference of GANs is a major advantage.

However, for offline high-quality generation where latency is not a constraint, modern diffusion models (DALL-E 3, Stable Diffusion XL) now match or exceed GAN quality with better diversity and text-conditional control.

**GAN training failure modes and solutions:**
1. *Mode collapse*: the generator learns to produce a small subset of faces (only one demographic, only one expression). The discriminator cannot distinguish real from fake for this small subset, so the generator has no incentive to explore other modes. Solution: mini-batch discrimination (discriminator looks at batch statistics, penalizing low diversity), Wasserstein distance (WGAN).
2. *Gradient vanishing*: when D is too good, log(1-D(G(z))) ≈ 0, providing near-zero gradient to G. Solution: use `-log D(G(z))` (non-saturating loss) instead.
3. *Training instability*: loss oscillates without convergence. Solution: WGAN-GP (gradient penalty), careful learning rate scheduling, balanced D/G update ratios.

---

## Section 3: Data & Preprocessing (8 min)

### Interviewer Prompt

> "What training data do you use for face generation, and how do you preprocess it?"

### Signal Being Tested

Does the candidate know the standard face generation datasets (FFHQ, CelebA-HQ) and understand the preprocessing pipeline specific to face images?

### Follow-up Probes

- "What is FFHQ and why was it created specifically for GAN training?"
- "Why is face alignment preprocessing critical for face generation?"
- "How do you ensure demographic diversity in your training data?"

---

### Model Answers — Section 3

**No Hire:**
"I would scrape photos from the internet." No understanding of face-specific datasets or preprocessing.

**Lean No Hire:**
Mentions CelebA but cannot describe FFHQ or explain face alignment.

**Lean Hire:**
Describes FFHQ (70K high-quality 1024×1024 images, diverse, CC0 license), explains face alignment, and mentions demographic diversity assessment.

**Strong Hire Answer (first-person):**

**FFHQ (Flickr-Faces-HQ):** The standard training dataset for high-quality face generation, introduced alongside the original StyleGAN paper. 70,000 images at 1024×1024 resolution, scraped from Flickr under Creative Commons licenses. Carefully curated: includes diverse ages, ethnicities, accessories (glasses, hats), expressions, and lighting conditions. This diversity is what makes FFHQ superior to earlier datasets like CelebA (which was dominated by celebrity faces with limited demographic diversity).

**Face alignment preprocessing:**
All faces must be aligned to a canonical orientation before training. Without alignment, the generator would have to simultaneously learn facial structure AND the transformation from arbitrary orientations — massively increasing task difficulty.

Standard alignment for FFHQ:
1. Detect facial landmarks (68-point or 5-point models: eye centers, nose tip, mouth corners)
2. Compute the affine transformation that maps the detected landmarks to a canonical template position
3. Apply transformation to crop and resize the image to 1024×1024

The canonical template positions are standardized: eyes at specific pixel coordinates, face centered, appropriate scale. StyleGAN was trained with faces aligned so that eye locations are always at specific pixel positions.

**Additional preprocessing:**
- Color normalization: normalize pixel values to [-1, 1] (suitable for tanh output activation in generator)
- Quality filtering: remove blurry images (Laplacian variance < threshold), images with occlusions, or images where face alignment failed
- Data augmentation: horizontal flip (faces are approximately symmetric) — doubles the effective dataset size without distorting facial structure

**Demographic diversity assessment:**
Before training, I audit the training data distribution using a face attribute classifier. Key dimensions: apparent gender (binary labels are imperfect but useful), apparent age group, skin tone (using the Fitzpatrick scale or ITA — Individual Typology Angle). If any demographic group is under-represented by more than 2× relative to its estimated global population frequency, I augment with additional data from targeted Flickr searches or synthetic augmentation.

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "Walk me through the StyleGAN architecture in detail. What are the key innovations over a vanilla GAN, and why does each one improve quality?"

### Signal Being Tested

Does the candidate understand StyleGAN's style injection mechanism, the mapping network, progressive growing (StyleGAN1), and the key differences from a vanilla GAN? Can they explain what AdaIN does?

### Follow-up Probes

- "What is the mapping network and why is it needed? What problem does mapping z → w solve?"
- "Explain AdaIN (Adaptive Instance Normalization). What does it do in each layer?"
- "What is the path length regularization in StyleGAN2, and what problem does it solve?"

---

### Model Answers — Section 4

**No Hire:**
"I would use a GAN with convolutions." Cannot describe StyleGAN's innovations or why they improve quality.

**Lean No Hire:**
Knows StyleGAN is a state-of-the-art GAN but cannot describe the mapping network, AdaIN, or progressive growing.

**Lean Hire:**
Correctly describes the mapping network, AdaIN style injection, and progressive growing. Can explain what each innovation solves in terms of training quality or generated image quality.

**Strong Hire Answer (first-person):**

StyleGAN (Karras et al., NVIDIA) introduced several innovations over vanilla GANs that collectively produce significantly higher quality and more controllable generation.

**Innovation 1: Mapping Network (Z → W)**
Vanilla GANs sample z from a Gaussian distribution N(0, I) and feed directly to the generator. The problem: the Gaussian prior forces the model to match a disentangled normal distribution to a highly structured distribution (human faces). Features that are correlated in face space (e.g., head rotation and shadow position) will be entangled in z-space.

StyleGAN's mapping network is an 8-layer MLP that transforms z ∈ R^{512} → w ∈ R^{512}. The W-space learned by the mapping network is empirically more disentangled than Z-space — changing one dimension of w is more likely to change one interpretable attribute (age, smile) without affecting others. This is measured by the Perceptual Path Length (PPL) metric.

**Innovation 2: Style Injection via AdaIN**
Instead of feeding z or w directly to the generator layers, StyleGAN uses a synthesis network that generates images from a learned constant starting point. The W vector controls the synthesis via Adaptive Instance Normalization (AdaIN) at each layer:

```
AdaIN(x_i, y) = y_{s,i} · (x_i - μ(x_i)) / σ(x_i) + y_{b,i}
```

where x_i is the feature map at layer i, μ(x_i) and σ(x_i) are its mean and standard deviation, and y_{s,i}, y_{b,i} are scale and bias computed from w via learned affine transforms. This injects the style (the w vector) at each resolution level, allowing coarse styles (overall structure, identity) to be controlled by early layers and fine styles (texture, color details) by later layers.

**Innovation 3: Stochastic Variation via Noise Injection**
At each layer, Gaussian noise B is added to the feature maps before AdaIN. This provides a source of stochastic fine-grained variation (individual hair strands, skin pores, freckles) that the style vector w doesn't need to control. Stochastic variation produces more realistic texture at high resolution.

**Innovation 4: Progressive Growing (StyleGAN1)**
Training starts at 4×4 resolution and progressively doubles (8×8, 16×16, ..., 1024×1024). Each resolution is trained until stable, then the next resolution is faded in. This allows the model to first learn coarse facial structure (face shape, overall proportions) before learning fine details (eyelash texture, skin pores). Without progressive growing, training at 1024×1024 from scratch often diverges.

**StyleGAN2 improvements:**
- *Weight demodulation* replaces AdaIN: normalizes the convolutional weights by the expected magnitude of the input feature maps — avoids "droplet" artifacts that AdaIN can produce
- *Path length regularization*: penalizes the generator if a fixed-magnitude step in W-space produces images that differ by more than expected. This encourages smooth, predictable latent space traversal.

**Discriminator architecture:**
The discriminator is a ResNet-style architecture that progressively downsamples the input image to a scalar. In StyleGAN, the discriminator uses a mirror of the generator's multi-scale structure.

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How do you evaluate the quality of a face generation model? Walk me through FID and IS, and their limitations."

### Signal Being Tested

Does the candidate understand FID and IS computation, their limitations, and how to complement them with face-specific evaluation?

### Follow-up Probes

- "Walk me through the FID calculation. What does it actually measure?"
- "Why is IS insufficient for face generation specifically?"
- "How do you evaluate diversity separately from quality?"

---

### Model Answers — Section 5

**No Hire:**
"I would look at the generated images and check if they look real." Cannot describe FID or IS.

**Lean No Hire:**
Mentions FID and IS but cannot explain how FID is computed or what the Fréchet distance measures.

**Lean Hire:**
Correctly explains FID as distribution distance in Inception feature space. Can explain IS formula. Notes that FID correlates better with human quality judgment than IS.

**Strong Hire Answer (first-person):**

Face generation quality is measured with two complementary automated metrics plus human evaluation.

**FID (Fréchet Inception Distance):**

FID computes the Fréchet distance between the distributions of real and generated images in the feature space of a pretrained Inception-v3 network:

```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^{1/2})
```

where μ_r, Σ_r are the mean and covariance of Inception features for real images, and μ_g, Σ_g are for generated images. Lower FID = better quality.

The Fréchet distance assumes both distributions are Gaussian in Inception feature space (an approximation) and measures both the difference in means (quality: generated features should have similar mean to real) and covariances (diversity: the spread of generated features should match real).

Typical FID scores:
- FFHQ ground truth: ~4 (self-consistency)
- StyleGAN2 on FFHQ: ~3–4 (near perfect)
- DCGAN on FFHQ: ~80+

Standard practice: compute FID using 50K real images and 50K generated images. Computing on < 10K images produces unreliable estimates.

**IS (Inception Score):**

```
IS = exp(E_{x~p_g}[KL(p(y|x) || p(y))])
```

where p(y|x) is the Inception classifier's conditional label distribution and p(y) is the marginal. High IS requires each generated image to be clearly recognizable (low entropy p(y|x)) and the set of images to be diverse (high entropy p(y)).

IS is problematic for face generation: Inception was trained on ImageNet, not faces. All face images produce similar conditional distributions (Inception doesn't distinguish individual face quality). FID is more appropriate for face generation than IS.

**Face-specific evaluation:**

*Demographic diversity*: run a face attribute classifier on 10K generated images and measure the distribution over gender, age, and ethnicity. Compare to target distribution or FFHQ distribution.

*Face quality*: use a face quality assessment model (e.g., SER-FIQ or FaceQAN) to score sharpness, symmetry, and artifact-freeness.

*Identity coverage*: for face generation, are the generated faces genuinely unique (no mode collapse producing near-duplicate faces)? Measure by computing ArcFace embedding similarity between random pairs — low average similarity = high diversity.

*PPL (Perceptual Path Length)*: measure the perceptual difference between images obtained by interpolating adjacent points in W-space. Lower PPL = smoother latent space = more disentangled representations.

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "How do you serve the face generation model in production for an avatar creation use case?"

### Signal Being Tested

Does the candidate understand GAN inference characteristics (single forward pass, no iterative decoding) and the specific serving optimizations available for GANs?

### Follow-up Probes

- "How do you handle the latency requirement for interactive avatar generation?"
- "What is truncation trick and when would you use it in production?"
- "How do you implement attribute editing (age, expression) at inference time?"

---

### Model Answers — Section 6

**No Hire:**
"I would run the model on a GPU." No understanding of GAN-specific serving characteristics.

**Lean No Hire:**
Notes that GANs are fast at inference (single forward pass) but cannot describe truncation trick, attribute editing, or batch generation for production.

**Lean Hire:**
Correctly explains GAN inference as a single forward pass (~100ms for 1024×1024). Describes truncation trick for quality-diversity trade-off. Explains attribute editing via W-space manipulation.

**Strong Hire Answer (first-person):**

GANs have fundamentally different serving characteristics from diffusion models: inference is a single forward pass through the generator, not an iterative process. This makes GANs naturally suited for real-time interactive applications.

**Inference latency:**
StyleGAN2 generating a 1024×1024 face on an A100 GPU: ~30ms. Generating a 512×512 face: ~10ms. This is fast enough for interactive avatar creation — a user clicks "generate" and gets an immediate result.

**Truncation trick for quality-diversity control:**
The full W-space covers the entire training distribution, including low-quality outliers (unusual faces). For production avatar generation, I want consistently high-quality outputs. The truncation trick replaces each w with:
```
w' = w̄ + ψ · (w - w̄)
```
where w̄ is the mean W-vector and ψ ∈ (0, 1] is the truncation factor. ψ=1 gives the full distribution; ψ=0.7 constrains generation to the "safe" high-quality region near the mean. Trade-off: smaller ψ = higher average quality but less diversity.

**Attribute editing at inference:**
StyleGAN's W-space supports attribute editing: identify the direction d_attr in W-space corresponding to a specific attribute (smiling, younger, etc.) using a linear classifier on labeled data. At inference, edit an attribute by:
```
w_edited = w + α · d_attr
```
where α controls the editing magnitude. This requires no re-training — all attribute control happens at inference time. This is a key selling point for avatar customization: users can fine-tune age, smile, etc. after generating a base face.

**Batch generation for dataset creation:**
For batch face dataset generation (e.g., creating training data for other models), I run the generator in large batches with mixed precision (FP16). A 1024-image batch on 8 A100s: ~2 minutes for 1M faces (50K per A100, each taking ~2.4 seconds at batch 64).

**Serving infrastructure:**
- REST API: `POST /generate` with optional `{attributes: {age: 0.8, gender: 0.6, ethnicity: ...}}`
- Returns base64-encoded 1024×1024 JPEG
- Safety filter: run a face detection check on each generated image (ensure it's a valid face, not an artifact); run a deepfake detector to ensure output passes as "synthetic" (flagged as AI-generated)
- Watermarking: invisible watermark embedded in each generated image for provenance tracking

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the critical failure modes for a face generation system, both technical and ethical?"

### Signal Being Tested

Does the candidate identify mode collapse, demographic bias, artifact generation, and misuse/deepfake risks? Can they propose concrete mitigations?

### Follow-up Probes

- "What is mode collapse and how do you detect it in a face GAN?"
- "The generated faces are demographically skewed — primarily lighter skin tones and certain age groups. What caused this and how do you fix it?"
- "How do you prevent this system from being used to generate realistic fake identities for fraud?"

---

### Model Answers — Section 7

**No Hire:**
Cannot describe GAN-specific failure modes. Generic "bad images."

**Lean No Hire:**
Mentions mode collapse but cannot define it precisely or describe detection methods.

**Lean Hire:**
Correctly defines mode collapse and its detection (FID diversity component, face embedding similarity). Identifies demographic bias as a training data issue. Mentions deepfake risk.

**Strong Hire Answer (first-person):**

Face generation failure modes span technical and ethical dimensions.

**Technical: Mode collapse**
The generator produces only a small subset of the face distribution — perhaps only one demographic group, one age range, or one expression type. The discriminator can't distinguish these from real (because they are realistic), so the generator has no incentive to explore other modes.

Detection: measure pairwise ArcFace embedding similarity between random samples. If average similarity is high (> 0.5), mode collapse is occurring. Also monitor FID — the diversity component of FID (covariance term) will diverge if the generated distribution is less spread than the real distribution.

Mitigation: mini-batch discrimination in the discriminator, WGAN-GP (Wasserstein loss with gradient penalty prevents the discriminator from becoming too strong, which is a common collapse trigger), increased latent code dimension.

**Technical: High-frequency artifacts ("checkerboard" patterns)**
Spectral artifacts from upsampling operations (nearest-neighbor or bilinear upsampling with stride-2 convolutions). StyleGAN2's weight demodulation addresses the "blob artifact" seen in StyleGAN1.

Detection: compute spectral analysis of generated images (FFT); real images have smooth frequency spectra while artifact images show high-frequency spikes.

**Ethical: Demographic bias**
Generated faces over-represent lighter skin tones and under-represent older faces, because FFHQ itself (scraped from Flickr) has these biases.

Mitigation: (1) targeted data augmentation with additional images from under-represented demographics, (2) conditional GAN with demographics as conditioning signal + balanced sampling, (3) post-hoc fairness auditing with attribute classifiers.

**Ethical: Deepfake/fraud misuse**
Photorealistic faces can be used to create synthetic identities for fraud (fake LinkedIn profiles, fake IDs) or to create non-consensual deepfake imagery.

Mitigation: (1) invisible watermarking via steganography in pixel values or frequency domain — the watermark is invisible to humans but detectable by scanners; (2) C2PA metadata (Content Provenance and Authenticity) embedded in file metadata; (3) usage gating — require API registration with stated use case; (4) rate limiting on per-user generation volume.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "You've built a face generation system. Now your company wants to offer a face generation API to game developers, app developers, and marketing teams. How does building a platform change your design and responsibilities?"

### Signal Being Tested

Does the candidate think about API design for creative use cases, safety at the platform level, and the responsibilities that come with widely deploying a face generation system?

### Follow-up Probes

- "What API controls do you expose to developers to prevent misuse?"
- "How do you implement watermarking at the platform level, and why is it your responsibility not the developer's?"

---

### Model Answers — Section 8

**No Hire:**
"Expose a REST API and let developers use it." No consideration of safety responsibilities.

**Lean No Hire:**
Mentions rate limiting as a safety measure but doesn't address deepfake prevention, watermarking, or the platform's legal responsibility for misuse by third-party developers.

**Lean Hire:**
Describes API with attribute controls, mandatory watermarking, and usage policy enforcement. Notes that the platform bears responsibility for misuse even by third-party developers.

**Strong Hire Answer (first-person):**

Moving from a product to a platform multiplies both the impact and the responsibility.

**API design:**
```
POST /v1/faces/generate
{
  "resolution": 1024,
  "count": 1,
  "attributes": {
    "diversity_mode": "balanced",  // balanced | custom
    "age_range": [20, 40],         // optional
    "expression": "neutral"        // optional
  },
  "usage_context": "game_asset"    // declared use case
}
```

The `usage_context` is recorded for audit purposes. Different use cases have different risk profiles: `game_asset` is low risk; `social_media_avatar` is medium risk; `realistic_identity_document` is rejected.

**Mandatory watermarking as platform responsibility:**
Every generated face must be watermarked regardless of what the developer does downstream. The watermark contains: API key hash (to trace which developer generated it), generation timestamp, model version. This is the platform's responsibility — developers cannot be trusted to apply watermarks themselves, and omitting watermarks enables fraud.

Implementation: invisible steganographic watermark using HiDDeN or similar invisible watermarking models that embed a 48-bit payload in the image's frequency domain. Survives JPEG compression at quality > 70%, resizing > 50%, and mild color adjustments.

**Content provenance standard compliance:**
Implement C2PA (Coalition for Content Provenance and Authenticity) manifest embedded in file metadata. This is becoming the industry standard for AI-generated content disclosure.

**Developer safety responsibilities:**
- Rate limiting per API key
- Identity verification for developers handling large-scale generation (> 10K faces/day)
- Prohibition on: identity document generation, non-consensual imagery, minor generation
- Right to terminate access for policy violations

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**GAN minimax objective:**
```
min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1-D(G(z)))]
```

**Non-saturating generator loss (practical improvement):**
```
L_G = -E_{z~p_z}[log D(G(z))]   (replaces -E[log(1-D(G(z)))])
```

**Wasserstein distance (WGAN):**
```
W(p_r, p_g) = sup_{||f||_L ≤ 1} E_{x~p_r}[f(x)] - E_{x~p_g}[f(x)]
L_D = -E_{x~p_r}[D(x)] + E_{z~p_z}[D(G(z))]
L_G = -E_{z~p_z}[D(G(z))]
```

**WGAN-GP gradient penalty:**
```
L_D = -E[D(x)] + E[D(G(z))] + λ·E[(||∇_x̂ D(x̂)||_2 - 1)²]
x̂ = εx + (1-ε)G(z), ε ~ Uniform(0,1)
```

**AdaIN (Adaptive Instance Normalization):**
```
AdaIN(x_i, y) = y_{s,i}·(x_i - μ(x_i))/σ(x_i) + y_{b,i}
y_s, y_b from affine transform of w
```

**FID:**
```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^{1/2})
```

**IS (Inception Score):**
```
IS = exp(E_{x~p_g}[KL(p(y|x) || p(y))])
```

**Truncation trick:**
```
w' = w̄ + ψ·(w - w̄),  ψ ∈ (0, 1]
```

**Attribute editing in W-space:**
```
w_edited = w + α·d_attr
```

### Vocabulary Cheat Sheet

| Term | Definition |
|---|---|
| **GAN** | Generative Adversarial Network; generator vs. discriminator adversarial game |
| **StyleGAN** | GAN with mapping network, style injection via AdaIN, and stochastic noise |
| **Mode collapse** | Generator learns to produce only a small subset of the data distribution |
| **WGAN** | Wasserstein GAN; uses Wasserstein distance, stabilizes GAN training |
| **WGAN-GP** | WGAN with gradient penalty; enforces 1-Lipschitz constraint on discriminator |
| **AdaIN** | Adaptive Instance Normalization; injects style (scale/bias) into feature maps |
| **Mapping network** | MLP that maps z → w; more disentangled than direct z usage |
| **W-space** | Intermediate latent space of StyleGAN; empirically more disentangled than Z |
| **Progressive growing** | Train GAN starting at low resolution and progressively increase |
| **Truncation trick** | Shrink w toward mean; trades diversity for quality in sampling |
| **FID** | Fréchet Inception Distance; distribution distance in Inception feature space |
| **PPL** | Perceptual Path Length; measures W-space smoothness/disentanglement |
| **ArcFace** | Face recognition model; embeddings used for face identity comparison |
| **FFHQ** | Flickr-Faces-HQ; 70K diverse 1024×1024 face images for GAN training |
| **C2PA** | Content Provenance and Authenticity; standard for AI content watermarking |

### Key Numbers Table

| Metric | Value |
|---|---|
| FFHQ dataset size | 70,000 images at 1024×1024 |
| StyleGAN2 FID on FFHQ | ~3–4 |
| StyleGAN2 inference (1024×1024, A100) | ~30ms |
| StyleGAN2 generator parameters | ~30M |
| FID: excellent face generation | < 5 |
| FID: good face generation | 5–15 |
| FID: poor | > 50 |
| Truncation ψ (quality-focused production) | 0.5–0.7 |
| Truncation ψ (maximum diversity) | 1.0 |
| Typical W-space dimension | 512 |
| WGAN-GP λ (gradient penalty weight) | 10 |
| Mapping network depth | 8 layers |
| StyleGAN2 training resolution steps | 4,8,16,32,64,128,256,512,1024 |

### Rapid-Fire Day-Before Review

1. **GAN minimax objective?** `min_G max_D E[log D(x)] + E[log(1-D(G(z)))]`
2. **Why non-saturating loss?** `log(1-D(G(z)))` saturates early in training; `-log D(G(z))` provides larger gradients when G is weak
3. **What is WGAN improvement?** Replaces JS divergence with Wasserstein distance; stabilizes training, enables meaningful loss curves
4. **AdaIN purpose?** Injects style (from w vector) into feature maps via per-channel scale and bias
5. **Mapping network purpose?** Transforms z → w; W-space is more disentangled — changing one dimension changes one attribute
6. **Truncation trick trade-off?** ψ < 1 constrains sampling near mean; higher quality, less diversity
7. **FID formula summary?** Squared Fréchet distance between Gaussian fits to Inception features of real and generated images
8. **Mode collapse detection?** High pairwise ArcFace similarity between random generated samples; FID diversity term diverges
9. **Attribute editing?** `w_edited = w + α·d_attr` where d_attr is a linear direction in W-space
10. **Why watermark at platform level?** Developers cannot be trusted to apply watermarks; platform bears responsibility for tracing misuse
