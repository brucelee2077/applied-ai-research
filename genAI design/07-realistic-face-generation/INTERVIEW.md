# Chapter 07: Interviewer Guide — Realistic Face Generation (GANs)

## Opening Problem Statement

"Design a system that generates photorealistic human face images for a gaming company that needs unique NPC (non-player character) portraits — thousands of distinct, high-quality faces at various ages, ethnicities, and expressions. I want to focus on the core architecture trade-offs for face generation, how you'd diagnose and fix mode collapse, the limits of FID as your evaluation metric, and your view on whether you'd still choose a GAN over a diffusion model for this use case in 2025."

---

## Probing Question Tree

### Area 1: Wasserstein Distance — Why It Solves Vanishing Gradients

**Surface:** "Why does Wasserstein GAN have better training stability than the original GAN?"

**Probe 1:** "The original GAN's discriminator saturates — outputs 0.0 for all fakes when the generator is bad. Why does this cause vanishing gradients mathematically, and why does WGAN's Earth Mover's distance not have this problem?"
> *Looking for:* Original GAN: discriminator loss is based on cross-entropy (log likelihood). When discriminator is perfect (p_fake and p_real don't overlap), D(G(z)) ≈ 0 → log(1-D(G(z))) ≈ 0 → gradient of generator loss = ∇G[log(1-D(G(z)))] ≈ 0. No gradient signal. WGAN: critic outputs an unconstrained real-valued score. Critic loss = E[C(x)] - E[C(G(z))]. This is the Wasserstein-1 distance, which is defined even when the distributions are completely disjoint — it measures the 'earth moving cost' between distributions. The gradient is always well-defined because the critic's output can still be very different for real vs. fake (e.g., 10 vs. -10) even when they don't overlap. Formally: W(p_r, p_g) is continuous and differentiable under mild conditions, unlike the Jensen-Shannon divergence used by vanilla GAN.

**Probe 2:** "WGAN requires the critic (not discriminator) to be K-Lipschitz continuous. The original paper uses weight clipping to enforce this. What's wrong with weight clipping, and how does spectral normalization fix it more elegantly?"
> *Looking for:* Weight clipping problems: (1) the constraint forces weights to be in [-c, c], which causes the critic to learn very simple functions (gradients explode or vanish for deep networks). (2) Clipping strength c is a hyperparameter that's hard to tune — too large and the Lipschitz constraint isn't enforced; too small and the critic is too weak. (3) Gradient penalty (WGAN-GP) or spectral normalization are better alternatives. Spectral normalization: normalize each weight matrix by its spectral norm (largest singular value), which guarantees the network is globally 1-Lipschitz. Implemented efficiently using power iteration to estimate the spectral norm, adding negligible overhead. Key benefit: spectral normalization stabilizes both the generator and discriminator, and is the standard approach in modern GANs (StyleGAN2 uses it).

**Expert probe:** "Mode collapse has a mathematical signature: the generator's output distribution has low entropy even when sampled from diverse latent codes. Derive what the GAN training objective converges to when mode collapse occurs, and explain why this is a Nash equilibrium — even though it's a bad one."
> *Looking for:* With mode collapse, the generator learns a mapping G that maps all z to a small set of outputs. The discriminator then learns to recognize this small set as fake. But if the discriminator adapts, the generator has little incentive to change (D is already confused or not — it can always adapt to the small set). More formally: if the generator collapses to a single mode m*, the discriminator converges to D(m*) = 0.5 (can't tell real from fake for that mode if p_real has mass there) and D(x) ≈ 1 for all other real images. The generator is at a local Nash equilibrium: changing G to generate another mode doesn't increase its payoff because D immediately adapts. This is a suboptimal Nash equilibrium — it satisfies the local optimality conditions but not the global minimax solution (which would have the generator covering all modes). Strong hire connects this to the impossibility of simultaneously satisfying the generator and discriminator objectives in a non-convex game.

---

### Area 2: Mode Collapse — Diagnosis and Remediation

**Surface:** "How do you detect and fix mode collapse during training?"

**Probe 1:** "You're training a face GAN and notice the FID stops improving after epoch 50, then starts getting worse. The generated images look realistic individually. How do you diagnose whether this is mode collapse vs. overfitting vs. training instability?"
> *Looking for:* Mode collapse diagnosis: (1) Compute intra-batch diversity — generate a large batch (e.g., 1000 samples) and compute pairwise distances between their features. Low variance = mode collapse. (2) Coverage metric: sample 10K generated images and 10K real images, compute nearest-neighbor distances in inception feature space — if many real images have no close generated neighbor, coverage is low. (3) Truncation trick: sample z values with varying standard deviations; if all produce visually similar faces, mode collapse has occurred. Overfitting diagnosis: FID on validation set diverges from training set FID. Training instability: loss curves show high oscillation, generated image quality varies dramatically between epochs. Key distinction: mode collapse produces stable but low-diversity outputs; instability produces erratic quality; overfitting produces memorization of training faces.

**Probe 2:** "Minibatch discrimination is one fix for mode collapse. Explain the mechanism: what does the discriminator see, and why does seeing batch-level statistics help the generator diversify?"
> *Looking for:* Standard discriminator: sees one image at a time, makes a real/fake decision. Problem: the generator can fool the discriminator by producing one convincing mode — the discriminator never knows if all generated images look the same. Minibatch discrimination: each image in the discriminator's input is augmented with features derived from the entire batch (e.g., distances to other samples in a learned feature space). The discriminator can now detect if all images in the batch look similar (batch-level diversity signal). Generator must now produce diverse outputs to fool a discriminator that notices low diversity. Implementation: a learned 'diversity layer' computes pairwise distances across the batch and appends them to each image's feature vector before the final classification. Training batch size must be large enough for the diversity signal to be meaningful.

**Expert probe:** "StyleGAN uses progressive growing during training. Compare the gradient flow in a progressively grown GAN vs. training on full resolution from scratch, and explain why the progressive approach stabilizes training from first principles."
> *Looking for:* Training from scratch at 1024×1024: at initialization, the generator produces random noise at high resolution. The discriminator easily distinguishes random noise from real faces — gradients are saturated (near-zero for generator). The model must simultaneously learn all scales of image structure (global pose, coarse features, fine texture). Progressive growing: starts at 4×4 resolution. At this scale, the generator only needs to learn coarse blob shapes. The discriminator's task is simpler — global structure only. As training stabilizes at low resolution, new layers are faded in gradually (via alpha blending). This means each new layer receives gradient signals on a foundation of already-learned representations rather than on random noise. Formally: the loss landscape at low resolution is smoother — the generator and discriminator are in a more benign optimization regime where the initial distribution mismatch is smaller. Strong hire also knows StyleGAN3's fix for progressive growing's alias artifacts: alias-free generator with continuous equivariance.

---

### Area 3: FID Score Limitations

**Surface:** "The README says FID is the gold standard. What are its limitations?"

**Probe 1:** "FID uses Inception-v3 features trained on ImageNet. For a face generation task, what specific aspects of face quality does FID fail to capture that a human evaluator would notice immediately?"
> *Looking for:* FID limitations for faces: (1) No identity consistency: FID doesn't penalize faces with inconsistent features (one eye larger than the other, asymmetric features). (2) No semantic accuracy: FID doesn't detect age/gender inconsistencies or attributes. (3) Inception's features are not face-specialized: the features used to compute FID were trained for 1000 ImageNet classes, not for face identity, expression, or attribute quality. (4) Distribution-level only: FID compares distributions, so a model that generates 50K perfect but identical faces might have a low FID if those faces are in the training distribution. Better face-specific metrics: (a) ID (identity preservation) using ArcFace — are generated faces consistent identities? (b) FID computed in a face-recognition feature space rather than Inception space. (c) FRS (Face Recognition Score) for attribute-conditioned generation.

**Probe 2:** "FID requires at least 50,000 generated images for reliable estimation. A team generates only 5,000 images for FID computation during development iterations. What is the mathematical problem with small-sample FID estimation?"
> *Looking for:* FID estimates the Frechet distance between two Gaussian distributions fit to the feature vectors. Estimating a Gaussian requires estimating the mean vector (d-dimensional, d=2048 for Inception) and covariance matrix (d×d = 4M entries for d=2048). With n=5,000 samples in d=2048 dimensions, n << d² — the covariance estimate is severely ill-conditioned (rank-deficient). The computed FID will have high variance and will typically underestimate the true FID. Kernel Inception Distance (KID) is an alternative that uses a polynomial kernel estimator and is unbiased even with small samples — recommended for small-sample evaluation. Strong hire knows n > d is not sufficient; you need n >> d for reliable covariance estimation, which requires tens of thousands of samples for d=2048.

**Expert probe:** "FID measures the distance between the real and generated feature distributions. But consider two generators: Generator A produces highly diverse faces with occasional artifacts; Generator B produces less diverse but artifact-free faces. Both could have the same FID. How would you design a composite evaluation framework for face generation that distinguishes precision (quality) from recall (diversity), and what existing metrics implement this?"
> *Looking for:* Precision-recall framework: Precision = fraction of generated images that fall within the support of the real distribution (quality/realism). Recall = fraction of real images that have a close neighbor in the generated set (diversity/coverage). Improved Precision and Recall (Kynkäänniemi et al.) estimates both by computing k-NN neighborhoods in feature space. Generator A: high recall (covers diverse modes), potentially lower precision (artifacts). Generator B: high precision, lower recall (covers fewer modes). Combined: use both precision and recall as separate metrics alongside FID. Additional for faces: Perceptual Path Length (PPL) in the StyleGAN latent space measures smoothness of the generator's mapping — abrupt changes correlate with artifacts. Attribute accuracy (age, gender, expression) evaluated by a separate classifier measures semantic controllability.

---

### Area 4: GANs vs. Diffusion — 2025 Perspective

**Surface:** "Why would you still choose a GAN over diffusion for face generation?"

**Probe 1:** "The README says GANs are better for real-time applications because of single-pass sampling. The game studio needs to generate 1,000 unique NPC faces during game loading (5 second budget). GAN: 1 forward pass ≈ 20ms each. Diffusion: 50 DDIM steps × 30ms each = 1,500ms. Is GAN the right choice? What's the calculation you'd do before answering?"
> *Looking for:* GAN: 1,000 faces × 20ms = 20 seconds (serial) or ~200ms with batch size 100. Diffusion: 1,000 faces / batch, 50 steps × 30ms = 1,500ms — but with a batch size of 100, that's 10 batches × 1,500ms = 15 seconds. Comparable! Key insight: diffusion's per-step cost amortizes across batch size — a batch of 100 faces takes the same time as 1 face. The question is whether batch processing during load time is acceptable. GAN advantage: real-time per-face generation (20ms), ideal if faces must be generated interactively. Diffusion advantage: higher quality and diversity, practical if all 1,000 faces can be batch-generated. Strong candidate recommends SDXL-Turbo (4-step distilled diffusion) or LCM which achieves diffusion quality in 4 steps (~120ms) as a middle ground.

**Probe 2:** "A diffusion model team claims they've achieved GAN-level face quality with FID < 5 using 1,000 timesteps. Their solution for real-time needs is consistency distillation (1-4 step generation). Compare the training complexity and quality tradeoffs of GAN training vs. consistency distillation from a diffusion teacher."
> *Looking for:* GAN training: adversarial minimax game — generator and discriminator must stay balanced. Failure modes: mode collapse, training instability, discriminator overfitting. Quality ceiling: determined by the discriminator's ability to detect fine-grained differences. Consistency distillation: train a student model to map any point on the diffusion trajectory directly to the clean image, using the diffusion model as teacher. Benefit: inherits the diffusion model's quality and diversity (no mode collapse). Training is simpler — supervised distillation with a regression loss. Quality tradeoff: each distillation step introduces approximation error; 1-step is lower quality than 4-step. Strong candidate notes that consistency models trained from scratch (vs. distillation) are harder to train but don't require a pretrained diffusion teacher. Recommendation: for a new system, consistency distillation from a strong diffusion backbone gives better quality than a new GAN without the training instability.

**Expert probe:** "StyleGAN3 introduced alias-free generation with equivariance properties. From first principles, explain what equivariance means for a face generator, why aliasing in StyleGAN2 breaks equivariance, and what the practical consequence was for applications like face morphing or animation."
> *Looking for:* Equivariance: if you transform the input (e.g., shift the latent code in a direction corresponding to head rotation), the output should transform predictably (the face rotates continuously). StyleGAN2 aliasing: traditional convolutional networks downsample and upsample using discrete operations that introduce high-frequency aliasing artifacts. When you walk along a smooth path in latent space, the image texture 'sticks' to the image grid rather than following the underlying 3D structure — textures appear to crawl along the face as the head rotates. Practical consequence: face animation by walking in latent space produces texture-sticking artifacts that break temporal consistency. For video or morphing applications, this creates unnatural 'texture swimming.' StyleGAN3 uses continuous interpolation with sinc-based filtering to prevent aliasing — textures are equivariantly attached to the underlying surface, not the image plane. Strong hire can draw the distinction between 'image coordinates' (breaks equivariance) and 'object surface' (equivariance-preserving).

---

## Red Flags to Watch For

- **FID is sufficient for face generation evaluation.** Cannot identify precision vs. recall split or face-specific metrics.
- **WGAN = WGAN-GP.** Cannot explain why weight clipping is bad and how spectral normalization or gradient penalty improves on it.
- **Progressive growing is just a training trick.** Cannot explain the gradient flow argument from first principles.
- **"Just use diffusion for everything."** Cannot do the latency calculation or identify when GANs remain competitive.
- **Minibatch discrimination = data augmentation.** Doesn't understand that it operates on batch-level statistics, not individual samples.
- **Mode collapse is a hyperparameter problem.** Cannot explain the Nash equilibrium nature of mode collapse.

---

## Hiring Criteria

| Tier | Criteria |
|------|----------|
| **No Hire** | Cannot explain Wasserstein distance vs. JS divergence at training. Identifies mode collapse only as "the model repeats outputs." Cannot state when GANs vs. diffusion is the right choice. FID is sufficient and has no failure modes. |
| **Weak Hire** | Correctly explains GAN minimax game and WGAN's benefit. Can identify mode collapse symptoms. Knows FID limitations conceptually but cannot specify face-specific metrics. Cannot handle the mathematical argument for progressive growing or small-sample FID bias. |
| **Hire** | Explains Lipschitz constraint and spectral normalization mechanistically. Can diagnose mode collapse vs. overfitting vs. instability with specific diagnostic approaches. Identifies FID limitations for faces and proposes alternatives. Does the GAN vs. diffusion latency calculation. |
| **Strong Hire** | Derives the Nash equilibrium nature of mode collapse. Explains alias-free generation (equivariance) and its practical consequences. Knows the precision-recall decomposition and existing metrics (Improved P&R). Proactively mentions SDXL-Turbo or consistency distillation as alternatives before being asked about diffusion vs. GAN. Has a specific, justified recommendation for the gaming use case. |
