# Personalized Headshot Generation — Staff/Principal Interview Guide

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

> "Design a personalized headshot generation system — a user uploads 10–20 photos of themselves, and the system generates professional-quality headshots in different styles (formal, casual, artistic). Think of products like Photoroom or Aragon.ai. Walk me through your approach."

### Signal Being Tested

Does the candidate recognize the per-user fine-tuning challenge, the identity preservation requirement, and the trade-off among fine-tuning methods (DreamBooth vs. LoRA vs. Textual Inversion)?

### Six Clarification Dimensions

| Dimension | Why It Matters |
|---|---|
| **Number of input photos** | 3–5 (minimal) vs. 10–20 (rich) — determines fine-tuning quality |
| **Style variety** | Professional headshots only vs. artistic styles — determines output diversity requirement |
| **Identity preservation strictness** | "Looks like me" vs. "Is recognizably me" — different quality thresholds |
| **Fine-tuning latency budget** | 10 minutes vs. 1 hour — constrains fine-tuning method choice |
| **Privacy requirements** | User photos are highly sensitive; data retention and deletion policies matter |
| **Batch generation** | How many headshots per session? (affects per-user serving cost) |

### Follow-up Probes

- "If a user provides only 3 photos vs. 20 photos, how does your system's output quality change?"
- "What is the trade-off between DreamBooth (full fine-tuning) and LoRA (low-rank fine-tuning)?"
- "How do you delete all traces of a user's identity from the system when they request data deletion?"

---

### Model Answers — Section 1

**No Hire:**
"I would use a text-to-image model with a description of the person." No understanding that the system must capture a specific identity not describable by text.

**Lean No Hire:**
Knows fine-tuning is needed but cannot distinguish DreamBooth from LoRA or Textual Inversion at a conceptual level.

**Lean Hire:**
Identifies the three main personalization approaches (DreamBooth, LoRA, Textual Inversion). Can explain the trade-off: DreamBooth is highest quality but most expensive; LoRA is efficient but may have slightly lower identity fidelity; Textual Inversion only optimizes the text embedding.

**Strong Hire Answer (first-person):**

Personalized headshot generation is fundamentally different from standard text-to-image generation because we need to capture a specific identity — an individual person who is not in the model's training data. A text description like "a 35-year-old woman with brown hair and green eyes" cannot capture unique facial features; we need to encode the person's identity into the model directly.

This requires per-user fine-tuning, and the three main approaches have very different cost-quality trade-offs.

First, I want to clarify the fine-tuning budget. DreamBooth fine-tunes all model weights (or a large subset) on the user's ~20 photos — this takes 10–60 minutes on a GPU and costs ~$0.50–2 per user. LoRA fine-tunes small rank-decomposition matrices (~1% of parameters) — this takes 5–15 minutes at ~$0.10 per user. Textual Inversion only optimizes a text embedding vector — this is cheapest but quality is lowest for complex identity features.

Second, I want to clarify the identity preservation requirement. The strictest requirement is that a stranger viewing the generated headshot would confidently say it depicts the same person as the input photos. This requires DreamBooth or a high-rank LoRA. A weaker requirement ("looks generally like me, roughly") can be met with Textual Inversion or low-rank LoRA.

Third, privacy is critical. User photos are highly sensitive biometric data. I need to understand data retention policy: are user photos stored on our servers? For how long? What happens to fine-tuned LoRA weights when the user requests deletion? Under GDPR, all user-specific model artifacts must be deletable on request.

Fourth, the fine-tuning throughput requirement. At scale (1M users/day), even 10 minutes of GPU time per user requires 10M GPU-minutes/day ≈ 6,900 A100-hours/day ≈ 300 A100 GPUs running continuously. This is significant infrastructure.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "How do you formally frame personalized headshot generation as an ML problem? What are you optimizing for?"

### Signal Being Tested

Does the candidate understand the per-user adaptation problem, the prior preservation objective, and the tension between identity fidelity and style control?

### Follow-up Probes

- "What is the prior preservation loss in DreamBooth and why is it needed?"
- "What does LoRA's weight decomposition look like mathematically?"
- "How does the identity-style trade-off manifest during inference?"

---

### Model Answers — Section 2

**No Hire:**
"Fine-tune the model on the user's photos." Cannot describe the prior preservation loss or why naive fine-tuning fails.

**Lean No Hire:**
Knows DreamBooth fine-tunes the model but cannot explain the prior preservation loss or describe LoRA's mathematical formulation.

**Lean Hire:**
Correctly describes DreamBooth's two-loss objective (reconstruction + prior preservation). Can describe LoRA's W = W_0 + BA formulation. Explains that textual inversion only optimizes an embedding vector.

**Strong Hire Answer (first-person):**

The personalized headshot problem has two competing objectives: (1) learn the user's specific identity so the generated headshots look like them, and (2) preserve the base model's generative capability so headshots can be rendered in diverse styles, backgrounds, and poses.

**DreamBooth formulation:**

We associate the user with a unique identifier token `[V]` (a rare token like "sks person" that the base model has no prior associations with). The model is fine-tuned to associate `[V]` with the user's appearance.

The training loss has two components:

1. *Instance loss* (teach the model the user's appearance):
```
L_instance = E_{t,ε} [||ε - ε_θ(x_t^u, t, c([V] person))||²]
```
where x^u are the user's photos. This minimizes diffusion MSE on the user's images.

2. *Prior preservation loss* (prevent forgetting the class prior):
```
L_prior = E_{t,ε} [||ε - ε_θ(x_t^c, t, c(person))||²]
```
where x^c are class images (generated by the base model without fine-tuning). This ensures the model doesn't forget how to generate diverse non-user faces.

Full DreamBooth loss:
```
L = L_instance + λ · L_prior,  λ ≈ 1.0
```

Without L_prior, the model undergoes language drift — the token "person" becomes synonymous with the specific user, so generating "a person in a park" produces the user's face instead of a diverse person. L_prior prevents this by maintaining the model's prior over the class.

**LoRA (Low-Rank Adaptation):**

LoRA avoids full fine-tuning by decomposing weight updates into low-rank matrices:
```
W_fine-tuned = W_0 + BA
B ∈ R^{d × r}, A ∈ R^{r × k}, rank r << min(d, k)
```

For d=4096, k=4096, r=16: W_0 has 16.7M parameters; BA has 131K parameters — 128× fewer. Only B and A are trained; W_0 is frozen. The adapter is stored separately (~50MB per user at r=16) and loaded dynamically at serving time.

DreamBooth + LoRA combines both: apply LoRA adapters while using DreamBooth's prior preservation loss. This achieves near-DreamBooth quality at LoRA compute cost.

**Textual Inversion:**

Only optimizes a new text embedding vector v* for the token `[V]`:
```
v* = argmin_v E_{t,x,ε} [||ε - ε_θ(x_t, t, f(v))||²]
```
The model weights are frozen; only v* is learned. This is the cheapest approach but captures only low-dimensional identity information — it works for textures and styles but may not capture fine-grained facial identity.

---

## Section 3: Data & Preprocessing (8 min)

### Interviewer Prompt

> "What are the requirements for the user's input photos, and how do you preprocess them for fine-tuning?"

### Signal Being Tested

Does the candidate understand input photo quality requirements, automatic face quality assessment, and the preprocessing pipeline for fine-tuning?

### Follow-up Probes

- "What input photo quality issues cause fine-tuning to produce poor results?"
- "How do you ensure diversity in the user's input photos?"
- "What data augmentation is appropriate for fine-tuning on 10–20 images?"

---

### Model Answers — Section 3

**No Hire:**
"I would use any photos the user provides." No understanding of quality requirements.

**Lean No Hire:**
Knows photos should be high quality but cannot describe quality checks or diversity requirements.

**Lean Hire:**
Describes face detection, alignment, quality filtering, and diversity requirements for input photos. Explains why similar-pose photos reduce fine-tuning quality.

**Strong Hire Answer (first-person):**

Fine-tuning on 10–20 user photos sounds like a small dataset, but quality and diversity of those photos are far more important than quantity. A set of 20 identical-angle, identical-lighting selfies produces worse results than 10 photos with varied poses, lighting, and backgrounds.

**Input photo quality requirements:**

1. *Face detection*: verify exactly one face is prominently visible per image. Reject group photos (multiple faces would confuse identity) and photos where the face is occluded, cropped, or too small (< 150×150 pixels).

2. *Face quality assessment*: use a face quality model (e.g., SER-FIQ) to score each photo on:
   - Sharpness (Laplacian variance > 100)
   - Exposure (histogram not clipped at extremes)
   - Face angle (reject > 30° yaw/pitch — profile photos confuse identity learning)
   - Resolution (face bounding box > 150×150 pixels)

3. *Diversity requirement*: detect if all photos are too similar (same pose, same lighting). Use face embedding cosine similarity — if average pairwise similarity between face crops is > 0.95, prompt the user to add more varied photos. Diversity in pose, expression, and background significantly improves fine-tuning output diversity.

**Face alignment preprocessing:**
Align all input faces to a canonical orientation using the same procedure as FFHQ alignment: detect 5 landmarks, compute affine transform, crop to 512×512.

**Data augmentation during fine-tuning:**
With only 10–20 images, augmentation is important. Safe augmentations:
- Horizontal flip (faces are approximately symmetric; works for some poses)
- Brightness/contrast jitter ±20% (doesn't change identity)
- Random crop (maintain face crop, slightly varied composition)

Do NOT apply: extreme color jitter (may confuse skin tone learning), rotation beyond ±15° (distorts facial geometry), or cutout (may corrupt facial features).

**Privacy preprocessing:**
Before storing user photos on the server, extract and store only the cropped, aligned face regions (not the full image). Background information is not needed for fine-tuning and reduces the privacy surface. Store with encryption at rest and in transit.

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "Compare DreamBooth, LoRA, and Textual Inversion in detail. For each, explain what parameters are optimized, the compute and memory cost, and the quality trade-offs."

### Signal Being Tested

Does the candidate understand the parameter count, compute cost, and quality for each fine-tuning method? Can they make a principled recommendation based on production constraints?

### Follow-up Probes

- "DreamBooth fine-tunes the full U-Net on 10–20 images. What is the risk of overfitting?"
- "At LoRA rank r=4 vs. r=64, what changes? How does rank affect identity fidelity?"
- "What is the DreamBooth-LoRA combination and why is it often the production choice?"

---

### Model Answers — Section 4

**No Hire:**
"Fine-tuning updates the model weights." Cannot distinguish the three methods or explain the LoRA decomposition.

**Lean No Hire:**
Knows LoRA uses low-rank matrices but cannot quantify compute savings or explain the rank-quality trade-off.

**Lean Hire:**
Correctly describes all three methods and their trade-offs. Can explain W = W_0 + BA and why prior preservation loss is needed in DreamBooth.

**Strong Hire Answer (first-person):**

The three personalization methods sit at very different points on the compute-quality frontier.

**Textual Inversion:**
- *What is optimized*: a single text embedding vector v* ∈ R^{768} for the token `[V]`. Model weights are fully frozen.
- *Parameters*: ~768 parameters
- *Compute*: ~100 gradient steps × very fast (frozen model) ≈ 1–2 minutes
- *Quality*: good for textures and artistic styles; poor for complex facial identity. The 768-dimensional embedding cannot capture the full complexity of a specific person's face.
- *Use case*: style transfer, texture learning

**LoRA Fine-tuning:**
- *What is optimized*: rank-r decomposition matrices for selected weight matrices in the U-Net (typically Q, K, V projection matrices in attention layers)
- *LoRA decomposition*: `W = W_0 + BA`, B ∈ R^{d×r}, A ∈ R^{r×k}, r ∈ {4, 8, 16, 64}
- *Parameters*: for SDXL U-Net with r=16: ~12M parameters (vs. 2.6B base — 0.5% of model)
- *Compute*: 200–500 gradient steps × forward+backward through adapted layers ≈ 5–15 minutes on A100
- *Quality*: rank r=4: captures rough identity (skin tone, hair color, rough face shape). r=16: good identity fidelity for most users. r=64: near-DreamBooth quality, more compute
- *Overfitting*: lower risk than DreamBooth (fewer parameters); regularization via prior preservation still recommended

**DreamBooth (full fine-tuning):**
- *What is optimized*: all U-Net parameters (2.6B for SDXL) or optionally + text encoder
- *Parameters*: 2.6B
- *Compute*: 800–1200 gradient steps ≈ 20–60 minutes on A100, depending on resolution
- *Quality*: highest identity fidelity; generates clear recognizable likeness across diverse styles and poses
- *Overfitting risk*: high — 1200 gradient steps on 15 images can overfit. The prior preservation loss is essential: without it, the model collapses the token `[V]` to a single face image regardless of prompt. With it, the model generalizes.
- *Storage*: full model per user (~5GB) — expensive at scale

**Production recommendation: DreamBooth + LoRA**

The production choice is typically DreamBooth-style fine-tuning *applied only to LoRA adapter weights* with prior preservation loss:

```
Train only B, A matrices (LoRA) using DreamBooth's dual loss:
L = L_instance + λ·L_prior
```

This achieves: DreamBooth-quality identity learning + LoRA's efficient storage (~50MB per user at r=16) + LoRA's fast fine-tuning (5–15 minutes). The prior preservation loss is equally important in the LoRA context — without it, the LoRA adapter learns to associate all generation with the user's identity.

**The rank-quality trade-off:**
At rank r=4: the adapter captures ~2% of the identity signal of a full fine-tune. The faces look generally similar (correct skin tone, hair type) but fail on fine-grained identity (distinguishing siblings). At r=16: captures ~70% of identity signal for most users. At r=64: ~90%. For production where identity fidelity is the core product value, r=16–32 is the sweet spot.

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How do you evaluate the quality of personalized headshots? What metrics measure identity preservation separately from aesthetic quality?"

### Signal Being Tested

Does the candidate know face recognition-based identity similarity metrics (ArcFace, FaceNet) and understand that identity fidelity and aesthetic quality must be measured separately?

### Follow-up Probes

- "What is ArcFace and how do you use it to measure identity preservation?"
- "How do you measure whether the style prompt was followed?"
- "What is the evaluation ground truth? We don't have reference 'correct' headshots."

---

### Model Answers — Section 5

**No Hire:**
"Ask users if they look like themselves." Cannot describe automated identity evaluation.

**Lean No Hire:**
Mentions FID for image quality but cannot describe identity-specific evaluation.

**Lean Hire:**
Correctly describes ArcFace cosine similarity as the identity preservation metric. Distinguishes identity score from aesthetic quality (FID, CLIP Score for style adherence).

**Strong Hire Answer (first-person):**

Personalized headshot evaluation requires measuring two orthogonal dimensions: does the person look like the user? (identity preservation) and does the headshot look professional/aesthetic? (image quality + style adherence).

**Identity Preservation (ArcFace Similarity):**

Use a pretrained face recognition model (ArcFace, FaceNet, or similar) to extract identity embeddings from the user's input photos and from the generated headshots. Measure cosine similarity:
```
identity_score = cos(ArcFace(generated), mean(ArcFace(input_photos)))
```

Threshold for "passes as the same person": identity_score > 0.5 (ArcFace cosine similarity scale). Above 0.7 is strong identity preservation; below 0.3 is a different-looking person.

The benchmark: compute identity_score on a held-out set of users (100 users, 20 input photos each, generate 10 headshots each). Target: median identity_score > 0.6, < 10% of generations below 0.3.

**Image Quality (Aesthetic Score + FID):**

*Aesthetic score*: run a CLIP-based aesthetic scorer (LAION-Aesthetics) on generated headshots. Professional headshots should score > 6/10.

*FID*: compare distribution of generated headshots to a reference set of professional LinkedIn/corporate headshots. Lower FID = more realistic professional appearance.

**Style Adherence (CLIP Score):**

For each generated image, compute CLIP Score between the image and the style prompt ("professional corporate headshot in formal attire"). This measures whether the style instruction was followed.

**Human evaluation (ground truth):**
Show human raters pairs of (input photo, generated headshot) and ask: "On a scale of 1-5, how closely does this headshot resemble the person in the input photo?" Collect 3 ratings per pair. This is the ultimate ground truth — automate it using a fine-tuned VQA model trained on human identity ratings.

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "Walk me through the serving architecture for a personalized headshot service. What are the key infrastructure decisions around fine-tuning pipeline and per-user model serving?"

### Signal Being Tested

Does the candidate understand the fine-tuning pipeline (async, queue-based), per-user adapter storage, and the challenge of serving millions of users with per-user LoRA adapters?

### Follow-up Probes

- "How do you serve millions of users when each has a unique LoRA adapter?"
- "What is the fine-tuning job queue and how do you prioritize jobs?"
- "How do you handle a user who wants more headshots 3 months after initial fine-tuning — do you re-fine-tune?"

---

### Model Answers — Section 6

**No Hire:**
"I would fine-tune a model for each user when they request it." No understanding of async pipeline or storage.

**Lean No Hire:**
Knows fine-tuning is async but cannot describe the job queue, adapter storage, or efficient LoRA serving.

**Lean Hire:**
Correctly describes async fine-tuning pipeline with job queue, per-user adapter storage (~50MB at r=16), and dynamic adapter loading for inference.

**Strong Hire Answer (first-person):**

Personalized headshot serving has two distinct phases: fine-tuning (per-user, one-time) and inference (per-generation, per-user).

**Fine-tuning pipeline (async):**
1. User uploads 10–20 photos
2. Photo validation and preprocessing (face detection, quality check, alignment) — ~30 seconds
3. Job submitted to fine-tuning queue (SQS, Redis Queue, or similar)
4. GPU worker picks up job, runs DreamBooth+LoRA training — 10–20 minutes
5. Trained LoRA adapter (~50MB) saved to object storage (S3), indexed by user_id
6. User notified (email/push notification): "Your headshots are ready"
7. Initial batch of 20–50 headshots generated and delivered

**LoRA adapter serving:**
Each user has a unique LoRA adapter. The base model (SDXL, ~7GB) is loaded once per GPU worker. Adapters are loaded on top of the base model:
```
W_effective = W_base + B_user · A_user
```
Loading an r=16 LoRA adapter (~50MB) takes ~200ms — acceptable for per-user cold starts.

For a high-traffic user, cache the loaded base+adapter combination in GPU memory (LRU cache across users). The cache can hold:
```
A100 (80GB) - base model (7GB) = 73GB for adapters
73GB / 50MB per adapter = ~1460 adapters simultaneously cached
```
For a user with >1000 requests/day, their adapter stays warm. For long-tail users, the adapter is loaded from S3 on demand.

**Re-fine-tuning policy:**
If the user uploads new photos (3 months later), run fine-tuning again on the combined old + new photo set. If only requesting more headshots without new photos, reuse the existing adapter — no re-fine-tuning needed.

**Cost per user:**
At scale:
- Fine-tuning: 15 min × $2/hour (A100 spot) = $0.50 per user one-time
- Per-generation: ~5s × $0.001/second = $0.005 per headshot
- 20 headshots per user: $0.10 generation cost

Total lifetime cost per user: ~$0.60 at 20 headshots. Product pricing: $12–$25 per session → ~20–40× gross margin on compute.

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the critical failure modes of personalized headshot generation, and how do you detect and mitigate them?"

### Signal Being Tested

Does the candidate identify identity drift (generated face looks different from user), style-identity conflict, misuse (generating faces of others), and privacy failure modes?

### Follow-up Probes

- "What is identity drift and when does it occur?"
- "How do you prevent a user from uploading photos of another person and generating their headshots?"
- "What happens if the fine-tuned adapter 'forgets' how to generate certain styles after overfitting?"

---

### Model Answers — Section 7

**No Hire:**
Cannot describe personalized generation failure modes. Generic "bad quality."

**Lean No Hire:**
Mentions overfitting but cannot describe identity drift or misuse prevention.

**Lean Hire:**
Correctly identifies identity drift, overfitting-induced style loss, and misuse (non-consensual generation of others' likeness). Proposes ArcFace-based quality gating and consent verification.

**Strong Hire Answer (first-person):**

Personalized headshot systems have failure modes at both technical and ethical levels.

**Technical: Identity drift**
Generated faces gradually deviate from the user's actual identity — they look like a plausible face but not the specific user. Common causes: (1) too few diverse input photos (model generalizes from limited poses), (2) high CFG scale during inference (strong style guidance overrides identity), (3) insufficient fine-tuning steps (adapter doesn't capture enough identity signal).

Detection: compute ArcFace similarity between each generated headshot and the user's reference photos. Auto-reject generations with identity_score < 0.4. Trigger re-fine-tuning with additional photos if rejection rate > 30%.

**Technical: Overfitting — style inflexibility**
Overfitting the LoRA adapter causes the model to generate only images that look exactly like the input photos — same pose, same background. Even with a style prompt "professional formal headshot," the output looks like a copy-paste of the input photos with a new background. Detection: measure pose diversity across generated images using face orientation estimation. If all generated poses are within ±10° of the dominant input pose, overfitting is likely.

Mitigation: reduce fine-tuning steps, increase λ (prior preservation loss weight), use lower LoRA rank.

**Ethical: Non-consensual generation of another person's likeness**
A user uploads 10 photos of another person (public figure, ex-partner) and generates headshots of them without consent.

Mitigation: (1) run face recognition on input photos and compare to a database of opted-out individuals (celebrities with explicit opt-out requests) — block if match found; (2) detect if input photos are scraped social media photos (perceptual hash comparison to known social media images); (3) identity consistency check — if the detected face in all 10 input photos belongs to the same identity and that identity is in the opt-out database, reject.

**Ethical: Data deletion compliance**
User requests GDPR deletion of all their data. The system must delete: input photos, aligned face crops, fine-tuned LoRA adapter weights, and all generated headshots. The adapter weights are an ML representation of biometric data and must be deleted within the GDPR 30-day window.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "You've built personalized headshots. Now you want to offer a personalization SDK that enables any app (professional social networks, dating apps, gaming avatar creators) to embed personalized generation. What are the platform decisions?"

### Signal Being Tested

Does the candidate think about identity portability (the user's adapter working across products), privacy architecture at the platform level, and abuse prevention at scale?

### Follow-up Probes

- "Should the user's LoRA adapter be portable across apps, or siloed per app?"
- "How does the consent model work when the user's identity is shared across products?"

---

### Model Answers — Section 8

**No Hire:**
"License the API to partners." No consideration of identity portability, privacy, or abuse.

**Lean No Hire:**
Mentions API licensing but doesn't address identity portability or consent architecture.

**Lean Hire:**
Correctly identifies identity portability as a privacy design choice. Proposes user-controlled adapter storage and consent model.

**Strong Hire Answer (first-person):**

Identity portability is the core platform design question: does a user's LoRA adapter belong to our platform or to the user?

**User-controlled identity architecture:**
I advocate for a user-controlled model where the LoRA adapter is the user's property. The user holds a signed credential (JWT or similar) that includes their adapter location in encrypted storage. Any app that the user grants permission to can request adapter access.

This creates a privacy-preserving identity layer: the adapter is stored in a user-controlled encrypted bucket; each app gets a temporary read token with expiry; the platform never directly shares the adapter — it generates images in a sandboxed environment and returns only the output images to the third-party app.

**Consent architecture:**
Each app accessing the user's identity must present a clear consent screen: "This app will use your personalized AI identity to generate [specific content type]. Your identity data will be used for [duration]. You can revoke access at any time." This is similar to OAuth scopes but for biometric ML adapters.

**Abuse prevention at platform level:**
The platform enforces: (1) output watermarking on all generated images from any partner app; (2) prohibited use cases (no political ads, no non-consensual intimate imagery, no impersonation of public figures in misleading contexts) — these are non-negotiable regardless of partner; (3) identity matching on outputs to verify the generated face matches the authorized user's identity (not someone else's photos used to generate their likeness).

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**DreamBooth instance loss:**
```
L_instance = E_{t,ε} [||ε - ε_θ(x_t^u, t, c([V] person))||²]
```

**DreamBooth prior preservation loss:**
```
L_prior = E_{t,ε} [||ε - ε_θ(x_t^c, t, c(person))||²]
```

**Full DreamBooth loss:**
```
L = L_instance + λ·L_prior,  λ ≈ 1.0
```

**LoRA weight decomposition:**
```
W_fine-tuned = W_0 + BA
B ∈ R^{d×r}, A ∈ R^{r×k}, rank r << min(d,k)
```

**ArcFace identity similarity:**
```
identity_score = cos(ArcFace(generated), mean(ArcFace(input_photos)))
```

**DreamBooth parameter count comparison:**
```
Full fine-tune: ~2.6B params (SDXL U-Net)
LoRA r=16: ~12M params (0.5% of U-Net)
Textual Inversion: ~768 params (1 text embedding)
```

**LoRA adapter storage per user:**
```
Size = 2 × r × (d + k) × num_adapted_layers × bytes_per_param
r=16, d=k=4096, 64 layers, FP16: ~50MB
```

**CFG for identity-style balance:**
```
ε̃ = ε_uncond + γ·(ε_cond - ε_uncond)
Lower γ (5-7): more identity preservation
Higher γ (9-12): stronger style adherence, possible identity drift
```

### Vocabulary Cheat Sheet

| Term | Definition |
|---|---|
| **DreamBooth** | Fine-tunes full model (or LoRA) with prior preservation loss on user images |
| **LoRA** | Low-Rank Adaptation; efficient fine-tuning via B·A weight decomposition |
| **Textual Inversion** | Optimizes only a new text embedding vector; model weights frozen |
| **Prior preservation loss** | Regularizes DreamBooth to prevent language drift and forgetting |
| **Language drift** | [V] becomes synonymous with the user's face, destroying generality |
| **[V] / sks** | Rare token used as the identity placeholder in DreamBooth |
| **ArcFace** | Face recognition model; identity embeddings used for similarity scoring |
| **FaceNet** | Alternative face recognition model for identity similarity |
| **Identity score** | ArcFace cosine similarity between generated face and input reference |
| **Identity drift** | Generated faces gradually deviate from user's actual appearance |
| **Overfitting** | Fine-tuned model copies input photos without generalization |
| **LoRA rank r** | Controls number of parameters: higher r = more capacity = better quality |
| **Adapter hot-loading** | Dynamically load user's LoRA adapter onto frozen base model |
| **GDPR deletion** | Must delete input photos, face crops, adapters, and generated images on request |

### Key Numbers Table

| Metric | Value |
|---|---|
| Recommended input photos | 10–20 (with pose/expression variety) |
| DreamBooth fine-tuning (A100) | 20–60 min (full), 5–15 min (LoRA) |
| DreamBooth fine-tuning cost (A100 spot) | ~$0.50–2.00 per user |
| LoRA adapter storage (r=16, SDXL) | ~50 MB |
| LoRA parameters (r=16, SDXL) | ~12M (0.5% of U-Net) |
| Textual Inversion parameters | ~768 |
| Identity score (same person) | > 0.5 (ArcFace cosine) |
| Identity score (strong match) | > 0.7 |
| Identity score (different person) | < 0.3 |
| A100 LRU adapter cache capacity | ~1460 adapters (80GB) |
| Inference per-headshot latency (20 steps) | ~5s at 1024×1024 |
| Typical headshots per session | 20–50 |
| Per-headshot compute cost | ~$0.005 |

### Rapid-Fire Day-Before Review

1. **Three personalization methods in order of compute?** Textual Inversion < LoRA < DreamBooth
2. **LoRA formula?** `W = W_0 + BA` where B ∈ R^{d×r}, A ∈ R^{r×k}, rank r << min(d,k)
3. **DreamBooth prior preservation loss purpose?** Prevent language drift — without it, [V] becomes synonymous with the user's identity, destroying model generality
4. **What is language drift?** Fine-tuning causes `person` → user's face; prior loss prevents this
5. **Identity evaluation metric?** ArcFace cosine similarity between generated and reference face embeddings
6. **Adapter cache capacity (A100)?** ~80GB - 7GB base model = 73GB for adapters; at 50MB each = ~1460 users simultaneously
7. **LoRA rank r trade-off?** Higher r = more parameters = better identity fidelity = more compute
8. **Misuse prevention?** Face recognition on input photos vs. opt-out database; identity consistency check across batch
9. **GDPR deletion scope?** Delete input photos, face crops, LoRA adapter, and all generated images
10. **Production fine-tuning choice?** DreamBooth + LoRA: DreamBooth's dual loss, LoRA's efficient storage (~50MB vs. 5GB)
