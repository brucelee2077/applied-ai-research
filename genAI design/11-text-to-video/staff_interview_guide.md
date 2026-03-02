# Text-to-Video Generation — Staff/Principal Interview Guide

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

> "Design a text-to-video generation system — a user provides a text prompt and the system generates a short video clip (3–15 seconds) matching the description. Think of systems like Sora, Runway Gen-2, or Google Veo. Walk me through your approach."

### Signal Being Tested

Does the candidate recognize that text-to-video is dramatically more complex than text-to-image, and that temporal consistency is the core challenge that shapes every design decision?

### Six Clarification Dimensions

| Dimension | Why It Matters |
|---|---|
| **Video length and resolution** | 3s at 512×512 vs. 15s at 1080p — orders of magnitude difference in compute |
| **Frame rate** | 8fps (cinematic) vs. 24fps (smooth) vs. 30fps (realistic) |
| **Temporal consistency requirement** | Physics-compliant motion vs. stylized/artistic motion |
| **Latency** | 30 seconds per clip (acceptable for generation) vs. real-time preview |
| **Conditional inputs** | Text-only vs. text + reference image/video |
| **Output format** | MP4, GIF, streaming frames |

### Follow-up Probes

- "What makes temporal consistency hard, and how does it differ from generating N independent images?"
- "Sora generates 1-minute videos at 1080p. What is the approximate compute cost relative to text-to-image?"
- "How does your design change if users can provide a reference image to guide the video generation?"

---

### Model Answers — Section 1

**No Hire:**
"I would generate each frame with a text-to-image model." No recognition of temporal consistency as a core challenge.

**Lean No Hire:**
Recognizes temporal consistency as a challenge but cannot articulate why generating N independent images fails or describe any temporal modeling approach.

**Lean Hire:**
Correctly identifies temporal consistency as the core challenge. Asks about frame rate, resolution, and length. Notes that video generation is much more computationally expensive than image generation. Can describe at a high level that video diffusion models add temporal dimensions.

**Strong Hire Answer (first-person):**

Text-to-video is the most computationally demanding generative AI task currently deployed in production. Understanding the scope before designing is critical.

First, the scale. A 5-second video at 24fps and 512×512 resolution is 120 frames × 512×512×3 pixels = ~94M pixels — about 30× more data than a single 512×512 image. At 1080p, it's 120× more data per second of video than a comparable still image. This means the compute cost scales roughly linearly with video length and quadratically with resolution.

Second, the core challenge: temporal consistency. If I generate 120 frames independently with a text-to-image model, each frame will be individually photorealistic but will show a dog that teleports, a fire that reverses, and a face that morphs between different people on adjacent frames. Temporal consistency requires: (1) the same identity persisting across frames, (2) physically plausible motion (objects follow trajectories, lighting is consistent), (3) no "flickering" — subtle color/texture changes frame-to-frame that produce an unpleasant shimmering effect.

Third, the resolution-length-quality triangle. You cannot simultaneously have: long duration, high resolution, and short generation time. Sora achieves 1-minute 1080p videos at the cost of minutes-to-hours of generation time per clip. Consumer applications (Runway Gen-2) target 4-second clips at 512×512 in ~30 seconds.

For this design, I'll assume: consumer-grade product, 4–8 second clips, 512×512–720p resolution, < 60 seconds generation time, text + optional reference image conditioning.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "How do you formally frame text-to-video generation as an ML problem? How does the diffusion framework extend from images to video?"

### Signal Being Tested

Does the candidate understand how latent video diffusion works, the role of temporal attention, and the training objective for video generation?

### Follow-up Probes

- "Write out the diffusion training loss for video. How does it differ from image diffusion?"
- "What is temporal attention and why is it needed in addition to spatial attention?"
- "What is a spacetime patch embedding and how does it work in a Video DiT?"

---

### Model Answers — Section 2

**No Hire:**
"Train the model on videos." Cannot formalize the video diffusion objective or describe temporal modeling.

**Lean No Hire:**
Knows that video diffusion extends image diffusion but cannot explain how temporal attention works or the mathematical form of the video diffusion loss.

**Lean Hire:**
Correctly extends image diffusion to video by adding a temporal dimension. Explains temporal attention as attending across frames at each spatial location. Describes Video DiT's spacetime patches.

**Strong Hire Answer (first-person):**

Video generation via latent diffusion extends the image formulation by treating a video V ∈ R^{T×H×W×3} as a 4D tensor rather than a 3D image tensor.

**Video diffusion training objective:**

The video latent z ∈ R^{T×h×w×C} is obtained by encoding each frame independently with the VAE encoder (or with a 3D VAE that encodes spatiotemporal patches). The diffusion model ε_θ is trained on noisy video latents:

```
L = E_{t,z_0,ε,c} [||ε - ε_θ(z_t, t, c)||²]
```

where z_t is the noisy video latent at diffusion timestep t, c is the text conditioning (CLIP/T5 embedding), and ε is the noise. This is identical to image diffusion except z is now a video latent — the model must predict spatially AND temporally coherent noise.

**Temporal Attention:**

Standard spatial attention processes each frame independently. Temporal attention attends across frames at each spatial location, enabling the model to track motion, maintain identity, and ensure consistency:

```
SpatialAttn: for each frame t, attend over all spatial positions within frame t
TemporalAttn: for each spatial position (i,j), attend over all frames t=1..T
```

In practice, video diffusion models interleave spatial and temporal attention blocks:
```
For each layer:
  x = SpatialSelfAttn(x)  # within each frame
  x = TemporalSelfAttn(x)  # across frames at each position
  x = CrossAttn(x, text_embedding)  # text conditioning
  x = FFN(x)
```

Temporal attention at position (i,j) sees the sequence `[f_1(i,j), f_2(i,j), ..., f_T(i,j)]` — the history of that pixel position across all frames. This allows the model to detect that a dog at position (100, 150) in frame 1 has moved to (105, 148) in frame 2 (consistent motion), rather than appearing at an unrelated position (frame inconsistency).

**Video DiT (Spacetime Patches):**

Video DiT (used in Sora, CogVideoX) extends ViT/DiT to video by using spacetime patches: a 3D patch of size p_t × p_h × p_w pixels covering both spatial and temporal extent. For T=16 frames, H=W=32 (in latent space), with spacetime patches 2×2×2:
```
Num patches = (T/p_t) × (H/p_h) × (W/p_w) = 8 × 16 × 16 = 2048 patches
```
Each patch is flattened and linearly projected to d_model. The transformer processes this sequence of 2048 spacetime tokens with full 3D attention (each token attends to all other spacetime tokens).

Spacetime patches are powerful because 3D patches inherently encode local motion (a patch spanning 2 consecutive frames sees the local optical flow pattern). Full 3D attention (vs. factorized spatial + temporal attention) produces better quality at the cost of O((T×H×W/p³)²) attention complexity.

---

## Section 3: Data & Preprocessing (8 min)

### Interviewer Prompt

> "What training data do you use for text-to-video generation, and how do you preprocess videos for training?"

### Signal Being Tested

Does the candidate know the major video training datasets (HD-VILA, Panda-70M, WebVid), the captioning pipeline, and the specific preprocessing challenges for video (scene cut detection, optical flow quality filtering)?

### Follow-up Probes

- "What is the biggest data quality challenge specific to video vs. image training data?"
- "Why is scene cut detection important in video training data preprocessing?"
- "How do you generate text captions for millions of training videos?"

---

### Model Answers — Section 3

**No Hire:**
"I would download videos from YouTube." No understanding of video-specific data processing.

**Lean No Hire:**
Knows video-text pairs are needed but cannot describe scene cut detection, optical flow quality, or video captioning pipelines.

**Lean Hire:**
Describes scene cut detection, optical flow quality filtering, and automated video captioning using a video-language model. Names major datasets (WebVid, HD-VILA, Panda-70M).

**Strong Hire Answer (first-person):**

Video training data is significantly harder to prepare than image training data because of temporal-specific quality issues.

**Major training datasets:**
- *WebVid-10M*: 10M video-text pairs scraped from stock video websites. High alignment (stock videos have professional descriptions) but limited diversity (heavily commercial/stock photo aesthetic).
- *HD-VILA-100M*: 100M high-definition video clips from YouTube with associated text from video metadata.
- *Panda-70M*: 70M video-caption pairs, re-captioned by video-language models for higher quality captions.
- *InternVid*: 234M video-text pairs, large-scale with quality filtering.

**Video-specific preprocessing pipeline:**

1. *Scene cut detection*: a single YouTube video clip may contain multiple scene changes (different camera angles, jump cuts). Training on multi-scene clips teaches the model that objects can teleport between frames. I use TransNetV2 or PySceneDetect to detect scene boundaries; keep only clips without scene cuts within the training window.

2. *Optical flow quality filtering*: compute optical flow (RAFT or similar) between consecutive frames. Filter out:
   - Static clips (near-zero optical flow): these don't teach the model about motion. The model should learn to generate motion, not static images.
   - Blur/flicker clips (very high variance in optical flow across frames): corrupted video quality.
   - Extremely fast motion (optical flow magnitude > threshold): motion blur degrades training signal.

3. *Video captioning*: stock video metadata captions vary dramatically in quality. For all videos without existing high-quality captions, run a video-language model (Video LLaVA, CogVLM-Video) to generate dense captions describing: the scene, camera motion, visible objects and actions, temporal dynamics. Dense captions significantly improve the model's ability to follow complex motion prompts.

4. *Temporal downsampling*: for consistency, sample all training videos at a fixed frame rate (e.g., 8fps or 16fps). Store as latent codes (VAE-encoded frames) rather than raw pixels to reduce storage and I/O during training.

5. *Resolution bucketing*: group training clips by aspect ratio and resolution. Train with multi-resolution batches to teach the model aspect-ratio awareness. This is critical for avoiding stretching/squishing artifacts at non-standard aspect ratios.

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "Walk me through the Video DiT architecture. How does it handle temporal modeling? How is it different from inflating a 2D image diffusion model?"

### Signal Being Tested

Does the candidate understand the Video DiT architecture with spacetime patches, the alternative of pseudo-3D convolutions (temporal inflation), and why full 3D attention vs. factorized attention is a key architectural trade-off?

### Follow-up Probes

- "What is temporal inflation of a 2D image model? Why is it a good initialization strategy?"
- "What is the trade-off between full 3D attention and factorized spatial + temporal attention?"
- "How does CausalVAE differ from a standard 2D VAE for video compression?"

---

### Model Answers — Section 4

**No Hire:**
"I would use a transformer." Cannot describe temporal modeling or Video DiT.

**Lean No Hire:**
Knows temporal attention is needed but cannot describe spacetime patches, CausalVAE, or temporal inflation strategies.

**Lean Hire:**
Correctly describes Video DiT with spacetime patches and temporal attention. Can explain temporal inflation as a way to initialize from a 2D image model. Distinguishes full 3D vs. factorized attention.

**Strong Hire Answer (first-person):**

Video generation architecture has evolved rapidly, and understanding the trade-offs between approaches is important for designing a production system.

**Temporal Inflation (Baseline Approach):**

The simplest way to add video capability to a pretrained image diffusion model is temporal inflation: extend 2D spatial convolutions and attention to process the temporal dimension. For a 2D U-Net:
- 2D convolution `C_2d: R^{H×W×C} → R^{H×W×C}` is inflated to a pseudo-3D convolution: first apply 2D spatial conv per frame, then apply a 1D temporal conv across frames
- 2D self-attention within each frame is supplemented with a 1D temporal attention across frames at each spatial position

This approach is computationally efficient and initializes temporal weights from pretrained image weights (the identity transformation — temporal conv initialized to only attend to the current frame), allowing fine-tuning from a strong image generation prior.

Limitation: pseudo-3D convolutions separate spatial and temporal processing. They can't model the joint spatiotemporal patterns that require seeing spatial changes across frames simultaneously (e.g., a ball rolling while also changing color).

**Video DiT with Spacetime Patches:**

Video DiT (Sora-style) uses a pure transformer applied to 3D spacetime patches. Let the video latent be z ∈ R^{T×H×W×C}. Divide into 3D patches of size p_t × p_h × p_w:
```
Num patches N = (T/p_t) × (H/p_h) × (W/p_w)
Each patch flattened: p_t × p_h × p_w × C → d_model
```

The resulting sequence of N spacetime tokens is processed by standard transformer blocks. Each token attends to all other tokens via full 3D attention, capturing global spatiotemporal dependencies.

3D positional encoding: extend 2D ViT positional embeddings to 3D. The positional embedding for each token encodes (time, height, width) position. This allows the model to understand that adjacent tokens in time are near in temporal space.

**CausalVAE for video compression:**

Standard 2D VAEs encode each frame independently. CausalVAE extends the VAE to video by using causal temporal convolutions: the latent representation of frame t depends on frames 1..t but not frames t+1..T (causal — no future information). This allows streaming generation (process and decode frames sequentially) and provides better temporal compression than independent frame encoding.

At compression factor f=8 spatially and f_t=4 temporally: a 24-frame, 512×512 video compresses to 6 frames × 64×64 × 16 channels — a 256× reduction in token count vs. pixel-space processing.

**Full 3D vs. factorized attention trade-off:**

Full 3D attention: attention between all N spacetime tokens simultaneously. Complexity: O(N²) where N grows with T×H×W. For N=2048 tokens: 2048² = 4M attention pairs per head per layer. This is expensive but models all spatiotemporal dependencies.

Factorized attention: separately compute spatial attention (within each frame) and temporal attention (across frames at each position). Complexity: O((H×W/p²)²) spatial + O(T²) temporal. Roughly 4–8× cheaper than full 3D, with modest quality degradation.

Production choice: Video DiT (Sora-scale models) uses full 3D attention for quality at the cost of compute. Consumer-speed models (Runway Gen-2, AnimateDiff) use factorized attention for 2–4× speedup.

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How do you evaluate text-to-video generation quality? Walk me through FVD and the temporal consistency metrics."

### Signal Being Tested

Does the candidate know FVD (Fréchet Video Distance) and temporal consistency evaluation methods, and understand how they complement CLIP Score for text-video alignment?

### Follow-up Probes

- "What is FVD and how does it differ from FID for images?"
- "How do you measure temporal consistency? What failure does it catch that FVD misses?"
- "What human evaluation dimensions are specific to video vs. static images?"

---

### Model Answers — Section 5

**No Hire:**
"I would watch the videos and see if they look good." Cannot describe FVD or temporal evaluation.

**Lean No Hire:**
Mentions FVD by name but cannot explain how it's computed or what it measures differently from FID.

**Lean Hire:**
Correctly explains FVD as FID extended to video features (using I3D or similar). Can describe temporal consistency metrics (LPIPS between adjacent frames, optical flow consistency). Knows CLIP Score for text-video alignment.

**Strong Hire Answer (first-person):**

Video evaluation requires measuring three orthogonal dimensions: visual quality, temporal consistency, and text alignment.

**FVD (Fréchet Video Distance):**

FVD extends FID from images to videos by computing the Fréchet distance in the feature space of a pretrained video understanding model (I3D, trained on Kinetics action recognition):

```
FVD = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^{1/2})
```

where μ_r, Σ_r are computed from I3D features of real videos, and μ_g, Σ_g from generated videos. Unlike FID (which uses a single frame's Inception features), FVD's I3D features capture both spatial appearance AND temporal dynamics — a flickering video will produce different I3D features from a temporally smooth video, even if frame-by-frame quality is similar.

Lower FVD = better video quality. Typical FVD for state-of-the-art text-to-video on UCF-101: FVD~150–300. FVD requires 2048+ video samples for reliable estimation.

**Temporal Consistency Metrics:**

*Frame-to-frame LPIPS*: compute LPIPS between consecutive frames. Low average = smooth transitions. But this can be fooled by a model that generates very slow/minimal motion.

*Optical flow consistency*: compute optical flow between frames using a flow estimator (RAFT). Smooth, consistent optical flow indicates physically plausible motion. Jerky, chaotic flow indicates temporal inconsistency.

*Warping error*: warp frame t by the estimated optical flow to predict frame t+1, compare to actual frame t+1. Low warping error = the video's motion is accurately predicted by optical flow = physically consistent.
```
warping_error = LPIPS(warp(f_t, flow_t), f_{t+1})
```

*CLIP feature consistency*: compute CLIP image embeddings for each frame. The standard deviation of these embeddings over time measures appearance drift — how much the visual content changes from frame to frame beyond what motion would predict.

**Text-Video Alignment:**

*CLIP Score (frame-averaged)*: compute CLIP score between text prompt and each frame, then average across frames. This measures whether the video content matches the prompt at each timestep.

*VideoCLIP Score*: run a video-language model (VideoCLIP or similar) that computes alignment between the text and the full video (incorporating temporal context). This is more accurate than frame-averaged CLIP for dynamic prompts ("a person walks toward the camera") where the key content is temporal.

**Human evaluation dimensions (video-specific):**
- *Motion quality*: do objects move naturally with realistic physics?
- *Temporal coherence*: does the scene remain consistent? (same object, same lighting)
- *Prompt following (temporal)*: does the motion described in the prompt (e.g., "camera pans left") occur correctly?
- *Flicker/shimmering artifacts*: is there unpleasant noise between frames?

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "Walk me through the serving infrastructure for a text-to-video system. What makes video serving fundamentally different from image serving?"

### Signal Being Tested

Does the candidate understand the much higher memory and compute requirements for video generation, and the serving optimizations specific to video (latent compression, temporal streaming)?

### Follow-up Probes

- "How much GPU memory does a 5-second 512×512 video generation require?"
- "What is temporal streaming and how does it improve user experience without reducing quality?"
- "How do you serve millions of users when a single video generation takes 60+ seconds?"

---

### Model Answers — Section 6

**No Hire:**
"I would use more GPUs." No understanding of video-specific memory or compute.

**Lean No Hire:**
Notes that video is more expensive than images but cannot quantify or describe serving optimizations.

**Lean Hire:**
Correctly estimates memory requirements and describes async serving (submit job, receive notification). Explains temporal streaming and compressed latent representation.

**Strong Hire Answer (first-person):**

Video generation has a compute profile unlike any other ML serving task — it is the most resource-intensive generative task currently in production.

**Memory requirements for a 5-second 512×512 video at 24fps:**

Video latent (CausalVAE compressed, f=8 spatial, f_t=4 temporal):
- Input: 120 frames × 64×64 × 4 channels (latent) = 120 × 64 × 64 × 4 × 2 bytes ≈ 250 MB
- But with 3D compression, reduce temporal by 4×: 30 latent frames × 64×64 × 4 channels ≈ 60 MB

Video DiT activations during 50-step DDIM: at each step, the full video latent (60 MB) passes through the DiT. With a large Video DiT (~7B parameters, ~14 GB), plus activations (~4× parameter size): ~70 GB total.

Practically: video generation at this scale requires 8× A100 GPUs or 4× H100 GPUs. A single generation is a distributed multi-GPU job, not a single-GPU request.

**Async serving architecture (required for 60+ second generation time):**

Unlike image generation where users wait for ~5s, video generation takes 30–300 seconds. Interactive serving (user waits) is not appropriate.

Queue-based async:
1. User submits prompt → receives job_id and ETA
2. Job enters priority queue → assigned to a GPU cluster node
3. Generation runs (30–300s)
4. Completed video stored in S3 → push notification sent to user
5. User downloads video via pre-signed URL

**Temporal streaming for progressive preview:**

Users dislike waiting for the full video. Generate and stream frames as they are produced:
- Generate keyframes first (frames 1, 6, 12, 18, 24) using fewer diffusion steps
- Stream low-quality preview frames immediately (appear within ~5s)
- Continue generating remaining frames and full-quality video
- Update client with final video when complete

This requires a streaming-capable generation pipeline where intermediate latent codes are decoded to preview frames during the DDIM reverse process.

**Model parallelism for large Video DiT:**

A 7B parameter Video DiT requires distributed inference:
- *Tensor parallel*: shard weight matrices across 4 GPUs (each GPU holds 1/4 of attention heads)
- *Sequence parallel*: for very long spacetime sequences (N=2048+), shard the sequence across GPUs for attention computation
- Communication overhead: ~15% on NVLink between A100/H100 GPUs in the same node

**Cost at scale:**
At $2/hour per A100 and 8 A100s per video job for 2 minutes: $0.53 per 5-second video. At a $1.99 product price point: 3.75× gross compute margin, before infrastructure overhead.

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the most critical failure modes for text-to-video generation, both technical and ethical?"

### Signal Being Tested

Does the candidate identify temporal inconsistency, catastrophic flickering, motion artifacts (impossible physics), and the deepfake/misinformation risk specific to video?

### Follow-up Probes

- "What is temporal flickering and what causes it in video diffusion models?"
- "What happens when the text prompt describes complex multi-step actions?"
- "Why is synthetic video more dangerous for misinformation than synthetic images?"

---

### Model Answers — Section 7

**No Hire:**
Cannot describe video-specific failure modes. Generic "bad quality."

**Lean No Hire:**
Mentions temporal inconsistency but cannot describe its mechanistic causes or mitigations.

**Lean Hire:**
Correctly identifies flickering (temporal inconsistency), impossible motion, and the higher deepfake risk for video vs. images. Can describe detection approaches.

**Strong Hire Answer (first-person):**

Text-to-video failure modes are more severe and harder to detect than image failures.

**Technical: Temporal flickering**
Adjacent frames have inconsistent appearance — texture, lighting, or color shifts rapidly between frames even when no motion is occurring. Mechanistic cause: the model processes frames with some spatial independence; even with temporal attention, noise in the generation process introduces per-frame variability that wasn't averaged out.

Detection: compute CLIP feature variance across frames (should be low for static scene regions). Measure optical flow between consecutive frames and compare to expected flow from motion — inconsistencies indicate flickering.

Mitigation: temporal consistency regularization during training (penalize high-frequency temporal changes in static regions), video-aware diffusion noise scheduling (correlate noise across frames during the forward diffusion process).

**Technical: Impossible motion / physics violation**
Objects defy gravity, pass through each other, or move at physically impossible speeds. The model has learned statistical correlations from training videos but doesn't have an explicit physics model.

Detection: apply a physics consistency checker (keypoint tracking + velocity consistency, depth estimation for spatial occlusion consistency).

Mitigation: training on physics-annotated video data, reinforcement from physics simulators (use a physics engine to score motion plausibility as a training signal).

**Technical: Identity drift across frames**
A person's face, hair color, or clothing changes gradually across frames — the model "forgets" the identity of the subject over the video's temporal span.

Detection: extract face identity embeddings at regular intervals; alert if identity drift > threshold (ArcFace similarity drops below 0.5).

Mitigation: longer temporal attention context (model sees more frames simultaneously), identity anchor injection (periodically re-inject reference appearance information).

**Ethical: Deepfake and misinformation**
Synthetic video is significantly more persuasive than synthetic images for spreading misinformation — realistic motion and temporally consistent faces are cognitively processed as more credible than still images. A 10-second synthetic video of a public figure making a fabricated statement is more dangerous than a fake image.

Mitigation: (1) all generated videos must contain a C2PA provenance watermark in metadata AND an invisible steganographic watermark in the pixel data — both must be present; (2) video deepfake detection tools should be made publicly available; (3) face generation should be gated (require explicit acknowledgment that generated videos may not depict real events).

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "You've built a text-to-video generation system. You're now asked to build a video generation platform that powers a film studio's creative pipeline, an advertising agency's campaign generation, and a social media app's short-form video features. What platform decisions matter most?"

### Signal Being Tested

Does the candidate think about quality tiers (creative/professional vs. consumer), infrastructure cost optimization across wildly different usage patterns, and the safety responsibilities unique to video?

### Follow-up Probes

- "Film studios need 4K 60fps video; social media apps need 720p 8fps. How do you serve both from the same platform without inefficiency?"
- "What is the most expensive shared infrastructure investment, and why should it be shared?"

---

### Model Answers — Section 8

**No Hire:**
"Build separate systems for each use case." No consideration of shared platform economics.

**Lean No Hire:**
Suggests shared model but doesn't address quality tiers, cost optimization, or the fundamentally different safety requirements for professional vs. consumer use.

**Lean Hire:**
Describes quality tiers (consumer 512p/8fps vs. professional 4K/24fps), shared base model with tier-specific upsampling, and shared watermarking infrastructure.

**Strong Hire Answer (first-person):**

A video generation platform serving film studios, advertising agencies, and social media apps must handle wildly different requirements without maintaining N separate platforms.

**Tiered generation architecture:**
I design a base model + cascade architecture:
- *Base model*: generates 512×512 at 8fps with short temporal context (16 frames) — fast, relatively cheap. Serves all tiers as a starting point.
- *Spatial upsampler*: 512→1080p or 512→4K using a video super-resolution model. Film studio tier uses 4K upsampling; social media uses 1080p.
- *Temporal upsampler*: 8fps → 24fps frame interpolation model. Film and advertising use 24fps; social media uses 8fps.
- *Long video model*: for film studio use, generate consistent long-form video (1–10 minutes) using a sliding window approach with cross-window identity anchoring.

**Shared infrastructure investments:**
1. *Base model serving cluster*: highest leverage — all products use the same base model, so a single shared cluster serving the base model benefits all tiers.
2. *Video processing pipeline*: encoding/decoding, frame rate conversion, format standardization — shared across all tiers.
3. *Safety infrastructure*: video deepfake detection, C2PA watermarking, harmful content classifier — non-negotiable for all tiers. Shared once.
4. *Evaluation harness*: FVD, temporal consistency metrics, human rating pipeline — shared with tier-specific benchmarks.

**Safety tier differentiation:**
Consumer (social media): strict content policy, mandatory watermarking visible in corner, no real person face generation. Advertising: brand safety filters (no competitor brand imagery), watermark optional (many clients want unbranded content), real person generation gated behind talent consent workflows. Film studio: least restricted (can generate photorealistic people for pre-approved storyboard use cases), C2PA metadata required, no visible watermark but invisible steganographic watermark always present.

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**Video diffusion training objective:**
```
L = E_{t,z_0,ε,c} [||ε - ε_θ(z_t, t, c)||²]
z ∈ R^{T×h×w×C} (video latent)
```

**Video forward diffusion:**
```
q(z_t | z_0) = N(z_t; √ᾱ_t·z_0, (1-ᾱ_t)·I)
(same as image diffusion, now applied to video latent)
```

**FVD (Fréchet Video Distance):**
```
FVD = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^{1/2})
(computed in I3D feature space, not Inception)
```

**Spacetime patches in Video DiT:**
```
Num patches N = (T/p_t) × (H/p_h) × (W/p_w)
Example: T=16, H=W=32, p_t=p_h=p_w=2 → N=8×16×16=2048
```

**Temporal attention:**
```
TemporalAttn(x_pos) = Softmax(Q_pos K_pos^T / √d_k) · V_pos
where Q, K, V from x_pos = [f_1(i,j), ..., f_T(i,j)]
```

**Warping error (temporal consistency):**
```
warping_error = LPIPS(warp(f_t, flow_t), f_{t+1})
```

**CFG for video:**
```
ε̃_θ(z_t, t, c) = ε_θ(z_t, t, ∅) + γ·(ε_θ(z_t, t, c) - ε_θ(z_t, t, ∅))
```

**Video latent dimensionality (CausalVAE):**
```
Raw: T×H×W×3 (e.g., 120×512×512×3 = 94M pixels for 5s @24fps @512p)
Compressed: (T/f_t)×(H/f_s)×(W/f_s)×C_z (e.g., 30×64×64×4 = 500K values)
Compression ratio: ~190×
```

### Vocabulary Cheat Sheet

| Term | Definition |
|---|---|
| **Video DiT** | Video Diffusion Transformer; processes spacetime patches with full 3D attention |
| **Spacetime patch** | 3D patch (time × height × width) as token unit for video transformer |
| **Temporal attention** | Attention across frames at each spatial position; enables motion coherence |
| **CausalVAE** | VAE with causal temporal convolutions; enables frame-by-frame streaming decode |
| **FVD** | Fréchet Video Distance; FID analog using I3D features for video quality |
| **Temporal flickering** | Frame-to-frame appearance inconsistency in static regions |
| **Warping error** | Inconsistency between optical flow prediction and actual next frame |
| **Pseudo-3D convolution** | 2D spatial conv + 1D temporal conv; cheap alternative to full 3D conv |
| **Temporal inflation** | Extending 2D image model to video by adding temporal conv/attention layers |
| **Optical flow** | Per-pixel motion vectors between consecutive frames |
| **I3D** | Inflated 3D CNN for video understanding; used for FVD feature extraction |
| **VideoCLIP** | Video-language model; computes text-video alignment for evaluation |
| **Cascaded generation** | Generate at low resolution, then upsample (spatial and/or temporal) |
| **C2PA** | Content Provenance and Authenticity standard; mandatory for AI video |
| **Frame interpolation** | Generate intermediate frames between keyframes (8fps → 24fps) |

### Key Numbers Table

| Metric | Value |
|---|---|
| Sora generation (1 min, 1080p) | Minutes to hours per clip |
| Consumer video gen (5s, 512p) | 30–90 seconds on 8× A100 |
| CausalVAE compression ratio | ~190× (pixel to latent) |
| Video DiT patches (typical) | 2048 spacetime tokens per video |
| FVD: excellent text-to-video | < 200 |
| FVD: good | 200–500 |
| FVD: poor | > 1000 |
| Cost per 5s video generation | ~$0.50 (8× A100, 2 min) |
| A100 tensor parallel for Video DiT | 4–8 GPUs |
| Temporal attention complexity | O(T² × H × W / p_s²) |
| Full 3D attention complexity | O((T×H×W / p_t·p_s²)²) |
| Target generation latency (consumer) | 30–90 seconds |
| Typical training video length | 3–15 seconds |
| Temporal consistency metric threshold | LPIPS < 0.10 between adjacent frames |

### Rapid-Fire Day-Before Review

1. **Why can't you just generate N independent images for a video?** Each frame would be independently photorealistic but have no temporal consistency — objects teleport, identity drifts, physics is violated
2. **Video diffusion loss?** Same as image: `E[||ε - ε_θ(z_t, t, c)||²]` but z is now a video latent tensor
3. **Temporal attention purpose?** Attends across frames at each spatial position — enables motion tracking and temporal consistency
4. **FVD vs. FID?** FVD uses I3D features (captures temporal dynamics), FID uses Inception features (single frame only)
5. **Spacetime patches?** 3D patches spanning time × height × width; tokenized for Video DiT input
6. **CausalVAE purpose?** Compresses video spatiotemporally; causal design enables frame-streaming decode
7. **Warping error measures?** LPIPS between next frame predicted by optical flow warp vs. actual next frame — physical motion consistency
8. **Full 3D vs. factorized attention?** Full 3D: O(N²) where N = T×H×W/p³ — best quality; Factorized: O(spatial²) + O(temporal²) — 4-8× cheaper
9. **Why is video deepfake more dangerous than image deepfake?** Temporal coherence and realistic motion are more cognitively persuasive; harder to visually identify as synthetic
10. **Async serving reason?** Video generation takes 30–300 seconds; users cannot interactively wait; job queue + notification is the production pattern
