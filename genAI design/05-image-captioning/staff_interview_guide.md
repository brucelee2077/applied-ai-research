# Image Captioning System — Staff/Principal Interview Guide

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

> "Design an image captioning system — given an image, generate a natural language description of its content. This could power accessibility features (alt text for screen readers), image search indexing, or social media auto-descriptions. Walk me through your approach."

### Signal Being Tested

Does the candidate recognize that image captioning is a multimodal problem requiring both vision and language components, and that the use case shapes the quality requirements?

### Six Clarification Dimensions

| Dimension | Why It Matters |
|---|---|
| **Use case** | Accessibility (screen readers) has higher recall requirement; search indexing needs factual accuracy |
| **Caption length and style** | Short descriptive phrase vs. long narrative — affects architecture and training data |
| **Image type** | Natural photos, charts/diagrams, screenshots — different models for different distributions |
| **Latency requirements** | Batch indexing (seconds OK) vs. real-time accessibility (< 500ms) |
| **Language** | English-only vs. multilingual captions |
| **Hallucination tolerance** | Accessibility captions with wrong details can seriously mislead blind users |

### Follow-up Probes

- "How does the system change if captioning must run entirely on-device for privacy?"
- "What changes about your approach if the images are medical scans vs. social media photos?"
- "How do you handle images that are deliberately misleading or contain embedded text?"

---

### Model Answers — Section 1

**No Hire:**
Immediately proposes "training a CNN on image-caption pairs" without clarifying use case, caption style, or latency. No recognition of the multimodal nature.

**Lean No Hire:**
Identifies the task as image-to-text but misses the use case dimension entirely. Doesn't ask about hallucination tolerance (which is critical for accessibility use cases) or the distinction between natural photos and non-photographic images.

**Lean Hire:**
Asks about use case, latency, and caption length. Identifies that the hallucination risk is higher for accessibility than for search indexing. Notes that multimodal architectures (vision encoder + language decoder) will be needed.

**Strong Hire Answer (first-person):**

Image captioning is a deceptively wide problem space — the right design depends entirely on what "good" means for the specific use case.

First, the use case. Accessibility alt-text for screen readers has zero tolerance for hallucination: a blind user who hears "a woman holding a red balloon" when the image shows a man holding a sign will be actively misled. Search indexing captioning can tolerate slightly imprecise descriptions as long as key objects and scenes are identified. Social media auto-captions are more conversational and can sacrifice precision for engaging prose. Each use case has a different quality-hallucination trade-off.

Second, caption length and style. A single sentence ("A dog playing in a park") is sufficient for alt-text. A detailed narrative description ("Two golden retrievers chasing a frisbee in a green park, with a red barn visible in the background") is better for rich search indexing. Each requires different training data.

Third, image type. Natural photographs are the most common case, but the system may encounter charts, diagrams, screenshots, and memes. A model trained on natural photo-caption pairs performs poorly on charts. I'd want to know if we need to handle non-photographic images.

Fourth, latency. Real-time accessibility in a web browser requires < 500ms. Batch indexing of an image corpus can run overnight. These are entirely different infrastructure requirements.

Fifth, hallucination tolerance. For accessibility, I would add a confidence-gated approach: only generate a caption when the model confidence is above a threshold; show "image" as fallback for low-confidence cases rather than a potentially wrong caption.

Let me proceed assuming: accessibility + search indexing use case, English-only, natural photos primarily, batch + near-real-time serving, and low hallucination tolerance.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "How do you formally frame image captioning as an ML problem? What are the components and how do they interact?"

### Signal Being Tested

Does the candidate correctly decompose image captioning into a vision encoder and a language decoder connected via cross-attention? Can they specify the training objective and explain why this architecture is appropriate?

### Follow-up Probes

- "Why is image captioning harder than text translation, even though both are seq-to-seq problems?"
- "What does the visual encoder learn? What representation does it produce?"
- "Could you solve this with retrieval instead of generation? When would that be better?"

---

### Model Answers — Section 2

**No Hire:**
"I would classify image regions and concatenate the class labels into a caption." Cannot formalize as conditional generation.

**Lean No Hire:**
Identifies image-to-text as seq2seq but cannot describe what the visual encoder produces or how it interacts with the decoder.

**Lean Hire:**
Correctly describes the architecture as visual encoder (CNN or ViT) + text decoder (transformer), connected via cross-attention or prefix conditioning. Specifies cross-entropy training loss. Can explain why this differs from text-to-text generation.

**Strong Hire Answer (first-person):**

Image captioning is conditional text generation where the conditioning signal is a visual representation rather than a text sequence. The model estimates:

```
p(y_1,...,y_n | I) = Π_{t=1}^{n} p(y_t | y_1,...,y_{t-1}, f(I))
```

where I is the image, f(I) is a visual representation (image embedding or patch token sequence), and y_1..y_n is the caption.

The architecture has two components:

**Visual Encoder:**
The encoder maps an image I ∈ R^{H×W×3} to a sequence of visual features V ∈ R^{P×d} where P is the number of image patches or regions and d is the feature dimension. Modern approaches use a Vision Transformer (ViT): the image is divided into non-overlapping 16×16 pixel patches, each patch is linearly projected to a d-dimensional embedding, positional embeddings are added, and the sequence of patch embeddings is processed by a stack of transformer encoder layers.

For a 224×224 image with 16×16 patches: P = (224/16)² = 196 patches. Each patch is a 16×16×3 = 768-dimensional vector linearly projected to d_model.

**Text Decoder:**
A causal transformer decoder generates the caption token by token, conditioning on the visual features via cross-attention at each layer:
```
CrossAttn(Q_dec, K_vis, V_vis) = softmax(Q_dec K_vis^T / √d_k) · V_vis
```
This allows each generated token to attend to all visual patches — focusing on the relevant image regions at each generation step.

**Training objective:**
```
L = -Σ_{(I, y)} Σ_{t=1}^{|y|} log p_θ(y_t | y_{<t}, f(I))
```

**Why harder than text translation?**
In MT, source and target are in the same modality (both are sequences of linguistic tokens). In captioning, we must bridge the modality gap between pixel patches and language tokens. The model must learn: (1) to extract semantically meaningful features from pixel patches, (2) to ground language tokens in visual features, and (3) to generate coherent text while attending to image regions. The cross-modal alignment is what makes captioning fundamentally harder.

**Could retrieval solve this?** Yes — nearest-neighbor image captioning finds the most similar image in a database and uses its caption. This is exact and non-hallucinatory. But it fails for novel images not well-represented in the database, and it can't compose descriptions of novel visual combinations.

---

## Section 3: Data & Preprocessing (8 min)

### Interviewer Prompt

> "What training data do you use? How do you preprocess images and captions, and what augmentation helps here?"

### Signal Being Tested

Does the candidate know the standard captioning datasets (COCO, Conceptual Captions)? Can they describe image preprocessing and augmentation strategies that are appropriate for multimodal training?

### Follow-up Probes

- "How do you handle the fact that each image in COCO has 5 reference captions? Do you use all 5 during training?"
- "What image augmentations could hurt captioning quality? Which help?"
- "How do you scale to a web-scale captioning dataset like Conceptual Captions 12M?"

---

### Model Answers — Section 3

**No Hire:**
Cannot name a captioning dataset or describe caption annotation. "I would scrape images from the web."

**Lean No Hire:**
Knows COCO exists but cannot describe its structure (5 captions per image) or how to use multiple references during training.

**Lean Hire:**
Describes COCO's structure and Conceptual Captions. Can explain image preprocessing (resize, normalize). Notes that augmentation must preserve the semantic content referenced in the caption.

**Strong Hire Answer (first-person):**

**Captioning datasets:**

*COCO Captions*: 330K images, each annotated with 5 human-written captions. The standard benchmark for image captioning. The 5 captions per image provide diversity — different annotators describe different aspects ("a man kicking a soccer ball" vs. "a soccer player in a red jersey on a grass field"). During training, I randomly sample one of the 5 captions per image per epoch — this provides implicit augmentation and exposes the model to diverse description styles.

*Conceptual Captions (CC3M, CC12M)*: 3–12 million web-scraped image-alt-text pairs. Automatically filtered using CLIP score (image-text alignment). Lower quality than COCO but much larger scale — useful for pretraining before fine-tuning on COCO.

*LAION-COCO*: 600M COCO-style captions generated by CoCa, filtered by quality. Provides synthetic large-scale pretraining data.

**Image Preprocessing:**
For ViT-based encoders, standard preprocessing is:
1. Resize to 224×224 (or 336×336 for higher-resolution variants)
2. Normalize: subtract ImageNet mean, divide by std
   `μ = (0.485, 0.456, 0.406), σ = (0.229, 0.224, 0.225)`
3. For ViT: divide into 16×16 patches, linearly project each patch

**Caption Preprocessing:**
1. Lowercase and strip special characters
2. Tokenize with a vocabulary matching the language model's tokenizer
3. Wrap with `[BOS]` and `[EOS]` tokens
4. Pad/truncate to max caption length (typically 25–50 tokens)

**Augmentation strategy:**
The key constraint: augmentation must not change the semantic content of the image, since the caption describes that content.

Safe augmentations:
- *Horizontal flip* (if caption doesn't reference left/right orientation): "a dog on the left" would become incorrect after flip, so this requires care. For captions without spatial references, horizontal flip is safe.
- *Color jitter* (brightness, contrast, saturation): doesn't change what objects are present
- *Random crop* (at least 80% of original area): preserves main content; teaches robustness to partial views

Unsafe augmentations:
- *Aggressive random crop* (< 50% area): may crop out the main subject, making the caption incorrect
- *Vertical flip*: produces unnatural images not represented in test distribution
- *Text overlay*: could conflict with embedded text already referenced in captions

**Handling web-scale data quality:**
For CC12M, I apply CLIP-score filtering (cosine similarity > 0.25) to remove image-text mismatches. I also filter captions containing personal data (names, phone numbers), captions shorter than 5 tokens, and captions that are clearly templated boilerplate ("Click here for more information").

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "Walk me through the specific architecture. How does a Vision Transformer encode images, and how does the decoder attend to those visual features?"

### Signal Being Tested

Does the candidate understand ViT patch embedding, positional encoding for images, and the cross-attention mechanism connecting visual and language representations?

### Follow-up Probes

- "Walk me through ViT's patch embedding step formally — what is the shape of each tensor?"
- "Why does image captioning use cross-attention rather than simply prepending image features to the text sequence?"
- "What is the difference between OFA, BLIP, and CoCa architecturally?"

---

### Model Answers — Section 4

**No Hire:**
"I would use a CNN to extract features and then an RNN to generate text." Cannot describe ViT or cross-attention for captioning.

**Lean No Hire:**
Knows ViT exists but cannot describe the patch embedding step or explain how cross-attention connects ViT features to the language decoder.

**Lean Hire:**
Correctly describes ViT patch embedding and self-attention. Explains decoder cross-attention over visual tokens. Can describe at least one modern captioning architecture (BLIP, CoCa).

**Strong Hire Answer (first-person):**

Let me walk through the full architecture from pixels to caption.

**Vision Transformer (ViT) Encoder:**

Step 1 — Patch embedding:
```
Input: I ∈ R^{H×W×C} (e.g., 224×224×3)
Patches: divide into P = (H×W)/p² patches, each of size p×p×C (p=16)
Flatten each patch: P vectors of dimension p²×C = 768
Linear projection: each patch → d_model (e.g., 768)
Result: X_patches ∈ R^{P×d_model} (e.g., 196×768)
```

Step 2 — CLS token and positional embedding:
```
Prepend learnable [CLS] token: X ∈ R^{(P+1)×d_model}
Add 2D positional embeddings (learned absolute positions)
```

Step 3 — Transformer encoder layers:
```
For each of L layers:
  X = X + MultiHeadSelfAttn(LayerNorm(X))
  X = X + FFN(LayerNorm(X))
```

The output is V ∈ R^{(P+1)×d_model}: one vector per patch + the CLS token. For captioning, I use all patch vectors (not just CLS) as the cross-attention keys/values — this gives the decoder fine-grained spatial access to different image regions.

**Language Decoder with Cross-Attention:**

Each decoder layer has three sub-modules:
1. Causal self-attention over previously generated tokens
2. Cross-attention over visual patch features:
```
CrossAttn(Q_dec, K_vis, V_vis) = softmax(Q_dec K_vis^T / √d_k) · V_vis
```
3. Feed-forward network

The cross-attention at step t asks: "given what I've generated so far, which image patches are most relevant for generating the next token?" When generating "dog", the attention weights peak at image patches containing the dog. When generating "green", they peak at patches containing grass.

**Modern captioning architectures:**

*BLIP (Bootstrapping Language-Image Pre-training)*: uses a mixture of three objectives — image-text contrastive (ITC), image-text matching (ITM), and image-conditioned language modeling (LM). This jointly trains understanding and generation, making the model strong on both retrieval and captioning tasks.

*CoCa (Contrastive Captioners)*: joint training with contrastive loss (like CLIP) + captioning loss. The contrastive objective aligns the CLS token embedding between image and text; the captioning objective trains the decoder on all patch tokens. This produces a single model that handles both retrieval and generation.

*OFA (One For All)*: sequence-to-sequence framework where the visual input is represented as discrete tokens (via VQ-VAE or image tokenizer) and the language model processes vision + language in a unified sequence.

For production, I would choose BLIP-2 or CoCa — they combine pretraining efficiency with strong generation quality and are designed to work with frozen pretrained vision encoders (ViT-G) and frozen LLMs, reducing fine-tuning cost dramatically.

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How do you evaluate caption quality? Walk me through BLEU, CIDEr, and human evaluation."

### Signal Being Tested

Does the candidate know CIDEr specifically (the dominant image captioning metric) and understand why it differs from BLEU? Can they describe human evaluation approaches for captioning?

### Follow-up Probes

- "What does CIDEr measure that BLEU doesn't?"
- "Why is SPICE a good complement to CIDEr for captioning evaluation?"
- "How would you evaluate hallucination specifically — a separate concern from overall caption quality?"

---

### Model Answers — Section 5

**No Hire:**
"I would use accuracy." Cannot describe any captioning metric.

**Lean No Hire:**
Mentions BLEU for captioning. Cannot describe CIDEr or why it was developed specifically for captioning.

**Lean Hire:**
Knows CIDEr is the standard captioning metric and can explain that it upweights n-grams specific to the image rather than common across all captions. Distinguishes CIDEr from BLEU. Mentions SPICE.

**Strong Hire Answer (first-person):**

Image captioning evaluation uses a different metric from MT precisely because common words ("a", "the", "is") should not dominate the score — what matters is whether the caption captures the image-specific content.

**CIDEr (Consensus-based Image Description Evaluation):**

CIDEr computes similarity between a candidate caption and a set of reference captions, weighted by the TF-IDF importance of each n-gram across the dataset. The key insight: n-grams that are common across all image captions (e.g., "a photo of") carry little information and should be downweighted; n-grams specific to this image (e.g., "a red bicycle") should be upweighted.

Formally:
```
CIDEr_n(c, S) = (1/m) Σ_{i=1}^{m} [g_n(c) · g_n(s_i)] / [||g_n(c)|| · ||g_n(s_i)||]
```
where g_n(c) is a vector of TF-IDF weighted n-gram counts for candidate c, S is the set of reference captions, and the dot product computes cosine similarity in n-gram space. CIDEr = average over n=1..4.

CIDEr typically correlates better with human judgments of caption quality than BLEU for the captioning task, achieving Pearson r ≈ 0.85–0.90 vs. ≈ 0.70 for BLEU on COCO.

**SPICE (Semantic Propositional Image Caption Evaluation):**
SPICE parses captions into scene graphs (subject-relation-object triples) and measures how well the candidate caption's scene graph matches the reference. This captures semantic structure rather than surface n-gram overlap. Strong complement to CIDEr: a caption can score well on CIDEr by matching surface forms without correct semantic relationships.

**Standard benchmarks:**
- COCO test set (Karpathy split): 5K test images, 5 references each
- Typical strong model scores: CIDEr ~140 (CIDEr@1), SPICE ~23–24

**Hallucination evaluation:**
CHAIR (Caption Hallucination Assessment with Image Relevance) measures what fraction of objects mentioned in the caption are actually present in the image:
```
CHAIR_i = |hallucinated_objects| / |all_mentioned_objects|
CHAIR_s = |hallucinated_captions| / |all_captions|
```
Lower CHAIR is better. A model with CIDEr=130 but CHAIR_i=20% may be less suitable for accessibility use cases than a model with CIDEr=120 and CHAIR_i=5%.

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "How do you serve the image captioning system at scale — for both real-time accessibility use and batch indexing of an image corpus?"

### Signal Being Tested

Does the candidate understand the two-phase inference (vision encoding + text decoding), batching strategies for each, and the different requirements for real-time vs. batch serving?

### Follow-up Probes

- "How do you batch image captioning requests efficiently?"
- "What can you pre-compute and cache in a captioning pipeline?"
- "How does the encoder vs. decoder cost split affect your hardware choices?"

---

### Model Answers — Section 6

**No Hire:**
"Run the model on a GPU." No understanding of two-phase inference or batching differences.

**Lean No Hire:**
Recognizes two phases but cannot describe batching strategies or the compute asymmetry between encoder and decoder.

**Lean Hire:**
Correctly explains encoder pre-computation and caching, decoder beam search, and the different batching strategies for real-time vs. batch serving.

**Strong Hire Answer (first-person):**

Image captioning has a natural two-phase structure: encode the image once, then decode the caption token by token. These phases have very different compute profiles and can be optimized independently.

**Phase 1: Visual Encoding**
The ViT encoder processes the full image in a single forward pass, producing P+1 visual tokens. This is highly parallelizable (no sequential dependencies). For batch processing, I can process images in batches of 64–128 with full GPU utilization.

Latency: ViT-L/16 on an A100 GPU encodes one 224×224 image in ~10ms; batches of 64 images in ~15ms total (effective 0.23ms per image with batching).

The encoder output is computed once per image and can be cached. For batch indexing (indexing an image corpus), I store the visual encoder features alongside each image and only run the decoder when a caption is needed. This allows decoupling the encoding and captioning jobs.

**Phase 2: Caption Decoding**
The decoder generates tokens sequentially (autoregressive). Each step attends to the cached visual features (cross-attention) and all previous tokens (causal self-attention). This cannot be parallelized across tokens within a single caption.

For beam search (beam width 4), the decoding of a 25-token caption requires 25 sequential steps × 4 beams = 100 forward passes. I use KV-cache for the decoder's self-attention: the decoder's own K/V are cached as each token is generated.

**Serving mode comparison:**
- *Batch indexing*: process images in large batches; latency per image < 100ms is acceptable; throughput is the primary metric
- *Real-time accessibility*: caption must appear within 500ms of image load; single-request latency is primary metric; beam width can be reduced to 2–3 for speed

**Hardware allocation:**
The encoder is compute-intensive but parallelizable — ideal for GPU with high memory bandwidth. The decoder is memory-bandwidth-bound (large model weights loaded for each token step) — benefits from INT8 quantization and high-memory GPUs.

For very high throughput, I would run encoder and decoder on separate GPU pools and pipeline requests through both stages asynchronously.

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the most critical failure modes for image captioning, especially in the accessibility use case?"

### Signal Being Tested

Does the candidate identify hallucination, racial/gender bias in captions, failure on non-photographic images, and text-in-image handling? Can they propose detection and mitigation strategies?

### Follow-up Probes

- "A blind user receives a caption 'a man in a red shirt' but the image is of a woman. What went wrong and how do you prevent it?"
- "How does the model handle images that contain text (memes, screenshots, signs)?"
- "What happens when the model encounters image distributions it was never trained on?"

---

### Model Answers — Section 7

**No Hire:**
Cannot describe captioning-specific failure modes. Generic "the model might be wrong."

**Lean No Hire:**
Mentions hallucination but cannot describe the specific types or propose CHAIR-based detection.

**Lean Hire:**
Identifies hallucination (CHAIR), demographic bias in gender/race attribution, and text-in-image as failure modes. Proposes confidence thresholding and bias auditing.

**Strong Hire Answer (first-person):**

Captioning for accessibility has much higher stakes than captioning for search indexing — a wrong caption actively misleads a blind user.

**Object hallucination:**
The model describes objects not present in the image. This is the most studied failure mode (CHAIR metric). Common causes: training data bias (models learn that kitchens usually have microwaves, so they hallucinate microwaves in kitchen images even when absent), attention collapse (cross-attention doesn't focus on the relevant regions).

Mitigation: (1) reduce beam search temperature (more conservative decoding), (2) ground each noun in the caption to a detected object via an object detector (refuse to mention objects with detection confidence < threshold), (3) use a factual consistency model to verify caption against image.

**Demographic misattribution:**
The model assigns incorrect gender, age, or apparent race to people in images. Training data biases (COCO captions use "man" and "woman" based on annotator perception, with systematic biases toward Western appearance norms) are directly encoded in the model.

Detection: run the model on a gender-balanced, racially diverse evaluation set and measure accuracy of gender/race attribution when ground truth is known. Any disparity > 5 percentage points between demographic groups is a signal for bias mitigation.

**Text in images:**
Images containing visible text (signs, book covers, memes) require OCR capability, not just object recognition. A model without OCR ability will generate captions like "a sign on a building" instead of "a sign reading 'EXIT' on a building." For accessibility, the text content is often the most important information.

Mitigation: augment the captioning pipeline with an OCR module; inject recognized text into the caption template or provide it as a separate field to screen readers.

**Non-photographic images:**
Charts, graphs, infographics, and technical diagrams have completely different visual structure from natural photos. A model trained on COCO may generate "a colorful pattern" for a bar chart. For accessibility, these are particularly important — data visualizations are inaccessible to blind users without data-specific captions.

Mitigation: image type classifier at inference time; route charts/diagrams to a specialized captioning model fine-tuned on chart/diagram data.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "You've built image captioning for web accessibility. Now you want to build this as a platform: image captioning, visual question answering, image search, and OCR, all from shared infrastructure. What is the shared foundation?"

### Signal Being Tested

Does the candidate identify that a shared vision encoder foundation underlies all vision-language tasks, and that the specialization happens at the task-specific head/decoder level?

### Follow-up Probes

- "What is the shared representation that powers all these tasks?"
- "How do you evaluate a shared vision encoder across diverse downstream tasks?"

---

### Model Answers — Section 8

**No Hire:**
"Build a separate model for each task." No consideration of shared visual backbone.

**Lean No Hire:**
Suggests shared CNN backbone but doesn't identify the language decoder layer or explain how different tasks specialize.

**Lean Hire:**
Correctly identifies shared ViT encoder as the foundation, with task-specific decoders for captioning, VQA, and retrieval. Notes that CLIP-style pretraining provides the shared representation.

**Strong Hire Answer (first-person):**

All vision-language tasks benefit from the same high-quality visual representation. The platform architecture is: shared frozen ViT encoder → task-specific adapters.

**Shared foundation: CLIP-pretrained ViT**
A ViT encoder pretrained with CLIP (contrastive language-image pretraining) produces visual features that are semantically aligned with language. This alignment is what makes one encoder work across captioning, VQA, image search (retrieval), and OCR-enhanced understanding — the encoder has learned to produce features that language models can "understand."

The training cost of a ViT-L (307M parameters) pretrained on LAION-5B is ~$1M in compute. This cost is paid once; all downstream tasks share this encoder.

**Task-specific decoders/heads:**
- *Image captioning*: frozen ViT → cross-attention decoder (BLIP-2 style with Q-Former bridge)
- *Visual question answering*: frozen ViT → Q-Former → LLM that generates the answer
- *Image search/retrieval*: frozen ViT → CLS embedding → FAISS nearest-neighbor index
- *OCR-enhanced captioning*: frozen ViT + OCR model → LLM decoder with both visual and text inputs

**What to share in infrastructure:**
- Visual encoding serving fleet: shared across all tasks, high throughput batch mode
- Visual feature store: computed ViT features cached per image (all tasks share)
- CLIP embeddings: dual vision/text embedding space for retrieval — shared with captioning and VQA

The key insight: keeping the ViT encoder frozen (or lightly fine-tuned via LoRA) across all tasks prevents catastrophic forgetting and allows all tasks to benefit when the shared encoder is updated.

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**Image captioning conditional probability:**
```
p(y | I) = Π_{t=1}^{n} p(y_t | y_1,...,y_{t-1}, f(I))
```

**Training cross-entropy loss:**
```
L = -Σ_{(I,y)} Σ_{t=1}^{|y|} log p_θ(y_t | y_{<t}, f(I))
```

**ViT patch embedding:**
```
Input image: I ∈ R^{H×W×C}
Patches: P = HW/p², each patch ∈ R^{p²C}
Projection: x_i = W_e · patch_i + b_e, W_e ∈ R^{d × p²C}
Sequence: [x_cls; x_1; ...; x_P] with positional embeddings
```

**Cross-attention in decoder:**
```
CrossAttn(Q_dec, K_vis, V_vis) = softmax(Q_dec K_vis^T / √d_k) · V_vis
Q_dec ∈ R^{n_dec × d_k}, K_vis, V_vis ∈ R^{P × d_k}
```

**CIDEr metric:**
```
CIDEr_n(c, S) = (1/m) Σ_{i} [g_n(c) · g_n(s_i)] / [||g_n(c)|| · ||g_n(s_i)||]
g_n: TF-IDF weighted n-gram vector
CIDEr = Σ_{n=1}^{4} w_n CIDEr_n,  w_n = 1/4
```

**CHAIR hallucination metric:**
```
CHAIR_i = |hallucinated_objects| / |all_objects_mentioned|
CHAIR_s = |captions_with_hallucination| / |total_captions|
```

**SPICE scene graph F-score:**
```
SPICE = F_1(scene_graph(c), scene_graph(S))
= 2·P·R/(P+R) where P,R over semantic propositions
```

**BLEU for comparison:**
```
BLEU = BP · exp(Σ w_n log p_n), w_n = 1/4
```

### Vocabulary Cheat Sheet

| Term | Definition |
|---|---|
| **ViT** | Vision Transformer; divides image into patches and processes as sequence |
| **Patch embedding** | Linear projection of each image patch to d_model dimension |
| **CLS token** | Learnable token prepended to patch sequence; aggregate image representation |
| **Cross-attention** | Decoder attends to encoder (visual) features; Q from decoder, K/V from ViT |
| **CIDEr** | Consensus-based image description metric; TF-IDF weighted n-gram similarity |
| **SPICE** | Semantic scene graph F-score for caption evaluation |
| **CHAIR** | Caption Hallucination Assessment; fraction of objects hallucinated |
| **BLIP** | Bootstrapping Language-Image Pretraining; joint ITC+ITM+LM objectives |
| **CoCa** | Contrastive Captioner; joint contrastive + captioning training |
| **Q-Former** | Query-based cross-modal bridge in BLIP-2; lightweight adapter between ViT and LLM |
| **CLIP** | Contrastive Language-Image Pretraining; aligns text and image embeddings |
| **Teacher forcing** | Training decoder receives gold target tokens at each step |
| **Exposure bias** | Gap between teacher-forced training and free-generation inference |
| **OCR** | Optical Character Recognition; extract text embedded in images |
| **Constrained decoding** | Force model to ground mentions in detected objects |

### Key Numbers Table

| Metric | Value |
|---|---|
| COCO Captions dataset size | 330K images, 5 captions each |
| Conceptual Captions (CC12M) size | 12M image-text pairs |
| LAION-COCO size | 600M synthetic captions |
| ViT-L/16 parameters | 307M |
| ViT-L/16 patches (224×224, p=16) | 196 patches |
| ViT-L inference latency (A100) | ~10ms per image |
| Max caption length (typical) | 25–50 tokens |
| State-of-art COCO CIDEr | ~140–150 |
| State-of-art COCO SPICE | ~23–25 |
| Good CHAIR_i | < 5% |
| CLIP ViT-L/14 embedding dimension | 768 |
| CIDEr human correlation | ~0.85–0.90 Pearson r |
| BLIP-2 parameters (total) | ~8.7B (frozen ViT + Q-Former + LLM) |

### Rapid-Fire Day-Before Review

1. **What is the ViT patch embedding step?** Divide image into 16×16 patches, flatten each to p²C vector, linearly project to d_model
2. **Why cross-attention not self-attention for image-to-text?** Visual tokens (source) and text tokens (target) are separate sequences; cross-attention lets text decoder attend to image patches without mixing modalities in self-attention
3. **CIDEr vs BLEU key difference?** CIDEr uses TF-IDF weighting — upweights image-specific n-grams, downweights common ones; BLEU treats all n-grams equally
4. **What does CHAIR measure?** Fraction of objects mentioned in caption that are not actually in the image (hallucination rate)
5. **Why is captioning harder than MT?** Modality gap: must bridge pixel patches to language tokens; no parallel structure like source-target sentences
6. **What is BLIP's training objective?** Three objectives: ITC (contrastive), ITM (matching), LM (captioning) — joint understanding and generation
7. **How do you cache in captioning serving?** Store visual encoder features per image; re-run decoder only when caption needed
8. **ViT number of patches for 224×224, p=16?** P = (224/16)² = 196 patches
9. **What failure mode does CHAIR detect?** Object hallucination — model describes objects not present in the image
10. **Platform shared foundation?** CLIP-pretrained ViT encoder shared across captioning, VQA, image search, OCR tasks
