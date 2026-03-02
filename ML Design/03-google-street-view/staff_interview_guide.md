# Google Street View Blurring System — Staff/Principal Interview Guide

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
- The principal-level bar requires the candidate to connect individual design decisions to broader organizational or platform impact.

**Time Budget:**

| Section | Time |
|---|---|
| Problem Statement & Clarification | 5 min |
| ML Problem Framing | 5 min |
| Data & Feature Engineering | 8 min |
| Model Architecture Deep Dive | 12 min |
| Evaluation | 5 min |
| Serving Architecture | 7 min |
| Edge Cases & Failure Modes | 5 min |
| Principal-Level Platform Thinking | 3 min |

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

> "Google Street View captures imagery of streets around the world. Before publishing, Google must blur personally identifiable information — specifically human faces and license plates. You are asked to design a machine learning system to automate this blurring. Please start by clarifying the problem, then walk me through your approach."

### Signal Being Tested

Does the candidate ask the right scoping questions before jumping to a solution? A staff engineer should recognize that requirement ambiguity drives the entire system design, and should surface the six dimensions below without being prompted.

### Six Clarification Dimensions

| Dimension | Why It Matters |
|---|---|
| **Scale** | Determines offline vs. online, batch vs. streaming, model size constraints |
| **Latency requirements** | Batch is fundamentally different from real-time serving |
| **What to detect** | Face + license plate only? Body? Text? Signs? |
| **User correction pipeline** | Shapes online feedback loop design |
| **Geographic scope** | License plate formats vary internationally |
| **Privacy/legal requirements** | GDPR may require near-100% recall even at cost of precision |

### Follow-up Probes

- "What would change about your design if latency had to be under 200ms?"
- "If you had no labeled data at all, what would you do?"
- "Who are the stakeholders who care about this system's performance?"

---

### Model Answers — Section 1

**No Hire:**
The candidate immediately begins describing a neural network without asking any clarifying questions. There is no acknowledgment that requirements shape design. Example: "I would use YOLO to detect faces and blur them." No mention of scale, latency, data availability, or correction pipelines.

**Lean No Hire:**
The candidate asks one or two surface questions ("How many images are there?" or "Does it need to be real-time?") but does not probe for legal/compliance requirements, the scope of what objects to detect, or the feedback/correction pipeline. They treat the problem as a single-shot batch job without considering ongoing maintenance.

**Lean Hire:**
The candidate asks questions across at least four of the six dimensions. They identify that this is a batch processing job (latency not critical), that there are roughly 1 million labeled images, that the objects of interest are faces and license plates, and that user corrections exist. They note that the system has asymmetric error costs — a missed face is far worse than a false positive — and they mention that recall should be prioritized over precision for privacy reasons.

**Strong Hire (500+ words, first-person):**

Before I draw a single box on a whiteboard, I want to make sure I'm solving the right problem, because the requirement space here is much larger than it first appears.

The first thing I want to confirm is the detection scope. The obvious targets are human faces and license plates, but I would ask: do we also need to blur partial faces (someone turned away), faces in reflections (car mirrors, shop windows), children specifically (who may warrant extra caution under GDPR's child data provisions), and what about vehicle identification numbers (VINs) visible through windshields? Each additional class increases annotation burden and model complexity. For this design I will assume faces and license plates are the primary targets, but I want to flag that scope creep here is a real operational risk.

Second, latency and throughput. Street View ingests imagery continuously as Street View cars drive. The question is whether blurring needs to happen before an image is published — which is a hard latency requirement — or whether there is an acceptable delay between capture and publication. If publication can lag by hours or days, we design a batch processing pipeline with high-throughput GPU inference clusters. If images need to be published within minutes of capture, we design for streaming inference. I'll assume batch is acceptable, which simplifies the architecture considerably and lets us use heavier, more accurate models.

Third, the error cost asymmetry is critical and I want to make it explicit. A false negative — missing a face — exposes a real person's identity and creates legal liability under GDPR, CCPA, and similar regulations. A false positive — blurring a car door handle that isn't a face — is annoying but harmless. This asymmetry means we should tune our system to maximize recall, potentially at the cost of precision. I want to understand if there is a hard recall floor (e.g., "99.9% of faces must be blurred") versus a soft preference.

Fourth, the user correction pipeline. Street View has a "Report a problem" feature where users can flag images with unblurred PII. I want to understand: how many reports come in per day, what the validation workflow looks like (human review or automated?), and whether validated corrections feed back into training. This is not just a UX feature — it is a continuous learning signal that can dramatically improve the model over time, especially for long-tail cases like unusual license plate formats.

Fifth, geographic scope. License plates vary enormously by country. A US plate is a standard rectangle; a UK plate has a fixed yellow rear/white front format; some European plates are much narrower; some Asian plates use non-Latin characters. A model trained primarily on US data may have very poor recall on international plates. I want to know if the initial scope is US-only, and what the roadmap looks like for international expansion.

Sixth, what data do we already have? I understand there are roughly 1 million annotated images with bounding boxes for faces and license plates. I want to understand annotation quality — were they labeled by professional annotators or crowdsourced? What is the inter-annotator agreement? And are there hard negatives (images without faces or plates) included?

These six dimensions — detection scope, latency, error cost asymmetry, correction pipeline, geographic scope, and data quality — collectively determine every major architectural decision I will make. Let me now proceed with the design assuming batch processing, faces and license plates as targets, a high-recall requirement, and 1 million labeled images.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "How would you frame this as an ML problem? What type of model task is this?"

### Signal Being Tested

Does the candidate correctly identify this as an object detection problem (not classification, not segmentation) and explain why? Can they articulate the output format and the loss function structure?

### Follow-up Probes

- "Why not semantic segmentation?"
- "What's the difference between object detection and image classification here?"
- "Could you use a simpler rule-based system instead of ML?"

---

### Model Answers — Section 2

**No Hire:**
"This is an image classification problem — we classify each image as containing a face or not." This fundamentally misunderstands the task. We need to locate where faces are, not just detect their presence.

**Lean No Hire:**
Correctly identifies object detection but cannot articulate the output format. Says "we detect objects in the image" without specifying that the model must output class labels and bounding box coordinates simultaneously, or explaining the regression component.

**Lean Hire:**
Correctly frames this as object detection: the model must output a set of bounding boxes, each with a class label (face or license plate) and a confidence score. Notes that the output is `{(x1, y1, x2, y2, class, confidence)}` for each detected object. Distinguishes this from segmentation (pixel-level mask, overkill here — we just need a box to blur) and from classification (no localization). Mentions that blurring the bounding box region is the final step after detection.

**Strong Hire (500+ words, first-person):**

Let me be precise about the ML task formulation because it directly determines the model architecture, loss function, evaluation metric, and serving pipeline.

This is a **multi-class object detection** problem. The input is a raw Street View image — typically high-resolution, perhaps 4000x2000 pixels or larger when stitched from multiple cameras. The output is a set of predicted bounding boxes, where each bounding box is characterized by four coordinates (typically parameterized as center_x, center_y, width, height or as top-left and bottom-right corners), a class label drawn from {face, license_plate}, and a confidence score in [0, 1].

Formally, given an image I ∈ R^(H×W×3), the model produces:

```
ŷ = {(b_i, c_i, s_i) | i = 1, ..., N}
```

where b_i ∈ R^4 is the bounding box, c_i ∈ {face, license_plate} is the class, s_i ∈ [0,1] is the confidence score, and N is the number of detected objects (variable per image).

Now let me explain why I chose detection over alternatives:

**Why not image classification?** Classification would tell us "does this image contain a face?" but not where the face is. We cannot blur what we cannot locate. Classification is a necessary sub-task within detection, but it is insufficient on its own.

**Why not semantic segmentation?** Semantic segmentation produces a per-pixel class mask, which gives us precise object boundaries. This is accurate but computationally expensive and unnecessary for blurring. We do not need pixel-perfect boundaries — a slightly over-generous bounding box that blurs a few extra pixels is perfectly acceptable and actually preferable (conservative blurring is safer from a privacy standpoint). Detection with bounding boxes gives us exactly the resolution we need.

**Why not instance segmentation (Mask R-CNN)?** Instance segmentation adds per-instance masks on top of detection. It is more accurate for irregular shapes, like a partially occluded face at an angle. However, the additional complexity and inference time are not justified for blurring — bounding box blur is sufficient, and mask inference would add latency and memory overhead without meaningful privacy benefit.

**Could we use a rule-based system?** For license plates in controlled conditions (high resolution, standard lighting, standard angles), traditional computer vision approaches like edge detection followed by a Viola-Jones classifier or HOG+SVM might work with 85-90% precision. But Street View images have enormous variability in lighting, angle, resolution, occlusion, and geographic diversity. Rule-based systems degrade rapidly at the tail of this distribution. ML is the right choice here.

The output of the ML model feeds directly into a blur operation. After detection, for each bounding box (x1, y1, x2, y2), we apply a Gaussian blur kernel to the corresponding image region before storing or publishing the image. The blur radius should scale with the bounding box size to ensure adequate privacy protection regardless of how close or far the face is from the camera.

One nuance I want to flag: for privacy, we want to blur slightly beyond the bounding box boundary, not just within it. A tight bounding box that clips the edges of a face might leave enough recognizable features. I would add a small padding (e.g., 10-20% of the bounding box dimensions) to the blur region.

---

## Section 3: Data & Feature Engineering (8 min)

### Interviewer Prompt

> "Walk me through your data pipeline. What does your training data look like, and how would you preprocess and augment it?"

### Signal Being Tested

Does the candidate understand the annotation format, identify potential data quality issues, and apply domain-appropriate augmentation strategies? Do they recognize the difference between online and offline augmentation?

### Follow-up Probes

- "What augmentations would be harmful here? Which ones should you avoid?"
- "How would you handle class imbalance between faces and license plates?"
- "If you only had 10,000 labeled images, how would you adapt?"

---

### Model Answers — Section 3

**No Hire:**
"I would normalize the images and feed them into the model." No mention of bounding box annotation format, augmentation, or data quality issues.

**Lean No Hire:**
Describes standard image augmentation (flipping, rotation) but does not connect augmentation choices to the specific domain. For example, does not note that vertical flips are inappropriate for street-level imagery, or that bounding box coordinates must be transformed along with the image.

**Lean Hire:**
Describes the annotation schema correctly (image path, object class, bounding box coordinates). Lists domain-appropriate augmentations: horizontal flip, 90-degree rotation, brightness/contrast adjustments, rescaling. Notes that bounding box coordinates must be updated when the image is transformed. Mentions offline vs. online augmentation trade-off. Addresses class imbalance between faces and license plates.

**Strong Hire (500+ words, first-person):**

Let me walk through the full data pipeline from raw annotation to training-ready batches.

**Annotation Schema**

The training data has the following structure per sample:

```
{
  "image_path": "gs://streetview-images/2024/img_001.jpg",
  "objects": [
    {
      "class": "human_face",
      "bounding_box": {"x1": 412, "y1": 230, "x2": 467, "y2": 295}
    },
    {
      "class": "license_plate",
      "bounding_box": {"x1": 820, "y1": 640, "x2": 920, "y2": 680}
    }
  ],
  "metadata": {
    "location": {"lat": 37.7749, "lng": -122.4194},
    "camera": {"pitch": -5.2, "yaw": 90.0, "roll": 0.1}
  }
}
```

With 1 million annotated images, I have a substantial but not unlimited dataset. I expect roughly 2-5 objects per image on average (some images have crowds with many faces; many have zero or one), giving me on the order of 2-5 million annotated object instances total.

**Data Quality Checks**

Before any training, I want to run a data audit. Key checks:
- Bounding box validity: are all coordinates within image bounds? Are x2 > x1 and y2 > y1?
- Class distribution: how many faces vs. license plates? If severely imbalanced, I may need class-weighted sampling.
- Annotation consistency: spot-check a random sample of 1,000 images for annotation quality. What is the inter-annotator agreement on ambiguous cases (partially occluded face, very small face)?
- Difficult cases: what fraction of faces are smaller than 32x32 pixels? Small faces are notoriously difficult to detect and may warrant special treatment.

**Preprocessing**

The model expects fixed-size inputs. For a two-stage detector like Faster R-CNN, the standard approach is to:
1. Resize the image so the shorter side is 800 pixels (while maintaining aspect ratio), with a maximum longer side of 1333 pixels. This is the standard COCO training resolution.
2. Normalize pixel values to zero mean and unit variance using ImageNet mean/std: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).
3. Scale bounding box coordinates proportionally with the resize factor.

**Augmentation Strategy**

I strongly prefer **offline augmentation** for this pipeline. Offline augmentation pre-generates augmented images and stores them to disk (or cloud storage), which means:
- Training throughput is not bottlenecked by augmentation compute
- Augmented data can be versioned and reproduced exactly
- Workers spend all their time on forward/backward passes

The trade-off is storage cost — 3x augmentation on 1M images at ~2MB each requires ~6TB of additional storage, which is acceptable at Google scale.

**Domain-Appropriate Augmentations:**

1. **Horizontal flip** — Street images are symmetric; a face on the left side of frame is equivalent to one on the right. Bounding boxes must be mirrored: x1_new = W - x2_old, x2_new = W - x1_old. Always valid for street scenes.

2. **90-degree rotation** — This simulates cameras mounted at different orientations on the Street View car. This is valid here but must be applied carefully: a full 90-degree clockwise rotation transforms coordinates as (x, y) → (H - y, x) in the new coordinate system. I would apply rotations of 0, 90, 180, 270 degrees.

3. **Rescaling (multi-scale training)** — Randomly resize images to multiple scales (e.g., shorter side in {600, 800, 1000, 1200} pixels). This forces the model to handle faces and plates at multiple resolutions, which directly improves performance on the most challenging cases (tiny distant faces vs. large close-up faces). This is one of the most important augmentations for detection.

4. **Brightness and contrast adjustments** — Street View images are captured in all lighting conditions: bright noon sun, overcast skies, dawn, dusk, and even night with artificial lighting. Randomly adjusting brightness by ±30% and contrast by ±20% significantly improves robustness. Implemented as multiplicative pixel-level transforms.

5. **Color jitter** — Randomly adjust hue, saturation. Faces in particular vary across ethnicities and lighting; color jitter ensures the model does not overfit to specific skin tones.

**Augmentations I Would NOT Apply:**

- **Vertical flip** — A sky-below, road-above image is never seen in deployment. This would introduce distribution shift.
- **Aggressive crop** that removes all objects from the image — this creates a hard negative, which is useful only if intentionally managed.
- **Cutout/CutMix** on face regions — this might teach the model that faces can have large occluded regions, but it could also teach it to ignore faces. I would be cautious here.

**Class Imbalance Handling**

License plates are typically less numerous per image than faces in urban scenes. However, the more critical concern is the ratio of images with hard-to-detect objects (small faces, unusual angles) to easy ones. I would use:
- Oversampling of images with small objects (< 32x32 pixels) in the training batch sampler
- Hard example mining: track which images the model gets wrong most often and sample them more frequently

**Train/Val/Test Split**

I would split geographically rather than randomly. A random split might put images from the same block in both train and test, which inflates test performance due to near-duplicate scenes. Geographic split ensures the model must generalize to new locations.
- Train: 80% of geographic regions
- Validation: 10% (used for hyperparameter tuning)
- Test: 10% (held out until final evaluation, used only once)

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "Describe the model architecture you would use. Walk me through how it works mechanically, not just by name."

### Signal Being Tested

Does the candidate understand the internal mechanics of their chosen architecture? Can they explain the RPN, anchor boxes, ROI pooling, and loss functions without just naming them? Can they compare one-stage vs. two-stage approaches with concrete trade-offs?

### Follow-up Probes

- "Walk me through exactly what the Region Proposal Network does."
- "Why does Faster R-CNN use anchor boxes? What problem do they solve?"
- "How does Feature Pyramid Network improve detection of small objects?"
- "What is the regression target for bounding box refinement?"

---

### Model Answers — Section 4

**No Hire:**
"I would use a CNN." No description of detection-specific components. Cannot distinguish detection from classification architecturally.

**Lean No Hire:**
Names Faster R-CNN or YOLO but cannot explain how proposals are generated, what anchor boxes are, or how the two-stage architecture works. Says "the model outputs bounding boxes" without explaining the mechanism.

**Lean Hire:**
Correctly describes the two-stage architecture: backbone → RPN → ROI pooling → classification + regression head. Explains anchor boxes and the RPN's role. Describes the combined loss function. Can compare one-stage (YOLO/SSD) vs. two-stage (Faster R-CNN) with some trade-offs.

**Strong Hire (500+ words, first-person):**

I will design a two-stage detector based on the Faster R-CNN paradigm, because the accuracy-precision trade-off heavily favors two-stage for our use case. Let me explain the full mechanism.

**Why Two-Stage Over One-Stage**

One-stage detectors (YOLO, SSD, RetinaNet) are optimized for speed. They perform detection in a single forward pass, predicting class probabilities and bounding box offsets for a dense grid of anchor boxes directly. This makes them excellent for real-time applications — YOLO can run at 30-100 FPS.

However, our problem has batch latency, not real-time latency. We are not detecting faces on a live video stream; we are processing a batch of Street View images on a GPU cluster. Under these constraints, accuracy dominates speed. Two-stage detectors consistently achieve higher mAP, especially for small objects — and small, distant faces are exactly our hardest case. The "proposal → classify" decomposition allows the model to focus its classification capacity on plausible object regions rather than wasting capacity classifying background.

**Architecture Overview**

```
Input Image
    ↓
[Backbone: ResNet-50 + FPN]
    ↓
Feature Maps (multi-scale: P2, P3, P4, P5)
    ↓
[Stage 1: Region Proposal Network (RPN)]
    ↓
Proposals (ROI bounding boxes, ~2000 per image)
    ↓
[ROI Align]
    ↓
Fixed-size feature vectors per proposal
    ↓
[Stage 2: Classification + Regression Head]
    ↓
(class_label, confidence, refined_bbox) per proposal
    ↓
[Non-Maximum Suppression]
    ↓
Final detections
```

**Backbone: ResNet-50 with Feature Pyramid Network (FPN)**

The backbone is a ResNet-50 pretrained on ImageNet. ResNet-50 is a reasonable choice: deep enough to capture complex facial features, but computationally tractable. For a Google-scale deployment, we might upgrade to ResNet-101 or even a transformer-based backbone (Swin Transformer) if compute allows.

Raw ResNet-50 produces a single feature map at 1/32 of input resolution. At this resolution, a face that is 32x32 pixels in the original image maps to roughly 1x1 pixel in the feature map — the model literally cannot see it. This is why FPN is essential.

FPN adds a top-down pathway with lateral connections. Starting from the deepest ResNet layer (coarse spatial resolution, rich semantic features), FPN progressively upsamples and merges with earlier layers (fine spatial resolution, low-level features). This produces a pyramid of feature maps:
- P5: 1/32 scale (large objects)
- P4: 1/16 scale
- P3: 1/8 scale
- P2: 1/4 scale (small objects)

Each level in the pyramid is responsible for detecting objects of a specific size range. Small faces and license plates (< 64px) are detected at P2/P3; large objects at P4/P5. This dramatically improves small object detection.

**Stage 1: Region Proposal Network (RPN)**

The RPN is a small convolutional network that slides over each FPN feature map and predicts objectness scores and coarse bounding box offsets at every spatial location.

At each spatial location on a feature map, the RPN considers k anchor boxes — predefined rectangles of various scales and aspect ratios. For our problem, I would use 3 scales × 3 aspect ratios = 9 anchors per location. The aspect ratios (0.5, 1.0, 2.0) cover wide plates and tall faces. The scales are assigned per FPN level.

For each anchor, the RPN predicts:
1. An objectness score p ∈ [0,1]: "does this anchor contain an object (any class) vs. background?"
2. Four regression offsets (Δx, Δy, Δw, Δh) to refine the anchor into a better-fitting proposal

The regression offsets are parameterized relative to the anchor:
```
t_x = (x_gt - x_a) / w_a
t_y = (y_gt - y_a) / h_a
t_w = log(w_gt / w_a)
t_h = log(h_gt / h_a)
```

where (x_a, y_a, w_a, h_a) are anchor center and dimensions, and (x_gt, y_gt, w_gt, h_gt) are ground truth box center and dimensions.

During training, anchors with IoU > 0.7 with any ground truth box are labeled positive; anchors with IoU < 0.3 are labeled negative; anchors between 0.3 and 0.7 are ignored. We sample 256 anchors per image (128 positive, 128 negative) for the RPN loss.

**RPN Loss:**
```
L_RPN = (1/N_cls) * Σ L_cls(p_i, p_i*) + λ * (1/N_reg) * Σ p_i* * L_reg(t_i, t_i*)
```

where:
- p_i is the predicted objectness score for anchor i
- p_i* ∈ {0,1} is the ground truth label
- t_i is the predicted offset vector
- t_i* is the ground truth offset vector
- L_cls is binary cross-entropy
- L_reg is smooth L1 loss
- λ balances classification and regression terms (typically λ = 10)

**ROI Align**

The RPN produces ~2000 proposal boxes per image. We need to extract a fixed-size feature representation for each proposal to pass to the classification head.

The original Faster R-CNN used ROI Pooling, which rounds pool boundaries to integer pixel positions, introducing quantization error. ROI Align fixes this by using bilinear interpolation to sample feature map values at exact floating-point positions. This matters for small objects: for a 16x16 pixel face, quantization error in ROI Pooling can account for 10-20% of the object size.

ROI Align produces a 7x7 feature map for each proposal.

**Stage 2: Classification + Regression Head**

Each 7x7 proposal feature is passed through two fully-connected layers (2048 units each), then two parallel output heads:

1. **Classification head**: FC → softmax over {background, face, license_plate}
2. **Regression head**: FC → 4 * num_classes offsets (class-specific refinement)

**Stage 2 Loss:**
```
L_stage2 = L_cls + λ * L_reg
```

```
L_cls = -Σ y_k * log(p_k)   (cross-entropy)
```

```
L_reg = Σ_k y_k * smooth_L1(t_k - t_k*)
```

where smooth L1 is:
```
smooth_L1(x) = 0.5 * x²           if |x| < 1
             = |x| - 0.5          otherwise
```

Smooth L1 is preferred over L2 because outlier proposals (with large offset errors) do not dominate the gradient, making training more stable.

**Total Loss:**
```
L_total = L_RPN_cls + λ₁ * L_RPN_reg + L_stage2_cls + λ₂ * L_stage2_reg
```

**Training Details**

- Pretrain backbone on ImageNet, fine-tune end-to-end
- Optimizer: SGD with momentum 0.9, weight decay 1e-4
- Learning rate schedule: warm-up for 500 steps, then step decay at 60k and 80k iterations
- Batch size: 2 images per GPU (memory-constrained by high-resolution images), 8 GPUs
- Total training: ~90k iterations on 1M images

**Training-Serving Skew Prevention**

A critical issue: the preprocessing applied at training time (resize, normalize, augmentation) must be exactly replicated at serving time. I would:
1. Package all preprocessing logic as a shared library used by both the training pipeline and the serving pipeline
2. Log a random sample of serving inputs and their preprocessed versions, and compare against training distribution statistics
3. Version the preprocessing configuration alongside the model weights

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How would you evaluate this system, both offline and online?"

### Signal Being Tested

Does the candidate understand IoU as a threshold with sensitivity, mAP calculation across classes and recall levels, and the distinction between offline model evaluation and online business metrics? Do they understand the privacy-specific nuance of recall priority?

### Follow-up Probes

- "Why is IoU threshold choice significant? What happens if you lower it to 0.5?"
- "Walk me through the mAP calculation step by step."
- "How would you design the online evaluation pipeline to catch regressions?"

---

### Model Answers — Section 5

**No Hire:**
"I would use accuracy." Accuracy is meaningless for object detection with class imbalance. No mention of IoU, mAP, or online metrics.

**Lean No Hire:**
Mentions IoU and precision/recall but cannot explain mAP calculation. Does not distinguish online vs. offline metrics. Does not address the privacy-specific recall requirement.

**Lean Hire:**
Correctly defines IoU, explains precision-recall tradeoff, describes mAP as area under the precision-recall curve averaged over classes. Mentions user report rate as an online metric. Notes the high-recall requirement for privacy.

**Strong Hire (500+ words, first-person):**

Evaluation for this system has three distinct layers: per-detection correctness (IoU), aggregate model performance (mAP), and real-world system performance (online metrics). Let me walk through each.

**Layer 1: Intersection over Union (IoU)**

IoU measures the overlap between a predicted bounding box P and a ground truth bounding box G:

```
IoU(P, G) = Area(P ∩ G) / Area(P ∪ G)
```

More explicitly:
```
Area(P ∩ G) = max(0, min(x2_P, x2_G) - max(x1_P, x1_G)) 
            × max(0, min(y2_P, y2_G) - max(y1_P, y1_G))

Area(P ∪ G) = Area(P) + Area(G) - Area(P ∩ G)
```

A detection is counted as a **true positive** if IoU ≥ threshold; otherwise it is a **false positive**. An unmatched ground truth box is a **false negative**.

**IoU Threshold Sensitivity:**

I would use IoU = 0.7 as the primary threshold. Here is why this choice matters:

- IoU = 0.5 (PASCAL VOC standard) accepts a prediction that covers only 50% of the ground truth box. For blurring faces, a prediction that covers 50% of a face leaves part of the face visible. This is unacceptable from a privacy standpoint.
- IoU = 0.7 requires substantial overlap — the predicted box must capture at least 70% of the ground truth box area. This is a meaningful privacy threshold.
- IoU = 0.9 (very tight) would penalize predictions that are essentially correct but slightly off in their exact boundaries. This is unnecessarily strict for blurring (a slightly larger box is fine).

I would also report metrics at IoU = 0.5 and IoU = 0.75 to understand the model's sensitivity, but IoU = 0.7 is the primary threshold for deployment decisions.

**Layer 2: Mean Average Precision (mAP)**

For each class c ∈ {face, license_plate}, I compute Average Precision (AP_c) as follows:

1. Sort all predictions (across all test images) by descending confidence score.
2. Accumulate predictions one by one, computing precision and recall at each step:
   ```
   Precision(k) = TP(k) / (TP(k) + FP(k))
   Recall(k)    = TP(k) / (TP(k) + FN)
   ```
3. Plot the precision-recall curve.
4. Compute AP as the area under this curve using the 11-point interpolation (PASCAL VOC) or all-point interpolation (COCO):
   ```
   AP = (1/11) * Σ_{r ∈ {0, 0.1, ..., 1.0}} max_{r' ≥ r} P(r')
   ```

5. Average over classes:
   ```
   mAP = (1/C) * Σ_{c=1}^{C} AP_c
   ```

With C = 2 classes (face and license plate): mAP = (AP_face + AP_license_plate) / 2.

**Why mAP and not accuracy or F1?**

mAP integrates over the full range of confidence thresholds, giving a complete picture of the precision-recall trade-off. F1 at a single threshold tells us performance only at that operating point. Since we tune the confidence threshold post-training (we can lower it to increase recall at the cost of precision), mAP better characterizes model quality independently of the chosen operating threshold.

**Class-Specific Reporting:**

I would report AP_face and AP_license_plate separately, not just aggregate mAP. License plates tend to be easier to detect (rigid shape, high contrast) but harder to precisely localize for unusual formats. Faces are harder overall due to occlusion, lighting, and pose variation. Separate reporting lets us identify class-specific degradations.

**Recall Prioritization:**

Given the privacy requirement, I want to explicitly evaluate at the high-recall operating point. After training, I select the confidence threshold that achieves recall ≥ 0.99 on the validation set, then report precision at that threshold. Operationally:

```
Threshold_deploy = min{s : Recall(s) ≥ 0.99}
Precision_at_deploy = Precision(Threshold_deploy)
```

The false positive rate at this threshold may be significant (many non-face regions blurred), but this is the acceptable trade-off.

**Layer 3: Online Evaluation Metrics**

- **User report rate**: Number of privacy complaints (unblurred PII reports) per million page views. This is the primary online metric — a direct signal of false negatives reaching users.
- **Valid appeal rate**: Of user reports flagged as "wrongly blurred" (false positives), what fraction are confirmed valid by human reviewers? A high false positive rate erodes user experience, especially for businesses whose signage is incorrectly blurred.
- **Manual spot-check accuracy**: Weekly sample of 500 randomly selected images, reviewed by annotators. Compare ML blur coverage against human-labeled faces/plates.
- **Geographic breakdown**: Decompose user reports by country/region to detect model degradation on international plates or underrepresented populations.

**Monitoring and Alerting**

I would set up automated alerts on:
- User report rate spike > 2x baseline (potential model regression)
- Prediction confidence distribution shift (serving inputs drifting from training distribution)
- Processing pipeline throughput (operational health)

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "How would you design the serving pipeline? Walk me through the full lifecycle of a Street View image from capture to publication."

### Signal Being Tested

Does the candidate understand batch pipeline design, the role of NMS, the user feedback loop, and how corrections feed back into retraining? Can they identify failure points in the pipeline?

### Follow-up Probes

- "Walk me through Non-Maximum Suppression — why is it needed and how does it work?"
- "How would you handle the case where a new batch of images comes from a country with a novel license plate format?"
- "How would you implement the user correction pipeline end-to-end?"

---

### Model Answers — Section 6

**No Hire:**
Describes a single forward pass through the model. No mention of NMS, batch processing, pipeline stages, or user feedback.

**Lean No Hire:**
Describes a linear pipeline but omits NMS or cannot explain it. Does not connect user reports to retraining.

**Lean Hire:**
Describes the full pipeline: preprocess → detect → NMS → blur → store. Explains NMS correctly. Mentions user report queue and retraining loop. Identifies at least one pipeline failure mode.

**Strong Hire (500+ words, first-person):**

Let me describe the serving architecture in two parts: the primary batch processing pipeline and the user correction pipeline.

**Primary Batch Processing Pipeline**

```
[Street View Capture System]
    ↓ (raw images + metadata)
[Image Ingestion Queue (Pub/Sub or Kafka)]
    ↓
[Preprocessing Workers]
    - Resize to model input dimensions
    - Normalize pixel values
    - Validate image format and dimensions
    ↓
[Inference Cluster (GPU fleet)]
    - Batch images: batch_size = 4-8 per GPU
    - Forward pass through Faster R-CNN + FPN
    - Output: raw (bbox, class, confidence) tuples per image
    ↓
[Post-processing Workers]
    - Non-Maximum Suppression (NMS)
    - Confidence threshold filtering
    - Bounding box padding (privacy margin)
    ↓
[Blur Application Workers]
    - For each accepted bounding box, apply Gaussian blur
    - Write blurred image to output storage
    ↓
[Output Storage (GCS)]
    ↓
[Publication Pipeline]
    - Images served to Street View users
```

**Non-Maximum Suppression (NMS) — Detailed Mechanism**

The detector produces many overlapping proposals, because the RPN generates ~2000 proposals per image and many will fire on the same face from different anchor positions. Without NMS, the same face might have 50 overlapping detections, each slightly different.

NMS algorithm per class:
1. Sort all detections by confidence score (descending).
2. Select the highest-confidence detection d₁; add it to the output set.
3. Compute IoU(d₁, dᵢ) for all remaining detections dᵢ.
4. Remove any dᵢ where IoU(d₁, dᵢ) > NMS_threshold (typically 0.5). These are suppressed as duplicates.
5. Repeat with the next highest-confidence detection among remaining candidates.
6. Continue until no candidates remain.

NMS is applied per class independently. A face prediction and a license plate prediction may legitimately overlap (e.g., someone holding a plate in front of them), so cross-class suppression is not applied.

**Soft-NMS** is an improvement that reduces confidence scores of overlapping detections rather than hard-removing them:
```
s_i = s_i * e^(-IoU(d₁, dᵢ)² / σ)
```
This avoids removing true positives in dense crowd scenes where two faces genuinely overlap in the image. For our use case, faces in crowded scenes (protests, events) are exactly the cases where we most need high recall.

**Throughput Estimation**

Approximate numbers for infrastructure sizing:
- Street View captures approximately 10 million new images per day globally
- Faster R-CNN + FPN on a T4 GPU: ~4 images/second at batch size 4
- 10M images / 86400 seconds = ~115 images/second required throughput
- Required GPUs: 115 / 4 ≈ 29 GPUs minimum; with safety margin and redundancy: ~50 T4 GPUs
- At Google scale this is trivially achievable; the real constraint is network I/O from GCS

**User Correction Pipeline**

Street View provides a "Report a problem" feature. Users can flag:
1. **Unblurred PII** (missed face or license plate) — false negative
2. **Incorrectly blurred region** (non-PII blurred, e.g., a sign) — false positive

The correction pipeline:

```
[User Report Form]
    ↓
[Report Queue]
    ↓
[Automated Triage]
    - Filter spam (reported region has no bounding box overlap with any detection)
    - Prioritize by report frequency per image location
    ↓
[Human Review Queue (1-5% of reports)]
    - Annotators confirm: valid report or not?
    - If valid: add to correction dataset
    ↓
[Correction Dataset]
    - Maintain a separate collection of correction examples
    - Weight corrections more heavily in next training run
    ↓
[Retraining Trigger]
    - Accumulate 50,000 corrections → trigger retraining
    - Or: user report rate exceeds threshold → emergency retraining
```

**Automated Emergency Blurring**

For high-priority unblurred PII reports (confirmed by automated heuristics or human review), the image should be taken offline within minutes:
1. Report flagged as high-priority
2. Image URL added to a blocklist — 404 is served for that image
3. Manual blur applied by human annotator
4. Image re-published with human-verified blur

This ensures that confirmed missed PII is not served while the model is being retrained.

**Model Versioning and Rollout**

I would maintain model versions with shadow deployment:
- New model version runs in shadow mode on 5% of traffic
- Compare shadow predictions against production predictions
- If shadow mAP on recent data significantly exceeds production: gradual rollout (5% → 25% → 50% → 100%)
- Maintain easy rollback: production prediction store keeps predictions from both current and previous model version for 30 days

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the most important edge cases and failure modes for this system? How would you address them?"

### Signal Being Tested

Does the candidate proactively identify failure modes beyond the obvious? A staff engineer should enumerate at least 5 distinct failure modes with mitigation strategies.

---

### Model Answers — Section 7

**No Hire:**
Mentions "bad lighting" vaguely. No concrete mitigations.

**Lean No Hire:**
Identifies 2-3 failure modes (occlusion, night images) without mitigations or without connecting them to system design changes.

**Lean Hire:**
Identifies 4-5 failure modes with mitigations. Shows awareness that failure modes have different severity (privacy risk vs. UX degradation).

**Strong Hire:**
Identifies 6+ failure modes, categorizes them by severity, and proposes both model-level and system-level mitigations. Thinks about adversarial cases and international diversity.

**Detailed Failure Mode Analysis:**

**1. Occluded Faces**
- **Description**: Face partially blocked by an object (sunglasses, hat, hand, another person). The model may not recognize a heavily occluded face.
- **Severity**: High — a partially visible face is still identifiable.
- **Mitigation**: Augment training data with synthetic occlusion (random rectangular masks over faces). Include crowdsourced hard-negative examples from correction pipeline. Consider using a lower confidence threshold for face class specifically.

**2. Small Faces / Long-Range Detection**
- **Description**: People far from the camera appear as very small (< 16px) faces. The model's smallest anchor boxes may not capture these.
- **Severity**: Medium — very distant faces may not be identifiable, but blur should still be applied for safety.
- **Mitigation**: FPN with P2 layer captures small objects. Multi-scale test-time augmentation: run inference at 2x scale as well, then merge predictions. Add a separate post-processing step that runs a dedicated small-face detector on regions of the image where people are expected (based on detected body parts or shadows).

**3. Unusual Lighting Conditions**
- **Description**: Night scenes with artificial lighting, harsh direct sunlight creating overexposed regions, severe motion blur from vehicle movement.
- **Severity**: Medium — harder to detect in these conditions, but same privacy risk.
- **Mitigation**: Aggressive brightness/contrast augmentation during training. Collect dedicated hard examples from nighttime captures. Consider a separate nighttime-specialized model or domain adaptation.

**4. International License Plate Formats**
- **Description**: License plates vary dramatically by country. Japanese kana plates, European narrow plates, Chinese plates with different aspect ratios — a model trained on US data will generalize poorly.
- **Severity**: High in international markets.
- **Mitigation**: Build a per-region training data collection strategy. Use geographic metadata (lat/lng) to know which country the image was captured in, and route to a country-specific model or apply a country-specific confidence threshold. Use few-shot learning or domain adaptation when labeled data for a new country is limited.

**5. Reflections and Indirect Views**
- **Description**: Faces visible in shop windows, car mirrors, puddles. The person is not in the image directly, but their face is visible.
- **Severity**: High — these faces are fully identifiable.
- **Mitigation**: This is hard. Reflections have different statistical properties (flipped, distorted, lower contrast). Annotate a small set of reflection examples and fine-tune. At minimum, monitor user reports specifically tagged as "face in reflection."

**6. Adversarial Blurring Requests**
- **Description**: A malicious actor submits repeated "wrongly blurred" reports for legitimate blur-outs (e.g., reporting a correctly blurred license plate as "incorrect" to try to get it published unblurred).
- **Severity**: High — this would undermine the entire correction pipeline.
- **Mitigation**: Human review for any report claiming a currently-blurred region should be unblurred. Rate-limit correction requests per user. Track user report validity history — users with high invalid report rates get deprioritized.

**7. Crowd Scenes and Occlusion Chains**
- **Description**: In a crowded scene with 50+ people, some faces are behind others. Dense overlap may confuse NMS — suppressing a real face as a duplicate of a nearby face.
- **Severity**: High.
- **Mitigation**: Use Soft-NMS instead of hard NMS. For crowd scenes (detected by scene classifier or detected count > threshold), lower the NMS IoU threshold to be more conservative about suppression.

**8. Model Staleness for New Vehicle Models**
- **Description**: A new car model with an unusual front-end design (or temporary plates) may have a license plate configuration unlike anything in training data.
- **Severity**: Low to medium.
- **Mitigation**: User report pipeline acts as a continuous sensor for novel cases. Monitor user report rate by vehicle type (can be inferred from context).

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "Step back from this specific problem. How would you build this capability as a platform that serves multiple privacy-related products at Google, not just Street View?"

### Signal Being Tested

This is the differentiating question for principal-level. Can the candidate generalize from a specific solution to a platform that generates leverage? Do they think about organizational impact, API design, and shared infrastructure?

---

### Model Answers — Section 8

**No Hire / Lean No Hire:**
Restates the Street View design with slightly different inputs. Does not demonstrate platform thinking.

**Lean Hire:**
Mentions a shared model that other products could use, or a shared annotation pipeline. Does not fully articulate the API contract, operational model, or organizational implications.

**Strong Hire (500+ words, first-person):**

What I have described so far is a good solution for Street View specifically. But at the principal level, I want to ask: how many other Google products have a privacy blurring problem?

Google Photos has 4 trillion photos. Google Maps has user-contributed imagery. YouTube has billions of videos. Google Meet has live video. Workspace has document scanning (Lens). In each case, there is a need to detect PII in visual content — faces, license plates, sensitive documents, medical images — and either blur, suppress, or alert on it.

Building a privacy blurring model per product creates massive duplication: duplicated training pipelines, duplicated annotation effort, duplicated model serving infrastructure, duplicated compliance review. More dangerously, it creates inconsistent privacy guarantees across products — a face detected by Street View's model might be missed by Maps's independent model, creating inconsistent user privacy protection across the same company.

The principal-level opportunity is to build a **Visual Privacy Platform** with the following components:

**1. Shared PII Detection Model Zoo**

A centralized model registry with:
- A general-purpose PII detector (faces + license plates + text + documents)
- Specialized models for specific domains (medical images, financial documents)
- A model versioning and lifecycle management system
- Common evaluation benchmarks across all models (so cross-model comparisons are meaningful)

Products consume model endpoints via a standard API:
```
POST /privacy/detect
{
  "image_url": "...",
  "detect_types": ["face", "license_plate", "text"],
  "recall_threshold": 0.99
}

Response:
{
  "detections": [
    {"type": "face", "bbox": {...}, "confidence": 0.97},
    ...
  ],
  "model_version": "v4.2.1"
}
```

**2. Shared Annotation Pipeline**

A centralized annotation platform where:
- All products contribute images to a shared annotation queue
- Specialized annotators develop expertise in PII detection across modalities
- Cross-product annotation reuse: a face annotated for Street View can be used to train the Photos model
- Consistent annotation guidelines enforced across all labeling work (standardized bounding box definition, handling of ambiguous cases)

This dramatically reduces annotation cost — instead of each product paying for their own annotation, they contribute to and benefit from a shared pool.

**3. Privacy Compliance Integration**

A critical platform capability is compliance audit logging:
- Every detection call is logged with model version, input image hash, and detection results
- This log is queryable by legal/compliance teams: "show all images processed containing faces in region X on date Y"
- GDPR Right to be Forgotten: when a user requests deletion of their data, the audit log enables us to identify which published images may contain their face, triggering a review

**4. Federated Retraining**

Different products have different data distributions. The platform needs to support:
- A shared base model trained on common data
- Product-specific fine-tuning layers trained on product-specific data
- Federated contributions: when Street View's correction pipeline produces 50,000 new annotations, those improvements propagate to the shared model, which benefits all products

**5. Privacy-by-Default Policy Enforcement**

Rather than individual product teams making ad hoc decisions about when to blur, the platform enforces policy:
- All user-generated content containing faces must be processed through the privacy pipeline before publication
- Product teams opt into specific detection types based on their use case
- Exceptions require explicit privacy review and approval

This transforms privacy from a feature that individual teams remember to implement into a platform constraint that is automatically enforced.

The organizational impact is significant: a team of 10 ML engineers owning the Visual Privacy Platform can serve the privacy needs of hundreds of product teams, at a higher quality level than each team could achieve independently, while creating a single compliance surface for legal and regulatory requirements.

---

## Section 9: Appendix — Key Formulas & Reference

### Core Formulas

**Intersection over Union:**
```
IoU(P, G) = Area(P ∩ G) / Area(P ∪ G)
```

**Precision and Recall:**
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

**Average Precision (11-point interpolation):**
```
AP = (1/11) * Σ_{r ∈ {0, 0.1, 0.2, ..., 1.0}} max_{r' ≥ r} Precision(r')
```

**Mean Average Precision:**
```
mAP = (1/C) * Σ_{c=1}^{C} AP_c
```

**RPN Regression Targets:**
```
t_x = (x_gt - x_a) / w_a
t_y = (y_gt - y_a) / h_a
t_w = log(w_gt / w_a)
t_h = log(h_gt / h_a)
```

**Smooth L1 Loss:**
```
smooth_L1(x) = { 0.5 * x²         if |x| < 1
               { |x| - 0.5        otherwise
```

**Total Loss (Faster R-CNN):**
```
L = L_RPN_cls + λ₁ * L_RPN_reg + L_det_cls + λ₂ * L_det_reg
```

**Soft-NMS Score Decay:**
```
s_i ← s_i * exp(-IoU(M, b_i)² / σ)
```

### Reference Numbers

| Parameter | Value |
|---|---|
| Training images | 1,000,000 |
| Primary IoU threshold | 0.7 |
| Target recall | ≥ 0.99 |
| Inference throughput (T4 GPU) | ~4 images/sec |
| Daily image volume | ~10M images |
| Required GPU fleet | ~50 T4 GPUs |
| NMS IoU threshold | 0.5 (hard) |
| Soft-NMS σ | 0.5 |
| RPN proposals per image | 2,000 |
| Training resolution (short side) | 800 px |
| Max resolution (long side) | 1,333 px |
| ResNet stages | C2–C5 |
| FPN output levels | P2–P5 |
| Anchor aspect ratios | 0.5, 1.0, 2.0 |
| Anchors per location | 9 |
| ROI Align output size | 7×7 |
| FC layer size (head) | 2,048 |

### Architecture Comparison

| Model Type | Example | Speed | mAP | Best For |
|---|---|---|---|---|
| One-stage | YOLOv8, RetinaNet | 30–100 FPS | Moderate | Real-time, latency-critical |
| Two-stage | Faster R-CNN | 5–15 FPS | High | Accuracy-critical, batch |
| Anchor-free | FCOS, CenterNet | 15–30 FPS | Moderate-High | Flexible shapes |

### Failure Mode Summary

| Failure Mode | Severity | Mitigation |
|---|---|---|
| Occluded faces | High | Synthetic occlusion augmentation |
| Small faces (< 32px) | High | FPN P2, multi-scale inference |
| Night / unusual lighting | Medium | Brightness augmentation, domain adaptation |
| International plates | High | Geographic routing, per-region fine-tuning |
| Reflections | High | Annotate reflection examples, monitor reports |
| Adversarial reports | High | Human review for unblur requests |
| Crowd scene NMS errors | High | Soft-NMS, lower NMS threshold for dense scenes |

### Evaluation Framework

| Metric | Type | Primary Use |
|---|---|---|
| mAP @ IoU=0.7 | Offline | Model comparison, deployment decision |
| AP_face | Offline | Class-specific analysis |
| AP_license_plate | Offline | Class-specific analysis |
| Recall @ 0.99 threshold | Offline | Privacy compliance gate |
| User report rate | Online | Primary production health signal |
| Valid appeal rate | Online | False positive monitoring |
| Manual spot-check accuracy | Online | Weekly regression detection |

