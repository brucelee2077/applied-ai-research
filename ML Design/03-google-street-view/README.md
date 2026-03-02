# Google Street View Blurring System -- ML System Design Interview Guide

> **Source**: ByteByteGo -- Machine Learning System Design Interview, Chapter 03
>
> **Philosophy**: Every concept explained simply enough for a 12-year-old, with staff-level technical depth underneath.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Clarifying Requirements](#clarifying-requirements)
3. [Framing the Problem as an ML Task](#framing-the-problem-as-an-ml-task)
4. [Data Preparation](#data-preparation)
5. [Model Development](#model-development)
6. [Evaluation](#evaluation)
7. [Serving](#serving)
8. [Other Talking Points](#other-talking-points)
9. [Interview Cheat Sheet](#interview-cheat-sheet)

---

## The Big Picture

### Simple Explanation (for a 12-year-old)

Imagine Google has special cars that drive around every street in the world, taking pictures. These pictures go on Google Maps so anyone can see what a street looks like before they visit. But here is the problem: the pictures capture people's faces and car license plates. That is private information! So Google needs a computer program that can look at millions of pictures, find every face and every license plate, and blur them out -- like putting a fuzzy circle over them -- before anyone sees the photos.

### Staff-Level Technical Summary

Google Street View's blurring system is an **offline batch-processing object detection pipeline** that identifies and obscures personally identifiable information (PII) -- specifically human faces and license plates -- in street-level panoramic imagery. The system prioritizes **recall** (catching every face/plate) over precision, because a missed face is a privacy violation, while a false positive blur is merely a cosmetic issue. The system operates offline, meaning latency is not a concern. Existing images are displayed to users while new images are processed in the background.

---

## Clarifying Requirements

In an interview, the first thing you do is ask questions. Here is the conversation structure from the PDF:

| Question | Answer | Why It Matters |
|----------|--------|----------------|
| Is the business objective to protect user privacy? | Yes | Establishes the "why" -- this is not about aesthetics, it is about legal/ethical obligations |
| We detect faces + license plates, then blur before display? | Yes | Defines the two-class detection problem |
| Users can report incorrectly blurred images? | Yes | Creates a feedback loop and defines an online metric |
| Do we have annotated data? | 1 million images with bounding box annotations | Tells us supervised learning is feasible |
| Dataset bias (race, age, gender)? | Good point, but out of scope today | Shows awareness of fairness issues |
| Latency constraints? | Not a concern -- offline processing | Unlocks batch prediction, no need for real-time inference |

### Key Takeaways from Requirements

- **Business objective**: Protect user privacy
- **ML objective**: Accurately detect objects of interest (faces, license plates) in images
- **Dataset**: 1 million annotated images with bounding boxes
- **Processing mode**: Offline (batch) -- no real-time latency requirements
- **Feedback mechanism**: Users can report missed blurs

---

## Framing the Problem as an ML Task

### Simple Explanation

Think of it like a game of "I Spy." You show the computer a photo and say: "Find all the faces and license plates." The computer draws rectangles around each one it finds (these rectangles are called "bounding boxes"). Then a blurring tool smudges everything inside those rectangles.

### Defining the ML Objective

- **Business objective**: Protect user privacy (NOT directly an ML objective)
- **ML objective**: Accurately detect objects of interest in an image
- **Translation**: If we can detect objects accurately, we can blur them before display

### Input and Output

- **Input**: A street-level image (potentially containing zero or more faces/license plates)
- **Output**: A list of bounding boxes, each with:
  - Position: (x, y) coordinates of the top-left corner
  - Size: width and height
  - Class: "human_face" or "license_plate"
  - Confidence score

### ML Category: Object Detection

Object detection has **two sub-tasks**:

1. **Localization (Regression)**: Predict the (x, y, w, h) coordinates of each bounding box -- these are continuous numeric values, so this is a regression problem.
2. **Classification (Multi-class)**: Predict which class each bounding box belongs to (face vs. license plate vs. background) -- this is a classification problem.

### Architecture Families

#### Two-Stage Networks

**Simple explanation**: Imagine you are looking for your friend in a crowded stadium. First, you scan the whole stadium and pick out sections where you think people might be (stage 1). Then, you look closely at each section to see if your friend is actually there (stage 2).

**Technical detail**:
1. **Region Proposal Network (RPN)**: Scans the image and proposes candidate regions likely to contain objects
2. **Classifier**: Processes each proposed region and classifies it into an object class

**Examples**: R-CNN, Fast R-CNN, Faster R-CNN

**Characteristics**: Slower but more accurate. The two components run sequentially.

#### One-Stage Networks

**Simple explanation**: Instead of scanning then looking closely, you look at the whole picture all at once and immediately say "There is a face here, a license plate there."

**Technical detail**: Both bounding box prediction and classification happen simultaneously in a single forward pass, without explicit region proposals.

**Examples**: YOLO (You Only Look Once), SSD (Single Shot MultiBox Detector)

**Characteristics**: Faster but potentially less accurate.

#### Transformer-Based (DETR)

A newer approach using Transformer architectures (the same family behind GPT and BERT). DETR (DEtection TRansformer) treats object detection as a set prediction problem. Promising results but not the main focus of this chapter.

#### Which One Do We Choose?

For this problem: **Two-stage network** (e.g., Faster R-CNN).

**Rationale**:
- Dataset is 1M images -- not huge by modern standards, so training cost is manageable
- Privacy is the primary concern, so we prioritize accuracy over speed
- Processing is offline, so slower inference is acceptable
- If training data grows or we need faster predictions later, we can switch to one-stage

---

## Data Preparation

### Data Engineering

#### Available Data

**1. Annotated Dataset (1 million images)**

Each image has a list of bounding boxes and associated object classes:

| Image Path | Objects | Bounding Boxes |
|------------|---------|---------------|
| dataset/image1.jpg | human_face | [10, 10, 25, 50] |
| dataset/image1.jpg | human_face | [120, 180, 40, 70] |
| dataset/image1.jpg | license_plate | [80, 95, 35, 10] |
| dataset/image2.jpg | human_face | [170, 190, 30, 80] |

Each bounding box is `[top_left_x, top_left_y, width, height]`.

**2. Street View Images (unlabeled production data)**

| Image Path | Location (lat, lng) | Pitch, Yaw, Roll | Timestamp |
|-----------|-------------------|-----------------|-----------|
| tmp/image1.jpg | (37.432567, -122.143993) | (0, 10, 20) | 1609459200 |
| tmp/image2.jpg | (37.387843, -122.091086) | (0, 10, -10) | 1609459200 |

### Feature Engineering

#### Standard Preprocessing
- **Resizing**: Scale images to a consistent input size (e.g., 800x800 or 1024x1024)
- **Normalization**: Scale pixel values to [0, 1] or standardize to zero mean and unit variance

#### Data Augmentation

**Simple explanation**: Imagine you have one photo of a cat. If you flip it, rotate it slightly, make it brighter, and make it darker, you now have five photos of a cat. This helps the computer learn to recognize cats in many different conditions.

**Technical detail**: Data augmentation creates modified copies of original data to increase dataset size and improve model generalization. Especially useful for imbalanced datasets.

**Common augmentation techniques for images**:
- Random crop
- Random saturation
- Vertical or horizontal flip
- Rotation and/or translation
- Affine transformations
- Changing brightness, saturation, or contrast

**Critical note**: When augmenting images, the ground truth bounding boxes must be transformed accordingly. If you flip an image, the bounding box coordinates must also be flipped.

#### Online vs. Offline Augmentation

| Approach | Pros | Cons |
|----------|------|------|
| **Offline** (augment before training) | Faster training -- no augmentation overhead during training | Requires additional storage for all augmented images |
| **Online** (augment on-the-fly during training) | No additional storage needed; more variety per epoch | Slower training due to augmentation overhead |

**Decision**: Offline data augmentation (in this design).

**Result**: Dataset grows from 1 million to ~10 million images after augmentation.

### Dataset Preparation Flow

```
Raw Images (1M) --> Preprocessing (resize, scale, normalize) --> Augmentation --> Augmented Dataset (10M)
```

---

## Model Development

### Two-Stage Architecture Components

#### 1. Convolutional Layers (Backbone)
- Processes the input image and outputs a **feature map**
- The feature map is a compressed representation of the image that captures spatial patterns
- Common backbones: ResNet-50, ResNet-101, VGG-16

#### 2. Region Proposal Network (RPN)
- Takes the feature map as input
- Outputs candidate regions (bounding box proposals) that may contain objects
- Uses neural networks internally
- Generates hundreds to thousands of proposals per image

#### 3. Classifier (Detection Head)
- Takes the feature map AND proposed candidate regions as input
- Assigns an object class to each region
- Also refines the bounding box coordinates
- Based on neural networks

### Loss Functions

Object detection requires **two loss functions** because there are two tasks:

#### Regression Loss (Bounding Box Alignment)

Measures how well predicted bounding boxes align with ground truth. Uses Mean Squared Error (MSE):

```
L_reg = (1/M) * SUM_i [(x_i - x_hat_i)^2 + (y_i - y_hat_i)^2 + (w_i - w_hat_i)^2 + (h_i - h_hat_i)^2]
```

Where:
- `M` = total number of predictions
- `(x_i, y_i)` = ground truth top-left coordinates
- `(x_hat_i, y_hat_i)` = predicted top-left coordinates
- `(w_i, h_i)` = ground truth width and height
- `(w_hat_i, h_hat_i)` = predicted width and height

#### Classification Loss (Object Class Accuracy)

Measures how accurate the predicted class probabilities are. Uses cross-entropy (log loss):

```
L_cls = -(1/M) * SUM_i SUM_c [y_ic * log(y_hat_ic)]
```

Where:
- `M` = total number of detected bounding boxes
- `C` = total number of classes
- `y_ic` = ground truth label for detection i, class c
- `y_hat_ic` = predicted class probability for detection i, class c

#### Combined Loss

```
L = L_cls + lambda * L_reg
```

Where `lambda` is a balancing parameter that controls the relative importance of classification vs. localization accuracy.

---

## Evaluation

### Simple Explanation

How do we know if our computer is doing a good job finding faces? We need a scoring system. Imagine the computer draws boxes around things it thinks are faces. We compare those boxes to the correct answers (drawn by humans). If the computer's box overlaps a lot with the correct box, it gets a point.

### Intersection Over Union (IoU)

**Simple explanation**: If you draw a rectangle and your friend draws a rectangle on the same picture, IoU measures how much they overlap. If they are exactly the same, IoU = 1 (perfect). If they do not overlap at all, IoU = 0 (terrible).

**Formula**:
```
IoU = Area of Overlap / Area of Union
```

**How it determines correctness**: Set an IoU threshold (e.g., 0.5). If a predicted bounding box has IoU >= 0.5 with a ground truth box, it counts as a correct detection (true positive). Otherwise, it is a false positive.

### Offline Metrics

#### 1. Precision

```
Precision = Correct Detections / Total Detections
```

**Example from the PDF** (6 total detections, 2 ground truth objects):

| IoU Threshold | Correct Detections | Precision |
|--------------|-------------------|-----------|
| 0.7 | 2 | 2/6 = 0.33 |
| 0.5 | 3 | 3/6 = 0.50 |
| 0.1 | 4 | 4/6 = 0.67 |

**Problem**: Precision changes depending on the IoU threshold you pick. This makes it hard to understand overall performance.

#### 2. Average Precision (AP)

**Solution**: Compute precision across various IoU thresholds and average them.

```
AP = integral from 0 to 1 of P(r) dr
```

This can be approximated by a discrete summation. For example, Pascal VOC 2008 uses 11 evenly-spaced thresholds:

```
AP = (1/11) * SUM_{n=0}^{10} P(n)
```

AP summarizes the model's overall precision for a **specific object class** (e.g., human faces).

#### 3. Mean Average Precision (mAP)

```
mAP = (1/C) * SUM_{c=1}^{C} AP_c
```

Where C = total number of object classes.

**mAP** is the gold standard metric for object detection. It averages AP across ALL object classes, giving a single number that represents the model's overall detection performance.

### Online Metrics

Since the business objective is privacy protection:

- **User reports/complaints**: Count how many users report missed blurs -- this is the primary online metric
- **Human annotator spot-checks**: Percentage of incorrectly blurred images found by manual review
- **Bias metrics** (out of scope but important): Measure whether face detection works equally well across different races, ages, and genders

### Summary of Metrics

| Type | Metric | What It Measures |
|------|--------|-----------------|
| Offline | mAP | Overall model detection performance across all classes |
| Offline | AP | Per-class detection performance |
| Offline | Precision@IoU | Detection accuracy at a specific overlap threshold |
| Online | User reports | Real-world privacy protection quality |

---

## Serving

### Non-Maximum Suppression (NMS)

#### Simple Explanation

When the computer looks for faces, it often gets excited and draws many overlapping rectangles around the same face. We need a way to pick just the best rectangle for each face and throw away the duplicates. That is what NMS does.

#### Technical Detail

NMS is a **post-processing algorithm** that removes redundant overlapping bounding boxes:

1. Sort all detections by confidence score (highest first)
2. Pick the detection with the highest confidence -- keep it
3. Calculate IoU between this detection and all remaining detections
4. Remove any detection with IoU above a threshold (e.g., 0.5) -- these are duplicates
5. Repeat from step 2 with the remaining detections
6. Stop when no detections remain

**NMS is a very commonly asked algorithm in ML system design interviews.** Be prepared to explain it step by step.

### ML System Design (Overall Architecture)

The system has two main pipelines:

#### 1. Batch Prediction Pipeline

```
Raw Street View Images
        |
        v
  [Preprocessing]  (CPU-bound: resize, normalize, scale)
        |
        v
  [Blurring Service]  (GPU-bound)
        |-- Object detection model --> list of detected objects
        |-- NMS --> refined list (remove overlapping boxes)
        |-- Blur detected regions
        |-- Store blurred image in object storage
        |
        v
  Blurred Street View Images (served to users)
```

**Why separate preprocessing from blurring?**
- Preprocessing is **CPU-bound** (resizing, normalization)
- Blurring service uses **GPU** (neural network inference)
- Separating them allows:
  - Independent scaling based on each service's workload
  - Better utilization of CPU and GPU resources

#### 2. Data Pipeline (Feedback Loop)

```
User Reports (missed blurs)
        |
        v
  [Process Reports] --> [Generate New Training Data]
        |
        v
  [Hard Negative Mining]
        |
        v
  [Prepare Training Data] --> Retrain Model
```

#### Hard Negative Mining

**Simple explanation**: The computer makes mistakes -- sometimes it looks at a tree trunk and thinks it is a person. We take these mistakes, label them as "NOT a face," and add them to the training data. Next time, the computer will learn "oh, a tree trunk is not a face."

**Technical detail**: Hard negatives are examples explicitly created from incorrectly predicted examples. These are cases where the model was confidently wrong. By adding them to the training dataset and retraining, the model learns to avoid these specific failure modes.

---

## Other Talking Points

If time allows in the interview, discuss these advanced topics:

1. **Transformer-based detection (DETR)**: How it differs from one-stage/two-stage models. It treats detection as a set prediction problem using bipartite matching loss. Pros: no need for NMS, anchor boxes, or hand-designed components. Cons: slower to converge, needs more data.

2. **Distributed training**: Techniques for training on larger datasets across multiple GPUs/machines (data parallelism, model parallelism). Relevant frameworks: PyTorch DistributedDataParallel, TensorFlow distributed strategies.

3. **GDPR compliance**: The European General Data Protection Regulation may impose additional requirements on how images are collected, stored, and processed. May require opt-out mechanisms.

4. **Bias in face detection**: Models may perform unevenly across racial groups, age groups, or genders. Evaluation should include fairness metrics and per-demographic-group performance analysis.

5. **Continuous fine-tuning**: As new data comes in (from user reports), periodically retrain the model to handle new edge cases.

6. **Active learning / Human-in-the-loop ML**: Strategically select the most informative data points for human annotation, rather than labeling randomly. Focuses labeling effort on images the model is most uncertain about.

---

## Interview Cheat Sheet

### The 5-Minute Structured Answer

1. **Clarify requirements** (~1 min): Privacy protection, offline processing, 1M annotated images, user feedback loop
2. **Frame as ML task** (~1 min): Object detection (regression + classification), two-stage vs. one-stage, choose two-stage (Faster R-CNN)
3. **Data preparation** (~1 min): Annotated data format, preprocessing (resize/normalize), data augmentation (offline, 1M to 10M), bounding box transformation
4. **Model + training** (~1 min): Backbone CNN + RPN + Classifier, combined loss (L_cls + lambda * L_reg)
5. **Evaluation** (~30 sec): IoU, Precision, AP, mAP (offline), user reports (online)
6. **Serving** (~1 min): NMS for overlapping boxes, batch prediction pipeline (CPU preprocessing + GPU blurring), data pipeline with hard negative mining
7. **Discussion** (~30 sec): DETR, distributed training, GDPR, bias, active learning

### Key Trade-offs to Mention

| Decision | Option A | Option B | Our Choice | Why |
|----------|----------|----------|------------|-----|
| Architecture | Two-stage (accurate) | One-stage (fast) | Two-stage | Privacy demands accuracy; offline = no latency constraint |
| Augmentation | Offline (fast training) | Online (less storage) | Offline | We have storage; faster training iteration |
| Primary metric | Precision | Recall | Both via mAP | mAP captures precision-recall across thresholds |
| Processing | Real-time | Batch | Batch | Requirements explicitly state latency is not a concern |
| CPU/GPU services | Combined | Separated | Separated | Independent scaling, better resource utilization |

### Common Follow-Up Questions and Answers

**Q: Why not use YOLO instead of Faster R-CNN?**
A: YOLO is faster but less accurate. Since we process offline and privacy is critical, accuracy matters more than speed. However, as data scales or if we move to real-time processing, YOLO is a strong alternative.

**Q: What if a face is partially occluded?**
A: Data augmentation (random cropping, occlusion simulation) helps. The RPN should still propose regions around partial faces. We can also lower the confidence threshold to catch more edge cases, accepting more false positives.

**Q: How do you handle edge cases like faces in mirrors, posters, or TV screens?**
A: These should still be blurred (they could contain real people). Hard negative mining and diverse training data help. User reports provide ongoing feedback.

**Q: What if the system blurs something that is not a face?**
A: A false positive (blurring a non-face) is far less harmful than a false negative (missing a real face). We optimize for recall, accepting some false positives.

**Q: How would you handle a 10x increase in images?**
A: Switch to one-stage detection (YOLO/SSD) for faster inference. Use distributed training. Scale preprocessing and blurring services independently. Consider model distillation for inference efficiency.

---

## Notebooks in This Module

| # | Notebook | What It Covers |
|---|----------|---------------|
| 01 | `01_street_view_system_design.ipynb` | **Full system design walkthrough** -- All 7 steps of the ML system design framework applied to Street View blurring. Covers requirements, ML framing, data pipeline, Faster R-CNN architecture, IoU/AP/mAP evaluation, NMS, batch serving pipeline, and key trade-offs. The complete picture from start to finish. |
| 02 | `02_object_detection_deep_dive.ipynb` | **Object detection deep dive** -- Detection vs classification vs segmentation, bounding box formats, anchor boxes, the R-CNN family evolution (R-CNN -> Fast R-CNN -> Faster R-CNN with diagrams), Region Proposal Network implementation, one-stage detectors (YOLO, SSD), speed vs accuracy trade-offs, combined loss function (L_cls + lambda * L_reg), and data augmentation with bounding box transforms. |
| 03 | `03_evaluation_and_serving.ipynb` | **Evaluation and deployment** -- IoU implementation and visualization, precision at different IoU thresholds, step-by-step AP computation, mAP across classes, PR curves, NMS implementation with step-by-step visualization, batch prediction pipeline architecture (CPU/GPU separation), hard negative mining, user feedback loop, and scaling calculations for billions of images. |
| 04 | `04_interview_walkthrough.ipynb` | **Mock interview simulation** -- Complete 45-minute interview in interviewer/candidate dialogue format with timing for each phase, whiteboard-style architecture diagrams, tensor shape walkthrough, common follow-up questions (YOLO vs Faster R-CNN, occluded faces, GDPR, bias in face detection, scaling), scoring rubric, 30-second elevator pitch, and key phrases cheat sheet. |

### Recommended Study Order

1. Start with **Notebook 01** for the full system design overview
2. Deep dive into detection mechanics with **Notebook 02**
3. Master evaluation and production deployment with **Notebook 03**
4. Practice the complete interview flow with **Notebook 04**
