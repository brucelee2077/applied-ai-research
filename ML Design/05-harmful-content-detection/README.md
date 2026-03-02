# Harmful Content Detection - ML Design Interview Module

## What is Harmful Content Detection? (The Simple Version)

**Imagine this:** You are a hall monitor for the entire internet. Millions of people post photos, videos, and messages every single minute. Your job is to spot the mean, dangerous, or inappropriate stuff before anyone gets hurt by it -- and you have to do it almost instantly.

That is what a **harmful content detection system** does. Social media platforms like Facebook, LinkedIn, and Twitter have community guidelines -- rules about what you can and cannot post. The system proactively monitors every new post, figures out if it violates the rules, and either removes it or pushes it down so fewer people see it.

**Why does it matter?**
- Billions of posts are created every day -- humans alone cannot review them all.
- Some harmful content (violence, self-harm) is time-sensitive and needs to be caught in seconds.
- Without automated detection, platforms become unsafe and users leave.
- Legal and regulatory requirements demand that platforms control harmful content.

### Two Categories of Integrity Enforcement

| Category | Examples | Focus |
|----------|----------|-------|
| **Harmful content** | Violence, nudity, self-harm, hate speech | The content itself is dangerous |
| **Bad acts / bad actors** | Fake accounts, spam, phishing, organized manipulation | The behavior or person is dangerous |

This module focuses on detecting **harmful content** specifically.

---

## Complete System Design

### The Big Picture

Think of the system like a school cafeteria security setup. Before any food (post) reaches the students (users), it passes through a quality checker. If the checker is very confident the food is bad, it gets thrown out immediately. If the checker is only slightly suspicious, the food gets set aside for a human inspector to look at later.

```
POST CREATION FLOW
==================
User creates a new post (text + image + video)
    --> Harmful content detection service
        --> ML model predicts probability of harm
            --> HIGH confidence harmful?
                --> Violation enforcement service (immediately remove + notify user)
            --> LOW confidence harmful?
                --> Demoting service (reduce visibility + queue for human review)
            --> Not harmful?
                --> Post goes live normally
```

### Step 1: Clarifying Requirements (Always Ask These in an Interview!)

| Question | Answer | Why It Matters |
|----------|--------|----------------|
| Detect harmful content or bad actors? | Both matter, but focus on harmful content | Scopes the problem |
| Post modalities? | Text, image, video, or any combination | Determines model architecture |
| Language support? | Multiple languages | Need multilingual models |
| Human annotators available? | Limited number, expensive | Cannot label everything manually |
| User reporting? | Yes, users can report harmful posts | Additional signal for the model |
| Latency requirements? | Varies by harm type -- some need real-time | Architectural constraints |
| What happens to flagged posts? | Either removed or demoted based on confidence | Need calibrated probabilities |

### Step 2: Frame as an ML Task

**ML Objective:** Accurately predict the probability that a post is harmful.

**Input:** A post P (which can contain text, images, video, or any combination) along with metadata about the post and its author.

**Output:** Probability scores for each harmful class (violence, nudity, hate speech, etc.).

**ML Category:** This is a **multi-task classification** problem. We predict multiple harm categories simultaneously using shared and task-specific layers.

### Step 3: Handling Heterogeneous (Multi-Modal) Data

A post can contain an image, text, author info, or any combination. Think of it like a report card that has grades in different subjects -- you need to look at all of them to understand the whole picture.

#### Two Fusion Strategies

**Late Fusion (Separate Models Combined at the End):**
```
Image -----> [Image Model] -----> prediction_1 \
                                                 --> [Combine Predictions] --> Final prediction
Text  -----> [Text Model]  -----> prediction_2 /
```

| Pros | Cons |
|------|------|
| Train, evaluate, and improve each model independently | Need separate training data per modality |
| Easy to debug (which model is wrong?) | Cannot capture cross-modal interactions |
| Can add new modalities easily | Fails on memes (benign text + benign image = harmful together) |

**Early Fusion (Combine Features, Then One Model):**
```
Image -----> [Feature Extractor] ---\
                                     --> [Fuse Features] --> [Single Model] --> Final prediction
Text  -----> [Feature Extractor] ---/
```

| Pros | Cons |
|------|------|
| Captures cross-modal interactions (memes!) | Harder to train -- one model for everything |
| Only need one training dataset | More complex debugging |
| Only one model to maintain | Needs more training data |

**Our Choice: Early Fusion.** Because content that looks harmless in isolation (a normal image + a normal sentence) can become harmful when combined (like hateful memes). With 500 million posts per day, we have enough data for the model to learn these cross-modal relationships.

### Step 4: ML Category -- Multi-Task Classification

We examined four options:

#### Option A: Single Binary Classifier
One model outputs: harmful (yes/no).
- **Problem:** Cannot tell users WHY their post was removed. Cannot identify which harm categories underperform.

#### Option B: One Binary Classifier Per Harmful Class
Separate models for violence, nudity, hate speech, etc.
- **Pros:** Can explain why a post was removed. Can improve each model independently.
- **Cons:** Must train and maintain multiple models separately. Expensive.

#### Option C: Multi-Label Classifier
One model outputs probabilities for all classes simultaneously using shared weights.
- **Pros:** Cheaper to train and maintain (one model). Less costly.
- **Cons:** A single shared model may not be ideal since different harm categories may need features to be transformed differently.

#### Option D: Multi-Task Classifier (Our Choice)
One model with **shared layers** (learn common patterns) and **task-specific heads** (specialize per harm category).

```
                  [Violence Head]    [Nudity Head]    [Hate Speech Head]
                       |                  |                  |
                  [Task-Specific]    [Task-Specific]   [Task-Specific]
                       |                  |                  |
                  ===================================================
                  |              Shared Layers                      |
                  ===================================================
                       |
                  [Fused Features]
                       |
                  [Feature Fusion]
                  /       |       \
            [Image]    [Text]   [Author]
```

**Why Multi-Task?**
1. **Efficient:** Only one model to train and maintain.
2. **Smart sharing:** Shared layers transform features in a way beneficial for all tasks, preventing redundant computation.
3. **Data sharing:** Training data for violence detection also helps the shared layers learn patterns useful for hate speech detection (transfer learning across tasks).

**Shared Layers:** A set of hidden layers that transform raw input features into higher-level representations useful for all harm categories.

**Task-Specific Layers (Classification Heads):** Independent ML layers, one per harm category. Each head transforms the shared features in a way optimized for predicting a specific type of harm.

---

## Data Preparation

### Data Sources

Three main data sources feed the system:

**1. Users Table**

| Field | Description |
|-------|-------------|
| ID | Unique user identifier |
| Username | Display name |
| Age | User age |
| Gender | User gender |
| City, Country | Location information |

**2. Posts Table**

| Field | Description |
|-------|-------------|
| Post ID | Unique post identifier |
| Author ID | Who created the post |
| On-device IP | Device information |
| Timestamp | When the post was created |
| Textual content | The text body of the post |
| Image/Video content | Media attachments |

**3. User-Post Interactions Table**

| Field | Description |
|-------|-------------|
| User ID | Who interacted |
| Post ID | Which post they interacted with |
| Interaction type | Impression, Like, Comment, Share, Report |
| Interaction content | Comment text, report reason, etc. |

---

## Feature Engineering

### The Five Feature Categories

#### 1. Textual Content Features
- Use a pre-trained multilingual model (like DistilBERT) to convert text into an embedding vector.
- **Why DistilBERT over BERT?** Faster inference, smaller model, same basic capability. BERT was trained on English-only data and embedding generation is slow due to its size.
- Text is first tokenized (broken into pieces), then embedded into a feature vector.

#### 2. Image or Video Features
- **Preprocessing:** Decode, resize, normalize pixel values.
- **Feature extraction:** Use a pre-trained image model (like CLIP's visual encoder or SimCLR) to convert images/videos into feature vectors.
- For videos, process sampled frames through the video model to get a video feature vector.

#### 3. User Reactions to the Post
- As comments accumulate and reports come in, the system becomes more confident about whether content is harmful (shown in Figure 5.13 of the PDF).
- **Comment features:** Embed each comment using the same pre-trained text model, then aggregate (e.g., average) the embeddings into a single comments feature vector.
- **Reaction counts:** Number of reports, likes, comments, shares -- scaled and concatenated.

#### 4. Author Features

| Feature | Type | Description |
|---------|------|-------------|
| Number of violations | Numerical | How many times the author's posts have been removed |
| Profane words rate | Numerical | Rate of profanity in the author's previous posts and comments |
| Age | Numerical | User's age |
| Gender | Categorical | One-hot encoded |
| City and Country | Categorical | Embedded into feature vectors using an embedding layer |
| Number of followers/followings | Numerical | Social network size |
| Account age | Numerical | How old the account is (newer accounts are more suspicious) |

#### 5. Contextual Information

| Feature | Type | Description |
|---------|------|-------------|
| Time of day | Categorical | Bucketed into morning, noon, afternoon, evening, night |
| Device | Categorical | Smartphone, tablet, or desktop (one-hot encoded) |

All features are concatenated into a single large feature vector that feeds into the model.

---

## Model Development

### Model Selection
A neural network is the most common model for multi-task learning. Key architectural decisions:
- Number of layers and neurons per layer in shared vs. task-specific sections.
- These are determined by hyperparameter tuning.

### Constructing the Training Dataset
Each data point consists of:
- **Input features:** All the engineered features (text embedding, image embedding, reactions, author features, context).
- **Labels:** Binary labels for each harm category (violence=0/1, nudity=0/1, hate speech=0/1, etc.).

### Labeling Strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Hand labeling** | Human annotators label each post | Most accurate | Expensive, slow |
| **Natural labeling** | Use user reports and automated signals | Free, scales well | Noisy, inconsistent |
| **Hybrid (Our Choice)** | Natural labels for training, hand labels to evaluate and prioritize | Best of both worlds | Requires careful pipeline design |

### Choosing the Loss Function
Multi-task training assigns a loss function per task. Since each task is binary classification, we use **cross-entropy loss** for each harm category:

```
Overall Loss = L_violence + L_nudity + L_hate + ... + L_n
```

Each task-specific loss is standard binary cross-entropy. The overall loss is the sum of all task-specific losses.

### Training Challenges

**Overfitting in multimodal models:** When learning speed varies across modalities, one modality (e.g., image) can dominate the learning process. Two techniques to address this:
- **Gradient blending:** Balance gradients from different modalities.
- **Focal loss:** Focus training on hard examples that the model currently gets wrong.

---

## Evaluation

### Offline Metrics

**PR-AUC (Precision-Recall Area Under Curve):**
- Plots precision vs. recall at different thresholds.
- Summarizes the trade-off between precision and recall.
- Higher PR-AUC = better model.
- Especially useful for imbalanced datasets (harmful posts are rare compared to normal posts).

**ROC-AUC (Receiver Operating Characteristic Area Under Curve):**
- Plots true positive rate (recall) vs. false positive rate at different thresholds.
- Summarizes overall classification performance.
- We use BOTH ROC-AUC and PR-AUC as our offline evaluation metrics.

### Online Metrics

| Metric | Formula | What It Tells Us |
|--------|---------|------------------|
| **Prevalence** | Harmful posts not prevented / Total posts | How much harmful content slips through |
| **Harmful impressions** | Harmful impressions not prevented / Total impressions | Better than prevalence -- weighs by reach |
| **Valid appeals** | Reversed appeals / Harmful posts detected | How often we incorrectly remove posts (false positive rate) |
| **Proactive rate** | Harmful posts found by system / Harmful posts detected by system | How much the system catches before user reports |
| **User reports per harmful class** | Reports for each category | Performance per harm type |

**Why harmful impressions > prevalence?** A harmful post with 100K views is worse than one with 10 views. Prevalence treats them equally; harmful impressions does not.

---

## Serving

### System Architecture (Figure 5.19 from the PDF)

```
User creates post (post_id, text, image, video)
        |
        v
[Harmful Content Detection Service]
        |
        v
[Trained ML Model] --> predicts harm probability
        |
        +--> High confidence harmful --> [Violation Enforcement Service]
        |                                   - Immediately removes post
        |                                   - Notifies user why
        |
        +--> Low confidence harmful  --> [Demoting Service]
                                         - Temporarily reduces post visibility
                                         - Queues for human review
                                         - Post stored for manual inspection
```

### Three Serving Components

**1. Harmful Content Detection Service:**
Given a new post, predicts the probability that it belongs to each harm category.

**2. Violation Enforcement Service:**
When the system predicts harm with HIGH confidence, this service immediately removes the post and notifies the user about which guideline was violated.

**3. Demoting Service:**
When the system predicts harm with LOW confidence, this service temporarily demotes the post (reduces its visibility in feeds) while it waits for human review. This avoids both:
- Letting potentially harmful content spread unchecked.
- Incorrectly removing legitimate content (which frustrates users).

---

## Interview-Ready Quick Reference

### The 7-Step Framework for This Problem

```
1. CLARIFY: Multi-modal posts, multiple harm categories, real-time + batch, multiple languages
2. FRAME: Multi-task classification with early fusion of modalities
3. DATA: Users + Posts + Interactions; natural labels (user reports) + hand labels (for evaluation)
4. FEATURES: Text embeddings (DistilBERT) + Image embeddings (CLIP/SimCLR) + Reactions + Author + Context
5. MODEL: Neural network with shared layers + task-specific classification heads
6. TRAINING: Cross-entropy per task, sum of losses, handle modality imbalance (gradient blending, focal loss)
7. METRICS: Offline = PR-AUC + ROC-AUC per task; Online = Harmful impressions, Valid appeals, Proactive rate
8. SERVING: Detection service --> Violation enforcement (high confidence) or Demoting (low confidence)
```

### Common Follow-Up Questions

| Question | Strong Answer |
|----------|---------------|
| Why early fusion over late fusion? | Memes and combined content can be harmful even when each modality is benign in isolation. Early fusion captures cross-modal interactions. |
| Why multi-task over separate models? | Shared layers enable knowledge transfer between tasks, are cheaper to maintain, and limited labeled data for one task benefits from data in other tasks. |
| How do you handle class imbalance? | Focal loss, oversampling, and strategic threshold tuning per harm category. |
| How do you handle adversarial content? | Obfuscation detection, character normalization, image perturbation augmentation during training. |
| How do you decide the confidence threshold? | Tune per category based on the cost of false positives vs. false negatives. Violence has a low threshold (better safe); satire/controversy has a higher threshold. |
| What about appeals? | Track valid appeal rate as an online metric. High appeal rate means the model is too aggressive (too many false positives). |
| How do you handle new types of harmful content? | Active learning pipeline: surface uncertain predictions for human review, retrain periodically with new labeled data. |
| What about latency? | Feature extraction happens at post creation time. Model inference is fast (single forward pass). Heavy content (video) can be processed asynchronously. |

### Key Figures from the PDF

- **Figure 5.1:** System overview -- post goes in, harm probability comes out
- **Figure 5.3:** Heterogeneous post data (image + text + author)
- **Figure 5.4:** Late fusion architecture
- **Figure 5.5:** Early fusion architecture
- **Figure 5.6:** Single binary classifier
- **Figure 5.7:** One binary classifier per harmful class
- **Figure 5.8:** Multi-label classifier
- **Figure 5.9-5.11:** Multi-task classification (shared layers + task-specific heads)
- **Figure 5.12:** Complete ML task framing
- **Figure 5.13:** User reactions providing increasing confidence over time
- **Figure 5.14:** Feature engineering for reactions and content
- **Figure 5.15:** Summary of all feature engineering
- **Figure 5.16:** A constructed data point with labels
- **Figure 5.17:** Model training with multi-task loss
- **Figure 5.18:** PR curve
- **Figure 5.19:** ML system design (serving architecture)

---

## Notebooks in This Module

| Notebook | Topic |
|----------|-------|
| `01_harmful_content_system_design.ipynb` | Full system design: problem framing, multi-task classification, early/late fusion, data pipeline |
| `02_multimodal_fusion_and_features.ipynb` | Feature engineering, text/image embeddings, multimodal fusion architectures |
| `03_multi_task_training_and_evaluation.ipynb` | Multi-task training, focal loss, PR-AUC/ROC-AUC, online metrics, serving architecture |
