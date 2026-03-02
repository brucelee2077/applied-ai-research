# Visual Search System - ML Design Interview Module

## What is Visual Search? (The Simple Version)

**Imagine this:** You see a pair of cool sneakers on someone walking by. You snap a photo with your phone. Now you want to find those exact sneakers -- or ones that look just like them -- so you can buy them online. That is **visual search**.

Instead of typing words like "red Nike running shoes" into a search bar, you just use a picture. The system looks at your photo, figures out what is in it, and then finds other images that look similar. Pinterest, Google Lens, and Amazon all do this.

**Why does it matter?**
- Sometimes you do not know the words to describe what you are looking for ("that weird lamp shaped like a cactus").
- Pictures carry way more information than words. A photo of a dress captures the color, pattern, fabric texture, cut, and style all at once.
- It enables entirely new product discovery experiences -- "I like this, show me more like it."

---

## Complete System Design for Visual Search

### The Big Picture (How Everything Fits Together)

Think of visual search like a library system. Before anyone searches for a book, the librarian has already organized every book on shelves with labels. When you come in with a question, the librarian does not read every book -- she goes to the right shelf and pulls a few good matches.

Our visual search system has two main pipelines:

```
INDEXING PIPELINE (the librarian organizing books ahead of time)
================================================
Every image on the platform
    --> Preprocessing (resize, normalize)
    --> CNN/ViT Model (extract embedding)
    --> Store embedding in Index (like a giant lookup table)

PREDICTION PIPELINE (answering a user's search query)
================================================
User uploads query image
    --> Preprocessing (resize, normalize)
    --> CNN/ViT Model (extract embedding)
    --> Nearest Neighbor Search (find closest embeddings in the index)
    --> Re-ranking Service (filter duplicates, inappropriate content, apply business rules)
    --> Return ranked list of similar images to the user
```

### Step 1: Clarifying Requirements

In an interview, always start by asking questions. Here are the key clarifications from the PDF:

| Question | Answer |
|----------|--------|
| Should results be ranked by similarity? | Yes, most similar first |
| Images only, or also video/text? | Images only |
| Support image crops (select a region)? | Yes |
| Personalized results? | No, same query image gives same results for everyone |
| Use image metadata (tags, etc.)? | No, only pixel data |
| User actions available? | Only image clicks (impressions + clicks) |
| Content moderation? | Out of scope |
| How to construct training data? | From user interactions (online) |
| Scale? | 100-200 billion images, must be fast |

### Step 2: Frame as an ML Task

**ML Objective:** Accurately retrieve images that are visually similar to the query image.

**Input:** A query image (or image crop) from the user.

**Output:** A ranked list of images, ordered from most similar to least similar.

**ML Category:** This is a **ranking problem** solved with **representation learning**.

#### What is Representation Learning?

Imagine you could describe every image as a list of numbers -- like coordinates on a map. Two photos of golden retrievers would end up at nearby spots on the map, while a photo of a pizza would be far away. That list of numbers is called an **embedding**, and the "map" is called the **embedding space**.

- The model learns to transform images into embedding vectors (lists of numbers).
- Similar images get embeddings that are close together in this space.
- Dissimilar images get embeddings that are far apart.

**How do we rank?**
1. Convert the query image into an embedding.
2. Compute similarity scores between the query embedding and all other image embeddings.
3. Rank images by these similarity scores.

### Step 3: Data Preparation

#### Available Data

**Images Table:**
| ID | Owner ID | Upload Time | Manual Tags |
|----|----------|-------------|-------------|
| 1 | 8 | 1658451341 | Zebra |
| 2 | 5 | 1658451841 | Pasta, Food, Kitchen |
| 3 | 19 | 1658821820 | Children, Family, Party |

**Users Table:**
| ID | Username | Age | Gender | City | Country |
|----|----------|-----|--------|------|---------|
| 1 | johnduo | 26 | M | San Jose | USA |
| 2 | hs2008 | 49 | M | Paris | France |
| 3 | alexish | 16 | F | Rio | Brazil |

**User-Image Interactions:**
| User ID | Query Image ID | Displayed Image ID | Position | Interaction Type |
|---------|---------------|-------------------|----------|-----------------|
| 8 | 2 | 6 | 1 | Click |
| 6 | 3 | 9 | 2 | Click |
| 91 | 5 | 1 | 2 | Impression |

#### Feature Engineering (Image Preprocessing)

Before feeding images into the model, we preprocess them:

1. **Resizing:** Models need fixed input sizes (e.g., 224x224 pixels).
2. **Scaling:** Pixel values scaled to [0, 1] range.
3. **Z-score Normalization:** Scale pixels to mean=0, variance=1.
4. **Consistent Color Mode:** Ensure all images are RGB (not CMYK or grayscale).

### Step 4: Model Development

#### Why Neural Networks?

- They handle **unstructured data** (images) natively.
- They can produce **embeddings** -- traditional ML models cannot do representation learning as effectively.

#### Architecture Choices

| Architecture | Description | Pros | Cons |
|-------------|-------------|------|------|
| **CNN-based (ResNet)** | Stacks of convolutional layers | Battle-tested, efficient, great for images | Limited global context |
| **Transformer-based (ViT)** | Vision Transformer, treats image patches as "tokens" | Captures global relationships, state-of-the-art | Needs more data, computationally heavier |

The simplified architecture:
```
Input Image (224x224x3)
    --> Convolutional Layers (extract visual features)
    --> Fully Connected Layers (compress features)
    --> Embedding Vector (e.g., 256-dimensional)
```

#### Contrastive Training

**The core idea:** Train the model to distinguish similar images from dissimilar ones.

Think of it like a quiz: "Here is a photo of a golden retriever. Which of these 10 images looks most like it?" The model learns to pick the right answer.

Each training example contains:
- **Query image (q):** The reference image
- **Positive image:** An image similar to q
- **n-1 Negative images:** Images dissimilar to q
- **Ground truth label:** The index of the positive image

#### Three Ways to Get Positive Images

| Method | How It Works | Pros | Cons |
|--------|-------------|------|------|
| **Human Judgment** | Annotators manually find similar images | Most accurate | Expensive, slow |
| **User Clicks** | Clicked image = similar to query | Automatic, no manual work | Noisy (users click random things), sparse |
| **Self-Supervision (Data Augmentation)** | Rotate/crop/flip the query image to create a "similar" image | Free, not noisy, automated | Augmented images differ from real similar images |

**Best approach for our system:** Self-supervision (using frameworks like SimCLR or MoCo). It is free, scalable, and works well with billions of images. We can always switch to click-based or human labeling later.

#### Loss Function: Contrastive Loss

The loss computation has three steps:

1. **Compute Similarities:** Use dot product or cosine similarity between the query embedding and all candidate embeddings. (Avoid Euclidean distance in high dimensions due to the curse of dimensionality.)
2. **Softmax:** Convert similarity scores into probabilities that sum to 1.
3. **Cross-Entropy:** Measure how close predicted probabilities are to the ground truth label.

**Pro tip for interviews:** Mention that you can use a **pre-trained model** (e.g., pre-trained on ImageNet with contrastive learning) and **fine-tune** it on your data. This saves massive training time.

### Step 5: Evaluation

#### Offline Metrics

The evaluation dataset has: query image, candidate images, and similarity scores (0-5 scale).

| Metric | What It Measures | Use It? | Why/Why Not |
|--------|-----------------|---------|-------------|
| **MRR (Mean Reciprocal Rank)** | Rank of the first relevant item, averaged | No | Only considers the first relevant item, ignores the rest |
| **Recall@k** | Fraction of all relevant items captured in top-k | No | Denominator can be huge (millions of dog images); does not measure ranking quality |
| **Precision@k** | Fraction of top-k items that are relevant | No | Does not measure ranking quality (reordering does not change score) |
| **mAP (Mean Average Precision)** | Average Precision across output lists | Partial | Works for binary relevance only (relevant/irrelevant) |
| **nDCG (Normalized DCG)** | Ranking quality compared to ideal ranking | YES | Works with continuous relevance scores; measures both precision and ranking quality |

**nDCG is our primary offline metric** because:
- It handles continuous relevance scores (0-5 scale).
- It penalizes relevant items ranked too low.
- It normalizes against the ideal ranking so the score is between 0 and 1.

**nDCG Computation (3 steps):**
1. Compute DCG: Sum of (relevance_i / log2(i+1)) for each position i
2. Compute IDCG: DCG of the ideal (perfectly sorted) ranking
3. nDCG = DCG / IDCG

#### Online Metrics

| Metric | Description |
|--------|-------------|
| **Click-Through Rate (CTR)** | Number of clicked images / Total suggested images |
| **Time Spent** | Average daily/weekly/monthly time on suggested images |

### Step 6: Serving Architecture

#### Prediction Pipeline (Real-Time)

1. **Embedding Generation Service:** Preprocesses the query image and runs it through the trained model to get its embedding vector.
2. **Nearest Neighbor Service:** Finds the closest embeddings in the index to the query embedding.
3. **Re-ranking Service:** Applies business logic -- filters inappropriate content, removes duplicates/near-duplicates, removes private images, and enforces policies.

#### Indexing Pipeline (Offline/Batch)

- All platform images are pre-processed and their embeddings are stored in an **index table**.
- When new images are uploaded, the indexing service computes their embeddings and adds them to the index.
- Memory optimization techniques: **vector quantization** and **product quantization** reduce storage requirements.

### Step 7: Nearest Neighbor Algorithms (Deep Dive)

#### Exact Nearest Neighbor (Linear Search)

- Computes distance between query and every single point in the index.
- Time complexity: O(N x D), where N = number of points, D = dimension.
- With billions of images, this is way too slow.

#### Approximate Nearest Neighbor (ANN)

ANN trades a tiny bit of accuracy for massive speed gains. Three categories:

| Category | How It Works | Examples |
|----------|-------------|----------|
| **Tree-based** | Splits embedding space into partitions using a tree structure; only searches the partition the query falls in | R-trees, Kd-trees, Annoy |
| **LSH (Locality-Sensitive Hashing)** | Hash functions map nearby points to the same "bucket"; only search within the same bucket | LSH variants |
| **Clustering-based** | Groups points into clusters; only search within the query's cluster | k-means based approaches |

Time complexity: O(D x log N) -- sublinear, much faster than exact search.

**Libraries to know:**
- **FAISS** (by Meta): Industry standard, supports all major ANN methods
- **ScaNN** (by Google): Optimized for large-scale retrieval

### Step 8: Other Talking Points (Senior/Staff Level)

These are follow-up topics to prepare for:

- **Content Moderation:** Identify and block inappropriate images
- **Positional Bias:** Users tend to click items at the top of the list regardless of relevance
- **Image Metadata:** Using tags to improve search quality
- **Smart Crop:** Using object detection to crop relevant regions
- **Graph Neural Networks:** Learning better representations from image relationship graphs
- **Text-to-Image Search:** Supporting text queries to find images
- **Active Learning / Human-in-the-Loop:** More efficiently annotating training data

---

## Interview Cheat Sheet

### Framework for Answering (5-Step Structure)

1. **Clarify Requirements** (~5 min): Ask about scope, scale, user actions, data availability
2. **Frame as ML Task** (~5 min): Define objective, input/output, ML category (ranking + representation learning)
3. **Data & Model** (~15 min): Data sources, preprocessing, architecture (CNN/ViT), contrastive training, loss function
4. **Evaluation** (~5 min): Offline (nDCG) and online (CTR, time spent) metrics
5. **Serving** (~10 min): Prediction pipeline, indexing pipeline, ANN algorithms, re-ranking

### Key Phrases to Drop in Your Interview

- "We frame this as a ranking problem using representation learning"
- "Contrastive training with self-supervision (SimCLR/MoCo) is our starting point"
- "nDCG is our primary offline metric because it handles continuous relevance scores"
- "We use ANN (e.g., FAISS) for sublinear retrieval at scale"
- "The re-ranking service applies business logic post-retrieval"
- "We can fine-tune a pre-trained model to reduce training time"

---

## References

1. Visual search at Pinterest - https://arxiv.org/pdf/1505.07647.pdf
2. Visual embeddings for search at Pinterest - https://medium.com/pinterest-engineering/unifying-visual-embeddings-for-visual-search-at-pinterest-74ea7ea103f0
3. Representation learning - https://en.wikipedia.org/wiki/Feature_learning
4. ResNet paper - https://arxiv.org/pdf/1512.03385.pdf
5. Transformer paper - https://arxiv.org/pdf/1706.03762.pdf
6. ViT (Vision Transformer) - https://arxiv.org/abs/2010.11929
7. SimCLR - https://arxiv.org/abs/2002.05709
8. MoCo - https://arxiv.org/abs/1911.05722
9. Contrastive losses overview
10. Dot product similarity
11. Cosine similarity
12. Euclidean distance
13. Curse of dimensionality
14-30. Various references on ANN algorithms, FAISS, ScaNN, content moderation, positional bias, object detection, GNNs, active learning

---

## Notebooks in This Module

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 01 | `01_visual_search_system_design.ipynb` | End-to-end system design overview | Problem framing, data pipeline, model architecture, evaluation metrics, serving overview |
| 02 | `02_embedding_and_contrastive_learning.ipynb` | Deep dive into the embedding model and training | Embedding spaces, SimCLR, MoCo, NT-Xent loss, temperature parameter, hard negative mining, fine-tuning |
| 03 | `03_ann_search_and_serving.ipynb` | Approximate nearest neighbor search and full serving pipeline | Exact NN vs ANN, KD-trees, LSH, IVF, FAISS IVFFlat demo, product quantization, prediction and indexing pipelines, re-ranking service, scale analysis |
| 04 | `04_interview_walkthrough.ipynb` | Complete 45-minute mock interview simulation | Interview timeline, phase-by-phase scripts, architecture diagrams, staff-level power moves, 6 follow-up Q&As with WEAK vs STRONG answers, scoring rubric, 30-second elevator pitch, vocabulary cheat sheet |
