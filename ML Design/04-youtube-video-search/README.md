# YouTube Video Search System Design

## What Is This?

Imagine you type **"funny cat videos"** into YouTube's search box. Behind the scenes, the computer has to look through **over one billion videos** and figure out which ones you'd actually want to watch -- in less than a second. That's the problem we're solving here.

This module is a comprehensive, interview-ready guide to designing a YouTube-scale video search system. It covers everything from the ByteByteGo ML System Design Interview chapter on YouTube Video Search, expanded with staff-level technical depth and explained so clearly that a 12-year-old could follow along.

---

## Table of Contents

1. [Why Video Search Is Hard](#why-video-search-is-hard)
2. [Clarifying Requirements](#clarifying-requirements)
3. [Framing It as an ML Problem](#framing-it-as-an-ml-problem)
4. [High-Level Architecture](#high-level-architecture)
5. [Data Preparation](#data-preparation)
6. [Feature Engineering](#feature-engineering)
7. [Model Development](#model-development)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Serving Architecture](#serving-architecture)
10. [Advanced Topics](#advanced-topics)
11. [Interview Cheat Sheet](#interview-cheat-sheet)
12. [Module Files](#module-files)

---

## Why Video Search Is Hard

Think of it like this: you walk into the world's biggest library, except instead of books, there are **one billion videos**. You tell the librarian "I want videos about dogs playing indoors." The librarian has to:

1. **Understand what you mean** -- not just the words, but the *intent*. Are you looking for training videos? Funny compilations? Livestreams?
2. **Look through a billion videos** -- but not one by one (that would take years). They need shortcuts.
3. **Understand what each video is about** -- some videos have great titles ("Golden Retriever Playing Fetch Inside"), but others have terrible titles ("VID_20240301_001.mp4"). The librarian needs to actually *look at* the video content too.
4. **Rank the results** -- put the best ones first, not just any match.
5. **Do all of this in under 200 milliseconds.**

That's why video search is a fascinating ML system design problem. It combines:
- **Natural Language Processing (NLP)** to understand the text query
- **Computer Vision (CV)** to understand video content
- **Information Retrieval (IR)** to efficiently search billions of items
- **Ranking/Learning to Rank** to put the best results first
- **Systems Engineering** to serve results at low latency

---

## Clarifying Requirements

In an interview, always start by asking clarifying questions. Here's how the conversation goes:

| Question | Answer |
|----------|--------|
| Is the input text-only, or can users search with images/video? | **Text queries only** |
| Is the content only videos, or also images/audio? | **Videos only** |
| What determines video relevance? | **Visual content + textual metadata (title, description)** |
| Is training data available? | **Yes, 10 million (video, text query) pairs** |
| Language support? | **English only (for simplicity)** |
| How many videos on the platform? | **1 billion videos** |
| Do we need personalization? | **No -- unlike recommendations, search does not require personalization for this problem** |

**Problem Statement Summary:** Design a search system where the input is a text query and the output is a ranked list of relevant videos. We leverage both visual content and textual metadata. We have 10 million annotated (video, text query) pairs for training.

---

## Framing It as an ML Problem

### ML Objective
Rank videos based on their **relevance** to the text query.

### Input/Output
- **Input:** A text query (e.g., "dogs playing indoor")
- **Output:** A ranked list of videos, sorted by relevance to the query

### ML Category
This is a **representation learning** + **ranking** problem. We need to:
1. Encode text queries and videos into a shared embedding space
2. Compute similarity between query and video embeddings
3. Rank videos by similarity scores

Think of it like this: imagine every video and every search query lives as a point in a giant map. If a video is relevant to a query, their points should be **close together** on the map. If they're not relevant, they should be **far apart**.

---

## High-Level Architecture

The system has **two main search pathways** that work together:

```
                          Text Query: "dogs playing indoor"
                                    |
                    +---------------+---------------+
                    |                               |
              Visual Search                    Text Search
              (Embedding-based)            (Inverted Index / ES)
                    |                               |
            Top-K similar videos          Videos with matching
            by visual content             titles, tags, descriptions
                    |                               |
                    +---------------+---------------+
                                    |
                              Fusing Layer
                         (weighted score combination)
                                    |
                            Re-ranking Service
                         (business logic, policies)
                                    |
                          Final Ranked Results
```

### Visual Search Path
- Encode the text query using a **text encoder** (e.g., BERT)
- Compare the text embedding against pre-computed **video embeddings** using dot product similarity
- Use **Approximate Nearest Neighbor (ANN)** for fast retrieval from billions of videos
- Returns: videos whose *visual content* matches the query

### Text Search Path
- Use **Elasticsearch** (inverted index) to find videos with titles, descriptions, and tags matching the query
- No ML training cost -- purely text matching
- Returns: videos whose *metadata* matches the query

### Fusing Layer
- Combines results from both paths
- Simple approach: **weighted sum** of relevance scores from each path
- Complex approach: a separate ML model to re-rank (more expensive, slower)

### Re-ranking Service
- Applies **business logic** and **policies** (e.g., remove inappropriate content, boost fresh content, apply diversity rules)

---

## Data Preparation

### Training Data
We have an annotated dataset of 10 million (video, text query) pairs:

| Video Name | Query | Split |
|-----------|-------|-------|
| 76134.mp4 | Kids swimming in a pool! | Training |
| 92167.mp4 | Celebrating graduation | Training |
| 2867.mp4 | A group of teenagers playing soccer | Validation |
| 28543.mp4 | How Tensorboard works | Validation |
| 70310.mp4 | Road trip in winter | Test |

In practice, you could also use **user interaction data** (clicks, likes, watch time) to construct and label data, enabling continuous model training.

---

## Feature Engineering

Since ML models only accept numbers, we need to convert text and video into numerical representations.

### Preparing Text Data

The text pipeline has three stages:

```
Raw Text --> Text Normalization --> Tokenization --> Token IDs
```

**1. Text Normalization (cleanup)**

Think of this like spell-checking and standardizing. "DOG!", "dogs", and "dog" should all mean the same thing.

| Technique | What It Does | Example |
|-----------|-------------|---------|
| Lowercasing | Make all letters lowercase | "DOG" --> "dog" |
| Punctuation removal | Remove periods, commas, etc. | "hello!" --> "hello" |
| Trim whitespace | Remove extra spaces | "  hello  " --> "hello" |
| NFKD normalization | Decompose combined characters | "Malaga" (with accent) --> "Malaga" |
| Strip accents | Remove accent marks | "Noel" (with accent) --> "Noel" |
| Lemmatization/Stemming | Reduce to root word | "walking, walks, walked" --> "walk" |

**2. Tokenization**

Breaking text into smaller pieces:
- **Word tokenization:** "I have an interview" --> ["I", "have", "an", "interview"]
- **Subword tokenization:** splits into sub-words (n-gram characters) -- used by BERT, GPT
- **Character tokenization:** splits into individual characters

**3. Tokens to IDs**

Two approaches:

| | Lookup Table | Feature Hashing |
|---|-------------|----------------|
| **Speed** | Fast lookup | Need to compute hash function |
| **ID-to-token** | Easy reverse lookup | Not possible |
| **Memory** | Stores full table in memory | Hash function is sufficient |
| **Unseen tokens** | Cannot handle new words | Handles any word via hash |
| **Collisions** | No collisions | Potential hash collisions |

### Preparing Video Data

Raw videos are preprocessed through:
1. **Decode** the video file
2. **Sample frames** (not every frame -- that would be too expensive)
3. **Resize and normalize** each frame
4. **Feed frames through a visual encoder** (e.g., ViT) to get frame embeddings
5. **Aggregate** frame embeddings (e.g., average pooling) to get a single video embedding

---

## Model Development

### Text Encoder Options

**Statistical Methods (Simple but Limited):**

| Method | How It Works | Limitations |
|--------|-------------|-------------|
| **Bag of Words (BoW)** | Count word occurrences in a matrix | Ignores word order; no semantic meaning; sparse vectors |
| **TF-IDF** | Like BoW but normalizes by word frequency | Ignores word order; no semantic meaning; sparse vectors |

**ML-Based Methods (Powerful):**

| Method | How It Works | Strengths |
|--------|-------------|-----------|
| **Embedding Layer** | Maps each token ID to a learned dense vector | Simple, effective for sparse features |
| **Word2Vec** | Learns embeddings from word co-occurrences (CBOW, Skip-gram) | Captures semantic similarity |
| **Transformer-based (BERT, GPT)** | Context-aware embeddings -- same word gets different embeddings based on context | State-of-the-art; captures context and semantics |

**Our Choice:** BERT (or similar Transformer) as the text encoder because it produces context-aware, semantically meaningful embeddings.

### Video Encoder Options

| Type | How It Works | Pros | Cons |
|------|-------------|------|------|
| **Video-level models** | Process the entire video (3D convolutions, Video Transformers) | Captures temporal information (actions, motions) | Computationally expensive |
| **Frame-level models** | Sample frames, encode each frame, aggregate embeddings | Faster, less compute | Misses temporal aspects |

**Our Choice:** Frame-level model using **ViT (Vision Transformer)** because:
- Faster training and inference
- Lower computational cost
- Temporal understanding is not critical for search relevance in most cases

### Model Training: Contrastive Learning

The text encoder and video encoder are trained together using **contrastive learning**:

Think of it like a matching game. You have a batch of (text, video) pairs. The goal is:
- **Positive pairs** (matching text-video): embeddings should be CLOSE (high dot product)
- **Negative pairs** (non-matching text-video): embeddings should be FAR (low dot product)

The loss function pushes matching pairs together and non-matching pairs apart in the embedding space.

---

## Evaluation Metrics

### Offline Metrics

| Metric | Formula / Description | Good for Our Problem? | Why? |
|--------|----------------------|----------------------|------|
| **Precision@k** | (relevant items in top k) / k | Not ideal | With only 1 relevant video per query, precision@10 maxes at 0.1 |
| **Recall@k** | 1 if relevant video is in top k, else 0 | Decent | But depends on choosing the right k; cannot distinguish "almost found" |
| **MRR (Mean Reciprocal Rank)** | Average of 1/rank of first relevant item | **Best choice** | Addresses recall@k shortcomings; rewards finding the relevant video earlier |

**MRR Formula:**
```
MRR = (1/m) * SUM(1/rank_i) for i = 1 to m
```

Where `m` is the number of queries and `rank_i` is the rank of the first relevant video for query `i`.

**Example:** If for 3 queries, the relevant video is ranked 1st, 3rd, and 2nd:
```
MRR = (1/3) * (1/1 + 1/3 + 1/2) = (1/3) * 1.833 = 0.611
```

### Online Metrics

| Metric | What It Measures | Limitation |
|--------|-----------------|------------|
| **Click-Through Rate (CTR)** | How often users click on results | Doesn't track if clicked videos are actually relevant |
| **Video Completion Rate** | How many search-result videos are watched to the end | Users may find a video relevant but not watch it fully |
| **Total Watch Time** | Total time spent watching search results | **Best proxy for relevance** -- users watch more when results are good |

---

## Serving Architecture

The full serving system has three pipelines:

### 1. Prediction Pipeline (Online, at Query Time)

```
User Query
    |
    v
[Text Encoder] --> query embedding
    |
    +---> [ANN Service] --> top-K visual matches
    |
    +---> [Elasticsearch] --> text-matched videos
    |
    v
[Fusing Layer] --> combined ranked list
    |
    v
[Re-ranking Service] --> final results with business logic
    |
    v
Display to User
```

- **ANN (Approximate Nearest Neighbor):** Used for fast embedding similarity search. Algorithms like HNSW or FAISS can search billions of vectors in milliseconds.
- **Elasticsearch:** Full-text search on titles, descriptions, tags.
- **Fusing:** Weighted combination of scores from both paths.
- **Re-ranking:** Apply business rules (freshness, diversity, content policy).

### 2. Video Indexing Pipeline (Offline, When Videos Are Uploaded)

```
New Video Uploaded
    |
    v
[Video Encoder] --> video embedding
    |
    v
[ANN Index] (stored for fast retrieval)
```

### 3. Text Indexing Pipeline (Offline, When Videos Are Uploaded)

```
New Video Uploaded
    |
    v
[Extract title, description, tags]
    |
    +---> Manual tags (from uploader)
    +---> Auto-generated tags (from auto-tagger model)
    |
    v
[Elasticsearch Index]
```

The **auto-tagger** is especially valuable when uploaders don't provide tags. It uses a standalone model to generate tags from video content.

---

## Advanced Topics

These are "bonus points" topics that can elevate your interview answer:

### 1. Multi-Stage Design
- **Candidate Generation:** Retrieve thousands of candidates quickly (ANN + ES)
- **Ranking:** Score and rank the candidates with a more expensive model
- **Re-ranking:** Apply business logic

### 2. Query Understanding
- **Spelling correction:** "funy cat" --> "funny cat"
- **Query category identification:** Is this entertainment? Education? News?
- **Entity recognition:** "Taylor Swift concert" --> entity = Taylor Swift

### 3. Multimodal Processing
- Process **speech and audio** from videos (not just visual frames)
- Speech-to-text transcription adds another signal for relevance

### 4. Head/Torso/Tail Queries
- **Head queries** (e.g., "music video"): Very common, can be pre-computed
- **Torso queries** (e.g., "python tutorial for beginners"): Moderate frequency
- **Tail queries** (e.g., "how to fix leaky faucet in 1970s ranch house"): Rare, require deeper understanding

### 5. Near-Duplicate Detection
- Remove near-duplicate videos from results to improve diversity
- Use perceptual hashing or embedding similarity to detect duplicates

### 6. Freshness and Popularity
- Boost recently uploaded videos for trending/news queries
- Incorporate video popularity signals (view count, engagement)

### 7. Multi-Language Support
- Extend to support queries and videos in multiple languages
- Use multilingual Transformers (e.g., mBERT, XLM-R)

### 8. Additional Video Features
- Video length, video popularity, channel authority, upload date
- Engagement metrics: likes, shares, comments

---

## Interview Cheat Sheet

### The 30-Second Pitch

> "I'd design a dual-path search system. The **visual search path** uses contrastive learning to train a text encoder (BERT) and video encoder (ViT) that map queries and videos into a shared embedding space. At serving time, we use ANN for fast retrieval. The **text search path** uses Elasticsearch on video metadata. A fusing layer combines results from both paths with weighted scoring, and a re-ranking service applies business logic. For evaluation, I'd use MRR offline and total watch time online."

### Key Design Decisions to Defend

1. **Why dual-path (visual + text)?** -- Videos with bad titles but good content would be missed by text-only search. Videos with great titles but irrelevant content would be caught by visual search disagreeing.

2. **Why frame-level over video-level encoder?** -- Much faster, lower compute cost. Temporal understanding isn't critical for most search queries.

3. **Why contrastive learning?** -- Naturally maps text and video into the same embedding space. Scales well with batch size (more negatives = better learning).

4. **Why MRR over Precision@k?** -- With only one relevant video per query in our dataset, precision@k gives artificially low values. MRR directly measures how quickly we find the relevant video.

5. **Why weighted fusion over ML fusion?** -- Simpler, faster, no additional training needed. ML fusion is better but adds complexity and latency.

6. **Why auto-tagging?** -- Many videos have no or poor manual tags. Auto-generated tags improve text search coverage.

---

## Module Files

| File | Description |
|------|-------------|
| `README.md` | This comprehensive guide |
| `01_video_search_system_design.ipynb` | System design: problem definition, metrics, architecture, data pipeline, features, model choices |
| `02_multimodal_embeddings.ipynb` | Text understanding, video understanding, multimodal embeddings, two-tower retrieval |
| `03_ranking_and_serving.ipynb` | Candidate generation, learning to rank, re-ranking, serving infrastructure, A/B testing |
| `04_interview_walkthrough.ipynb` | Complete mock interview walkthrough from start to finish |

---

## References

1. Elasticsearch - https://www.tutorialspoint.com/elasticsearch/
2. Preprocessing text data - https://huggingface.co/docs/transformers/
3. NFKD normalization - https://unicode.org/reports/tr15/
4. Tokenization algorithms
5. Feature hashing
6. Text embeddings
7. TF-IDF mathematics
8. Word2Vec - Mikolov et al.
9. CBOW model
10. Skip-gram model
11. BERT - Devlin et al.
12. GPT-3 - Brown et al.
13. BLOOM
14. Transformer architecture - Vaswani et al., "Attention Is All You Need"
15. 3D Convolutions for video
16. ViT (Vision Transformer) - Dosovitskiy et al.
17. Query understanding
18. Multimodal search systems
19. Multilingual search
20. Near-duplicate video detection
21. Head/torso/tail query strategies
22. Freshness and popularity in search
23-25. Real-world search system references
