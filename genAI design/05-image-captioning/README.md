# Chapter 05: Image Captioning 📸➡️📝

> **Teaching AI to play Pictionary in reverse -- it sees the picture and writes the description.**

---

## What Is Image Captioning?

Imagine you show a photo to your friend and ask, "What's happening here?" Your friend looks at the picture, understands what's in it, and says something like:

> *"A golden retriever is catching a frisbee on a sunny beach."*

That's image captioning! The AI needs to:
1. **See** the image (understand objects, actions, relationships, scenes)
2. **Describe** it in natural language (form a grammatically correct, accurate sentence)

It's one of the hardest AI tasks because it bridges two completely different worlds: **vision** (pixels) and **language** (words). The AI must be fluent in both.

### Real-World Examples 🌍

| Product | How It Uses Image Captioning |
|---------|------------------------------|
| **iPhone Photos** | Auto-generates descriptions for search ("beach sunset") |
| **Instagram / Facebook** | Alt-text for accessibility (screen readers for visually impaired users) |
| **Google Lens** | Describes scenes for users who can't see them |
| **Self-driving cars** | Describes the driving scene for logging and debugging |
| **Medical imaging** | Generates radiology reports from X-rays / MRIs |
| **E-commerce** | Auto-generates product descriptions from product photos |

---

## Key Concepts 🧠

### 1. The Encoder-Decoder Architecture (The Brain Split)

Image captioning uses a **two-part brain**:

```
                    IMAGE CAPTIONING SYSTEM
                    =======================

    📷 Image                                    📝 Caption
       |                                           ^
       v                                           |
  +-----------+      bridge        +-----------+
  |  IMAGE    | -- (features) -->  |   TEXT    |
  |  ENCODER  |   cross-attention  |  DECODER  |
  | "What do  |                    | "How do I |
  |  I see?"  |                    |  say it?" |
  +-----------+                    +-----------+

  Sees the picture                 Generates words
  (CNN or ViT)                     one at a time
                                   (Transformer decoder)
```

**Analogy** 🎨: Think of a courtroom sketch artist and a news reporter working together. The sketch artist (encoder) carefully observes the scene and creates a detailed visual summary. The reporter (decoder) takes that summary and writes a news story about it, word by word.

---

### 2. Image Encoders: CNN vs Vision Transformer (ViT) 🔬

The encoder's job is to look at the image and extract **meaningful features** -- "there's a dog", "it's on a beach", "the sky is blue."

#### Option A: CNN Encoder (The Classic) 🏛️

CNNs (Convolutional Neural Networks) process images with sliding filters that detect edges, textures, shapes, and eventually whole objects.

```
CNN Encoder: Hierarchical Feature Extraction
=============================================

  Raw Image (224x224x3)
       |
       v
  [Conv Layer 1] → edges, corners
       |
       v
  [Conv Layer 2] → textures, patterns
       |
       v
  [Conv Layer 3] → parts (ears, wheels, windows)
       |
       v
  [Conv Layer 4] → whole objects (dog, car, house)
       |
       v
  Feature Map (7x7x2048)  ← "Here's everything I see!"

  Popular choices: ResNet-101, Inception V3
```

**Analogy** 🔍: A CNN is like reading a book with a magnifying glass, starting with individual letters, then words, then sentences, then understanding the whole paragraph.

#### Option B: Vision Transformer (ViT) Encoder (The Modern Way) 🚀

ViTs treat an image like a sentence -- they chop it into **patches** (like words) and process them with a Transformer.

```
ViT Encoder: Patchify + Transformer
=====================================

  Raw Image (224x224)
       |
  Step 1: PATCHIFY (cut into puzzle pieces)
       |
       v
  ┌────┬────┬────┬────┐
  │ P1 │ P2 │ P3 │ P4 │    Each patch = 16x16 pixels
  ├────┼────┼────┼────┤    Total patches = (224/16)² = 196
  │ P5 │ P6 │ P7 │ P8 │
  ├────┼────┼────┼────┤    Like cutting a photo into
  │ P9 │P10 │P11 │P12 │    196 puzzle pieces! 🧩
  ├────┼────┼────┼────┤
  │P13 │P14 │P15 │P16 │
  └────┴────┴────┴────┘
       |
  Step 2: FLATTEN + LINEAR PROJECTION
       |
       v
  Each 16x16x3 patch → 768-dim vector (like a word embedding!)
       |
  Step 3: ADD POSITIONAL ENCODING
       |
       v
  [P1+pos1, P2+pos2, P3+pos3, ... P196+pos196]
  (So the Transformer knows WHERE each patch came from)
       |
  Step 4: TRANSFORMER ENCODER
       |
       v
  Self-attention across ALL patches
  (every puzzle piece can "look at" every other piece)
       |
       v
  Rich feature vectors for each patch
```

**Analogy** 🧩: Imagine you have a jigsaw puzzle. Instead of putting it together piece by piece (CNN style), you lay ALL the pieces on a table and look at them all at once, figuring out how each piece relates to every other piece. That's the ViT approach!

#### CNN vs ViT: Head-to-Head ⚔️

| Aspect | CNN (ResNet) | ViT |
|--------|-------------|-----|
| How it sees | Sliding window, local → global | All patches at once, global from start |
| Inductive bias | Built-in: locality, translation equivariance | Minimal: learns everything from data |
| Data hunger | Works with moderate data | Needs LOTS of data (or pretraining) |
| Positional info | Implicit from convolution structure | Explicit positional encoding needed |
| Scalability | Harder to scale | Scales beautifully with more data/compute |
| Modern winner? | Still good, but... | ViT wins when you have enough data 🏆 |

---

### 3. Patchify: Turning Images into "Sentences" 🧩

Patchify is the magic trick that lets Transformers eat images. It's the bridge between the pixel world and the sequence world.

**How it works:**
1. Take an image of size H × W × C (e.g., 224 × 224 × 3)
2. Divide it into non-overlapping patches of size P × P (e.g., 16 × 16)
3. Flatten each patch into a 1D vector: 16 × 16 × 3 = 768 values
4. Apply a linear projection (learned) to get patch embeddings
5. Add positional encodings so the model knows spatial layout

**Positional Encoding for Images:**

```
1D Positional Encoding (simpler):
  Just number the patches left-to-right, top-to-bottom:
  [1, 2, 3, 4, 5, 6, 7, 8, 9, ... 196]
  (Used in original ViT -- surprisingly, this works well!)

2D Positional Encoding (fancier):
  Give each patch a (row, col) coordinate:
  (0,0) (0,1) (0,2) ...
  (1,0) (1,1) (1,2) ...
  ...
  (Preserves spatial structure better for some tasks)
```

---

### 4. Cross-Attention: The Bridge Between Vision and Language 🌉

Cross-attention is how the text decoder "asks questions" about the image. It's the critical mechanism that connects what the model **sees** with what it **says**.

```
CROSS-ATTENTION IN ACTION
===========================

  Text Decoder is generating: "A dog is playing on the ___"

  Query (Q):  Comes from the TEXT decoder
              "What word should come next?"

  Key (K):    Comes from the IMAGE encoder
              "Here are all the visual features I extracted"

  Value (V):  Also from the IMAGE encoder
              "Here's the detailed info about each visual feature"

  Attention = softmax(Q · K^T / √d) · V

  Result: The decoder focuses on the beach-related patches
          and generates "beach" 🏖️
```

**Analogy** 🎯: Imagine you're writing an essay about a painting in a museum. Cross-attention is like glancing up at the painting every time you need to write the next sentence. Your eyes focus on the relevant part of the painting (attention weights) to decide what to write next.

---

### 5. CIDEr Metric: The Gold Standard for Caption Evaluation 🏅

CIDEr (Consensus-based Image Description Evaluation) is THE metric for measuring caption quality. It answers: "Does this caption describe the image the way most humans would?"

#### How CIDEr Works (Step by Step):

```
CIDEr CALCULATION
=================

Step 1: Extract n-grams (1-gram, 2-gram, 3-gram, 4-gram)
  Candidate: "A dog plays on the beach"
  → 1-grams: [A, dog, plays, on, the, beach]
  → 2-grams: [A dog, dog plays, plays on, on the, the beach]
  → 3-grams: [A dog plays, dog plays on, plays on the, on the beach]
  → 4-grams: [A dog plays on, dog plays on the, plays on the beach]

Step 2: Compute TF-IDF for each n-gram
  TF  = How often this n-gram appears in THIS caption
  IDF = log(total images / images where this n-gram appears)
  → Common words like "a" get LOW weight (appear everywhere)
  → Specific words like "frisbee" get HIGH weight (rare & informative)

Step 3: Create TF-IDF vector for candidate and each reference

Step 4: Cosine similarity between candidate and each reference
  sim = (candidate · reference) / (|candidate| × |reference|)

Step 5: Average across all references and all n-gram sizes
  CIDEr = (1/4) × Σ [average cosine similarity for n-gram size n]
```

#### Why CIDEr is Better Than BLEU for Captioning:

| Feature | BLEU | CIDEr |
|---------|------|-------|
| Weights words equally? | Yes 😬 | No! Uses TF-IDF 🎯 |
| "a dog on a beach" vs "a cat on a beach" | High score (most words match) | Lower score ("dog" vs "cat" is important) |
| Rewards specific descriptions? | Not really | Yes! Rare, informative words score higher |
| Consensus with humans? | Moderate | High (designed for it!) |

**Analogy** 📊: BLEU is like grading an essay by counting how many words match the answer key. CIDEr is like grading by checking if the IMPORTANT words match -- it knows that getting "golden retriever" right matters more than getting "the" right.

---

## The Full System Design 🏗️

```
IMAGE CAPTIONING: END-TO-END SYSTEM
=====================================

┌─────────────────────────────────────────────────────┐
│                   INPUT PIPELINE                     │
│                                                     │
│  📷 Raw Image                                       │
│     │                                               │
│     ├── Resize to 224×224 (or 384×384)             │
│     ├── Normalize (ImageNet mean/std)               │
│     ├── Data augmentation (random crop, flip)       │
│     │                                               │
│     v                                               │
│  Preprocessed Image Tensor                          │
└─────────────────┬───────────────────────────────────┘
                  │
                  v
┌─────────────────────────────────────────────────────┐
│                 MODEL PIPELINE                       │
│                                                     │
│  ┌───────────┐      ┌──────────────────────┐       │
│  │  Image    │      │   Text Decoder       │       │
│  │  Encoder  │─────>│   (autoregressive)   │       │
│  │  (ViT)    │cross │                      │       │
│  │           │attn  │ <BOS> → A → dog →    │       │
│  └───────────┘      │  → is → playing →    │       │
│                     │  → fetch → <EOS>     │       │
│                     └──────────────────────┘       │
└─────────────────┬───────────────────────────────────┘
                  │
                  v
┌─────────────────────────────────────────────────────┐
│                OUTPUT PIPELINE                       │
│                                                     │
│  Raw tokens → Detokenize → Post-process             │
│     │                                               │
│     ├── Remove <BOS>/<EOS> tokens                   │
│     ├── Capitalize first letter                     │
│     ├── Add period at end                           │
│     ├── Filter profanity / harmful content          │
│     ├── Length check (too short? regenerate)         │
│     │                                               │
│     v                                               │
│  📝 "A dog is playing fetch in the park."           │
└─────────────────────────────────────────────────────┘
```

### Training Pipeline 🎓

```
TRAINING STRATEGY
==================

Phase 1: PRETRAINING (learn general knowledge)
  ├── Image encoder: pretrained on ImageNet (ViT-B/16, ViT-L/14)
  ├── Text decoder: pretrained on large text corpus
  └── Why? Starting from scratch would need 100x more data!

Phase 2: SUPERVISED FINETUNING (learn to caption)
  ├── Dataset: image-caption pairs (COCO Captions: 330K images, 5 captions each)
  ├── Loss: Cross-entropy on next-token prediction
  ├── Teacher forcing: feed ground-truth tokens during training
  └── Learning rate: small! Don't destroy pretrained knowledge

Phase 3: (Optional) REINFORCEMENT LEARNING
  ├── Optimize directly for CIDEr score
  ├── SCST (Self-Critical Sequence Training)
  └── Why? Cross-entropy doesn't perfectly correlate with CIDEr
```

### Inference: Beam Search for Better Captions 🔍

```
BEAM SEARCH (beam_width = 3)
=============================

Step 1: <BOS>
  → "A"    (score: -0.2)
  → "The"  (score: -0.3)
  → "Two"  (score: -0.8)

Step 2: Expand each
  → "A dog"      (score: -0.5)  ✅ keep
  → "A cat"      (score: -0.7)  ✅ keep
  → "The dog"    (score: -0.6)  ✅ keep
  → "A bird"     (score: -0.9)  ❌ pruned
  → "The cat"    (score: -1.0)  ❌ pruned
  → "Two dogs"   (score: -1.1)  ❌ pruned

Step 3: Continue expanding top 3...
  → "A dog is playing fetch." (WINNER! 🏆)

Why beam search instead of greedy?
  Greedy: picks best word at EACH step → can miss better overall sequences
  Beam:   keeps multiple candidates → finds globally better captions
```

---

## Interview Cheat Sheet 🎯

### Must-Know Talking Points

#### 1. "Walk me through an image captioning system."

> "It's an encoder-decoder architecture. The encoder -- typically a Vision Transformer or CNN like ResNet -- processes the image into a sequence of feature vectors. The decoder is an autoregressive Transformer that generates the caption one token at a time. The bridge between them is cross-attention: at each decoding step, the decoder attends to the image features to decide what word comes next. We pretrain both components separately, then finetune end-to-end on image-caption pairs using cross-entropy loss with teacher forcing."

#### 2. "Why ViT over CNN?"

> "ViTs split the image into patches and process them with self-attention, which captures global relationships from the very first layer -- a CNN needs many layers to build up a global receptive field. ViTs also scale better with data and compute. The tradeoff is they need more training data due to fewer inductive biases (no built-in locality or translation equivariance). In practice, we use pretrained ViTs (like ViT-L/14 from CLIP) which solves the data problem."

#### 3. "Explain CIDEr."

> "CIDEr measures caption quality using TF-IDF weighted n-gram similarity. It extracts n-grams (1 through 4) from both the candidate and reference captions, weights them by TF-IDF -- so informative words like 'golden retriever' count more than common words like 'the' -- then computes cosine similarity between TF-IDF vectors. It averages across all reference captions and all n-gram sizes. This consensus-based approach correlates better with human judgment than BLEU because it rewards specific, informative descriptions."

#### 4. "How do you handle the vision-language gap?"

> "Three strategies: (1) Cross-attention layers where text queries attend to image keys/values -- this is the standard approach. (2) A learned projection layer that maps image features into the same embedding space as text tokens. (3) Prefix tuning where image features are prepended as 'visual tokens' to the decoder input, so the decoder treats them like any other tokens in the sequence."

#### 5. "What are the failure modes?"

> "Object hallucination -- the model describes objects not in the image (says 'a cat on a mat' when there's no cat). Attribute errors -- wrong colors, sizes, or counts. Repetitive/generic captions -- 'a photo of a room' instead of specific details. Bias from training data -- gender/racial stereotypes from COCO. Mitigation: constrained decoding, faithfulness metrics, debiasing datasets."

#### 6. "How do you evaluate beyond CIDEr?"

> "BLEU (precision-focused n-gram overlap), ROUGE (recall-focused), METEOR (synonym-aware, stemming). But all automated metrics have limits -- human evaluation is the gold standard. Key human dimensions: adequacy (does it describe the image?), fluency (is it grammatical?), and specificity (is it detailed enough?). At scale, we use human eval on a sample and automated metrics for the rest."

### Common Follow-Up Questions

| Question | Key Points to Hit |
|----------|-------------------|
| "How do you handle varying image sizes?" | Resize + center crop, or use flexible patch sizes |
| "What's teacher forcing?" | During training, feed ground-truth tokens (not model predictions) as decoder input |
| "How do you prevent hallucination?" | Constrained decoding, object detection verification, faithfulness rewards |
| "BLEU vs CIDEr?" | BLEU weights all n-grams equally; CIDEr uses TF-IDF to emphasize informative words |
| "What dataset would you use?" | COCO Captions (330K images, 5 caps each), Conceptual Captions (3M), SBU Captions (1M) |
| "How do you scale inference?" | Batch beam search, KV-cache for decoder, quantize encoder, distill to smaller model |
| "What about multilingual captions?" | Multilingual decoder, or translate after generating English caption |

---

## Notebook

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | [Image Captioning System](01_image_captioning_system.ipynb) | ViT patchification, cross-attention, CIDEr metric, full system walkthrough |

---

## Key Terms

| Term | Plain-English Meaning |
|------|-----------------------|
| **Patchify** | Cutting an image into a grid of small squares (patches) so a Transformer can process them like words |
| **ViT** | Vision Transformer -- applies the Transformer architecture to image patches instead of words |
| **Cross-attention** | The mechanism where the text decoder "looks at" the image features to decide what word to generate next |
| **CIDEr** | A metric that measures caption quality using TF-IDF weighted n-gram similarity against human references |
| **Beam search** | A decoding strategy that keeps multiple candidate captions and picks the best overall sequence |
| **Teacher forcing** | During training, feeding the ground-truth previous word (instead of the model's own prediction) to the decoder |
| **Multi-modal** | A system that understands multiple types of data (e.g., both images AND text) |
| **TF-IDF** | Term Frequency - Inverse Document Frequency: a way to weight words by how informative they are |
| **Hallucination** | When the model describes objects or details that aren't actually in the image |
| **Autoregressive** | Generating output one token at a time, where each token depends on all previous tokens |

---

## References 📚

- [An Image is Worth 16x16 Words (ViT paper)](https://arxiv.org/abs/2010.11929) -- Dosovitskiy et al., 2020
- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) -- Xu et al., 2015
- [CIDEr: Consensus-based Image Description Evaluation](https://arxiv.org/abs/1411.5726) -- Vedantam et al., 2015
- [BLIP-2](https://arxiv.org/abs/2301.12597) -- Li et al., 2023
- [MS COCO Captions Dataset](https://cocodataset.org/)

---

[Back to GenAI Design Guide](../README.md)
