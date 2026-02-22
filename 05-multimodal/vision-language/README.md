# Vision-Language Models

## What Are Vision-Language Models?

Imagine showing a photo to a friend and asking "What's happening in this picture?"
Your friend uses both their **eyes** (vision) and their **language skills** (words)
to give you an answer.

**Vision-language models** do the same thing -- they combine computer vision
(understanding images) with language understanding (reading and writing text).

```
+-------------------------------------------------------------------+
|              What Vision-Language Models Can Do                    |
|                                                                   |
|   Image Captioning:                                                |
|     [photo of sunset over ocean] --> "A beautiful sunset over     |
|                                       the Pacific Ocean"          |
|                                                                   |
|   Visual Question Answering (VQA):                                 |
|     [photo of kitchen] + "How many chairs are there?" --> "Four"  |
|                                                                   |
|   Image-Text Matching:                                             |
|     "A dog playing fetch" + [5 photos] --> picks the right photo  |
|                                                                   |
|   Zero-Shot Classification:                                        |
|     [photo of animal] + ["cat", "dog", "bird"] --> "dog" (85%)    |
+-------------------------------------------------------------------+
```

---

## CLIP: The Breakthrough Model

**CLIP (Contrastive Language-Image Pre-training)** by OpenAI (2021) changed
everything. It's the foundational model for modern vision-language AI.

### How CLIP Works

CLIP learns to connect images and text by training on **400 million
image-text pairs** scraped from the internet.

```
+-------------------------------------------------------------------+
|                    CLIP Architecture                               |
|                                                                   |
|   Image: [photo of a dog]     Text: "a photo of a dog"           |
|           |                          |                            |
|           v                          v                            |
|     [Image Encoder]           [Text Encoder]                      |
|     (ViT or ResNet)           (Transformer)                       |
|           |                          |                            |
|           v                          v                            |
|     Image Embedding            Text Embedding                     |
|     [0.3, 0.7, -0.2, ...]     [0.31, 0.69, -0.18, ...]          |
|                                                                   |
|     These should be CLOSE together (same concept)                 |
|     "A photo of a cat" would be FAR away                         |
+-------------------------------------------------------------------+
```

### Contrastive Learning: How CLIP Trains

CLIP uses **contrastive learning** -- it learns by seeing matching and
non-matching pairs.

```
Training batch with 4 image-text pairs:

                    Text 1      Text 2      Text 3      Text 4
                   "a dog"    "a sunset"  "a pizza"   "a car"
Image 1 (dog)     [MATCH]    [mismatch]  [mismatch]  [mismatch]
Image 2 (sunset)  [mismatch] [MATCH]     [mismatch]  [mismatch]
Image 3 (pizza)   [mismatch] [mismatch]  [MATCH]     [mismatch]
Image 4 (car)     [mismatch] [mismatch]  [mismatch]  [MATCH]

Goal: Maximize similarity for diagonal (matches)
      Minimize similarity for off-diagonal (mismatches)
```

### Why CLIP Is Special: Zero-Shot Classification

CLIP can classify images into categories **it has never been trained on**,
just by comparing image embeddings to text embeddings of category names.

```
+-------------------------------------------------------------------+
|              CLIP Zero-Shot Classification                         |
|                                                                   |
|   1. Encode the image:                                             |
|      [photo of a husky] --> image_embedding                       |
|                                                                   |
|   2. Encode each category as text:                                 |
|      "a photo of a cat"   --> text_embedding_1                    |
|      "a photo of a dog"   --> text_embedding_2                    |
|      "a photo of a bird"  --> text_embedding_3                    |
|                                                                   |
|   3. Compare similarities:                                         |
|      Image vs "cat":  similarity = 0.15                           |
|      Image vs "dog":  similarity = 0.92  <-- HIGHEST!             |
|      Image vs "bird": similarity = 0.08                           |
|                                                                   |
|   4. Result: "This is a dog" (92% confident)                      |
|                                                                   |
|   No training on these specific categories was needed!            |
+-------------------------------------------------------------------+
```

---

## Beyond CLIP: Modern Vision-Language Models

### BLIP / BLIP-2

**BLIP (Bootstrapping Language-Image Pre-training)** extends CLIP to also
**generate** text, not just match images with text.

```
CLIP:   Image + Text --> "Do these match?" (yes/no)
BLIP:   Image --> "Describe this image" (generates caption)
BLIP-2: Image --> Connects to an LLM for complex reasoning
```

### LLaVA (Large Language-and-Vision Assistant)

Connects a vision encoder to an LLM, creating a model that can have
conversations about images.

```
User: [uploads photo of a messy room] "What should I clean first?"
LLaVA: "I'd start with the clothes on the floor, then organize
        the desk. The bookshelf looks manageable after that."
```

### GPT-4V / Claude Vision / Gemini

The latest commercial models that natively understand both text and images
as part of their core training, enabling sophisticated visual reasoning.

---

## The Image Encoder: How AI "Sees"

Vision-language models use an **image encoder** to convert images into
embeddings. The two main approaches:

| Encoder | How It Works | Used In |
|---------|-------------|---------|
| **CNN (ResNet)** | Slides filters across the image to detect patterns | Older CLIP models |
| **ViT (Vision Transformer)** | Splits image into patches, processes like text tokens | Modern CLIP, BLIP-2 |

```
+-------------------------------------------------------------------+
|          Vision Transformer (ViT): How It Works                   |
|                                                                   |
|   1. Split image into 16x16 patches (like puzzle pieces)          |
|                                                                   |
|      [patch1][patch2][patch3][patch4]                              |
|      [patch5][patch6][patch7][patch8]                              |
|      [patch9][...  ][...  ][patch16]                              |
|                                                                   |
|   2. Flatten each patch into a vector                              |
|                                                                   |
|   3. Process through a Transformer (same as text!)                |
|                                                                   |
|   4. Output: one embedding that captures the whole image          |
|                                                                   |
|   Key insight: Transformers work for images too!                  |
|   Just treat image patches like words in a sentence.              |
+-------------------------------------------------------------------+
```

---

## Applications

| Application | Description | Models Used |
|-------------|-------------|-------------|
| **Image search** | Find images matching a text query | CLIP |
| **Image captioning** | Generate text describing an image | BLIP, LLaVA |
| **Visual QA** | Answer questions about images | BLIP-2, GPT-4V |
| **Content moderation** | Detect inappropriate images | CLIP + classifier |
| **Image generation** | Create images from text descriptions | DALL-E, Stable Diffusion (use CLIP) |
| **Accessibility** | Describe images for visually impaired users | BLIP, GPT-4V |

---

## Summary

```
+------------------------------------------------------------------+
|           Vision-Language Models Cheat Sheet                      |
|                                                                  |
|  What:     AI that understands both images and text              |
|  Key model: CLIP (contrastive learning on image-text pairs)      |
|  Key idea: Map images and text to the same embedding space       |
|                                                                  |
|  Evolution:                                                      |
|    CLIP (2021) -- match images with text                         |
|    BLIP (2022) -- generate text from images                      |
|    LLaVA (2023) -- have conversations about images               |
|    GPT-4V (2023) -- native multimodal understanding              |
+------------------------------------------------------------------+
```

---

## Further Reading

- **CLIP** -- Radford et al., 2021 -- The foundational paper
- **BLIP-2** -- Li et al., 2023 -- Efficient vision-language pre-training
- **LLaVA** -- Liu et al., 2023 -- Visual instruction tuning
- **An Image is Worth 16x16 Words (ViT)** -- Dosovitskiy et al., 2020 -- Vision Transformers

---

[Back to Multimodal](../README.md)
