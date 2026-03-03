# Vision-Language Models

In 2021, a model called CLIP did something that no one expected. Without being trained to identify a single animal species, it could look at a photo of a pangolin and tell you it was a pangolin — just by comparing the photo to the words "a photo of a pangolin." It had never seen a labeled pangolin training example. How?

The answer is one of the most elegant ideas in AI: teach a model to connect images and text in the same space, and it can do things you never trained it to do.

**Before you start, you need to know:**
- What an embedding is (a list of numbers that captures meaning) — covered in [multimodal README](../README.md)
- What a neural network does at a high level — covered in [00-neural-networks](../../00-neural-networks/)

---

## The Analogy: Showing a Photo to a Friend

Imagine showing a photo to a friend and asking "What's happening in this picture?" Your friend uses both their **eyes** (vision) and their **language skills** (words) to give you an answer.

Vision-language models do the same thing — they combine understanding images with understanding text.

**What the analogy gets right:**
- Your friend processes two different types of information (the image they see, the words they know) and connects them
- The model also has two separate parts — one for images, one for text — and learns to connect their outputs
- Your friend can answer questions about photos they have never seen before, and so can the model

**The concept in plain words:**
A vision-language model turns images and text into lists of numbers (embeddings) that live in the same space. A photo of a dog and the sentence "a photo of a dog" end up close together in that space. A photo of a dog and the sentence "a photo of a car" end up far apart. Once images and text live in the same space, you can compare them, search through them, and even classify images using only text descriptions.

**Where the analogy breaks down:** Your friend deeply understands what a dog is — they know dogs bark, have four legs, and are pets. The model does not understand any of this. It only knows that photos labeled "dog" tend to produce embeddings close to the text "dog." It learned patterns from 400 million image-text pairs, not from understanding the world.

---

## CLIP: The Breakthrough

**CLIP (Contrastive Language-Image Pre-training)** by OpenAI (2021) changed how we think about vision-language models. It has two separate parts:

1. **An image encoder** — takes a photo and outputs a list of numbers
2. **A text encoder** — takes a sentence and outputs a list of numbers

Both lists live in the same space. If the photo and sentence describe the same thing, their lists will be close together.

```
   Image: [photo of a dog]     Text: "a photo of a dog"
           |                          |
           v                          v
     [Image Encoder]           [Text Encoder]
           |                          |
           v                          v
     Image Embedding            Text Embedding
     [0.3, 0.7, -0.2, ...]     [0.31, 0.69, -0.18, ...]

     These are CLOSE together (same concept)
     "A photo of a cat" would be FAR away
```

### How CLIP Learns: Contrastive Training

CLIP learns by seeing matching and non-matching pairs. In each training batch, the model sees several images and several texts. It learns to push matching pairs together and non-matching pairs apart.

```
Training batch with 4 image-text pairs:

                    Text 1      Text 2      Text 3      Text 4
                   "a dog"    "a sunset"  "a pizza"   "a car"
Image 1 (dog)     [MATCH]    [mismatch]  [mismatch]  [mismatch]
Image 2 (sunset)  [mismatch] [MATCH]     [mismatch]  [mismatch]
Image 3 (pizza)   [mismatch] [mismatch]  [MATCH]     [mismatch]
Image 4 (car)     [mismatch] [mismatch]  [mismatch]  [MATCH]

Goal: Make diagonal scores (matches) high
      Make off-diagonal scores (mismatches) low
```

### Zero-Shot Classification: The Surprising Superpower

CLIP can classify images into categories **it has never been trained on**, just by comparing image embeddings to text embeddings of category names.

```
   1. Encode the image:
      [photo of a husky] --> image_embedding

   2. Encode each category as text:
      "a photo of a cat"   --> text_embedding_1
      "a photo of a dog"   --> text_embedding_2
      "a photo of a bird"  --> text_embedding_3

   3. Compare similarities:
      Image vs "cat":  similarity = 0.15
      Image vs "dog":  similarity = 0.92  <-- HIGHEST
      Image vs "bird": similarity = 0.08

   4. Result: "This is a dog" (92% confident)

   No training on these specific categories was needed!
```

---

## Beyond CLIP

CLIP can match images with text, but it cannot generate text. Newer models build on CLIP's idea:

- **BLIP / BLIP-2** — Can also *generate* captions for images, not just match them
- **LLaVA** — Connects a vision encoder to an LLM so you can have conversations about images
- **GPT-4V / Claude Vision / Gemini** — Commercial models that natively understand both text and images as part of their core training

---

## Quick Check — Can You Answer These?

- What does CLIP do differently from a regular image classifier?
- Why is zero-shot classification possible with CLIP but not with a standard image classifier?
- In the contrastive training matrix above, what does it mean for a diagonal score to be high?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## What You Just Learned

You now understand the core idea behind image search engines, visual question answering, image captioning, content moderation systems, and AI accessibility tools. The principle — map images and text to the same embedding space, then compare — is used in production at Google, OpenAI, Meta, and thousands of other companies. CLIP alone has been cited over 10,000 times and inspired an entire family of models.

Ready to go deeper? The math behind contrastive loss, failure modes, and interview questions are in [vision-language-interview.md](./vision-language-interview.md).

---

[Back to Multimodal](../README.md)
