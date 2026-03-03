# 5. Multimodal Models

You can read a book with your eyes closed if someone reads it to you. You can describe a photo to someone who can't see it. Your brain connects what you see, hear, and read — automatically. What if an AI could do the same thing?

That is what multimodal models do. And how they pull it off is one of the most surprising ideas in modern AI.

**Before you start, you need to know:**
- What an embedding is (a list of numbers that captures meaning) — covered in [01-transformers/architecture/attention-mechanisms.md](../01-transformers/architecture/attention-mechanisms.md)
- What a neural network does at a high level — covered in [00-neural-networks](../00-neural-networks/)

---

## The Analogy: A Translator at the United Nations

Imagine the United Nations. Delegates speak dozens of different languages. They cannot understand each other directly. But there is a team of translators who convert every speech into a shared language that everyone can read on their screens.

A multimodal model works the same way. Images, text, and audio are like delegates who "speak" completely different languages. The model acts as a translator — it converts each type of input into a shared list of numbers (an embedding) where similar meanings end up close together.

**What the analogy gets right:**
- Each modality (text, image, audio) starts in its own "language" that the others cannot read
- The model converts all of them into a shared space where meanings can be compared
- A photo of a dog and the words "a dog" end up in the same place in that shared space

**The concept in plain words:**
A multimodal model takes different types of input — text, images, audio — and maps them all into the same number space. Once they are in the same space, the model can compare them, match them, and reason across them. "Does this photo match this caption?" becomes a math problem: "Are these two lists of numbers close together?"

**Where the analogy breaks down:** Real UN translators understand meaning deeply before translating. Multimodal models learn the translation purely from seeing millions of paired examples (photos with captions, audio with transcripts). They never "understand" — they find patterns in data.

---

## Key Concepts

### Cross-Modal Alignment

The central challenge: how do you get a model to understand that a **photo of a cat**, the **word "cat"**, and the **sound of meowing** all refer to the same concept?

This is called **alignment** — mapping different types of input into a shared space where related concepts are close together.

```
   Text: "cat"        --+
                        |
   Image: [photo]     --+--> SHARED EMBEDDING SPACE
                        |    (related things are close together)
   Audio: [meow]      --+
```

### Fusion Strategies

How do you combine information from different types of input?

| Strategy | How It Works | Example |
|----------|-------------|---------|
| **Early Fusion** | Combine raw inputs before processing | Put image pixels and text tokens together, then feed them into one model |
| **Late Fusion** | Process each type separately, combine at the end | Separate image and text models, merge their outputs |
| **Cross-Attention Fusion** | Let the types "look at" each other during processing | Image features attend to text tokens and the other way around |

```
Early Fusion:       Late Fusion:        Cross-Attention:
  Image + Text        Image   Text       Image <--> Text
      |                 |       |            |         |
      v                 v       v            v         v
   [Model]          [Model] [Model]     [Model A] [Model B]
      |                 \     /          (they communicate
      v                  v   v            during processing)
   Output              [Merge]                |
                          |                   v
                        Output              Output
```

---

## Quick Check — Can You Answer These?

- In your own words: what does it mean for two different types of input (like a photo and a sentence) to be "aligned"?
- Why is it useful to map images and text into the same embedding space?
- What is the difference between early fusion and late fusion?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## What You Just Learned

You now understand the core idea behind GPT-4V, Claude Vision, Gemini, CLIP, and Whisper. Every one of these models takes different types of input and maps them into a shared space where meanings can be compared. That single idea — cross-modal alignment — powers image search, voice assistants, image captioning, and dozens of other applications you use every day.

---

## Study Plan

```
    START HERE
        |
        v
+---------------------------+
|  1. This README            |  Understand what multimodal means
|     (you are here)         |  and why it matters
+-----------+---------------+
            |
            v
+---------------------------+
|  2. Vision-Language        |  CLIP, image captioning, visual QA
|     (vision-language/)     |  -- how AI "sees" and "reads"
+-----------+---------------+
            |
            v
+---------------------------+
|  3. Audio-Language         |  Speech recognition, text-to-speech
|     (audio-language/)      |  -- how AI "hears" and "speaks"
+---------------------------+
```

---

## Coverage Map

### Vision-Language

| Topic | Depth | Files |
|-------|-------|-------|
| Vision-Language Models — CLIP, contrastive learning, zero-shot classification | [Core] | [README.md](./vision-language/README.md) · [vision-language-interview.md](./vision-language/vision-language-interview.md) · [01_vision_language.ipynb](./vision-language/01_vision_language.ipynb) · [01_vision_language_experiments.ipynb](./vision-language/01_vision_language_experiments.ipynb) |

### Audio-Language

| Topic | Depth | Files |
|-------|-------|-------|
| Audio-Language Models — spectrograms, Whisper, speech recognition | [Core] | [README.md](./audio-language/README.md) · [audio-language-interview.md](./audio-language/audio-language-interview.md) · [01_audio_language.ipynb](./audio-language/01_audio_language.ipynb) · [01_audio_language_experiments.ipynb](./audio-language/01_audio_language_experiments.ipynb) |

---

## Key Terms

| Term | Simple Explanation |
|------|-------------------|
| **Modality** | A type of data: text, image, audio, video |
| **Multimodal** | Using more than one type of data at a time |
| **Embedding** | A list of numbers that captures the meaning of some input |
| **Alignment** | Teaching the model that a photo of X and the word "X" are the same concept |
| **Contrastive Learning** | Training by showing the model matching pairs (correct) and non-matching pairs (incorrect) |
| **Encoder** | The part of the model that reads and processes an input (text encoder, image encoder, etc.) |
| **Zero-shot** | Using a model on a task it was not trained for directly |

---

## Key Papers

- **CLIP: Learning Transferable Visual Models From Natural Language Supervision** — Radford et al., 2021
  - Showed you can align images and text by training on 400M image-text pairs from the internet
- **BLIP-2: Bootstrapping Language-Image Pre-training** — Li et al., 2023
  - Efficient method to connect frozen image encoders with frozen LLMs
- **Flamingo: a Visual Language Model for Few-Shot Learning** — Alayrac et al., 2022
  - Showed multimodal models can learn new tasks from just a few examples
- **Whisper: Robust Speech Recognition via Large-Scale Weak Supervision** — Radford et al., 2022
  - OpenAI's speech recognition model trained on 680K hours of audio

---

[Back to Main](../README.md) | [Previous: Prompt Engineering](../04-prompt-engineering/README.md) | [Next: Evaluation](../06-evaluation/README.md)
