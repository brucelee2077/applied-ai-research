# 5. Multimodal Models

## What Are Multimodal Models?

Imagine a person who can only read text -- no pictures, no sounds. They'd miss out
on a huge part of the world. Now imagine someone who can see, hear, AND read.
That's the difference between a text-only AI and a **multimodal** AI.

**Multimodal models** are AI systems that can understand and work with **multiple
types of input** -- text, images, audio, video, and more -- at the same time.

```
+-------------------------------------------------------------------+
|                  Types of "Modalities"                             |
|                                                                   |
|   Text:    "A dog playing in the park"                            |
|   Image:   [photo of a golden retriever fetching a ball]          |
|   Audio:   [sound of a dog barking]                               |
|   Video:   [clip of a dog running through grass]                  |
|                                                                   |
|   A multimodal model can understand ALL of these and              |
|   connect them: "The barking sound matches the dog in             |
|   the image, which matches the text description."                 |
+-------------------------------------------------------------------+
```

---

## Why Does This Matter?

The real world is multimodal. We don't experience life through text alone --
we see, hear, touch, and read simultaneously. Multimodal AI can:

- **Describe images** -- "What's in this photo?"
- **Answer questions about images** -- "How many people are in this picture?"
- **Generate images from text** -- "Draw a cat riding a bicycle"
- **Transcribe speech** -- Convert audio to text
- **Generate speech** -- Convert text to natural-sounding audio
- **Understand videos** -- Summarize what happens in a video clip

```
+-------------------------------------------------------------------+
|              Multimodal Model Examples                             |
|                                                                   |
|   GPT-4V / GPT-4o:  Text + Images + Audio                        |
|   Claude (Vision):   Text + Images                                |
|   CLIP:              Connects text and images                     |
|   Whisper:           Audio --> Text                               |
|   DALL-E / Midjourney: Text --> Images                            |
|   Gemini:            Text + Images + Audio + Video                |
+-------------------------------------------------------------------+
```

---

## Key Concepts

### Cross-Modal Alignment

The central challenge of multimodal AI: how do you get a model to understand
that a **photo of a cat**, the **word "cat"**, and the **sound of meowing**
all refer to the same concept?

This is called **alignment** -- mapping different modalities into a shared
understanding.

```
+-------------------------------------------------------------------+
|              Cross-Modal Alignment                                 |
|                                                                   |
|   Text: "cat"        --+                                          |
|                        |                                          |
|   Image: [photo]     --+--> SHARED UNDERSTANDING SPACE            |
|                        |    (all representations are close         |
|   Audio: [meow]      --+     together in this space)              |
|                                                                   |
|   The model learns that these three things                        |
|   all mean "cat" even though they look                            |
|   completely different as raw data.                                |
+-------------------------------------------------------------------+
```

### Fusion Strategies

How do you combine information from different modalities?

| Strategy | How It Works | Example |
|----------|-------------|---------|
| **Early Fusion** | Combine raw inputs before processing | Concatenate image pixels with text tokens |
| **Late Fusion** | Process each modality separately, combine at the end | Separate image and text models, merge their outputs |
| **Cross-Attention Fusion** | Let modalities "look at" each other during processing | Image features attend to text tokens and vice versa |

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

## Directory Structure

```
05-multimodal/
+-- README.md                    # You are here
+-- vision-language/             # Models that combine vision + text
|   +-- README.md                #   CLIP, BLIP, image captioning, visual QA
+-- audio-language/              # Models that combine audio + text
|   +-- README.md                #   Whisper, TTS, speech understanding
+-- experiments/                 # Hands-on practice
    +-- (your experiments go here!)
```

---

## Key Terms

| Term | Simple Explanation |
|------|-------------------|
| **Modality** | A type of data: text, image, audio, video |
| **Multimodal** | Using more than one modality at a time |
| **Embedding** | A list of numbers representing the meaning of some input |
| **Alignment** | Teaching the model that a photo of X and the word "X" are the same concept |
| **Contrastive Learning** | Training by showing the model matching pairs (correct) and non-matching pairs (incorrect) |
| **Encoder** | The part of the model that reads/processes an input (text encoder, image encoder, etc.) |
| **Zero-shot** | Using a model on a task it wasn't explicitly trained for |

---

## Key Papers

- **CLIP: Learning Transferable Visual Models From Natural Language Supervision** -- Radford et al., 2021
  - The breakthrough paper that showed you can align images and text by training on 400M image-text pairs from the internet
- **BLIP-2: Bootstrapping Language-Image Pre-training** -- Li et al., 2023
  - Efficient method to connect frozen image encoders with frozen LLMs
- **Flamingo: a Visual Language Model for Few-Shot Learning** -- Alayrac et al., 2022
  - Showed multimodal models can learn new tasks from just a few examples
- **Whisper: Robust Speech Recognition via Large-Scale Weak Supervision** -- Radford et al., 2022
  - OpenAI's speech recognition model trained on 680K hours of audio

---

[Back to Main](../README.md) | [Previous: Prompt Engineering](../04-prompt-engineering/README.md) | [Next: Evaluation](../06-evaluation/README.md)
