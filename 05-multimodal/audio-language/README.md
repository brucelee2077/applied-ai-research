# Audio-Language Models

Close your eyes and listen. You can tell if someone is happy or sad just from their voice. You can recognize a song after hearing two seconds. You can follow a conversation in a noisy restaurant. Your brain turns sound waves into meaning effortlessly.

Computers struggled with this for decades. Then, in 2022, a model called Whisper listened to 680,000 hours of audio and learned to do what seemed impossible: transcribe speech in 99 languages, translate between them, and handle accents, background noise, and technical terms — all in a single model. How?

The answer starts with a surprisingly visual trick: turn sound into a picture.

**Before you start, you need to know:**
- What a neural network does at a high level — covered in [00-neural-networks](../../00-neural-networks/)
- What an encoder and decoder are — covered in [multimodal README](../README.md)

---

## The Analogy: A Conversation with a Friend

Think about how you experience a conversation. You **hear** words (sound waves hit your ears), your brain converts them into **meaning** (you understand what was said), and you can respond with **speech** (your vocal cords produce sound waves). Audio-language models do the same three things:

- **Speech-to-text (ASR):** Hear audio, write it down
- **Text-to-speech (TTS):** Read text, speak it aloud
- **Audio understanding:** Listen to any sound and describe what is happening

**What the analogy gets right:**
- Your brain processes sound in stages: raw sound waves first, then patterns (phonemes, words), then meaning. Audio models follow the same pipeline.
- You can handle noise, accents, and overlapping voices — and so can modern models like Whisper.
- You can both listen (ASR) and speak (TTS) — models can do both directions too.

**Where the analogy breaks down:** Your brain learns language from a few thousand hours of conversation over years. Whisper learned from 680,000 hours of audio in one massive training run. You understand *meaning* — Whisper finds statistical patterns. You can ask "what did you mean by that?" — Whisper cannot.

---

## Turning Sound into a Picture: The Spectrogram

Here is the surprising trick at the heart of speech recognition: **convert audio into an image, then process that image with a neural network.**

A **spectrogram** is a picture of sound. The horizontal axis is time. The vertical axis is pitch (frequency). The brightness shows how loud each pitch is at each moment.

```
   Spectrogram: "Hello"

   High pitch  |  .   .       .  .
               |  ..  ..     ..  ..
               | ... ....   ... ...
   Low pitch   | .... ..... ........
               +------------------------
                  H   e   l   l   o
                       Time -->
```

Your voice makes different pitch patterns for each sound. The letter "s" has energy at high pitches. The letter "o" has energy at low pitches. A spectrogram captures all of this in one picture.

Once speech is a picture, you can use the same tools that work for images — transformers, convolutions, attention — to process it. That is why modern speech models are so powerful: they reuse all the advances from computer vision.

---

## How Speech Recognition Works

The pipeline is simple once you know the spectrogram trick:

```
   1. RAW AUDIO
      Sound waves from a microphone
      (just a list of numbers: air pressure over time)

   2. SPECTROGRAM
      Convert audio to a picture of sound
      (which pitches are present at each moment)

   3. ENCODER
      A neural network reads the "picture"
      (extracts patterns: phonemes, words, context)

   4. DECODER
      Generates text tokens one at a time
      "Hello" --> "how" --> "are" --> "you"
```

This is the architecture Whisper uses. The encoder is a transformer that processes the spectrogram. The decoder is a transformer that generates text. The two are connected by cross-attention — the decoder can look back at any part of the audio while generating each word.

---

## Text-to-Speech: Going the Other Way

Text-to-speech (TTS) does the reverse — it takes written text and produces natural-sounding audio. Modern TTS models sound so natural that most people cannot tell the difference from a real human voice.

The key breakthrough: instead of hand-coding pronunciation rules, modern TTS models learn to generate audio by training on thousands of hours of recorded speech. They learn rhythm, stress, emotion, and natural pauses directly from data.

---

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Spectrogram** | A picture of sound — shows which pitches are present at each moment in time |
| **Mel scale** | A way of measuring pitch that matches how humans hear — we are better at telling apart low sounds than high sounds |
| **ASR** | Automatic Speech Recognition — audio to text |
| **TTS** | Text-to-Speech — text to audio |
| **Diarization** | Figuring out "who spoke when" in a conversation with multiple speakers |

---

## Quick Check — Can You Answer These?

- Why is converting audio to a spectrogram useful? What does it allow us to reuse?
- What are the four steps in the speech recognition pipeline?
- What is the difference between ASR and TTS?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## What You Just Learned

You now understand how Siri, Alexa, Google Assistant, and every voice-powered app works under the hood. The spectrogram trick — turning sound into a picture — is the foundation of all modern speech recognition. And the encoder-decoder transformer architecture that Whisper uses is the same architecture behind machine translation, text generation, and dozens of other AI breakthroughs.

Ready to go deeper? The math behind spectrograms, Whisper's architecture, failure modes, and interview questions are in [audio-language-interview.md](./audio-language-interview.md).

---

[Back to Multimodal](../README.md)
