# Audio-Language Models

## What Are Audio-Language Models?

Think about how you experience a conversation. You **hear** words (audio), your brain
converts them to **meaning** (language), and you respond with **speech** (back to audio).
Audio-language models do the same thing for computers.

These models bridge the gap between **sound** and **text**, enabling:
- **Speech-to-text** (transcription) -- Hear audio, write it down
- **Text-to-speech** (synthesis) -- Read text, speak it aloud
- **Audio understanding** -- Understand what's happening in audio (music, sounds, speech)

```
+-------------------------------------------------------------------+
|              Audio-Language Model Tasks                            |
|                                                                   |
|   Speech Recognition (ASR):                                       |
|     [audio: "Hello, how are you?"] --> "Hello, how are you?"     |
|                                                                   |
|   Text-to-Speech (TTS):                                           |
|     "Welcome to the show" --> [natural-sounding audio]           |
|                                                                   |
|   Audio Classification:                                            |
|     [sound clip] --> "Dog barking" / "Car horn" / "Piano music"  |
|                                                                   |
|   Speech Translation:                                              |
|     [audio in French] --> "Hello" (translated English text)       |
+-------------------------------------------------------------------+
```

---

## Speech Recognition: How AI Hears

**Automatic Speech Recognition (ASR)** converts spoken audio into written text.
This is what powers voice assistants (Siri, Alexa), video subtitles, and
meeting transcription tools.

### How It Works (Simplified)

```
+-------------------------------------------------------------------+
|              Speech Recognition Pipeline                          |
|                                                                   |
|   1. RAW AUDIO                                                    |
|      Sound waves captured by a microphone                         |
|      (just a list of numbers representing air pressure)           |
|                                                                   |
|   2. SPECTROGRAM                                                   |
|      Convert audio to a visual representation                     |
|      (an "image" of the sound over time)                          |
|                                                                   |
|      Time  -->                                                    |
|      Freq  ........###.......                                     |
|       |    .....###...###....                                     |
|       v    ..###.........###.                                     |
|            ###.............##                                     |
|                                                                   |
|   3. ENCODER                                                      |
|      Process the spectrogram with a neural network                |
|      (like reading the "image" of the sound)                      |
|                                                                   |
|   4. DECODER                                                      |
|      Generate text tokens one at a time                           |
|      "Hello" --> "how" --> "are" --> "you"                        |
+-------------------------------------------------------------------+
```

### Whisper: OpenAI's Speech Model

**Whisper** (2022) is the most popular open-source speech recognition model.

```
+-------------------------------------------------------------------+
|                    Whisper Overview                                |
|                                                                   |
|   Trained on: 680,000 hours of audio from the internet            |
|   Languages:  99 languages                                        |
|   Tasks:      Transcription, translation, language detection      |
|                                                                   |
|   Model sizes:                                                    |
|     tiny   (39M params)  -- Fast, less accurate                  |
|     base   (74M params)  -- Good balance                          |
|     small  (244M params) -- Better accuracy                       |
|     medium (769M params) -- High quality                          |
|     large  (1.5B params) -- Best quality                          |
|                                                                   |
|   Architecture: Encoder-Decoder Transformer                       |
|     Encoder: Processes the audio spectrogram                      |
|     Decoder: Generates text tokens                                |
+-------------------------------------------------------------------+
```

**What makes Whisper special:**
- Works across 99 languages
- Can translate speech from one language to English text
- Handles background noise, accents, and technical terms well
- Completely open-source and free

---

## Text-to-Speech: How AI Speaks

**Text-to-Speech (TTS)** does the reverse -- it converts written text into
natural-sounding speech.

### The Evolution of TTS

```
+-------------------------------------------------------------------+
|              TTS Through the Ages                                 |
|                                                                   |
|   1960s-1990s: Rule-based                                          |
|     "Hello" --> [robotic voice, like old GPS]                     |
|     Method: Manually programmed pronunciation rules               |
|                                                                   |
|   2000s-2015: Concatenative                                        |
|     "Hello" --> [more natural but slightly choppy]                |
|     Method: Stitch together recorded audio snippets               |
|                                                                   |
|   2016-2020: Neural TTS (WaveNet, Tacotron)                       |
|     "Hello" --> [very natural sounding]                           |
|     Method: Neural networks generate audio directly               |
|                                                                   |
|   2023+: LLM-based TTS                                            |
|     "Hello" --> [indistinguishable from human]                    |
|     Method: Treat speech as "tokens" like text                    |
|     Models: ElevenLabs, OpenAI TTS, Bark                         |
+-------------------------------------------------------------------+
```

### Modern TTS Models

| Model | Type | Key Feature |
|-------|------|-------------|
| **OpenAI TTS** | API | High quality, multiple voices |
| **ElevenLabs** | API | Voice cloning, emotional control |
| **Bark** | Open-source | Can generate music and sound effects too |
| **Coqui TTS** | Open-source | Many languages, voice cloning |
| **XTTS** | Open-source | Cross-lingual voice cloning |

---

## Cross-Modal Learning

The frontier of audio-language research is building models that truly
**understand** the connection between audio and language, not just convert
one to the other.

### Audio-Language Models (Beyond ASR/TTS)

```
+-------------------------------------------------------------------+
|              Audio Understanding Models                            |
|                                                                   |
|   CLAP (Contrastive Language-Audio Pre-training):                 |
|     Like CLIP but for audio! Aligns audio and text in             |
|     a shared embedding space.                                      |
|                                                                   |
|     "Sound of rain" <--> [audio of rainfall]                     |
|     "Dog barking"   <--> [audio of dog bark]                     |
|                                                                   |
|   AudioPaLM / Gemini:                                              |
|     Natively multimodal -- can process text, images, AND          |
|     audio in a single model.                                       |
|                                                                   |
|   Use cases:                                                       |
|     - "What instrument is playing?" (audio QA)                    |
|     - "Find sounds similar to this clip" (audio search)           |
|     - "Describe what you hear" (audio captioning)                 |
+-------------------------------------------------------------------+
```

---

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Spectrogram** | A visual representation of audio -- shows which frequencies are present at each moment in time |
| **Mel scale** | A frequency scale that matches how humans perceive sound (we're better at distinguishing low-pitched sounds) |
| **ASR** | Automatic Speech Recognition -- audio to text |
| **TTS** | Text-to-Speech -- text to audio |
| **Diarization** | Figuring out "who spoke when" in a conversation with multiple speakers |
| **Voice cloning** | Creating a synthetic voice that sounds like a specific person |
| **Prosody** | The rhythm, stress, and intonation of speech (what makes it sound natural vs robotic) |

---

## Summary

```
+------------------------------------------------------------------+
|            Audio-Language Models Cheat Sheet                      |
|                                                                  |
|  Speech-to-Text (ASR):                                           |
|    Best model: Whisper (open-source, 99 languages)               |
|    How: Audio --> Spectrogram --> Encoder --> Decoder --> Text    |
|                                                                  |
|  Text-to-Speech (TTS):                                           |
|    Best models: OpenAI TTS, ElevenLabs (API), Bark (open)       |
|    How: Text --> Neural network --> Audio waveform               |
|                                                                  |
|  Audio Understanding:                                             |
|    Emerging area: CLAP, AudioPaLM                                |
|    Like CLIP but for audio + text alignment                      |
+------------------------------------------------------------------+
```

---

## Further Reading

- **Whisper: Robust Speech Recognition via Large-Scale Weak Supervision** -- Radford et al., 2022
- **WaveNet: A Generative Model for Raw Audio** -- van den Oord et al., 2016
  - The paper that made neural TTS sound natural
- **CLAP: Learning Audio Concepts From Natural Language Supervision** -- Elizalde et al., 2023
- **AudioPaLM: A Large Language Model That Can Speak and Listen** -- Rubenstein et al., 2023

---

[Back to Multimodal](../README.md)
