> **What this file covers**
> - 🎯 Why mel spectrograms are the standard audio representation (derived from human perception)
> - 🧮 Full STFT formula, mel scale conversion, and mel-spectrogram computation — worked examples
> - 🧮 Whisper architecture: encoder-decoder transformer, special tokens, multitask training
> - 🗺️ Concept flow: raw audio → STFT → mel-spectrogram → encoder → decoder → tokens → text
> - ⚠️ 4 failure modes: hallucination on silence, accent bias, background noise, timestamp drift
> - 📊 Whisper model sizes vs accuracy vs latency — real numbers
> - 💡 CTC vs seq2seq decoding, streaming vs batch, model size trade-offs — comparison tables
> - 🏭 Whisper deployment: chunking long audio, diarization pipeline, real-time factor
> - Staff/Principal Q&A with all four hiring levels shown

---

# Audio-Language Models — Interview Deep-Dive

This file assumes you have read [audio-language README](./README.md) and understand the spectrogram trick, the encoder-decoder pipeline, and the difference between ASR and TTS. Everything here is for Staff/Principal depth.

---

## 🧮 From Sound to Numbers: The Short-Time Fourier Transform (STFT)

### Step 1 — Raw audio

Sound is air pressure changing over time. A microphone samples this pressure at a fixed rate. At 16kHz (Whisper's sample rate), you get 16,000 numbers per second.

```
    Raw audio: x[0], x[1], x[2], ..., x[T-1]
    Sample rate: 16,000 Hz
    1 second of audio = 16,000 numbers
    30 seconds = 480,000 numbers
```

### Step 2 — Why not feed raw audio directly?

Raw audio is a 1D signal. A 30-second clip is 480,000 numbers. This is too long for a transformer — attention is O(n²), so 480,000 tokens would require 230 billion attention cells. We need a compressed representation.

The solution: split time into short windows, and for each window, decompose the sound into its component frequencies using the Fourier transform.

### Step 3 — The STFT formula

The Discrete Fourier Transform tells you which frequencies are present in a signal. The Short-Time Fourier Transform (STFT) applies this to overlapping windows of the audio.

First, we explain what the Fourier transform does in plain words: it takes a chunk of audio (a list of numbers representing air pressure) and breaks it into a sum of pure tones at different frequencies. Each frequency gets a number saying how loud it is.

```
🧮 STFT formula:

    X(t, f) = Σₙ x[n] · w[n - t·hop] · e^(-j·2π·f·n/N)

    Where:
      x[n]         = raw audio sample at position n
      w[n]         = window function (usually Hann window) centered at time frame t
      hop          = hop size (number of samples between consecutive windows)
      N            = window size (number of samples per window, typically 400 at 16kHz = 25ms)
      f            = frequency bin index (0 to N/2)
      j            = imaginary unit (√-1)
      X(t, f)      = complex number encoding amplitude and phase at time t, frequency f
```

The magnitude |X(t, f)|² is the **power spectrogram** — how loud frequency f is at time t.

**Practical parameters (Whisper defaults):**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Sample rate | 16,000 Hz | 16K samples per second |
| Window size (N) | 400 samples | 25ms per window |
| Hop size | 160 samples | 10ms between windows |
| FFT size | 400 | Produces 201 frequency bins (N/2 + 1) |

**Worked example:**

30 seconds of audio at 16kHz = 480,000 samples.
Number of time frames = (480,000 - 400) / 160 + 1 = 2,997 frames.
Frequency bins = 400/2 + 1 = 201.
Spectrogram shape: (201, 2997) — 201 frequencies × 2,997 time steps.

This is much smaller than 480,000 raw samples, and it organizes the information by frequency and time — exactly what we need for speech patterns.

---

## 🧮 The Mel Scale: Matching Human Perception

The linear frequency spectrogram treats all frequencies equally: 100 Hz and 200 Hz get the same spacing as 8,000 Hz and 8,100 Hz. But human hearing does not work this way.

Humans are much better at telling apart low-pitched sounds than high-pitched sounds. The difference between 100 Hz and 200 Hz is obvious (like the difference between a bass and a tenor). The difference between 8,000 Hz and 8,100 Hz is barely noticeable.

The **mel scale** matches this. It spaces frequencies according to how humans perceive them — more resolution at low frequencies, less at high frequencies.

```
🧮 Mel scale conversion:

    mel(f) = 2595 · log₁₀(1 + f/700)

    Where:
      f     = frequency in Hz
      mel(f) = frequency on the mel scale

    Inverse:
      f = 700 · (10^(mel/2595) - 1)
```

**Worked example:**

| Frequency (Hz) | Mel value | Perceptual meaning |
|----------------|-----------|-------------------|
| 100 | 150 | Low bass |
| 200 | 284 | +134 mel (big perceptual difference) |
| 1,000 | 1,000 | Mid range |
| 2,000 | 1,405 | +405 mel |
| 8,000 | 3,071 | High frequency |
| 8,100 | 3,085 | +14 mel (tiny perceptual difference) |

Notice: 100 Hz to 200 Hz = 134 mel units. But 8,000 Hz to 8,100 Hz = only 14 mel units. The mel scale compresses high frequencies and expands low ones — matching human perception.

### Mel filter bank

To create a mel-spectrogram, we apply a set of triangular filters to the power spectrogram. Each filter covers a range of frequencies on the mel scale.

```
🧮 Mel-spectrogram computation:

    S_mel(t, m) = Σ_f H_m(f) · |X(t, f)|²

    Where:
      |X(t, f)|²  = power spectrogram at time t, frequency f
      H_m(f)      = triangular filter m (centered at mel-spaced frequency)
      m           = mel band index (0 to M-1, typically M = 80 or 128)
      S_mel(t, m) = mel-spectrogram value at time t, mel band m
```

Whisper uses 80 mel bands. So the final mel-spectrogram has shape: **(80, T)** where T is the number of time frames.

For 30 seconds: shape = (80, 3000). This is the input to Whisper's encoder.

**Log scaling:** In practice, we take the log: `log(S_mel + 1e-10)`. This compresses the dynamic range — quiet sounds become visible instead of being drowned out by loud ones. The 1e-10 prevents log(0).

---

## 🧮 Whisper Architecture

Whisper is an encoder-decoder transformer. The encoder processes the mel-spectrogram. The decoder generates text tokens autoregressively (one at a time, left to right).

### Encoder

```
    Input: log mel-spectrogram (80, 3000)

    1. Two 1D convolutions:
       Conv1: (80, 3000) → (d_model, 1500)  stride=1, kernel=3
       Conv2: (d_model, 1500) → (d_model, 1500)  stride=2, kernel=3
       This downsamples time by 2x and projects to d_model dimensions

    2. Add sinusoidal position embeddings (1500, d_model)

    3. N transformer encoder layers:
       Each layer: self-attention → layer norm → FFN → layer norm
       Output: (1500, d_model)
```

| Model | d_model | Encoder layers | Decoder layers | Total params |
|-------|---------|----------------|----------------|-------------|
| tiny | 384 | 4 | 4 | 39M |
| base | 512 | 6 | 6 | 74M |
| small | 768 | 12 | 12 | 244M |
| medium | 1024 | 24 | 24 | 769M |
| large-v3 | 1280 | 32 | 32 | 1.55B |

### Decoder

```
    Input: previously generated tokens [<|startoftranscript|>, <|en|>, ...]

    1. Token embedding + positional embedding → (seq_len, d_model)

    2. N transformer decoder layers:
       Each layer:
         - Causal self-attention (can only see past tokens)
         - Cross-attention to encoder output (can see all audio frames)
         - FFN
       Output: (seq_len, d_model)

    3. Linear projection to vocabulary → softmax → next token probabilities
```

### Special tokens

Whisper uses special tokens to control its behavior. This is how a single model handles transcription, translation, timestamps, and language detection.

```
    <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|> Hello how are you <|endoftext|>
         ↑                  ↑         ↑              ↑
     start signal       language    task mode     timestamp mode
```

| Token | Purpose |
|-------|---------|
| `<\|startoftranscript\|>` | Signals the start of decoding |
| `<\|en\|>`, `<\|fr\|>`, ... | Language ID (99 languages) |
| `<\|transcribe\|>` | Transcribe in the same language |
| `<\|translate\|>` | Translate to English |
| `<\|notimestamps\|>` | Do not produce timestamps |
| `<\|0.00\|>`, `<\|0.02\|>`, ... | Timestamp tokens (30ms resolution) |
| `<\|endoftext\|>` | End of sequence |

This multitask design means one model handles all tasks. The decoder "decides" what to do based on the special tokens in its prompt.

---

## 🗺️ Concept Flow

```
     RAW AUDIO (480,000 samples for 30s at 16kHz)
            │
            ▼
     Short-Time Fourier Transform (STFT)
     Window: 25ms, Hop: 10ms
            │
            ▼
     Power Spectrogram (201 freq bins × 3000 frames)
            │
            ▼
     Mel Filter Bank (80 triangular filters)
            │
            ▼
     Log Mel-Spectrogram (80 × 3000)
            │
            ▼
     Two 1D Convolutions (downsamples to 1500 frames)
            │
            ▼
     + Sinusoidal Position Embeddings
            │
            ▼
     Transformer Encoder (N layers of self-attention)
     Output: (1500, d_model) — encoded audio representations
            │
            ╔══════════════════════════════════╗
            ║  Cross-attention from decoder     ║
            ║  (decoder queries, encoder K/V)   ║
            ╚══════════════════════════════════╝
                           │
                           ▼
     Transformer Decoder (causal self-attn + cross-attn)
     Generates tokens one at a time:
     <|startoftranscript|> → <|en|> → <|transcribe|> → Hello → how → ... → <|endoftext|>
```

---

## 💡 CTC vs Sequence-to-Sequence Decoding

Two main approaches to speech recognition. Whisper uses seq2seq. Understanding both is important for interviews.

### CTC (Connectionist Temporal Classification)

CTC aligns audio frames directly to characters without needing explicit alignment labels.

```
    Audio frames: [frame1] [frame2] [frame3] [frame4] [frame5] [frame6] [frame7]
    CTC output:   [ h ]   [ h ]   [ e ]   [ - ]   [ l ]   [ l ]   [ o ]
                                      ↑ blank token
    After collapsing: "hello"
```

The key idea: CTC inserts a **blank token** between characters. Repeated characters are collapsed. The loss function marginalizes over all valid alignments.

### Comparison

| | CTC | Seq2Seq (Whisper) |
|---|---|---|
| **Output** | Characters/subwords, one per frame | Tokens, generated autoregressively |
| **Alignment** | Monotonic (left to right, no reordering) | Flexible (cross-attention can attend anywhere) |
| **Streaming** | Yes — can output as audio arrives | Difficult — decoder needs full context for cross-attention |
| **Conditional independence** | Each frame prediction is independent | Each token depends on all previous tokens |
| **Error correction** | Cannot — each frame is independent | Can — decoder attends to its own output history |
| **Punctuation** | Difficult — no explicit language model | Natural — decoder is a language model |
| **Best for** | Streaming ASR, low latency | Batch ASR, high accuracy, translation |

### Why Whisper chose seq2seq

Whisper prioritizes accuracy over latency. The seq2seq decoder acts as an implicit language model — it can correct errors, add punctuation, and handle translation. CTC would require a separate language model for these capabilities.

---

## ⚠️ Failure Modes

### 1. Hallucination on Silence

**What happens:** When given silent or near-silent audio, Whisper generates plausible-sounding but completely fabricated text. It may produce "Thank you for watching," URLs, or other common web text.

**Why:** Whisper was trained on internet audio, which often has standard intros/outros. The decoder is a language model — when it receives no useful signal from the encoder, it falls back to high-probability text from its training data.

**How to detect:** Check if encoder outputs have low energy. Compare transcription length to audio energy — long transcription of quiet audio is a hallucination.

**How to mitigate:** Voice Activity Detection (VAD) preprocessing. Skip segments where audio energy is below a threshold. In production: Silero VAD → chunk only active speech → Whisper.

### 2. Accent Bias

**What happens:** Whisper performs worse on non-standard accents, regional dialects, and non-native speakers. Word Error Rate (WER) can vary 2-5x across accents.

**Why:** Training data is dominated by American and British English. The internet has less transcribed audio for African, South Asian, and other accents.

**Impact:** Disproportionate error rates for certain demographics. This is an equity concern in production systems.

**Mitigation:** Fine-tune on domain-specific accented data. Use WER disaggregated by accent as a metric, not just overall WER.

### 3. Background Noise Degradation

**What happens:** Performance drops significantly with background music, overlapping speakers, or environmental noise.

**Why:** The mel-spectrogram mixes all audio sources. The model cannot separate the target speaker from background sounds. Whisper is more robust than older models (trained on noisy data) but still degrades at low SNR (Signal-to-Noise Ratio).

**How to measure:** Plot WER vs SNR. Whisper degrades gracefully above 10dB SNR but fails below 5dB.

**Mitigation:** Speech enhancement (noise removal) as a preprocessing step. Source separation models (e.g., Demucs) for music + speech mixtures.

### 4. Timestamp Drift

**What happens:** Whisper's word-level timestamps gradually drift from the actual audio timing, especially in long recordings.

**Why:** Timestamps are generated as special tokens by the decoder. Small errors accumulate over the 30-second processing window. When chunking long audio, boundaries between chunks can cause jumps.

**Impact:** Subtitle generation becomes misaligned. Users see text that does not match what is being said.

**Mitigation:** Use Dynamic Time Warping (DTW) to re-align timestamps with audio. Process overlapping chunks and merge timestamps at boundaries.

---

## 📊 Complexity Analysis

### Whisper Model Sizes

| Model | Params | Encoder layers | d_model | English WER | Multilingual WER | Speed (RTF) |
|-------|--------|----------------|---------|-------------|------------------|-------------|
| tiny | 39M | 4 | 384 | ~7.5% | ~14% | ~0.03 (33x real-time) |
| base | 74M | 6 | 512 | ~5.5% | ~11% | ~0.05 |
| small | 244M | 12 | 768 | ~4.0% | ~8% | ~0.12 |
| medium | 769M | 24 | 1024 | ~3.5% | ~6.5% | ~0.3 |
| large-v3 | 1.55B | 32 | 1280 | ~3.0% | ~5.5% | ~0.6 |

RTF = Real-Time Factor. RTF < 1 means faster than real-time. Whisper-tiny processes audio 33x faster than real-time on a V100 GPU.

### Computational Complexity

```
    Encoder self-attention:   O(T² · d_model) where T = 1500 frames
    Decoder self-attention:   O(S² · d_model) where S = output sequence length
    Cross-attention:          O(S · T · d_model)
    Total per step:           O(T² · d + S · T · d + S² · d)
```

The encoder processes 1500 frames. At d_model=1280 (large), encoder self-attention has 1500² = 2.25M attention cells per layer per head. With 32 layers and 20 heads, this is 1.44 billion attention computations — the dominant cost.

### Memory

| Component | Memory |
|-----------|--------|
| Model weights (large) | ~3.1 GB (float16) |
| Encoder KV cache (32 layers) | ~240 MB per batch item |
| Decoder KV cache | Grows with output length |
| Mel-spectrogram (30s) | ~960 KB |

---

## 🏭 Production and Scaling

### Chunking Long Audio

Whisper processes 30-second segments. For longer audio:

```
Long audio (10 minutes)
    │
    ▼
Split into 30s chunks with 5s overlap
    │
    ├── Chunk 1: [0s - 30s]
    ├── Chunk 2: [25s - 55s]
    ├── Chunk 3: [50s - 80s]
    └── ...
    │
    ▼
Transcribe each chunk independently
    │
    ▼
Merge transcriptions (resolve overlapping regions)
    │
    ▼
Final transcript with timestamps
```

The 5-second overlap ensures no words are lost at chunk boundaries. Merging uses timestamp alignment — keep the version from whichever chunk has the word further from its boundary.

### Diarization Pipeline

Whisper transcribes but does not identify speakers. A full pipeline:

```
Audio → VAD (voice activity detection)
      → Speaker embedding (extract voice prints)
      → Clustering (group segments by speaker)
      → Whisper (transcribe each segment)
      → Merge: "Speaker A: Hello... Speaker B: Hi there..."
```

Common tools: pyannote.audio for diarization, Whisper for transcription.

### Deployment Considerations

| Concern | Solution |
|---------|----------|
| Latency | Use whisper-tiny or distil-whisper for real-time applications |
| Cost | Batch processing on GPU; CPU inference for small models |
| Streaming | Use CTC-based models (Wav2Vec2) for streaming; Whisper for batch |
| Accuracy | Whisper-large for offline/batch; fine-tune on domain data |
| Hallucination | VAD preprocessing + confidence thresholding |
| Languages | Whisper supports 99 languages but quality varies — test per-language |

---

## Staff/Principal Interview Depth

---

**Q1: Walk me through how a mel-spectrogram is computed from raw audio, and why each step exists.**

---
**No Hire**
*Interviewee:* "A spectrogram is a picture of audio showing frequencies over time. Mel makes it match human hearing."
*Interviewer:* The candidate has a vague idea but cannot describe any step concretely. No mention of STFT, window size, hop size, or the mel filter bank. Cannot explain what "match human hearing" means quantitatively.
*Criteria — Met:* Knows spectrogram exists / *Missing:* STFT formula, window/hop parameters, mel scale formula, filter bank mechanics, log scaling

---
**Weak Hire**
*Interviewee:* "You window the audio into overlapping frames, apply FFT to each frame to get frequencies, then apply mel filters to group frequencies into perceptually-spaced bands, then take the log."
*Interviewer:* Correct high-level pipeline. The candidate knows the four steps (window → FFT → mel → log). What is missing: specific parameters (window size, hop, number of mel bands), why overlapping windows, the mel scale formula, and why log scaling is needed.
*Criteria — Met:* Four-step pipeline in correct order / *Missing:* Specific parameters, mel formula, time-frequency resolution trade-off, log scaling justification

---
**Hire**
*Interviewee:* "Start with raw audio at 16kHz. Apply STFT with a 25ms Hann window (400 samples) and 10ms hop (160 samples). This gives overlapping frames to avoid boundary artifacts. FFT of each frame produces a frequency representation with N/2+1 = 201 bins, linearly spaced from 0 to 8kHz. The power spectrum |X(t,f)|² captures energy at each frequency. Then apply 80 triangular mel-spaced filters. The mel scale is mel(f) = 2595 · log₁₀(1 + f/700). This compresses high frequencies where human perception has less resolution — 100-200 Hz gets the same mel range as 4000-8000 Hz. Multiply the power spectrum by each filter, sum within each band, giving (80, T) mel-spectrogram. Finally, take log to compress dynamic range — speech energy varies by 60+ dB, and without log, quiet phonemes are invisible compared to loud ones. Whisper uses log mel-spectrogram as its encoder input."
*Interviewer:* Solid. Gives exact parameters, the mel formula, explains why each step exists, and correctly describes the output shape. What would push to Strong Hire: discussing the time-frequency trade-off (longer windows give better frequency resolution but worse time resolution), the Hann window's purpose (reducing spectral leakage), and alternatives like learned filterbanks.
*Criteria — Met:* STFT with exact parameters, mel formula, perceptual justification, log scaling purpose, output shape / *Missing:* Time-frequency trade-off, window function purpose, learned filterbank alternatives

---
**Strong Hire**
*Interviewee:* "The mel-spectrogram pipeline has five stages, and each involves a design trade-off. (1) Framing: 25ms window at 16kHz = 400 samples. This is long enough to capture the fundamental frequency of most voices (80-300 Hz, period = 3-12ms) but short enough that the signal is approximately stationary within the window. (2) Windowing: Hann window tapers the edges to zero. Without it, the rectangular truncation creates spectral leakage — energy at one frequency bleeds into neighboring bins. The Hann window has a 6dB/octave sidelobe rolloff vs 0dB for rectangular. (3) FFT: Produces 201 bins from 0 to 8kHz. The frequency resolution is fs/N = 16000/400 = 40 Hz per bin. The time-frequency uncertainty principle applies: halving the window doubles frequency resolution but halves time resolution. 25ms is the standard compromise for speech. (4) Mel filter bank: 80 triangular filters spaced by mel(f) = 2595·log₁₀(1+f/700). Below ~1kHz, the mel scale is approximately linear; above ~1kHz, it is approximately logarithmic. This matches the critical bands of the human cochlea — the basilar membrane has roughly equal-width critical bands on the mel scale. 80 bands is standard; 128 gives marginal improvement at 60% more compute. (5) Log: Converts power to approximately perceptual loudness. Weber-Fechner law says perceived loudness is proportional to log of intensity. This also makes the features more Gaussian, which helps neural network training. An alternative is power-law compression (x^0.1) used in PCEN. Whisper uses standard log. The final shape is (80, 3000) for 30 seconds, which the encoder downsamples to (1500, d_model) through strided convolutions."
*Interviewer:* Staff-level. The candidate connects each step to its physical or perceptual justification. Mentioning the uncertainty principle, Hann window spectral leakage, critical bands of the cochlea, and Weber-Fechner law shows deep understanding of why these specific choices were made, not just what they are. The PCEN alternative and 80 vs 128 band trade-off show awareness of the design space.
*Criteria — Met:* Complete pipeline with exact parameters, physical justifications for each step, uncertainty principle, spectral leakage, critical bands, Weber-Fechner law, PCEN alternative, output dimensions

---

**Q2: What are the advantages and disadvantages of Whisper's seq2seq architecture compared to CTC-based models?**

---
**No Hire**
*Interviewee:* "Whisper uses an encoder-decoder transformer. It's more accurate than older methods."
*Interviewer:* The candidate does not know what CTC is and cannot compare architectures. "More accurate" is asserted without reasoning.
*Criteria — Met:* Knows Whisper is encoder-decoder / *Missing:* CTC definition, comparison on any dimension, reasoning about accuracy difference

---
**Weak Hire**
*Interviewee:* "CTC outputs one label per audio frame with a blank token for no-output frames. It is good for streaming because each frame is independent. Whisper uses attention-based decoding which is more accurate but requires processing the full audio first."
*Interviewer:* Correct high-level comparison. The candidate identifies the key trade-off: CTC enables streaming, seq2seq enables higher accuracy. What is missing: why seq2seq is more accurate (implicit language model, error correction), the conditional independence assumption, and concrete scenarios where each wins.
*Criteria — Met:* CTC mechanism (frame-level with blank), streaming trade-off / *Missing:* Conditional independence limitation, implicit language model advantage, punctuation handling, concrete use cases

---
**Hire**
*Interviewee:* "CTC makes a conditional independence assumption: the label at each audio frame is predicted independently of other frames. This means CTC cannot model dependencies between output tokens — it cannot correct 'recognize speech' vs 'wreck a nice beach' because each frame does not know what adjacent frames predicted. Seq2seq (Whisper) has no such limitation: the decoder attends to its own past output and to all encoder frames via cross-attention. This gives three advantages: (1) implicit language model — the decoder learns grammar and common phrases, so it can resolve ambiguities and add punctuation naturally; (2) error correction — if the decoder generates a wrong word, subsequent tokens can implicitly compensate; (3) translation — the decoder can output in a different language than the audio because the target sequence is not constrained to follow the audio's temporal order. The disadvantage is latency: seq2seq needs the full encoder output before decoding starts, making it unsuitable for streaming. CTC models like Wav2Vec2-CTC can output tokens frame by frame with ~50ms latency."
*Interviewer:* Excellent. The candidate explains conditional independence precisely, gives three concrete advantages of seq2seq, and quantifies the latency disadvantage. What would push to Strong Hire: discussing hybrid CTC-Attention models, the label smoothing and beam search strategies Whisper uses, and how streaming Whisper variants (faster-whisper, whisper-streaming) work around the latency issue.
*Criteria — Met:* Conditional independence limitation, three seq2seq advantages with reasoning, latency quantification / *Missing:* Hybrid approaches, beam search, streaming workarounds

---
**Strong Hire**
*Interviewee:* "The core trade-off is conditional independence vs autoregressive modeling. CTC assumes P(y₁,...,yₜ|x) = ∏ₜ P(yₜ|x), marginalizing over all valid alignments via the forward-backward algorithm. This makes training efficient and enables streaming, but the model cannot learn output dependencies — it misses things like 'the' almost always follows 'in' in English. Seq2seq models P(y₁,...,yₜ|x) = ∏ₜ P(yₜ|y₁,...,yₜ₋₁, x), which is exact autoregressive factorization. The decoder is a full language model conditioned on audio. This is why Whisper produces punctuated, grammatically correct output without a separate language model — the decoder handles it internally. The practical implication: CTC needs an external language model for production-quality output (typically shallow fusion with a neural LM, adding 100-300ms latency), while Whisper needs none. Hybrid CTC-Attention models (like Conformer-Transducer) get the best of both: CTC alignment for streaming with attention for accuracy. Whisper's specific choices: beam size 5 for decoding, temperature fallback (if beam search produces high repetition, retry with temperature sampling), and forced alignment via cross-attention weights for timestamps. For streaming Whisper variants: faster-whisper uses CTranslate2 for 4x speedup; whisper-streaming processes audio in chunks with speculative decoding, giving ~1s latency at some accuracy cost."
*Interviewer:* Staff-level. The candidate writes the probabilistic factorizations for both CTC and seq2seq, explains the practical implication (external LM requirement), discusses hybrid approaches, and gives Whisper-specific implementation details (beam search, temperature fallback, faster-whisper). The probabilistic framing shows the candidate understands these as different factorizations of the same joint distribution, which is the fundamental insight.
*Criteria — Met:* Probabilistic factorizations, CTC forward-backward, external LM requirement, hybrid CTC-Attention, Whisper beam search and temperature fallback, streaming variants, forced alignment

---

**Q3: How would you deploy Whisper to transcribe 10,000 hours of audio per day in production?**

---
**No Hire**
*Interviewee:* "Run Whisper on a GPU server and process the files."
*Interviewer:* No system design at all. No mention of chunking, parallelism, latency requirements, model selection, or failure handling.
*Criteria — Met:* Knows Whisper runs on GPU / *Missing:* Everything — chunking, parallelism, model selection, failure handling, monitoring

---
**Weak Hire**
*Interviewee:* "10,000 hours per day = ~417 hours per hour. Whisper-large processes at ~0.6 RTF, so one GPU handles ~1.67 hours per hour. You would need about 250 GPUs running in parallel to meet the throughput. Split files into 30-second chunks, distribute across GPUs, merge results."
*Interviewer:* The arithmetic is correct and shows systems thinking. What is missing: model size selection (large may not be needed), cost analysis, failure handling, quality monitoring, and the chunking overlap strategy.
*Criteria — Met:* Throughput calculation, parallelism strategy / *Missing:* Model size trade-off, cost analysis, chunking overlap, failure handling, quality monitoring

---
**Hire**
*Interviewee:* "10,000 hours/day at 24 hours processing time means ~417 hours per hour throughput. First, model selection: Whisper-medium (RTF ~0.3 on A100) gives WER within 0.5% of large at half the compute. With float16 and batch processing, we get ~3.3 hours per GPU-hour. We need 417/3.3 ≈ 127 A100s. Use a queue-based architecture: audio files go into a message queue (SQS/Kafka), GPU workers pull jobs, process, and write results to object storage. Each file is split into 30-second chunks with 5-second overlap to avoid boundary word loss. Workers process chunks in parallel across a file. Post-processing: merge chunk transcriptions resolving overlaps by comparing timestamps, then run a light language model pass for punctuation normalization. Monitoring: track WER on a golden test set of 100 manually-transcribed files, alert if WER increases by >10%. Track processing latency, queue depth, and GPU utilization. Failure handling: if a chunk fails (OOM, timeout), retry on a different worker with a smaller batch size. After 3 retries, mark the chunk as failed and log for manual review."
*Interviewer:* This is a solid production design. The candidate chooses medium over large with a cost-accuracy reasoning, designs a queue-based distributed system, handles chunking correctly, and includes monitoring and failure handling. What would push to Strong Hire: discussing VAD preprocessing to skip silence (saving 20-30% compute), cost optimization (spot instances, dynamic scaling), and the diarization pipeline for multi-speaker audio.
*Criteria — Met:* Model size selection with reasoning, throughput calculation, queue-based architecture, chunking with overlap, monitoring strategy, failure handling / *Missing:* VAD preprocessing savings, cost optimization, diarization, multi-language handling

---
**Strong Hire**
*Interviewee:* "I'll design this in four layers. **Preprocessing:** Before any transcription, run Silero VAD to detect speech regions. Typical audio has 20-40% silence (hold music, dead air). Skipping silence saves proportional GPU compute and prevents hallucination. Also detect language per file to route to the right Whisper model. **Compute layer:** Use distil-whisper-medium (distilled from large, 50% fewer params, ~90% of large's accuracy at 6x speed). On A100 with float16 and batch size 16: RTF ≈ 0.05, meaning one GPU processes 20 hours per hour. After VAD filtering (~30% silence), effective input is ~7,000 hours. 7000/24/20 = 14.6 A100s needed. Use Kubernetes with spot instances (3x cheaper than on-demand), with on-demand fallback for the 5% spot interruptions. **Quality layer:** Maintain a golden set of 500 files across 10 languages and accent groups. Run daily regression. Track WER disaggregated by language, accent, and audio quality (SNR). Alert on per-group WER regression, not just overall — overall WER can be stable while a minority group degrades. **Output layer:** Chunk with 5s overlap, merge using DTW timestamp alignment. Run speaker diarization (pyannote) on files with multiple speakers. Output format: JSON with word-level timestamps, speaker labels, confidence scores. Store in a searchable index for downstream consumers. Total cost estimate: ~15 A100 spot instances × $1.50/hr × 24h = ~$540/day for 10,000 hours of audio."
*Interviewer:* Staff-level system design. The candidate quantifies the VAD savings, chooses distilled models for cost efficiency, calculates concrete costs, includes equity monitoring (disaggregated WER by accent group), and adds diarization for multi-speaker files. The cost breakdown ($540/day for 10K hours) is specific enough to take to a budget meeting. This is someone who has thought about production ML systems at scale.
*Criteria — Met:* VAD preprocessing with savings quantified, distilled model selection, cost calculation, Kubernetes spot strategy, disaggregated quality monitoring, DTW timestamp alignment, diarization pipeline, concrete cost estimate

---

**Q4: Why does Whisper hallucinate on silent audio, and how would you prevent it?**

---
**No Hire**
*Interviewee:* "Whisper sometimes makes mistakes on quiet audio."
*Interviewer:* The candidate knows hallucination exists but cannot explain why it happens or how to fix it. "Makes mistakes" is vague — hallucination is a specific failure mode distinct from misrecognition.
*Criteria — Met:* Knows the problem exists / *Missing:* Root cause, difference from misrecognition, detection method, prevention strategy

---
**Weak Hire**
*Interviewee:* "The decoder is a language model. When the audio is silent, the encoder gives the decoder no useful signal, so the decoder generates the most likely text from its training data — things like 'thank you for watching' or 'subscribe to my channel' because those appeared frequently in the training data."
*Interviewer:* Correct root cause analysis. The candidate understands that the decoder acts as an unconditioned language model when encoder output is uninformative. What is missing: how to detect hallucination and how to prevent it in production.
*Criteria — Met:* Root cause (decoder as language model, encoder gives no signal) / *Missing:* Detection methods, prevention strategies, VAD preprocessing

---
**Hire**
*Interviewee:* "Whisper's hallucination on silence has a clear mechanism. The encoder processes the mel-spectrogram of silent audio, which has near-zero energy across all mel bands. The encoder output is therefore uninformative — close to the bias terms of the final layer norm. The decoder receives this via cross-attention but gets no useful gradient signal from it. Since the decoder is autoregressive and trained on internet text, it generates the highest-probability continuation given its own prefix tokens. Training data from YouTube includes frequent 'thanks for watching' outros, which become the default generation. Detection: (1) check if RMS energy of the audio chunk is below a threshold (e.g., -40 dB); (2) compute the no-speech probability — Whisper outputs a logit for this in its decoding; (3) check if generated text has high perplexity under a separate language model (hallucinated text is usually generic, low-perplexity). Prevention: run VAD as preprocessing, skip chunks below the energy threshold, and use Whisper's no_speech_threshold parameter (default 0.6). In production: Silero VAD → filter → Whisper → post-hoc confidence check."
*Interviewer:* Strong. The candidate explains the mechanism at the feature level (zero-energy mel-spectrogram), gives three detection methods, and proposes a practical pipeline. What would push to Strong Hire: discussing why fine-tuning does not fully solve this (the decoder always has some language model behavior), the relationship to exposure bias in seq2seq models, and how newer models like Whisper-v3 partially mitigate it.
*Criteria — Met:* Feature-level mechanism, three detection methods, VAD prevention pipeline, no_speech_threshold / *Missing:* Why fine-tuning is insufficient, exposure bias connection, v3 improvements

---
**Strong Hire**
*Interviewee:* "The hallucination is a consequence of Whisper's architecture. The decoder is a causal transformer language model conditioned on encoder output via cross-attention. When encoder output is uninformative (silent audio → near-zero mel features → encoder output ≈ layer norm bias), the cross-attention weights become near-uniform over encoder positions (attention entropy is maximal). The decoder effectively becomes an unconditional language model. Training on 680K hours of internet audio means high-probability generations include YouTube boilerplate. This is fundamentally related to exposure bias: during training, the decoder always receives ground-truth previous tokens; at inference with silence, the decoder receives its own hallucinated tokens and spirals into confident nonsense. Fine-tuning on silence-labeled data helps but does not eliminate the problem because you cannot enumerate all possible hallucinations. Three prevention layers: (1) VAD preprocessing — Silero VAD detects speech with 96%+ accuracy at 1ms latency, essentially free; (2) Whisper's internal no_speech_prob — if the token <|nospeech|> has probability > 0.6, suppress generation; (3) post-hoc: compare transcript length to audio duration — if the ratio is anomalous (e.g., 50 words for 0.5 seconds of audio), flag as hallucination. Whisper v3 partially mitigates this by adding more labeled silence in training and adjusting the loss to penalize generation during silence, but the fundamental issue remains because the decoder architecture inherently has language model behavior."
*Interviewer:* Staff-level. The candidate traces the mechanism from features through architecture to training dynamics, connects to exposure bias (a fundamental seq2seq issue), explains why fine-tuning is insufficient (you cannot cover all hallucinations), and gives three practical prevention layers ordered by cost. The ratio-based anomaly detection is practical and deployable. Understanding that the problem is architectural (any autoregressive decoder will hallucinate when the condition is uninformative) shows deep reasoning.
*Criteria — Met:* Mechanism at feature/architecture/training levels, exposure bias connection, why fine-tuning is insufficient, three prevention layers, v3 improvements, fundamental architectural limitation

---

## Key Takeaways

🎯 1. The mel-spectrogram converts audio to an 80×T matrix that matches human perception — each step (STFT, mel filter, log) has a specific purpose
🎯 2. Whisper is an encoder-decoder transformer that treats ASR as a sequence-to-sequence task, with special tokens controlling language, task, and timestamps
   3. The STFT window size controls the time-frequency resolution trade-off: 25ms is the standard compromise for speech
   4. Mel scale: mel(f) = 2595·log₁₀(1 + f/700) — compresses high frequencies to match the cochlea's critical bands
⚠️ 5. Whisper hallucinates on silence because the decoder becomes an unconditioned language model when encoder output is uninformative
⚠️ 6. Accent bias causes 2-5x WER variation across demographics — monitor per-group WER, not just overall
   7. CTC enables streaming but assumes conditional independence; seq2seq sacrifices latency for accuracy and implicit language modeling
🎯 8. Production Whisper deployment requires VAD preprocessing, 30s chunking with overlap, and speaker diarization for multi-speaker audio
   9. Whisper-medium gives ~95% of large's accuracy at half the compute — model size selection is a key production decision
  10. The decoder's autoregressive nature is both its strength (error correction, punctuation, translation) and weakness (hallucination, no streaming)
