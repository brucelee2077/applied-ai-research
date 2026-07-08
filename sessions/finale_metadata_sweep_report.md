# Finale Metadata Sweep Report

_Metadata/coherence-only sweep of lesson **finale banners** across the whole `sessions/` curriculum. No lesson content, navigation, quiz, playground, or section structure was changed — only the truncated `<b>topic</b>` inside each finale's "you've completed …" line. Date: 2026-07-06._

Companion: [`batch_1_cleanup_report.md`](./batch_1_cleanup_report.md) §7 first flagged the out-of-Batch-1 truncations this sweep resolves.

---

## 1. TL;DR

A known generator bug truncated the completed-topic `<b>…</b>` in many lesson finales (e.g. `Deriving the`, `The Tower of`, `Batch`). This sweep scanned **all 155 lesson pages**, isolated **58 finales that differ from their `<title>` main-name**, adjudicated each (truncated vs. deliberate nickname vs. already-complete vs. acceptable short name) with a 20-agent classify + adversarial-verify workflow, and **fixed the 44 genuinely truncated finales** by restoring the correct name from each lesson's own `<title>`. **14 were intentionally left** (deliberate analogy nicknames, complete short names, or already-complete). Each fix is a single surgical `<b>…</b>` replacement inside the finale. **`lesson_audit.py`: 44/44 OK, 0 DEGRADED; all pass jsdom + JS-syntax with 7/7 sections, `BUILD`/`QS` intact, and no dangling finale.**

---

## 2. Method

1. **Scan (deterministic).** Extract every lesson's `<title>` topic (text after the em-dash) and its finale `you've completed <b>TOPIC</b>`. Compute each title's **main-name** = the title topic minus any `: subtitle` and trailing `(parenthetical)`. Flag lessons where `TOPIC` differs from the main-name.
2. **Adjudicate (workflow).** 58 candidates → 10 batches → **10 classify agents** (each reads the lesson: title, hero heading, finale) → **10 adversarial-verify agents** (re-check each verdict, reject over-fixes of nicknames/short names). Verdicts: `truncated` (fix) · `nickname` · `already_complete` · `acceptable_short` (leave). Classify and verify agreed on all 58.
3. **Apply (surgical).** For each `truncated` verdict, replace only the `<b>…</b>` immediately after "you've completed", using the file's exact bytes as the search key; the corrected topic is HTML-encoded (`&` → `&amp;`).
4. **Audit.** `lesson_audit.py` on all changed files + a jsdom/JS-syntax/structure gate + a re-scan confirming no dangling finales remain.

---

## 3. Scope

| Metric | Count |
|--------|:---:|
| Lesson pages scanned (`sessions/**/lesson.html`) | 155 |
| With a `you've completed <b>…</b>` finale | 150 |
| Non-standard custom finale (no completed-bold slot) | 5 |
| Finale differs from `<title>` main-name (candidates) | 58 |
| **Fixed (truncated)** | **44** |
| Left unchanged | 14 |

---

## 4. Fixes applied (44)

Each finale's `<b>topic</b>` restored to the lesson's proper name from its `<title>`. "Before" is the truncated text; "After" is the corrected display text.

| Lesson | Before | After |
|--------|--------|-------|
| m03 · day-05-positional | `Positional Encoding &` | `Positional Encoding & the Shuffle Problem` |
| m04 · day-03-minibatch-loop | `The Mini-Batch Loop` | `The Mini-Batch Loop & Overfit-One-Batch` |
| m04 · day-05-pytorch-version | `The Same Model` | `The Same Model in PyTorch` |
| m04 · day-06-why-pytorch | `Why PyTorch` | `Why PyTorch Mirrors NumPy` |
| m05a · day-02-layer-norm | `Keeping the` | `Layer Normalization` |
| m05a · day-04-full-block | `Assembling the` | `The Full Decoder Block` |
| m05a · day-06-encoder-vs-decoder | `Reader, Writer,` | `Encoder vs Decoder` |
| m05a · day-07-text-generation | `Picking the` | `Text Generation and Sampling` |
| m05a · day-08-full-transformer | `The Tower of` | `The Full Transformer` |
| m06 · day-01-convolution-pooling | `Convolution` | `Convolution & Pooling` |
| m06 · day-02-output-dim-arithmetic | `Output-Size` | `Output-Size Arithmetic` |
| m06 · day-03-cnn-classifier | `A Small CNN` | `A Small CNN Classifier` |
| m06 · day-04-batch-normalization | `Batch` | `Batch Normalization` |
| m06 · day-05-data-augmentation | `Data` | `Data Augmentation` |
| m06 · day-06-transfer-learning-embeddings | `Transfer Learning` | `Transfer Learning & Embeddings` |
| m06 · day-08-clip-contrastive | `CLIP &` | `CLIP & Contrastive Learning` |
| m08 · day-06-scaling-exercises-rehearsal | `Scaling Book Exercises` | `Scaling Book Exercises & Screencast Rehearsal` |
| m09a · day-01-rooflines | `Rooflines` | `Rooflines & Arithmetic Intensity` |
| m09a · day-06-distributed-checkpointing | `Distributed Deploy` | `Distributed Deploy & Activation Checkpointing` |
| m09c · day-06-pytorch-sync-consolidation | `PyTorch Parallelism Sync` | `PyTorch Parallelism Sync & Phase 1 Consolidation` |
| m12 · day-04-custom-tokenizer-design | `Custom Tokenizer` | `Custom Tokenizer Design` |
| m12 · day-08-training-loop-baseline-training | `Training Loop &` | `Training Loop & Baseline Training` |
| m12 · day-10-profiling-step-time | `Profiling` | `Profiling the Step Time` |
| m14a · day-06-routing-design-deep-synthesis | `Routing Design` | `MoE Routing Design & Deep Synthesis` |
| m14b · day-01-moe-architecture-upgrade | `Mixture of Experts` | `MoE Architecture Upgrade` |
| m14b · day-03-moe-isoflops-matrix-law-derivation | `MoE IsoFlops Matrix` | `MoE IsoFlops Matrix & Law Derivation` |
| m14b · day-06-consolidation-publishing | `Consolidation` | `Consolidation & Publishing` |
| m15a · day-02-basic-kernel-construction | `Basic Kernel` | `Basic Kernel Construction` |
| m15a · day-06-moe-kernel-challenge-performance-proof | `MoE Kernel Challenge` | `MoE Kernel Challenge & Performance Proof` |
| m15b · day-02-tensor-memory-acceleration | `Tensor Memory` | `Tensor Memory Acceleration` |
| m15b · day-04-thunderkittens-architecture | `ThunderKittens` | `ThunderKittens Architecture` |
| m15b · day-06-cuda-tk-setup-implementation | `CUDA/TK Setup` | `CUDA/TK Setup & TK Implementation` |
| m16a · day-04-serving-llama3-tpus | `Serving LLaMA 3` | `Serving LLaMA 3 on TPUs` |
| m16a · day-06-inference-calculator-roofline | `Inference Calculator` | `Inference Calculator & Roofline Graphing` |
| m17a · day-05-memory-economics-synthesis | `Memory Economics` | `Memory Economics Synthesis` |
| m17a · day-06-ptq-implementation-perplexity-benchmarking | `PTQ Implementation` | `PTQ Implementation & Perplexity Benchmarking` |
| m26 · day-03-case-studies-automation | `Case Studies` | `Case Studies in Automation` |
| m26 · day-04-less-is-more-principle | `The "Less is More"` | `The "Less is More" Principle` |
| m26 · day-05-reward-hacking-threat | `The Reward Hacking` | `The Reward Hacking Threat` |
| m26 · day-06-evaluation-engineering-holdout | `Evaluation Engineering` | `Evaluation Engineering & Hold-out Workloads` |
| m26 · day-12-adrs-implementation-simulation-execution | `ADRS Implementation` | `ADRS Implementation & Simulation Execution` |
| m29 · day-05-cold-email-engineering | `Cold Email` | `Cold Email Engineering` |
| m29 · day-06-targeted-leadership-outreach | `Targeted Leadership` | `Targeted Leadership Outreach` |
| m29 · day-07-researcher-network-outreach | `Researcher Network` | `Researcher Network Outreach` |

---

## 5. Left unchanged (14) — not truncated

The adversarial pass confirmed these are **complete on purpose**; fixing them would mutate intentional content.

| Lesson | Finale | Category | Why left |
|--------|--------|----------|----------|
| m02 · day-07-optimizers | `Optimizers: SGD → Momentum → Adam` | already complete | Finale is the full title verbatim (subtitle used as a complete phrase); nothing is cut |
| m04 · day-02-backward-pass | `The Backward Pass` | complete short name | 'The Backward Pass' is the title's main name before the comma and is a complete, standard, unambiguous ML concept name (hero h1 'The Backward Pass' + sub 'Wired by Hand') |
| m05a · day-01-residual-connections | `The Highway` | deliberate nickname | Title 'Residual Connections'; hero 'The Highway / Around the Town' |
| m05a · day-03-feed-forward | `Each Word Thinks` | deliberate nickname | Title 'The Feed-Forward Layer'; hero 'Each Word Thinks / in a Private Room' |
| m05a · day-05-causal-masking | `No Peeking` | deliberate nickname | Title 'Causal Masking'; hero 'No Peeking / at the Future' |
| m06 · day-07-object-detection-intuition | `Object Detection` | already complete | Complete standard concept name; matches the hero heading (title only adds ‘Intuition’) |
| m07 · day-03-vmap | `vmap` | complete short name | Complete standard JAX term; matches the hero heading |
| m07 · day-04-jit | `jit` | complete short name | Complete standard JAX term; matches the hero heading |
| m09c · day-05-distillation | `Distillation` | complete short name | Title main-name is 'Knowledge Distillation' (pre-colon), but the hero H1 itself is just 'Distillation' and the sibling day-06 prev-nav links to this lesson as 'Distillation' |
| m14a · day-01-moe-fundamentals | `Mixture-of-Experts` | complete short name | Complete concept name; the verify pass confirmed not-a-fix |
| m14b · day-04-report-finalization | `Report Finalization` | complete short name | Title main-name is 'Scaling Report Finalization'; the finale drops only the leading adjective 'Scaling' |
| m15a · day-05-hardware-evolution | `Hardware Evolution` | complete short name | Title main-name is 'FlashAttention Hardware Evolution'; the finale drops only the leading qualifier 'FlashAttention' |
| m17b · day-06-systems-review-essay | `The Hardware Lottery` | already complete | Title is 'Systems Review Essay: The Hardware Lottery' but the hero FLIPS it — h1 is 'The Hardware Lottery' with 'Systems Review Essay' as the subtitle format label |
| m26 · day-11-agentic-scaffolding | `Agentic Scaffolding` | complete short name | Complete phrase; drops only the trailing ‘for ADRS’ |

**5 non-standard finales** (custom closing sentence, no `you've completed <b>…</b>` slot) were confirmed complete and untouched: `m01/day-03`, `m01/day-05`, `m08/day-01`, `m08/day-03`, `m10a/day-04`.

---

## 6. Preservation & verification

- **Surgical edits only.** Each change replaces exactly one `<b>…</b>` inside the finale's "you've completed …" line. Navigation `href`s, `data-quest-id`, `data-sec`/section structure, `.gotit` buttons, `var QS`/`var DEMOS`/`var BUILD`, and tooltip machinery are untouched.
- **`lesson_audit.py`** (targeted; `_recover_set.json` unchanged) on the changed files: **44/44 OK, 0 MISSING / 0 LEFTOVER / 0 DEGRADED.**
- **jsdom + JS-syntax gate** on all changed files: 0 syntax errors, 0 jsdom runtime errors, 7/7 `data-sec` sections, 7/7 `.gotit`, `BUILD`/`QS` present, no finale ends on a dangling connector.
- **Re-scan:** the only remaining finale-vs-main-name differences are the 14 intentional leaves plus 2 fixed finales trimmed to a complete short form (`Basic Kernel Construction`, `The "Less is More" Principle`) — **zero dangling/truncated finales remain.**

---

## 7. Notes / remaining risks

1. **Resolves `batch_1_cleanup_report.md` §7.** The out-of-scope truncations flagged there (the `m05a-text-transformer` set and others) are now fixed curriculum-wide.
2. **No real-browser pixel check** (sandbox constraint) — verification is headless (jsdom + audit + re-scan). Finale text is plain content in a `<p>`, so visual risk is minimal.
3. **Editorial choices, not truncations, were left alone:** deliberate analogy nicknames (`The Highway`, `No Peeking`, `Each Word Thinks`), the subtitle-as-title case (`The Hardware Lottery`), and complete short names matching the hero heading (`vmap`, `jit`, `Distillation`, `Object Detection`). A future pass could align these to `<title>` verbatim, but they are not bugs.
