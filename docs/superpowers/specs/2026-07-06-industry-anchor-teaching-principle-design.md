# Industry-anchor teaching principle — design

## Problem

The user singled out `sessions/m01-shape-of-data/day-03-broadcasting-dtypes/lesson.html`'s
section 4 ("Why a frontier lab cares") as content they want more of: a short passage
connecting the day's mechanism to why it matters at a real lab, plus a "remember this chain"
takeaway. They want (a) more of this kind of context, and (b) for it to link to the
**current, named industry** rather than stay abstract.

Auditing the curriculum found this pattern already exists in exactly 19 lessons — M01 Days
1–5, M02 Days 1–9, M03 Days 1–4, M04 Day 1 — each with a `<h4>Why a frontier lab cares</h4>`
block in section 4 ("机制/为什么 Mechanism & why"). Every instance is abstract: "a
7-billion-parameter model," "a frontier lab," "modern frameworks" — never a named system,
paper, or company. The other 137 built lessons (M05–M29) don't have this section at all; they
use different "Why it matters" headers.

## Decision

**Scope for this pass:** deepen the 19 existing M01–M04 lessons now. Propagating the pattern
to M05–M29 is deferred — tracked as a watchlist note, not scheduled. (Note: the 2026-07-04
Staff-lens retrofit was in fact swept curriculum-wide the same day it was decided — see
`ROADMAP.md` lines ~553–560 — so this narrower M01–M04-only scope is a deliberate choice for
this pass, per the user's explicit answer during brainstorming, not a repeat of how Staff-lens
happened to play out.)

**Content specificity:** name real systems by title — a specific model, paper, or product
with a verifiable number — rather than staying abstract. Facts were checked against primary
sources (arXiv abstracts/sections) where feasible this session; see the Verification ledger
below for what's paper-table-confirmed vs. well-established secondary knowledge. Anything not
independently confirmed is phrased with a hedge ("Meta has reported…", "per its published
config…") rather than presented as a verbatim paper citation.

**Structural placement — two touchpoints per lesson, no new sections:**
1. **Earlier hook** — a new short callout appended to the end of section 1 ("是什么 What is
   it"), reusing the existing `.callout.c-info` style with a new 🏭 icon (already used
   elsewhere in the curriculum for factory/production-line imagery; reused here for
   "real production system" callouts — a new but consistent pairing, not an existing
   Layer-2-markdown convention). One or two sentences naming the same system used in section
   4, as a teaser. No new `.sec` element — the progress-bar JS counts `.sec` elements, so
   this must live inside the existing section 1 body.
2. **Deepened section 4** — the existing `<h4>Why a frontier lab cares</h4>` paragraph is
   rewritten to *open* with the named system + verified number, keep the existing abstract
   mechanism explanation after it, and leave the "Remember this chain" callout as-is (it's
   already good — the timeless takeaway, not the concrete anchor).

Worked example (Day 3, approved by user):
- Hook (end of section 1): "DeepSeek-V3 (671B params) trains its matrix multiplies almost
  entirely in FP8 — half the bytes of bf16."
- Section 4 opens: existing "A 7B model in float32 needs ~28GB…" sentence, then adds:
  "DeepSeek-V3 goes further: 671B total parameters (37B active per token, it's a
  mixture-of-experts) trained with FP8 matrix multiplies — half of bf16's bytes again."
  Existing bias-vector broadcasting example and chain callout are unchanged.

## Fact table (approved)

| Lesson | Named system | Fact | Confidence |
|---|---|---|---|
| M01 D1 Arrays | GPT-3 175B config (Table 2.1) | 96 layers, d_model 12288, 96 heads → one FFN weight matrix is shape (12288, 49152) | Well-established (not re-verified from primary text this session; this table is among the most widely reproduced in ML pedagogy) |
| M01 D2 Indexing/slicing | vLLM's PagedAttention (Kwon et al. 2023) | KV cache is paged into fixed-size blocks, OS-virtual-memory-inspired | Verified (abstract) |
| M01 D3 Broadcasting/dtypes | DeepSeek-V3 | 671B total / 37B active (MoE); GEMMs trained in FP8 | Verified (tech report §3.3) |
| M01 D4 Matmul/shapes | Kaplan et al. C≈6ND; NVIDIA H100 spec | Compute formula from Kaplan (2020); H100 published BF16 dense throughput | C≈6ND verified; H100 spec is a public vendor datasheet, not a paper |
| M01 D5 Logs/exponents | Kaplan scaling exponent; Chinchilla | α≈0.050–0.057 (Kaplan); Chinchilla 70B/1.4T tokens = 20 tok/param | Verified (both papers) |
| M02 D1 Single neuron | (reuses GPT-3 FFN matrix from D1/D4) | one neuron = one row of that matrix | Same confidence as D1 |
| M02 D2 Activations | SwiGLU (Llama) vs GELU (GPT-2/3, BERT) | which activation family ships in which model family | Well-established |
| M02 D3 Forward pass = inference | vLLM / TensorRT-LLM / SGLang | named serving engines running this exact forward pass in production | Well-established (named OSS projects) |
| M02 D4 Loss | Cross-entropy as literal GPT pretraining loss; Chinchilla's loss-vs-compute framing | — | Well-established / verified |
| M02 D5 Gradients/backprop | jax.grad / torch.autograd; Kaplan's "6" in C≈6ND | — | Verified (Kaplan); autodiff APIs are named public library functions |
| M02 D6 Training loop | Llama 3 | Meta reported pretraining the 405B model on >15T tokens | Well-publicized, not primary-verified this session (full-text fetch truncated twice) |
| M02 D7 Optimizers | ZeRO / DeepSpeed (Rajbhandari et al.) | shards Adam's optimizer state across GPUs specifically because it's ~2× model memory | Verified (abstract; Adam-specific wording is well-known Stage-1 mechanism) |
| M02 D8 Learning rate | (kept qualitative — no unverified exact LR value) | warmup+decay schedules are published as first-class methodology in GPT-3/Llama3/Chinchilla papers | N/A — deliberately no unverified number |
| M02 D9 Train/val/test | Benchmark decontamination practice (MMLU, GSM8K) | described in Llama 3 / GPT-4-class technical reports | Well-established practice, not a single verified number |
| M03 D1 Embeddings | Llama 3 vocab (128,256 tokens) vs GPT-4o's o200k_base (~200k tokens) | tokenizer vocab sizes | Llama 3 verified via published config; GPT-4o figure well-known public tiktoken fact |
| M03 D2 QKV | Llama 3 405B | 128 query heads / 8 KV heads (GQA), hidden 16384 | Well-established via Meta's published model config |
| M03 D3 Attention scores | FlashAttention (Dao et al.); Gemini 1.5 context | avoids materializing seq×seq matrix; Gemini 1.5's 1–2M token context as the payoff for dodging O(seq²) | FlashAttention verified; Gemini context length well-publicized |
| M03 D4 Multi-head | GQA paper (Ainslie et al., Google) | Llama 3 / Mistral use GQA in production | Verified (paper); production usage well-established |
| M04 D1 MLP MNIST | Cross-entropy loss identity | same loss function trains this MNIST MLP and GPT-style next-token pretraining | Well-established (definitional fact, no citation needed) |

## Changes

### 1. `.claude/skills/frontier-session-coach/TEACHING_PRINCIPLES.md` — new "Industry Anchor" rule
Sibling to the existing "Staff Lens" rule. Requires: section 4's frontier-relevance sentence
must open by naming one real, verifiable system with a specific number (not "a frontier lab"
in the abstract); section 1 gets a matching one-to-two-sentence teaser callout naming the same
system. Facts must be well-established or paper-verified; hedge language required when not
independently confirmed.

### 2. `SESSION_TEMPLATE.md` and `SKILL.md`
One sentence each in the section 1 and section 4 descriptions pointing at the new Industry
Anchor rule in `TEACHING_PRINCIPLES.md` — same pattern as the existing Staff Lens pointers.

### 3. `ROADMAP.md`
- **Global rules** gains one dated line ("Industry-anchor bar, added 2026-07-06") pointing at
  the new rule.
- A new subsection, **"Industry-anchor retrofit — M01–M04 (2026-07-06)"**, recording which 19
  lessons were retrofitted and noting M05–M29 as a deferred watchlist item (not scheduled),
  mirroring the Staff-lens retrofit section's format.

### 4. The 19 lesson files themselves
Each of the 19 `lesson.html` files listed in the fact table gets:
- One new `.callout.c-info` (🏭 icon) appended at the end of section 1.
- Section 4's frontier-relevance paragraph rewritten per the fact table.
No other section is touched. No new `.sec` elements. `lesson-before.html`, `compare.html`,
`experiment.py`, `log.md` in `day-03-broadcasting-dtypes/` are the user's own in-progress
files and are left untouched.

**Structural exceptions found while auditing the 19 targets** (confirmed by grep, not
assumed):
- 17 of 19 use `<h4>Why a frontier lab cares</h4>` — rewrite that paragraph directly.
- **M02 D8 (learning-rate) and M02 D9 (train-val-test)** don't have an `<h4>` at all — the
  phrase is the bold lead-in inside the existing `<div class="callout c-info">…<b>Why a
  frontier lab cares:</b> …</div>` block. Retarget that inline lead-in instead of looking for
  an `<h4>`.
- **M02 D7 (optimizers) and M02 D9 (train-val-test)** each have a *second*, shorter echo of
  the same phrase inside a section-5 BUILD-step `note:` JS string (the final scroll-reveal
  step restates the section-4 takeaway in one line). When rewriting section 4 for these two,
  also update that echoed note so it stays consistent with the new named-system claim — but
  keep it a one-line callback, not a second full deepening.
- General fallback for any further mismatch discovered mid-pass: if section 4's
  frontier-relevance text isn't in `<h4>` form, retarget the equivalent prose block; don't
  stop to ask, just keep the meaning and note the deviation in the commit.

## Scope

- The 19 M01–M04 lessons listed above, plus the three skill-guidance files and ROADMAP.md.
- Files are written one at a time, in the order listed in the fact table (M01→M04), per this
  repo's "one file per session, no parallel writing" rule — content edits are not
  parallelized across subagents.

## Out of scope

- M05–M29 propagation (137 lessons) — deferred to a future session, tracked only as a
  ROADMAP watchlist note.
- Re-verifying every "well-established" fact against primary full text — two arXiv full-text
  fetches (GPT-3, Llama 3 Herd of Models) truncated twice each this session; per the repo's
  retry policy (try once, maybe twice, then stop) these are not re-attempted. Copy uses hedge
  language for these specific facts instead of claiming table-level verification.
- Any change to `sessions/staff_lens_audit.js` or other structural audit tooling — this is a
  content/copy change, not a new structural rule that needs mechanical enforcement.
