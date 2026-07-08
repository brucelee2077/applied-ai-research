# Inference-Systems Batch — Coach Layer Rollout Report

_Execution report for the `frontier-curriculum-refactor` Coach Layer rollout to the **inference-systems tier** (`m06` / `m16a` / `m17a` / `m17b`). Companion to [`inference_systems_rollout_plan.md`](./inference_systems_rollout_plan.md). Date: 2026-07-08. Effort: xhigh._

Prior art this follows: [`batch_3_rollout_report.md`](./batch_3_rollout_report.md) (JAX/systems-tier rollout) · [`batch_3_cleanup_report.md`](./batch_3_cleanup_report.md) (the earlier defect-only pass over this same scope, which explicitly deferred the Coach Layer). This report closes that deferral.

---

## 1. TL;DR

Rolled the full Coach Layer onto **all 26 lessons** of the inference-systems tier — every lesson now carries a warm bilingual pain-point, an intuition-first + systems analogy, a Math Ladder, a labeled Staff Lens, a 🎤 interview-ready answer, explicit acceptance criteria, and a bilingual 5-minute research-log/explain-back — while preserving every piece of interactive machinery byte-for-behaviour.

- **26/26 lessons `lesson_audit.py` OK**, 0 DEGRADED. Coach advisories cleared everywhere except **one documented, intentional** advisory (the `m17b/day-06` synthesis essay legitimately has no `experiment.py`).
- **6 P0 visuals shipped** — **5 new inline `.vlab` interactive labs** (m06 conv-slide, m06 embedding-space, m06 CLIP grid+contrastive, m16a batching/KV-budget, m17b ring-rotation) + **1 existing-viz embed** (m17a/day-05 → `roofline.html`). All 5 labs re-render on control input under the jsdom `--p0` harness.
- **26 Math Ladders added** (one per lesson — more than the plan's 21, because every lesson turned out to have a genuine single core formula, including the two "survey"/"essay" days).
- **In-scope coherence fixes:** ~15 undefined-CSS-var sub-lines (`--text2`/`--surface2`/`--border`/`--bg-warm`) fixed across edited files; **5 truncated `var BUILD` note captions** completed in `m16a/day-05`+`day-06` (finale-truncation generator-bug class); 2 undefined-var tables (`m06/day-07`, `m17b/day-02`) repaired.
- **jsdom + JS-syntax: 26/26 PASS.** Nav, `data-quest-id`, quiz (4 each), playground (`var DEMOS`, 3 demos), `var BUILD`, and tooltip machinery preserved everywhere; git-diff guard shows **no protected line net-deleted**.

---

## 2. Scope & method

26 lessons: `m06-cnns-vision-encoders` (8) + `m16a-inference-economics` (6) + `m17a-quantization` (6) + `m17b-long-context-decoding` (6). Workflow: **Inspect → Visual Audit → Plan (written first, no edits) → Edit (one file at a time, sequential) → per-file verify gate → batch verify → report**.

The visual audit ran as **4 read-only per-module auditors in parallel** (read-only parallelization allowed; only lesson *writing* was sequential). One contested auditor finding — that `m16a/day-05`+`day-06` were "broken generator artifacts" — was **directly re-verified and corrected**: those lessons are already-visual (embed `quantization.html` / `roofline.html`) plus a legitimate verbose code-walkthrough; the real, smaller defect was a handful of mid-word-truncated note captions (the complete text lived in the parallel `viz` code strips), which were completed rather than rebuilt.

---

## 3. Files changed

**26 lesson files** (all `sessions/<module>/day-*/lesson.html`):

| Module | Lessons edited |
|--------|----------------|
| `m06-cnns-vision-encoders` | day-01 … day-08 (all 8) |
| `m16a-inference-economics` | day-01 … day-06 (all 6) |
| `m17a-quantization` | day-01 … day-06 (all 6) |
| `m17b-long-context-decoding` | day-01 … day-06 (all 6) |

**2 new artifacts:** [`inference_systems_rollout_plan.md`](./inference_systems_rollout_plan.md) (plan-first), [`inference_systems_rollout_report.md`](./inference_systems_rollout_report.md) (this file).

Every edit is **additive** inside `.sec-body` (Coach blocks) or into the `<style>`/`<script>` for the 5 P0 labs, plus the small documented coherence fixes. No lesson outside the batch was touched.

---

## 4. P0 visuals added (6)

All labs are **dependency-free inline SVG + vanilla JS** on the proven repo `.vlab` pattern (CSS from `m10a/day-01`), offline-safe, theme-aware, keyboard-reachable, inserted into §3 **above** the existing fake-terminal (which keeps its 3-`data-demo` completion gate). **All lab math was hand-verified in Node before shipping.** Each carries the 7 required cards (learning question, labeled axes/legend, clear change-on-control, what-to-notice, misconception, where-it-simplifies, frontier relevance).

| # | Lesson | Lab | What it makes visible | Grammar item |
|---|--------|-----|-----------------------|--------------|
| 1 | `m06/day-01-convolution-pooling` | **Sliding-kernel + pooling sweep** | Position slider slides a 3×3 filter over a 6×6 image, fills a 4×4 feature map cell-by-cell (verified FMAP `[[6,3,5,8]…]`), then a 2×2 max-pool → 2×2 (`[[7,8],[6,9]]`) | feature-map / pooling |
| 2 | `m06/day-06-transfer-embeddings` | **Embedding-space explorer** | Two sliders move a query point through a 2D scatter; its 3 nearest neighbours + predicted class flip live as you enter a cluster (kNN verified) | embedding space |
| 3 | `m06/day-08-clip-contrastive` | **CLIP grid + contrastive pull/push** | A "training progress" slider brightens the N×N similarity grid's diagonal + dims off-diagonal while matched image/text dots pull together on a circle (diag rises −0.36→1.00, off-diag falls 0.12→−0.33, verified) | CLIP grid **+** contrastive pull/push |
| 4 | `m16a/day-03-batching-economics` | **Batching & KV-budget lab** | Batch-size slider bends the step-time curve at ridge `B*≈167`, throughput/latency readouts move, and the HBM bar shows KV OOM — MHA at B≈62 (before B*!), GQA≈246, MQA≈1970 (all verified); MHA/GQA/MQA toggle | request-queue/batching + latency-vs-throughput |
| 5 | `m17b/day-05-ring-attention` | **Ring-rotation stepper** | Step slider rotates K/V blocks around an N-device ring; an N×N coverage grid fills as blocks visit devices, hitting full after N−1 steps; N-devices slider shows per-device memory staying flat | ring attention |
| 6 | `m17a/day-05-memory-economics-synthesis` | **Embed `../../viz/roofline.html`** (existing viz, not new D3) | Drag arithmetic intensity across the ridge on the one roofline-shaped lesson that omitted its matching interactive; mirrors day-01/day-03's `quantization.html` embed | memory-economics via roofline |

**Held as documented P1 (not built):** `m17b/day-03-gqa-mqa` (topology) — `kv-cache.html` already gives this lesson a live group-size slider, so a topology-only viz was marginal. Everything already-visual (8 iframe lessons) or with adequate static builds got Coach prose only. The honest strict-rubric finding remains "this tier is visually mature"; the 6 shipped upgrades are where interactivity beats the current visual and the parent named the grammar item — 3 concentrated in `m06`, the one module with zero prior interactivity in the most spatial subject.

---

## 5. Math Ladders added (26)

A compact 4-rung Math Ladder (words → labeled formula → tiny numbers → sanity check) was added to **every one of the 26 lessons** — more than the plan's projected 21. The plan reserved 5 "documented skips," but on close reading each of those lessons carried a genuine single core formula worth laddering, so a ladder was added instead of skipped (a positive, honest deviation):

- `m06/day-03` (shape-trace `⌊(W−K+2P)/S⌋+1 → pool → flatten`), `m06/day-05` (horizontal-flip index map), `m06/day-07` (IoU) — these three were plan-skips, laddered because the math is real.
- `m17a/day-03` (QuIP#, "4-method survey") — laddered on its true central formula `tr(ΔW·H·ΔWᵀ)`.
- `m17b/day-06` (essay) — laddered on the roofline verdict `min(Peak, Intensity×BW)` / ridge point.

Representative core formulas laddered: conv dot-product, output-dim `⌊(W−K+2P)/S⌋+1`, batch-norm `(x−μ)/√(σ²+ε)`, cosine similarity, InfoNCE grid, arithmetic intensity, ridge point, `B*`, KV-cache bytes, absmax int8 scale, incoherence μ, Hessian-weighted objective, additive-codebook sum, perplexity `exp(mean NLL)`, SnapKV vote/top-k budget, GQA cache ratio, speculative expected-accept `(1−α^{K+1})/(1−α)`, ring per-device memory `O(s/N)`.

---

## 6. Coach Layer artifacts improved (all 26)

Per lesson (only what was missing was added; existing good blocks were left):

- **Pain-point** (😕, bilingual) ×26 — one Chinese "why this trips people up" clause + the specific English confusion.
- **Bilingual intuition + systems analogy** ×26 — a `直觉 / Intuition` line paired with an "in code/systems terms" line (2–4 total Chinese touches per lesson, pain/intuition/log only — never in formulas, code, or interview phrasing).
- **Interview-ready 🎤** ×26 — a 20–30s spoken answer (headline + proof number + the nuance that separates Hire from Weak Hire). The auditors confirmed **0/26 had a real block** beforehand (the audit's substring match under-reported).
- **Labeled Staff Lens** ×1 added (`m17b/day-06`, the only lesson lacking one); present + preserved on the other 25.
- **Acceptance criteria** ×8 added (all `m06`, which lacked the explicit `<h4>`); present + preserved on the other 18.
- **Explain-back + 5-minute research log** (📓, bilingual) ×26 — the old one-line `log.md` nudge folded into a richer 3-line log whose lines carry the explain-back (what was built, why it scales, what breaks in production).
- **Produce artifact** — every lesson keeps its concrete artifact (`experiment.py`, or the `m17b/day-06` synthesis essay `.md`); none fabricated.

---

## 7. In-scope coherence fixes (guide §7 — factual/label bugs in edited files)

- **Undefined CSS vars** → defined shell vars, in edited files only: `var(--text2)` → `var(--ink2)` (in ✅/🧮/📦 sub-lines across ~11 lessons in m16a/m17a/m17b), `var(--surface2)`/`var(--border)` → `var(--panel2)`/`var(--line2)` (the `m06/day-07` detection table), `var(--bg-warm)`/`var(--border)` → `var(--panel2)`/`var(--line2)` (the `m17b/day-02` GQA-vs-SnapKV table). These previously rendered with no background/border.
- **Truncated `var BUILD` note captions** (finale-truncation generator-bug class) completed from the parallel complete `viz` code strips: `m16a/day-05` — 4 notes (`…by 150`→`150x+`, `…0.50`→`0.5000] (max err ≈ 0.0013)`, `…relative erro`→`error)`, `…already-trained w`→`weights`); `m16a/day-06` — 1 note (a stray `…seq_len;  byt` trimmed to match the parallel `decode_step` note). No BUILD rebuild — the code-walkthrough steps and embedded iframes were preserved.

`m06/review.html` (a module review-gate page, **not one of the 26 lesson files and not in scope**) still contains an undefined var — pre-existing, untouched, noted below as a non-blocking observation.

---

## 8. Audit result

- **`lesson_audit.py` (targeted, all 26):** `26/26 OK · 0 MISSING · 0 LEFTOVER · 0 DEGRADED`.
- **Coach advisories remaining: 1**, intentional and documented — `m17b/day-06` "no concrete Produce artifact (experiment.py)". Its artifact is `hardware-lottery-essay.md`, correct for a synthesis-essay day. Every other pain / interview / ladder / accept / log / bilingual / staff-lens advisory is cleared on all 26.
- **Duplicate-callout scan** (memory `coach-layer-duplicate-callout-defect`): **0 adjacent duplicate callouts** across the batch.
- **Per-block counts:** exactly **1 pain + 1 interview + 1 log** block per lesson (no double-paste); **1 Math Ladder** per lesson.

---

## 9. jsdom / shell verification

- **`node /tmp/batch1_verify.js` on all 26: 26/26 PASS.** Each asserts JS-syntax clean (`vm.Script` on every inline `<script>`), jsdom render with no runtime error, and 7 sections / 7 got-it / 3 demos / 4 quiz / BUILD present / tooltip controller.
- **P0 interaction (`--p0`) on the 5 lab files:** all 5 assert `.vlab` present, a range slider present, and lab innerHTML re-renders on `input` (content changed). The 6th P0 (`m17a/day-05` embed) verified via the iframe now present + the shell's `viz-height` resize receiver + `roofline.html`'s existing height sender.
- **Preservation guard (git diff):** quest-ids untouched (0 removed); quiz literals = 4 on all 26; 7/7/3 section/gotit/demo counts intact on all 26; the only flagged "removed" protected lines were the 3 single-line `var BUILD=` literals modified in place (m16a/d05, m16a/d06 note-fixes; m17a/d05 embed-prepend) — each matched by a re-added `+var BUILD=` line, and all 26 retain `var BUILD` / `var DEMOS` / `var QS`.
- **Batch hygiene:** no stale run commands (paths are full `sessions/…` or prose), no title-spacing glitches, no residual undefined CSS vars in edited files, no remaining truncated notes, visual captions hand-checked against verified lab math.

---

## 10. Remaining non-blocking risks

1. **No real-browser pixel check** (`no-real-browser-in-sandbox`). All verification is headless (jsdom + JS-syntax + hand-checked lab math). The 5 labs reuse the `.vlab` classes proven in Batch 1/2/3; a Light/Dim eyeball would need a user screenshot if pixel fidelity matters. The two new SVG lab types (ring-rotation, CLIP grid) are novel layouts — most worth a glance.
2. **P0 count is a documented judgment.** This tier is visually mature (8 iframes + rich static builds), so the strict-rubric P0 set is small. The 6 upgrades are directive-aligned (parent named each grammar item) and concentrated where interactivity genuinely beats the existing visual. `m17b/day-03` (GQA topology) was deliberately held as P1 — the adjustable knob if a reviewer wants it built.
3. **One intentional advisory persists** (`m17b/day-06` essay, no `experiment.py`) — by design, not a defect.
4. **`m06/review.html`** (out of scope) carries a pre-existing undefined CSS var — noted for a future cleanup pass, not touched here.
5. **Concurrent edits.** `sessions/` is edited by other sessions; each file was re-read immediately before editing and the tooling enforces "file modified since read." Changes are staged in the working tree, **not committed** (awaiting the usual explicit go-ahead).

---

## 11. Definition of done — met

- [x] Visual audit run (4 read-only per-module auditors); one contested finding directly re-verified and corrected.
- [x] Plan written **before** any lesson edit (`inference_systems_rollout_plan.md`).
- [x] All 26 lessons get their full missing Coach-element set; nav / quest-id / quiz / playground / build / tooltips preserved.
- [x] 6 P0 visuals shipped (5 inline `.vlab` labs + 1 existing-viz embed); all interactive checks pass.
- [x] `lesson_audit.py` 26/26 OK; jsdom 26/26 PASS; batch hygiene clean; git-diff additive-safe.
- [x] Report written (this file).
