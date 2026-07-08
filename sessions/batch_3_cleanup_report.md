# Batch 3 — Cleanup Pass Report

_Cleanup-only pass over the Batch 3 inference-systems modules (`m06-cnns-vision-encoders`, `m16a-inference-economics`, `m17a-quantization`, `m17b-long-context-decoding`). No new visuals, no Coach Layer rollout, no broad content refactor. Date: 2026-07-07._

Companion docs: [`batch_2_cleanup_report.md`](./batch_2_cleanup_report.md) (the pass this repeats) · [`batch_2_rollout_report.md`](./batch_2_rollout_report.md) §11 (defines Batch 3) · [`lesson_audit.py`](./lesson_audit.py) (the automated gate).

---

## 1. TL;DR

Ran the Batch 2 cleanup pass — the same four defect classes — over the **26** Batch 3 lessons. The batch was found **almost entirely clean**: the only defect was **one** `<h1>` ampersand-spacing glitch (`m06/day-08`), now fixed. There were **0** duplicate callouts and **0** stale run commands.

Why so clean: **Batch 3 never received the Coach Layer rollout** that introduced Batch 2's defects. These modules carry the curriculum-wide ⚠️/⚖️ staff-lens callouts (which did not double-paste here), but 0/26 have 🎤 interview blocks, pain-points, or P0 `.vlab` labs. So the duplicate-callout defect class simply does not exist here, and the native run-command format is not stale.

Final state: **26/26 OK in `lesson_audit.py`, 0 DEGRADED**; the one edited file renders clean in jsdom with JS syntax intact; diff-verified single-line change with no protected machinery touched.

---

## 2. Scope

Batch 3 = the inference-systems tier named in `batch_2_rollout_report.md` §11:

| Module | Lessons | Theme |
|--------|:---:|-------|
| `m06-cnns-vision-encoders` | 8 | conv/pooling → CNN classifier → BN → augmentation → transfer/CLIP |
| `m16a-inference-economics` | 6 | prefill/decode → memory wall → batching roofline → serving → quant intro |
| `m17a-quantization` | 6 | int8 theory → QuIP/QuIP# → AQLM/QTIP → memory economics → PTQ |
| `m17b-long-context-decoding` | 6 | KV cache → SnapKV → GQA/MQA → speculative → ring attention → essay |

**26 lessons total.** All on the current sidebar shell; all 26 pass `lesson_audit.py` structurally.

---

## 3. Defect inventory (full detection suite, all 26)

| Class | Result |
|-------|--------|
| **Duplicate callouts** — adjacent identical lines + whole-file exact `<div class="callout` scan | **0.** The only repeated callout line is the `🧮` math-ladder mono-wrapper opener in `m06/day-02` — a legitimate multi-line ladder box (non-adjacent), the same verified false-positive noted in Batch 2. Not touched. |
| **Stale run commands** | **0.** `m06`'s 8 lessons use `Run with <code>python3 sessions/…/experiment.py</code>` — all already the correct full path. `m16a`/`m17a`/`m17b` use a prose instruction ("…then run it to confirm it works") that references the `Create <code>sessions/…/experiment.py</code>` path directly — no stale flat filenames anywhere. |
| **Title spacing glitch** | **1** — `m06/day-08-clip-contrastive` (fixed, §4). |
| **Markdown report hygiene** | **N/A** — no `batch_3_*` docs existed; this report is the new artifact and is itself 0-issue Markdown. |
| **Finale integrity** | All 26 `you've completed <b>…</b>` finales intact (no truncation). |
| **Baseline `lesson_audit.py`** | 26/26 OK, 0 MISSING, 0 LEFTOVER, 0 DEGRADED. |

---

## 4. The one fix — title spacing glitch

`m06-cnns-vision-encoders/day-08-clip-contrastive/lesson.html`, line 333. The `<h1>` left the ampersand orphaned against the subtitle span (the exact pattern fixed in Batch 2's `m03/day-05-positional`). Moved `&amp;` into the subtitle span so it renders cleanly and matches the sibling convention (`m06/day-01`: `<span class="sub">&amp; Pooling</span>`):

- Before: `<h1>CLIP &amp;<span class="sub">Contrastive Learning</span></h1>`
- After:  `<h1>CLIP<span class="sub">&amp; Contrastive Learning</span></h1>`

The `<title>` (`… CLIP &amp; Contrastive Learning`) was already correct and was not touched. A precise scan (`&amp;` abutting a tag) confirms **no** orphaned/abutting ampersand remains in any Batch 3 title. The other four `m06`/`m16a`/`m17a` h1s that a loose scan flags (`&amp; Pooling`, `&amp; Embeddings`, `&amp; Roofline`, `&amp; Perplexity`) already use the correct in-span convention and were left alone.

---

## 5. Verification

- **`lesson_audit.py` (targeted — `_recover_set.json` left unchanged):** `26/26 OK, 0 DEGRADED`.
- **JS syntax:** every inline `<script>` in the edited file parses clean (`vm.Script`).
- **jsdom render:** the edited file loads with no runtime error and passes the structural gate (`RESULT: PASS`).
- **Diff audit:** exactly one `<h1>` line changed (−1/+1); guard confirms **no** `data-quest-id` / `data-demo` / `data-sec` / `var QS|DEMOS|BUILD` / nav-`href` / `.gotit` line removed.

---

## 6. Notes / out of scope

- **Batch 3 is a future Coach Layer rollout candidate, not part of this cleanup.** It currently lacks the Batch 1/2 Coach Layer: **0/26** lessons have a 🎤 interview block, a bilingual pain-point, or an interactive P0 `.vlab` lab. Adding those is a *rollout* (much larger, one lesson per session), explicitly out of scope for this cleanup-only pass.
- `m17b/day-06` ("The Hardware Lottery" systems-review essay) has no ⚖️ trade-off callout and no `experiment.py` by design — its Produce artifact is an essay, not a code experiment. Not a defect; not changed.
- `sessions/` is edited by concurrent sessions; only the two files this pass owns were staged/committed.

---

## 7. Definition of done — Batch 3 cleanup

- [x] Full defect-detection suite run over all 26 lessons (duplicates, run commands, title glitches, finales).
- [x] The one real defect — `m06/day-08` title glitch — fixed and precisely re-verified.
- [x] Confirmed 0 duplicate callouts and 0 stale run commands (with documented rationale for why the batch was clean).
- [x] `lesson_audit.py` 26/26 OK / 0 DEGRADED; edited file JS-clean + jsdom render-clean; diff additive-safe.
- [x] No Coach Layer rollout, no new visuals, no non-Batch-3 lesson touched.
- [x] `batch_3_cleanup_report.md` written (this file).
