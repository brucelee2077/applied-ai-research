# Inference-Systems Batch — Coach Layer Rollout Plan

_Plan-first artifact for the `frontier-curriculum-refactor` Coach Layer rollout to the **inference-systems tier** (`m06` / `m16a` / `m17a` / `m17b`). Written **before any lesson edit**, at xhigh effort. Date: 2026-07-08._

Companion docs: [`COACH_STYLE_GUIDE.md`](./COACH_STYLE_GUIDE.md) (blocks + rules) · [`enhance_lesson_checklist.md`](./enhance_lesson_checklist.md) (per-lesson checklist) · [`lesson_audit.py`](./lesson_audit.py) (automated gate) · [`batch_3_rollout_plan.md`](./batch_3_rollout_plan.md) + [`batch_3_rollout_report.md`](./batch_3_rollout_report.md) (the JAX/systems-tier rollout this follows) · [`batch_3_cleanup_report.md`](./batch_3_cleanup_report.md) (the earlier defect-only pass over **this same scope**).

> **Naming note.** An earlier `batch_3_cleanup_report.md` ran a **defect-only cleanup** over these exact 26 lessons (found them structurally clean — one title glitch fixed) but explicitly deferred the Coach Layer as "a future rollout candidate, out of scope for that pass." **This** plan is that deferred Coach Layer rollout. It drives `inference_systems_rollout_report.md`. No module overlap with the JAX-tier `batch_3_rollout_*` files (`m07`/`m08`/`m09a`/`m09c`).

---

## 1. Scope

The **inference-systems tier** — vision encoders + the three inference/serving modules:

| Module | Lessons | Theme |
|--------|:---:|-------|
| `m06-cnns-vision-encoders` | 8 | conv/pooling → output-dim arithmetic → CNN classifier → batch-norm → augmentation → transfer/embeddings → detection → CLIP |
| `m16a-inference-economics` | 6 | prefill/decode → memory wall → batching → LLaMA-3/TPU serving → quant intro → roofline calculator |
| `m17a-quantization` | 6 | int8 theory → incoherence/QuIP → QuIP# → AQLM/QTIP → memory economics → PTQ lab |
| `m17b-long-context-decoding` | 6 | KV cache → SnapKV eviction → GQA/MQA → speculative → ring attention → systems essay |

**26 lessons total.** All on the **new sidebar shell** (`class="module-section"`, Light/Dim/Dark/Midnight switcher). Baseline `lesson_audit.py`: **26/26 OK, 0 DEGRADED**. This is a pure Coach Layer overlay + a **small, evidence-driven** interactive-visual pass — no structural repair work.

**Baseline Coach advisories (`lesson_audit.py`, targeted run, all 26):**
- **All 26 are missing:** a pain-point block, a 5-minute research log, and a bilingual scaffold. → add to all 26.
- **Interview 🎤:** `lesson_audit.py` flags 17/26 (the word "interview" appears in prose on the other 9), but the **four independent visual auditors confirmed 0/26 carry a real 🎤 spoken-answer block.** → add a real one to **all 26**.
- **Acceptance criteria:** flagged on all **8 `m06`** lessons (missing); present on all 18 of `m16a`/`m17a`/`m17b`. → add to the 8 `m06`; verify the 18.
- **Staff Lens:** present on 25/26; missing only on `m17b/day-06` (synthesis essay). → add a labeled Staff Lens there.
- **Produce artifact (`experiment.py`):** missing only on `m17b/day-06` — its artifact is a synthesis **essay `.md`** by design. → keep as-is (documented).
- **Math-Ladder advisory:** fires on 25/26 (all but the essay). Ladder added **only** where a lesson has a genuine single core formula (§4); documented skips otherwise.

---

## 2. Method (how this plan was built)

1. **Inspect** — listed all 26 lessons; confirmed all on the new shell; ran `lesson_audit.py` (targeted → `_recover_set.json` untouched) → 26/26 OK; captured the exact per-file missing-Coach-element set from the advisory list.
2. **Visual inventory** — grepped every lesson for embedded `../../viz/*.html` iframes vs. `var BUILD` scroll-reveal SVG steps; catalogued the 17-file `viz/` library. Corrected an initial line-based miscount: every lesson has **either** an interactive iframe **or** a rich 6–8-step static SVG build. This tier is **visually mature**, like the JAX tier — not weak-static like Batch 2.
3. **Visual audit** — ran the `frontier-visual-auditor` rubric via **4 read-only per-module auditors in parallel** (read-only auditing is explicitly allowed to parallelize; only lesson *writing* is sequential). Each classified every lesson's playground {visual-mechanism / interactive-calculator / terminal-walkthrough / static-explanation / missing-visual}, described what the current visual renders, scored it 1–5, and returned a P0/P1/terminal-acceptable/already-visual verdict with evidence — and, for each P0, exactly which interactive to build, why the static/existing-iframe cannot serve it, and confirmation that no existing `viz/*.html` already covers it.
4. **Direct verification of the one contested finding** — the `m16a` auditor flagged `day-05`/`day-06` as "broken generator artifacts (~29/23 empty steps)." I verified directly: they are **already-visual (embed `quantization.html` / `roofline.html`) + a legitimate — if verbose — annotated code-walkthrough build** (same iframe+walkthrough pattern `m17b/day-04` uses and scored 5/5). The real, smaller defect is **~5–6 `note` captions truncated mid-word** (`erro`→`error`, `w`→`weights`, `0.50`→`0.5000…`, `150`→`150x+`, `byt`→`bytes`) — the finale-truncation generator-bug class — while the parallel `viz` code strips hold the complete text. → **complete the captions, no rebuild.**

**Upgrade decision rule (auditor skill + this batch's explicit directive):** P0 = a hard, inherently-**spatial/dynamic** concept whose current visual (static SVG build **or** existing iframe) does **not** serve the learning question, where a **live, draggable/steppable** interactive shows something a static frame cannot, **and** no existing `viz/*.html` already serves it. **Terminal-acceptable** = code-idiom / recipe / comparison / awareness lesson where a fake terminal + shape prints is the right medium. **Already-visual** = embeds an interactive `viz/*.html` that serves the question. **Do not blindly add D3.**

---

## 3. Visual audit results (26 lessons)

"Current visual" = interactive iframe embedded in `var BUILD`, or the static SVG scroll-reveal build.

| Lesson | Concept | Diff. | Playground | Current visual | Score | Verdict |
|--------|---------|:---:|---|---|:---:|:---:|
| m06 · d01 · convolution-pooling | filter slides → feature map; pooling | med | terminal | static: image→filter→one-patch→3 frozen "stops"→map→pool | 2 | **P0** |
| m06 · d02 · output-dim-arithmetic | `out=⌊(W−K+2P)/S⌋+1` | easy | calculator (terminal) | static: count-the-fitting-stops + knob terminal | 4 | terminal-acceptable |
| m06 · d03 · cnn-classifier | conv/pool stack; shape flow | med | terminal | static: 28→14→7 shape-tracker pipeline | 3 | P1 |
| m06 · d04 · batch-normalization | per-channel μ/σ², then γ/β | med | calculator (terminal) | static: numeric raw→norm→scale-shift | 3 | terminal-acceptable |
| m06 · d05 · data-augmentation | flip/crop/jitter → less overfit | easy | static-explanation | static: transforms + overfit-gap curve | 3 | P1 (needs raster) |
| **m06 · d06 · transfer-embeddings** | penultimate layer = embedding; close=similar | med | terminal | static: **3 frozen dots** (2 cats close, 1 car far) | 2 | **P0** |
| m06 · d07 · object-detection | boxes; IoU; 1- vs 2-stage | easy | static-explanation | static: IoU overlap rectangles (awareness day) | 3 | P1 / terminal-ok |
| **m06 · d08 · clip-contrastive** | shared space; N×N grid; pull/push | hard | terminal | static: **pre-coloured 3×3 grid + frozen pull/push dots** | 2 | **P0** |
| m16a · d01 · inference-mechanics | prefill vs decode; roofline; KV | med | visual-mechanism | static: rich 8-step (split, shapes, roofline, KV, batch, MQA) | 4 | P1 |
| m16a · d02 · memory-wall | arithmetic intensity; ridge | med | calculator | **iframe `roofline.html`** + static | 5 | already-visual |
| **m16a · d03 · batching-economics** | batch amortizes weights to `B*`; KV vs HBM | med→adv | calculator | static: strong 8-step (crossover, KV bars, MHA/GQA/MQA) | 4 | **P0 (directive)** |
| m16a · d04 · serving-llama3-tpus | KV in GB; PagedAttention; disaggregation | adv | calculator | static: richest 8-step (paging, disaggregation) | 4 | P1 |
| m16a · d05 · quantization-intro | int8 halves bytes → ~2× decode tok/s | med | visual-mechanism | **iframe `quantization.html`** + code-walkthrough | 4 | already-visual (+note fix) |
| m16a · d06 · calculator-roofline | ridge = FLOPs/s ÷ BW; predict tok/s | med→adv | calculator | **iframe `roofline.html`** + code-walkthrough | 4 | already-visual (+note fix) |
| m17a · d01 · quantization-theory-int8 | absmax int8; outliers; LLM.int8() | core | terminal | **iframe `quantization.html`** + 9-step build | 5 | already-visual |
| m17a · d02 · incoherent-processing-quip | rotate-then-round smears outliers | med | terminal | static: ~8 SVGs (rotation smear, Hadamard) | 4.5 | already-visual / P1 |
| m17a · d03 · quip-sharp | 2-bit; Hessian / E₈ / trellis | hard | terminal | **iframe `quantization.html`** + 7 SVGs (E₈, trellis) | 4.5 | already-visual |
| m17a · d04 · additive-quantization-qtip | AQ = sum of codebooks; AQLM/QTIP | hard | terminal | static: 7-step (codebook sum, AQLM pipeline) | 4 | already-visual / P1 |
| **m17a · d05 · memory-economics-synthesis** | roofline AI of decode matmul @ fp16/int8/2-bit | med | terminal | static only — **roofline lesson with no roofline interactive** | 3 | **P0 (embed existing viz)** |
| m17a · d06 · ptq-perplexity | PTQ via bitsandbytes; perplexity tax | applied | terminal | static: 7-step (re-teaches d01 mechanism) | 3.5 | terminal-acceptable / P1 |
| m17b · d01 · decoding-innovations | KV cache growth past 100k | intro | terminal | **iframe `kv-cache.html`** + 8-step build | 5 | already-visual |
| m17b · d02 · cache-eviction-snapkv | vote→pool→top-k→gather | med | terminal | static: strong 8-step SnapKV storyboard | 4 | P1 |
| m17b · d03 · gqa-mqa | 32 Q → 8/1 shared K/V heads | med | calculator | static: 7-step head boxes (topology) | 3 | P1 (borderline) |
| m17b · d04 · speculative-decoding | draft K, verify in parallel | med→adv | visual-mechanism | **iframe `speculative-decoding.html`** + 13-step | 5 | already-visual |
| **m17b · d05 · ring-attention** | rotate K/V blocks around device ring | adv | terminal | static: **frozen 4-device ring** (rotation narrated) | 3 | **P0** |
| m17b · d06 · systems-review-essay | synthesis: roofline, Flash, hardware lottery | adv | static-explanation | static: 8-step roofline storyboard (essay artifact) | 4 | terminal-acceptable |

### 3.1 The finding: this tier is visually mature — 8 iframes + uniformly rich static builds

- **8 lessons are already-visual** — they embed an interactive `viz/*.html` that serves the learning question: `m16a/d02` + `m16a/d06` (roofline), `m16a/d05` + `m17a/d01` + `m17a/d03` (quantization), `m17b/d01` (kv-cache), `m17b/d04` (speculative-decoding). Per "don't blindly convert every playground to D3," these get **Coach prose only, no new lab**.
- **The `m16a`/`m17a`/`m17b` systems lessons ship rich static builds** — 7–10 purpose-built inline SVGs per lesson (prefill/decode shapes, roofline verdicts, KV-growth bars, SnapKV storyboards, E₈ lattices, AQLM pipelines). Auditors scored these **4–4.5/5** → mostly **P1/terminal-acceptable, not P0**.
- **`m06` (vision) is the genuinely under-served module** — it is the **most spatial topic in the whole curriculum yet embeds zero interactive iframes**; 4 of its 8 lessons score ≤2 on their crux step because the crux is *motion* or *relative position* shown only as frozen SVG snapshots.

### 3.2 P0 set — **6 upgrades, evidence-driven and directive-aligned**

Concentrated where interactivity genuinely beats the existing visual **and** the parent request named the grammar item. Five are new inline `.vlab` labs; one is an embed of an **existing** `viz/*.html` (not new D3).

1. **`m06/day-01-convolution-pooling` — "Sliding-kernel + pooling sweep"** _(grammar m06-1: feature-map/pooling)._ A filter window slides across a small pixel grid via a slider (or step control); the covered patch highlights, the multiply-add runs live, and the matching feature-map cell fills in; a second control runs the 2×2 max-pool sweep. **Why static can't serve it:** the whole idea is *motion + weight-reuse across positions + one-patch→one-cell correspondence*; the build shows only three disconnected "stops." Foundational — everything downstream builds on it.
2. **`m06/day-06-transfer-learning-embeddings` — "Embedding-space explorer"** _(grammar m06-2: embedding space)._ A 2D scatter of image points (cats/dogs/cars); drag a query point and its nearest-neighbour distances/ranking update live, so "close = similar" becomes something you *do*. **Why static can't serve it:** the learning question is *relative distance in a space*, meaningless without moving a point; the build shows three frozen dots. Reusable intuition feeds `day-08`.
3. **`m06/day-08-clip-contrastive` — "CLIP similarity grid + contrastive pull/push"** _(grammar m06-3 **and** m06-4 — two items)._ An N×N image×text cosine-similarity grid where hovering a cell shows the score and a "train step" control brightens the diagonal while dimming off-diagonal; a companion 2D space pulls matched image/caption dots together and pushes mismatched ones apart per step. **Why static can't serve it:** contrastive learning *is* a dynamic (diagonal vs off-diagonal tension over steps); the build shows only the pre-coloured endpoint. Module finale, hardest concept.
4. **`m16a/day-03-batching-economics` — "Batching & KV-budget lab"** _(grammar m16a-2 **and** m16a-3: request-queue/batching + latency vs throughput)._ Drag batch size `B`: the step-time-per-token curve bends at the ridge `B*`, total throughput rises while per-user latency rises, and the KV-cache bar simultaneously eats into an HBM budget line; an MHA/GQA/MQA toggle shrinks the cache. **Why static can't serve it:** the concept is *two curves fighting as one dial moves* — throughput up, latency up, memory up together. **No batching/queue viz exists in the library** (roofline/quantization iframes cover m16a's other grammar items, so this is the one lesson carrying m16a's distinctive grammar).
5. **`m17b/day-05-ring-attention` — "Ring-rotation stepper"** _(grammar m17b-4: ring attention)._ N devices on a ring; a step/play control rotates each K/V block one hop per step, highlighting the local `Q_i @ K_j` compute and the online-softmax running max/sum accumulating, with a counter showing per-device memory staying flat while total sequence grows; an N-devices slider. **Why static can't serve it:** the concept *is* motion + comm/compute overlap + a running-softmax merge across visits — a static ring animates none of it. **No `ring-attention.html` exists** (`parallelism.html` is DP/FSDP/TP/PP on a mesh, zero overlap).
6. **`m17a/day-05-memory-economics-synthesis` — embed existing `../../viz/roofline.html`** _(grammar m17a: memory-economics via roofline — NOT new D3)._ This is a roofline-arithmetic-intensity lesson (decode matmul at fp16/int8/2-bit) that is currently static-only; the perfect-fit interactive **already exists and is unused here**. Add it as build step 1 (drag AI/bandwidth across the ridge), mirroring how `m17a/day-01` + `day-03` embed `quantization.html`. Lowest-cost, high-fit P0.

**Deliberately NOT given a new lab (avoid gold-plating / duplication):**

- **`m17b/day-03-gqa-mqa`** (head-sharing topology) — `viz/kv-cache.html` **already** gives this lesson a live group-size slider + MHA/GQA/MQA selector (the quantitative half). A new viz would add only the head-regrouping *topology* picture — marginal. Held as **documented P1**.
- **`m16a/day-01`** (prefill/decode timeline), **`m16a/day-04`** (paging/fragmentation), **`m17a/day-02`** (Hadamard smear), **`m17a/day-04`** (codebook sum), **`m17b/day-02`** (SnapKV eviction) — all have strong static builds (score 4–4.5). Coach prose only.
- **`m06/day-02,03,04,05,07`** — arithmetic / pipeline / distributional / raster-image-dependent / awareness. Static-adequate or wrong medium for a light conceptual viz. Coach prose only.

**P0 count = 6** (5 new inline `.vlab` + 1 existing-viz embed), of which **3 are in `m06`** — appropriate, because `m06` is the one module with zero interactivity in the most spatial subject. This covers **all four `m06` grammar items**, `m16a`'s batching grammar, `m17a`'s memory-economics-via-roofline, and `m17b`'s ring — while leaving every already-visual and adequate-static lesson alone.

---

## 4. Per-lesson Coach Layer work (add only what is missing)

Missing Coach blocks are inserted **inside `.sec-body`, before that section's `.gotit`**, using the exact templates in `COACH_STYLE_GUIDE.md` §5. Legend: **●** add · **○** already present (leave it) · **–** intentionally skip (documented).

| Lesson | 😕 pain | 🪜 Ladder | 🎤 interview | Staff Lens | ✔ Accept | 📓 log | 中文 | P0 visual |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| m06 · d01 · convolution-pooling | ● | ● (dot-product) | ● | ○ | ● | ● | ● | **● lab** |
| m06 · d02 · output-dim-arithmetic | ● | ● (`⌊(W−K+2P)/S⌋+1`) | ● | ○ | ● | ● | ● | – |
| m06 · d03 · cnn-classifier | ● | – (shape-trace) | ● | ○ | ● | ● | ● | – |
| m06 · d04 · batch-normalization | ● | ● (`(x−μ)/√(σ²+ε)`) | ● | ○ | ● | ● | ● | – |
| m06 · d05 · data-augmentation | ● | – (no core formula) | ● | ○ | ● | ● | ● | – |
| m06 · d06 · transfer-embeddings | ● | ● (cosine sim) | ● | ○ | ● | ● | ● | **● lab** |
| m06 · d07 · object-detection | ● | ● (IoU) | ● | ○ | ● | ● | ● | – |
| m06 · d08 · clip-contrastive | ● | ● (InfoNCE, compact) | ● | ○ | ● | ● | ● | **● lab** |
| m16a · d01 · inference-mechanics | ● | ● (arithmetic intensity) | ● | ○ | ○ | ● | ● | – |
| m16a · d02 · memory-wall | ● | ● (ridge point) | ● | ○ | ○ | ● | ● | – |
| m16a · d03 · batching-economics | ● | ● (`B*`) | ● | ○ | ○ | ● | ● | **● lab** |
| m16a · d04 · serving-llama3-tpus | ● | ● (KV GB) | ● | ○ | ○ | ● | ● | – |
| m16a · d05 · quantization-intro | ● | ● (bytes→tok/s) | ● | ○ | ○ | ● | ● | – (note fix) |
| m16a · d06 · calculator-roofline | ● | ● (ridge = FLOPs/s÷BW) | ● | ○ | ○ | ● | ● | – (note fix) |
| m17a · d01 · quantization-theory-int8 | ● | ● (absmax scale) | ● | ○ | ○ | ● | ● | – |
| m17a · d02 · incoherent-processing-quip | ● | ● (incoherence μ) | ● | ○ | ○ | ● | ● | – |
| m17a · d03 · quip-sharp | ● | – (survey of 4) | ● | ○ | ○ | ● | ● | – |
| m17a · d04 · additive-quantization-qtip | ● | ● (codebook sum) | ● | ○ | ○ | ● | ● | – |
| m17a · d05 · memory-economics-synthesis | ● | ● (AI at 1 byte) | ● | ○ | ○ | ● | ● | **● embed** |
| m17a · d06 · ptq-perplexity | ● | ● (perplexity) | ● | ○ | ○ | ● | ● | – |
| m17b · d01 · decoding-innovations | ● | ● (KV bytes) | ● | ○ | ○ | ● | ● | – |
| m17b · d02 · cache-eviction-snapkv | ● | ● (top-k score) | ● | ○ | ○ | ● | ● | – |
| m17b · d03 · gqa-mqa | ● | ● (cache ∝ n_kv) | ● | ○ | ○ | ● | ● | – |
| m17b · d04 · speculative-decoding | ● | ● (expected accept) | ● | ○ | ○ | ● | ● | – |
| m17b · d05 · ring-attention | ● | ● (per-device mem) | ● | ○ | ○ | ● | ● | **● lab** |
| m17b · d06 · systems-review-essay | ● | – (essay) | ● | ● | ○ | ● | ● | – |

**Totals to add:** pain-point ×26 · Math Ladder ×21 (5 documented skips: m06/d03 shape-trace, m06/d05 no core formula, m17a/d03 4-method survey, m17b/d06 essay — plus ladders kept compact where a derivation already exists) · interview ×26 · Staff Lens ×1 (`m17b/d06`) · Acceptance criteria ×8 (all `m06`) · research log ×26 · bilingual scaffold ×26 · **5 inline `.vlab` labs + 1 `roofline.html` embed**.

**Notes / guardrails:**

- **Math Ladder only for a genuine single core formula.** 5 documented skips (above). Where a full "Step 1/2/3" derivation already exists in a lesson, the ladder is **compact** — it labels the existing steps and adds the missing **sanity-check rung** rather than duplicating the derivation. The math-ladder advisory persists on the skipped files **by design** (reported, not "fixed").
- **Produce artifact — do NOT fabricate.** Every lesson already references a concrete artifact. `m17b/day-06` intentionally uses a synthesis **essay `.md`** (not `experiment.py`) — correct for a capstone-essay day; **keep it.** Its `lesson_audit.py` "no experiment.py" advisory is expected and documented.
- **Acceptance criteria.** Present on all 18 `m16a`/`m17a`/`m17b` lessons (verify, keep). Add an explicit `<h4>Acceptance criteria</h4>` + checklist to all 8 `m06` lessons (they only have prose today).
- **Interview + Staff Lens.** Interview: all 26 lack a real 🎤 spoken-answer block (auditors confirmed; `lesson_audit.py` under-reports because "interview" appears in prose) → add one to each. Staff Lens: present on 25/26 (preserve); add a labeled §4 Staff Lens to the `m17b/day-06` essay.
- **Chinese dosage:** 2–4 short touches per lesson, concentrated in §1 (pain) and §2 (intuition) only — never in formulas, code, or interview phrasing (guide §3; the batch directive "Keep Chinese light: 2–4 intuition touches"; memory `prefers-english-learning-content`). These lessons are currently English-only, so this adds the light bilingual scaffold the batch asks for.
- **No motivational fluff** (explicit directive). Every added block carries technical content: a named confusion, a labeled formula, a spoken answer with a real caveat, a concrete log prompt.

---

## 5. P0 visual specs (5 inline `.vlab` labs + 1 embed)

The five inline labs are **dependency-free inline SVG + vanilla JS** using the repo `.vlab` template proven in Batch 1/2/3 (the CSS block at `m10a-scaling-laws/day-01-kaplan-paradigm/lesson.html:196–224`, copied verbatim — new-shell CSS vars only). **No CDN, no new fonts, offline-safe, theme-aware, keyboard-reachable controls.** Each lab is inserted into **Section 3** (`data-sec="play"`) **above** the existing fake-terminal playground; the terminal keeps its 3 `data-demo` buttons and its **completion-gate** role (untouched). Being inline (not iframed), the `viz-iframe-autoresize-bug` concern does not apply. Each carries the 7 required cards: learning question (`.vlab-q`) · labeled axes/legend · clear change on control move · "what you should notice" · misconception note · "where this simplifies reality" (`.vlab-note`) · frontier-lab relevance. **All lab math is hand-verified in Node before shipping.**

### 5.1 `m06/day-01` — "Sliding-kernel + pooling sweep"
- **Learning question:** how does one small shared filter, slid over every position, turn an image into a feature map — and what does pooling then throw away?
- **Panel A:** a small input grid (e.g. 5×5 of integers) with a 3×3 filter window. A **position** slider (or ◀▶ steps) moves the window across valid positions; the covered patch highlights, the elementwise multiply-add is shown live, and the corresponding **feature-map cell fills in** as the window passes over it.
- **Panel B / control 2:** a **2×2 max-pool** toggle that sweeps the finished feature map and greys the three discarded values per window, keeping the max.
- **Notice / misconception / simplifies:** the *same 9 weights* are reused at every stop (weight sharing); misconception = "each output pixel has its own filter"; simplification = one channel, integer weights, `stride=1`, no padding/activation.

### 5.2 `m06/day-06` — "Embedding-space explorer"
- **Learning question:** why is "similar images sit close together" a claim about *distance in a space*, not about the pixels?
- **One panel:** a 2D scatter of ~9 labelled points (cats / dogs / cars) in a fixed embedding layout. A **draggable query point**; as you drag it, the **k nearest neighbours** re-rank and highlight, and a readout shows the cosine/Euclidean distance to each class centroid — so dragging a point *into* the cat cluster flips its nearest neighbours to cats.
- **Notice / misconception / simplifies:** clusters = classes the network learned to separate; misconception = "the axes mean something specific" (a 2D projection of a 512-D space has no named axes); simplification = 2D toy of a high-D space, fixed points, no real encoder.

### 5.3 `m06/day-08` — "CLIP grid + contrastive pull/push"
- **Learning question:** what does contrastive training actually *do* to the image and text vectors, and why does the similarity matrix end up bright on the diagonal?
- **Panel A:** an N×N (e.g. 4×4) image×text **cosine-similarity grid**; hovering a cell shows its score. A **"train step"** button runs a few steps: diagonal cells brighten toward 1, off-diagonal cells dim toward 0.
- **Panel B:** a 2D space with matched image/caption dot-pairs; each train step **pulls matched pairs together** and **pushes mismatched pairs apart**, in sync with the grid.
- **Notice / misconception / simplifies:** the diagonal is the "correct pairing"; the loss maximizes diagonal similarity while suppressing off-diagonal; misconception = "CLIP classifies images" (it *aligns two encoders*; zero-shot classification is a consequence); simplification = toy scores updated by a fixed pull/push rule, not a real InfoNCE gradient.

### 5.4 `m16a/day-03` — "Batching & KV-budget lab"
- **Learning question:** when you raise the decode batch size, why do throughput, per-user latency, and memory pressure all move at once — and where's the sweet spot?
- **One panel, two synced readouts:** a **batch-size `B`** slider drives (1) a step-time-per-token curve that is flat-then-rising, bending near the ridge `B*` where decode crosses from memory-bound to compute-bound, with tokens/sec and per-user latency readouts moving in opposite helpfulness; and (2) a stacked **HBM budget bar** — weights + KV-cache(B) — where the KV block grows with `B` until it crosses the device-RAM line (the OOM moment).
- **Control 2:** an **MHA / GQA / MQA** toggle that shrinks the KV block (and pushes the OOM point right).
- **Notice / misconception / simplifies:** bigger batches buy throughput but cost latency *and* memory; misconception = "just crank the batch size" (KV cache OOMs, and latency SLOs break first); simplification = one layer's arithmetic, fixed seq-len, no continuous batching / paging (that's `day-04`).

### 5.5 `m17b/day-05` — "Ring-rotation stepper"
- **Learning question:** how do N devices compute *exact* full attention over a sequence too long for any one chip's memory, by passing K/V blocks around a ring?
- **One panel:** N device nodes on a ring (N-slider, 2→8), each holding its local Q block and one K/V block. A **step/play** control rotates every K/V block one hop clockwise per step; at each step the local `Q_i @ K_j` cell highlights and the **online-softmax running max/sum** per device updates; after N−1 steps every device has seen every block → exact attention.
- **Side counter:** per-device memory stays flat at O(block) while a **total sequence length** readout grows with N — "context scales with devices, not chip memory."
- **Notice / misconception / simplifies:** the ring gives *exact* attention (not an approximation); the win is overlapping the block transfer with compute; misconception = "ring attention approximates attention like sparse/eviction does" (it does not — it's exact); simplification = perfect ring, no bandwidth stalls, no causal-mask load-imbalance.

### 5.6 `m17a/day-05` — embed `../../viz/roofline.html`
- **Not a new lab.** Insert one `build-embed` step at the **front of `var BUILD`** using the exact markup `m17a/day-01` uses (`{viz:'<div class="build-embed"><iframe src="../../viz/roofline.html" title="Interactive" loading="lazy"></iframe><div class="cap">interactive — drag the controls. <a href="../../viz/roofline.html" target="_blank" rel="noopener">open full screen ↗</a></div></div>', note:'…'}`), matching the lesson's existing BUILD quoting style. The parent already carries the iframe-auto-resize receiver script and `roofline.html` already broadcasts its height (proven in `m16a/day-02`/`day-06`), so no new plumbing is needed — verify both at edit time.

> **Reuse note:** the five inline labs reuse the exact `.vlab` CSS + drawing-IIFE pattern from Batch 1/2/3. Where a lesson's learning question is already served by an embedded interactive iframe (8 lessons) or an adequate static build, **no new lab is built** — Coach prose only.

---

## 6. Artifact + Explain-back requirements (per the parent request)

The parent request requires each lesson to end with a **concrete artifact**, each Produce section to carry **acceptance criteria**, and each artifact to include an **Explain-back requirement**: 5–7 sentences explaining what was built, why it scales that way, and what would break in production/research if ignored.

- **Concrete artifact:** every lesson already has one (`experiment.py` or, for the `m17b/day-06` capstone, an essay `.md`). **Kept as-is.**
- **Acceptance criteria:** present on the 18 `m16a`/`m17a`/`m17b` lessons; the explicit `<h4>Acceptance criteria</h4>` heading + checklist is **added to all 8 `m06`** lessons.
- **Explain-back:** added to **all 26** as an explicit line folded into the 📓 research-log block — a required `log.md` (or the essay `.md`) entry of 5–7 sentences: (1) what was computed/built, (2) why the result scales the way it does, (3) what breaks in production/research if ignored. This upgrades the existing one-line `log.md` nudge into the richer explain-back, without removing the original question.

---

## 7. Preservation contract (never violate — guide §7)

For every edited lesson, these must be byte-behaviour-identical after the edit:

- `data-quest-id` on `<body>` (the localStorage key `frontier-lesson:<id>`).
- Every prev / next / hub `href` (top and sidebar nav).
- 7 sections, their `id`s and `data-sec` keys; 7 `.gotit` buttons; the progress/reset/checklist script.
- `var QS` (4 quiz entries `{q,opts,ans,fb}`), `var DEMOS` (3 keys ↔ 3 `data-demo` buttons), `var BUILD` (scroll-reveal, incl. every interactive iframe and every existing SVG step).
- Glossary tooltip script + every `.term[data-tip]` (new terms get a `data-tip`).
- Self-contained / offline: no new external assets, CDN, or fonts. Chinese renders in the existing system font stack.
- Section wrapper tags, `id`s, and `<div class="sec-body">…</div></section>` boundaries kept intact so `_shell_migrate.py` can still extract cleanly.

**Additive-only.** No failure mode, trade-off, equation, or complexity claim is deleted; no precise language is swapped for cheerleading; no motivational fluff. The one permitted deletion per file is the old one-line `log.md` nudge, folded into the richer research-log + explain-back block.

**In-scope coherence fixes** (guide §7 — factual/label bugs in the edited file): the ~5–6 **truncated `var BUILD` `note` captions** in `m16a/day-05` + `day-06` (finale-truncation class) — completed from the parallel, complete `viz` code strips. Fixed only in those two files (already being edited for Coach Layer); logged in the report. **No BUILD rebuild** — the code-walkthrough steps and the embedded iframes are correct and preserved.

**P0 specifics:** the five inline `.vlab` labs are added **above** the existing terminal playground; the terminal's 3 `data-demo` buttons remain each section's completion gate. Labs are **inline** (not iframed). The one embed (`m17a/day-05`) is added as a `build-embed` step **at the front of `var BUILD`**, exactly matching the proven pattern.

---

## 8. Execution order (sequential — one file per session, no parallel lesson writing)

Per CLAUDE.md §4 and memory `capability-spiral-build`: **no subagents for parallel writing** (tone/analogy drift). Edit lessons one at a time, in module order (foundational day first), running the per-file verify gate after each before moving on. (The visual audit was, and final verification will be, parallelized — read-only only.)

1. `m06-cnns-vision-encoders` — d01 **(+lab)**, d02, d03, d04, d05, d06 **(+lab)**, d07, d08 **(+lab)** (8; Coach prose + Accept criteria + 3 labs)
2. `m16a-inference-economics` — d01, d02, d03 **(+lab)**, d04, d05 **(+note fix)**, d06 **(+note fix)** (6)
3. `m17a-quantization` — d01, d02, d03, d04, d05 **(+embed)**, d06 (6)
4. `m17b-long-context-decoding` — d01, d02, d03, d04, d05 **(+lab)**, d06 **(+Staff Lens)** (6)

P0 lessons get the Coach Layer prose **and** the visual in the same session.

---

## 9. Verification gate (run after each file; summarized in the report)

1. **JS syntax check** — `node --check` (or `vm.Script`) on every inline `<script>` block.
2. **jsdom headless load** — the file loads with no thrown error; asserts 7 sections / 7 got-it / 4 quiz / 3 demos / BUILD / tooltip controller. For each P0 lab file, dispatch `input`/`click` on a control and assert the `.vlab` SVG re-renders and readouts respond.
3. **`python3 sessions/lesson_audit.py <changed paths…>`** (targeted — leaves `_recover_set.json` alone) → **OK**, with the core COACH advisories (pain / interview / accept / log / bilingual + Staff Lens) cleared for that file (except documented Math-Ladder skips and the one intentional essay "no experiment.py" advisory).
4. **Duplicate-callout scan** (memory `coach-layer-duplicate-callout-defect`): after each file, check for adjacent identical `<div class="callout…>` lines (the Batch 2 double-paste defect). Ignore the legitimate 🧮 / 🪜 math-ladder mono-wrapper false-positive.
5. **`git diff` audit per file:** additive callouts + (P0) one `.vlab` CSS block + IIFE (or one embed step); confirm **no** `data-quest-id` / `data-demo` / `data-sec` / `var QS|DEMOS|BUILD` / nav-`href` / `.gotit` line removed.
6. **Batch-wide final checks** (per the request): stale run commands, duplicated paragraphs, title-spacing glitches, Produce-path vs run-command mismatch, misleading visual captions.

**Definition of done for the batch:** all 26 edited lessons OK with no hard failures; each has its full missing-Coach-element set added; the 5 P0 labs render + interact and the 1 embed loads; nav / quest-id / quiz / playground / build / tooltips preserved everywhere; Chinese light (2–4 touches); depth ≥ before; the truncated-note defect fixed in the two `m16a` files. Then write `sessions/inference_systems_rollout_report.md`.

---

## 10. Risks

1. **P0 is a judgment call, documented transparently.** This tier is visually mature (8 iframes + rich static builds), so the honest strict-rubric P0 set is small. The 6 chosen upgrades are where interactivity genuinely beats the current visual **and** the parent named the grammar item — 3 concentrated in `m06` (zero prior interactivity, most spatial subject). If the reviewer wants fewer (e.g. m06 only) or a different `m16a`/`m17b` day, that is the adjustable knob.
2. **Concurrent edits.** `sessions/` is edited by other sessions; re-read each file immediately before editing (the tooling enforces "file modified since read").
3. **Per-lesson BUILD quoting style varies** (`{"viz":…}` JSON-style vs `{viz:'…'}` single-quoted). Match each file's existing style exactly when adding the `m17a/day-05` embed step; verify JS parses after.
4. **Documented persistent advisories:** the 5 Math-Ladder skips and the 1 essay "no experiment.py" advisory remain **by design** — listed as intentional in the report, not failures.
5. **No real-browser pixel check** (`no-real-browser-in-sandbox`). All verification is headless (jsdom + JS-syntax + hand-checked lab math); the `.vlab` labs reuse the classes proven in Batch 1/2/3. A final Light/Dim eyeball would need a user screenshot if pixel fidelity matters.
6. **Truncated-note fix scope.** Only the genuinely mid-word-truncated captions in `m16a/day-05`+`day-06` are completed (from the complete `viz` code strips). Complete captions and all other lessons' builds are left untouched.
