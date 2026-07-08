# Batch 3 — Coach Layer Rollout Plan

_Plan-first artifact for the `frontier-curriculum-refactor` Coach Layer rollout to **Batch 3 — the JAX & systems tier**. Written **before any lesson edit**. Date: 2026-07-07._

Companion docs: [`COACH_STYLE_GUIDE.md`](./COACH_STYLE_GUIDE.md) (blocks + rules) · [`enhance_lesson_checklist.md`](./enhance_lesson_checklist.md) (per-lesson checklist) · [`lesson_audit.py`](./lesson_audit.py) (automated gate) · [`batch_1_rollout_report.md`](./batch_1_rollout_report.md) + [`batch_2_rollout_plan.md`](./batch_2_rollout_plan.md) + [`batch_2_rollout_report.md`](./batch_2_rollout_report.md) (the precedent this batch follows).

> **Naming note (two "Batch 3" files, no overlap).** This plan drives `batch_3_rollout_report.md` — the **Coach Layer rollout** for the JAX & systems tier (`m07` / `m08` / `m09a` / `m09c`). It is unrelated to the older [`batch_3_cleanup_report.md`](./batch_3_cleanup_report.md), a **defect-only cleanup** of a *different* scope (the inference-systems tier `m06` / `m16a` / `m17a` / `m17b`) — **no module overlap**. See [`batch_3_cleanup_note.md`](./batch_3_cleanup_note.md) for the one-line disambiguation.

---

## 1. Scope

Batch 3 = the **JAX mental-model + transformer-arithmetic + hardware/systems** tier:

| Module | Lessons | Theme |
|--------|:---:|-------|
| `m07-thinking-in-jax` | 6 | immutability → PRNG keys → vmap → jit → flax/optax → ViT capstone |
| `m08-transformer-math` | 6 | transformer arithmetic → Q/K/V params → KV cache → production step → pretraining logic → rehearsal |
| `m09a-hardware-physics` | 6 | rooflines → TPU/MXU → sharding → multi-device → memory footprint → checkpointing |
| `m09c-sharding-parallelism` | 6 | DP/FSDP → tensor-parallel → pipeline-parallel → LLaMA-3 on TPU → distillation → PyTorch consolidation |

**24 lessons total. 2 are already Coach-complete and are excluded:** `m08/day-01-transformer-arithmetic` and `m08/day-03-kv-cache` were the original `frontier-curriculum-refactor` **pilot** lessons — `lesson_audit.py` fires **zero** COACH advisories on either (full pain/ladder/interview/staff-lens/artifact/accept/log/bilingual set present). Per the guide's "add only what is missing" rule, they are left untouched.

**→ 22 lessons in play.** All 24 are on the **new sidebar shell** (`class="module-section"`, Light/Dim/Dark/Midnight switcher) and all pass `lesson_audit.py` structurally (**24/24 OK**, 0 DEGRADED — 7 sec / 7 got-it / 3 demos / 4 quiz / BUILD present). This is a pure Coach Layer overlay + a **small, focused** interactive-visual pass; **no structural repair work**.

**Baseline advisories (`lesson_audit.py`, all 22):** every lesson already carries an **everyday analogy** (`.relate`/`.card`) and a concrete **Produce artifact**, and all but two carry a **Staff Lens**. Every one of the 22 is **missing**: a pain-point block, a 5-minute research log, a bilingual scaffold, and a real 🎤 interview block. Two lack **Acceptance criteria** (`m07/day-01`, `m07/day-06`); two lack a labeled **Staff Lens** (`m08/day-04`, `m08/day-05`). The "math-heavy but no Math Ladder" advisory fires on most files, but the ladder is added **only** where a lesson has a genuine single core formula (see §5).

---

## 2. Method (how this plan was built)

1. **Inspect** — listed all 24 lessons; confirmed all on the new shell; ran `lesson_audit.py` (targeted, so `_recover_set.json` untouched) → 24/24 OK; identified the 2 pilot-done lessons (zero advisories) and the per-file missing-Coach-element set from the advisory list.
2. **Visual inventory** — grepped every lesson's `var BUILD` literal for an embedded `../../viz/*.html` iframe vs. static inline SVG vs. terminal-text; catalogued the 17-file `viz/` library (roofline, kv-cache, parallelism, transformer-flops, …).
3. **Visual audit** — ran the **`frontier-visual-auditor` rubric over the 22 lessons via 22 read-only per-lesson auditors** (a background workflow; read-only auditing is explicitly allowed to parallelize — only lesson *writing* must be sequential). Each cited evidence: the 3 `data-demo` terminal keys, the `var BUILD` step types (iframe / static-SVG / terminal-text), the §4 core formula, difficulty, a 1–5 visual score, a P0/P1/terminal-acceptable/already-visual verdict, and — for any P0 — *which* systems visual is needed and *why the existing `viz/*.html` does not already serve it*. The 23rd lesson (`m09a/day-01-rooflines`, already-visual) was read in full directly.
4. **Precise Coach-element scan** — the `lesson_audit.py` advisory list + each auditor's `has_analogy`/`has_staff_lens`/`has_artifact`/`has_accept_criteria`/`has_interview` flags give the exact per-file missing set, so this plan adds **only what is missing** and never duplicates an existing Staff Lens / analogy / artifact.

**Upgrade decision rule (auditor skill + this batch's explicit directive):** P0 = a hard, inherently-**spatial systems** concept whose **playground is terminal-only**, where a genuinely **new interactive systems visual** (one of: memory-movement, roofline, device-mesh, all-reduce timeline, batching/KV-growth, sharding-communication) adds value a static frame cannot, **and** no existing `viz/*.html` already serves that learning question. **Terminal-acceptable** = a programming-model / code-idiom / recipe / comparison / case-study lesson where a fake terminal + shape prints is the right medium (explicit rule: *do not blindly add D3 if a terminal walkthrough is appropriate*). **Already-visual** = BUILD embeds an interactive `viz/*.html` that serves the question.

---

## 3. Visual audit results (24 lessons)

Verdicts from the 22 read-only auditors (+ direct read of `m09a/day-01`, + the 2 pilot-done). "Build viz" = what the `var BUILD` scroll-reveal actually renders.

| Lesson | Concept | Diff. | Playground | Build viz | Score | Verdict |
|--------|---------|:---:|---|---|:---:|:---:|
| m07 · d01 · jax-immutability | in-place vs functional `.at[].set()` + purity | easy | terminal | static SVG (memory boxes, causal chain) | 4 | terminal-acceptable |
| m07 · d02 · prng-keys | explicit PRNG keys + `split` | easy | terminal | static SVG (data-flow) | 4 | terminal-acceptable |
| m07 · d03 · vmap | auto-batching a fn; `in_axes` | easy | terminal | terminal-text divs | 3 | terminal-acceptable |
| m07 · d04 · jit | trace → XLA fuse → cache by shape | med | terminal | static SVG (dispatch/fuse/cache/timeline) | 4 | terminal-acceptable |
| m07 · d05 · flax-optax | explicit params + opt_state pytrees | med | terminal | static SVG (2 pytrees, pure step) | 4 | terminal-acceptable |
| m07 · d06 · vit-capstone | ViT shape-flow; overfit-one-batch | med | terminal | static SVG (architecture assembly) | 4 | terminal-acceptable |
| m08 · d01 · transformer-arithmetic | C≈6ND FLOPs | — | — | **iframe `transformer-flops.html`** | 5 | **DONE (pilot)** |
| m08 · d02 · qkv-matrices | 12·d² params/block | easy | terminal | static SVG (shapes, funnel, d² curve) | 4 | P1 |
| m08 · d03 · kv-cache | KV memory growth | — | — | **iframe `kv-cache.html`** | 5 | **DONE (pilot)** |
| m08 · d04 · production-code | 6-stage step; where all-reduce/ckpt sit | med | terminal | static SVG (shard, all-reduce, timeline) | 4 | terminal-acceptable |
| m08 · d05 · pretraining-logic | scaling×data×systems intersection | med | terminal | static SVG (U-curve, Venn, fit) | 3 | P1 |
| m08 · d06 · scaling-rehearsal | recap: AI, KV, C≈6ND (spoken) | med | terminal | static SVG (KV-growth build-up) | 4 | terminal-acceptable |
| m09a · d01 · rooflines | AI vs ridge; roofline | med | terminal | **iframe `roofline.html`** + SVG | 5 | already-visual |
| **m09a · d02 · tpu-architecture** | MXU = systolic array; bf16 | med | terminal | static SVG (2×2 matmul beats) | 4 | **P0 (directive)** |
| m09a · d03 · sharding | AllGather/ReduceScatter/AllReduce cost | med | terminal | static SVG (mesh shard, bytes bar) | 4 | P1 |
| **m09a · d04 · multi-device** | `pmap`/`pmean` SPMD gradient sync | med | terminal | static SVG (split, all-reduce avg) | 4 | **P0 (directive)** |
| **m09a · d05 · memory-footprint** | 16N budget + activations vs batch | med | terminal | static SVG (4N stack, crossover) | 4 | **P0 (directive)** |
| m09a · d06 · checkpointing | remat: O(L)→O(1) memory-for-compute | med | terminal | static SVG (memory stack, recompute) | 4 | P1 |
| m09c · d01 · data-parallel-fsdp | DP 16N vs FSDP 16N/dev | med | terminal | **iframe `parallelism.html`** + text | 5 | already-visual |
| m09c · d02 · tensor-parallel | col-parallel / row-parallel all-reduce | med | terminal | **iframe `parallelism.html`** + text | 5 | already-visual |
| m09c · d03 · pipeline-parallel | bubble = (p−1)/(m+p−1) | med | terminal | **iframe `parallelism.html`** + text | 4 | already-visual |
| m09c · d04 · llama3-tpu | DP/FSDP/TP/PP on a pod (case study) | med | terminal | **iframe `parallelism.html`** + SVG | 5 | already-visual |
| m09c · d05 · distillation | `KL(softmax(z_t/T)‖softmax(z_s/T))` | med | terminal | static SVG (softmax spread, KL gap) | 4 | terminal-acceptable |
| m09c · d06 · pytorch-consolidation | JAX↔PyTorch map + memory calc | med | terminal | **iframe `transformer-flops.html`** + text | 4 | already-visual |

### 3.1 The honest finding: **0 strict-rubric P0**

Every one of the 22 auditors — running independently — concluded that **no lesson is a strict-rubric P0**. The reason is specific to this tier and worth stating plainly:

- **8 lessons are already-visual** — they embed a genuinely interactive `viz/*.html` that serves the learning question (`m08/d01` transformer-flops, `m08/d03` kv-cache, `m09a/d01` roofline, `m09c/d01–d04` parallelism, `m09c/d06` transformer-flops). Per the "don't blindly convert every playground to D3" rule, these get **Coach prose only, no new lab**.
- **The m09a systems lessons already ship strong static-SVG builds.** Unlike Batch 2's terminal-only lessons (weak static SVGs), every `m09a` day carries **7–9 purpose-built inline SVGs** — a beat-by-beat 2×2 systolic matmul, device-mesh shard splits, all-reduce averaging, a four-tenant memory stack, activations-vs-batch crossover curves. Auditors scored these **4/5** ("adequate; interactive would be a nice-to-have") → **P1, not P0**.
- **m07 (JAX) is genuinely terminal-appropriate.** Immutability, PRNG threading, `vmap`/`jit`/Flax/Optax are **programming-model idioms**, not spatial systems mechanisms, and have **no single core formula**. A fake terminal + shape prints is the correct medium; forcing D3 here would violate the explicit rule.

### 3.2 P0 set — **3 directive-driven elevations**, all in `m09a`

The parent request explicitly **named** the systems visuals it prefers (memory-movement, roofline, device-mesh, **all-reduce timelines**, **batching/memory growth**, sharding-communication) and asked to "prefer systems visuals" and "identify P0 lessons where the playground is terminal-only but the concept needs a real systems visual." Following the **Batch 2 precedent** (which elevated auditor-P1 lessons the request named — qkv/residual/layer-norm — to P0 and documented it), three `m09a` P1 lessons are elevated to **P0** because (a) each instantiates a systems-visual type the request named, (b) each is currently terminal-playground + static-only, and (c) **interactivity adds value the frozen SVG cannot** — a draggable parameter that *reshapes the picture live*. Each auditor independently flagged these as the strongest interactive-upgrade candidates.

1. **`m09a/day-02-tpu-architecture` — "Systolic-array beat scrubber."** Named type: *GPU/TPU bottlenecks*. The static build walks a 2×2 matmul beat-by-beat, but the **rhythm** (weights loaded, inputs streaming on staggered beats, partial sums accumulating cell-by-cell) is exactly what static frames can only approximate. A beat slider + a batch-1 toggle that shows the 128×128 grid sitting **idle** makes the staff-lens failure (small matmuls waste the MXU) a thing you *see*. Not covered by any `viz/*.html`.
2. **`m09a/day-04-multi-device` — "All-reduce ring timeline."** Named type: *all-reduce timelines / communication overhead*. A device-count slider drives the ring cost (`2(N−1)` steps, bytes/step) and a compute-vs-comm overlap bar, so the learner watches **communication come to dominate as N grows** — a time-axis view `parallelism.html`'s static mesh does not provide.
3. **`m09a/day-05-memory-footprint` — "Training memory stack."** Named type: *batching memory growth / memory-movement*. Sliders for batch size, N, dtype, and optimizer redraw the four-tenant stack (weights 4N + grads 4N + Adam 8N = 16N) and drag the **activations-cross-the-16N-budget** point live. The auditor called this "the strongest candidate for an interactive upgrade." Not covered by any `viz/*.html`.

**Deliberately NOT given a new lab (to avoid duplication / gold-plating):**

- `m09a/day-03-sharding` (device-mesh / collectives) — `viz/parallelism.html` **already** provides the interactive Device-Mesh + AllGather/ReduceScatter/AllReduce visual (used by `m09c/d01–d04`). A new mesh lab would duplicate it; the strong static SVG build + Coach prose is enough. (If ever wanted, embed `parallelism.html` — noted, not done.)
- `m09a/day-06-checkpointing`, `m08/day-02-qkv`, `m08/day-05-pretraining` — strong static-SVG builds (score 4). Coach prose only.

**P0 count = 3 (all m09a), down from Batch 2's 7** — appropriate: this tier is already the most visually mature (8 interactive iframes + uniformly strong static builds), so the honest, non-gold-plated set is small and concentrated where interactivity genuinely adds a live, draggable systems picture.

---

## 4. Per-lesson Coach Layer work (add only what is missing)

Every lesson gets its missing Coach blocks inserted **inside `.sec-body`, before that section's `.gotit`**, using the exact templates in `COACH_STYLE_GUIDE.md` §5. Legend: **●** add · **○** already present (leave it) · **–** intentionally skip (documented).

| Lesson | 😕 pain | 🪜 Ladder | 🎤 interview | Staff Lens | ✔ Accept | 📓 log | 中文 | P0 lab |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| m07 · d01 · jax-immutability | ● | – (no formula) | ● | ○ | ● | ● | ● | – |
| m07 · d02 · prng-keys | ● | – | ● | ○ | ○ | ● | ● | – |
| m07 · d03 · vmap | ● | – | ● | ○ | ○ | ● | ● | – |
| m07 · d04 · jit | ● | – | ● | ○ | ○ | ● | ● | – |
| m07 · d05 · flax-optax | ● | – (recipe) | ● | ○ | ○ | ● | ● | – |
| m07 · d06 · vit-capstone | ● | – (shape-trace) | ● | ○ | ● | ● | ● | – |
| m08 · d02 · qkv-matrices | ● | ● (`12·d²`) | ● | ○ | ○ | ● | ● | – |
| m08 · d04 · production-code | ● | ● (all-reduce avg) | ● | ● | ○ | ● | ● | – |
| m08 · d05 · pretraining-logic | ● | ● (`D≈20N`) | ● | ● | ○ | ● | ● | – |
| m08 · d06 · scaling-rehearsal | ● | ● (`C≈6ND`) | ● | ○ (verify) | ○ | ● | ● | – |
| m09a · d01 · rooflines | ● | ● (`AI`, ridge) | ● | ○ | ○ | ● | ● | – |
| **m09a · d02 · tpu-architecture** | ● | – (no formula) | ● | ○ | ○ | ● | ● | **●** |
| m09a · d03 · sharding | ● | ● (bytes `3N`/`6N`) | ● | ○ | ○ | ● | ● | – |
| **m09a · d04 · multi-device** | ● | ● (`pmean`) | ● | ○ | ○ | ● | ● | **●** |
| **m09a · d05 · memory-footprint** | ● | ● (`16N`) | ● | ○ | ○ | ● | ● | **●** |
| m09a · d06 · checkpointing | ● | ● (O(L)→O(√L)) | ● | ○ | ○ | ● | ● | – |
| m09c · d01 · data-parallel-fsdp | ● | ● (`16N/dev`) | ● | ○ | ○ | ● | ● | – |
| m09c · d02 · tensor-parallel | ● | ● (col/row) | ● | ○ | ○ | ● | ● | – |
| m09c · d03 · pipeline-parallel | ● | ● (`(p−1)/(m+p−1)`) | ● | ○ | ○ | ● | ● | – |
| m09c · d04 · llama3-tpu | ● | – (case study) | ● | ○ | ○ | ● | ● | – |
| m09c · d05 · distillation | ● | ● (`KL(·/T)`) | ● | ○ | ○ | ● | ● | – |
| m09c · d06 · pytorch-consolidation | ● | ● (memory calc) | ● | ○ | ○ | ● | ● | – |

**Totals to add:** pain-point ×22 · Math Ladder ×14 (8 documented skips) · interview ×22 · Staff Lens ×2 (+1 re-verify) · Acceptance criteria ×2 · research log ×22 · bilingual scaffold ×22 · **3 P0 inline `.vlab` labs**.

**Notes / guardrails:**

- **Math Ladder only for a genuine single core formula.** 8 documented skips — all 6 `m07` lessons (programming-model idioms, `core_formula = none`), `m09a/day-02` (systolic-array mechanism), `m09c/day-04` (case study). Forcing a ladder there would be noise; `lesson_audit.py`'s math-ladder advisory persists on those **by design** (reported, not "fixed"). Where a full "Step 1/2/3" derivation already exists (`m09a/d05`, `m09c/d03`, `d05`), the ladder is **compact** — it labels the existing steps and adds the missing **sanity-check rung** rather than duplicating the derivation.
- **Produce artifact — do NOT fabricate.** Every lesson already references a concrete artifact. **3 lessons intentionally use a `.md` artifact instead of `experiment.py`** — `m08/day-05` (`feinberg-gemini-flash-pretraining.md`, a synthesis/reading day), `m08/day-06` (`scaling_book_exercises.md` + `rehearsal_script.md`, a spoken-rehearsal day), `m09c/day-04` (`llama3-on-tpus.md`, a case-study day). These are correct for their lesson type; **keep them**, do not invent an `experiment.py`. The `lesson_audit.py` "no experiment.py" advisory on these is expected and documented.
- **Interview + Staff Lens.** Interview: **all 22 lack a real 🎤 spoken-answer block** (some contain the word "interview" only in prose, so `lesson_audit.py` under-reports; the visual auditors confirmed `has_interview=false` on all 22) → add one to each. Staff Lens: present on 20/22 (preserve); add a labeled §4 Staff Lens to `m08/day-04` and `m08/day-05`; re-verify `m08/day-06` (has trade-off callouts + a failure warning but maybe no labeled block) and add only if genuinely absent.
- **Chinese dosage:** 2–4 short touches per lesson, concentrated in §1 (pain) and §2 (intuition) only — never in formulas, code, or interview phrasing (guide §3; the user's `Keep Chinese light: 2–4 intuition touches` directive for this batch; memory `prefers-english-learning-content`). These lessons are currently English-only, so this adds the light bilingual scaffold the batch asks for.

---

## 5. P0 visual upgrades (3 labs) — design specs

All three are **dependency-free inline SVG + vanilla JS** using the repo `.vlab` template proven in Batch 1/2 (the CSS block at `m10a-scaling-laws/day-01-kaplan-paradigm/lesson.html:197–224`, copied verbatim — it uses only new-shell CSS vars). **No CDN, no new fonts, offline-safe, theme-aware, keyboard-reachable sliders.** Each lab is inserted into **Section 3** (`data-sec="play"`) **above** the existing fake-terminal playground; the terminal keeps its 3 `data-demo` buttons and its **completion-gate** role (untouched). Being inline (not iframed), the `viz-iframe-autoresize-bug` concern does not apply. Each carries the 7 required cards: learning question (`.vlab-q`) · labeled axes · clear change on control move · "what you should notice" · misconception note · "where this simplifies reality" (`.vlab-note`) · frontier-lab relevance. All lab math is hand-verified in Node before shipping.

### 5.1 `m09a/day-02-tpu-architecture` — "Systolic-array beat scrubber"

- **Learning question:** how does a fixed grid of multiply-accumulate cells do a whole matmul by *streaming* data through on staggered beats — and why does a batch-1 matmul waste it?
- **One panel:** an `n×n` MXU grid (default 3×3) computing a concrete small matmul `C = A·B`. A **beat** slider (0 → 2n−1) advances the staggered wavefront: inputs enter from the left/top, each cell shows its running partial sum, and finished `C` entries light up as they emerge.
- **Control 2:** a **utilization** toggle (full batch vs batch-1) greys the cells that sit idle when the operand is too small — the grid lights up ~`1/n` of its cells, throughput readout drops, no error. That *is* the staff-lens silent failure.
- **Notice / misconception / simplifies:** the array reuses each loaded weight across many beats (that is the whole point); misconception = "the TPU does the matmul in one clock" (it takes ~`2n−1` beats to fill/drain); simplification = tiny integer grid, no HBM/VMEM latency, one pass.

### 5.2 `m09a/day-04-multi-device` — "All-reduce ring timeline"

- **Learning question:** as you add devices, why does the *communication* to keep gradients in sync start to dominate the step, even though each device does less math?
- **One panel, time on the x-axis:** per-device rows showing a **compute** block then a ring **all-reduce** block (`2(N−1)` steps, each moving `~data/N` bytes). A total-step-time readout splits into compute vs comm.
- **Controls:** a **device-count `N`** slider (2 → 64) and a **link-bandwidth** slider. Raising `N` shrinks per-device compute but lengthens the ring; lowering bandwidth stretches the comm block until it dominates — the memory-movement bottleneck, made visible on a timeline.
- **Notice / misconception / simplifies:** ring all-reduce time grows with `N` and shrinks with bandwidth, not FLOPs; misconception = "more devices always = proportionally faster" (comm overhead caps the speedup); simplification = perfect ring, no overlap of compute/comm, uniform links. (Distinct from `parallelism.html`, which shows *which* collective on a mesh, not a *time-axis* cost.)

### 5.3 `m09a/day-05-memory-footprint` — "Training memory stack"

- **Learning question:** the number people quote is "weights = 4N" — so why does a 1B model OOM on a 40 GB card long before you'd expect?
- **One panel:** a stacked memory bar — **weights (4N) + gradients (4N) + Adam state (8N) = 16N static**, then **activations** stacked on top — against a horizontal **device-RAM budget** line.
- **Controls:** **batch-size** and **sequence-length** sliders grow the activations block until it crosses the budget line (the OOM moment); a **dtype** toggle (fp32/bf16) halves the static bytes; an **optimizer** toggle (Adam 8N ↔ SGD 0N) drops the tallest static tenant.
- **Notice / misconception / simplifies:** activations — not the static 16N — are what usually blow the budget, and they scale with batch×seq; misconception = "model memory = parameter count" (that's 1 of 4 tenants); simplification = fixed per-layer activation estimate, no gradient-checkpointing (that's `day-06`), no fragmentation.

> **Reuse note:** these three inline labs reuse the exact `.vlab` CSS + drawing-IIFE pattern from Batch 1/2. Where a lesson's learning question is already served by an embedded interactive iframe (`m09a/d01`, `m09c/d01–d04`, `d06`) or by an adequate static-SVG build (`m08/d02`, `m08/d05`, `m09a/d03`, `d06`, `m09c/d05`), **no new lab is built** — Coach prose only.

---

## 6. Artifact + Explain-back requirements (per the parent request)

The parent request requires that **each lesson end with a concrete artifact**, each Produce section carry **acceptance criteria**, and each artifact include an **Explain-back requirement**: *5–7 sentences explaining what was built, why it scales that way, and what would break in production/research if ignored.*

- **Concrete artifact:** every lesson already has one (`experiment.py` or a lesson-appropriate `.md`). **Kept as-is**; only `m07/day-01` and `m07/day-06` need the explicit `<h4>Acceptance criteria</h4>` heading added (the others already have one).
- **Acceptance criteria:** verified present on 20/22; added to the 2 above.
- **Explain-back:** added to **all 22** as an explicit line inside the Produce section's Acceptance-criteria list **and** folded into the 📓 research-log block — a required `log.md` (or the lesson's `.md` artifact) entry of **5–7 sentences**: (1) what was computed/built, (2) why the result scales the way it does, (3) what breaks in production/research if this is ignored. This upgrades the existing "write one line in log.md" nudge into the richer explain-back the request asks for, without removing the original question.

---

## 7. Preservation contract (never violate — guide §7)

For every edited lesson, these must be byte-behaviour-identical after the edit:

- `data-quest-id` on `<body>` (the localStorage key `frontier-lesson:<id>`).
- Every prev / next / hub `href` (top and sidebar nav).
- 7 sections, their `id`s and `data-sec` keys; 7 `.gotit` buttons; the progress/reset/checklist script.
- `var QS` (4 quiz entries `{q,opts,ans,fb}`), `var DEMOS` (3 keys ↔ 3 `data-demo` buttons), `var BUILD` (scroll-reveal, incl. every interactive iframe).
- Glossary tooltip script + every `.term[data-tip]` (new terms get a `data-tip`).
- Self-contained / offline: no new external assets, CDN, or fonts. Chinese renders in the existing system font stack.
- Section wrapper tags, `id`s, and `<div class="sec-body">…</div></section>` boundaries kept intact so `_shell_migrate.py` can still extract cleanly.

**Additive-only.** No failure mode, trade-off, equation, or complexity claim is deleted; no precise language is swapped for cheerleading; no motivational fluff. The one permitted deletion per file is the old one-line `log.md` nudge, folded into the richer research-log + explain-back block (as in Batch 1/2).

**In-scope coherence fixes** (guide §7 — factual/label bugs in the edited file): the undefined-CSS-var `var(--text2)` → `var(--ink2)` sub-line (seen in `m07/d02`, `m07/d04`, `m08/d02`), and the truncated `var BUILD` note text in `m09c/d06` (finale-truncation class). Fixed only in files already being edited; logged in the report. The stale `M1 Day 3` CSS *comment* is shared boilerplate across all lessons — left untouched.

**P0 specifics:** the inline `.vlab` is added **above** the existing terminal playground; the terminal's 3 `data-demo` buttons remain the section's completion gate. Labs are **inline** (not iframed).

---

## 8. Execution order (sequential — one file per session, no parallel lesson writing)

Per CLAUDE.md §4 and memory `capability-spiral-build`: **no subagents for parallel writing** (tone/analogy drift). Edit lessons one at a time, in module order (foundational day first), running the per-file verify gate after each before moving on. (The visual audit and final verification are read-only and *were*/*will be* parallelized — only writing is sequential.)

1. `m07-thinking-in-jax` — d01, d02, d03, d04, d05, d06 (6; Coach prose, no labs, no ladders)
2. `m08-transformer-math` — d02, d04, d05, d06 (4; Coach prose + ladders + 2 Staff Lens)
3. `m09a-hardware-physics` — d01, d02 **(+P0)**, d03, d04 **(+P0)**, d05 **(+P0)**, d06 (6)
4. `m09c-sharding-parallelism` — d01, d02, d03, d04, d05, d06 (6)

P0 lessons get the Coach Layer prose **and** the inline `.vlab` in the same session.

---

## 9. Verification gate (run after each file; summarized in the report)

1. JS syntax check via `/tmp/batch1_verify.js` (`vm.Script` on every inline `<script>`).
2. jsdom headless load (`/tmp/node_modules/jsdom`, via `node /tmp/batch1_verify.js <path> [--p0]`) — no thrown error; asserts 7 sections / 7 got-it / 4 quiz / 3 demos / BUILD / tooltip controller. For the 3 P0 files, `--p0` dispatches `input`/`click` and asserts the `.vlab` SVG re-renders and readouts respond.
3. `python3 sessions/lesson_audit.py <changed paths…>` (targeted — leaves `_recover_set.json` alone) → **OK**, with the six core COACH advisories (pain / interview / accept / log / bilingual + Staff Lens) cleared for that file (except the documented Math-Ladder skips and the 3 intentional `.md`-artifact "no experiment.py" advisories).
4. **Duplicate-callout scan** (memory `coach-layer-duplicate-callout-defect`): after each file, check for adjacent identical `<div class="callout…>` lines (the Batch 2 double-paste defect). Ignore the legitimate 🧮 math-ladder mono-wrapper false-positive.
5. `git diff` audit per file: additive callouts + (P0) one `.vlab` CSS block + IIFE; confirm **no** `data-quest-id` / `data-demo` / `data-sec` / `var QS / DEMOS / BUILD` / nav-`href` / `.gotit` line removed.

**Definition of done for the batch:** all 22 edited lessons OK with no hard failures; each has its full missing-Coach-element set added; the 3 P0 labs render and interact; nav / quest-id / quiz / playground / build / tooltips preserved everywhere; Chinese light (2–4 touches); depth ≥ before. Then write `sessions/batch_3_rollout_report.md`.

---

## 10. Risks

1. **0 strict-rubric P0 — P0 count is a directive judgment.** The audit honestly found no lesson meets the pure P0 rubric (all are already-visual, or have adequate static-SVG builds, or are terminal-appropriate). The 3 P0 labs are **directive-driven elevations** of auditor-P1 `m09a` lessons — justified by the parent request explicitly naming those systems-visual types and preferring interactive systems visuals, and mirroring the documented Batch 2 elevation. Documented transparently so the scope is defensible; if the reviewer wants 0 labs (prose-only) or a different `m09a` day, this is the adjustable knob.
2. **Concurrent edits.** `sessions/` is edited by other sessions; re-read each file immediately before editing (the tooling enforces "file modified since read").
3. **3 `.md`-artifact lessons + 8 Math-Ladder skips** will keep their `lesson_audit.py` advisories by design — listed as intentional in the report, not failures.
4. **No real-browser pixel check** (`no-real-browser-in-sandbox`). All verification is headless (jsdom + JS-syntax + hand-checked lab math); a final Light/Dim eyeball of the 3 labs would need a user screenshot if pixel fidelity matters. The SVGs reuse the `.vlab` classes proven in Batch 1/2.
5. **Bilingual on English-only files.** These lessons are currently English-only (matching the user's general preference). This batch's explicit directive is to add 2–4 light Chinese intuition touches — so the scaffold is added deliberately and kept minimal (pain/intuition only), never in formulas/code/interview phrasing.
