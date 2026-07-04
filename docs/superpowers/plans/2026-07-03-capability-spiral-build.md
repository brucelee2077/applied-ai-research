# Capability Spiral — Curriculum Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 24-module Capability Spiral ([`ROADMAP.md`](../../../ROADMAP.md)) as interactive HTML daily lessons + one hands-on experiment + one review gate per module, **reusing existing repo assets wherever they exist** rather than authoring from scratch.

**Architecture:** Reuse-first. A single `sessions/MANIFEST.md` maps every module → its daily lessons → source file → action (`reuse` / `convert` / `wrap` / `author`) → status. Every module is built with the **same 7-step pipeline**. We build in the learner's consumption order (M1 first), just-in-time — not all 24 up front. This plan specifies the build *system*, the manifest, and **M1 in full as the worked template**; M2–M24 are executed by instantiating the pipeline against the source/action table.

**Tech stack:** HTML daily lessons (`frontier-session-coach` format, self-contained, vendored JS — no CDN), Jupyter notebooks as source, COACH (`coach/state.json`), `scripts/validate_notebook.py`, `git`. System `python3` (the repo `.venv` is broken).

---

## Environment constraints (read before building)

- **Branch first.** The repo is on `main`. All build work happens on a feature branch (e.g. `build/capability-spiral`). Commit per task. **Do not push until the user asks.**
- **No accelerators here.** This sandbox cannot run GPU/TPU/CUDA. Lessons and code for M13–M14, M18, M19, M20 are *authored*; their hands-on runs are marked **"verify on cloud"** — the learner executes them on real hardware.
- **CDN is blocked (403).** Any JS/CSS library must be vendored locally (`npm pack` from the internal registry), never `<script src="https://cdn…">`.
- **Use system `python3`** (3.11), not `.venv`. `pytest.ini` makes tests cwd-independent.
- **COACH module IDs must be exact** or XP tracking silently breaks. This plan assigns them explicitly.

---

## File structure

**Physical layout follows the `frontier-session-coach` skill's native scheme** (`sessions/week-<NN>/day-<NN>-<slug>.html`) — the skill cannot be redirected to a custom folder, so we do **not** invent `sessions/modules/`. The **module is a virtual layer**: `MANIFEST.md` maps each module → its `(week folder, day lessons)` and defines the learner order; the regenerated index (`sessions/wire_index.py`) presents modules M1→M24 regardless of folder number.

```
docs/superpowers/plans/2026-07-03-capability-spiral-build.md   ← this plan
sessions/MANIFEST.md                                            ← source of truth: module → (week,day) → source → action → status + learner order
sessions/week-<NN>/day-<NN>-<slug>.html                         ← every daily lesson (skill-native path); new blocks get newly-allocated week ids
sessions/week-<NN>/experiment.ipynb                             ← the module's hands-on artifact (a notebook → COACH-tracked)
sessions/week-<NN>/review.html                                  ← the module's quiz/gate
sessions/progress.json                                          ← HTML-lesson progress ledger (w<NN>-d<NN> quest ids + localStorage)
coach/core.py, coach/state.json                                 ← COACH module ids for the EXPERIMENT NOTEBOOKS only (see Task 0.3)
```

**Two tracking models, do not conflate them:**
- **HTML lessons** track via `sessions/progress.json` + `w<NN>-d<NN>` quest ids + `data-quest-id` + browser `localStorage`. They do **not** touch `coach/state.json` (those COACH functions are Jupyter-only).
- **Experiment notebooks** track via COACH (`render_session_start/end`), keyed by a module id that **must be registered in `coach/core.py`** first (Task 0.3).

- **New content** (M1–M7 foundations, M17 RLHF, M19 generative, design-checkpoint wrappers): authored with `frontier-session-coach` into newly-allocated `sessions/week-<NN>/` folders; the manifest maps them to their module and places them in learner order.
- **Frontier-spine content** (M8–M14, M16-frontier, M18, M20, M22, M24): already exists as ~120 HTML lessons in `sessions/week-01…24/`. Reused **in place**; Phase 3 applies the audit edits and the manifest re-labels the week as its module. (Two-week blocks live under a single folder — `week-17` covers 17–18, `week-21` covers 21–22; there is no `week-18/` or `week-22/`.)
- **Design checkpoints** (M9, M12, M15, M16-design, M21, M23): short HTML lessons that summarize each case study and link into the full `ML Design/` and `genAI design/` files.

---

## The per-module build pipeline (the repeatable recipe)

Every module M*n* is built by running these seven steps. This recipe IS the plan for M2–M24.

1. **Scope** — read the module's `ROADMAP.md` entry + its source files; confirm the "builds_on" modules are already built.
2. **Break down** — write the module's day-by-day lesson list into `MANIFEST.md` (~5–15 daily steps, one atomic idea each).
3. **Lessons** — produce each daily HTML lesson (skill-native path `sessions/week-<NN>/day-<NN>-<slug>.html`):
   - `author` (foundations/gaps) → `frontier-session-coach` from the source notebook/MD, into a newly-allocated `week-<NN>` folder.
   - `reuse` (frontier spine) → edit the existing `sessions/week-*` HTML **in place**, applying the ROADMAP audit edits (do not relocate).
   - `wrap` (design checkpoints) → short HTML lesson summarizing the case study + linking to its `ML Design`/`genAI design` files.
   - Section 5 of each lesson = **scroll-reveal visual build-up** (per house style), not a click-through code stepper; no CDN `<script src>`.
4. **Experiment** — build the module's one hands-on artifact via `frontier-experiment-lab` (code + run command + expected result). Mark accelerator-only runs "verify on cloud."
5. **Review gate** — build `review.html` via `frontier-review-quiz`: 3–5 questions + the "can I explain X?" gate that must pass before advancing.
6. **Wire** — add each lesson's `w<NN>-d<NN>` quest id to `sessions/progress.json`; run `sessions/wire_index.py` to regenerate the index so the module appears in **learner order** (M1→M24) regardless of folder number; register the experiment notebook's COACH id (Task 0.3). HTML lessons themselves do NOT write to `coach/state.json`.
7. **Validate + commit** — `python3 scripts/validate_notebook.py` on any notebook (exit 0), open each HTML to eyeball it, then commit (`feat(mNN): …`).

**Hard rule carried from ROADMAP:** do not build M*n* until its `builds_on` modules exist and their review gates pass.

---

## Phase 0 — Manifest & scaffolding

### Task 0.1: Create the build branch
- [ ] **Step 1:** `git checkout -b build/capability-spiral`
- [ ] **Step 2:** confirm clean status: `git status`

### Task 0.2: Create `sessions/MANIFEST.md`
**Files:** Create `sessions/MANIFEST.md`
- [ ] **Step 1:** Write the manifest skeleton — a table with columns: `Module | Title | Capability | Lessons (planned) | Source | Action | COACH id | Status`, one row per module M1–M24, seeded from the ROADMAP table below.
- [ ] **Step 2:** Add a per-module sub-section placeholder where the day-by-day lesson list will go (filled in Step 2 of each module's pipeline).
- [ ] **Step 3:** Commit: `docs(manifest): seed 24-module build manifest`

### Task 0.3: Register COACH module IDs for the experiment notebooks
**Files:** Modify `coach/core.py`, `coach/state.json`
> COACH ids are needed only for the per-module `experiment.ipynb` notebooks (HTML lessons use `progress.json`, not COACH). Ids are **hardcoded in `coach/core.py`** (`ALL_MODULES` list + `ALWAYS_UNLOCKED` set, seeded by `_default_state()`), and `render_session_end` does a bare `state["modules"][module_id]` lookup that **KeyErrors on an unregistered id** — so registration is mandatory before any experiment notebook runs.
- [ ] **Step 1:** Add the 24 ids (`m01-numbers-arrays` … `m24-artifact-outreach`) to `ALL_MODULES` (and `ALWAYS_UNLOCKED` as appropriate) in `coach/core.py`.
- [ ] **Step 2:** Add matching dict entries (full `_default_state` shape) to `coach/state.json`, or re-seed it.
- [ ] **Step 3:** Smoke-test: run a throwaway notebook cell calling `render_session_start(module_id="m01-numbers-arrays", ...)` then `render_session_end(...)` → no KeyError.
- [ ] **Step 4:** Commit: `chore(coach): register 24 capability-spiral module ids`

---

## Phase 1 — Module 1 built in full (the worked template)

M1 = "Numbers, Arrays, and the Shape of Data." This is the pattern every later module copies.

### Task 1.1: Break down M1 in the manifest
**Files:** Modify `sessions/MANIFEST.md`
- [ ] **Step 1:** Allocate M1 a folder id in the manifest (`week-f1` — a new foundations folder; confirm `frontier-session-coach` accepts it, else use an unused numeric id and let the index order it first). Write M1's 5-lesson list: `day-01-arrays`, `day-02-indexing-slicing`, `day-03-broadcasting-dtypes`, `day-04-matmul-and-shapes`, `day-05-logs-and-exponents`. Source = Python/NumPy warm-up (new) + light-math primer. Action = `author`. Learner order = first.
- [ ] **Step 2:** Commit: `docs(manifest): break down M1 into 5 lessons`

### Task 1.2–1.6: Author the 5 daily HTML lessons
**Files:** Create `sessions/week-f1/d01-arrays.html` … `d05-logs-and-exponents.html`
For each lesson (repeat the sub-steps):
- [ ] **Step A:** Invoke `frontier-session-coach` with module_id `m01-numbers-arrays` and the lesson topic; it emits one self-contained HTML lesson (click-through sections + interactive demo + quiz + COACH start/end).
- [ ] **Step B:** Confirm section 5 is a **scroll-reveal visual build-up**; no CDN `<script src>` (vendor any JS locally).
- [ ] **Step C:** Open it to eyeball: `open sessions/week-f1/d01-arrays.html`
- [ ] **Step D:** Commit: `feat(m01): add lesson dNN <topic>`

### Task 1.7: Build the M1 experiment artifact
**Files:** Create `sessions/week-f1/experiment.ipynb`
- [ ] **Step 1:** `frontier-experiment-lab` — a NumPy notebook that loads a small dataset, reshapes/broadcasts, does a matmul by hand and with NumPy, prints every shape.
- [ ] **Step 2:** Validate: `python3 scripts/validate_notebook.py sessions/week-f1/experiment.ipynb m01-numbers-arrays` → expect exit 0.
- [ ] **Step 3:** Commit: `feat(m01): add hands-on experiment notebook`

### Task 1.8: Build the M1 review gate
**Files:** Create `sessions/week-f1/review.html`
- [ ] **Step 1:** `frontier-review-quiz` — 3–5 questions + the gate: "Can I read a shape/broadcast/matmul statement and predict the output shape?"
- [ ] **Step 2:** Open to verify. Commit: `feat(m01): add review gate`

### Task 1.9: Wire M1 into the site + COACH
**Files:** Modify `sessions/index.html` (or the index builder), `coach/state.json`
- [ ] **Step 1:** Add M1 + its 5 lessons to the sessions index; set next-link to M2.
- [ ] **Step 2:** Confirm COACH `m01-numbers-arrays` records XP when the lesson start/end cells run.
- [ ] **Step 3:** Commit: `feat(m01): wire module into index + coach`

### Task 1.10: Manifest status + checkpoint
- [ ] **Step 1:** Mark M1 rows `✅ Done` in `sessions/MANIFEST.md`.
- [ ] **Step 2:** Commit: `docs(manifest): M1 complete`.
- [ ] **Step 3:** **Checkpoint with the user** — review the finished M1 before mass-producing the rest.

---

## Phases 2–6 — remaining modules via the pipeline

Each module below is executed by running the 7-step pipeline (Phase 1 is the template). Build order = module order. `builds_on` is enforced from the ROADMAP.

### Phase 2 — Foundations M2–M7 (action: `author`/`convert`)

| M | Title | Source | COACH id |
|---|---|---|---|
| 2 | One Neuron, Then a Layer | `00-neural-networks/fundamentals` 01–05 | `m02-neuron-layer` |
| 3 | Loss, Gradients & Backprop | `00-neural-networks/fundamentals` 06–08 | `m03-loss-backprop` |
| 4 | Attention as a Pure Idea | `01-transformers/days` 01–17 | `m04-attention-idea` |
| 5 | First Model: MLP → PyTorch | `00-neural-networks/fundamentals` 09–10 | `m05-mlp-pytorch` |
| 6 | Full Transformer Block + ViT | `01-transformers/days` 18–28 + frontier ViT | `m06-transformer-vit` |
| 7 | CNNs & Vision Encoders ⭐ | `00-neural-networks/cnn` 01–06 + `05-multimodal/vision-language` | `m07-cnn-vision` |

### Phase 3 — Frontier spine (action: `reuse` + audit edits)

Each maps to existing `sessions/week-*` HTML; copy into the module folder, renumber, apply the ROADMAP audit edit.

| M | Title | Reuse source | Key audit edit |
|---|---|---|---|
| 8 | Thinking in JAX | `week-01/` (day-01…day-05) | Drop the ViT capstone (moved to M6); first JAX build stays an MLP |
| 9 | Transformer Math + Search 🎨 | `week-03/` day-01–03 | Add the search design checkpoint (Phase 5) |
| 10 | Hardware Physics (sim → 1 real) | `week-02/` + `week-04/` | Keep simulated; isolate one capped real run; add cost cap |
| 11 | Scaling Laws + Simulator | `week-05/` | — |
| 12 | Inference + Language-Assist 🎨 | `week-06/` | Add BPE lesson; add smart-compose/translate checkpoint |
| 13 | Addition Transformer — Build | `week-09/`, `week-10/` | Config runner; persistent paid TPU; billing cap; fold in Week-8 profiling |
| 14 | Your Own IsoFlops Law | `week-11/` | Shrunk matrix; caps; durable checkpoints |
| 16(fr) | MoE (frontier half) | `week-07/` | Seed the reusable toy router |
| 18 | Custom Kernels I: Pallas | `week-12/` consolidation + `week-13/` | MoE capstone upgrade + report; kernel bar = "correct first" |
| 20 | Extreme Optimization | `week-14/`, `week-15/`, `week-16/` | C++/CUDA on-ramp BEFORE ThunderKittens; "verify on cloud" |
| 22 | AI-Driven Research (ADRS) | `week-19/`, `week-20/` | Re-anchor loop to the learner's own router/kernel |
| 24 | Artifact, TLA+, Outreach | `week-17` (17–18 block) + `week-21` (21–22 block) + `week-23`/`week-24` | Add TLA+ on-ramp |

> **Boundary-split note (M13 / M14 / M18):** the source weeks don't align 1:1 with these modules. `week-10` days 04–06 (IsoFlops experiment-design/automation) belong to **M14**, not M13. `week-11` days 04–06 (MoE architecture upgrade / MoE IsoFlops) belong to **M18**, not M14. When reusing, **split these lessons across module boundaries** — do not copy a week folder wholesale.

### Phase 4 — Gap modules (action: `convert`)

| M | Title | Source | COACH id |
|---|---|---|---|
| 17 | Post-Training: RLHF & DPO ⭐ | `10-reinforcement-learning/rlhf` (what-is-rlhf, reward-modeling, ppo, dpo) | `m17-rlhf-dpo` |
| 19 | Generative Modeling ⭐ | genAI design `07`(GANs), `08`(VQ-VAE), `09`(diffusion) | `m19-generative` |

### Phase 5 — Design checkpoints (action: `wrap`)

Short HTML lessons that summarize each case study and link to its full `ML Design`/`genAI design` files (README + `staff_interview_guide.md` + notebooks).

| M | Cluster | Case studies wrapped |
|---|---|---|
| 9 | Retrieval/search | ml-design-prep, visual-search, youtube-video-search, google-street-view-blurring |
| 12 | Language assist | gmail-smart-compose, google-translate |
| 15 | Recommendation/ranking | video-recommendation, ad-click-prediction, event-recommendation, similar-listings, personalized-news-feed |
| 16 | Graph | people-you-may-know |
| 21 | Generation + RAG | intro-and-framework, image-captioning, RAG, realistic-face, high-res, text-to-image, headshots, text-to-video |
| 23 | Assistant + safety | chatgpt-chatbot, harmful-content-detection |

### Phase 6 — Wire, track, deploy check
- [ ] Full `sessions/` index rebuilt with all 24 modules + inter-module gates.
- [ ] Verify GitHub Pages root index still redirects correctly (`brucelee2077.github.io/applied-ai-research/`); `ML Design/examples/` stays gitignored.
- [ ] `MANIFEST.md` all rows `✅ Done`.

---

## Verification strategy (content build, not code)

- **Notebooks:** `python3 scripts/validate_notebook.py <nb> <module_id>` → exit 0 (checks COACH cells, module IDs, banned savefig, stale links).
- **HTML lessons:** open each (`open <file>`); confirm no CDN dependency, COACH start/end present, section 5 is scroll-reveal.
- **Review gates:** each module's `review.html` questions are answerable from its lessons.
- **Accelerator experiments (M13, M14, M18, M19, M20):** authored + code-reviewed; runs marked **"verify on cloud"** — NOT executed in this sandbox.
- **Hardest content (M18, M19, M20, M24-TLA+):** each gets one adversarial correctness-review pass (subagent) before its manifest row flips to Done.

## Risks

- **Accelerator verification gap** — the advanced hands-on artifacts can't be run here; correctness of that code rests on review + the learner's cloud runs. State this in each affected lesson.
- **M19 is the densest new build** (VAE→VQ-VAE→GAN→DDPM→latent diffusion). If it overruns, split its applied module (M21) across two blocks.
- **Reuse edits** — the old `sessions/week-*` HTML was built for the broken ordering. Each reused lesson must get its audit edit (esp. M8 ViT removal, M10 real-hardware/cost, M20 C++/CUDA on-ramp) or a forward-dependency leaks back in.
- **Scope size** — 24 modules × (5–15 lessons + experiment + gate) is a large, multi-session build. The manifest + just-in-time order keep it tractable; do not batch-generate ahead of the learner.

## Commit & handoff

- Feature branch `build/capability-spiral`; one commit per task; no push until the user asks.
- After M1 (Task 1.10), checkpoint with the user before mass-producing M2–M24.
