# Module 2 — Intuition Rescue Plan

**Rollout type:** Intuition Rescue (foundation tier)
**Module:** `sessions/m02-the-neuron/` (9 daily lessons)
**Skill protocol:** Scope → Audit → **Plan (this file)** → Edit → QA → Report
**Effort:** xhigh · **Rule:** fix **P0 only**; record P1/P2 for later.
**Date:** 2026-07-08

---

## 1. Scope

Upgrade all 9 `lesson.html` files in `sessions/m02-the-neuron/` from *structurally-correct-but-math-first*
into *warm, coach-led, intuition-first* lessons that follow the arc:

> **Pain → Intuition → Analogy → Tiny Example → Formula → Mechanism → Visual → Artifact**

This is the **Coach Layer** retrofit (already applied to m01, m05a, m06, m07, m08, m09a, m09c, m10a, m13,
m16a, m17a, m17b). m02 received the earlier **staff-lens** retrofit but **not** the Coach Layer — so the
work here is well-defined and repeatable, not a rewrite.

**Out of scope (deliberately preserved, not fixed):**
- The m02/m03 label quirk: days 4–9 are titled "Module 3"/"Module 2-3" with quest-ids `wf3-*`. Changing
  titles/quest-ids would break localStorage keys and the nav chain. **Leave every title, quest-id, and
  brand-sub exactly as-is.**
- Shared files (`viz/*.html`, `index.html`, audit scripts). No shared-file edits during this module rollout.

## 2. Lessons in scope

| # | Folder | quest-id | Topic |
|---|--------|----------|-------|
| 1 | day-01-single-neuron | wf2-d01-neuron | A single neuron |
| 2 | day-02-activations | wf2-d02-activations | Activation functions |
| 3 | day-03-layers-forward-pass | wf2-d03-forward | Layers & forward pass |
| 4 | day-04-loss | wf3-d01-loss | Loss functions |
| 5 | day-05-gradients-backprop | wf3-d02-backprop | Gradients & backprop |
| 6 | day-06-training-loop | wf3-d03-training-loop | The training loop |
| 7 | day-07-optimizers | wf3-d04-optimizers | SGD → Momentum → Adam |
| 8 | day-08-learning-rate | wf3-d05-lr | Learning-rate intuition |
| 9 | day-09-train-val-test | wf3-d06-split | Train/val/test & overfitting |

## 3. Current teaching problems

`lesson_audit.py m02-the-neuron` → **9 OK / 0 hard failures**, but every lesson trips the same six
COACH advisories. Confirmed by marker grep (all zeros across all 9): `🎤=0 🪜=0 😕=0 📓=0 直觉=0`.

1. **No pain-point (😕).** Lessons state facts before naming *why the idea is confusing*. Intuition is
   never primed before the formula. This is the core "too math-first" complaint.
2. **No Math Ladder (🪜).** Formulas appear without the explicit `words → labeled formula → tiny numbers
   → sanity check` scaffold. The tiny-number step exists informally in some lessons but isn't laddered.
3. **No interview-ready explanation (🎤).** Staff-lens callouts exist, but there's no crisp "say this in
   an interview" line — the artifact that makes the depth portable.
4. **No explicit Acceptance criteria** in the Produce step. Learners can't self-verify the artifact.
5. **No 5-minute research log (📓).** Some lessons have a light `📝 write one line in log.md` note; it's
   not the bilingual 3-line reflective log the Coach Layer standard uses.
6. **No bilingual scaffold.** Zero Chinese intuition touches (target: 2–4 light touches per lesson).

**Not a problem (verified — do NOT "fix"):**
- **Hooks are warm, not formula-first.** Every hero lede opens with motivation/mystery ("is it any
  good?", "which way should each weight move?"). Days 07/08 open with a one-line formula the learner
  *already knows*, framed as "here's the twist" — acceptable.
- **Analogies are solid, not shallow.** judge scoring a contestant · valve+dimmer · assembly line ·
  golf score · hiker in the fog · free throws · ball downhill · stepping stones · practice vs exam.
  Each has 2 `.relate` cards, a one-line summary, and most have a "where the analogy breaks" callout.
  → We **deepen with a Chinese intuition touch**, we do **not** replace them.

## 4. Visual audit summary

- **All 9** have a fake-terminal Playground (§3) + an SVG **scroll-reveal build** (§5, `BUILD` array).
  The static SVG builds are genuinely strong (per-step diagrams), so §5 is not visually empty.
- **Days 07 & 08** additionally embed the interactive `viz/gradient-descent.html` (build-embed iframe).
  Confirmed working; the viz posts `viz-height` (no autoresize bug). The `.build-embed` CSS ships in
  **every** lesson's shell, so adding an embed elsewhere needs **no CSS change**.
- **Days 01–06, 09** have terminal-only Playgrounds (no interactive viz).

## 5. P0 / P1 / P2 visual upgrades

**Visual decision rule** (P0 requires all four): hard concept · terminal-only/static now · a visual gives
a clearer mental model · a specific learning question. Foundation module → lean toward visuals, but
**reuse existing proven viz over authoring new ones** (new HTML/JS = higher risk, out of an intuition rescue).

| Lesson | Decision | Rationale |
|--------|----------|-----------|
| **day-05 gradients-backprop** | **P0 — embed `viz/gradient-descent.html` in §5** | Module's hardest, most inherently-visual concept (slope + descent); playground is terminal-only; the exact proven viz already exists and is embedded by days 07/08 → zero new-viz risk. Learning question: "which direction reduces loss, and how far does one step move?" |
| day-02 activations | **P1** | An interactive ReLU/sigmoid-curve + "linear layers collapse" plot would help, but a *new* viz; strong static SVG build already present. Record, don't build now. |
| day-06 training-loop | **P1** | Live loss-curve-descending viz would help; new viz; static build adequate. |
| days 01, 03, 04, 09 | **P2** | Static SVG builds already carry the mental model; no P0 gap. |
| days 07, 08 | none | Already have the interactive embed. |

**Net P0 visual work: 1 embed (day-05), reusing an existing file.**

## 6. Lessons that should stay terminal/code-oriented

All Playgrounds (§3) stay as fake-terminals — they walk tiny numbers through the steps, which *is* the
"Tiny Example" rung and is the right medium for showing code/values. We are not converting §3 terminals
into visuals; the §5 build is where visuals live. day-05 gets an interactive §5 embed on top of its build.

## 7. Expected content edits (per lesson — the Coach Layer 6-pack)

Applied to **all 9**, using the uniform anchors verified in the audit. Each element matches the
curriculum-standard pattern (reference: `m01/day-04-matmul-and-shapes/lesson.html`).

1. **Pain-point (😕)** — insert in **§1**, immediately after the `🏭 "A real one"` callout, before the
   `.gotit`. Form: `<div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 ·
   Why this trips people up:</b> …</div></div>`. Names the specific misconception for that day.
2. **直觉 / Intuition touch (Chinese)** — insert in **§2**, after the `<strong>In one line:</strong>`
   summary, before the `.gotit`. Form: `<p><b>直觉 / Intuition:</b> [Chinese] — [English bridge]. [1
   technical sentence].</p>`. Deepens the existing analogy; adds the bilingual scaffold.
3. **Math Ladder (🪜)** — insert in **§4**, right after the formula/worked-small block. 4 rungs:
   `1 · In words · 2 · The formula (every symbol labeled) · 3 · Tiny numbers · 4 · Sanity check`.
4. **Interview answer (🎤)** — insert in **§4**, after the existing staff-lens trade-off callout, before
   the `.gotit`. Form: `<div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an
   interview:</b> "…"</div></div>`.
5. **Acceptance criteria** — insert in **§7**, after the Option-B prompt block. Form: `<h4>Acceptance
   criteria</h4><ul>…</ul>` — 2–3 checkable, artifact-specific bullets.
6. **5-minute research log (📓)** — in **§7**, **convert** the existing light `📝` log.md note into the
   standard bilingual `📓` 3-line log (avoids a duplicate callout — see risk R3).

Chinese budget: pain-point + 直觉 + research-log ≈ **3 touches/lesson** (within the 2–4 target), key
technical terms and all interview phrasing stay in **English**.

## 8. Expected artifact / Produce fixes (P0)

**Stale run command / Produce path mismatch** — the create-path is correct but the run command is a
stale flat filename. Fix the "Run with" command to match the created artifact path, in **7 lessons**:

| Lesson | Stale (now) | Fix to |
|--------|-------------|--------|
| day-02 | `python3 m02b_activations.py` | `python3 sessions/m02-the-neuron/day-02-activations/experiment.py` |
| day-03 | `python3 m02c_forward.py` | `…/day-03-layers-forward-pass/experiment.py` |
| day-05 | `python3 m03b_gradients.py` | `…/day-05-gradients-backprop/experiment.py` |
| day-06 | `python3 m03c_training_loop.py` | `…/day-06-training-loop/experiment.py` |
| day-07 | `python3 m03d_optimizers.py` | `…/day-07-optimizers/experiment.py` |
| day-08 | `python3 m03e_learning_rate.py` | `…/day-08-learning-rate/experiment.py` |
| day-09 | `python3 m03f_train_val_test.py` | `…/day-09-train-val-test/experiment.py` |

Days 01 & 04 already correct — no change. (The `experiment.py` stubs themselves already print the
correct path, confirming the flat names are the stale side.)

## 9. Protected files — do NOT touch

- Navigation (`.side-nav` prev/next), quest-ids, `localStorage` keys, brand-subs, titles.
- Sidebar shell, Appearance switcher, quiz JS, BUILD/scroll-reveal JS, DEMOS playground JS, `.gotit`
  completion logic, tooltip machinery, copy-button JS, viz-height receiver script.
- `sessions/viz/*` (including `gradient-descent.html`), `sessions/index.html`, audit scripts.
- Section count must remain **7** (`data-sec`), `gotit=7`, `data-demo=3`, `quiz=4`, `var BUILD=` present.
  Coach Layer elements are inserted **inside existing sections** — no new sections.

## 10. Risks and stop conditions

- **R1 — duplicate callouts.** (Known defect class from Batch 2.) Insert each element **once**, at a
  unique anchor; after edits, grep each marker (`😕 🪜 🎤 📓`) must equal **1** per lesson.
- **R2 — breaking the 7-section invariant.** Never add/remove a `<section>`, `.gotit`, `data-demo`, or
  quiz object. Re-run `lesson_audit.py` after each lesson.
- **R3 — double log note.** Convert the existing `📝` log.md callout into the `📓` research log; do not
  add a second reflective note.
- **R4 — day-05 embed autoresize.** Reuse the exact `.build-embed` + `<iframe src="../../viz/
  gradient-descent.html">` + cap/reload pattern from day-07/08 verbatim (proven; height-sender present).
- **R5 — Chinese overreach.** Cap at 2–4 touches; never translate formulas or interview phrasing.
- **STOP and report** if: any hard audit failure appears, nav/JS/quiz/playground breaks, a shared file
  seems to need changing, or math would have to be altered to fit a ladder (means the lesson math is
  wrong — flag, don't paper over).

## Execution order

1. Fix content + run-command per lesson, day-01 → day-09 (one file at a time — no parallel writes).
2. day-05 also gets the P0 viz embed.
3. After each lesson: targeted `lesson_audit.py <folder>` + marker grep.
4. After all 9: full `lesson_audit.py m02-the-neuron` + jsdom/`node --check` shell verification +
   duplicate/title-spacing/stale-command sweep.
5. Write `sessions/m02_intuition_rescue_report.md`.
