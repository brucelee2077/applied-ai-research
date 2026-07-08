# v8 Exemplar Distillation Report

**Loop:** `v8_exemplar_distillation`
**Date:** 2026-07-08
**Owner skills:** `frontier-curriculum-architect`, `frontier-lesson-builder`, `frontier-refactor-qa`, `frontier-visual-evidence-builder`
**Scope:** Diagnose why hand-refactoring `lesson.html` still feels less smooth than authoring the notebook, and distill transferable authoring rules from the exemplar notebook.
**Mode:** Report + design (companion: `v8_source_first_authoring_plan.md`). **No lesson files edited.** m03 not touched.

**Exemplar (style / reader-flow reference ONLY — not content to copy):**
`00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb` (13 cells).

**Calibration set (pilot):** `sessions/m02-the-neuron/` (9 lessons, v7.6 LX-stabilized).

> **Guardrail respected throughout:** the notebook is a *reader-flow yardstick*, not a source of truth for content, and **future modules must not depend on having a notebook.** The rules below are distilled so they hold in all three authoring modes (Exemplar / Reference / First-Principles) — see the plan.

---

## TL;DR

- The notebook feels smooth for one structural reason: **its authoring surface equals its reader surface.** You write markdown + code cells top-to-bottom, in the exact order the reader experiences them, and the runtime (Jupyter) owns rendering.
- `lesson.html` feels rough because **the author is also the compiler.** Every paragraph is hand-wrapped in nested styled `<div>`s; the interactive content (`DEMOS`, `BUILD`, `QS`) lives as JS arrays ~200–350 lines away from the prose it serves; the reader's journey is 7 *hidden* sections reassembled at runtime by JS; and ~15 frozen invariants must survive every edit. **Reader order ≠ source order, and authoring is conflated with rendering, interactivity, and state.**
- The v7.5/v7.6 loops proved this indirectly: they kept re-finding the *same* structural inversions (picture-after-vocabulary, formula-first, homework-register artifact) and fixing them by **HTML micro-surgery**, not authoring. A successful fix that still "feels less smooth" is the tell that the medium — not the writer — is the bottleneck.
- Fix (detailed in the plan): introduce a **source layer** (`source.md`, authored in reader-flow order like a notebook) and a **compiler** that owns the shell, the section mapping, the JS arrays, and every invariant. The notebook only donates the **Reader Flow Blueprint**; `source.md` is authored fresh from whatever the mode provides, so **no notebook is required** for future modules.

---

## 1. The two media, side by side (grounded, not abstract)

Both artifacts were read in full for this report. The contrast is structural.

| Axis | Notebook `01_what_is_a_neural_network.ipynb` | `m02/day-01/lesson.html` (801 lines) |
|---|---|---|
| **Authoring surface** | Markdown + code cells | Raw HTML: `<section class="module-section"><div class="sec-head">…<div class="sec-body"><div class="callout c-info"><span class="ic">🧭</span><div>…` per paragraph, inline styles on every table cell |
| **Reader order vs source order** | Identical (linear scroll = cell order) | **Divergent.** Hero @331, s1 @339, but `DEMOS` (s3 content) @608, `BUILD` (s5) @635, `QS` (s6) @681 — the journey is scattered and re-assembled by JS |
| **Reader navigation** | Scroll | 7 **hidden** `.module-section`s revealed by sidebar `data-target` clicks + per-section `.gotit` gates |
| **Claim ↔ evidence** | Markdown claim → code cell → **plotted output directly below** (adjacent, always in sync) | Prose claim → **static SVG with baked numbers** (can drift) or `<iframe src="viz/*.html">` **in a separate file** |
| **State / interactivity** | None to maintain | `data-quest-id`, `frontier-lesson:<qid>`, playground ≥3 gate, quiz `q:4 o:16`, `.fin` banner, prev/next hrefs, theme key |
| **Rendering owner** | Jupyter | **The author** (hand-writes the shell every time) |
| **To move a paragraph earlier** | Drag a cell | Cut nested HTML from `<section id="s2">`, splice into `<section id="s1">`, preserve wrappers + `data-sec` + `.gotit` |

The notebook is not "better writing." It is a **medium where authoring is decoupled from rendering.** That single property is what "smooth" means here.

---

## 2. Diagnosis — six structural reasons `lesson.html` refactoring feels rough

### D1 — The author is the compiler (authoring conflated with rendering)
In the notebook, the author writes content; Jupyter renders. In `lesson.html`, the author hand-emits the presentation layer: every paragraph is wrapped in `sec-body / callout / ic / term` markup, and s2's mapping table carries inline `style="padding:.5rem .6rem;color:var(--ink)"` on **every cell**. Changing one sentence means navigating scaffolding, not prose. Cognitive load that the notebook spends on *ideas* is spent here on *markup*.

### D2 — Reader order ≠ source order
The reader's experience is: hero → s1 → … → s7. But what s3 shows lives in `DEMOS` (line 608), what s5 shows lives in `BUILD` (line 635), what s6 shows lives in `QS` (line 681). **You cannot read the reader's journey by reading the file top-to-bottom.** The narrative is only assembled at runtime by JS. Authoring flow in a medium whose source doesn't preserve flow is inherently rough.

### D3 — The frozen-invariant tax dominates attention
Every edit must preserve ~15 invariants: `data-quest-id` (frozen `wf2-*/wf3-*`), 7 `.module-section`s, `data-target` links, `DEMOS/BUILD/QS` shapes, playground ≥3 gate, quiz `q:4 o:16`, `frontier-lesson:<qid>` + `frontier-theme` keys, `.fin`, prev/next hrefs, and every anchor number. During a reorder, most attention goes to *not breaking* these — not to reader flow. The notebook has **zero** such tax.

### D4 — Claim and evidence are decoupled across files
In the notebook, a claim's evidence is the **plotted output of the next cell** — adjacent and guaranteed in sync. In `lesson.html`, the evidence is a static SVG with baked-in numbers (can silently drift from the prose) or an `<iframe>` pointing at a *separate* `viz/*.html`. This is the structural cause of the recurring behavioral-viz gaps (`BL-VE-D1`, `BL-VE-D2` in the manifest) and the anchor-consistency P1s: the medium makes it **hard** to keep a moving quantity next to its claim and **easy** to ship prose-only.

### D5 — Reordering is destructive surgery, not composition
The v7.6 "seed stabilization" fix — move the mental picture before the vocabulary — was literally: cut HTML out of `<section id="s2">`, splice it into `<section id="s1">`, keep the `data-sec` and `.gotit` intact. In a notebook you drag a cell. The v7.6 report reads as careful splice-work precisely because the operation is HTML micro-surgery. **A correct fix that still feels rough is the signature of a medium problem, not a writing problem.**

### D6 — The fixed 7-section container fights the natural reader arc
The container's sections are **What / Intuition / Playground / Mechanism / Build / Quiz / Produce**, and section 1 is literally titled *"What is it."* That title pulls the author toward a **definition-first** opening — which is exactly the `LX-D1-PICTURE-ORDER` / `LX-D2-FORMULA-FIRST` inversion the loops kept finding. The notebook's arc (hook → jargon buster → human situation → analogy → picture → mechanism → discovery) has no such gravity well. **The recurring LX findings are the container leaking into the content.**

**Synthesis.** D1–D6 are all one thing: **there is no source layer.** `lesson.html` *is* the source and the compiled artifact at once, so authoring reader-flow means editing a compiled artifact by hand while fighting the compiler's concerns. The notebook is smooth because Jupyter is the compiler. v8 gives the HTML medium the same separation — **without** making a notebook a prerequisite.

---

## 3. Distilled authoring rules (from the exemplar notebook)

Each rule is stated, grounded in a specific place in the notebook, mapped to how it lands in a compiled lesson, and — critically — written so it holds **with or without a notebook**.

### 3.1 Reader Flow Blueprint
**Rule.** Every lesson follows one canonical top-to-bottom reader sequence. The author writes in this order; the compiler maps it onto the shell.

```
1  Curiosity Hook        — a mystery/surprising fact, no definition yet
2  Jargon Buster         — plain-English gloss of every term used today
3  Human Situation       — something the reader has lived
4  Narrative-Spine Analogy — the one analogy that runs the whole lesson
5  Mental-Picture Visual — a picture BEFORE the first formula
6  Concrete Worked Example — small real numbers, every step shown
7  Mechanism             — built up piece by piece (formula + Math Ladder)
8  Frontier / Real-World Payoff — why a frontier lab cares (AFTER mechanism)
9  Failure Modes / Misconceptions — what breaks, what people get wrong
10 Artifact-as-Discovery — predict → run → observe something surprising
11 Checkpoint            — 2–3 low-pressure questions
12 Takeaways / What's Next
```

**Grounded in the notebook:** cell 0 (title + "Prerequisites: just curiosity!") → cell 0 (**Jargon Buster** table) → cell 1 ("Let's start with something familiar: your brain!") → cell 2 (Team-of-Specialists analogy) → cells 3–5 (network diagram) → cell 7 (components, each with a mini-analogy) → cell 6 (real-world apps) → cell 10 (misconceptions) → cells 8–9 (learning / worked example) → cell 11 (takeaways) → cell 12 (checkpoint w/ reveal answers) → cell 13 (what's next).

**Lands as:** hero = 1–2; s1 = 2–5; s2 = 4–5; s3/s5 = 6 + hands-on; s4 = 7–8 + staff; s6 = 11; s7 = 10; `.fin` = 12.

### 3.2 Jargon Ladder Rule
**Rule.** Front-load a plain-English glossary of **every** term the lesson will use (the notebook's "Jargon Buster"). Thereafter introduce each term **exactly once, defined-before-use**, at the moment it is needed. No term may appear before its plain-English gloss.

**Grounded in:** the notebook's Jargon Buster defines Neuron, Weight, Bias, Layer, Activation Function, Training, etc. *before* cell 1 uses any of them ("A number that says how important a connection is — like how much you trust a friend's movie recommendation").

**Why it transfers:** this is the structural cousin of the non-negotiable *"no formula before felt intuition."* The Jargon Ladder makes "define every word before you use it" (CLAUDE.md §6) **mechanically checkable**: the compiler can assert that every `.term` tooltip's first use is preceded by its gloss. In the current `lesson.html`, `.term` tooltips exist but glosses are scattered inline — a front-loaded ladder removes the ordering risk entirely.

### 3.3 Narrative Spine Rule
**Rule.** Choose one analogy at the top and run it through **every** section — intro, mechanism, failure mode, and artifact. An analogy that appears once and is abandoned is decorative and is a defect.

**Grounded in:** the notebook's "Team of Specialists / Movie Recommendation Company" recurs across components (voter, bouncer), learning ("learning to bake"), and the worked example. Contrast the v7.5 finding `LX-D2-NARRATIVE-SPINE`: m02 D2's valve/dimmer analogy appeared in the build viz but **not** in the staff-lens failure — a thin spine, later fixed in v7.6 ("one-way valve jammed shut").

**Transfers without a notebook:** in Reference/First-Principles mode the author *chooses* the spine up front and the Reader Flow Gate checks it appears in ≥3 blocks including staff.

### 3.4 Curiosity-before-Rigor Rule
**Rule.** The reader's first contact is wonder, never a definition or a prerequisites checklist. Prerequisites are framed as "just curiosity." The hook opens a "why do I need this?" tension the lesson then resolves.

**Grounded in:** "Your brain has about **86 billion neurons**…"; "Prerequisites: Just curiosity!" The m02 v7.6 fix mirrors this — the hero now opens on *"GPT-3 has a weight matrix with 49,152 columns… every column is the same tiny thing"* (line 334), and the goal box was demoted from a capability checklist to a quiet promise. That fix **is** this rule; v8 makes it the default, not a retrofit.

### 3.5 Artifact-as-Discovery Rule
**Rule.** Hands-on work is framed "predict, then run and see," prints shapes/intermediate values, and visualizes at least once. It is a discovery moment, never a homework submission with "acceptance criteria."

**Grounded in:** every notebook code cell prints shapes and annotates ("📊 What you're seeing:"). The m02 v7.6 fix converted Produce from "Acceptance criteria" to *"Predict the output, then run it — watch ReLU decide"* (line 467, 485). `lesson_audit.py` was broadened to accept the discovery register so it doesn't false-fire on future modules — that tooling change is a v8 precondition, already in place.

### 3.6 Staff-Depth-after-Orientation Rule
**Rule.** Depth (failure modes, trade-offs, scale, interview framing) appears **only after** the learner is oriented by hook → picture → mechanism, and is framed as **empowerment**, not pressure. Frontier/scale relevance follows orientation. Every staff claim must trace to something taught earlier (the S4 grounding rule).

**Grounded in:** the notebook defers "Common Misconceptions" to cell 10 and closes with "what you just unlocked." In m02, the staff lens (silent-failure + trade-off + grounded interview line) lives in s4 *after* the mechanism, and the v7.5 gate marks "staff/frontier relevance appears before orientation" as a **P0**. This rule is the positive statement of that P0.

---

## 4. Why these six rules are notebook-independent

The notebook is where the rules were *observed*, but none of them **requires** a notebook to *apply*:

| Rule | What it needs | Notebook required? |
|---|---|---|
| Reader Flow Blueprint | a fixed sequence to author in | No — it's a template |
| Jargon Ladder | a term list + glosses | No — author writes them |
| Narrative Spine | one chosen analogy | No — author chooses it |
| Curiosity-before-Rigor | a hook | No — a paper's surprising result or a real bug works |
| Artifact-as-Discovery | a predict→run moment | No — any runnable stub works |
| Staff-Depth-after-Orientation | a late, grounded depth block | No — a paper *supplies* the depth |

This is the hinge of the whole v8 design: **the exemplar calibrates the flow; the flow then applies to modules sourced from papers/docs (Reference Mode) or from nothing but the Learning Barrier Lens (First-Principles Mode).** The plan operationalizes all three modes on the same `source.md` format and the same gates.

---

## 5. What the pilot (m02) already validates

- The v7.6 seed stabilization already re-derived these rules by hand and applied them to all 9 lessons — so m02 is a **known-good target**: a v8 compiler must be able to reproduce the *current* shipped `lesson.html` behavior from a `source.md`, and the compiled reader flow must match what v7.6 hand-authored.
- The recurring loops (v7 → v7.6) that kept re-finding the same LX inversions are the empirical evidence that **the medium re-introduces the inversion each time a lesson is built from the container template.** A source layer breaks that cycle at the root.
- `lesson_audit.py`, `nav_audit.py`, `staff_lens_audit.js` already exist and pass on m02 — they become the v8 QA backstop under and behind the new gates.

---

## 6. Handoff to the plan

The companion `v8_source_first_authoring_plan.md` defines: the canonical `source.md` format, the file layout, compiler responsibilities, the three authoring modes, and the gates — Reader Flow, Notebook Smoothness (pilot only), No-Notebook Authoring (future modules), Staff Depth, Shell Invariant — plus the exact v8 rollout loop and the JAX / Scaling-Laws no-notebook authoring path.

**Status:** v8 is **proposed**. No lesson files were edited; module statuses are unchanged; m03 is untouched.
