# v8 Source-First Authoring Plan

**Loop:** `v8_source_first_authoring`
**Date:** 2026-07-08
**Owner skills:** `frontier-curriculum-architect` (rollout + modes), `frontier-lesson-builder` (source craft), `frontier-refactor-qa` (gates), `frontier-visual-evidence-builder` (behavioral viz)
**Companion:** `v8_exemplar_distillation_report.md` (diagnosis + the six rules)
**Status:** **PROPOSED.** No lesson files edited. No compiler code written yet (this is the design). Module statuses unchanged. m03 untouched.

> **North star:** give the HTML medium the notebook's one good property — *authoring surface = reader surface* — by adding a **source layer** and a **compiler**, so the author writes reader-flow and the compiler owns the shell/JS/invariants. **Never require a notebook for future modules.**

---

## 0. Design principles (non-negotiable)

1. **Source ≠ artifact.** The author edits `source.md` (reader-flow order). `lesson.html` is *compiled output*, never hand-edited under v8.
2. **The compiler owns the invariants.** Shell, 7-section mapping, `DEMOS/BUILD/QS`, quest-ids, gates, localStorage, nav, `.fin` are generated — not authored.
3. **Single-sourced anchors.** Every numeric anchor is declared once and referenced by name; the compiler expands it into prose, playground, quiz, and produce, so drift is impossible.
4. **Notebook-independent.** The Reader Flow Blueprint is a template, not a notebook import. Three modes (Exemplar / Reference / First-Principles) feed the *same* format and gates.
5. **Additive, reversible, opt-in.** v8 is layered over v7.6. Existing hand-authored `lesson.html` stays authoritative until a module is explicitly "sourced." Nothing in the frozen `do_not_modify` list changes.
6. **Deterministic + idempotent.** Same `source.md` → byte-identical `lesson.html`, so the Shell Invariant Gate can diff.

---

## 1. Canonical source format (`source.md`)

One `source.md` per day. YAML front-matter (contract + single-sourced anchors) + a reader-flow-ordered body of typed blocks. The block fences map to shell regions; the author never names the shell.

### 1.1 Front-matter

```yaml
---
module: m02-the-neuron
day: 1
quest_id: wf2-d01-neuron          # FROZEN — compiler asserts, never invents
title: "A Single Neuron"
subtitle: "The Smallest Piece of a Learning Machine"
kicker: "Module 2 · Train · Day 1"
mode: exemplar                    # exemplar | reference | first_principles
notebook_yardstick: 00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb  # OPTIONAL; omit for reference/first_principles
references: []                    # REQUIRED (>=1) when mode: reference (arXiv ids / doc URLs / paths)
spine: "brain cell: weigh → add → decide"   # the ONE narrative-spine analogy
nav:
  prev: { href: "../../m01-shape-of-data/review.html", label: "M1 · Review Gate" }
  next: { href: "../day-02-activations/lesson.html",   label: "Activation Functions" }
anchors:                          # single source of truth for every number in the lesson
  x: [2, 3]
  w: [0.5, -1]
  b: 1
  wsum: -2
  prebias: -1
  out: 0
coverage: [C01, C02, C03, C04]    # must-cover ids from the module coverage contract
capability_limit: "one neuron draws one line; XOR needs a hidden layer"   # S3 lens — REQUIRED for core mechanisms
geometry: "w orients the boundary; b shifts it off the origin"            # S3 lens
---
```

### 1.2 Body — typed blocks, in reader-flow order

The block **order in the file is the reader's order.** Fences are `::: <type>` … `:::`. Prose inside is plain markdown; `{{anchor:x}}` references expand at compile time; `[[term:weight]]` marks a glossary term (compiler wires the tooltip).

```markdown
::: hook            # → hero lede. Curiosity first, no definition.
GPT-3 has a weight matrix with {{const:49152}} columns. The surprise: every
column is the same tiny thing — one neuron...
:::

::: jargon          # → front-loaded glossary; compiler emits the term-tooltip table
neuron | The basic unit: weighted sum → add bias → activation.
weight | How much one input matters. Training adjusts these.
bias   | One number added after the weighted sum; shifts the result.
:::

::: picture         # → top of s1, BEFORE any vocabulary (Jargon Ladder + Curiosity rules)
Think about one brain cell. It listens to many signals, weighs them by how
much each matters, adds them up, and decides whether to fire...
:::

::: analogy spine   # → s2. The spine that must recur later.
Brain neuron ↔ artificial neuron, piece by piece: | dendrites | inputs x | ...
:::

::: mechanism math_ladder   # → s4. Formula + 4-step Math Ladder + worked example.
formula: output = activation(w·x + b)
worked: x={{anchor:x}}, w={{anchor:w}}, b={{anchor:b}} → wsum={{anchor:wsum}} → +b={{anchor:prebias}} → ReLU→{{anchor:out}}
:::

::: playground steps=3       # → s3 DEMOS. Compiler builds the JS; enforces >=3.
- id: wsum   label: "① weighted sum (w·x)"   console: "w·x = {{anchor:wsum}}"
- id: bias   label: "② + bias"                console: "+ b = {{anchor:prebias}}"
- id: act    label: "③ activation → output"  console: "ReLU → {{anchor:out}}"
:::

::: build behavioral=false   # → s5 BUILD scroll-reveal pieces (data, not JS)
- "one neuron"  | "weighted sum + bias + activation"
- "a whole layer" | "many neurons at once = one matmul x@W+b"
:::

::: payoff         # → s4 tail. Frontier relevance — AFTER mechanism (Staff-Depth-after-Orientation).
Every one of GPT-3's 49,152 columns is one neuron wired to 12,288 inputs...
:::

::: staff          # → s4 staff lens. failure + trade-off + grounded interview line.
failure: symmetric initialization — all-equal weights collapse a layer to one neuron.
tradeoff: scale of the random start (fan-in; Xavier/He named, deferred).
interview: "A neuron is activation(w·x+b)... symmetric init collapses the layer..."
:::

::: quiz q=4 o=4   # → s6 QS. Compiler enforces q:4 o:16 total.
- q: "..."  options: [...]  answer: 2  why: "..."
:::

::: produce discovery   # → s7. predict → run → observe. NOT "acceptance criteria".
predict: "will the output be positive, negative, or exactly zero?"
optionA: "write experiment.py: def neuron(x,w,b)=relu(dot(w,x)+b)..."
observe: "the pre-activation lands at {{anchor:prebias}}, ReLU squashes to {{anchor:out}}"
log: "three lines in log.md: the three steps / why w·x is a dot product / one fuzzy thing"
:::
```

**Key properties:** the file reads top-to-bottom as the reader experiences it (fixes D2); prose is plain markdown, not nested divs (fixes D1); anchors are single-sourced (fixes D4's drift + the anchor-consistency P1s); `playground`/`build`/`quiz` are **data**, so the author never writes JS (fixes D3); a `behavioral=true` flag on `mechanism`/`build` forces a live-viz reference (fixes D4's prose-only tendency, enforces S1).

---

## 2. Proposed file layout

```
sessions/
  _compiler/                       # NEW — the v8 engine (shared, single source of shell)
    compile_lesson.py              # source.md -> lesson.html (deterministic, idempotent)
    shell_template.html            # the frozen sidebar + Appearance switcher + 4 themes (dark default)
    blocks.py                      # block-type -> shell-region mapping (the 12->7 map)
    gates/
      reader_flow_gate.py
      shell_invariant_gate.py
      staff_depth_gate.py
      notebook_smoothness_gate.py  # pilot/exemplar only
      no_notebook_gate.py          # reference / first_principles
  <module>/
    day-XX-slug/
      source.md                    # NEW — canonical source (authored)
      lesson.html                  # COMPILED (never hand-edited under v8)
      experiment.py                # guided stub (unchanged)
      log.md                       # learner's log (unchanged)
    _refactor/manifest.yaml        # records mode + gate results (unchanged location)
  _refactor/rollout_tracker.yaml   # records v8 state (unchanged location)
```

**Migration policy (opt-in, per module):**
- A shipped module stays hand-authored-authoritative until it is *sourced*: extract `source.md` from the current `lesson.html`, recompile, and prove no behavioral regression via the **Shell Invariant Gate** (diff of shell + interactive regions against the shipped file).
- **v8 does not require re-sourcing any shipped module.** New modules (m03+) are authored source-first from day one; migrating m01/m02 is optional and separately scheduled.

---

## 3. Compiler responsibilities (`source.md → lesson.html`)

The compiler is the "Jupyter" of this system. It must:

1. **Emit the frozen shell** from `shell_template.html` — sidebar + Appearance switcher, 4 themes, dark default, `frontier-theme` key. Authored **zero** times by hand.
2. **Map the 12 reader-flow blocks onto the 7 `.module-section`s + hero + `.fin`** via `blocks.py`. The author writes flow; the compiler decides section placement, `data-sec`, `data-target`, sidebar labels, and `.gotit` gates.
3. **Generate `DEMOS`, `BUILD`, `QS` JS arrays** from the `playground`/`build`/`quiz` block data. Enforce **playground ≥3** and **quiz `q:4 o:16`** at compile time (a violation is a compile error, not a QA finding).
4. **Wire state + nav:** `data-quest-id` from front-matter (asserted against the frozen list, never invented), `frontier-lesson:<qid>` localStorage, `.fin` completion banner, prev/next hrefs, hub linkage.
5. **Wrap prose** in the canonical `callout` / `.term`-tooltip / Math-Ladder markup; build the front-loaded glossary from `::: jargon`; expand every `{{anchor:*}}` / `{{const:*}}` so numbers are single-sourced across prose, playground, quiz, and produce.
6. **Embed behavioral evidence:** a `mechanism`/`build` block with `behavioral=true` **must** name a `viz/*.html`; the compiler inserts the auto-resizing iframe + the `postMessage` height sender. A behavioral block with no viz is a **compile error** (S1 enforced structurally, not by audit).
7. **Be deterministic + idempotent:** same `source.md` → byte-identical `lesson.html`. This is what lets the Shell Invariant Gate diff a recompile against the shipped file.
8. **Preserve `experiment.py`/`log.md`** untouched (guided-stub contract §31).

The compiler is the mechanism that **moves the entire frozen-invariant tax (D3) off the author and onto tested code.**

---

## 4. Three authoring modes

All three feed the **same** `source.md` format and the **same** gates. Only the *input* and the *Notebook Smoothness Gate* differ.

| Mode | When | Input | Notebook Smoothness Gate | Example modules |
|---|---|---|---|---|
| **Exemplar** | A strong companion notebook exists | Distill flow from the notebook; source.md mirrors its arc; reuse its analogy/worked example as calibration | **Applies** (pilot only) | m02 (↔ `00-*/01,02,03`), m03/m04 if a `00-*` notebook matches |
| **Reference** | Papers/docs exist, **no notebook** | Read refs for correctness + depth; author source.md fresh via the six rules; refs supply mechanism/staff, Blueprint supplies flow | **Skipped** (not failed) | m07 JAX (JAX docs), m10a/m13 scaling laws (arXiv), m14 MoE, m15 kernels, m17 quantization |
| **First-Principles** | No strong ref, no notebook | Author from the Learning Barrier Lens + Blueprint alone; self-generate the worked example + analogy | **Skipped** (not failed) | synthesis/framing modules; design case studies without a canonical text |

**Mode is declared in front-matter and recorded in the manifest.** The rollout loop (§9) selects the mode automatically: notebook present → Exemplar; else references discoverable → Reference; else First-Principles.

---

## 5. Reader Flow Gate

Validates the **compiled** lesson against the Reader Flow Blueprint. Runs on **all modes**. Mechanical checks against the ordered block list:

- [ ] **Curiosity before rigor:** `::: hook` contains no definition/formula; first `::: mechanism`/formula appears only after `::: picture`.
- [ ] **Jargon Ladder:** `::: jargon` glossary present; every `[[term:*]]` first-used *after* its gloss.
- [ ] **Picture before term:** `::: picture` precedes the first technical vocabulary in the reading order.
- [ ] **Narrative spine:** the `spine` analogy appears in ≥3 blocks **including `::: staff`** (guards against decorative analogy — the `LX-*-NARRATIVE-SPINE` failure).
- [ ] **Artifact as discovery:** `::: produce` has a `predict:` and an `observe:`, and uses the discovery register (not "acceptance criteria").
- [ ] **Staff after orientation:** `::: staff` and `::: payoff` occur after `::: mechanism`; no frontier/scale claim before `::: picture`.
- [ ] **Staff grounding (S4):** every claim in `::: staff` (esp. the interview line) traces to something in an earlier block.

This is the v7.5 Learning Experience Gate made **mechanical** against typed source blocks, so it can't be judged loosely per-lesson.

---

## 6. Notebook Smoothness Gate (PILOT / Exemplar ONLY)

Fires **only** when `notebook_yardstick` is set (Exemplar mode; today: m02). Compares the compiled lesson's reader flow to the notebook on the **barrier axis only** (from the v7.5 held-out table):

| Barrier dimension | Requirement |
|---|---|
| First move | Compiled hook ≥ notebook's warmth (curiosity, not a capability bridge) |
| Picture vs term order | Picture precedes terms, as in the notebook |
| "Why it exists" framing | Plain words + picture before any identity/formula |
| Goal framing | Invitation register, not a capability checklist |

**Pass condition:** the compiled lesson reads **at least as smoothly** as the notebook on all four; the reader would not prefer the notebook for the first screen.

> **Explicitly pilot-scoped.** This gate is how we *calibrate* the system on m02. It is **never** a gating requirement for Reference/First-Principles modules — those cannot and must not be blocked for lacking a notebook. See §7.

---

## 7. No-Notebook Authoring Gate (FUTURE MODULES)

The rule that makes future modules **not depend on a notebook.** For any module where `mode: reference` or `mode: first_principles`:

- [ ] **Mode declared** in front-matter and manifest.
- [ ] **A missing notebook is not a defect.** The Notebook Smoothness Gate is **skipped**, recorded `N/A (no-notebook mode)` — never `fail`.
- [ ] **Reference integrity (Reference mode):** `references:` lists ≥1 authoritative source (arXiv id / doc URL / path); staff-depth claims trace to a reference.
- [ ] **Self-sourced example (First-Principles mode):** `::: mechanism` contains a worked numeric example authored from scratch (the thing Exemplar/Reference borrow).
- [ ] **The six rules still hold** via the Reader Flow Gate — no exemption.
- [ ] **Staff Depth Gate + Shell Invariant Gate** run unchanged.

**This is the gate the goal names.** It is what lets JAX and Scaling Laws (§9.1) ship at full reader-flow quality with no notebook in sight.

---

## 8. Staff Depth Gate & Shell Invariant Gate

### 8.1 Staff Depth Gate (all modes)
- [ ] one **named** failure mode (silent-failure preferred),
- [ ] one **trade-off** stated as a real design decision,
- [ ] one **interview line** grounded in something taught earlier (S4),
- [ ] one **diagnostic** quiz question.
Reuses `staff_lens_audit.js` as the backstop.

### 8.2 Shell Invariant Gate (all modes — the safety net that replaces D3's manual vigilance)
- [ ] Recompile is **idempotent** (compile twice → byte-identical).
- [ ] `data-quest-id` matches the frozen list; 7 `.module-section`s present; `data-target` links complete.
- [ ] `DEMOS/BUILD/QS` shapes valid; playground ≥3; quiz `q:4 o:16`.
- [ ] `frontier-lesson:<qid>` + `frontier-theme` keys, `.fin`, prev/next hrefs intact.
- [ ] **Migration diff:** for a sourced-from-existing module, diff the compiled shell + interactive regions against the last shipped `lesson.html`; assert no behavioral regression.
- [ ] Backstop: `lesson_audit.py`, `nav_audit.py`, `node --check`, jsdom-if-available all pass.

---

## 9. The exact future rollout loop (v8)

Extends the v7.6 Autonomous Rollout Loop with **mode selection**, **author-source**, and **compile** steps. New/changed steps are marked ★.

```
1.  Read sessions/_refactor/rollout_tracker.yaml → current_target
2.  Read seed manifest(s) + recent reports referenced by tracker/manifests
3.  Detect seed propagation risk; stabilize seed if required (v7.6 rule); update seed manifest/tracker
4.  Select target; create/read sessions/<target>/_refactor/manifest.yaml
5. ★ DETERMINE AUTHORING MODE:
        companion notebook exists (00-* or module)  → Exemplar
        else authoritative papers/docs available    → Reference (arXiv MCP / doc fetch)
        else                                         → First-Principles
      Record mode in the manifest.
6.  Source-free coverage discovery → coverage contract (must-cover + S3 capability-limit + geometry rows)
7. ★ AUTHOR source.md per day in reader-flow order (the six rules), anchors single-sourced,
      spine chosen up front. (Exemplar: mirror notebook arc. Reference: refs → mechanism/staff,
      Blueprint → flow. First-Principles: derive worked example + analogy.)
8. ★ COMPILE source.md → lesson.html (compiler owns shell/JS/invariants; behavioral blocks name a viz)
9.  GATES (on compiled output):
        Reader Flow · Staff Depth · Shell Invariant · Visual/Evidence(behavioral) · Artifact
        + Notebook Smoothness (Exemplar/pilot only)
        + No-Notebook Authoring (Reference / First-Principles)
10. QA: lesson_audit.py, nav_audit.py, staff_lens_audit.js, node --check, jsdom if available
11. ★ Fix P0 only IN source.md, then RECOMPILE (never hand-edit lesson.html) until 0 P0
12. Update manifest + tracker (mode, gate results, counts, report paths, rollout_history)
13. Stop at pass / pass_with_p1, or write a phase-specific blocker report
```

### 9.1 Worked example — how JAX (m07) and Scaling Laws (m10a/m13) get authored WITHOUT notebooks

**m07 `thinking-in-jax` → Reference Mode.**
- Step 5: no `00-*` notebook for JAX → **Reference Mode**. `references:` = JAX docs (`jax.readthedocs.io` transforms/pytrees/sharding pages) + relevant repo `sessions/m07-*` prior material.
- Step 7: author `source.md` per day in reader-flow order. Hook = a curiosity ("the same NumPy function runs 100× faster and you changed one line" / `jit` surprise). Spine = one analogy (e.g. "a recipe you hand to a compiler once, then it cooks fast forever"). Mechanism/staff-depth (tracing, `vmap`, `pmap`, sharding pitfalls) come **from the JAX docs**; the Blueprint supplies the *order*. Worked example = a tiny traced function with printed shapes.
- Step 9: Reader Flow + Staff Depth + Shell Invariant + Artifact gates run. **Notebook Smoothness Gate = skipped/N-A.** No-Notebook Gate checks: mode=reference, ≥1 ref cited, staff claims trace to docs.
- Result: full reader-flow quality, **no notebook required.**

**m10a/m13 Scaling Laws → Reference Mode (arXiv).**
- `references:` = Kaplan et al. 2001.08361, Hoffmann et al. ("Chinchilla") 2203.15556, fetched via the `mcp__arxiv__*` server (`get_abstract` → `download_paper`).
- Hook = the surprising empirical fact ("loss falls as a straight line on a log-log plot across 7 orders of magnitude"). Spine = "buying compute like a budget — how do you split it between a bigger model and more data?" Mechanism = the power-law form and the compute-optimal frontier, sourced from the papers; the **behavioral** iso-FLOP / power-law curves are `behavioral=true` blocks that name a live `viz/*.html` (S1). Staff-depth = the Kaplan↔Chinchilla disagreement as a real trade-off, interview line grounded in the plotted frontier.
- Gates identical to m07. **No notebook anywhere in the pipeline.**

**First-Principles fallback:** a module with neither a notebook nor a clean reference (e.g. a synthesis/framing day) is authored from the Learning Barrier Lens + Blueprint; the author *invents* the worked example and analogy, and the No-Notebook Gate requires that self-sourced example to exist.

---

## 10. Gate summary (which gate runs in which mode)

| Gate | Exemplar (pilot) | Reference | First-Principles |
|---|:--:|:--:|:--:|
| Reader Flow | ✅ | ✅ | ✅ |
| Staff Depth | ✅ | ✅ | ✅ |
| Shell Invariant | ✅ | ✅ | ✅ |
| Visual/Evidence (behavioral) | ✅ | ✅ | ✅ |
| Artifact | ✅ | ✅ | ✅ |
| Notebook Smoothness | ✅ | ⛔ N/A | ⛔ N/A |
| No-Notebook Authoring | ⛔ N/A | ✅ | ✅ |

---

## 11. Implementation phasing (proposal — not executed in this goal)

1. **Phase A — spec freeze (this goal):** reports + skills patch + tracker record. *Done here.*
2. **Phase B — compiler MVP:** `compile_lesson.py` + `shell_template.html` + `blocks.py`; round-trip **one** m02 day (extract `source.md`, compile, Shell-Invariant-diff against shipped `lesson.html`). Proves the compiler reproduces known-good output.
3. **Phase C — gates as code:** the five gate scripts under `sessions/_compiler/gates/`, wired to the QA loop.
4. **Phase D — pilot calibration:** source all 9 m02 days; run the Notebook Smoothness Gate against the `00-*` notebooks; confirm parity.
5. **Phase E — first no-notebook rollout:** author m07 (JAX) or m10a (Scaling Laws) source-first in Reference Mode — the first module with **no notebook** — and validate the No-Notebook Gate end to end.

Each phase updates the tracker; no module status flips to `pass`/`pass_with_p1` under v8 until its gates are green on compiled output.

---

## 12. What this plan explicitly does NOT do

- Does **not** edit any `lesson.html` or `source.md` (none exist yet). m02 stays v7.6-shipped.
- Does **not** refactor m03 or change any module status.
- Does **not** make future modules depend on a notebook — the entire §4/§7/§9 machinery exists to prevent exactly that.
- Does **not** remove any v7.6 rule; v8 is additive and reversible.
