# V9 Concept-Mode Authoring Reference

**This file is the source of truth for authoring a V9 `source.md`.** Every grammar
claim below matches the shipped compiler (`sessions/_compiler/v8lib.py` +
`compile_lesson.py` + the gates in `sessions/_compiler/gates/`). If the compiler and
this file ever disagree, the compiler wins — fix this file.

> An earlier document, `sessions/_refactor/v8_source_first_authoring_plan.md`,
> describes an **obsolete** grammar (`:::` fences, `{{anchor}}`, `[[term:weight]]`)
> that the current compiler **rejects**. Do not author from it.

---

## 1. What V9 is

V9 is an additive lesson mode. You set `mode: concept` in the front-matter and write
the lesson body as **N concept units** instead of the fixed 7-section V8 template.
Each concept unit is one idea: a plain-words intro → its **own** inline visual →
a build-up (worked example, mechanism, failure mode). The compiler assembles the
units, a quiz, a produce section, and a completion banner into the neutral
`v9-base.donor` shell. V8 lessons (no `mode:` or `mode` != `concept`) are untouched —
they still use `@@@ section` / `@@@ region` blocks and the Shell Invariant Gate.

---

## 2. Front-matter

The file MUST start with a `---` YAML block. Keys:

| Key | Required? | Used by | Notes |
|-----|-----------|---------|-------|
| `quest_id` | **required** | `_compile_concept`, concept_shell_gate | Replaces the `__QUEST_ID__` token in the donor; localStorage key. |
| `mode` | **required** | dispatch | Must be `concept` to get the V9 path. |
| `donor` | **required** | orchestrator | `v9-base.donor` (resolved under `sessions/_compiler/shells/`). |
| `page_title` | optional | `<title>` | Empty string if omitted. |
| `module_label` | **required** | `render_hero` (kicker) + sidebar `nav-group-label` | **KeyError if missing.** Drives the sidebar group heading AND the hero kicker chip. |
| `title` | **required** | `render_hero` (`<h1>`) | **KeyError if missing.** |
| `subtitle` | **required** | `render_hero` (`<h1><span class="sub">`) | **KeyError if missing.** |
| `brand_sub` | optional | `<div class="brand-sub">` | Brand line under the Frontier.Lab wordmark. Empty if omitted; set it or the shipped donor's own brand line leaks through. |
| `spine` | optional (strongly recommended) | reader_flow_gate | The narrative-spine word. Gate lower-cases the first word of the first `:`-split segment and requires it in ≥3 blocks. Defaults to `bend` if absent. |
| `nav_prev_href` / `nav_prev_label` | optional | prev link | Empty if omitted. |
| `nav_next_href` / `nav_next_label` | optional | next link | Empty if omitted. |
| `fin_title` | **required** | `render_fin` | **KeyError if missing** (bracket access). Completion banner heading. |
| `fin_body` | **required** | `render_fin` | **KeyError if missing** (bracket access). Completion banner body (HTML allowed). |
| `notebook_yardstick` | optional | notebook_smoothness_gate + coverage_gate | Repo-relative notebook path, or `null`. `null`/absent → both gates return **N/A** (skipped, never fail). |
| `coverage_topics` | optional | coverage_gate | The in-source **skill-drafted** coverage spec (check A). A list whose items are each **a plain string** OR a `{topic: "…", keywords: […]}` mapping. **Overrides** the manifest `coverage.<day>.covers`. This is the expected-coverage list; the notebook is a separate held-out test (check B), never the spec source (see §7). |
| `require_artifact` | optional | concept_shell_gate | Defaults to `true`; when true the produce section must contain `experiment.py`. |

`coverage_topics` schema — a topic is COVERED if any of its keywords (default: the topic
string itself) appears as a substring in the compiled lesson's normalized text:
```yaml
coverage_topics:
  - relu                                           # plain string: keyword = "relu"
  - {topic: vanishing gradient, keywords: [gradient shrinks, vanish]}   # any keyword matches
```

Missing `module_label`, `title`, or `subtitle` raises a `KeyError` in `render_hero`;
missing `fin_title` or `fin_body` raises a `KeyError` in `render_fin`. Either crash exits
the compiler with code `1` **before any gate runs** — so all five are hard-required.

---

## 3. Block grammar

Top-level blocks are delimited by a line beginning with `@@@ ` (three at-signs, a
space, then the type). Everything after that line, until the next `@@@ ` line, is the
block body. Arguments are `key="value"` pairs on the `@@@` line.

Concept-mode uses these block types (in body order):

```text
@@@ hero
@lede <one-two sentences: the curiosity hook — human/you/imagine/?, NO frontier-pressure>
@goal <what the reader can do by the end>

@@@ concept id=c1 tag="Short nav label" title="Section heading" gotit="Got it"
<intro → inline visual → build-up>

@@@ concept id=c2 tag="…" title="…" gotit="…"
<…>

@@@ concept id=c3 tag="…" title="…" gotit="…"
<…>   (you need at least 3 concept units)

@@@ concept id=cN tag="Recap" title="Today in one page" gotit="Got the recap"
<REQUIRED final concept: the day's summary in a few beats + a reference cheat-sheet
 (%%% jargon and/or %%% table) + its own visual>

@@@ quiz id=quiz tag="Quiz" title="Section heading" gotit="answer all first"
%%% quiz
<exactly 4 question lines — see §4>
%%%

@@@ produce id=produce tag="Produce" title="Section heading" gotit="Done"
<discovery-framed prose that literally contains the string experiment.py>

@@@ fin
```

Argument syntax: `key="value"` (quoted; use quotes whenever the value has spaces).
Bare `key=value` also parses but stops at the first space, so always quote `title=`,
`tag=`, `gotit=`. Parsed by `parse_args_kv` — the regex is
`(\w+)=(?:"([^"]*)"|(\S+))`.

Block notes:
- `hero` — uses `@lede` then `@goal` markers inside the body. The kicker chip comes
  from `module_label`, the `<h1>` from `title`, the sub-heading from `subtitle`.
- `concept` — `id=` MUST match `c\w+` (e.g. `c1`, `c2`) so the concept_shell_gate
  finds it. `tag=` is the sidebar label; `title=` is the section heading; `gotit=` is
  the button label (each concept gets exactly one Got-it button, auto-added).
  Concepts are auto-numbered 1..N in source order.
- `quiz` — put the `%%% quiz` widget in the body. Exactly one quiz block, 4 questions.
- `produce` — the body must reference `experiment.py` (artifact gate).
- `fin` — empty body; the banner text comes from `fin_title` / `fin_body`.

---

## 4. Inline widgets (`%%% <type> … %%%`)

A widget is a block-level directive: a line starting `%%% <type> <args>`, body lines,
then a closing line that is exactly `%%%`. Args on the opening line use the same
`key="value"` syntax. Types:

**`svg`** — a static labeled figure. Body is raw SVG, passed through, wrapped in
`<div class="build-viz">`. This is the cheapest way to satisfy the per-concept
visual requirement.
```text
%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="…">…</svg>
%%%
```

**`viz`** — an interactive embed (iframe) for behavioral concepts. Args: `src=`
(path to a `sessions/viz/*.html`, relative to the lesson — it should point at a file that
actually exists; a bad path still compiles and passes gates but renders an empty iframe at
runtime), `title=`, `caption=`. Emits a `.build-embed` iframe the shared auto-resize script
drives.
```text
%%% viz src=../../viz/activation-derivatives.html title="slopes" caption="drag z"
%%%
```

**`demo`** — inline run-demo: a code line, hidden output + takeaway revealed on click.
Args: `id=`, `label=`. Body is `key: value` lines (`code:`, `out:`, `take:`).
```text
%%% demo id=relu label="run it"
code: relu(np.array([-3, -1, 0, 2, 5]))
out: array([0, 0, 0, 2, 5])
take: <b>ReLU = max(0, z).</b> Zeroes negatives, passes positives.
%%%
```

**`hint`** — a tiered, OFFLINE progressive-disclosure hint ladder for a STUCK learner
(get-unstuck). `t1:` a gentle nudge, `t2:` a worked micro-step, `t3:` the idea in one
sentence (`t4:` optional). Tiers ship hidden; donor JS un-hides ONE per click. No network.
```text
%%% hint
t1: What is the slope doing right where the ball sits?
t2: Read the ground under your boots — a steeper tilt means a bigger step.
t3: The gradient IS that local slope; you step the opposite way (downhill).
%%%
```

**`quiz`** — one question per body line, `|`-separated:
`q: <ask> | a:<IDX> | <opt0> | <opt1> | <opt2> | <opt3> | fb: <feedback>`.
`a:IDX` is the **0-based** index of the correct option (only counts as the answer
marker when the tail after `a:` is all digits — otherwise it is treated as an
option). Give 4 options for a standard question. Concept mode needs exactly 4 lines.
```text
%%% quiz
q: What does an activation add? | a:1 | params | non-linearity | speed | bias | fb: The bend.
q: Ten linear layers = ? | a:1 | more power | one linear layer | random | sigmoid | fb: One matrix.
q: ReLU(z) = ? | a:1 | 1/(1+e^-z) | max(0,z) | z^2 | -z | fb: keep positives.
q: All ReLUs output 0? | a:1 | sigmoid bug | dead ReLUs | OOM | converged | fb: dead units.
%%%
```

**`table`** — comparison table, cells `::`-separated, first row = headers.
```text
%%% table
:: ReLU :: sigmoid :: tanh
Range :: `[0, ∞)` :: `(0, 1)` :: `(−1, 1)`
%%%
```

**`jargon`** — Jargon Ladder. Each line `word | plain-English gloss`.
```text
%%% jargon
linear | a straight-line relationship
activation | the little bend added at the end of a neuron
%%%
```

**`cards`** — analogy cards. Each line `emoji | title | body`.
```text
%%% cards
🚰 | One-way valve | ReLU lets positives through, blocks negatives
💡 | Dimmer | sigmoid eases smoothly from off to on
%%%
```

**`formula`** — a formula callout. Body `key: value` lines `expr:` and `note:`.
```text
%%% formula
expr: ReLU(z) = max(0, z)
note: keep positives, zero negatives; range [0, ∞)
%%%
```

**`mathladder`** — the 4-rung Math Ladder callout. Body `key: value` lines
`title:`, `words:`, `formula:`, `numbers:`, `sanity:`.
```text
%%% mathladder
title: why linear ∘ linear collapses
words: two matmuls back-to-back do the job of one
formula: g(f(x)) = (ab)·x + (bp+q)
numbers: x=[1,2], W1=[[1,2],[0,1]], W2=[[1,0],[3,1]] → (xW1)W2 = x(W1W2)
sanity: add a bend first and the two stop matching
%%%
```

**`prompt`** — Produce Option-B copy box. Args: `id=`, `label=`. Body is the verbatim
prompt text (leading/trailing blank lines trimmed).
```text
%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my artifact.
Create experiment.py that …
%%%
```

Unknown widget types raise `ValueError`.

---

## 5. Callouts, glossary, headings

**Callouts** — a line `!!! <class> <emoji>`, body lines, closing `!!!`.
Classes styled by the shell: `c-info` (blue), `c-warn` (amber), `c-ok` (green).
```text
!!! c-warn ⚠️
<b>Failure mode (silent):</b> a dead ReLU outputs 0 forever and stops learning.
!!!
```

**Glossary term** — `[[phrase||tooltip]]` renders `phrase` with a hover tooltip.
The `||` separates the visible phrase from the tooltip text.
```text
a [[linear||A straight-line relationship: output changes in fixed proportion.]] step
```

**Sub-heading** — `#### Heading text` → `<h4>`.

**Other inline markup** — backtick `` `code` ``, `**bold**`, `*italic*`.

**Lists** — `- item` (unordered) or `1. item` (ordered); consecutive lines group.

**Raw HTML escape** — `~~~html` … `~~~` passes a block through verbatim (fallback;
prefer the typed widgets above).

---

## 6. The concept-unit pattern

Every concept unit follows the same three beats, in this order:

1. **Intro** — plain words, define-before-use. Front-load a Jargon Ladder or glossary
   term the first time a word appears.
2. **Its OWN inline visual** — placed immediately after the intro, before the
   build-up. Choose:
   - `%%% svg` for a **fixed shape** (a curve, a matrix diagram, a boundary) — a
     static labeled figure.
   - `%%% viz src=…` for **behavioral** content (something that changes as you drag a
     control: a slope flattening, a distribution shifting). The `src` should point at a
     file that exists under `sessions/viz/` — this is a runtime-quality concern (a missing
     file renders an empty iframe), not a compile gate.
3. **Build-up** — worked example, mechanism, `#### ` headings, callouts, Math Ladder,
   failure mode, frontier payoff.

**Every concept unit MUST ship a real visual.** The gate does not accept prose that
merely mentions `build-embed` or the word "svg" — it needs a genuinely closed
`<svg>…</svg>` (via `%%% svg`) or a `%%% viz` (which compiles to a `.build-embed`
iframe). A concept without a visual FAILS the gate. Never defer a concept's picture to
a later unit or a single shared "build" section.

Keep the `spine` word running through the hero and most concepts (gate needs it in
≥3 blocks). Keep failure modes / frontier relevance AFTER the hook and mechanism.

**Beginner-mentality structural rules (user directive 2026-07-20 — enforced by the
Reader Flow Gate + tone/interest/structure judges + the concept_structure gate):**

1. **Everyday-analogy hero.** The `@@@ hero` `@lede` opens with a concrete everyday
   analogy (a physical thing a 12-year-old has done) BEFORE any aspiration/relevance.
2. **Jargon as-you-go, not a wall.** Introduce each term define-before-use with an
   inline `[[term||plain gloss]]` the first time it appears. Do NOT put a `%%% jargon`
   block in the FIRST concept — the gate hard-fails a front-loaded jargon wall there.
3. **A closing recap unit is REQUIRED.** The FINAL concept before the quiz is a
   one-page recap (`tag="Recap"`): the day's arc in a few beats + a reference
   cheat-sheet (`%%% jargon` and/or `%%% table`) + its own visual. The gate hard-fails
   if the last concept is not a recap.
4. **Math restraint.** Lead each concept with intuition + its visual + the analogy.
   Keep any main-flow formula to a single narrated line; DEMOTE heavy or multi-step
   math (derivations, matrix algebra, closed forms) into an `!!! c-info` box whose
   first words mark it "Optional (skippable)".
5. **Visualize the build-up.** The concept's opening inline visual (beat 2) anchors the
   OPENING intuition only — it does NOT discharge the duty to visualize the *build-up*.
   Whenever a concept's build-up is heavy or multi-step — a worked numeric example, a
   derivation, a growth/decay ladder (e.g. the `0.25×0.25×…` vanishing-signal fade), a
   matrix/shape transformation, a multi-symbol formula, or anything you put in an
   "Optional (skippable)" box — SHOW that build-up as a picture placed AT the build-up
   (after the opening visual): a **second `%%% svg`** that draws the transformation itself
   (the shape/curve/value changing as each term is added; a before→after in ONE figure
   when a step fixes a problem), a **`%%% demo`** whose `out:` makes the numbers watchable
   (predict-then-run), a draggable **`%%% viz`**, or a **`%%% mathladder`** (words→formula
   →numbers→sanity). Humans read a graph far faster than algebra — never leave a numeric
   ladder or worked example as monospace text when an SVG of bars/curves would let the
   reader SEE it. Net: a concept with a real build-up ships ≥2 visual elements (opening
   anchor + build-up figure); a light/definitional concept needs only its one opening
   visual. Enforced by the Concept-Structure Judge's `buildup_visualized` axis (MISSING
   → P0, WEAK → P1) and warned by the `concept_structure` gate (advisory, offline floor).
6. **Draw the analogy.** Each concept states an everyday analogy in words — *also draw it.*
   The concept's OPENING visual (beat 2) ILLUSTRATES the everyday THING the analogy names —
   a one-way valve (ReLU), a dimmer (sigmoid), a see-saw (tanh), a ruler (linear collapse),
   a pizza sliced into shares (softmax), an assembly-line conveyor (forward pass), a golf
   score (MSE), a weather forecaster's surprise (cross-entropy) — drawn with labelled parts,
   NOT the equation or a bare axis/curve plot. The reader should look and think "oh — it's
   like a ___" before meeting the math. The math/mechanism picture (a curve, a worked
   example, an interactive explorable widget) is the BUILD-UP visual (rule 5) and comes
   AFTER. So a full concept typically ships the analogy picture FIRST, then the mechanism
   picture. Enforced by the Concept-Structure Judge's `analogy` axis: GOOD requires the
   analogy be DRAWN; analogy-in-words-only is WEAK, which gates the build loop.

---

## 7. Coverage is skill-drafted; the notebook is a test

**Draft coverage from domain knowledge, not from a notebook.** The concepts a lesson
must teach come from the topic itself (see `frontier-curriculum-architect` → Coverage
Spec Rule): core mechanism family, historical ancestor (when it motivates the modern
form), **every failure mode paired with its remedy**, capability limits, forward-pointers
(deferred), and out-of-scope items (with a written reason). Record this as the manifest
`coverage.<day>.covers` (or in-source `coverage_topics`). This works even when no notebook
exists (JAX, scaling laws).

When a `notebook_yardstick` DOES exist it plays two roles, neither of which is "the list
of what to cover":

1. **Reader-flow yardstick** — the lesson must open at least as smoothly as the notebook's
   first screen (Notebook Smoothness Gate). It is not a source you mirror cell-for-cell.
2. **Held-out coverage test** — the Coverage Gate's check B compares the notebook's
   concepts to your skill-drafted spec. A notebook concept missing from `covers` /
   `deferred` / `out_of_scope` is a **SKILL-GAP**: fix the architect skill's derivation,
   re-draft `covers`, regenerate, re-eval — until the skill reproduces the notebook's
   concepts without reading it.

Authoring a lesson body from the spec:

1. Turn each `covers` concept into **one concept unit**, in teaching order.
2. For each unit pick the visual kind: **static `svg`** for a fixed shape (a curve, a
   diagram); **interactive `viz`** for behavioral content (a value moving as you drag).
3. Curation is allowed — "named ≠ covered." Defer or scope-out a concept rather than
   forcing a weak unit, and **record it in the manifest** so coverage stays traceable.

When `notebook_yardstick: null`, there is no notebook; the smoothness gate and check B are
both N/A, and the skill-drafted spec stands on its own (check A still runs).

---

## 8. Compile + gate loop

Command:
```bash
python3 sessions/_compiler/compile_lesson.py <source.md>
# iterate without writing output:
python3 sessions/_compiler/compile_lesson.py <source.md> --check-only
# explicit donor / output:
python3 sessions/_compiler/compile_lesson.py <source.md> --donor sessions/_compiler/shells/v9-base.donor --out /tmp/out.html
```

Flags: `--donor <html>`, `--out <html>`, `--check-only` (run gates, write nothing),
`--quiet`.

Exit codes:
- `0` — compiled and all gates pass.
- `2` — Reader Flow Gate failed (nothing written).
- `3` — Concept Shell Gate (or Notebook Smoothness) failed.
- `4` — Visual Integrity Gate failed (a visual would render blank).
- `1` — usage / parse error.

Gates that run in concept mode (`mode: concept`):

**Reader Flow Gate** (concept branch, on source):
- hero opens human-first / has a curiosity cue, NO frontier-pressure tokens
  (`GPT-3`, `49,152`, "billions of", "frontier model", …);
- **≥3 concept units**;
- **every concept ships a visual** (`%%% svg` / `%%% viz` / a closed `<svg>…</svg>`);
- **no front-loaded jargon wall** — a `%%% jargon` block in the FIRST concept fails;
- **a closing recap unit** — the LAST concept must be a recap (recap-ish `tag`/`title`,
  or a `%%% jargon` cheat-sheet in it);
- the `spine` word appears in ≥3 blocks (hero + concepts);
- the produce block is discovery-framed (contains a cue: predict / observe / notice /
  watch / "what you should see" / before you …).

**Concept Shell Gate** (on compiled HTML):
- `quest_id` is present (frozen);
- **≥3** `<section class="module-section" id="c…">`, each with a real visual
  (closed `<svg>` OR `class="build-embed"…<iframe`) **and exactly one** `class="gotit"`;
- **exactly one** quiz section, and it has **exactly 4** `class="q"` questions;
- **exactly one** produce section; if `require_artifact` (default true) its HTML
  literally contains `experiment.py`;
- sidebar nav parity — the set of `data-target`s (minus `home`) equals the set of
  section ids;
- localStorage keys present (`frontier-lesson:`, `frontier-theme`), `.fin` banner
  present, and NO leaked markers (`<!--V9_CONTENT-->`, `<!--V9_NAV-->`, `__QUEST_ID__`,
  `@@@`, `%%%`).

**Notebook Smoothness Gate**:
- `notebook_yardstick: null`/absent → **N/A** (skipped, never fails);
- otherwise compares the compiled hero lede to the notebook's opening: fails if the
  hero opens frontier-first, has no human/curiosity cue, or opens on a formula wall.

**Coverage Gate (ADVISORY — never fails the build)**:
Principle: **the skills draft coverage; the notebook is a held-out TEST of the skills, never the source of coverage.** (Many topics — JAX, scaling laws — have no notebook at all.) It runs two independent checks and writes an advisory `<day>/_coverage.md` sidecar (it never edits `lesson.html`):
- **Check A — execution** (always, no notebook needed): does the lesson realize the SKILL-DRAFTED spec? The spec is `coverage_topics` (front-matter), else the manifest `coverage.<day>.covers`. Each spec topic → COVERED / DEFERRED / **EXEC-GAP**. An EXEC-GAP means the builder didn't render a spec item → fix the lesson.
- **Check B — skill eval** (only when `notebook_yardstick` is set): does the notebook teach a concept the spec never listed / deferred / scoped-out? If so it is a **SKILL-GAP** → the architect skill's coverage-derivation missed it; fix the skill, re-draft `covers`, regenerate, re-eval. Matching is whole-concept and token-aware, so a spec entry `relu` does NOT account for `leaky relu`, and out-of-scope `elu` does NOT swallow `relu`.
- **N/A** only when there is neither a spec nor a notebook. Where there is no notebook, only check A runs and the skill is trusted (validated by check B on topics that do have notebooks).
- The gate is advisory: you **drive its status to PASS** by authoring a complete skill-drafted spec and recording legitimate curation, but it never blocks compilation.

**Visual Integrity Gate (HARD — blocks the build, exit 4)**:
`concept_shell_gate` proves a visual *marker* exists; this proves the visual will
actually *render*. It runs in concept mode and checks: every `%%% viz src=` file
EXISTS; each viz file's local `<script src=>` deps EXIST (the missing-`d3.v7.min.js`
case); each viz carries a `postMessage` height-sender whose message `type` matches
the donor receiver (`viz-height`); and every inline `%%% svg` is NON-degenerate (has a
`viewBox` and a real shape, not a bare `M0 0` path). It CANNOT verify pixels — there is
no browser, and a nested `file://` iframe may be blocked by the browser regardless of
content (serve over http to view). Run it standalone with
`python3 gates/visual_integrity_gate.py <source.md>`; the test suite also guards every
shipped concept lesson.

---

## 9. Donor-reuse contract

- Reuse `sessions/_compiler/shells/v9-base.donor` **verbatim**. Never hand-edit it.
- It is a **neutral** template. It carries the tokens `__QUEST_ID__` (in
  `data-quest-id`), `<!--V9_NAV-->` (sidebar nav insertion point), and
  `<!--V9_CONTENT-->` (main content insertion point). The compiler substitutes these.
- Module identity comes from **front-matter**, not the donor:
  - `module_label` drives BOTH the sidebar `nav-group-label` AND the hero kicker chip;
  - `brand_sub` drives the brand line under the Frontier.Lab wordmark;
  - `title` / `subtitle` drive the hero `<h1>`.
- The shipped donor snapshot may still carry an old module's `brand-sub` text. A new
  module MUST set its own `module_label` and `brand_sub`, or the lesson will mislabel.

---

## 10. Minimal compiling example

This compiles clean (exit 0, all gates pass). It has 3 concept units (each with a
visual), a 4-question quiz, and a discovery produce that references `experiment.py`.

```markdown
---
quest_id: demo-mini
mode: concept
donor: v9-base.donor
page_title: "Demo · Mini"
module_label: "Demo · Mini"
title: "Mini"
subtitle: "a tiny concept lesson"
brand_sub: "Demo · Mini"
spine: "bend"
notebook_yardstick: null
fin_title: "Mini complete!"
fin_body: "Done."
---

@@@ hero
@lede Ever wonder why a bend matters? Picture two straight rulers stacked.
@goal By the end you can explain the bend.

@@@ concept id=c1 tag="The collapse" title="Straight + straight is still straight" gotit="Got it"
Two straight rulers stacked are still one straight ruler — a [[bend||a non-linear step]] is missing.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="collapse"><path d="M0 0 L10 0"/></svg>
%%%
So depth without a bend buys nothing.

@@@ concept id=c2 tag="The bend" title="A bend between the layers" gotit="Got the bend"
Put a bend between layers and they can no longer fold flat.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="bend"><path d="M0 10 L5 0 L10 10"/></svg>
%%%
That bend is the activation.

@@@ concept id=c3 tag="Meet ReLU" title="ReLU — a one-way valve" gotit="Met ReLU"
ReLU passes positives and zeroes negatives — a one-way valve, and the modern default bend.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="relu"><path d="M0 10 L5 10 L10 0"/></svg>
%%%
%%% demo id=relu label="run it"
code: relu(np.array([-3, -1, 0, 2, 5]))
out: array([0, 0, 0, 2, 5])
take: ReLU zeros negatives, passes positives.
%%%
One cheap max.

@@@ concept id=c4 tag="Recap" title="Today in one page" gotit="Got the recap"
It all came down to the bend: straight+straight stays straight, a bend makes depth matter, and ReLU is the cheap default bend. Come back to this any time.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="recap"><path d="M0 8 L4 8 L8 2"/></svg>
%%%
%%% jargon
bend | the small non-linearity (activation) at the end of a neuron
ReLU | a one-way valve: passes positives, zeroes negatives
%%%

@@@ quiz id=quiz tag="Check" title="Four questions" gotit="answer all first"
%%% quiz
q: What does an activation add? | a:1 | params | non-linearity | speed | bias | fb: The bend.
q: Ten linear layers = ? | a:1 | more power | one linear layer | random | sigmoid | fb: One matrix.
q: ReLU(z) = ? | a:1 | 1/(1+e^-z) | max(0,z) | z^2 | -z | fb: keep positives.
q: All ReLUs output 0? | a:1 | sigmoid bug | dead ReLUs | OOM | converged | fb: dead units.
%%%

@@@ produce id=produce tag="Produce" title="Watch the collapse" gotit="Done"
Predict what `(x@W1)@W2` prints versus `x@(W1@W2)`, then run it and observe they match —
until you insert a ReLU. Write it in `experiment.py` and run it.

@@@ fin
```
