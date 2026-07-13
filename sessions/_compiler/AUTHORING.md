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
| `fin_title` | optional | `render_fin` | Completion banner heading. |
| `fin_body` | optional | `render_fin` | Completion banner body (HTML allowed). |
| `notebook_yardstick` | optional | notebook_smoothness_gate | Repo-relative notebook path, or `null`. `null`/absent → gate returns **N/A** (skipped, never fails). |
| `require_artifact` | optional | concept_shell_gate | Defaults to `true`; when true the produce section must contain `experiment.py`. |

Missing `module_label`, `title`, or `subtitle` raises a `KeyError` in `render_hero`
before any gate runs.

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
(path to an EXISTING `sessions/viz/*.html`, relative to the lesson), `title=`,
`caption=`. Emits a `.build-embed` iframe the shared auto-resize script drives.
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
     control: a slope flattening, a distribution shifting). The `src` MUST point at an
     EXISTING file under `sessions/viz/`.
3. **Build-up** — worked example, mechanism, `#### ` headings, callouts, Math Ladder,
   failure mode, frontier payoff.

**Every concept unit MUST ship a real visual.** The gate does not accept prose that
merely mentions `build-embed` or the word "svg" — it needs a genuinely closed
`<svg>…</svg>` (via `%%% svg`) or a `%%% viz` (which compiles to a `.build-embed`
iframe). A concept without a visual FAILS the gate. Never defer a concept's picture to
a later unit or a single shared "build" section.

Keep the `spine` word running through the hero and most concepts (gate needs it in
≥3 blocks). Keep failure modes / frontier relevance AFTER the hook and mechanism.

---

## 7. Notebook → concept-units recipe

When a `notebook_yardstick` exists:

1. Read the notebook. It is a **yardstick**, not a dependency — the lesson must open
   at least as smoothly as the notebook's first screen, but it is not a source you
   must mirror cell-for-cell.
2. Cluster its cells into topic groups (each group = one idea a reader can hold).
3. Author **one concept unit per cluster**, in teaching order.
4. For each unit pick the visual kind: **static `svg`** when the object is a fixed
   shape (a curve, a diagram); **interactive `viz`** when the point is behavioral
   (a value moving as you drag).
5. **Curation is allowed — "named ≠ covered."** If a notebook cell is out of scope for
   today, defer or omit it rather than forcing a weak unit. **Record every deferral in
   the module manifest** (`sessions/<module>/_refactor/manifest.yaml`) so coverage is
   traceable.

When `notebook_yardstick: null`, there is no notebook to match; the smoothness gate is
N/A. Reference / first-principles modules are the usual `null` case.

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
- `1` — usage / parse error.

Gates that run in concept mode (`mode: concept`):

**Reader Flow Gate** (concept branch, on source):
- hero opens human-first / has a curiosity cue, NO frontier-pressure tokens
  (`GPT-3`, `49,152`, "billions of", "frontier model", …);
- **≥3 concept units**;
- **every concept ships a visual** (`%%% svg` / `%%% viz` / a closed `<svg>…</svg>`);
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
