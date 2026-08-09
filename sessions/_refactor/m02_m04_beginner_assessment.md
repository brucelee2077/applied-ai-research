# m02–m04 — "Can a 12-year-old learn this?" assessment

**Date:** 2026-08-04 · **Scope:** 20 shipped days (m02 ×9, m03 ×5, m04 ×6)
**Method:** two independent reads compared against each other —
(1) the engine's own judge panel re-run on all 20 days, and
(2) a blind 12-year-old cold read (one fresh agent per day, told nothing about our
gates), every claimed stall then adversarially verified by a separate agent.
**Recommendation: Pass with P1.**

---

## 1. Verdict

**Yes — the teaching is solid.** This is not a "lessons are too hard" problem.
Where it is weak is the *enforcement*: several stated rules are not actually
checked by anything, and one compiler bug is corrupting the text every LLM judge
reads. Those are the skill-level items worth fixing before more modules ship.

Evidence the teaching itself is good:

| Signal | Result |
|---|---|
| Cold reader would return tomorrow | **19/20 YES_EAGER**, 1 YES_DUTIFUL (m04 d05) |
| Claimed stalls surviving adversarial verify | **9 of 120 (92% refuted)** — 8×P2, 1×P1, 0×P0 |
| Heroes opening on a concrete everyday analogy | **20/20** (mini-golf, ruler, assembly line, library, movie seats) |
| Visual density | **1.1–2.7 SVGs per concept**, every concept has its own |
| Doing leg (`experiment.py`) | **20/20 gate-passing, 20/20 golden-pinned** |
| Main-line prose walls >600 chars (m02+m03) | **0** |

The 92% refutation rate is itself the headline: a fresh reader's complaints were
overwhelmingly "I forgot what was explained earlier," not "this was never
explained."

---

## 2. The four findings

### P0 — Compiler corrupts tooltips, and feeds the corrupted text to every judge

Not a writing defect. Authors wrote correct sentences; `v8lib.inline()` breaks them.
`v8lib.py:66` substitutes `[[term||gloss]]` **first**, so the `*`/`**` rules at
lines 77–78 then rewrite *inside* the `data-tip` attribute. `attr_esc` (line 61)
escapes only `&` and `"`, so nothing stops it. Both shapes reproduce:

```
IN : the [[gradient||slopes; steepest *increase*, so step opposite]]: keep going.
OUT: ... data-tip="slopes; steepest <em>increase</em>, so step opposite">gradient</span>

IN : a [[step()||replaces W -= lr*dW lines]] and [[SGD||w := w - lr*grad, per param]]
OUT: ... data-tip="replaces W -= lr<em>dW lines">...  data-tip="w := w - lr</em>grad, ...
```

The second is worse: one `<em>` spans **two different attributes**.

Two consequences, both verified:
- **Reader:** `v9-base.donor` does `body.textContent = getAttribute('data-tip')`, so
  hovering `gradient` shows the literal string `steepest <em>increase</em>`.
- **Judges:** `coverage_judge._readable_text` (a naive `<[^>]+>` strip) desyncs on the
  stray `>` and hands the panel mangled prose. Confirmed on m04 d02, the judges read:
  *"…is exactly the increase, so you step the opposite way to go downhill.">gradient…"*
  Every lens graded that and returned MATCHES/GOOD.

**Corpus-wide: 4 tooltips in 3 lessons**, all m04 (d01 `argmax`, d02 `gradient`,
d06 ×2). Small blast radius, but it degrades the input to all the other lenses.

**Fix (tested):** stash gloss bodies behind `\x01` placeholders exactly as the
existing code-span guard already does, restore after the bold/em rules, and extend
`attr_esc` to escape `<`/`>`. Verified to fix all four cases with bold/em/code
behavior unchanged. Add a three-second post-compile assertion in
`concept_shell_gate.py`: no `data-tip` value may contain `<` or `>`.

### P1 — "No term appears before its gloss" is enforced by nothing

`frontier-lesson-builder/SKILL.md:343` states the rule. Its own "Checkable form"
(line 347) names two enforcers, and **neither checks it**:

- `reader_flow_gate.py:84` only asserts a jargon wall is *absent*, never that a
  gloss is *present*.
- The single place `[[` is read (line 139) sits behind `if strict:` on the v8
  non-concept path — **unreachable**, because `run()` returns at line 114 for
  `mode: concept`. All 20 days are `mode: concept`, so nothing reads `[[` at all.
- The tone judge's `progressive_disclosure` is notebook-relative and returns `N/A`
  when `notebook_yardstick: null` — true for 9 of 20 days, including every m03 day.

Measured (my scan, cumulative-aware, markup and front-matter stripped):
**14 genuinely undefined first uses across 10 days.** Note this is *far* smaller
than a raw scan suggests — of 39 bare first-uses in body prose, 25 are self-defining
at the point of use ("That whisper has a name — the **gradient** —"). The rule is
mostly being honored by good writing, not by tooling.

Terms never glossed anywhere in their module: `framework`, `ln`, `arrow`,
`slot-pair`, `symmetric`, `generalize`, `GELU`. (`Leaky ReLU` *is* glossed —
the synthesizer got that one wrong.)

### P1 — A number the reader is told to check but cannot derive

The clearest single reader-facing defect, and the cold read's only P1.
`m03-attention/day-01-embeddings:149` promises:

> "you can do them with all four numbers on a scrap of paper"

Then line 156 supplies `cat's length is 1.24`. **`sqrt`, `√`, and "square root"
appear zero times in that file.** `1.24 = √(0.9²+0.8²+0²+0.3²)` is never shown, not
even in the skippable box. A reader who accepts the invitation gets stuck on exactly
the step the day's payoff depends on, and ends up copying numbers instead of
understanding them.

Same shape elsewhere: `ln K` → 0.69/2.3 (m02 d04, and `ln` arrives with no note that
it is the `log` used all lesson); `1.2` (m02 d08, correct as `|1 − 2.2|` but derived
nowhere, and the adjacent word "doubling" contradicts it).

No lens grades arithmetic reproducibility. `buildup_visualized` asks only whether a
build-up is *drawn* — m03 d01's ladder is a `%%% steps` widget, so it scores GOOD
while `1.24` stays unsourced.

### P1 — m04 was never held to the m02/m03 bar

m04 is a genuine outlier and no gate noticed:

| | m02 | m03 | **m04** |
|---|---|---|---|
| `%%% steps` (chunking) | 71 | 45 | **0** |
| `%%% insight` (re-hook) | 72 | 42 | **0** |
| `predict:` (predict-then-reveal) | 42 | 29 | **0** |
| Main prose walls >600 chars | 0 | 0 | **7** (worst 1002) |

Cause: `_density_scan.py:20-21` hardcodes m02 and m03 only — **m04 has never been
density-scanned.** The Build-Up Register names `%%% steps`/`%%% insight` as the
primary chunking devices; m04 uses neither, and carries the only oversized walls
in the corpus.

Separately, **no day in m03 or m04 ships a single interactive iframe** (m02 has 7
across 3 days). The interest judge nonetheless credited m04 d02 with "interactive
%%%viz widgets" — there are zero `%%% viz` blocks in all of m04. That is a judge
hallucination crediting play that does not exist.

---

## 3. The engine grades itself perfect — that is the real risk

All 20 days: tone MATCHES_NOTEBOOK (where applicable), interest FLOOR_MET,
structure GOOD, body GOOD. Every day green on every axis.

But the panel is also **self-inconsistent**. The absolute interest floor's own
stated rule is *"FLOOR_MET only if NO lever is MISSING and fewer than two are WEAK."*
Across the corpus `momentum` is WEAK/BELOW on **18 of 20 days**, and **6 days have
≥2 WEAK levers yet still returned FLOOR_MET** (m03 d04, m03 d05, m04 d02, d03, d04,
d06). The judge is not applying its own threshold.

The deterministic backstop that should catch this never fires. I traced
`concept_structure_gate.py`'s failure-cluster check on m04 d04 (a WEAK-momentum day):
it requires ≥3 *consecutive* concepts whose **title** matches a failure token AND
that carry no `%%% demo`/`%%% viz`. Max observed run length: **1**. The clustering
the judge complains about happens *inside* build-ups and in prose, which a
title-only regex cannot see.

---

## 4. Recommended skill-level changes

Ordered by leverage. Prefer deterministic — a deterministic check is cheap and
cannot drift or hallucinate.

1. **Fix `v8lib.inline()`** (P0, ~5 lines, prototype tested) + the `data-tip`
   assertion in `concept_shell_gate.py`. Do this first: it is currently corrupting
   the input every other lens reads.
2. **New `term_grounding_gate.py`**, run beside `concept_structure_gate.py`. Build the
   gloss set from `[[term||…]]`, strip `<svg>` and `!!!` boxes, then flag (a) a bare
   first use preceding its gloss, and (b) a term in a quiz stem never glossed in the
   body. Land as warn, promote to fail after one cleanup pass. Also **delete or
   rewrite SKILL.md:347's "Checkable form"** — it currently names two enforcers that
   do not enforce, which is worse than admitting the rule is unchecked.
3. **Un-hardcode `_density_scan.py`** to take a module argument, and run it on every
   module at build time. Then bring m04 to the m02/m03 bar (`%%% steps`/`%%% insight`
   chunking, split the 7 oversized walls). This is content work, not a rewrite.
4. **Unsourced-number check** over `%%% steps`/`%%% mathladder` bodies: flag a decimal
   literal that neither appeared earlier nor sits on the RHS of an arithmetic `=`.
   Hard-fail when the source also promises "on a scrap of paper / check it yourself /
   by hand" — that pairing is the actual defect and is low-noise. Add a Build-Up
   Register beat: *every number in a ladder is either carried in from an earlier step
   or shown being computed.*
5. **Make the interest floor apply its own rule** — compute FLOOR_MET from the lever
   counts in code rather than trusting the model's `overall` field. And since
   `momentum` is WEAK on 18/20 days, either the bar or the guidance needs to change;
   right now it is a permanently-ignored warning, which trains everyone to ignore it.

---

## 5. What I did not change

No lesson content, gate, or skill was edited — this run is assessment only.
One incident worth recording: I clobbered
`m02-the-neuron/day-05-gradients-backprop/source.md` by importing `_density_scan.py`
in a way that ran its `main()`. Restored from git; the working tree is verified back
to its pre-session state (the only modified file, `concept_shell_gate.py`, predates
this session).
