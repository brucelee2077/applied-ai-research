# Skills Patch Report — v7.2 Frontier Lab Skills (from m02 held-out eval)

> **Trigger:** `m02_v7_2_heldout_eval_report.md` found 4 **systemic** skill gaps (S1–S4) — patterns that
> would recur in the next module, not m02-specific defects.
> **Scope honored:** edited **only** `.claude/skills/frontier-*/SKILL.md` (the installed skills under test).
> **No lesson, contract, notebook, or shell file was touched in Phase 3** (verified via `git status`).
> **Mirror:** the identical patches were synced into `frontier_lab_refactor_skills_v7_2/skills/*` so the
> installed skills stay byte-identical to the v7.2 package (the invariant verified at Phase-1 start).
> **Edit style:** purely additive — new labeled sections; no existing rule removed or weakened.

---

## Root cause the eval exposed

A plain beginner Jupyter notebook (the held-out reference) **out-evidenced** a polished v7.2 module on the
hardest-to-fake axes: **code that runs and curves that plot.** The v7.2 system reliably produces superior
*conceptual* depth (mapping tables, scalars-first derivations, staff-lens, interview framing) but has a
repeatable blind spot — it accepts **static SVGs + simulated "fake terminals" + prose tables** in place of
**plotted/behavioral evidence**, and the QA gate cannot catch this because it is calibrated to the same
contracts that chose those surfaces. The four patches close that loop at the skill level.

---

## Patches applied

### S1 — `frontier-visual-evidence-builder`: Behavioral vs structural visual rule
**Gap:** the "changing quantity" / "simulated-vs-runnable" rule did not distinguish a **behavioral** concept
(a curve/derivative/boundary/gradient *moving over a range*) from a **structural** diagram. So a static SVG
with baked-in numbers, or a prose "slope near 0 ≈ 0.25" table, was accepted as the P0 visual on
derivative-/geometry-centric topics (D1 boundary geometry, D2 activation derivatives, saturation/dead-ReLU).

**Edit:** added a **"Behavioral vs structural visual rule"** section — structural relationships may use a
static SVG; **behavioral** concepts require a **plotted or live-recomputed** viz (a real plot, or the
existing `build-embed` iframe → `viz/*.html`), and a prose/table readout does **not** satisfy the P0 visual.
Litmus: *if changing a parameter should change the picture, the picture must actually recompute.* Extended
the **"Show-the-failure rule"** so shape-signature failures (derivative flattening, loss diverging) must be
the plotted/run curve, and added the saturation/vanishing-gradient example. Added
"behavioral concept shown only via static SVG / prose table" to the workflow-reporting list.

*Evidence:* D2 has no plotted derivatives (reference plots every curve + derivative, cell 37); D1 boundary
geometry is prose-only (reference derives + plots it, cells 13–17). Convergent across all 3 held-out agents.

### S2 — `frontier-refactor-qa`: names-it≠covers-it · prose-for-behavioral · held-out yardstick · dead-escape-hatch
**Gap:** QA credited **"names it" as "covers it"** (Leaky ReLU / softmax / decision boundary named but never
shown → still passed) and credited **prose assertions** for behavioral claims; it also silently waved the
dead `/frontier-experiment-lab` Option-B path through as a sanctioned legacy reference.

**Edit:** added four sections:
- **"Names-it-is-not-covers-it rule"** — a concept only *named* is a coverage gap (PARTIAL at best), not coverage.
- **"Prose-for-behavioral rule"** — a behavioral claim stated only in prose/table is a **visual gap** ("asserts a shape it never shows").
- **"Held-out yardstick step"** — before sign-off, ask of each core topic *"would a canonical treatment SHOW what this lesson only TELLS?"*; do not let staff-lens/frontier polish offset weak beginner evidence (operationalizes the skill's own over-crediting warning).
- **"Dead-escape-hatch flag"** — if a Produce auto-build path routes to an uninstalled skill, flag it (≥P1) rather than silently waive, noting the learner's only shipped runnable evidence is then a stub.

*Evidence:* QA passed D1/D2/D3 clean despite S1/S3/S4; self-referential gate blind spot.

### S3 — `frontier-curriculum-architect`: Capability-Limits & Geometry Lens
**Gap:** coverage contracts enumerated forward formulas but not a topic's **capability limit** (single
neuron → XOR / linear-only boundary) or its **geometric interpretation** (what `w`/`b` do to a boundary),
so the built lessons omitted the two facts that define what a single neuron *can and cannot do*.

**Edit:** added **lens #10 "Capability-Limits & Geometry Lens"** to the Coverage Engine, requiring every core
mechanism's contract to add (a) a **capability-limit** row (the defining thing it *cannot* do — often the
reason the next concept exists) and (b) a **geometric-interpretation** row paired with a **plotted/behavioral**
visual requirement (coordinated with S1), never prose-only.

*Evidence:* XOR/linear-separability absent from both files; boundary geometry thin. Convergent across the
what-is-NN and single-neuron agents.

### S4 — `frontier-lesson-builder`: Interview-answer grounding rule
**Gap:** a scripted interview/staff answer could assert a picture or result the lesson never taught. D1's
interview line has the learner recite "…shifts the **decision boundary** off the origin" — but "boundary"
appears *only* in that scripted line; no boundary is ever drawn.

**Edit:** added an **"Interview-answer grounding rule"** — an interview/staff claim may only reference things
taught or shown earlier in the same lesson; trace each claim to a taught moment, and if a claim has no anchor,
teach it or cut it.

---

## Deliberately NOT patched (documented, not a skill-text gap)

| Item | Why not patched |
|---|---|
| Unfilled `experiment.py` guided stubs | **Deliberate, contract-sanctioned** learn-by-building design (`m02_artifact_contract.md` §31: "intentionally NOT a filled reference solution"). Not a bug. |
| Dead `/frontier-experiment-lab` Option-B | Known **curriculum-wide** infra/uninstall P1 (~243 files); an install/tooling issue, not skill text. S2's new **dead-escape-hatch flag** makes QA *surface* it instead of silently waiving — the appropriate skill-level response. |
| Bilingual mid-lesson callouts | User-preference (English-primary); skills already default to light-touch. A P2 lesson nit, not systemic. |
| HTML-lesson vs Jupyter medium difference | Legitimate; m02 proves it can host live viz (D5/D7/D8 `gradient-descent.html`). S1 targets *where* static SVG is wrongly chosen, which is the real lever. |

---

## Verification

| Check | Result |
|---|---|
| S1–S4 anchors present in target skills | **all 6 grep anchors ✓** (S1; S2a/b/c; S3; S4) |
| Frontmatter well-formed (`---` fences = 2) | **all 4 files ✓** |
| `## ` section counts | vis-evidence 6 · refactor-qa 13 · architect 10 · lesson-builder 8 (all sane) |
| Installed `.claude/skills/frontier-*` == `frontier_lab_refactor_skills_v7_2/skills/*` | **all 4 IDENTICAL** (mirror synced) |
| Any m02 lesson/contract edited in Phase 3? | **No** — `git status` shows only the pre-existing v7.1 lesson mods, none new |
| Edits additive only? | **Yes** — each `Edit` wrapped the prior text and appended; no rule removed |

---

## Status

**Systemic gaps S1–S4 are now closed in the v7.2 skill set.** Because these were **skill-guidance** gaps (not
module-P0s), **m02 itself remains ✅ Pass with P1** — it was not, and per Phase-3 rules must not be, edited
here. The patched skills will, on the next module, require plotted/behavioral evidence where the concept is a
curve/derivative/boundary, force contracts to name capability limits + geometry, refuse to credit
"named ≠ shown," apply the held-out yardstick, flag dead auto-build paths, and ground interview answers.

**Recommendation:** with S1–S4 patched, **v7.2 is ready for next-module rollout.** Suggested (optional) next
step for the *architect*, not this validation pass: re-run the built m02 lessons through the newly hardened
visual-evidence gate to decide whether to add one live `viz/*.html` boundary/derivative embed on D1/D2 — that
is a *forward* module-improvement choice, outside this end-to-end validation's scope.
