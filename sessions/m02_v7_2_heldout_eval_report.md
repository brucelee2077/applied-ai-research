# Held-Out Eval Report — Module 2 vs Fundamentals Notebooks (v7.2)

> **Purpose:** stress-test the **v7.2 refactor SYSTEM** (not just m02) against an external gold-standard
> yardstick the module authors did **not** see during QA. Per `/frontier-refactor-qa`: *"Held-out eval must
> include adversarial sections and must not over-credit Staff Lens or frontier relevance when beginner
> coverage, intuition, visuals, or runnable experiments are weak."*
> **This phase edited NO lesson files** (read-only, as required).
> **Held-out references (gold standard):**
> - `00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb` (13 cells; 1 real matplotlib plot)
> - `00-neural-networks/fundamentals/02_single_neuron.ipynb` (51 cells; **16 code cells render real plots**)
> - `00-neural-networks/fundamentals/03_activation_functions.ipynb` (41 cells; ~15 executed code cells)
> **Modules under test:** m02 D1 (single neuron), D2 (activations), D3 (forward pass).
> **Method:** 3 read-only adversarial subagents (one per reference↔lesson mapping), integrated + adjudicated.

---

## Headline

The v7.2 module is **stronger than the reference on conceptual depth** — brain→math mapping table,
scalars-first linear-collapse derivation, pre-activation/logits naming, symmetry-break + dead-ReLU staff-lens,
worked numeric anchors, interview framing. **The reference has none of that.**

But on the four axes this eval exists to protect — **runnable evidence, plotted behavior, a topic's defining
limits, and honest claim-grounding** — a plain beginner notebook **beats** the polished module. All three
agents converged, independently, on the same root cause: **v7.2 substitutes static SVGs + simulated "fake
terminals" + prose tables + guided-stub artifacts for the reference's real, executed, plotted evidence** — and
the QA gate cannot catch it because the gate is calibrated to the very contracts that chose those surfaces.

This is the "over-credited polish" failure mode. **Systemic skill gaps exist → proceed to Phase 3.**

---

## 1. Overclaiming

| # | Where | Claim | Reality | Tag |
|---|---|---|---|---|
| O1 | D1 hero/Produce (`:335`,`:463`) "compute a neuron by hand / building it is what makes the day count" | implies shipped runnable evidence | `experiment.py` is a 5-line placeholder; the reference ships executed code | module-P1 (see adjudication) |
| O2 | D1 interview line (`:436`) learner recites "a bias that **shifts the decision boundary** off the origin" | asserts a geometric picture | the word "boundary" appears **only** in that scripted line; no boundary is ever drawn/derived. Reference derives + plots it (cells 13–17) | **systemic** (lesson-builder grounding) |
| O3 | D3 §4 + quiz Q4 + interview: "`(xW1)W2 = x(W1W2)` — stacked linears collapse" | taught in 3 places | **shown in 0** — no numeric with-vs-without-ReLU run in D3 (D2 shows the collapse; D3 only asserts it) | **systemic** (show-the-failure) |
| O4 | D2 Produce Option B (`:460`) skill "creates the file, writes the code, and **runs it for you**" | promises auto-run evidence | `/frontier-experiment-lab` is **uninstalled**; the path is dead | known curriculum-wide P1 |

The universal-approximation hand-wave (D2 `:411`, hedged "loosely") and the deferred symmetry-init note (D1,
honestly hedged to D3/D5) are **acceptable**, not overclaim — checked explicitly.

## 2. Beginner intuition — **m02 wins (credit given, does not offset §4–§5)**

- m02's **biological→artificial mapping table** (D1 `:370`) is a cleaner first-contact than the reference's
  voter/movie hand-waving, and m02 states *where the analogy breaks* (D1 `:389`); the reference never does.
- m02's **scalars-first collapse** (D2 Math Ladder `:412`, `g(f(x))=(ab)x+(bp+q)` before the matrix form) is a
  genuine pedagogical win the reference lacks (the reference shows collapse numerically but not *why*).
- Fair counter: the reference builds one intuition m02 D1/D3 defer — **the learning loop** as a named
  5-step process (ref cell 8). m02 legitimately defers this to D4–D6 (acceptable scope split).
- One real regression: mid-lesson **Chinese-English bilingual callouts** (D1 `:358`,`:390`,`:488`) drop into an
  otherwise all-English lesson — against the user's stated English-only preference — which *raises* the reading
  cliff for a first-time non-native-English reader. (module-P2; lesson-builder default.)

## 3. Coverage — concept-by-concept vs the reference

**m02 bonuses the reference lacks (real added depth):** `output=activation(w·x+b)` with worked numbers;
pre-activation `z`; logits/inference/backprop naming; the full `3→4→2` shape chain + crash condition;
symmetry-break + dead-ReLU + fan-in staff-lens; frontier framing (GPT-3 matrix, GELU/SwiGLU up front).

**Reference teaches; m02 omits or under-covers:**
| Concept | Reference | m02 | Adjudication |
|---|---|---|---|
| **Decision-boundary geometry** (`w`=orientation, `b`=shift/threshold) | derived + plotted (02: cells 13–17, 27–31, 36) | **prose only** (D1 `:421` "shifts that line"); no picture | **systemic** — contract listed it must-cover but QA credited prose |
| **XOR / linear-separability** (single neuron's defining *limit*) | plotted (02: cells 40–41) | **absent** (both files) | Should-cover gap → **systemic** (architect: contracts omit *limits*) |
| **Plotted derivatives** of sigmoid/tanh/ReLU | every curve + derivative (03: cells 12,17,22,27,37) | **no derivatives**; a prose "slope near 0 ≈0.25" table row | **systemic** (visual-evidence) |
| Leaky ReLU / softmax | full formula + code + plot (03) | **named only** (D2 `:419`,`:425`) | "names-it ≠ covers-it" → **systemic** (QA) |
| loop-vs-vectorized + speed benchmark | 3 impls + timing (02: cell 10) | absent | module-P2 |
| beginner "common misconceptions" sweep | ref cell 10 | absent (only staff-lens warnings) | module-P2 |
| Step/ELU/PReLU, temperature, He/Xavier guide | covered (03) | absent | **acceptable-scope** (beginner day vs full notebook) |

**No P0 coverage omission** (m02's scope split is legitimate), but two *defining* facts of these exact topics —
the neuron's **geometry/boundary** and its **XOR limit**, and the activations' **derivative shapes** — are
present in the reference and thin/absent in m02.

## 4. Visual / Evidence — **the sharpest, most systemic shortfall**

- **Reference:** real, executed matplotlib — a rendered network (01), a 16-plot geometry suite incl.
  weight-rotation grid, bias-threshold plots, weight-vector quiver, XOR scatter (02), and every activation
  curve **with its derivative** + a 2×5 comparison grid printing `Sigmoid max=0.25 / ReLU grad=1` (03).
  These are computed from data and change when parameters change.
- **m02 D1/D2/D3:** **zero executed plots.** Every "run" is either a **simulated fake terminal** (outputs like
  `-2.0`, `array([0,0,0,2,5])` are literal HTML strings in `DEMOS`, not computed) or a **static inline SVG** with
  numbers baked in. Saturation, vanishing gradient, and dead-ReLU — whose entire substance is *a derivative
  going flat* — are conveyed in **prose + a table cell**, never a plotted curve.
- **Dead infra:** D1 even carries the `.build-embed iframe` CSS + postMessage auto-resize handler for a live
  embedded viz — with **no `<iframe>` in the file.** The mechanism to host a live boundary/curve viz exists but
  is unused. (m02 *does* use it correctly on D5/D7/D8 for `gradient-descent.html` — so the capability is proven;
  it's just not applied where the reference plots.)
- By the visual-evidence skill's **own** "simulated-vs-runnable rule," a simulated terminal counts as evidence
  only if the Produce artifact reproduces it — and the artifact is a stub — so the simulated playgrounds **do
  not qualify as evidence** under the skill's own definition, yet they are the primary "run it" surface.

## 5. Artifacts

- **Reference:** self-contained, fully executed — ~15 runnable cells per notebook, incl. a numerically-stable
  softmax (overflow-vs-stable side-by-side) and a loop-vs-vectorized speed benchmark. Meets the CLAUDE.md
  "every cell produces output an interviewer could see" bar.
- **m02:** the only artifact per day is the Produce `experiment.py`, shipped as a **5-line placeholder**, plus an
  empty `log.md`. Acceptance criteria are well-written and numerically **correct** (Phase-1 agents recomputed
  D5/D6/D7/D8 anchors and confirmed them), but **nothing runnable ships**, and Option B routes through the
  uninstalled `/frontier-experiment-lab`.

**Adjudication (fairness check):** the unfilled `experiment.py` is a **deliberate, contract-sanctioned design**
(`m02_artifact_contract.md` §31: *"intentionally NOT a filled reference solution — the learner produces it"*).
So "the stub is empty" is **not** itself a skill defect — it is a learn-by-building choice. The *genuine*
surviving finding is narrower and real: the module's only "auto-build + run" path is **dead** (uninstalled
skill), and the learn-by-building design ships **no** executed evidence on topics where a canonical treatment
runs everything. That is a **known curriculum-wide P1** (the dead skill) plus a **design-level tradeoff**, not a
new v7.2 skill-text bug — see the Skills-gap adjudication below.

## 6. Skills gaps (SYSTEMIC) — adjudicated

The three agents proposed several skill patches. Separating **genuine, generalizable v7.2 skill-text gaps**
(→ Phase 3) from **deliberate design / known items** (→ document, do not patch):

### ✅ Genuine systemic skill gaps → patch in Phase 3
| # | Skill | Missing/weak rule | Evidence (convergent across agents) |
|---|---|---|---|
| S1 | **frontier-visual-evidence-builder** | The "changing quantity" / "simulated-vs-runnable" rule does not distinguish **behavioral** concepts (a curve/derivative/boundary/gradient *behaving*) from **structural** diagrams. For behavioral concepts a static SVG with baked-in numbers or a prose "slope≈0.25" table must **not** satisfy the P0 visual — it needs a plotted or live-recomputed viz that shows the quantity move. | No plotted derivatives on D2; boundary geometry prose-only on D1; saturation/dead-ReLU prose-only |
| S2 | **frontier-refactor-qa** | Gate credits **"names it" as "covers it"** and credits **prose assertion** for a visual/behavioral claim. Needs: (a) a concept named in a callout but never shown (formula+curve or run) = coverage gap; (b) a derivative/geometry/failure claim with no plotted-or-run evidence = visual gap; (c) an explicit **held-out-yardstick** step ("would a canonical notebook *show* what this lesson only *tells*?"). | QA passed D1/D2/D3 clean despite S1/S3/S4 |
| S3 | **frontier-curriculum-architect** | Coverage contracts enumerate forward formulas but not a topic's **capability LIMITS** (single neuron → XOR) or **geometric interpretation** (decision boundary). Contracts must require both. | XOR absent; boundary geometry thin |
| S4 | **frontier-lesson-builder** | No **interview-answer grounding** rule: a scripted interview/staff claim may assert a picture or result the lesson never taught/showed. | D1 `:436` "shifts the decision boundary" — no boundary in the lesson |

### ➖ NOT skill-text gaps → documented, not patched
- **Unfilled `experiment.py` stub** — deliberate, contract-sanctioned learn-by-building design (§31). Not a bug.
- **Dead `/frontier-experiment-lab` Option-B path** — known curriculum-wide P1 (~243 files), already logged;
  it is an *infra/uninstall* issue, not skill text. (Phase-3 will add a QA **flag** for it, not a fix.)
- **Bilingual callouts** — user-preference item; skills already default to light-touch; a P2 lesson nit.
- **Medium difference** (HTML interactive lesson vs auto-executed Jupyter) — legitimate; m02 proves it can host
  live viz (D5/D7/D8). The gap is *where* it chooses static SVG, which S1 addresses.

---

## Adversarial self-check (did this eval over- or under-credit?)
- **Did not** let m02's staff-lens/frontier polish paper over the evidence gap — the polish is credited only in
  §2/§3-bonus and explicitly does **not** offset §4/§5.
- **Did not** inflate deliberate design into a defect — the empty stub and dead-skill path were down-graded to
  design/known-P1 after checking the artifact contract.
- **Did** confirm m02's numbers are correct (not fabricated) — so the gap is *evidence-medium*, not *accuracy*.

## Conclusion

Held-out eval **found systemic skill gaps** (S1–S4) that generalize beyond m02 and would recur in the next
module. They are the reason a plain beginner notebook out-evidences a polished v7.2 module on runnable/plotted
proof. **Proceed to Phase 3: patch `.claude/skills/frontier-*` for S1–S4** (no lesson edits). m02 itself remains
**Pass with P1** — none of S1–S4 is a module-P0; they are systemic authoring-guidance gaps, and fixing the
*skills* (not this module's lessons) is the correct lever.
