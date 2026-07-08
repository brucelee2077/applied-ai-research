# Held-Out Evaluation — Module 2 vs the fundamentals notebooks

_Post-generation evaluation. The three held-out notebooks were **not** used during generation (the
refactor was source-free); they are read here only as reference. **No lesson file was edited.**_

**Compared:**
| Held-out reference (notebook) | Generated (refactored lesson) |
|---|---|
| `00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb` | folded into `m02-the-neuron/day-01-single-neuron/lesson.html` (hero + §1) |
| `00-neural-networks/fundamentals/02_single_neuron.ipynb` | `m02-the-neuron/day-01-single-neuron/lesson.html` |
| `00-neural-networks/fundamentals/03_activation_functions.ipynb` | `m02-the-neuron/day-02-activations/lesson.html` |

**Method:** 4-agent workflow — 2 independent paired judges (each read the notebooks + the matching
lesson and scored 9 dimensions with re-executed numbers), 1 **adversarial** critic (tasked to make the
strongest case *for the notebooks* to counter authorship bias), 1 skill-gap analyst (read the notebooks
+ the three skill files). I independently read all three notebooks to ground the synthesis.

---

## 0. The headline: different instruments for different readers

The notebooks and the lessons are **not** competing versions of the same artifact — they optimize for
different readers, which explains almost every win/loss below:

- **Notebooks** = a wide, runnable, pure-beginner reference (Reader A). Real matplotlib plots, six
  activations with derivatives, decision-boundary geometry, XOR, applied classification with confusion
  matrices, everyday analogies (brain, coffee, movie, spam). **No** staff lens, failure modes, interview
  answers, or frontier framing.
- **Lessons** = a narrow, deep, **interview-shaped** intro for a senior engineer new to ML (Reader A
  intuition + Reader B depth), inside a chained 9-day narrative with progress tracking, gates, and
  self-contained offline interactivity. They fold in named silent-failure modes, trade-offs, verbatim
  interview answers, and concrete frontier anchors (GPT-3 `(12288, 49152)`, SwiGLU/GELU).

Both judges + the adversarial reviewer independently reached the same split. **All numbers in both
artifacts verified correct by re-execution — no technical discrepancies.**

---

## 1. Scoreboard (both judges, per dimension, 1–5)

Verdict is for the **lesson** vs the notebook.

| # | Dimension | Day 1 (neuron)  nb / lesson | Day 2 (activations)  nb / lesson | Net |
|---|---|---|---|---|
| 1 | Intuition quality | 4 / 5 · **win** | 4 / 5 · **win** | **Lesson** |
| 2 | Analogy depth | 4 / 4 · tie | 3 / 4 · **win** | **Lesson** |
| 3 | Concept coverage | 5 / 4 · **lose** | 5 / 3 · **lose** | **Notebook** |
| 4 | Visual depth | 5 / 3 · **lose** | 4 / 4 · tie | **Notebook** |
| 5 | Experiment quality | 5 / 3 · **lose** | 5 / 3 · **lose** | **Notebook** |
| 6 | Beginner friendliness | 4 / 5 · **win** | 4 / 5 · **win** | **Lesson** |
| 7 | Technical correctness | 5 / 5 · tie | 5 / 5 · tie | **Tie** (both correct) |
| 8 | Frontier-lab relevance | 2 / 5 · **win** | 3 / 5 · **win** | **Lesson** |
| 9 | Artifact quality | 4 / 4 · tie | 4 / 5 · **win** | **Lesson** |

**Tally:** lessons lead **5** dimensions (intuition, analogy, beginner-friendliness, frontier relevance,
artifact), notebooks win **3** (concept coverage, visual depth, experiment quality), **1** genuine tie
(correctness). The notebooks' three wins cluster tightly around one theme — **"evidence you can see run."**

---

## 2. Dimension-by-dimension

### 1 · Intuition quality — Lesson wins
The lessons carry the load-bearing *why* the notebooks bury: "`w·x` alone can only **rotate** the line —
it's stuck through the origin; the bias **shifts** it off." And they anchor everything in one durable
chain ("neuron → matmul → layer → network → training tunes the weights; everything later is scale on
top"). The activations lesson frames the linear-collapse as a payoff ("the little bend that lets deep
networks learn"), sharper than the notebook's fact-first bullets.

### 2 · Analogy depth — Lesson (Day 2), tie (Day 1)
Notebooks have **more** analogies (coffee with worked arithmetic; three bias analogies: race head-start,
mood, credit-score threshold). Lessons use one tight analogy per concept but **always name where it
breaks** ("a real brain neuron is far messier; one neuron alone is weak") — the analogy-discipline
CLAUDE.md S5 requires and the notebooks omit. Net: notebooks win breadth, lessons win craft.

### 3 · Concept coverage — Notebook wins (both)
The notebooks are strictly broader: decision-boundary derivation to `x₂ = −(w₁/w₂)x₁ − b/w₂`, weight
rotation, bias-as-threshold, spam + imbalanced-medical classification, **XOR non-separability**, the
full input/hidden/output network + learning loop (nb01); and **six activations** with derivatives +
temperature + init tables (nb03). The lessons deliberately scope tighter (one neuron + a scale teaser;
ReLU/sigmoid only) — **but add symmetric-init + fan-in scaling, which the notebooks entirely lack.**

### 4 · Visual depth — Notebook wins (Day 1), tie (Day 2)
Notebook 02 renders ~8 real data-driven figures (contour decision-boundary heatmaps, a 6-panel
weight-rotation grid, spam scatter, XOR). The lessons' visuals are hand-built inline SVGs (6 static
scroll-reveal steps) + a simulated terminal — clean, progressive, self-assembling, but static and
low-data-density, with **no plotted boundary**. Day 2 is a tie (the notebook's function+derivative plots
vs the lesson's conceptual collapse→cure SVG sequence).

### 5 · Experiment quality — Notebook wins (both, decisively)
This is the notebooks' strongest, most defensible edge. They **execute and print evidence**: nb02's
`85.3× faster` loop-vs-vectorized benchmark and per-setting sensitivity/specificity; nb03's live
**naive-softmax overflow to `[nan nan nan]`** on `[1000,1001,1002]` beside the stable version returning
`[0.09, 0.245, 0.665]` (the adversarial reviewer re-ran it — it reproduces). The lessons' "playgrounds"
are hard-coded HTML strings — **nothing executes**; real code is deferred to a Produce artifact the
learner writes. Against the repo's own bar ("every cell produces output an interviewer could see"), the
notebooks win.

### 6 · Beginner friendliness — Lesson wins (both)
Both are beginner-first, but the lessons are better **scaffolded**: explicit prerequisites callout,
one crisp goal, section-by-section `.gotit` gating, "why this trips people up" normalize-confusion
callouts, hover tooltips, bilingual EN/中文 touch. The notebooks pile XOR + confusion matrices +
sensitivity/specificity into a nominal *intro* — heavier load for a true novice.

### 7 · Technical correctness — Tie (both correct)
Re-executed by two independent judges + the adversarial critic. Lesson: `x=[2,3], w=[0.5,-1], b=1` →
`−2 → −1 → ReLU → 0.0`; positive case `w=[1,1]` → `5.0`; anchor pair `W1@W2=[[7,2],[3,1]]`,
`(x@W1)@W2 = x@(W1@W2) = [13,4]`; `sigmoid(2)=0.8808` (lesson's `0.119` for `−2` is correctly rounded).
Notebook algebra + `vectorized==loop` check also correct. **No numeric disagreement.** One benign
*definitional* divergence (see §4 below).

### 8 · Frontier-lab relevance — Lesson wins (both, large margin)
The notebooks stay at hobbyist/intro altitude (`2/5`, `3/5`). The lessons deliver staff signal the
notebooks entirely lack: named **silent failure modes** ("symmetric init → a 512-unit layer has the
power of one; catch it by spotting `np.zeros` / printing row-variance"; dead/saturated ReLU → "log
activation statistics"), real **trade-offs** (fan-in init, ReLU vs smooth GELU as a design-review call),
**verbatim interview answers**, and concrete frontier anchors (GPT-3 matrix, SwiGLU/GELU).

### 9 · Artifact quality — Lesson (Day 2), tie (Day 1)
Different classes. Notebooks: polished self-contained `.ipynb` with rendered outputs — **but** two nb02
exercise cells contain literal `?` placeholders that `SyntaxError` if run straight through, and (per the
skill-gap analyst) nb03 **has no COACH start/end cells**, so it would fail this repo's notebook
validation. Lessons: production-grade offline HTML (4 themes, localStorage, scrollspy, tooltip
controller, viz-iframe autoresize, wired nav). As a finished deliverable, the lesson edges it.

---

## 3. What each side teaches that the other misses

**Lessons teach, notebooks miss:** symmetric-initialization failure + detection; fan-in init scale
(Xavier/He); activation folded into the neuron from step 1 + the term *pre-activation*; the
scale-to-billions chain + GPT-3 anchor; a drop-in interview answer; the **matrix-form** linear-collapse
proof (scalars → anchor pair `[13,4]`); dead/saturated units as a *silent* failure with the
log-activation-statistics habit; named frontier activations (SwiGLU/GELU) as design choices; an
active-recall Produce artifact + research log.

**Notebooks teach, lessons miss:** plotted decision boundaries + the `x₂ = −(w₁/w₂)x₁ − b/w₂`
derivation; the **XOR** non-separability limit (concrete, plotted); the loop-vs-vectorized `~85×`
benchmark; applied classification with confusion matrix / sensitivity / specificity; the whole-network
orienting diagram + plain-words learning loop (nb01); every activation's **derivative** + the
`sigmoid' peaks at 0.25 vs tanh 1.0` vanishing-gradient number; **live softmax numerical-stability**
demo; softmax temperature scaling; a weight-init-to-activation pairing table + decision tree +
debugging checklist.

---

## 4. Honesty check — where the lessons overclaim or oversimplify (from the adversarial pass)

These are **not** errors, but places a critical reader (or interviewer) could push back. Recorded for
transparency; **not edited** per instruction.

1. **Day 2 "ReLU dodges vanishing gradients"** is asserted while gradients are deferred to Day 5 — a
   Day-2-only reader accepts the central "why ReLU won" claim on faith. The notebook instead *shows*
   `sigmoid' → 0.25` vs `tanh' → 1.0`. (This was a deliberate sequencing choice per the coverage
   contract, but it does leave the claim unbacked at Day 2.)
2. **Neuron = `activation(w·x+b)` from step 1** — the lessons never flag this is a *scope choice*; a
   regression/output unit has identity activation. A one-line note ("the third step can be identity")
   would prevent over-generalizing "a neuron always has a ReLU."
3. **Day 1 symmetric-init "gets the same update"** quietly depends on gradients not yet met (Day 5). A
   good staff point, but its mechanism is asserted before the reader has any notion of a gradient.
4. **Day 2 "approximate almost any function"** (universal approximation) is dropped as a hedged aside
   with no conditions — a Reader-B risk of repeating it as an unqualified soundbite.
5. **Simulated playgrounds** present pre-baked output styled as a live REPL — faithful to what the code
   *would* print (verified), but a learner could believe code ran when nothing did.
6. **Day 2 "ReLU/sigmoid are the two you'll see most"** slightly undersells that the frontier models it
   *cites as motivation* (Llama/GPT) actually run GELU/SwiGLU, not ReLU/sigmoid.

---

## 5. What the skills failed to specify (dimension 10 — the highest-value finding)

**Meta-insight (from the skill-gap analyst, and it holds up):** the m02 **contracts already captured**
nearly all the notebook content the lessons miss (decision-boundary geometry, XOR, MSE↔CE table,
softmax stability, temperature). So the gap is **not** the architect mis-scoping m02 — it is that the
underlying **skills never encode the generalizable *teaching moves*** that produced those notebook
features. On a *different* topic where a contract author didn't happen to enumerate each item, the
generator would not reliably reproduce them. Fix the skills, not just this module.

Eleven proposed skill patches, grouped by owner:

### → `frontier-lesson-builder` (6)
1. **Step-traced worked examples** — require the "Tiny numbers" ladder step to print *every*
   intermediate value on its own line (running total per term), never just the final answer.
2. **Derivative-alongside-function** — when teaching a function that will later be trained (activation,
   loss, kernel), require its derivative formula + a function-vs-derivative visual + the peak-slope
   number that motivates the next lesson (`sigmoid 0.25 vs tanh 1.0`). Don't defer *all* gradient content
   when a one-number preview lands the trade-off now.
3. **Show the failure executing** — a taught failure mode (overflow, dead ReLU, leakage, divergence)
   must show the broken output running (e.g. `[nan nan nan]`) beside the fix, not just name it.
4. **Naive-vs-idiomatic benchmark** — when an idiomatic form exists (vectorized vs loop, fused vs
   unfused), include a runnable benchmark printing the measured cost gap (`~85×`), not an assertion.
5. **Applied consequence step** — one worked scenario carrying the mechanism into *evaluation*
   vocabulary (confusion matrix, FP/FN, sensitivity/specificity), each term Jargon-Busted on first use.
6. **Decision tree + debugging checklist + pairing table** — required for any lesson comparing 3+
   interchangeable options (which to choose, symptom→cause→fix, companion setting e.g. activation→init).

### → `frontier-visual-builder` (3)
7. **Parameter-isolation small multiples** — for 2+ interacting params, a fixed grid varying ONE while
   freezing the others (sweep `w₁`, hold `w₂,b`), each panel annotated with the value + a metric —
   alongside (not instead of) the single slider.
8. **On-panel numeric readout** — every quantitative visual must print the derived quantity it teaches
   (slope, angle, intercept, accuracy, entropy) on the panel; a shape that moves without surfacing the
   number is half-built.
9. **Decision-boundary triptych** — for any linear classifier: (a) contour output field, (b) the
   `output=0` boundary, (c) the same line as `x₂ = −(w₁/w₂)x₁ − b/w₂` with slope/intercept labeled;
   controls that rotate (weights) and translate (bias) it.

### → `frontier-curriculum-architect` (2)
10. **Whole-system-first + the learning loop** — foundation modules must show a full-system orienting
    diagram *before* the first deep-dive, and state the plain-words end-to-end loop (init → predict →
    measure → adjust → repeat) early, referenced as pieces fill in.
11. **Reader-objection Q&A** — the blueprint must enumerate and *inline-answer* the top "why do we even
    need X / can't we just do Y" objections (bias-as-extra-input equivalence; why not stack linear
    layers), distinct from the misconception list.

---

## 6. Recommendations

**Primary (skills, not lessons):** adopt the 11 patches in §5 — especially #3 (show-the-failure) and
#9 (decision-boundary triptych), which close the two most consequential losses (experiment quality,
visual depth) across the whole curriculum, not just m02.

**Optional lesson enhancements (NOT applied — flagged for a future pass, since lessons must not be
edited here):**
- **P1 · Day 1** — add an embedded interactive **decision-boundary** viz (reuse the file's existing
  viz-iframe autoresize pattern) + the one-line `x₂ = −(w₁/w₂)x₁ − b/w₂` algebra as a Math-Ladder step,
  so "bias shifts the line" is *derived and seen*, not asserted. (Note: Day 1 geometry was
  *deliberately* kept a light preview per the coverage-contract's own "don't overload Day 1" note — this
  is a scoping trade-off, not an oversight.)
- **P1 · Day 2** — make the "log the fraction of dead ReLUs" staff habit a **runnable Produce acceptance
  criterion**, and borrow the live naive-softmax overflow demo, so the staff advice is backed by
  evidence instead of a slogan. Add a compact XOR "one line can't split this" visual + an activation
  cheat-table (range / peak-derivative / use-case) since the lesson already name-drops GELU/SwiGLU.
- **P2** — one-line honesty notes for the §4 overclaims (activation-can-be-identity; "loosely" on
  universal approximation; that the cited frontier models use GELU/SwiGLU).

---

## 7. Verdict

The **source-free generation held up well against the reference it never saw.** For a first pass and
for interview preparation, the lessons teach *better* (intuition, scaffolding, frontier signal, artifact
polish) and are **technically correct throughout** (no discrepancies on re-execution). The notebooks
remain the better place to **see the ideas run and move** — plotted decision boundaries, executed
failure demos, and reference breadth — which is exactly the cluster (experiment quality, visual depth,
concept coverage) where the lessons' *format* (static SVG + simulated terminal + deferred Produce)
structurally caps them. The most leverage is at the **skill layer**: encoding the notebooks' teaching
moves (§5) would let the generator reproduce those strengths on any topic. **No lesson files were
edited.**
