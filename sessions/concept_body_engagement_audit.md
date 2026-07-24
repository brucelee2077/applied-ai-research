# Concept-Body Engagement Audit (m02 / m03) — 2026-07-24

**Question:** intros are engaging now; why do the concept *bodies* (the build-up:
mechanism, math, worked examples) still read as "hard to digest, tedious, boring"?

**Method:** 4 parallel read-only readers examined (1) the LLM judge suite, (2) the
lesson-builder skill, (3) the widget toolkit + donor, (4) real m02/m03 concept bodies.

## Root cause — the "engagement cliff"

Every engagement device is prescribed + enforced for the **opening only**. The moment
the **build-up** begins, the skill gives only *structural* rules and the judges only
certify *structure* — nothing grades whether the body PROSE is engaging.

### 1. Judges (coverage_judge.py) — no body-voice axis

- Interest judges (relative + absolute) score **whole-lesson** momentum/payoff/delight
  → a brilliant hero carries a tedious body. `momentum` says "not a dense wall" but is
  never decomposed per concept-body.
- Tone judge is whole-lesson vs the notebook → warm intro + dense body can still score
  MATCHES.
- Concept-Structure judge grades `intuition_first` / `analogy` / `buildup` /
  `buildup_visualized` — all certify the build-up is *present / followable / visualized*.
  **No axis grades body voice, re-hook, narration quality, or discovery.**

### 2. Builder skill — silent on body engagement

- "Beginner Intuition Register" = 5 named beats for the concept OPENING (plain-words
  what-is-it, 4-beat analogy scaffold, why-it-matters, mechanism, when-to-use) + warmth
  rules. **Ends there.** Zero beats for INSIDE the build-up.
- "Visualize the Build-Up" / "Math restraint" / CLAUDE.md §6 "Breaking down hard
  equations" are all STRUCTURAL (where the 2nd visual goes, keep math to one line, show
  dimensions, one idea per equation). None address body voice, pacing, re-hook,
  narrative connectors, predict-then-reveal, or keeping the spine analogy alive.

### 3. Widget toolkit — thin for bodies

- Body widgets: `mathladder` (static 4-rung dump, all rungs at once), `demo`
  (one-shot binary reveal), `svg`, `viz`, `hint` (post-concept). Raw prose between
  widgets → flat `<p>` with no pacing/disclosure.
- **Dead scaffolding:** `.build` / `.build-step` / `.build.armed` scroll-reveal CSS
  exists in the donor; `if(window.__revealBuild) window.__revealBuild()` is called on
  nav-click but `__revealBuild` **is never defined** and nothing emits `.build`.
- No widget for: predict-then-reveal, inline "why this matters" callout, narrated
  stepped worked-example.

### 4. Real bodies — the ground truth (verbatim)

- **m02 d05 (backprop) c1:** after the hiker/fog intro → *"Each knob inside the neuron
  is a weight… Collect that answer for every weight and you get the gradient."* 3 flat
  mechanism sentences, analogy dropped.
- **m02 d02 (activations) c1:** ruler intro → *"Lay one straight ruler on top of another
  and you still get…a single straight ruler."* → *"Engineers have a name for this
  fold-back — the linear collapse."* Mechanism enumeration, no re-engagement.
- **m03 d03 (attention) c2:** warm taste-survey analogy → immediately a cold formula box
  *"score = Q · K = (q₁·k₁)+(q₂·k₂)+…"* with no narrative bridge.
- **Pattern:** after the warm intro, register switches to flat textbook; spine analogy
  disappears once mechanism starts; no "why this bites" beat inside the body; 3–5 dense
  blocks run before any re-engagement.

## Fix (see the design doc)

Body-side twin of the interest-floor fix, in four layers: a **Build-Up Register** in the
builder skill, a per-concept **`body_engagement` judge axis** gated as a hard floor, a
**body toolkit** (`%%% insight`, predict-then-reveal `%%% demo`, `%%% steps` + wire the
dead scroll-reveal), and a **full author-loop rebuild** of m02 (9) + m03 (5).

Design: `docs/superpowers/specs/2026-07-24-concept-body-engagement-design.md`
