# Module 2 — Refactor Plan

> Ordered plan to bring `sessions/m02-the-neuron` to notebook-quality against the four contracts beside
> this file. **No lesson file is edited until all five contract files exist** (they do). The current
> lessons are strong drafts (audit depth 4–5, full Coach Layer) — this is an **elevate + fix**, not a
> rewrite. Preserve the shell, quest-ids, nav, and every interaction behavior.

---

## Orchestration (how the edits are done)

- **Sequential, single-voice, main-thread.** Per repo CLAUDE.md ("one file per session — no
  parallelism; no subagents for parallel writing — tone/analogy drift"), lesson `.html` edits are made
  **one file at a time in dependency order (Day 1 → Day 9 → gates → experiment.py)**, by the main
  session, not fanned out to parallel writer agents. The analogy chain is shared across days, so voice
  coherence is protected by writing them in order against the locked contracts.
- **Workflows only for read-only work** (already done: audit + design) **and for QA verification**
  (`/frontier-refactor-qa` at the end; parallel read-only checks are fine).
- **Edit style:** surgical `Edit`s against the existing files (the shell/CSS/JS scaffolding is kept
  verbatim). Only the content payload changes — hero/goal, the 7 section bodies, `DEMOS`, `BUILD`, `QS`,
  Produce, and labels. No structural HTML/JS rewrites.

## Frozen invariants (must survive every edit)

1. **No `data-quest-id` changes.** `wf2-d01…d03`, `wf3-d01…d06`, `wf2-review`, `wf3-review` are
   localStorage + hub-pill + "First Neuron" badge keys. Untouchable.
2. **Shell/CSS/JS engine unchanged** — sidebar, 4-theme switcher (dim default, pre-paint script),
   progress bar, checklist, scrollspy, `.gotit` gating, tooltip controller, viz-iframe autoresize,
   review-gate `done.verdict` engine.
3. **7 sections, same `data-sec` keys, `count` stays "0/7".** Review gates stay 3-section
   (check/quiz/verdict).
4. **Nav chain unchanged** — every `prev`/`next`/hub href stays exactly as wired today.
5. **Offline/self-contained** — no new external dependencies; SVGs inline; embeds are local `viz/*.html`.

---

## Per-file change lists

### Global (applies to every one of Days 4–9 + review.html)
- Relabel all learner-facing module text to **"Module 2 · Day N"** (N = folder day number 1–9):
  `<title>`, `.kicker`, `.brand-sub` ("Foundations · M2 Day N"), `.nav-group-label` ("Module 02 · Train"),
  the `.fin` completion banner, and the Produce Option-B prompt header ("my Module 2 Day N artifact").
  **Quest-ids stay.**
- Fix intra-module cross-references that call an *earlier day of this module* "M2"/"M3" → "Day N
  (earlier this module)". Keep genuine "M1 Day 4/5/1" references.

### Day 1 — `day-01-single-neuron/lesson.html` (`wf2-d01-neuron`)
- Labels already correct (Module 2). Keep the judge analogy.
- **Coverage adds:** (a) bias-is-not-optional line tied to geometry (`b` shifts the boundary; without it
  the line is forced through the origin); (b) reframe the **staff-lens silent failure to symmetric
  initialization** (all weights equal ⇒ identical gradients ⇒ a whole layer acts as one unit) with the
  **init-scale (fan-in)** trade-off — this is the module's biggest coverage hole and Day 1 owns half of it.
  Move the current dead-ReLU material to Day 2 (its natural home).
- **Fixes:** build step 6 (width 2→3 jump) gets a one-line note that the layer example uses a wider
  input; unify `np.maximum` in the Produce Option-A snippet (drop built-in `max`).
- Update quiz Q4 (diagnostic) to the symmetric-init failure if the dead-ReLU Q moves to Day 2.

### Day 2 — `day-02-activations/lesson.html` (`wf2-d02-activations`)
- Keep valve/dimmer analogy.
- **Coverage adds:** show the collapse with **scalars first** (`a(bx+c)+d = (ab)x + (ad+c)`, still linear)
  *before* the 2×2 matrix version (matrix–matrix multiply isn't formally introduced until Day 3); frame
  saturation as "the curve goes flat" (visual), deferring the formal derivative to Day 5; **dead ReLU**
  becomes this day's staff-lens silent failure; add one-line Leaky-ReLU/GELU + perceptron/XOR mentions.
- **Fixes:** carry **one anchor matrix pair** through playground + Math Ladder + Produce (kill the
  3-way drift); mark sigmoid values approximate (`≈`/`~`).

### Day 3 — `day-03-layers-forward-pass/lesson.html` (`wf2-d03-forward`)
- Keep assembly-line analogy. Labels correct (Module 2).
- **Coverage adds:** define **logits / "raw scores"** where the last layer skips activation; add a
  single **concrete numeric** end-to-end forward pass (not shapes-only); staff-lens = missing
  nonlinearity + silent broadcasting; pin **one weight-orientation convention** (`x@W` with `W:(in,out)`)
  and hold it through Days 5–7. (Init-scale reinforcement can live here or Day 6 — keep it on Day 6 to
  avoid overloading Day 3.)
- **Fixes:** clean the Produce Option-B prompt authoring artifact ("W2 (4,4)... use W2 (4,2)" → "W2 (4,2),
  b2 (2,)").

### Day 4 — `day-04-loss/lesson.html` (`wf3-d01-loss`)  ← relabel M3→M2
- Keep golf analogy; add the **loss-landscape handoff** at the end (sets up Day 5 terrain picture).
- **Coverage adds:** an **MSE ↔ CE comparison table**; a **MSE Math Ladder** parallel to the CE one;
  the general multiclass CE sum form `−Σ yₖ log pₖ`; state **loss = MEAN over the batch**;
  promote **logits→softmax→prob + log-sum-exp** to a should-cover callout (why the loss takes logits);
  one-line note that softmax outputs are not calibrated confidences. Staff-lens = double-softmax /
  MSE-on-classification (keep, it's already the diagnostic quiz).
- **Fixes:** make CE ordering consistent between playground and acceptance criteria (`0.3→0.6→0.9` ⇒
  `1.204→0.511→0.105` everywhere).

### Day 5 — `day-05-gradients-backprop/lesson.html` (`wf3-d02-backprop`)  ← relabel M3→M2
- Keep hiker-in-fog analogy — **the single gradient picture** (terrain). Do not run a second metaphor.
- **Coverage adds:** explicitly **name "gradient descent"** as the baseline algorithm (`w ← w − η·dL/dw`,
  repeated) so Day 7 reads as "what optimizers ADD"; a **real numeric multi-link chain-rule** worked
  example (`dL/dw = dL/da·da/dz·dz/dw` with numbers), noting it's a from-scratch illustration; establish
  **non-convexity** (the hill is bumpy — we reach *a* good-enough basin, not the global bottom) and
  correct the "GD finds the global min" misconception; **own exploding/vanishing gradients** (so Day 8's
  clipping has a referent). Staff-lens = missing `zero_grad` (accumulation) + vanishing/exploding.
- **Fixes:** pick **one** forward-pass cost figure (**~3×**, drop "2–3x"); reword "M2 (forward pass)"
  cross-ref → "Day 3"; if the worked chain uses sigmoid+MSE, flag it's an illustration, not a
  recommended pairing.

### Day 6 — `day-06-training-loop/lesson.html` (`wf3-d03-training-loop`)  ← relabel M3→M2
- Keep free-throws analogy.
- **Coverage adds:** add **`zero_grad`** as an explicit step in the loop skeleton (forward→loss→backward→
  zero→update); a **worked epoch/batch count** (e.g. 100 examples / batch 25 = 4 steps/epoch); state the
  batch gradient is the **mean of per-example gradients** → a noisy **unbiased** estimate (earns the
  claim); a **random-init scale** callout (fan-in intuition — the second half of the init hole); note the
  loop should **track a held-out (val) loss** (forward-pointer to Day 9). Staff-lens = eval-mode/no_grad +
  loss ≠ reported metric. Add the **batch-size trade-off table**.
- **Fixes:** loss `0.747` → **`0.746`** (playground + anywhere it appears); **remove premature-close
  copy** ("Foundations done", "completes the how-a-model-learns story") — Days 7–9 remain; keep the
  banner's correct "Next up: Optimizers".

### Day 7 — `day-07-optimizers/lesson.html` (`wf3-d04-optimizers`)  ← relabel M2-3→M2 (Day 7)
- Keep ball-downhill analogy. Open with a one-line callback: "these all sit on the gradient-descent
  baseline from Day 5 — each just changes the hand that turns the fader."
- **Coverage adds:** give Adam's `β₂=0.999`, `ε=1e-8` defaults; a one-line "N weights → 2N optimizer
  numbers" to make the 2× state concrete; AdaGrad/RMSProp→Adam lineage (one line). Staff-lens = AdamW
  weight-decay coupling (keep).
- **Fixes:** momentum **step 2 must use the real continuation** at `w=1.8` (`g=−9.6`, `v=0.9·(−16)+(−9.6)=
  −24.0`, move `1.2` vs SGD `0.48`) so the playground, prose, and the Produce loop agree — remove the
  hypothetical `−12.8`/`−27.2`; tighten the "2× SGD memory" phrasing to "2 extra buffers per parameter".

### Day 8 — `day-08-learning-rate/lesson.html` (`wf3-d05-lr`)  ← relabel M2-3→M2 (Day 8)
- Keep stepping-stones analogy.
- **Coverage adds:** a one-line derivation/justification that GD on a quadratic gives the `(1−lr·c)`
  recurrence (`c` = curvature = 2nd derivative), tipping point `lr = 2/c`; input-scaling/ill-conditioning
  ↔ LR should-cover note; LR–batch linear-scaling one-liner; note `c` generalizes to a spectrum in nD.
  Staff-lens = LR-too-high **silent** high plateau (not always a clean NaN).
- **Fixes:** align the playground step count with the Produce step count (one N); add a one-line
  **symbol disambiguation** (`c` curvature here vs `v` momentum-velocity on Day 7).

### Day 9 — `day-09-train-val-test/lesson.html` (`wf3-d06-split`)  ← relabel M2-3→M2 (Day 9)
- Keep exam analogy. This is the capstone before the final gate.
- **Coverage adds:** tie overfitting to **model capacity/complexity** (the cause), so the degree-sweep
  Produce artifact rests on a taught idea; one-line mechanisms for L2 (smaller weights → smoother
  function) and dropout (≈ ensemble of subnetworks); data-leakage should-cover (already the diagnostic
  quiz) + k-fold one-liner. Staff-lens = data leakage (keep).
- **Fixes:** correct the internal contradiction — §4 + Math Ladder must say the best model is at **step
  200** (the val minimum), matching the playground early-stopping demo (not "near step 100"); make the
  **playground toy and the Produce toy the same framing** (choose one: either the step-vs-loss curve or
  the degree-sweep, and use it in both).

### `review-part-a.html` (`wf2-review`, mid-module gate after Day 3)
- Self-label already "Module 2" ✅. **Fix forward copy** only: "The Gate Before Module 3" → "Halfway Gate —
  before the training half of Module 2"; "M3 · Loss, Gradients & Backpropagation" / "Module 3 teaches it
  to learn" → "next: the training half (Days 4–9) — loss, gradients, the loop". Keep quest-id, nav, engine.

### `review.html` (`wf3-review`, final gate after Day 9)  ← P0 relabel
- Relabel every "Module 3"/"M3" → **"Module 2"**: `<title>`, nav-title, hero eyebrow, `h1`, verdict
  button ("Mark Module 2 complete"), finale ("Module 2 — passed!"), footer.
- "Ready for Module 4?" → **"Ready for Module 3?"** (next module is `m03-attention`);
  "(M1–M3)" → **"(M1–M2)"**; "M4 · Attention" → **"M3 · Attention"**.
- The internal self-check "Day 1/2/3" labels (the wf3 sub-arc's loss/gradients/loop) may read
  "Days 4–9 recap" or keep their local numbering — but must not claim to be a separate module.
- Keep quest-id `wf3-review`, nav (→ `../m03-attention/day-01-embeddings/lesson.html`), and the
  `done.verdict` engine.

### `experiment.py` × 9
- Ensure each stub's commented run path matches its folder and mentions Option A / Option B (per the
  artifact contract stub convention). No filled solutions.

---

## Sequence of work

1. Day 1 → 2 → 3 (Part A), then `review-part-a.html`.
2. Day 4 → 5 → 6 → 7 → 8 → 9 (Part B), then `review.html`.
3. `experiment.py` stub pass (all 9) — quick consistency sweep.
4. QA gate: `/frontier-refactor-qa` (contracts, visuals, artifact paths, stale commands, JS/shell
   behavior, legacy skill refs) + repo tooling: `sessions/staff_lens_audit.js`, `sessions/nav_audit.py`,
   `lesson_audit.py m02-the-neuron`, and the jsdom/`node --check` harness for the two review gates and any
   edited JS. Verify hub `pillStatus` reads `done.verdict` (MEMORY: review-gate-completion-contract).
5. Write `sessions/m02_refactor_report.md`.

## Definition of done (module passes only when all hold)

- [ ] Blueprint + 4 contracts exist (✅ this batch)
- [ ] All 29 must-cover concepts land at stated depth on their lesson; 4 promoted items visibly owned
- [ ] Every lesson keeps the full Coach Layer (Jargon · Math Ladder · silent-failure + trade-off ·
      interview answer · diagnostic quiz Q)
- [ ] P0 visuals present and show a changing quantity; playground == Math-Ladder == Produce numbers
- [ ] Every Produce run command matches its folder; no stale "Module 3/2-3 Day N" anywhere; quest-ids intact
- [ ] All 5 defect classes closed (labels, numeric self-consistency, playground↔produce, anchor spine,
      no premature close)
- [ ] Nav chain unchanged; both gates use the dark shell and record `done.verdict`; hub pill logic reads it
- [ ] QA gate passes; no broken JS / broken nav; no legacy skill references
- [ ] `m02_refactor_report.md` written

## Stop conditions (halt the batch and report if any occur)

- Hard audit failure, broken JS, or broken navigation after an edit
- A required quest-id would have to change to satisfy a contract (it must not — escalate instead)
- Coverage or visual contract cannot be satisfied within the frozen 7-section shell
- Scope drift beyond `sessions/m02-the-neuron/` + the hub label check + these contract/report files
