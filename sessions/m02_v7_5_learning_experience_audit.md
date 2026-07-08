# Module 2 — v7.5 Learning Experience Gate Audit

**Scope:** `sessions/m02-the-neuron/day-01-single-neuron/lesson.html` (`wf2-d01-neuron`) and `sessions/m02-the-neuron/day-02-activations/lesson.html` (`wf2-d02-activations`).
**Gate:** `frontier_lab_refactor_skills_v7_5/LEARNING_EXPERIENCE_GATE.md` + `frontier-refactor-qa` Learning Experience Gate + `frontier-lesson-builder` Learning Barrier Gate.
**Benchmark (barrier reference only, NOT source):** `00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb`.
**Mode:** Report-only. No lesson files edited.
**Date:** 2026-07-08 · **Loop:** `m02_v7_5_learning_experience`

---

## TL;DR

- **Verdict: Pass with P1.** Not blocked. Both lessons are technically strong, warm in the middle, and end with genuine empowerment. But the **Experience Layer is back-loaded**: on both days the *mental picture / analogy* lives in Section 2, while Section 1 opens with definitions (D1) or matrix algebra (D2) — the exact v7.5 inversion of "human situation → curiosity → analogy → tiny example → mechanism."
- **Full experience-layer rewrite: NOT required.** The analogies are real and structural; the fix is a **targeted reorder + reframe of the openings and Section 1** on each day, preserving all mechanism, staff, and artifact content (and all frozen shell/quest-id invariants).
- **Strategic risk (the real reason this matters now):** m02 is the **seed module**. If m03 is rolled out by mirroring m02's section pattern, the "define/formula-first, picture-second, checklist Goal box" inversion **propagates into every future module**. This is the strongest argument for acting before/at m03 — not to re-polish m02, but to stop the seed pattern from replicating.
- **Grandfathering:** consistent with the manifest's own v7.3 precedent, the two clearest findings **would be P0 for a fresh v7.5 build** but are recorded **P1 for the shipped seed** (correct content, no wrong model taught, strong payoff).

---

## 1. Learning Experience Gate — the 9 required checks

Scored against `LEARNING_EXPERIENCE_GATE.md` §"Required checks". ✅ pass · ⚠️ partial/weak · ❌ fail.

| # | Check | Day 1 | Day 2 | Note |
|---|-------|:----:|:----:|------|
| 1 | Why care before the math? | ⚠️ | ✅ | D1 hero = capability bridge ("multiply matrices → build the thing they power"); GPT-3 teaser is a good hook but is mechanical/scale, not felt. D2 hero = strong ("a *mysterious* third step… find out why"). |
| 2 | Familiar human experience? | ✅ | ⚠️ | D1 brain + judge = human. D2 "valve / dimmer switch" is mechanical, not a human situation; no bouncer/thermostat-style everyday anchor. |
| 3 | Fear/confusion named? | ✅ | ✅ | Both have the 😕 "why this trips people up" callout naming the exact trap. **Genuine strength — keep it.** |
| 4 | Curiosity hook that pulls you forward? | ⚠️ | ✅ | D1 hook (GPT-3 columns) is buried in a §1 industry callout (line 357), after the definitions. D2's "mysterious third step" is front-and-center in the hero. |
| 5 | Mental picture *before* the first formula? | ❌ | ❌ | **The core failure.** D1: brain picture is §2 (line ~367); "weighted sum / dot product / bias / activation" arrive first in §1 (lines 345–356). D2: valve/dimmer is §2 (line ~360); `(x@W₁)@W₂ = x@(W₁@W₂)` matrix algebra arrives first in §1 (line 348). |
| 6 | Analogy guides the whole lesson (spine)? | ✅ | ⚠️ | D1 brain analogy maps every term, is revisited in the judge card + build. D2 valve/dimmer is reused in the build visual but **not** in the staff-lens failure mode or the artifact — thinner spine. |
| 7 | Stays warm through the technical sections? | ✅ | ✅ | Callouts, Math Ladder "in words" step, and plain-language framing keep both readable through §4. |
| 8 | Staff depth = empowerment, not pressure? | ✅ | ✅ | "worth naming precisely," "say this in an interview," deferred-load notes ("you'll feel this on Day 3/5"). No "you're behind" tone. **Strength.** |
| 9 | Artifact = discovery, not homework? | ⚠️ | ⚠️ | Both Produce sections are framed as "leave today's evidence" + **"Acceptance criteria"** (homework register). Explain-back `log.md` is present (good). D2's collapse-then-break is an inherent "observe something surprising" moment but it's buried under the criteria list. |

**Reading:** No lesson fails on tone, warmth, or empowerment (checks 3, 7, 8 are strengths). The failures cluster on **ordering** (check 5, both days) and **opening framing** (checks 1, 4, 9). This is precisely the v7.5 target: "rewrite lessons so the experience layer runs through the whole lesson … not a checklist with a cute intro."

---

## 2. Where the lessons follow the v7.5 "bad pattern"

`LEARNING_EXPERIENCE_GATE.md` bad pattern: `Prerequisites → definition → formula → mechanism → staff lens → artifact`.
Desired pattern: `Human situation → curiosity → familiar analogy → tiny concrete example → mechanism → visual/evidence → artifact → staff depth`.

**Day 1, Section 1 (lines 339–361) is the bad pattern almost verbatim:**
1. line 342 — `🧭 What this assumes:` (**Prerequisites**)
2. line 344–345 — "What is a neural network?" (**definition**)
3. line 347–353 — neuron = weighted sum + bias + activation; "this is exactly a **dot product**" (**mechanism/terms**)
4. Only in **Section 2** (line 367) does the brain **mental picture** appear.

The analogy that should *precede* the vocabulary arrives *after* it. A learner meets `w·x + b`, "dot product," and "activation function" before they have any picture to hang them on.

**Day 2, Section 1 (lines 339–354) is sharper still — formula-first:**
1. line 342 — prerequisites callout
2. line 344–345 — definition of "activation function"
3. line 347–349 — "Why does it exist?" opens with **matrix algebra**: `(x @ W₁) @ W₂` equals `x @ (W₁ @ W₂)` (line 348)
4. The **valve/dimmer mental picture** is deferred to Section 2 (line 360).

`frontier-lesson-builder` marks **"No formula before felt intuition"** *non-negotiable*. D2 §1 puts a two-matmul associativity identity in front of a beginner before any picture — the highest-barrier possible opening for this concept.

---

## 3. Held-out barrier comparison vs the benchmark notebook

Per `frontier-refactor-qa` §"Held-out eval": compare on the **learning-experience axis only**, and do **not** over-credit m02 for being more advanced if it is less inviting. (Content coverage, visual evidence, and runnable evidence are already adjudicated in the m02 manifest — not re-litigated here.)

| Barrier dimension | Notebook `01_what_is_a_neural_network` | m02 D1 / D2 | Who lowers the barrier |
|---|---|---|---|
| First move | "Don't worry if you're new to this." → "Let's start with something familiar: your brain! Right now, as you read this text…" (human situation first) | D1: "You can read shapes and multiply matrices." D2: matrix identity in §1 | **Notebook** |
| Picture vs term order | Brain story → *then* neuron/weight/bias defined | Terms/formula in §1 → picture in §2 | **Notebook** |
| "Why activations exist" | Plain words: "just one big multiplication and addition — it couldn't learn complex patterns" (bouncer/thermostat) | `(x@W₁)@W₂ = x@(W₁@W₂)` in §1 | **Notebook** |
| Goal framing | "By the end you'll understand…" + "Prerequisites: just curiosity!" | "By the end, you'll be able to: name the three steps… compute by hand" (capability checklist) | **Notebook** (lower pressure) |
| Depth, rigor, staff lens, failure modes, interview framing | Absent | Rich, precise, empowering | **m02 (by a wide margin)** |
| Curiosity mid/late | Flat | D2 "mysterious third step," GPT-3 payoff, "why this trips people up" | **m02** |

**Conclusion:** m02 is far stronger on rigor, staff depth, and mid-lesson curiosity. It is **weaker than a plain beginner notebook specifically on the first screen** — the opening ordering and goal framing. This is the gate's exact P0-adjacent condition: *"learner would reasonably prefer the reference notebook for readability"* — true here **only for the opening 30 seconds**, not the whole lesson. That bounded scope is why this lands **P1, targeted-reorder**, not Blocked/full-rewrite.

---

## 4. Findings — P0 / P1 / P2

No **open P0**. (Grandfathering note below matches the manifest's v7.3 precedent: new-gate findings on a shipped seed are recorded P1, flagged "would-be-P0 for a fresh v7.5 build.")

### P1 (would-be P0 on a fresh v7.5 build)

| ID | Day | Finding | Fix shape (preserve shell/quest-id/BUILD/DEMOS/QS) |
|----|-----|---------|-----|
| **LX-D1-PICTURE-ORDER** | D1 | Mental picture (brain analogy, §2 line ~367) arrives **after** §1 has introduced neuron/weighted sum/dot product/bias/activation (lines 345–356). Gate trigger: *"first technical term arrives before a mental picture."* | Seed a 2–3 sentence brain/"score-and-decide" picture into the hero or the top of §1 **before** the three-step vocabulary. Keep the full §2 mapping table as the payoff. |
| **LX-D2-FORMULA-FIRST** | D2 | §1 "Why does it exist?" leads with the matrix identity `(x@W₁)@W₂ = x@(W₁@W₂)` (line 348) before any analogy. Violates *"No formula before felt intuition."* | State the collapse in **plain words + the valve/"stacking rulers stays a straight ruler" picture first**; the matrix identity already lives in §4's Math Ladder (line 415), so §1 can defer it. |
| **LX-D1-OPENING-CHECKLIST** | D1 | Goal box (line 335) is a capability checklist ("By the end, you'll be able to: name the three steps… compute by hand"); hero (line 334) bridges from engineering ("multiply matrices") rather than wonder. Gate trigger: *"opening feels task-like or checklist-like."* | Lead the hero with the GPT-3 payoff hook (already on line 357) as *curiosity*; demote the objective list to a quieter "you'll be able to…" line under it. |
| **LX-D2-NARRATIVE-SPINE** | D2 | Valve/dimmer analogy is reused in the build visual but **not** in the staff-lens failure mode (dead ReLU/saturation, line 424) or the artifact. Spine is thinner than D1's. | Carry the valve image into the dead-ReLU failure ("the one-way valve rusts shut — nothing gets through") and into the Produce framing. |
| **LX-ARTIFACT-AS-DISCOVERY** | D1 & D2 | Produce framed as "leave today's evidence" + **"Acceptance criteria"** (homework register), not "run this and observe something surprising" (`frontier-lesson-builder` Artifact-as-discovery rule). Explain-back log **is** present (credit). | Reframe acceptance around an observation/surprise. D2 is easy: foreground "watch two layers collapse into one, then insert ReLU and watch the collapse break." D1: "predict the output, then check whether ReLU zeroed it." |

### P2 (nits — do not block)

| ID | Day | Finding |
|----|-----|---------|
| **LX-D1-FRONTIER-EARLY** | D1 | GPT-3 scale teaser (§1, line 357) lands before the learner is oriented by the §2 analogy. It *is* a good hook, so this is minor — but per the gate, frontier relevance ideally follows orientation. |
| **LX-D2-HUMAN-SITUATION** | D2 | No familiar *human* situation (valve/dimmer are mechanical). Notebook uses bouncer/thermostat. Minor because the "mysterious third step" hook carries the opening. |
| **LX-BOTH-GOAL-REGISTER** | D1 & D2 | Both Goal boxes use the "By the end, you'll be able to:" objective register. Consistent house style, but reads assessment-like vs the notebook's "Prerequisites: just curiosity!". |

### Already tracked — not re-flagged

- **BL-P1-legacy** (dead `/frontier-experiment-lab` in every Produce Option B, D1 line 470 / D2 line 460): curriculum-wide, fix globally, never per-module. Noted, not counted as new.
- **BL-VE-D1 / BL-VE-D2 / BL-COV-XOR** (boundary geometry, plotted derivatives, XOR limit): already open P1 in the manifest; visual/coverage, out of this learning-experience audit's scope.

---

## 5. Other required gates (brief — not the focus of this audit)

- **Coverage Traceability:** Unchanged by this audit — manifest holds 32/33 must-cover PASS, 1 waived (C12). No experience finding downgrades coverage. **PASS.**
- **Visual / Evidence:** Unchanged — behavioral-viz upgrades already tracked (BL-VE-D1/D2). **PASS_WITH_P1** (pre-existing).
- **Coach Voice / Foundation Framing:** D1 has the full 5-element foundation frame (brain intuition, "not biology" caveat line 389, mapping table, analogy-breaks, transition to `activation(w·x+b)`). Voice is warm. **PASS** — the only issue is *order*, not presence.
- **Staff Depth:** Strong on both days (silent failure + trade-off + grounded interview line). Empowering register. **PASS** — and explicitly credited (check 8).
- **Artifact / Anchor Consistency:** Anchor numbers consistent (playground = Math Ladder = Produce). Artifact framing is the P1 above; correctness intact. **PASS_WITH_P1.**

---

## 6. Recommendation

> ### Verdict: **Pass with P1.** Do **not** block m03 rollout. Do **not** do a full experience-layer rewrite.

**Why not a full rewrite:** the Experience, Mechanism, and Staff layers all exist and are high quality. The analogies are structural, not decorative. The defect is *sequence and framing on the first screen*, not missing content. A full rewrite would risk the frozen invariants (quest-ids `wf2-*`, BUILD/DEMOS/QS, completion wiring) for no content gain.

**What to actually do (targeted, ~openings + Section 1 of each day):**
1. **D1** — seed the brain/"score-and-decide" picture into the hero/top of §1 (before the vocabulary); reframe the Goal box around the GPT-3 curiosity hook. *(LX-D1-PICTURE-ORDER, LX-D1-OPENING-CHECKLIST)*
2. **D2** — open §1 with the collapse *in plain words + valve picture*; move the matrix identity to §4 (where it already appears). *(LX-D2-FORMULA-FIRST, LX-D2-NARRATIVE-SPINE)*
3. **Both** — reframe Produce from "acceptance criteria" to "run it and notice X." *(LX-ARTIFACT-AS-DISCOVERY)*

**The decision that actually gates m03 (most important):** m02 is the **seed module**. The v7.5 gate did not exist when m02 was built (v7.2/v7.3), so its Section-1 pattern predates the Experience Layer. Choose one before m03 ships:
- **(A) Fix the seed first** — apply the targeted D1/D2 reorder above so m03, if cloned from m02, inherits the *correct* v7.5 pattern; **or**
- **(B) Override at build time** — explicitly instruct the m03 rollout to follow the v7.5 desired pattern (**picture-before-term, curiosity-not-checklist opening, plain-words-before-matrix-algebra**) rather than mirroring m02's ordering.

Either is acceptable. What is **not** acceptable is rolling m03 by copying m02's structure without addressing the inversion — that replicates a would-be-P0 into every downstream module. **Recommended: (A)** — a small, bounded fix to the seed is cheaper than repeatedly overriding the pattern for 42 remaining modules.

---

## 7. Manifest Update (recommended follow-up — not applied here)

Per `frontier-refactor-qa` §"Manifest Update Gate", these findings must be appended to `sessions/m02-the-neuron/_refactor/manifest.yaml`. This audit is report-only (lesson files untouched, manifest/tracker left for the follow-up write):

- Add to `backlog.p1`: `LX-D1-PICTURE-ORDER`, `LX-D2-FORMULA-FIRST`, `LX-D1-OPENING-CHECKLIST`, `LX-D2-NARRATIVE-SPINE`, `LX-ARTIFACT-AS-DISCOVERY` (each `source: "v7.5 Learning Experience Gate"`, `retro_note: "would-be P0 on a fresh v7.5 build; P1 for shipped seed"`).
- Add to `backlog.p2`: `LX-D1-FRONTIER-EARLY`, `LX-D2-HUMAN-SITUATION`, `LX-BOTH-GOAL-REGISTER`.
- Add a `quality_gates` row: `Learning Experience Gate (v7.5) — PASS_WITH_P1`.
- Consider a `skill_gap_candidate` S5: *"seed/template lessons built pre-v7.5 place the mental picture in §2 after §1 definitions/formulas; the v7.5 desired pattern must gate cloned modules so the inversion does not propagate."*

## 8. Tracker Update (recommended follow-up — not applied here)

In `sessions/_refactor/rollout_tracker.yaml`:
- m02 `open_p1`: 9 → 14 (add the 5 new LX P1s); `open_p2`: 10 → 13.
- Append `reports:` on the m02 module entry and in `rollout_history`: `sessions/m02_v7_5_learning_experience_audit.md`.
- Record `skills.version` bump to `v7_5` and note the Learning Experience Gate is now in force for m03+ (`current_target: sessions/m03-attention`).
