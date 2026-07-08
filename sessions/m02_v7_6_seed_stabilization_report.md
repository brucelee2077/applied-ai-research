# Module 2 — v7.6 Seed Stabilization Report

**Loop:** `m02_v7_6_seed_stabilization`
**Owner skills:** `frontier-curriculum-architect` (Autonomous Rollout Loop · Seed Propagation Risk Rule · Seed Stabilization Phase), `frontier-lesson-builder`, `frontier-refactor-qa`
**Date:** 2026-07-08
**Scope:** **All 9 lessons of `sessions/m02-the-neuron/` (D1–D9)** — openings, Section 1, the Section 2 bridge, the Section 4 staff-lens, and the Produce sections. D1–D2 done first as the seed; D3–D9 as a propagation sweep (user-directed: "finish M2 D3–D9 LX first").
**Mode:** Lesson-editing (targeted per lesson). NOT a full rewrite.

> **Update (D3–D9 sweep, same day):** after the D1/D2 seed fix, the user asked to extend the same LX stabilization to the rest of Module 2 before advancing to m03. A read-only LX audit workflow diagnosed D3–D9; each lesson then received the same targeted reorder+reframe by hand. m03 rollout is parked this session per the user. See §6 below for the D3–D9 detail.

---

## TL;DR

- **Verdict: PASS (pass_with_p1).** The 5 seed-propagation Learning-Experience P1s named in the v7.5 report are **resolved**. 0 P0 introduced. Module 2 remains the seed module and now carries the correct v7.5/v7.6 pattern for m03 to inherit.
- **Why this loop ran at all:** the v7.6 Autonomous Rollout Loop reads the tracker, and before touching the current target (`m03-attention`) it applies the **Seed Propagation Risk Rule**. `sessions/m02_v7_5_learning_experience_audit.md` had already found that m02's Section-1 ordering ("define/formula first, mental picture second, homework-register Produce") would replicate into every module cloned from the seed. That made a targeted seed stabilization **required before m03** — while the findings stay **P1** for the shipped seed (grandfathered, per the manifest's own precedent), not retroactive P0.
- **Skills reconciliation done in the same loop:** the installed `.claude/skills/frontier-*` are byte-identical to the `frontier_lab_refactor_skills_v7_6/skills/` mirror, but the tracker still said `v7_4`. Bumped tracker + manifest to **v7_6** and recorded the v7.5 + v7.6 skill-history lineage.

---

## 1. What changed, finding by finding

All edits preserve the frozen invariants (quest-ids, `BUILD`/`DEMOS`/`QS`, playground≥3 + quiz-answered gates, `.fin` completion wiring, nav chain, localStorage keys, shared shell) and every anchor number (D1: `x=[2,3]`, `w=[0.5,−1]`, `b=1` → `0`; D2: `W₁=[[1,2],[0,1]]`, `W₂=[[1,0],[3,1]]`, `W₁@W₂=[[7,2],[3,1]]`, `[13,4]`, `sigmoid'(0)≈0.25`).

| ID | Day | Before | After (fix applied) |
|----|-----|--------|---------------------|
| **LX-D1-PICTURE-ORDER** | D1 | Brain analogy lived in §2; §1 introduced neuron / weighted-sum / dot-product / bias / activation first. | Seeded a 2–3 sentence brain "weigh → add → decide" picture at the **top of §1, before any vocabulary**. §2 reworded from "Start with the thing it was named after" to **"Now make that picture exact"** — it is now the full-mapping payoff, not a cold restart. |
| **LX-D1-OPENING-CHECKLIST** | D1 | Hero bridged from engineering ("you can read shapes and multiply matrices"); Goal box was a capability checklist. | Hero now **opens on the GPT-3 "49,152 columns" curiosity hook** plus a felt one-liner ("looks at numbers, decides how much each matters, adds them, gives a verdict"). Goal box demoted to a quiet "by the time you close this tab" promise. |
| **LX-D2-FORMULA-FIRST** | D2 | §1 "why it exists" led with the matrix identity `(x@W₁)@W₂ = x@(W₁@W₂)` before any felt intuition. | §1 now opens with **plain words + a straight-ruler-on-ruler picture** for the collapse; the matrix identity is **deferred to the §4 Math Ladder** (where it already lived — no duplication, no number changed). |
| **LX-D2-NARRATIVE-SPINE** | D2 | Valve/dimmer analogy used only in the build viz; absent from the staff-lens dead-ReLU failure and the artifact. | Valve carried into the **dead-ReLU failure** ("its one-way valve has jammed shut") and the **trade-off** (Leaky ReLU / GELU = "a valve left with a permanent small leak"); Produce reframed around the valve/collapse image. |
| **LX-ARTIFACT-AS-DISCOVERY** | D1 & D2 | Produce framed as "Leave today's evidence" + **"Acceptance criteria"** (homework register). | D1 Produce → **"Predict the output, then run it — watch ReLU decide"** (guess positive/negative/zero, then check). D2 Produce → **"Watch two layers collapse — then break the collapse"** (see the same numbers, then insert ReLU and watch them differ). Heading "Acceptance criteria" → **"What you should see"**. Explain-back `log.md`, run paths, and anchor cases preserved. |

### P2s (nits)
- **LX-D1-FRONTIER-EARLY → fixed:** the GPT-3 hook moved up to the hero as opening curiosity; the §1 picture now precedes the §1 industry callout, so frontier relevance follows orientation.
- **LX-D2-HUMAN-SITUATION → open (accepted):** the valve/dimmer stay mechanical; the "mysterious third step" hook carries the opening, and forcing a human situation would fracture the valve spine.
- **LX-BOTH-GOAL-REGISTER → partial:** D1's goal register softened; D2's kept (house style; D2's hero was already a strength per the v7.5 audit).

---

## 2. Tooling change (systemic, so the reframe does not false-fire forever)

`sessions/lesson_audit.py` previously grepped for the literal phrase **"acceptance criteria"** in Produce and raised a soft advisory when absent. The v7.5/v7.6 artifact-as-discovery gate deliberately replaces that homework register. Left unchanged, the tool would emit a false advisory on **every future module** that adopts the discovery framing. Fixed by broadening the check to accept either register:

```python
if not any(k in low for k in ('acceptance criteria', 'what you should see',
                              'what you should expect', 'check your prediction')):
    coach.append("no explicit success criteria in Produce (acceptance criteria / 'what you should see')")
```

This is a tooling alignment (not in the frozen `do_not_modify` list), and it keeps `lesson_audit.py m02-the-neuron` at **0 advisories**.

---

## 3. QA evidence (all green)

| Check | Result |
|-------|--------|
| Frozen quest-ids | `wf2-d01-neuron`, `wf2-d02-activations` intact |
| Section count | 7 `.module-section` on each day |
| D1 anchors | `x=[2,3]`, `w=[0.5,-1]`, `b=1`, `ReLU(−1)=0`, `relu(w·x+b)` all present |
| D2 anchors | `[[7,2],[3,1]]`, `[13,4]`, `W1/W2`, `0.25` all present |
| `python3 sessions/lesson_audit.py m02-the-neuron` | **9 OK / 0 MISSING / 0 LEFTOVER / 0 DEGRADED / 0 advisories** |
| `python3 sessions/nav_audit.py` | **0 BROKEN — PASS** |
| `node sessions/staff_lens_audit.js m02` | **9/9 staff-lens present, gap 0**; D1/D2 `q:4 o:16 errs:[]` (inline JS parses; quiz mechanics intact). `render:BROKEN`/`secs:0` is the documented-benign `.sec` vs `.module-section` selector mismatch. |

**Adversarial self-check:** did not touch mechanism/staff/quiz/BUILD content; did not change any computed value; confirmed the picture-first reorder did not orphan the §2 mapping table (it is now framed as the payoff); confirmed no analogy was made decorative — the brain (D1) and valve (D2) spines now run opening → mechanism → staff → artifact.

---

## 4. State updates

- **`sessions/m02-the-neuron/_refactor/manifest.yaml`** — meta `last_loop: v7.6-seed-stabilization`; `skills_state.version: v7.6`; new `quality_gates` row (Learning Experience Gate v7.5 → v7.6 PASS); 5 LX P1 added to `backlog.p1` as `status: fixed` (`resolved_in: v7.6-seed-stabilization`); 3 LX P2 added to `backlog.p2`; `skill_gap_candidate` **S5** recorded; `loop_history` v7.6 entry with verification evidence.
- **`sessions/_refactor/rollout_tracker.yaml`** — `version` and `skills.version` → `v7_6`; skills `mirror` → v7.6, `v7_6_additions` recorded; m02 module entry `skill_version → v7_6`, `open_p2 → 12`, reports += v7.5 audit + this report; `summary.open_p2_total → 12`; `rollout_history` += `m02_v7_6_seed_stabilization`; `skill_history` += v7_5 and v7_6.
- **`sessions/lesson_audit.py`** — Produce success-criteria heuristic broadened (see §2).

Counting convention: `open_*` = genuinely-unresolved findings. The 5 LX P1 are recorded `fixed`, so `open_p1` stays 9 (the pre-existing VE-D1/VE-D2/COV-XOR/BL-P1a–e + the wont-fix legacy). `open_p2` rose 10 → 12 (two still-open LX P2).

---

## 5. Gate for m03

The seed now models the v7.5 desired pattern. When m03-attention is rolled out, mirroring m02's structure is now **safe** — it inherits picture-before-vocabulary, curiosity-not-checklist openings, plain-words-before-matrix-algebra, a structural analogy spine, and artifact-as-discovery. The m03 rollout must still run its own Learning Experience Gate (it is not exempt just because the seed is fixed).

**Seed stabilization: COMPLETE.** (m03 rollout parked this session per the user — "stop all M3 work, just focus on M2".)

---

## 6. The D3–D9 propagation sweep

After the D1/D2 seed fix, the user directed: *"finish M2 D3–D9 LX first"* (then *"stop all M3 work, just focus on M2"*). D3–D9 were built from the same pre-v7.5 template as D1/D2, so they carried the same inversions. Process: a **read-only LX audit workflow** (7 parallel auditors) diagnosed each lesson and returned verbatim excerpts + per-lesson fix shapes; then each lesson was hand-edited sequentially (no parallel writing, per repo rule) with an analogy tailored to its topic.

| Day | quest-id | Analogy (spine) | Fixes applied |
|-----|----------|-----------------|---------------|
| D3 Layers & forward pass | `wf2-d03-forward` | Assembly line | picture-seed §1 · curiosity hero · spine → "missing-activation" failure · predict-shapes Produce |
| D4 Loss | `wf3-d01-loss` | Golf score | picture-seed §1 · curiosity hero · golf into trade-off + Produce · "score drops round by round" |
| D5 Gradients & backprop | `wf3-d02-backprop` | Hiker in fog | picture-seed §1 · **deferred `w=w−lr·grad` out of §1 to §4** · finite-difference-match Produce (hero + spine already strong) |
| D6 Training loop | `wf3-d03-training-loop` | Basketball free-throws | picture-seed §1 · curiosity hero · spine → "forgot to reset grads" failure · predict-where-it-lands Produce |
| D7 Optimizers | `wf3-d04-optimizers` | Rolling ball | picture-seed §1 · curiosity hero (demoted leading formula) · ball into "resume w/o optimizer state" failure · "watch momentum overtake SGD" Produce |
| D8 Learning rate | `wf3-d05-lr` | Stepping stones | picture-seed §1 · curiosity hero · **removed §1 formula callout** · stones into silent-rate failure · predict-which-hop-crosses Produce |
| D9 Train/val/test | `wf3-d06-split` | Exam vs answer-key | picture-seed §1 · demoted checklist goal · exam into data-leakage failure · predict-the-winner Produce |

**Every fix preserved:** the frozen quest-ids (verified above), 7 sections per lesson, `BUILD`/`DEMOS`/`QS` + gates, `.fin` completion wiring, prev/next nav, and **every anchor number** (D3 `3→4→2`/`out=[2]`; D4 CE `1.204→0.511→0.105`, MSE `0.167`; D5 `9→5.76`, `dL/dw≈−0.212`, finite-diff `<1e-6`; D6 `grad=−16`, `w→3`, `100/25=4`; D7 `v=−24.0`, `2.5×`, β₁/β₂/ε; D8 `c=8`, factors `0.96/0.2/−3.8`, tipping `2/c=0.25`; D9 val-min `0.28` at step 200, split `0.7/0.15/0.15`).

**QA after the sweep (all 9 lessons):** `lesson_audit.py m02-the-neuron` → 9 OK / 0 advisories; `nav_audit.py` → 0 BROKEN, PASS; `staff_lens_audit.js m02` → 9/9 staff-lens present, gap 0, every lesson `q:4 o:16 errs:[]` (inline JS + quiz mechanics intact); `render:BROKEN` benign. Recorded as `LX-D3toD9-SWEEP` (status `fixed`) in the manifest backlog.

**Module 2 is now LX-consistent end to end (D1–D9). 0 P0. pass_with_p1.**
