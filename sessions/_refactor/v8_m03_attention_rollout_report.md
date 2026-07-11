# v8 Rollout Report — m03-attention ("Module 4 · Represent")

**Loop:** `v8-m03-source-first-rollout`
**Date:** 2026-07-09
**Module:** `sessions/m03-attention` (5 lessons) · learner-facing **Module 4 · Represent**
**Authoring mode:** Reference (first non-seed, no-notebook v8 rollout)
**Recommendation:** **Pass with P1**

---

## Scope

Second module through the v8 source-first pipeline (after the m02 seed) and the **first**:
- **non-seed** module rollout,
- **no-notebook / Reference-mode** rollout (Notebook Smoothness Gate = N/A),
- rollout of a module that already shipped mature v7-era lessons.

Goal: put all 5 attention lessons on the source-first pipeline and lift their learning
experience to the v8 Reader Flow Blueprint with a **module-wide beginner spine**, while
preserving every frozen shell invariant and the interactive content byte-for-byte.

## Phase (in the Autonomous Rollout Loop)

Read tracker → (no seed-propagation risk: m02 already stabilized) → created target manifest →
determined authoring mode (Reference: no companion notebook) → authored `source.md` per day in
reader-flow order → compiled → fixed the one P1 in source + recompiled → QA → updated
manifest + tracker → report.

## Tracker status

- `current_target`: `sessions/m03-attention` → **`sessions/m04-first-model-mlp`**
- summary: `pass_with_p1: 2` (m02 + m03), `in_progress: 0`, `not_started: 42`
- m03 entry: `status: pass_with_p1`, `skill_version: v8`, `open_p0: 0`, `open_p1: 1`, `open_p2: 2`
- rollout_history: `v8_m03_attention_source_first_rollout` appended; v8 phasing → Phase E DONE

## Manifest status

`sessions/m03-attention/_refactor/manifest.yaml` — **created** and finalized to `pass_with_p1`,
`open_p0_count: 0`. Every finding maps to backlog.

---

## The narrative spine (module-wide)

> **A meeting of words, each deciding who to listen to.**

| Day | Concept | How the spine carries it |
|-----|---------|--------------------------|
| D1 embeddings | words → vectors | Before the meeting, each word gets a **profile card** of numbers; similar words get similar cards; a dot product reads how alike two cards are. |
| D2 Q/K/V | three projections | In the meeting each word plays **three roles** from its one card: a Query (what it seeks), a Key (name-tag it advertises), a Value (content it shares). |
| D3 scores+softmax | scaled dot-product attention | Each word scores its Query against everyone's Key, scales by √d_k, splits a **100% attention budget** with softmax, and blends the Values. `softmax(Q·Kᵀ/√d_k)·V`. |
| D4 multi-head | parallel heads | One conversation isn't enough — run several **breakout meetings** (heads), each `d_k=d_model/h`, then concat + `W_O`. |
| D5 positional | order | The meeting ignores **who spoke when** (permutation-invariant); stamp each word with a **seat number** (sinusoidal PE). |

## Authoring method (reusable for future non-seed modules)

- **Clean-authored** reader-flow prose regions: `hero`, `s1`, `s2`, `s4`, `s7`, `fin`.
- **Froze verbatim** (via `@@@ region` blocks): `title`, `brand_sub`, `sidebar_nav`, `nav_prev`, `nav_next`.
- **Preserved untouched**: `DEMOS`, `BUILD`, `QS` were **omitted from source**, so the compiler left
  the donor's interactive content byte-identical (zero regression). `s3`/`s5`/`s6` section shells
  and all live viz (`attention-heatmap`, `attention-pipeline`, `softmax-scaling`, `attention-multihead`)
  carried straight through.
- Donors: `sessions/_compiler/shells/m03-day-0N.donor` (pristine snapshots).
- **No compiler/shell/JS files were modified** — m02 unaffected.

---

## Gate results (v8)

| Gate | Result |
|------|--------|
| Reader Flow (source, strict/clean) | **PASS** all 5 — human-first hero, front-loaded Jargon Ladder, picture-before-vocab, frontier deferred to s4, spine "meeting" in ≥3 of hero/s1/s2/s4, discovery Produce |
| Shell Invariant (output vs donor) | **PASS** all 5 — quest-id frozen, 7 sections, 8 data-targets, DEMOS/BUILD/QS present, playground ≥3, quiz q:4 o:16, localStorage keys, .fin, 7 gotit, nav hrefs, no unresolved markers, experiment.py referenced; CSS + all `<script>` engines byte-identical to donor |
| Notebook Smoothness | **N/A** — Reference mode, no companion notebook (skipped per No-Notebook rule, never failed) |
| No-Notebook Authoring | **PASS** — mode declared; ≥1 authoritative source cited (Vaswani 2017 + FlashAttention + RoPE/ALiBi); staff claims trace to it |
| Staff Depth | **PASS** all 5 — 1 named silent-failure + 1 trade-off + 1 grounded interview line + diagnostic quiz Q each |
| Coverage Traceability | **PASS** — 17/17 must-cover attention concepts land at their day |
| Visual/Evidence | **PASS (preserved)** — behavioral viz carried verbatim from donors |
| Artifact | **PASS** — discovery-framed Produce (predict→run→observe); run paths match folders |
| Determinism | **PASS** — every day compiles twice byte-identical |

## Audit / check evidence

```
lesson_audit m03-attention  -> 5 OK / 0 MISSING / 0 LEFTOVER / 0 DEGRADED
nav_audit                   -> 0 CHAIN, 0 BROKEN, 0 CASE, 0 ORPHANS — PASS
staff_lens_audit m03        -> 5/5 staff-lens, gap 0, each fail:1 trade:1 q:4 o:16 errs:[]
                               (render:BROKEN is the known-benign .sec vs .module-section selector mismatch)
node --check                -> all inline scripts parse (4-5 per day)
git diff vs HEAD            -> only hero/s1/s2/s4/s7 (+fin d04/d05) changed; shell/nav/CSS/
                               interactive/s3/s5/s6 byte-identical; experiment.py + log.md untouched
DEMOS/BUILD/QS vs donor     -> byte-identical on all 5
compile x2                  -> byte-identical on all 5 (deterministic)
```

## Adversarial verification

5 independent read-only Explore verifiers (workflow `wf_2cb6623f-f1e`), one per day, each checking
attention-math correctness, Reader-Flow-Blueprint adherence, terminology, and — critically —
consistency between the authored prose and the preserved DEMOS/BUILD/QS.

Result: **technical_correctness CORRECT on all 5**, **spine coherent on all 5**, **0 P0**, **1 P1**.

## Findings

### P0 — none.

### P1 (resolved this loop)
- **BL-P1-d04-consistency** *(fixed)* — the authored d04 s4/s7 taught the canonical multi-head design
  (concat width `h·d_k = d_model`, `W_O` square `(d_model,d_model)`) which clashed with the **preserved**
  s3 `DEMOS.merge` toy (concat 4 → output 2, `W_O` 4×2). The playground is frozen, so the fix was on the
  authored side: added one sentence in s4 noting the playground uses deliberately tiny numbers so *its*
  `W_O` also trims the width, while the standard design keeps concat width = `d_model`. Recompiled clean;
  gates PASS; interactive still byte-identical.

### P1 (open, sanctioned)
- **BL-P1-legacy** — `/frontier-experiment-lab` (uninstalled) referenced in every Produce Option-B.
  Curriculum-wide; **wont_fix_per_module** (fix globally when the skill is reinstated).

### P2 (open, preserved/frozen)
- **BL-P2-d04-navlabel** — d04 nav "next" label "Positional Encoding & RoPE" vs d05 title "…the Shuffle Problem". Frozen nav; cosmetic.
- **BL-P2-d04-playground-toy** — s3 `DEMOS.merge` toy doesn't follow strict `h·d_k=d_model`. Preserved; bridged by the authored s4 note above.

## Edits applied

- Created `sessions/m03-attention/_refactor/manifest.yaml`.
- Authored `sessions/m03-attention/day-0N-*/source.md` (5 files) and compiled each to `lesson.html`.
- Created donors `sessions/_compiler/shells/m03-day-0N.donor` (5).
- One follow-up edit to `day-04-multihead/source.md` (P1 fix) + recompile.
- Updated `sessions/_refactor/rollout_tracker.yaml` (current_target, summary, m03 entry, history, phasing, next_rollout).

## Frozen invariants verified intact

quest-ids `wf4-d01-embeddings / wf4-d02-qkv / wf4-d03-attention-scores / wf4-d04-multihead / wf4-d05-position`;
"Module 4 · Represent" labels; nav chain (m02 review → d01 → … → d05 → m03 review.html); localStorage
`frontier-lesson:<qid>` + `frontier-theme`; quiz q:4 o:16; DEMOS/BUILD/QS behavior; `.fin` completion;
7 gotit buttons; `experiment.py` + `log.md` untouched.

## Recommendation

**Pass with P1.** Open P0 = 0; the one consistency P1 was resolved in-loop; remaining open items are one
sanctioned curriculum-wide P1 and two preserved/frozen cosmetic P2s. All required v8 gates pass or are
explicitly N/A (Reference mode). Compiled lessons are deterministic; shell + audits clean; interactive
content byte-identical to the shipped donors.

**Next target:** `m04-first-model-mlp` (check for a companion notebook — may be **Exemplar** mode, unlike m03's Reference mode).
