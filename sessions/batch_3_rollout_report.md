# Batch 3 — Coach Layer Rollout Report

_Completion report for the `frontier-curriculum-refactor` Coach Layer rollout to **Batch 3 — the JAX & systems tier** (`m07`/`m08`/`m09a`/`m09c`). Date: 2026-07-07._

Companion docs: [`batch_3_rollout_plan.md`](./batch_3_rollout_plan.md) (the plan this executed) · [`COACH_STYLE_GUIDE.md`](./COACH_STYLE_GUIDE.md) · [`enhance_lesson_checklist.md`](./enhance_lesson_checklist.md) · [`lesson_audit.py`](./lesson_audit.py) · [`batch_2_rollout_report.md`](./batch_2_rollout_report.md) (the precedent this followed).

> **Naming note.** This is the **Coach Layer rollout Batch 3**. It is distinct from [`batch_3_cleanup_report.md`](./batch_3_cleanup_report.md), a defect-only cleanup of the *inference-systems* tier (`m06`/`m16a`/`m17a`/`m17b`). No module overlap.

---

## 1. TL;DR

Applied the Coach Layer to **all 22 in-scope lessons** of Batch 3 (`m07-thinking-in-jax`, `m08-transformer-math`, `m09a-hardware-physics`, `m09c-sharding-parallelism`), purely **additively**, and built **3 new interactive P0 systems visuals** in `m09a`. Workflow was plan-first: visual audit (22 read-only per-lesson auditors) → P0 identification → written plan → user-confirmed P0 scope → edit → per-file verify → report. Final state: **24/24 OK in `lesson_audit.py`, 0 DEGRADED, 22/22 jsdom-render clean, all 3 P0 labs pass simulated-interaction checks and hand-verified math.** No navigation, `data-quest-id`, quiz, playground `DEMOS`, `BUILD`, or tooltip machinery changed — diff-verified **0 protected lines removed**.

The 2 original pilot lessons (`m08/day-01-transformer-arithmetic`, `m08/day-03-kv-cache`) were already Coach-complete and left untouched.

---

## 2. Files changed

**22 lesson files** (all additive):

| Module | Lessons edited | P0 labs built |
|--------|----------------|:---:|
| `m07-thinking-in-jax` | day-01…day-06 (6) | — (JAX programming-model: terminal-appropriate) |
| `m08-transformer-math` | day-02, day-04, day-05, day-06 (4) | — (day-01/day-03 pilot-done, untouched) |
| `m09a-hardware-physics` | day-01…day-06 (6) | **day-02-tpu, day-04-multi-device, day-05-memory-footprint** |
| `m09c-sharding-parallelism` | day-01…day-06 (6) | — (day-01–04, day-06 embed `parallelism.html`/`transformer-flops.html`) |

Plus **new artifacts** `sessions/batch_3_rollout_plan.md` (the pre-edit plan) and this report.

**`git diff --stat`:** 22 files changed, **548 insertions(+), 59 deletions(-)**. The 59 deletions are all documented replacements, not removals of content or machinery:

| Deletion | Count | Why |
|----------|:---:|-----|
| Old one-line `📝 log.md` nudge | 22 | Folded into the richer `📓` research-log + explain-back block |
| `color:var(--text2)` → `var(--ink2)` | 33 | Pre-existing undefined-CSS-var sub-lines, fixed in-file (guide §7 coherence) |
| Truncated `var BUILD` note text | 2 | `m09c/day-06` finale-truncation fix (`bytes_per_val=2):`, `across devices)`) |
| "Extend experiment.py into experiment.py" | 2 | `m07/day-03`, `day-04` generator glitch → "Create … (or extend …)" |
| "Then create … experiment.py" | 1 | `m07/day-06` → "Then extend the same … experiment.py" |

Diff audit confirms **no** `data-quest-id` / `data-demo` / `data-sec` / `var QS|DEMOS` / nav-`href` / `.gotit` line was removed. The single `var BUILD=` in the removed set is the `m09c/day-06` one-line literal re-emitted with the two truncated `note` strings repaired (structure intact — jsdom confirms BUILD present, 3 demos, 4 quiz).

---

## 3. Before → after (Coach markers per lesson)

`lesson_audit.py` + the shell-agnostic jsdom harness confirm every edited lesson kept **7 sections / 7 got-it / 3 demos / 4 quiz / BUILD / tooltip** intact. Markers added where missing (● added this batch, ○ already present, – intentionally skipped):

| Lesson | pain😕 | Ladder🪜 | interview🎤 | StaffLens | Accept | log📓 | 中文 | P0 lab |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| m07 · d01 jax-immutability | ● | – | ● | ○ | ● | ● | ● | — |
| m07 · d02 prng-keys | ● | – | ● | ○ | ○ | ● | ● | — |
| m07 · d03 vmap | ● | – | ● | ○ | ○ | ● | ● | — |
| m07 · d04 jit | ● | – | ● | ○ | ○ | ● | ● | — |
| m07 · d05 flax-optax | ● | – | ● | ○ | ○ | ● | ● | — |
| m07 · d06 vit-capstone | ● | – | ● | ○ | ● | ● | ● | — |
| m08 · d02 qkv-matrices | ● | ● | ● | ○ | ○ | ● | ● | — |
| m08 · d04 production-code | ● | ● | ● | ● | ○ | ● | ● | — |
| m08 · d05 pretraining-logic | ● | ● | ● | ● | ○ | ● | ● | — |
| m08 · d06 scaling-rehearsal | ● | ● | ● | ● | ○ | ● | ● | — |
| m09a · d01 rooflines | ● | ● | ● | ○ | ○ | ● | ● | — |
| m09a · d02 tpu-architecture **(P0)** | ● | – | ● | ○ | ○ | ● | ● | ● |
| m09a · d03 sharding | ● | ● | ● | ○ | ○ | ● | ● | — |
| m09a · d04 multi-device **(P0)** | ● | ● | ● | ○ | ○ | ● | ● | ● |
| m09a · d05 memory-footprint **(P0)** | ● | ● | ● | ○ | ○ | ● | ● | ● |
| m09a · d06 checkpointing | ● | ● | ● | ○ | ○ | ● | ● | — |
| m09c · d01 data-parallel-fsdp | ● | ● | ● | ○ | ○ | ● | ● | — |
| m09c · d02 tensor-parallel | ● | ● | ● | ○ | ○ | ● | ● | — |
| m09c · d03 pipeline-parallel | ● | ● | ● | ○ | ○ | ● | ● | — |
| m09c · d04 llama3-tpu | ● | – | ● | ○ | ○ | ● | ● | — |
| m09c · d05 distillation | ● | ● | ● | ○ | ○ | ● | ● | — |
| m09c · d06 pytorch-consolidation | ● | ● | ● | ○ | ○ | ● | ● | — |

**Added this batch:** pain-point ×22 · Math Ladder ×14 · interview ×22 · Staff Lens ×3 (m08 d04, d05, d06) · Acceptance criteria ×2 (m07 d01, d06) · research log ×22 · bilingual scaffold ×22 · **3 interactive P0 labs**. Interview: **all 22 lacked a real 🎤 spoken-answer block** (some contained the word "interview" only in prose, so `lesson_audit.py` under-reported; the auditors confirmed) — all 22 now carry one.

---

## 4. What became warmer (Reader A)

- A **bilingual pain-point** (😕 "为什么这里容易卡住 / Why this trips people up") now opens the confusion in §1 of every lesson — e.g. immutability "感觉像被无理由地限制"; PRNG keys "要你手里攥着一个 key 到处传"; systolic array "很多人以为芯片一下子就把整个矩阵乘完了"; memory footprint "照参数量买卡，结果一训练就 OOM"; distillation "为什么用 KL 而不是普通的对错标签".
- A **bilingual intuition line** (直觉 / Intuition:) in §2 carries the feeling in Chinese and the mechanism in English — e.g. JAX array as "冲印出来的照片", the MXU as "一条流水线，权重先装进网格", memory as "打包行李，衣服只是四样之一", FSDP as "一本菜谱撕成页传着用".
- Chinese is deliberately **light: ~2–3 touches per lesson** (pain / intuition / research-log prompt), intuition only — per the batch directive and memory `prefers-english-learning-content`. Per-lesson Chinese-char counts sit in the **74–124** range (short clauses), confirming the light dose. Never inside formulas, code, or interview phrasing.

## 5. Math Ladder + analogies + Staff Lens added

- **14 four-rung 🪜 Math Ladders** (words → labeled formula → tiny numbers → sanity check), each anchored to the lesson's single core formula: `12·d²` params/block (m08/d02), all-reduce averaging (m08/d04), Chinchilla `D≈20N` (m08/d05), `C≈6ND` (m08/d06), AI vs ridge (m09a/d01), collective bytes `(P−1)N` (m09a/d03), `pmean` (m09a/d04), `16N` memory (m09a/d05), remat `O(√L)` (m09a/d06), FSDP `16N/P` (m09c/d01), TP col/row (m09c/d02), pipeline bubble `(p−1)/(m+p−1)` (m09c/d03), distillation `KL(·/T)` (m09c/d05), memory-calculator `weights+grads+opt+act` (m09c/d06). Where a full derivation already existed the ladder is **compact** — it labels the steps and adds the sanity-check rung. **All ladder arithmetic hand-verified in Node** (e.g. the systolic `C=[[1,3,6],[4,9,15],[7,15,24]]`, memory `1B fp32 Adam → 16 GB static / 48 GB total`, all-reduce `P=8 → 41% comm / 4.7× of 8×`).
- **8 Math-Ladder skips** (documented, `lesson_audit.py` "math-heavy" advisory persists by design): all 6 `m07` lessons (programming-model idioms, no single formula), `m09a/day-02-tpu` (systolic-array mechanism), and `m09c/day-04-llama3-tpu` (case study — the audit heuristic doesn't even flag it).
- **Analogies:** every lesson already had a strong everyday analogy (photo, dice ticket, cookie cutter, chef, assembly line, suitcase, cookbook, orchestra, tutor, manual/automatic car). The Coach Layer added the **bilingual intuition line** beside each and preserved every existing "where the analogy breaks" callout.
- **Staff Lens:** 20/22 already had a named silent-failure + trade-off block (preserved). Added a labeled §4 Staff Lens to **`m08/day-04`** (all-reduce mean vs. missing divide), **`m08/day-05`** (compute-optimal vs. serving cost), and **`m08/day-06`** (wrong ridge-point comparison, distinct from the existing memorize-vs-derive gotcha).

## 6. Coherence fixes (in-scope, guide §7)

- **`var(--text2)` → `var(--ink2)`** across the batch (33 sub-lines in ~12 files): a pre-existing undefined-CSS-var in callout mono sub-lines that fell back to inherited colour. Fixed only in files already being edited.
- **`m09c/day-06` truncated `var BUILD` notes** (finale-truncation class, memory `finale-truncation-generator-bug`): `bytes_per_val=2` → `bytes_per_val=2):` and `across d` → `across devices)`.
- **`m07/day-03`, `day-04` "Extend experiment.py into experiment.py"** (a from/to path pair collapsed by the generator) → "Create … (or extend …)"; **`m07/day-06` "Then create … experiment.py"** → "Then extend the same … experiment.py".

## 7. P0 visual upgrades built (3 labs)

The visual audit found **0 strict-rubric P0 lessons** — this tier is the most visually mature so far (8 interactive `viz/*.html` iframes across `m08`/`m09a`/`m09c`, plus uniformly strong static-SVG builds; `m07` is genuinely terminal-appropriate). Following the **Batch 2 precedent** (directive-driven elevation of auditor-P1 lessons) and the **user-confirmed scope** ("3 labs in m09a"), three `m09a` lessons — each instantiating a systems-visual type the parent request named, each currently terminal-playground + static-only — were elevated to P0. All are **dependency-free inline SVG + vanilla JS** on the repo `.vlab` template (CSS block copied verbatim from `m10a/day-01`), inserted **above** the existing fake-terminal playground (which keeps its 3 `data-demo` completion-gate buttons). No CDN, offline-safe, theme-aware, keyboard-reachable controls. All math hand-verified in Node.

| Lesson | Lab | Interaction verified (jsdom `--p0` + Node math) |
|--------|-----|------------------------------------------|
| m09a · d02 tpu-architecture | **Systolic-array beat scrubber** — a 3×3 MXU grid computing `C=A·B`; a beat slider sweeps the staggered wavefront (waiting → accumulating → retired-green), and a batch-1 toggle greys 2 of 3 columns to show the idle-grid failure. | slider → grid re-renders; `C=[[1,3,6],[4,9,15],[7,15,24]]`, all cells retire by beat 6 ✓ |
| m09a · d04 multi-device | **All-reduce ring timeline** — a per-step timeline bar (fixed compute + a `2(P−1)`-step ring all-reduce) and an ideal-vs-real speedup pair; device-count and bandwidth sliders. | sliders → re-render; P=8,bw=5 → comm 41%, 4.7× of ideal 8×; slow link → 1.8×; P=32 → 18× (the tax) ✓ |
| m09a · d05 memory-footprint | **Training memory stack** — a stacked bar (weights 4N + grads 4N + Adam 8N + activations) vs a 40 GB budget line; N / batch sliders + dtype (fp32/bf16) + optimizer (Adam/SGD) toggles that drive OOM live. | sliders/toggles → re-render; 1B fp32 Adam b8 → 48 GB (OOM); bf16 → static halves; SGD → 8N tenant drops ✓ |

The **not-P0** lessons were deliberately left prose-only, documented to avoid gold-plating: `m09a/day-03-sharding` (device-mesh already served interactively by `viz/parallelism.html`, used by `m09c/d01–04`), `m09a/day-06-checkpointing` / `m08/day-02` / `m08/day-05` (strong static-SVG builds), and all 6 `m07` JAX lessons (programming-model idioms where a fake terminal is the right medium — the explicit "don't blindly add D3" rule).

---

## 8. Audit results

**`lesson_audit.py` (targeted, `_recover_set.json` left unchanged):**
```
TOTAL lessons audited: 24
  OK: 24   MISSING: 0   LEFTOVER: 0   DEGRADED: 0
```

**jsdom render + JS-syntax gate:** 22/22 edited lessons render clean, 0 runtime errors, 0 syntax errors; structural counts (7 sec / 7 got-it / 3 demos / 4 quiz / BUILD / tooltip) intact on all. The 3 P0 labs re-render on simulated `input`/`click`.

**Remaining soft COACH advisories (all expected, none a failure):**

- `math-heavy but no Math Ladder` ×7 — the intentional non-formula skips: all 6 `m07` lessons + `m09a/day-02-tpu` (systolic-array mechanism, no single equation). Correct by design.
- `no concrete Produce artifact (experiment.py)` ×3 — the lessons that intentionally use a lesson-appropriate `.md` artifact instead: `m08/day-05` (`feinberg-gemini-flash-pretraining.md`, synthesis/reading day), `m08/day-06` (`scaling_book_exercises.md` + `rehearsal_script.md`, spoken-rehearsal day), `m09c/day-04` (`llama3-on-tpus.md`, case study). Each has an explain-back requirement folded into its `.md`/`log.md`. Not fabricated into an `experiment.py`.

No other advisories fire (analogy, Staff Lens, pain-point, interview, acceptance, research log, bilingual all cleared on all 22 edited lessons). No **duplicate-callout** defects (adjacent-identical-line scan clean across all 4 modules — the Batch 2 double-paste class does not recur).

---

## 9. Verification method

1. `vm.Script` syntax check on every inline `<script>` (via `/tmp/batch1_verify.js`).
2. jsdom headless load (no thrown error) + shell-agnostic structural assertions (7 sec / 7 got-it / 3 demos / 4 quiz / BUILD / tooltip). For P0 labs: `--p0` dispatches `input`/`click` and asserts the `.vlab` SVG re-renders with changed content.
3. Hand-verified each P0 lab's core math in Node (systolic matmul + beat schedule, all-reduce cost model, memory-stack pools).
4. `lesson_audit.py` per lesson and combined (24/24 OK).
5. `git diff` audit: additive callouts + one `.vlab` CSS block + IIFE per P0 file; confirmed **0** `data-quest-id`/`data-demo`/`data-sec`/`QS`/nav-`href`/`.gotit` lines removed; the 59 deletions fully accounted for (§2). Interactive `viz/*.html` iframes (5) preserved; nav 3-links intact on all 24; Chinese-char count 74–124 per lesson (light).

Reusable harness: `/tmp/batch1_verify.js` (`node /tmp/batch1_verify.js <lesson.html> [--p0]`); jsdom at `/tmp/node_modules/jsdom`.

---

## 10. Remaining risks / notes

1. **0 strict-rubric P0 — the 3 labs are a documented directive judgment.** The audit found no lesson meets the pure P0 rubric; the 3 labs are directive-driven elevations of auditor-P1 `m09a` lessons, scoped to 3 by explicit user confirmation. If a reviewer prefers prose-only or a different `m09a` day, the P0 set is the adjustable knob.
2. **No real-browser pixel check** (`no-real-browser-in-sandbox`). All verification is headless (jsdom + JS-syntax + Node math re-derivation); a final Light/Dim eyeball of the 3 new labs would need a user screenshot if pixel fidelity matters. The SVGs reuse the `.vlab` classes proven in Batch 1/2.
3. **P0 labs are teaching idealizations.** Each lab's `.vlab-note` states its simplification (3×3 toy MXU with no HBM latency; ring all-reduce with no compute/comm overlap; fixed per-batch activation estimate, no gradient checkpointing).
4. **7 Math-Ladder + 3 `.md`-artifact advisories persist by design** (§8). Documented, not failures.
5. **Concurrent edits** to `sessions/` remain possible; each file was re-read immediately before editing (the harness enforces "file modified since read"; one such re-read was triggered by a mid-edit `sed` `--text2` fix).

---

## 11. Recommended next rollout order

Batch 3 (JAX & systems tier) is done. Per the pilot's rollout order, the remaining tiers:

3. **Inference systems** — `m16a`, `m17a/b`, `m06` (already had the *cleanup* pass; Coach Layer still pending).
4. **Frontier model-literacy** — `m14*`, `m15*`, `m19`–`m21`.
5. **Research-engineering & portfolio** — `m23`–`m29`.

Reuse this batch's mechanics: the `.vlab` CSS block + drawing-IIFE pattern for any new P0 lab (place inline in §3 above the terminal), `/tmp/batch1_verify.js` + `lesson_audit.py <path>` as the per-file gate. Keep bilingual light (~2–3 touches), Math Ladder only for genuine single-formula lessons, and add Staff Lens / interview only where missing. For systems modules already carrying interactive `viz/*.html` iframes or strong static-SVG builds, prefer Coach prose over new labs (this tier proved most concepts are already visually served).

---

## 12. Definition of done — Batch 3

- [x] Visual audit run (22 read-only per-lesson auditors, rubric-scored) before any edit; P0 set identified (0 strict, 3 directive-elevated, user-confirmed) with documented rationale.
- [x] `batch_3_rollout_plan.md` written **before** any lesson was modified.
- [x] All 22 in-scope lessons: missing Coach elements added (2 pilot lessons correctly untouched).
- [x] 3 P0 interactive labs built (dependency-free, offline-safe, verified interactive + Node math-checked).
- [x] Navigation, quest-ids, quiz, playground, BUILD, tooltips, shell structure preserved (diff-verified additive; 0 protected lines removed; 5 interactive iframes intact).
- [x] Bilingual scaffold present but light (74–124 zh chars/lesson, intuition only).
- [x] Math Ladder for every genuine single-formula lesson (14); skipped + documented for the 8 non-formula lessons.
- [x] Artifact + explain-back requirement on every lesson; `.md`-artifact lessons kept (not fabricated into `experiment.py`); acceptance-criteria heading added where missing.
- [x] Coherence fixes: `--text2`→`--ink2` (33), 2 truncated build notes, 3 experiment.py-path glitches.
- [x] No motivational fluff; technical depth preserved or increased.
- [x] `lesson_audit.py`: 24/24 OK, 0 DEGRADED. jsdom: 22/22 clean. All 3 P0 labs interactive. No duplicate callouts.
- [x] `batch_3_rollout_report.md` written (this file).
