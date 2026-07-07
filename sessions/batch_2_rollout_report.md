# Batch 2 — Coach Layer Rollout Report

_Completion report for the `frontier-curriculum-refactor` Coach Layer rollout to Batch 2. Date: 2026-07-07._

Companion docs: [`batch_2_rollout_plan.md`](./batch_2_rollout_plan.md) (the plan this executed) · [`COACH_STYLE_GUIDE.md`](./COACH_STYLE_GUIDE.md) · [`enhance_lesson_checklist.md`](./enhance_lesson_checklist.md) · [`lesson_audit.py`](./lesson_audit.py) · [`batch_1_rollout_report.md`](./batch_1_rollout_report.md) (the precedent this followed).

---

## 1. TL;DR

Applied the Coach Layer to **all 19 lessons** of Batch 2 (`m03-attention`, `m04-first-model-mlp`, `m05a-text-transformer`), purely **additively**, and built **7 new interactive P0 mechanism visuals**. Workflow was plan-first: visual audit (3 read-only module auditors) → P0 identification → written plan → edit → per-file verify → report. Final state: **19/19 OK in `lesson_audit.py`, 0 DEGRADED, 19/19 jsdom-render clean, all 7 P0 labs pass simulated-interaction checks.** No navigation, `data-quest-id`, quiz, playground `DEMOS`, `BUILD`, or tooltip machinery changed — diff-verified **0 protected lines removed**.

---

## 2. Files changed

**19 lesson files** (all additive):

| Module | Lessons edited | P0 labs built |
|--------|----------------|:---:|
| `m03-attention` | day-01…day-05 (5) | day-02-qkv, day-05-positional |
| `m04-first-model-mlp` | day-01…day-06 (6) | day-02-backward-pass |
| `m05a-text-transformer` | day-01…day-08 (8) | day-01-residual, day-02-layer-norm, day-05-causal-masking, day-07-text-generation |

Plus **new artifacts** `sessions/batch_2_rollout_plan.md` (the pre-edit plan) and this report.

**`git diff --stat`:** 19 files changed, **1025 insertions(+), 22 deletions(-)**. The 22 deletions are the per-lesson one-line `log.md` nudge (folded into the richer research-log block), 2 terminal-intro rewordings in P0 lessons (the terminal playground itself is untouched — a lead line was added above it), and **1 factual fix** (see §6). Diff audit confirms **no** `data-quest-id` / `data-demo` / `data-sec` / `var QS|DEMOS|BUILD` / nav-`href` / `.gotit` line was removed.

---

## 3. Before → after (Coach markers per lesson)

`lesson_audit.py` + the shell-agnostic jsdom harness confirm every edited lesson kept **7 sections / 7 got-it / 3 demos / 4 quiz / BUILD / tooltip** intact. Markers added where missing (● added this batch, ○ already present, – intentionally skipped):

| Lesson | pain😕 | Ladder🪜 | interview🎤 | StaffLens | Accept | log📓 | 中文 | P0 lab |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| m03 · day-01 embeddings | ● | ● | ● | ○ | ● | ● | ● | — |
| m03 · day-02 qkv **(P0)** | ● | ● | ● | ○ | ● | ● | ● | ● |
| m03 · day-03 attention-scores | ● | ● | ● | ○ | ● | ● | ● | — |
| m03 · day-04 multihead | ● | ● | ● | ○ | ● | ● | ● | — |
| m03 · day-05 positional **(P0)** | ● | ● | ● | ○ | ● | ● | ● | ● |
| m04 · day-01 mlp-mnist | ● | ● | ● | ○ | ● | ● | ● | — |
| m04 · day-02 backward-pass **(P0)** | ● | ● | ● | ○ | ● | ● | ● | ● |
| m04 · day-03 minibatch-loop | ● | ● | ● | ○ | ● | ● | ● | — |
| m04 · day-04 training-loss-dropout | ● | ● | ● | ○ | ● | ● | ● | — |
| m04 · day-05 pytorch-version | ● | – | ● | ○ | ● | ● | ● | — |
| m04 · day-06 why-pytorch | ● | – | ● | ○ | ● | ● | ● | — |
| m05a · day-01 residual **(P0)** | ● | ● | ● | ○ | ● | ● | ● | ● |
| m05a · day-02 layer-norm **(P0)** | ● | ● | ● | ○ | ● | ● | ● | ● |
| m05a · day-03 feed-forward | ● | ● | ● | ○ | ● | ● | ● | — |
| m05a · day-04 full-block | ● | ● | ● | ○ | ● | ● | ● | — |
| m05a · day-05 causal-masking **(P0)** | ● | ● | ● | ○ | ● | ● | ● | ● |
| m05a · day-06 encoder-vs-decoder | ● | – | ● | ○ | ● | ● | ● | — |
| m05a · day-07 text-generation **(P0)** | ● | ● | ● | ○ | ● | ● | ● | ● |
| m05a · day-08 full-transformer | ● | ● | ● | ○ | ● | ● | ● | — |

**Added this batch:** pain-point ×19 · Math Ladder ×16 · interview ×19 · Acceptance criteria ×19 · research log ×19 · bilingual scaffold ×19 · **7 interactive P0 labs**. **Staff Lens: 0 new** — all 19 already had one (preserved intact). Interview: 17 needed a real 🎤 block; day-05-positional (word "interview" only appeared in prose) and day-06-encoder-vs-decoder (had a 🔗 one-liner, not a spoken-answer block) each received a proper 🎤 block too — so all 19 now carry one.

---

## 4. What became warmer (Reader A)

- A **bilingual pain-point** (😕 "为什么这里容易卡住 / Why this trips people up") now opens the confusion in §1 of every lesson — e.g. "一个词为什么要变成三个向量？" (qkv), "'反向'听起来像另一套算法，其实是同一条链子倒着走" (backprop), "很多人盯着 training loss 一路下降就以为在进步" (loss/dropout), "为什么不直接把未来的权重清零，而要在 softmax 之前加 −∞？" (causal mask).
- A **bilingual intuition line** (直觉 / Intuition:) in §2 carries the feeling in Chinese and the mechanism in English — e.g. positional encoding as "时钟的时针和分针以不同速度转动", LayerNorm as "落到同一个'公平刻度'上", temperature as "调低，最可能的词几乎占满整个袋子".
- Chinese is deliberately **light: ~3 touches per lesson** (pain / intuition / research-log), intuition only — per `prefers-english-learning-content`. Never inside formulas, code, or interview phrasing. Per-lesson Chinese-char counts sit in the 63–87 range (the three short clauses), confirming the light dose.

## 5. What Math Ladder + analogies + Staff Lens were added

- **16 four-rung 🪜 Math Ladders** (words → labeled formula → tiny numbers → sanity check). For the ~8 attention/transformer lessons that already carried a full "Step 1/2/3" derivation (attention-scores, positional, backward-pass, residual, layer-norm, feed-forward, full-block, full-transformer, causal-masking), the ladder is **compact** — it labels the existing steps and adds the missing **sanity-check rung** rather than duplicating the derivation (guide §4: "leave it if present"). New from-scratch ladders were written for embeddings (dot-product similarity), qkv (`Q=x·W`), multihead (`d_k=d_model/h`), mlp-mnist (forward pass), minibatch-loop (`W←W−lr·(1/B)Σdw`), dropout (inverted dropout).
- **Analogies:** every lesson already had strong everyday analogies (library, theater seats, assembly line, whiteboard, cover sheet, weighted bag of tickets, tower of floors). The Coach Layer added the **bilingual intuition line** beside each and preserved every existing "where the analogy breaks" callout.
- **Staff Lens:** all 19 already had a named silent-failure + trade-off block (preserved). Every lesson also got a matching 🎤 **interview-ready block** (20–30s spoken answer: headline + proof + nuance) — the one Coach element that was genuinely absent almost everywhere.

## 6. Coherence fix (in-scope, factual)

- **m03/day-05-positional §4 Step 1** stated "Slots near the start use slow waves; slots near the end use fast waves" — this is **backwards**. `PE(pos,2i)=sin(pos/10000^(2i/d))` gives slot 0 divisor 1 (fast) and higher slots huge divisors (slow), and the lesson's own worked example (`pos 1 → [0.84, 0.54, 0.01, 1.00]`, slot 0 swinging, slot 2 barely moving) confirms start=fast. Verified numerically (`slot 0: 0,0.84,0.91,0.14` vs `slot 2: 0,0.01,0.02,0.03`) and corrected to "fast near the start, slow near the end", now consistent with the new PE-waves lab. Per guide §7, fixing a factual bug in the edited file is in scope.

## 7. P0 visual upgrades built (7 labs)

All are **dependency-free inline SVG + vanilla JS** on the repo `.vlab` template (CSS block copied verbatim from `m10a/day-01`) — no CDN, offline-safe, theme-aware, keyboard-reachable controls, inserted **above** the existing fake-terminal playground (which keeps its 3 `data-demo` completion-gate buttons). Each carries the 7 required cards (learning question · labeled axes · clear change on control · what-you-should-notice · misconception · where-it-simplifies · frontier relevance). All math was hand-verified in Node.

| Lesson | Lab | Interaction verified (jsdom + Node math) |
|--------|-----|------------------------------------------|
| m03 · day-02 qkv | **Projection fan** — one embedding → 3 labeled projections (W_Q/W_K/W_V) → Q/K/V boxes; a query-angle θ slider rotates the query and the q·key bars re-rank; token toggle. | slider → bars re-render; verified river@θ0 attends to bank, @θ90 flips to river ✓ |
| m03 · day-05 positional | **PE waves & fingerprint** — 8×8 PE grid (blue−1…red+1) + the sine/cosine wave for the selected dim; dim + pos sliders. | slider → grid + wave re-render; low dim fast / high dim flat confirmed ✓ |
| m04 · day-02 backward-pass | **Blame flows backward** — vertical loss→x flow with per-stage gradients revealed by a step slider; a ReLU-gate panel shows the dead unit's `dz1` gated to 0. | step slider → reveal; verified `z1=[2.5,−2]`, dead unit `dz1=0` ✓ |
| m05a · day-01 residual | **Two roads for the gradient** — log-scale gradient-vs-depth plot; depth-N slider + skip on/off toggle; plain line vanishes, residual survives. | slider/toggle → re-render; plain `0.2^N` vanishes, residual `1.2^N` alive ✓ |
| m05a · day-02 layer-norm | **Recenter & rescale** — raw dot-strip (auto-scaled) vs normalized dot-strip (fixed); spread + shift sliders; the normalized dots stay put — the invariance is the lesson. | slider → raw re-renders, μ/σ update, normalized fixed ✓ |
| m05a · day-05 causal-masking | **Who can see whom** — 6×6 attention grid (green allowed / grey blocked); row-i slider + correct-vs-off-by-one toggle that lights a leaked-future cell orange. | slider/toggle → re-render; off-by-one leaks cell `(i, i+1)` ✓ |
| m05a · day-07 text-generation | **Temperature dial** — 5-word probability bar chart; temperature slider re-heights `softmax(logits/T)` live; top-p slider greys the trimmed tail. | slider → bars re-render; verified top-prob 0.96→0.61→0.33 as T rises ✓ |

The **not-P0** hard-math attention lessons already embed genuinely interactive `viz/*.html` iframes (m03/day-01 `attention-heatmap`, m03/day-03 `attention-pipeline`+`softmax-scaling`, m03/day-04 `attention-multihead`, m04/day-04 `gradient-descent`) — per the plan and the "don't blindly convert every playground to D3" rule, those got Coach prose only, no new lab.

**P0 count note (7 vs Batch 1's 4):** the auditors flagged 4 on the pure rubric (positional, backward-pass, causal-masking, text-generation). The parent request explicitly named 5 mechanism visuals for this batch; three of them (Q/K/V flow, residual stream, layer-norm before/after) the auditors rated P1-adequate, but the directive elevated them to P0. All 7 scored 3 on the visual rubric (below the P0 threshold of 4). The 12 non-P0 lessons are genuinely terminal/static-appropriate (code translation, comparison, shape-flow, capstone assembly) or already interactive — none were gold-plated.

---

## 8. Audit results

**`lesson_audit.py` (targeted, `_recover_set.json` left unchanged):**
```
TOTAL lessons audited: 19
  OK: 19   MISSING: 0   LEFTOVER: 0   DEGRADED: 0
```

**jsdom render + JS-syntax gate:** 19/19 render clean, 0 runtime errors, 0 syntax errors; structural counts (7 sec / 7 got-it / 3 demos / 4 quiz / BUILD / tooltip) intact on all. The 7 P0 labs re-render on simulated `input`/`click`.

**Remaining soft COACH advisories (all expected, none a failure):**
- `math-heavy but no Math Ladder` ×3 — the intentional non-formula skips: `m04/day-05-pytorch-version` (API/idiom lesson), `m04/day-06-why-pytorch` (comparison finale), `m05a/day-06-encoder-vs-decoder` (contrast lesson — the difference is one `+causal_mask` line, not a single equation). Correct by design.

No other advisories fire (analogy, Staff Lens, `experiment.py`, pain-point, interview, acceptance, research log, bilingual all cleared on all 19).

---

## 9. Verification method

1. `vm.Script` syntax check on every inline `<script>` (via `/tmp/batch1_verify.js`).
2. jsdom 29.1.1 headless load (no thrown error) + shell-agnostic structural assertions.
3. For P0 labs: `--p0` dispatches `input`/`click`, asserts the `.vlab` SVG re-renders; hand-verified each lab's core math in Node (Q/K/V winner-flip, PE frequencies, backprop ReLU gate, residual vanish/survive, temperature reshaping).
4. `lesson_audit.py` per module and combined (19/19 OK).
5. `git diff` audit: additive callouts + one `.vlab` CSS block + new IIFE scripts per P0 file; confirmed no `data-quest-id`/`data-demo`/`DEMOS`/`QS`/`BUILD`/nav-`href`/`data-sec`/`.gotit` line removed.

Reusable harness: `/tmp/batch1_verify.js` (`node /tmp/batch1_verify.js <lesson.html> [--p0]`); jsdom at `/tmp/node_modules/jsdom`.

---

## 10. Remaining risks / notes

1. **No real-browser pixel check.** All verification is headless (jsdom) + math re-derivation; per `no-real-browser-in-sandbox`, a final Light/Dim eyeball of the 7 new labs should be a user screenshot if pixel fidelity matters. The SVGs reuse the `.vlab` classes proven in Batch 1.
2. **P0 labs are teaching idealizations.** Each lab's `.vlab-note` states its simplification. Two carry a deliberate bridge to the *next* concept: the residual lab notes that the raw `(1+r)^N` road would explode and that LayerNorm (day-02) tames it; the layer-norm lab's normalized dots are intentionally invariant to the sliders (the pedagogical point), which could read as "broken" without the note — the note calls it out.
3. **3 Math-Ladder advisories persist by design** (pytorch-version, why-pytorch, encoder-vs-decoder). Documented above, not a failure.
4. **Concurrent edits** to `sessions/` remain possible; each file was re-read immediately before editing (the harness enforces "file modified since read").

---

## 11. Recommended next rollout order

Batch 2 (transformer & first-model) is done. Per the pilot's rollout order, the transformer & training tier continues, then:

3. **Inference systems** — `m16a`, `m17a/b`, `m06`.
4. **Frontier model-literacy** — `m14*`, `m15*`, `m19`–`m21`.
5. **Research-engineering & portfolio** — `m23`–`m29`.

Reuse this batch's mechanics: the `.vlab` CSS block + drawing-IIFE pattern for any new P0 lab, `/tmp/batch1_verify.js` + `lesson_audit.py <path>` as the per-file gate. Keep bilingual light (~3 touches), Math Ladder only for genuine formula lessons (compact when a full derivation already exists), and add Staff Lens / interview only where missing.

---

## 12. Definition of done — Batch 2

- [x] Visual audit run (3 read-only module auditors, rubric-scored) before any edit; P0 set identified (7 lessons, documented rationale).
- [x] `batch_2_rollout_plan.md` written **before** any lesson was modified.
- [x] All 19 lessons: missing Coach elements added.
- [x] 7 P0 interactive labs built (dependency-free, offline-safe, verified interactive + math-checked).
- [x] Navigation, quest-ids, quiz, playground, BUILD, tooltips, shell structure preserved (diff-verified additive; 0 protected lines removed).
- [x] Bilingual scaffold present but light (~3 touches/lesson, intuition only).
- [x] Math Ladder for every formula-heavy lesson (16); skipped + documented for the 3 non-formula lessons.
- [x] One factual fix (positional-encoding fast/slow) made and verified.
- [x] No motivational fluff; technical depth preserved or increased.
- [x] `lesson_audit.py`: 19/19 OK, 0 DEGRADED. jsdom: 19/19 clean. All 7 P0 labs interactive.
- [x] `batch_2_rollout_report.md` written (this file).
