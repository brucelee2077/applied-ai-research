# Coach Layer Pilot — Report

_Pilot run of the `frontier-curriculum-refactor` Coach Layer on 4 lessons. Date: 2026-07-06._

Companion docs: [`COACH_STYLE_GUIDE.md`](./COACH_STYLE_GUIDE.md) (house style + block templates), [`enhance_lesson_checklist.md`](./enhance_lesson_checklist.md) (per-lesson checklist), updated [`lesson_audit.py`](./lesson_audit.py) (automated gate).

---

## 1. TL;DR

Applied the Coach Layer to **4 pilot lessons only**, purely **additively** — no depth removed, every interactive mechanism preserved. All four now pass a shell-agnostic jsdom + JS-syntax render check and the rewritten `lesson_audit.py` with **zero hard failures and zero COACH advisories**. Chinese is now treated as a bilingual scaffold, not a defect. A live concurrent shell-migration ran during the work; the approach was designed to compose with it and did.

---

## 2. Files changed

| File | Change |
|------|--------|
| `sessions/COACH_STYLE_GUIDE.md` | **new** — house style, bilingual rule, 15-element map, reusable block templates, shell + preservation contract |
| `sessions/enhance_lesson_checklist.md` | **new** — per-lesson checklist (warmth, math, staff depth, artifact, coherence, preservation, verify) |
| `sessions/coach_layer_pilot_report.md` | **new** — this report |
| `sessions/lesson_audit.py` | **rewritten** — glob discovery, shell-agnostic counts, Chinese-not-degraded, COACH soft-warnings, path-arg mode (original recoverable via `git show HEAD:sessions/lesson_audit.py`) |
| `sessions/m01-shape-of-data/day-05-logs-and-exponents/lesson.html` | Coach Layer applied |
| `sessions/m08-transformer-math/day-01-transformer-arithmetic/lesson.html` | Coach Layer applied + migration-bug fixes |
| `sessions/m08-transformer-math/day-03-kv-cache/lesson.html` | Coach Layer applied + migration-bug fixes |
| `sessions/m10a-scaling-laws/day-04-power-law-derivation/lesson.html` | Coach Layer applied + migration-bug fixes |

Explicitly **not** touched: any lesson outside the pilot; the DEMOS/BUILD/QS interactive data; the navigation, quest-ids, progress/localStorage, quiz, playground, or tooltip machinery.

---

## 3. Before → after, per lesson

Verified with a shell-agnostic jsdom harness (`/tmp/coach_verify.js`) + `lesson_audit.py`. `zh` = Chinese characters (bilingual scaffold); `coach{}` = presence of the 6 Coach markers (pain-point, Math Ladder, interview, acceptance, research-log, staff-lens).

| Lesson | Before | After |
|--------|--------|-------|
| m01 · day-05 · logs & exponents | `zh=0  coach{pain:0 ladder:0 interview:0 accept:0 log:0 staff:1}` | `zh=83  coach{all 6 = 1}` — render OK |
| m08 · day-01 · transformer arithmetic | `zh=0  coach{pain:0 ladder:0 interview:0 accept:1 log:0 staff:1}` | `zh=98  coach{all 6 = 1}` — render OK |
| m08 · day-03 · KV cache | `zh=0  coach{pain:0 ladder:0 interview:0 accept:1 log:0 staff:1}` | `zh=102 coach{all 6 = 1}` — render OK |
| m10a · day-04 · power-law derivation | `zh=0  coach{pain:0 ladder:0 interview:0 accept:1 log:0 staff:1}` | `zh=135 coach{all 6 = 1}` — render OK |

Every lesson still reports **7 sections / 7 got-it / 3 demos / 4 quiz / 16 options / 0 empty / BUILD present / tooltip element created / 0 JS errors / 0 syntax errors** after the edits — i.e. all mechanics are byte-behavior-identical.

---

## 4. What became warmer (Reader A)

- **Bilingual pain-point** (😕 "为什么这里容易卡住 / Why this trips people up") opens the confusion out loud in section 1 of each lesson — e.g. "people memorize `6ND` but can't say why it's 6", "people frame the KV cache as a compute optimization and miss that it's the memory bottleneck".
- **Bilingual intuition line** (直觉 / Intuition:) in section 2: Chinese carries the feeling, English carries the mechanism — e.g. the KV cache = "边读书边做笔记 / taking notes as you read".
- Warmer, lesson-correct **finales** replacing the wrong "Module 1 · Day 3 … broadcast" text.

The Chinese is deliberately **light** (83–135 chars/lesson, 2–4 touches, intuition only) so the English-first reader is scaffolded, not swamped — per the `prefers-english-learning-content` guidance and the skill's bilingual rule.

## 5. What Math Ladder was added (Reader B)

A 4-rung 🪜 Math Ladder (words → labeled formula → tiny numbers → sanity check) now anchors the core formula of each lesson:

| Lesson | Ladder target | Tiny-number rung | Sanity-check rung |
|--------|---------------|------------------|-------------------|
| logs | `y=a·x^b → log y = log a + b·log x` | `y=x²`, x=1,10,100 → slope 2 | slope must equal the power, else you forgot to log an axis |
| 6ND | `C ≈ 6·N·D` | 10M·1B·6 = 6e16 → ~60 s on an H100 | GPT-3 → 3.15e23; use params not bytes |
| KV cache | `2·L·H·d·s·b·bytes` | 7B → 262,144/token → 4.3 GB at 8k | linear in seq_len; GQA/MQA share KV heads |
| power law | `a = Δ log N / Δ log C` | two Table-3 vertices → a≈0.49 | `a+b≈1` because `C≈6ND` |

## 6. What analogies were added

The lessons already had strong everyday analogies (Richter scale, party handshakes, note cards, tire pressure). The Coach Layer **added the software/systems framing** where it was thin — e.g. logs as "log-space to prevent underflow" in code, and 6ND as "N = weight-matrix size, D = tokens flowing through, one step = forward + backward over a batch". The existing "where the analogy breaks" callouts were kept.

## 7. What Staff Lens was added

All four lessons **already had** a Staff Lens (named silent failure mode + trade-off, tied to code/design review) from a prior retrofit — this was **preserved intact**. The Coach Layer added the matching **interview-ready block** (🎤 "Say this in an interview") that turns the staff-lens content into a 20–30-second spoken answer with a headline + one proof + one nuance.

## 8. What artifacts improved

- Every lesson keeps its concrete `experiment.py` Produce task (Option A hand-write / Option B frontier-experiment-lab prompt) — untouched.
- Added an explicit **Acceptance criteria** list to m01/day-05 (the only pilot lacking one).
- Replaced the one-line `log.md` nudge with a **5-minute research log** block (📓, bilingual) asking for 3 concrete lines (teammate summary / surprising number / open question), folding in each lesson's original log question.

---

## 9. Audit results

`lesson_audit.py` was **rewritten** and is the automated gate. The original is recoverable via `git show HEAD:sessions/lesson_audit.py`.

**Why it needed a rewrite (two independent bugs):**
1. Stale discovery — it looked for `week-NN/slug.html`, a layout that no longer exists, so it reported **110 MISSING** and audited nothing real.
2. Shell blindness — it counted `class="sec"`, which is **0** in the new sidebar shell, so it would have **falsely marked every migrated lesson DEGRADED** (`sections=0`).

**What changed:**
- Discovery by glob over the real layout (`m*/**/lesson.html` + `week-*/**/lesson.html`); optional path arguments to audit specific files.
- Structural counts are shell-agnostic (`data-sec=`, `class="gotit"`, `data-demo=`, the quiz literal), so both the old (`.sec`) and new (`.module-section`) shells pass.
- **Chinese is no longer a defect.** Present → an informational "bilingual scaffold" note; absent → a soft COACH advisory to add one.
- New **COACH advisories** (soft, never fail the run): no analogy, math-heavy without a Math Ladder, no Produce artifact, no Staff Lens, no interview-ready explanation, no Acceptance criteria, no pain-point block, no research log, no bilingual scaffold.
- Path-arg (targeted) runs do **not** overwrite `_recover_set.json`, so a pilot run never disturbs another session's state.

**Result on the 4 pilots (targeted run):**
```
TOTAL lessons audited: 4
  OK: 4   MISSING: 0   LEFTOVER: 0   DEGRADED: 0
(no COACH advisories remain — all 4 satisfy pain-point, Math Ladder, interview, acceptance,
 research-log, staff-lens, analogy, artifact, and bilingual scaffold)
```
Baseline (pre-edit) COACH advisories that are now cleared: missing Math Ladder ×4, missing interview-ready ×4, missing pain-point ×4, missing research log ×4, missing bilingual scaffold ×4, missing acceptance criteria ×1.

**Render/syntax gate (jsdom 29.1.1 + `vm.Script`):** 4/4 render clean, 0 JS errors, 0 syntax errors.

---

## 10. Adversarial verification

Ran a 4-lens verification workflow over the 4 edited lessons: **technical accuracy**, **depth preservation / no-fluff**, **bilingual quality & dosage**, **HTML & interactivity integrity**. Each material finding was re-checked by an independent adversarial verifier (default: refute). 5 agents, ~754k tokens.

| Lens | Verdict | Findings |
|------|---------|----------|
| Depth preservation | ✅ Clean | Strictly additive; every pre-existing Staff Lens, causal chain, complexity claim, failure mode, and Produce task intact; added blocks are warm-but-rigorous, not fluff. **0 findings.** |
| Bilingual quality & dosage | ✅ Clean | All 12 Chinese touches (3/lesson) natural, grammatical, accurate; confined to intuition; never leak into formulas or interview phrasing; light dosage. **0 findings.** |
| HTML & interactivity | ✅ Clean | No structural breakage; nav, quest-ids, quiz, playground, BUILD, tooltips all preserved. **0 findings.** |
| Technical accuracy | ✅ w/ 1 fix | All **new** arithmetic (Math Ladders, interview blocks, pain-points) re-derives correctly. **1 confirmed material defect** (below) + 2 low-severity pre-existing nits. |

**Confirmed material finding — FIXED.** In `m10a/day-04`, the pre-existing `DEMOS.ninepoint` playground demo (which the refactor had deliberately preserved) labeled the IsoFLOP vertex-chaining method as Chinchilla **"Approach 1" (a=0.50, b=0.50)** — but that method is the paper's **Approach 2 (IsoFLOP profiles), a≈0.49, b≈0.51**, exactly as the lesson's own quiz Q3 states. This was an internal contradiction (demo vs. quiz vs. the new Math Ladder/interview blocks). Fixed with two string edits inside the demo: relabeled to **"Approach 2 (IsoFLOP)"** and aligned the numbers to **a≈0.49, b≈0.51**, matching the rest of the lesson. Verified: 0 remaining "Approach 1" occurrences; render still clean.

**Low-severity, pre-existing, left as-is (not caused by the Coach Layer):**
- The two-point demo displays `0.83 / 1.67 ≈ 0.494`; using the 2-decimal-rounded log intermediates the fraction is ~0.497, but the true slope (from unrounded logs) is 0.4946 ≈ 0.494, which the verifier itself re-derived — so `0.494` is correct and consistent lesson-wide. No change.
- The two-point demo uses the Gopher budget (`5.76e23`), above the stated 6e18–3e21 IsoFLOP sweep range — a legitimate extrapolation example, pre-existing, non-material. No change.

**Net:** every dimension passes; the one confirmed defect (a pre-existing contradiction the review surfaced) is fixed and re-verified.

---

## 11. The concurrent shell migration (important context)

During this work, a separate session ran `_shell_migrate.py --apply-all`, converting the whole `sessions/` tree from the old top-nav shell (`class="sec"`) to the new sidebar + appearance-switcher shell (`class="module-section"`, Light/Dim/Dark/Midnight). m01/day-05 flipped shells mid-task (its first Edit hit a "file modified since read" and was re-read).

This was **anticipated and composed with**, not fought:
- `_shell_migrate.py` migrates only the *shell* and re-extracts each `.sec-body`, the hero, and the `DEMOS/BUILD/QS` literals **byte-for-byte** from the live `lesson.html`. Coach Layer content lives exactly in those preserved regions, so it survives migration.
- The pre-existing `.new.html` preview files (all stamped 17:43) were left untouched; the migration applied and removed them.
- **Side effect discovered & fixed within the pilots:** the migrator substitutes a `<span class="eyebrow">` that the new template renders as `<span class="kicker">`, so the kicker never updated — every migrated lesson showed a leftover "Module 1 · Represent · Day 3" kicker and a wrong "broadcast" finale. Fixed in the 4 pilot files (authoritative source = `<title>` + `.brand-sub`).

---

## 12. Remaining risks

1. **Migrator kicker/finale bug is curriculum-wide.** Every lesson migrated by `_shell_migrate.py` likely has the wrong kicker (`Module 1 · Represent · Day 3`), a wrong `nav-group-label`, and the "broadcast" finale — the pilot fixed only its 4 files. **Recommended:** patch `_shell_migrate.py` to substitute `<span class="kicker">` (and re-derive the finale) before it processes the rest, or run a one-off sweep. Track separately from the Coach Layer rollout.
2. **Bilingual dosage is a judgment call.** The pilot uses a light touch (Chinese for intuition only). If the learner wants more or less Chinese, adjust the block templates in `COACH_STYLE_GUIDE.md` §3 before rolling out.
3. **`lesson_audit.py` full-scan** (no args) still owns `_recover_set.json`; run it only when you intend to refresh that file. Targeted runs are safe.
4. **Concurrent edits** to `sessions/` remain possible; re-read before editing (the tooling enforces this).

---

## 13. Recommended rollout order

Pilot passed → proceed in this order (per the skill), one file per session, no parallel lesson writing (avoids tone/analogy drift), running `lesson_audit.py <path>` + the jsdom check after each:

1. **Math foundation modules** — `m01`, `m10a`, `m13` (logs, scaling laws, isoFLOPs).
2. **Transformer & training modules** — `m02`–`m05`, `m08`, `m09*`, `m12`.
3. **Inference systems modules** — `m16a`, `m17a/b`, `m06`.
4. **Frontier model-literacy modules** — `m14*`, `m15*`, `m19`–`m21`.
5. **Research-engineering & portfolio modules** — `m23`–`m29`.

Do the migrator kicker/finale sweep (risk #1) **before or alongside** step 1 so the rollout isn't copying the Coach Layer onto lessons with broken chrome.

---

## 14. Definition of done — pilot

- [x] Lessons still open and render (jsdom, 4/4 clean).
- [x] Navigation, quest-ids, quiz, playground, build, tooltips preserved (verified).
- [x] All 15 Coach Layer elements present in each pilot.
- [x] Bilingual scaffold present but light.
- [x] Technical depth preserved or increased (nothing removed).
- [x] Each lesson asks for a concrete artifact with acceptance criteria.
- [x] `lesson_audit.py` reports the 4 pilots OK with no hard failures; Chinese not degraded.
- [x] Adversarial verification workflow findings addressed — see §10 (1 confirmed defect fixed; all lenses pass).
