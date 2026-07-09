# v8 Phase C — Compiler Hardening Report

**Loop:** `v8_phase_c_compiler_hardening`
**Date:** 2026-07-08
**Owner skills:** `frontier-curriculum-architect`, `frontier-lesson-builder`, `frontier-refactor-qa`, `frontier-visual-evidence-builder`
**Scope:** Compiler hardening only. **Day 1 recompiled. Day 2–9 NOT sourced. m03 NOT rolled out.**
**Spec:** `sessions/_refactor/v8_source_first_authoring_plan.md` · **Pilot:** `sessions/_refactor/v8_phase_b_m02_day1_pilot_report.md`

---

## TL;DR

- **Verdict: PASS.** The Day 1 source-first path is now **reusable** for Day 2–9. The compiler was split into a shared core + two standalone gates, and all five raw-HTML widget escapes in Day 1 were replaced with **typed source blocks**.
- **Day 1 still compiles deterministically** (twice → byte-identical) and the compiled `lesson.html` **preserves every frozen invariant**, with CSS + all 4 `<script>` blocks **byte-identical to the donor**.
- All shell/audit checks pass: Reader Flow Gate, Shell Invariant Gate, `lesson_audit` (9 OK / 0 advisories), `nav_audit` (PASS), `staff_lens_audit` (9/9, gap 0), `node --check` (all parse), and 11/11 widget spot-checks.
- **Remaining issues are P1/P2 only** (§6).

---

## 1. What changed (three moves)

### 1.1 Extracted a shared core — `sessions/_compiler/v8lib.py`
All parsing, the markdown-lite renderer, the typed-widget renderers, the region renderers, and the marker-based `compile_html()` now live in one importable, side-effect-free module. Every function is pure, so compilation stays deterministic.

### 1.2 Extracted the two gates into standalone + importable modules
```
sessions/_compiler/gates/reader_flow_gate.py       # Reader Flow Gate (on source)
sessions/_compiler/gates/shell_invariant_gate.py   # Shell Invariant Gate (on output; optional donor identity)
```
Each exposes `run(...) -> (ok, msgs)` **and** a CLI:
```
python3 sessions/_compiler/gates/reader_flow_gate.py <source.md>            # exit 0 / 2
python3 sessions/_compiler/gates/shell_invariant_gate.py <lesson.html> \
        --source <source.md> --donor <donor>                               # exit 0 / 3
```
`compile_lesson.py` is now a **thin orchestrator**: parse → Reader Flow Gate (block write on fail) → `compile_html` → Shell Invariant Gate (incl. donor shell-identity) → write. The gate that previously lived inline in the compiler is gone; the compiler imports the standalone module, so the gate logic has exactly one home.

### 1.3 Replaced raw-HTML escapes with typed source blocks
The Phase B pilot left five complex widgets as `~~~html` raw escapes. Phase C adds a **reusable typed-block vocabulary** and converts all five:

| Widget | Old (Phase B) | New typed block | Reused across… |
|---|---|---|---|
| Brain↔ANN mapping table | `~~~html` | `%%% table` (`::`-separated, 3-col) | every comparison table |
| Judge everyday-analogy cards | `~~~html` | `%%% cards` (`emoji \| title \| body`) | every `.relate` analogy |
| Formula callout (🧮) | `~~~html` | `%%% formula` (`expr:` / `note:`) | every formula callout |
| Math Ladder (🪜) | `~~~html` | `%%% mathladder` (`title/words/formula/numbers/sanity`) | **every Coach-Layer lesson** |
| Produce prompt box | `~~~html` | `%%% prompt id= label=` + verbatim text | every Produce Option-B |

Plus the existing `%%% jargon`. **Day 1 source.md now contains zero raw-HTML escapes.** The `~~~html` escape is retained in `v8lib` as a documented fallback for one-off widgets not yet typed.

**Donor-shell strategy kept.** No safer abstraction was justified: reusing the pristine donor snapshot + marker replacement is exactly what guarantees byte-identical CSS/JS, and it works. Only the *content-region renderers* were hardened.

---

## 2. Determinism & shell identity (preserved)

- **Deterministic:** two consecutive compiles → `diff` byte-identical.
- **Shell Invariant Gate now asserts donor identity automatically** (with `--donor`): CSS block **identical**; all **4 `<script>` blocks identical** (incl. the JS engine with `DEMOS/BUILD/QS`). Only the reader-flow content regions (hero, s1, s2, s4, s7) differ from the donor — as intended.
- Compiled size: 73,492 chars (vs Phase B's 73,624 — typed blocks emit slightly tighter HTML than the old raw indentation; semantically identical, renders the same).

---

## 3. Invariants verified on the recompiled Day 1

`data-quest-id="wf2-d01-neuron"` (compiler asserts against the donor, never invents) · 7 `.module-section` · 8 sidebar `data-target`s · `DEMOS`/`BUILD`/`QS` present · playground ≥3 · **quiz q:4 o:16** · `frontier-lesson:<qid>` + `frontier-theme` keys · 7 `.gotit` · `.fin` banner · prev/next nav hrefs · `experiment.py` artifact referenced · **no unresolved markers** (`{{`, `@@@`, `%%%`) · `experiment.py` + `log.md` untouched.

---

## 4. Gate & audit evidence

| Check | Result |
|---|---|
| Reader Flow Gate (standalone, on source) | **PASS** — 9/9 (human-first hero, Jargon Ladder, picture-before-vocab, frontier deferred to s4, spine hero+s1+s2, discovery produce) |
| Shell Invariant Gate (standalone, on output + donor) | **PASS** — all invariants + CSS/4-script byte-identity |
| `python3 sessions/lesson_audit.py m02-the-neuron` | **9 OK / 0 MISSING / 0 LEFTOVER / 0 DEGRADED / 0 advisories** |
| `python3 sessions/nav_audit.py` | **PASS** — 0 CASE, 0 BROKEN, 0 orphans |
| `node sessions/staff_lens_audit.js m02` | **9/9 present, gap 0**; day-01 `fail:1 trade:1 q:4 o:16 errs:[]` (`render:BROKEN` benign) |
| `node --check` on all 4 inline scripts | **all parse** |
| Typed-widget spot-checks (table/cards/formula/mathladder/prompt) | **11/11 OK** |

All lesson_audit **coach advisories stay at zero**: the typed blocks preserve every keyed phrase — `.relate`/`.card` (cards), the literal "Math Ladder" text + 🧮 (formula/mathladder), `experiment.py`, "the staff lens", 🎤 interview, "What you should see", the 😕 pain-point, 📓 5-minute log, and the Chinese bilingual touches.

---

## 5. What is now reusable for Day 2–Day 9

The Day 1 path is now lesson-agnostic. To source any further day, an author needs only:

1. **A `source.md`** — front-matter (`quest_id`, `donor`, nav, sidebar, `fin_*`, `mode`, `spine`) + reader-flow `@@@` blocks (`hero`, `section s1/s2/s4/s7`, `js DEMOS/BUILD/QS`), using the typed widget vocabulary (`%%% jargon/table/cards/formula/mathladder/prompt`) and `!!!` callouts — no hand-written HTML for standard widgets.
2. **A donor snapshot** — `sessions/_compiler/shells/<module>-day-NN.donor` (pristine shell of that day's current `lesson.html`, stored without `.html` so `nav_audit` skips it).
3. **One command** — `python3 sessions/_compiler/compile_lesson.py <source.md>`, which runs both gates and compiles.

`v8lib.py`, the two gate modules, and the typed-block renderers are all shared and unchanged per-day. **Nothing in the compiler is Day-1-specific** except the donor filename and quest-id, both read from front-matter. The Reader Flow Gate reads the `spine` word from front-matter, so it adapts per lesson.

---

## 6. Remaining issues (P1 / P2 — none blocking)

- **P1 — Day 2–9 not yet sourced** (Phase D). Each needs a donor snapshot + a `source.md`. The path is proven; only authoring remains.
- **P1 — s3/s5/s6 scaffolding still carried from the donor.** Their *content/data* (`DEMOS/BUILD/QS`) is authored in `source.md`; the static wrapper HTML (play buttons, `#build`, `#quiz` divs) is preserved shell. Typing these is a small Phase-D/E follow-up.
- **P2 — `~~~html` fallback retained.** Not used in Day 1 anymore, but kept in `v8lib` for genuinely one-off widgets a future day might need before a typed block exists.
- **P1 (carried, unchanged):** behavioral-viz `BL-VE-D1` / `BL-VE-D2` are unaffected by this hardening.

**No P0.** m02 remains `pass_with_p1`. No other module status changed. m03 untouched.

---

## 7. File map (Phase C)

```
sessions/_compiler/
  compile_lesson.py            # thin orchestrator (imports v8lib + both gates)
  v8lib.py                     # shared core: parsing, markdown-lite, typed widgets, compile_html
  gates/
    reader_flow_gate.py        # standalone CLI + importable run()
    shell_invariant_gate.py    # standalone CLI + importable run() (+ optional donor identity)
  shells/
    m02-day-01.donor           # pristine v7.6 shell snapshot (no .html ext -> nav_audit skips)
    m02-day-01.v7_6-backup     # rollback copy
sessions/m02-the-neuron/day-01-single-neuron/
  source.md                    # canonical source — now zero raw-HTML escapes
  lesson.html                  # compiled (deterministic)
```
