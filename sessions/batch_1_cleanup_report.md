# Batch 1 — Cleanup Pass Report

_Cleanup-only pass for the `frontier-curriculum-refactor` Coach Layer work. No lesson content was refactored and no new visuals were added — this pass only reformats docs, reconciles one inconsistent constant, fixes text-insertion glitches, and re-audits. Date: 2026-07-06._

Companion docs: [`visual_audit_report.md`](./visual_audit_report.md) (source of the H100 finding) · [`batch_1_rollout_plan.md`](./batch_1_rollout_plan.md) · [`batch_1_rollout_report.md`](./batch_1_rollout_report.md) · [`COACH_STYLE_GUIDE.md`](./COACH_STYLE_GUIDE.md) · [`enhance_lesson_checklist.md`](./enhance_lesson_checklist.md) · [`lesson_audit.py`](./lesson_audit.py).

---

## 1. TL;DR

Ran a scoped cleanup over the Coach Layer curriculum: (1) verified the five refactor Markdown docs already render cleanly on GitHub — no reformatting needed; (2) reconciled the H100 throughput mismatch in `m08/day-01` flagged by `visual_audit_report.md` §3.2/§4, so both in-lesson calculators now state and agree on which assumption they use; (3) fixed text-insertion glitches — one Cyrillic stray character and **six truncated lesson finales** in the audited set; (4) re-audited the 4 pilot lessons + 18 Batch 1 lessons — **20/20 OK, 0 DEGRADED**, all edited files pass syntax + headless (jsdom) checks. **9 files changed, all edits ≤ 12 lines, purely additive/label-level — no navigation, quest-id, quiz, `DEMOS`, `BUILD`, or tooltip machinery touched.**

---

## 2. Files changed

| File | What changed | Task |
|------|--------------|:---:|
| `m08-transformer-math/day-01-transformer-arithmetic/lesson.html` | Relabeled `1e15` as "idealized teaching estimate" in all 6 spots (inline lab note, Math Ladder, Produce prompt, `BUILD` scroll-reveal ×2, inline-lab SVG label); reconciliation note now names both constants with the preferred labels | 2 |
| `viz/transformer-flops.html` | Readout label + code comment now state the realistic effective throughput explicitly: `989 TFLOP/s × 40% MFU ≈ 3.96×10¹⁴` FLOP/s | 2 |
| `m01-shape-of-data/day-01-arrays/lesson.html` | Finale `Numbers, Arrays,` → `Numbers, Arrays, and the Shape of Data` | 3 |
| `m10a-scaling-laws/day-03-isoflops-methodology/lesson.html` | Finale `The IsoFlops` → `The IsoFlops Methodology` | 3 |
| `m10a-scaling-laws/day-06-scaling-simulator-visualization/lesson.html` | Finale `Scaling Simulator` → `Scaling Simulator & Visualization` | 3 |
| `m13-isoflops-scaling-law/day-01-isoflops-experiment-design/lesson.html` | Finale `IsoFLOPs` → `IsoFLOPs Experiment Design` | 3 |
| `m13-isoflops-scaling-law/day-04-plotting-parabolas/lesson.html` | Finale `Plotting` → `Plotting the Parabolas` | 3 |
| `m13-isoflops-scaling-law/day-05-deriving-scaling-law/lesson.html` | Finale `Deriving the` → `Deriving the Scaling Law` | 3 |
| `batch_1_rollout_plan.md` | §5.1 removed a stray Cyrillic word (`stacked/див bar` → `stacked bar`) | 3 |
| `batch_1_cleanup_report.md` | This report (new) | 5 |

**Explicitly NOT touched:** any `data-quest-id`, prev/next/hub `href`, `data-sec` key, `.gotit` button, quiz literal (`var QS`), playground literal (`var DEMOS`), or scroll-reveal literal (`var BUILD`) array shape. The one `BUILD` edit in `m08/day-01` changed only a comment string *inside* an existing element (the array's length, keys, and structure are byte-identical); the `1e15` teaching value itself was kept, so the scroll-reveal's worked numbers stay correct.

---

## 3. Task 1 — Markdown docs render cleanly on GitHub

Audited all five requested docs for GitHub-Flavored-Markdown rendering hazards. **Result: all five already render cleanly — no formatting changes were required.**

Checks run (all passed on every file):

- **Code fences balanced** — every fenced code block opens and closes (even number of fence markers); no unterminated blocks.
- **Tables** — each table has a preceding blank line and a valid separator row; no cell contains an unescaped `|`.
- **Emphasis** — the `_..._` lines are intentional whole-line italic subtitles; every filename underscore (`lesson_audit.py`, `_recover_set.json`, …) sits inside backticks, so no accidental italics.
- **No bare HTML tags** outside code spans (e.g. `<h4>…</h4>` mentions are all inside backticks or fenced blocks).
- **No accidental indented code blocks** (no stray 4-space / tab line starts), **no BOM, no CRLF**, trailing newline present.

The one edit to a doc in this set (`batch_1_rollout_plan.md`) was a copy glitch, not a formatting issue — see §5.

> Note: `visual_audit_report.md` still describes the H100 mismatch as an open finding (it is a dated, point-in-time snapshot that states "no `lesson.html` file was modified to produce this report"). Per the "preserve meaning" instruction it was left as a historical record; this report supersedes it on that point — see §4.

---

## 4. Task 2 — H100 throughput reconciled

**The problem (from `visual_audit_report.md` §3.2/§4):** the lesson had two calculators modeling the same GPU with a ~2.5× gap — the inline lab, Math Ladder, and experiment prompt used `~1e15` FLOP/s, while the embedded `viz/transformer-flops.html` used `989e12 × 0.40 MFU ≈ 3.96×10¹⁴` FLOP/s.

**A second, sharper defect surfaced during the fix:** the lesson *contradicted itself*. The reconciliation note already present in Section 5 called `1e15` the "simpler round number" and `3.96×10¹⁴` the "realistic" one — yet the inline-lab note, the Produce prompt, and the `BUILD` scroll-reveal all described `1e15` as "realistic utilization" / "real utilization". Same number, opposite labels.

**Fix (the preferred labeling, applied consistently):**

- `1e15` FLOP/s is now labeled **"idealized teaching estimate"** everywhere it appears in the lesson (one H100's bf16 peak, rounded for clean hand-arithmetic).
- `989 TFLOP/s × 40% MFU ≈ 3.96×10¹⁴` FLOP/s is now labeled **"realistic effective throughput at 40% MFU"** in the reconciliation note and stated explicitly in the calculator itself (readout label + source comment).
- Both calculators now say which assumption they use, and the reconciliation note keeps the "~2.5× longer wall-clock" explanation so nothing was silently changed. The old rounded `≈ 4×10¹⁴` in the note was tightened to `≈ 3.96×10¹⁴` to match the calculator exactly.

No numeric value was changed — the `1e15` teaching value and the calculator's `989e12 × 0.40` were both kept; only the *labels* were made explicit and consistent.

Verification grep confirms zero remaining "real/realistic utilization" labels on `1e15`, and both files now carry the "realistic effective throughput" / "idealized teaching estimate" wording.

---

## 5. Task 3 — Copy polish

- **`"Numbers, Arrays,and the Shape of Data"`** — the exact no-space string from the request was **not present**; the lesson `<title>` already reads `Numbers, Arrays, and the Shape of Data`. But the same phrase was truncated in the day-01 **finale** (`you've completed <b>Numbers, Arrays,</b>.`), which is what the request was really pointing at. Fixed.
- **Systematic finale truncation (same insertion glitch).** Several finale banners had their topic name cut off mid-phrase (a dangling comma or article). Each was completed from the lesson's own authoritative `<title>` — a label/coherence fix explicitly permitted by `COACH_STYLE_GUIDE.md` §7, not a content rewrite. Six were in the audited set and are now fixed:

  | Lesson | Before | After (from `<title>`) |
  |--------|--------|------------------------|
  | m01 · day-01 | `Numbers, Arrays,` | `Numbers, Arrays, and the Shape of Data` |
  | m10a · day-03 | `The IsoFlops` | `The IsoFlops Methodology` |
  | m10a · day-06 | `Scaling Simulator` | `Scaling Simulator & Visualization` |
  | m13 · day-01 | `IsoFLOPs` | `IsoFLOPs Experiment Design` |
  | m13 · day-04 | `Plotting` | `Plotting the Parabolas` |
  | m13 · day-05 | `Deriving the` | `Deriving the Scaling Law` |

- **Cyrillic stray character.** `batch_1_rollout_plan.md` §5.1 had `a stacked/див bar` (the `див` is Cyrillic, an IME/paste artifact). Fixed to `a stacked bar`; meaning preserved.
- No prose was rewritten. Every fix is a one-token or one-phrase correction of an obvious insertion artifact.

---

## 6. Task 4 — Audit result

**`lesson_audit.py` (targeted run — `_recover_set.json` left unchanged), 4 pilot + 18 Batch 1 = 20 unique lessons:**

```
TOTAL lessons audited: 20
  OK: 20   MISSING: 0   LEFTOVER: 0   DEGRADED: 0
```

**Headless verification of all 9 edited files** (JS syntax via `vm.Script` on every inline `<script>` + jsdom 29.x load): **all PASS** — 0 syntax errors, 0 jsdom runtime errors, and every lesson still reports 7 sections / 7 got-it / 3 demos / `QS` / `BUILD` intact. The viz (`transformer-flops.html`, not a lesson) parses clean.

**Remaining soft COACH advisories (all pre-existing and documented-intentional — never a failure):**

- `math-heavy but no Math Ladder` ×5 — the non-formula lessons (m01 arrays/indexing/random-seeds, m13 automation/aggregation/documentation) that have no single core equation.
- `no experiment.py referenced` ×6 — lessons that legitimately produce a notebook / `.md` / LaTeX artifact instead of `experiment.py`.

These match `batch_1_rollout_report.md` §10 exactly; this pass did not touch them.

---

## 7. Remaining non-blocking risks

1. **The finale truncation is a systemic generator bug.** A repo-wide scan for the same dangling-topic signature found **five more truncated finales outside this pass's scope**, all in `m05a-text-transformer`:

   | File | Finale shows | Should be (from `<title>`) |
   |------|--------------|----------------------------|
   | m05a · day-02-layer-norm | `Keeping the` | Layer Normalization |
   | m05a · day-04-full-block | `Assembling the` | The Full Decoder Block |
   | m05a · day-06-encoder-vs-decoder | `Reader, Writer,` | Encoder vs Decoder |
   | m05a · day-07-text-generation | `Picking the` | Text Generation and Sampling |
   | m05a · day-08-full-transformer | `The Tower of` | The Full Transformer |

   These were **left unfixed** — they are outside the requested scope (4 pilot + 18 Batch 1 lessons) and in a module other sessions may be editing. The dangling-comma/article signature also won't catch mid-phrase cuts (like the `Plotting` / `IsoFLOPs` cases fixed here), so the true count curriculum-wide is likely higher. **Recommendation:** a follow-up sweep that regenerates every finale's `<b>topic</b>` from its `<title>`, then re-runs `lesson_audit.py`.

2. **No real-browser pixel check.** All verification here is headless (jsdom + syntax + grep). Per the `no-real-browser-in-sandbox` constraint, the two H100 label edits in `viz/transformer-flops.html` (a longer readout `<small>` line) should get a quick eyeball at Light/Dim if pixel fidelity matters — the text may wrap to a second line on narrow screens, which is cosmetic only.

3. **`visual_audit_report.md` now describes a resolved issue.** Its H100 finding (§3.2/§4/§5) is fixed as of this pass, but the file was intentionally left as a dated snapshot (Task 1 was formatting-only for it). This report is the current source of truth on that item.

4. **Concurrent edits.** `sessions/` is edited by other sessions; each file here was re-read immediately before editing, and all edits are additive/label-level, so a later shell migration (`_shell_migrate.py`) still extracts cleanly.

---

## 8. Definition of done — this cleanup pass

- [x] Five refactor docs verified GitHub-clean (fences, tables, emphasis, HTML tags, indent/BOM/CRLF) — no reformatting needed.
- [x] H100 throughput reconciled: `1e15` = "idealized teaching estimate", `3.96×10¹⁴` = "realistic effective throughput at 40% MFU"; both calculators state their assumption; no number silently changed.
- [x] Copy glitches fixed: 6 truncated finales (in scope) + 1 Cyrillic stray char; the named `Numbers, Arrays,` case addressed.
- [x] Audit: `lesson_audit.py` 20/20 OK, 0 DEGRADED; all 9 edited files pass syntax + jsdom.
- [x] No content refactor, no new visuals, no navigation/quest-id/quiz/playground/build/tooltip changes.
- [x] Out-of-scope systemic finale bug documented with exact locations for a follow-up sweep.
