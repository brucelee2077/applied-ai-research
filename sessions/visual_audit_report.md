# Visual Audit Report — 4 Coach Layer Pilot Lessons

_Fresh, independent re-audit of the current visual/playground state of the 4 Coach Layer pilot lessons. Date: 2026-07-06. Companion doc: [`coach_layer_pilot_report.md`](./coach_layer_pilot_report.md) §15 (the prior audit + the visual-lab build it triggered)._

---

## 1. Method

`coach_layer_pilot_report.md` §15 already claims a P0 rescue and three P2 polish passes were built for these 4 lessons. Rather than take that write-up at face value, this report re-audits the **current on-disk state** of each `lesson.html` from scratch: 4 independent agents, one per lesson, each blind to the others' findings, each required to cite evidence (element `id`s, variable names, literal text) for every claim and to run a real jsdom headless simulation (drag the slider, click the toggle, click the presets) rather than reason from source code alone. No `lesson.html` file was modified to produce this report.

---

## 2. Summary

| Lesson | Visual category | Score | Priority | Claim verified? |
|---|---|:---:|:---:|:---:|
| m01 · day-05 · logs & exponents | Interactive dual-panel SVG lab + retained fake-terminal exact-numbers panel | 4.5 | P3 | ✅ Yes |
| m08 · day-01 · transformer arithmetic (6ND) | Interactive dual-panel SVG lab + D3 iframe calculator | 4.5 | P3 | ✅ Yes |
| m08 · day-03 · KV cache | Interactive dual-panel SVG lab (slider-driven) + secondary canned-text button demo | 4.5 | P3 | ✅ Yes |
| m10a · day-04 · power-law derivation | Interactive dual-panel SVG lab (log-log vs raw-axis power-law fit) | 4.5 | P3 | ✅ Yes |

All four of `coach_layer_pilot_report.md` §15's "what was built" claims independently verified true — the interactive labs described there are real, wired, and produce correct output under simulated interaction, not just present in the source. All four lessons moved from their pre-build priorities (P0 / P2 / P2-high) to **P3** (polish-only, nothing substantive missing). One new, previously unflagged defect turned up during this pass — see §4.

---

## 3. Per-lesson detail

### 3.1 m01 · day-05 · logs & exponents

**Visual category:** Interactive dual-panel SVG lab + retained fake-terminal exact-numbers panel
**Score:** 4.5 / 5 · **Priority:** P3

**What works**

- A genuinely interactive two-panel SVG lab in Section 3: left panel (`#lg-left`) renders `y = a·x^b` on normal axes as a bending curve, right panel (`#lg-right`) renders the same data on log-log axes as a straight line, both driven live by an exponent slider (`#lg-exp`, range −1 to 3).
- Verified by simulated drag: moving the slider to `b=3` re-renders both panels; the slope readout (`#lg-b`) tracks the slider in lockstep; a drawn rise/run triangle's numbers check out (rise/run = 2 for `b=2`, matching the exponent).
- A working "log y only" misconception toggle (`#lg-semi`) swaps the right panel to a curve that stays curved, with the caption literally stating "still a curve" — a real demonstration of the trap, not decoration.
- Two working presets (`y=x²`, Kaplan `b=−0.05`) verified to move the slider and re-render correctly, including the correct sign flip for the negative exponent.
- A stated learning question, a "what you should notice" callout, a named misconception callout, and a "where this simplifies reality" caveat are all present.
- The old fake-terminal playground is retained below the lab and still does real work — it's the section's completion gate (the 3 buttons must be clicked to unlock "got it"), not dead leftover content.
- Internally consistent with Section 4's Math Ladder worked example and the Kaplan exponent cited in prose — no wording/number mismatch.

**What's missing**

- No drag-to-fit interaction on the plotted points themselves — the learner can only move a slider, not try their own eyeball slope estimate and get feedback on it.
- The left (normal-axes) panel autoscales its y-range on every drag, so the learner never visually sees how dramatically that axis range would explode as `b` grows if it were held fixed.
- Minor: Google Fonts `<link>` tags fail silently offline (cosmetic only, does not affect the lab's functionality).

**Next recommended upgrade:** Add a fixed/reference y-scale option (or a small inset) on the left panel so the axis-explosion is visible as evidence, rather than auto-scaled away every time the slider moves.

---

### 3.2 m08 · day-01 · transformer arithmetic (6ND)

**Visual category:** Interactive dual-panel SVG lab + D3 iframe calculator
**Score:** 4.5 / 5 · **Priority:** P3

**What works**

- The "decompose the 6" stacked bar chart is real and live (`#fl-left` / `#fl-right`): a 2-unit (forward, blue) + 4-unit (backward, orange) stack, redrawn by `renderLeft()`/`renderRight()`, not a static image.
- A genuinely functional inference (2ND) vs. training (6ND) toggle (`#fl-inf` / `#fl-train`): flips bar heights, colors, and the readout text between "6 FLOPs = 2 fwd + 4 bwd" and "2 FLOPs = forward only".
- N/D log-scale sliders default to 175B / 300B (≈ GPT-3's own N and D), driving a real H100-time readout; the 3× training-vs-inference ratio shown checks out arithmetically (6ND / 2ND = 3).
- Learning question, "what you should notice", named misconception, and simplification caveat are all present and substantive.
- The wording fix claimed in the prior report is verified: `sessions/viz/transformer-flops.html` now reads "2 forward + 4 backward" everywhere, matching the lesson (zero remaining contradictions found via grep across both files).
- No new external CDN dependency — the iframe's D3 usage is a vendored local file with a graceful fallback if it's missing.

**What's missing (new finding, not in the prior audit)**

- The lesson's **two** interactive artifacts disagree on the H100 throughput constant used for the wall-clock-time estimate: the inline lab, the Math Ladder, and the `experiment.py` prompt all use `1e15` FLOP/s, while the iframed `viz/transformer-flops.html` uses `989e12 × 0.40 MFU = 3.96e14` FLOP/s — roughly a **2.5× discrepancy** between two calculators in the same lesson modeling the same GPU.
- `viz/transformer-flops.html` loads Google Fonts via an external `<link>`, a soft dependency on being online (degrades gracefully to system fonts, but isn't fully self-contained as the rest of the lesson claims to be).

**Next recommended upgrade:** Reconcile the H100 throughput constant between the inline lab/Math Ladder (`1e15`) and the iframed calculator (`989e12 × 0.40`) so a reader who plays with both tools gets the same wall-clock-time answer for the same N/D inputs, instead of two answers ~2.5× apart for the same stated assumption.

---

### 3.3 m08 · day-03 · KV cache

**Visual category:** Interactive dual-panel SVG lab (slider-driven) + secondary canned-text button demo
**Score:** 4.5 / 5 · **Priority:** P3

**What works**

- A verbatim "predict before you drag" question opens the section, asking the learner to predict which of two panels grows (and by how much) before touching the slider.
- Two real range sliders (`#kv-seq` 512–8192, `#kv-batch` 1–16) drive a live recompute of both SVGs and two readouts via a real `input` listener — confirmed genuine interactivity, not a decorative slider.
- The claimed grey "weights ~14 GB" band is real: a rendered rectangle stacks the KV cache on top of a 14 GB weights baseline, next to an explicit "H100 HBM ≈ 80 GB" reference line — exactly the shared-ceiling context the prior audit said was missing.
- The left y-axis is now literally relabeled "relative work / step (schematic)", matching the caveat text.
- A named misconception callout ("the KV cache saves recompute, but it does not save memory — it adds it") and a simplification caveat are both present.
- Every number checks out end-to-end: the bytes-per-token formula matches the Math Ladder, the canned console growth numbers, the live slider defaults, and the quiz answer text.

**What's missing**

- The secondary three-button "playground" below the slider lab is still a canned-text swap on click — the input doesn't change any computed output, it just displays pre-written console text. It sits right next to a genuinely interactive lab, so the contrast in interactivity quality is visible.
- The "≈1024×" recompute-saved multiplier is a crude `round(seq_len/2)` heuristic with no derivation shown at the readout itself (only indirectly excused by the general schematic caveat).
- The 80 GB HBM ceiling number has no citation or source link inside Section 3 (minor, given the caveat block already flags it as schematic).

**Next recommended upgrade:** Re-badge the canned-text three-button demo as illustrative narration (e.g., rename the buttons to "see the trace") rather than implying it's live-interactive, since it's the one remaining rough edge next to an otherwise reference-quality lab.

---

### 3.4 m10a · day-04 · power-law derivation

**Visual category:** Interactive dual-panel SVG lab (log-log vs. raw-axis power-law fit)
**Score:** 4.5 / 5 · **Priority:** P3

**What works**

- The learning question ("why do the per-budget best-model-size points fall on a straight line only after you take logs — and what does the slope tell you?") is present verbatim.
- The exponent-`a` slider (`#pl-exp`, range 0.30–0.80) is genuinely wired: dragging it recomputes both fitted lines and redraws both panels live, rotating around a fixed pivot point — confirmed by re-deriving the pivot coordinates from the code.
- The live regime badge is real: it reports "≈ Chinchilla" for `a` in [0.44, 0.56], and the correct "leaning Kaplan" / "too much into D" warnings outside that band, with `implied b = 1−a` shown alongside.
- The raw-mode slope readout is now genuinely dynamic — independently recomputed in Node using the file's own fit logic and got ≈9.7e-13 at `a=0.49`, confirming it's a live computed value and not the previously stale, hardcoded `0.49`.
- "What you should notice", a named misconception callout, and a simplification caveat are all present and accurate.
- Fully consistent with the lesson's own prose: the default `a=0.49` sits inside the "Chinchilla" band; Kaplan's cited `a=0.73` correctly falls into the "warn" band.

**What's missing**

- The "`a+b` sanity check" is not actually a check — `b` is *defined* as `1−a` in the code, so `a+b` is tautologically 1.00 on every drag; it can never fail, so it doesn't demonstrate the real independent cross-check the prose describes.
- The "Highlight budget" control only spotlights one of the 5 plotted points; there's no way to add noise or vary the number of budgets, even though the caveat text says real fits use ~9 noisy points.
- The regime thresholds (0.44–0.56 "ok" band) are a hardcoded judgment call with no visible justification in the widget.

**Next recommended upgrade:** Make the `a+b` readout a genuine cross-check instead of a tautology — independently least-squares-fit `b` from a second synthetic series (e.g., derived from `D = C/(6N)` with its own small perturbation) so `a+b` can actually drift from 1.00 and the badge has something real to flag.

---

## 4. Cross-cutting finding

The one substantive new defect this pass surfaced — not present in the prior `coach_layer_pilot_report.md` §15 audit — is the **H100 throughput mismatch** in m08/day-01 (§3.2): the in-lesson lab and Math Ladder assume `1e15` FLOP/s while the iframed `viz/transformer-flops.html` assumes `989e12 × 0.40 MFU ≈ 3.96e14` FLOP/s, a ~2.5× gap for modeling the same GPU. Both numbers are individually defensible (a round order-of-magnitude assumption vs. a named-MFU realistic estimate), but having both live in the same lesson without reconciliation or acknowledgment is the kind of "two calculators, two answers" inconsistency the Coach Layer's staff lens is supposed to catch. Recommended fix is a one-line note in the lesson pointing to the difference, or aligning the two constants — no lesson.html edit was made as part of this report.

---

## 5. Remaining risks

1. All four labs are now P3 (polish-only) — the P0 rescue for m01 and the P2/P2-high gaps for the other three are all closed and independently re-verified.
2. None of the "what's missing" items above are correctness bugs; they are depth/rigor upgrades (a real cross-check instead of a tautology, re-badging a canned demo, reconciling two throughput constants).
3. This report did not re-run the Coach-Layer prose audit (bilingual dosage, Math Ladder, staff lens, etc.) — see `coach_layer_pilot_report.md` §§4–10 for that, which is unaffected by anything in this report since no `lesson.html` file was modified here.
4. `sessions/lesson_audit.py` run targeted on these same 4 files during this pass: **4/4 OK, 0 hard failures, 0 COACH advisories** (see §6).

---

## 6. Automated audit gate (targeted run)

```
TOTAL lessons audited: 4
  OK: 4
  MISSING: 0
  LEFTOVER: 0
  DEGRADED: 0

(targeted run — _recover_set.json left unchanged)
```
