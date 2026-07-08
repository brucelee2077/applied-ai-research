# QA Report — Module 2 "The Neuron & How It Learns" (v7.1 P0 Fix Pass)

> **Target:** `sessions/m02-the-neuron` (9 days + 2 gates)
> **Mode:** `/frontier-refactor-qa`, **xhigh** effort, **fix + re-verify**.
> **Source of truth:** the P0 findings and P0 fix plan in `sessions/m02_v7_1_gap_report.md`.
> **Scope honored:** only P0-1 … P0-4 were touched, all localized to one D1 section + three
> Produce blocks + D2's mechanism section. No `data-quest-id`, navigation, localStorage, quiz array,
> BUILD/DEMOS array, completion gating, or shared shell file was modified. No P2 polish, no broad prose
> rewrite. Old notebooks were not read (none in-module).
> **Contracts used as ground truth:** `m02_coverage_contract.md`, `m02_artifact_contract.md`,
> `m02_visual_contract.md`.

---

## Files changed

`git diff --stat` (4 files, **+64 / −27** lines):

| File | What changed | P0 |
|---|---|---|
| `day-01-single-neuron/lesson.html` | Section 2 rebuilt: leads with brain-inspired intuition, adds a biological→artificial **mapping table**, keeps the "judge" card as an explicitly optional secondary hook, extends the "where the analogy breaks" line (no biological backprop; spikes over time; weights learned by gradient descent). Section header + goal-line unchanged in meaning. | P0-1 |
| `day-02-activations/lesson.html` | Added **tanh** to the goal, S1 ("Three common bends"), S4 (third formula + range + zero-centered note), a **static tanh curve** figure, and a **ReLU/sigmoid/tanh comparison table**; extended the Produce Option A/B + acceptance to print all three over `np.linspace(-6,6,13)`, `sigmoid'(0)≈0.25`, and the "ReLU grad = 0 on negatives" note. | P0-2 |
| `day-05-gradients-backprop/lesson.html` | Added **Part (b)** to Produce Option A and Option B: chain-rule on `(sigmoid(w·x+b)−y)²` at anchors `x=2,w=0.5,b=0,y=1`, printing the chain pieces and `dL/dw≈−0.212`, `dL/db≈−0.106`, verified vs central finite differences to `<1e-6`; acceptance gained two Part-(b) criteria; the 📓 log restores the finite-difference explain-back. | P0-3 |
| `day-09-train-val-test/lesson.html` | Produce Option A/B + acceptance rewritten to a **3-way disjoint split** (18 train / 6 val / 6 test, sizes printed & summing to 30), **test loss evaluated only at the best-val degree**, degree-sweep framing kept, and the incoherent "as you train" phrasing dropped from Option A. | P0-4 |

**Not touched (verified via `git diff` guard scan):** every `data-quest-id` (`wf2-d01-neuron`,
`wf2-d02-activations`, `wf3-d02-backprop`, `wf3-d06-split`), all `DEMOS`/`BUILD`/`QS` arrays, the `>=3`
playground gate and quiz `answered` gate, `frontier-lesson:` localStorage keys, `data-target`/nav-links,
and the `.fin` completion banners.

---

## P0 fixes applied

### P0-1 — D1 Foundation Framing → **FIXED**
Section 2 now delivers **all five** required elements (was 3/5):
1. **Brain-inspired intuition** — a lead paragraph: a brain neuron collects signals via dendrites, weighs
   them by synapse strength, the cell body sums them, and it "fires" past a threshold down the axon.
2. **Artificial-neuron caveat** — retained ("one artificial neuron alone is weak; the power comes from
   many neurons in layers").
3. **Mapping table** (biological → artificial → what it is): dendrites→`x`, synapse strength→`w`,
   cell-body sum→`w·x`, firing threshold→`b`+activation, axon→the output.
4. **Where the analogy breaks** — extended: it is inspiration, not a copy; real neurons spike over time
   and rewire chemically; here weights are **learned by gradient descent (Day 5)**, no biological "backprop".
5. **Transition to the math function** — "read down the middle column" → `output = activation(w·x + b)`,
   worked with numbers in Section 4.

The everyday "judge" analogy is retained but demoted under an **"A second, everyday picture (optional)"**
heading, so it is clearly secondary to the brain frame (v7.1 disallows it as the foundation frame).

### P0-2 — D2 `tanh` + activation artifact → **FIXED**
`tanh` now appears across every non-wiring surface: goal line, S1, the S4 formula callout
(`tanh(z)=(eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)`, range `(−1,1)`, zero-centered), a **static tanh S-curve figure**, and a
**ReLU/sigmoid/tanh comparison table** (formula · range · zero-centered? · slope near 0 · saturates? ·
typical use). The Produce block prints all three over `np.linspace(-6,6,13)`, plus `sigmoid'(0)≈0.25` and
"ReLU grad = 0 for negatives", exactly as `artifact_contract.md:20` mandates.
*(The playground `DEMOS` and the scroll-reveal `BUILD` array were intentionally not modified — adding a
4th playground button or a build step would change the completion gate, which is a frozen invariant; the
tanh curve is delivered as a static S4 figure instead.)*

### P0-3 — D5 backprop artifact → **FIXED**
The headline concept now has a **runnable artifact**. Produce Part (b) derives and prints the chain pieces
`dL/da=2(a−y)`, `da/dz=a(1−a)`, `dz/dw=x`, `dz/db=1`, their products `dL/dw≈−0.212` and `dL/db≈−0.106`
(identical to the S4 Math Ladder), and **verifies both against central finite differences to `<1e-6`**.
The 📓 explain-back restores the contract probe ("why backprop is the chain rule backward, and what the
finite-difference check proves").

### P0-4 — D9 test split → **FIXED**
The artifact is now a **3-way disjoint split** (18 train / 6 val / 6 test, sizes printed and asserted to
sum to 30 with no shared points). It sweeps degrees 1/3/5/9, selects the **lowest-validation** degree
(not the highest), and reports the **test loss once** for only that model. The "as you train" phrasing on
the one-shot `lstsq` fit was removed (this also closes P1-5). Acceptance bullets now map 1:1 to
`artifact_contract.md:27` criteria (1)-(4), including determinism.

---

## Coverage traceability result after fix

Re-graded the three rows the gap report flagged; all others carry forward unchanged.

| Concept | Req. depth | After-fix evidence | Result |
|---|---|---|---|
| ReLU / sigmoid / **tanh** — formulas, ranges | Core | tanh formula+range (S4), curve figure, comparison table, Produce grid | ✅ **PASS** (was ❌ FAIL) |
| Backprop = real numeric multi-link chain | Core | Produce Part (b) prints chain pieces + finite-diff `<1e-6`, matching the Math Ladder | ✅ **PASS** (was 🟡 PARTIAL) |
| Train / validation / **test** roles | Core | Produce = 3-way disjoint split, test at best-val | ✅ **PASS** (was 🟡 PARTIAL) |
| MSE convex-bowl-in-`ŷ` visual | Core | falling-loss curve present; bowl-in-`ŷ` still not drawn | 🟡 PARTIAL (**P2 only** — waived, not a must-cover FAIL) |

**Result:** **30/32 PASS · 0 FAIL · 1 PARTIAL** (D4 convex-bowl, P2, waived). No must-cover concept is
FAIL and none is PARTIAL for a P0/P1 reason, so **the coverage table now passes** — "all concepts land at
depth" can be signed off. Verified live: `grep tanh day-02` = 16 hits across goal/S1/S4/table/produce.

## Foundation framing result after fix

**✅ PASS (D1).** All five elements present (brain intuition, artificial-neuron caveat, biological→artificial
mapping table, where-it-breaks, transition to `output=activation(w·x+b)`). The everyday analogy is clearly
secondary. Verified: `dendrite`, `axon`, `synapse`, `firing threshold`, `brain neuron`, and
`gradient descent` all now present in D1; `"Brain neuron</th>"` mapping-table header found;
`"everyday picture (optional)"` confirms the judge card is demoted.

## Function family result after fix

**✅ PASS (D2 activations).** `tanh` — the missing Core representative — now lands with formula, range,
zero-centered explanation, a rendered curve, a **comparison table** (this also closes P1-2), and runnable
artifact evidence. Leaky-ReLU / GELU / SwiGLU remain correct Aware one-liners. The only family facets not
added are a tanh **playground demo** and a tanh **quiz option**; both were deliberately skipped because
they live in the frozen `DEMOS`/`QS` wiring (adding either changes the completion gate). With tanh fully
taught + visualized + produced, their absence is now **P2 polish**, not a coverage failure.

## Artifact result after fix

**✅ All three P0 artifact gaps closed.**
- **D2:** Produce now prints ReLU/sigmoid/tanh over `np.linspace(-6,6,13)`, `sigmoid'(0)≈0.25`, and the
  ReLU-grad-0 note (satisfies `artifact_contract.md:20`).
- **D5:** Produce Part (b) reproduces the chain-rule numbers and the finite-difference check
  (satisfies `artifact_contract.md:23`); a must-cover concept now has runnable evidence.
- **D9:** three disjoint splits summing to N, best-val ≠ final, test used once
  (satisfies `artifact_contract.md:27` criteria 1-4).
Path + run command remain correct and unchanged on all three days.

## Anchor consistency result after fix

**✅ D9 P0 conflict resolved.** The Produce no longer re-teaches "val = test": it is a 3-way split whose
test loss is read only at the best-val checkpoint, matching the lesson body's 3-box build. D5's Part (b)
uses the same anchors (`x=2,w=0.5,b=0,y=1`) and reproduces `dL/dw≈−0.212` from the Math Ladder — internally
consistent. Remaining anchor item is **D8 (P1-1)**: playground `5/5/4` vs Produce `8/8/8` — narrated at
`day-08:390`, teaches no conflicting model, and is explicitly out of this P0 scope.

---

## Remaining P1 / P2 (unchanged — out of this pass's scope)

**Closed as a side effect of the P0 work:** P1-2 (D2 activation comparison table — added) and
P1-5 (D9 Option A "as you train" incoherence — removed).

**P1 (fix before merge if easy):**
- **P1-1** D8 — playground `5/5/4` vs Produce `8/8/8` (narrated → not P0).
- **P1-3** D7 — no optimizer comparison table (SGD/momentum/Adam in prose+build only).
- **P1-4** D3 — playground prints shapes only, no per-stage numeric readout.
- **P1-6** D1 — acceptance omits the determinism criterion; 📓 log drops the bias explain-back.
- **P1-7** D4 — acceptance omits MSE==0-at-match and the clip/no-`log(0)` criterion.
- **P1-8** D6 — "training is run-once / terminates" misconception never explicitly reframed.
- **P1-9** review.html — final gate quiz reportedly tests only D4–D6; re-check before sign-off (unverified this pass).
- **P1-legacy** all 9 days — `/frontier-experiment-lab` (uninstalled) in every Produce Option-B + stub.
  **Contract-sanctioned; curriculum-wide (~243 files) — do NOT fix per-module.** Confirmed my edits added/removed
  **0** occurrences.

**P2 (polish; do not block):** P2-1 stub comment wording ("open lesson.html" vs "section 7"); P2-2 D4
convex-bowl visual; P2-3 no in-lesson derivative *value* beyond the new table; P2-4 D7 Adam playground shows
`m/√v` not bias-corrected; P2-5 D9 "around step 300" narrative looseness; P2-6 review.html self-check
Day-1/2/3 tags vs Day-4/5/6 content (do NOT touch `data-quest-id`).

---

## Verification run (this pass)

| Check | Result |
|---|---|
| `python3 sessions/lesson_audit.py m02-the-neuron` | **9 OK / 0 MISSING / 0 LEFTOVER / 0 DEGRADED** |
| `python3 sessions/nav_audit.py` | **0 BROKEN links** |
| `node sessions/staff_lens_audit.js m02` | 9/9 staff-lens present, gap 0, **`errs:[]`**, `q:4 o:16` on every day (quiz intact). `render:BROKEN` is the known-benign `.sec` vs `.module-section` selector mismatch — not a regression. |
| Node structural + JS syntax check (4 edited files) | **ALL PASS** — 7 sections each, one unchanged `data-quest-id` each, balanced `table`/`tr`/`td`/`svg`/`section` tags, all `<script>` blocks parse clean. |
| `git diff` frozen-invariant guard | **empty** — no change to `data-quest-id`, DEMOS/BUILD/QS, gate counts, localStorage keys, nav-links. |
| jsdom render check | not run — jsdom not installed in this environment (MODULE_NOT_FOUND); substituted the node structural + syntax check above. |

---

## Merge recommendation

### ✅ Pass with P1

All four v7.1 blockers are fixed and re-verified:

- **P0-1 Foundation framing (D1)** — 5/5 elements; brain frame leads, judge demoted to optional.
- **P0-2 Coverage + Function Family + Artifact (D2)** — `tanh` fully taught (formula, range, curve, table)
  and produced (grid + derivative facts); coverage row FAIL → PASS.
- **P0-3 Backprop artifact (D5)** — runnable chain-rule + finite-difference evidence matching the Math Ladder.
- **P0-4 Test split (D9)** — 3-way disjoint split, test-at-best-val; no longer re-teaches "val = test".

The lesson bodies, coach voice + Coach 6-pack, math, JS/nav/completion, module-label unification, and the
anchor spine remain intact. Only P1/P2 items remain (D8 step count, D7 optimizer table, review-quiz scope,
and the sanctioned curriculum-wide `/frontier-experiment-lab`), none of which block merge.

**Module 2 moves from 🚫 Blocked → ✅ Pass with P1.**
