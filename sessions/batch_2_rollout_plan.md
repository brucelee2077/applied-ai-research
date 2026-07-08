# Batch 2 — Coach Layer Rollout Plan

_Plan-first artifact for the `frontier-curriculum-refactor` Coach Layer rollout to Batch 2. Written **before any lesson edit**. Date: 2026-07-06._

Companion docs: [`COACH_STYLE_GUIDE.md`](./COACH_STYLE_GUIDE.md) (blocks + rules) · [`enhance_lesson_checklist.md`](./enhance_lesson_checklist.md) (per-lesson checklist) · [`lesson_audit.py`](./lesson_audit.py) (automated gate) · [`batch_1_rollout_plan.md`](./batch_1_rollout_plan.md) + [`batch_1_rollout_report.md`](./batch_1_rollout_report.md) (the precedent this batch follows).

---

## 1. Scope

Batch 2 = the **transformer & first-model** modules (rollout step 2 per the Batch 1 report §13: "Transformer & training modules"):

| Module | Lessons | Theme |
|--------|:---:|-------|
| `m03-attention` | 5 | embeddings → Q/K/V → scores → multi-head → positional |
| `m04-first-model-mlp` | 6 | MLP on MNIST → backprop → minibatch loop → loss/dropout → PyTorch |
| `m05a-text-transformer` | 8 | residual → LN → FFN → block → causal mask → enc/dec → generation → full stack |

**19 lessons total. All 19 are in play** (no Batch 2 pilot was pre-done). All 19 are on the **new sidebar shell** (`class="module-section"`, Light/Dim/Dark/Midnight switcher). `lesson_audit.py` reports all 19 structurally **OK** (7 sec / 7 got-it / 3 demos / 4 quiz / BUILD present) — no repair work; this is purely a Coach Layer overlay + a targeted P0 visual pass.

**Baseline advisories (all 19, from `lesson_audit.py`):** every lesson already has an analogy (`.relate`/`.card`), a **Staff / Research Engineer Lens**, and a concrete `experiment.py` artifact (none of those three advisories fired on any file). Every lesson is **missing**: a pain-point block, a 5-minute research log, a bilingual scaffold, and explicit Acceptance criteria. **17/19** are missing an interview-ready block (day-05-positional and day-06-encoder-vs-decoder already carry an `interview`/🎤 marker — to be re-verified as a real block when editing). The "math-heavy but no Math Ladder" advisory fired on all 19 — but that is the `<code>`-count heuristic; the ladder is added only where a lesson has a genuine single core formula (see §4).

---

## 2. Method (how this plan was built)

1. **Inspect** — listed all 19 lessons; confirmed all on the new shell; ran `lesson_audit.py` (targeted, so `_recover_set.json` untouched) → 19/19 OK with the soft COACH advisories above.
2. **Visual audit** — ran the `frontier-visual-auditor` rubric over all 19 lessons via 3 read-only module agents (one per module), each citing evidence: the 3 `data-demo` fake-terminal keys, the `var BUILD` scroll-reveal steps (static inline SVG vs `<iframe src=...>`), any embedded `../../viz/*.html`, and the section-4 core formula. Each returned difficulty + a 1–5 visual score + a P0/P1/terminal-acceptable verdict.
3. **Iframe reality check** — four lessons embed a genuinely interactive `../../viz/*.html` inside their scroll-reveal build: `attention-heatmap.html` (m03/day-01), `attention-pipeline.html`+`softmax-scaling.html` (m03/day-03), `attention-multihead.html` (m03/day-04), `gradient-descent.html` (m04/day-04). Those already score 4–5 and are **not** P0 — they get Coach prose only, no new lab (per "do not blindly convert every playground to D3").
4. **Precise Coach-element scan** — the `lesson_audit.py` advisory list per file tells us exactly which of the 15 Coach elements are missing, so this plan adds **only what is missing** and never duplicates an existing Staff Lens / analogy / artifact.

**Upgrade decision rule (auditor skill + this batch's explicit directive):** P0 = a hard **or** inherently-spatial concept, a current visual score **< 4**, a terminal-only playground, **and** no interactive mechanism viz anywhere — **OR** a mechanism the parent request explicitly named for this batch (attention heatmaps · Q/K/V flow · residual stream · layer-norm before/after · logits→sampling). P1 = useful but the existing static SVG build already carries the mechanism. Terminal-acceptable = a code / comparison / shape-flow / assembly lesson where a fake terminal + static diagram is genuinely the right medium.

---

## 3. Visual audit results (19 lessons)

| Lesson | Concept | Difficulty | Playground | Build viz | Score | Verdict |
|--------|---------|:---:|---|---|:---:|:---:|
| m03 · day-01 · embeddings | token → vector lookup, dot-product similarity | easy | terminal (`ids/lookup/sim`) | **iframe `attention-heatmap.html`** + SVG | 4 | already-visual |
| **m03 · day-02 · qkv** | one embedding → Q,K,V via three projections | medium | terminal (`makeqkv/match/roles`) | static SVG | 3 | **P0** (user-named: Q/K/V flow) |
| m03 · day-03 · attention-scores | `softmax(QKᵀ/√d_k)V` | hard | terminal (`scores/softmax/out`) | **iframe `attention-pipeline`+`softmax-scaling`** | 5 | already-visual |
| m03 · day-04 · multihead | split → h attentions → concat·W_O | med-hard | terminal (`onehead/twoheads/merge`) | **iframe `attention-multihead.html`** + SVG | 4 | already-visual |
| **m03 · day-05 · positional** | permutation-invariance; sinusoidal PE | hard | terminal (`shuffle/petable/addpe`) | static SVG (text rows, never the waves) | 3 | **P0** |
| m04 · day-01 · mlp-mnist | 784→128→10 MLP forward + eval | medium | terminal (`data/model/train`) | static SVG (flow) | 4 | P1 |
| **m04 · day-02 · backward-pass** | hand-wired backprop / chain rule | hard | terminal (`fwd/back/check`) | static SVG (directional but frozen) | 3 | **P0** |
| m04 · day-03 · minibatch-loop | mini-batch SGD + overfit-one-batch | medium | terminal (`batch/loop/overfit`) | static SVG (trade-off curves) | 4 | P1 |
| m04 · day-04 · training-loss-dropout | reading loss curves; inverted dropout | medium | terminal (`run/overfit/dropout`) | **iframe `gradient-descent.html`** + SVG | 4 | already-visual |
| m04 · day-05 · pytorch-version | rebuild the MLP in PyTorch (API map) | easy | terminal (`module/setup/loop`) | static SVG (NumPy→PyTorch map) | 3 | terminal-acceptable |
| m04 · day-06 · why-pytorch | tensor = array + autograd + GPU | easy | terminal (`mirror/autograd/gpu`) | static SVG (comparison) | 3 | terminal-acceptable |
| **m05a · day-01 · residual-connections** | `y=F(x)+x`; gradient `F'(x)+1` | medium | terminal (`fade/skip/grad`) | static SVG (two-road) | 3 | **P0** (user-named: residual stream) |
| **m05a · day-02 · layer-norm** | `y=(x−μ)/(σ+ε)`; pre-LN vs post-LN | medium | terminal (`raw/norm/check`) | static SVG (wild→tidy) | 3 | **P0** (user-named: LN before/after) |
| m05a · day-03 · feed-forward | Linear expand → ReLU → Linear compress | medium | terminal (`expand/relu/compress`) | static SVG (4→16→4) | 3 | terminal-acceptable |
| m05a · day-04 · full-block | assemble pre-LN block (attn+FFN+res) | medium | terminal (`input/attn/ffn`) | static SVG (sub-layers + tower) | 3 | terminal-acceptable |
| **m05a · day-05 · causal-masking** | −inf triangle before softmax | hard | terminal (`scores/mask/soft`) | static SVG (has 3×3/4×4 grids) | 3 | **P0** |
| m05a · day-06 · encoder-vs-decoder | enc(no mask)/dec(causal)/enc-dec; BERT/GPT/T5 | easy-med | terminal (`enc/dec/encdec`) | static SVG (arrows) + table | 3 | terminal-acceptable |
| **m05a · day-07 · text-generation** | autoregressive loop; `softmax(logits/T)`; top-k/p | hard | terminal (`greedy/temp/topk`) | static SVG (two frozen T frames) | 3 | **P0** (user-named: logits→sampling) |
| m05a · day-08 · full-transformer | capstone: embed+pos → N blocks → head | medium | terminal (`embed/stack/head`) | static SVG (vertical tower) | 3 | terminal-acceptable |

### P0 set — 7 lessons

Four are P0 on the pure rubric (hard/inherently-visual + terminal-only + no interactive viz + score 3):

1. **m03 · day-05 · positional** — sinusoidal PE is inherently a wave/heatmap pattern, yet every build step is a static text row (`pos 0 → [0,1,0,1]`); the waves are never drawn and the "shuffle → same output" permutation-invariance break is only described.
2. **m04 · day-02 · backward-pass** — backprop is an inherently *directional* mechanism (blame flowing right→left, ReLU gates zeroing units); a frozen SVG + terminal cannot show the chain-rule product accumulating.
3. **m05a · day-05 · causal-masking** — a 2-D "who-sees-whom" grid where the diagonal boundary is the whole point; the classic off-by-one (`j>i` vs `j>i+1`) silent bug is exactly what a probeable grid makes visible.
4. **m05a · day-07 · text-generation** — temperature is a *continuous* dial ("watch the bars flatten/sharpen"); two frozen frames (T=0.5, T=2.0) show endpoints but not the reshaping, and top-k/top-p cut-offs are naturally interactive.

Three more are elevated to P0 because the **parent request explicitly named them** as mechanism visuals this batch should prefer over terminal walkthroughs (the auditor rated them P1 — adequate static SVG — so this is a directive-driven elevation, documented for transparency):

5. **m03 · day-02 · qkv** — "Q/K/V flow" (user-named). One embedding fanning into three projections is the clearest small mechanism to make interactive.
6. **m05a · day-01 · residual-connections** — "residual stream diagram" (user-named). The existing two-road SVG is static; a depth + skip-toggle control makes the vanishing-vs-alive gradient a live contrast.
7. **m05a · day-02 · layer-norm** — "layer-norm before/after" (user-named). The wild→tidy boxes are static; a live recenter/rescale strip makes mean-0/std-1 a thing you *do*, not read.

**Not P0 — already interactive (score 4–5):** m03/day-01, m03/day-03, m03/day-04, m04/day-04. Coach prose only.

**Not P0 — terminal-acceptable (code / comparison / shape-flow / assembly):** m04/day-01 (P1), m04/day-03 (P1), m04/day-05, m04/day-06, m05a/day-03, m05a/day-04, m05a/day-06, m05a/day-08. Coach prose only; keep the terminal playground and static build.

---

## 4. Per-lesson Coach Layer work (add only what is missing)

Every lesson gets the missing Coach blocks inserted **inside `.sec-body`, before that section's `.gotit`**, using the exact templates in `COACH_STYLE_GUIDE.md` §5. Legend: **●** add · **○** already present (leave it) · **–** intentionally skip.

| Lesson | 😕 pain | 🪜 Ladder | 🎤 interview | Staff Lens | ✔ Accept | 📓 log | 中文 | P0 lab |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| m03 · day-01 · embeddings | ● | ● (dot/cosine sim) | ● | ○ | ● | ● | ● | – |
| m03 · day-02 · qkv | ● | ● (`Q=x·W_Q` ×3) | ● | ○ | ● | ● | ● | ● |
| m03 · day-03 · attention-scores | ● | ● (`softmax(QKᵀ/√d_k)V`) | ● | ○ | ● | ● | ● | – |
| m03 · day-04 · multihead | ● | ● (`d_k=d_model/h`; concat·W_O) | ● | ○ | ● | ● | ● | – |
| m03 · day-05 · positional | ● | ● (`PE=sin/cos(pos/10000^…)`) | ○ | ○ | ● | ● | ● | ● |
| m04 · day-01 · mlp-mnist | ● | ● (`relu(x·W1+b1)·W2+b2`) | ● | ○ | ● | ● | ● | – |
| m04 · day-02 · backward-pass | ● | ● (`dW2=hᵀ·dlogits`; chain) | ● | ○ | ● | ● | ● | ● |
| m04 · day-03 · minibatch-loop | ● | ● (`W←W−lr·(1/B)Σdw`) | ● | ○ | ● | ● | ● | – |
| m04 · day-04 · training-loss-dropout | ● | ● (`h·mask/(1−p)`) | ● | ○ | ● | ● | ● | – |
| m04 · day-05 · pytorch-version | ● | – (code idiom) | ● | ○ | ● | ● | ● | – |
| m04 · day-06 · why-pytorch | ● | – (comparison finale) | ● | ○ | ● | ● | ● | – |
| m05a · day-01 · residual-connections | ● | ● (`dy/dx=F'(x)+1`) | ● | ○ | ● | ● | ● | ● |
| m05a · day-02 · layer-norm | ● | ● (`(x−μ)/(σ+ε)`) | ● | ○ | ● | ● | ● | ● |
| m05a · day-03 · feed-forward | ● | ● (`ReLU(x·W1)·W2`) | ● | ○ | ● | ● | ● | – |
| m05a · day-04 · full-block | ● | ● (`x+attn(LN(x))`; `x+ffn(LN(x))`) | ● | ○ | ● | ● | ● | – |
| m05a · day-05 · causal-masking | ● | ● (`softmax(scores+M)`) | ● | ○ | ● | ● | ● | ● |
| m05a · day-06 · encoder-vs-decoder | ● | – (no single core formula) | ○ | ○ | ● | ● | ● | – |
| m05a · day-07 · text-generation | ● | ● (`softmax(logits/T)`) | ● | ○ | ● | ● | ● | ● |
| m05a · day-08 · full-transformer | ● | ● (`logits = x·W_head`, shape trace) | ● | ○ | ● | ● | ● | – |

**Totals to add:** pain-point ×19 · Math Ladder ×16 (skip the 3 non-formula lessons) · interview ×17 (2 already present, re-verify) · Acceptance criteria ×19 · research log ×19 · bilingual scaffold ×19 · **7 P0 inline `.vlab` labs**. **Staff Lens: 0 new** — all 19 already have one (preserve intact).

**Notes / guardrails:**

- **Math Ladder only for a genuine single core formula.** The 3 skips (m04/day-05 PyTorch API map, m04/day-06 why-PyTorch comparison, m05a/day-06 encoder-vs-decoder contrast) have no single central equation; forcing a ladder would be noise. `lesson_audit.py`'s math-ladder advisory will persist on those by design — noted in the report, not "fixed".
- **Produce artifact — do NOT fabricate.** Every lesson already references `experiment.py` (no advisory fired). Keep the existing artifact and file path; only add the missing `<h4>Acceptance criteria</h4>` list.
- **Interview + Staff Lens.** Staff Lens exists on all 19 (preserve). Interview: add a real 🎤 block to the 17 that lack one; for day-05-positional and day-06-encoder-vs-decoder, re-read first and only add if the existing marker is not already a real spoken-answer block (avoid duplication).
- **Chinese dosage:** 2–4 short touches per lesson, concentrated in §1 (pain) and §2 (intuition) only — never in formulas, code, or the interview phrasing (guide §3, memory `prefers-english-learning-content`).

---

## 5. P0 visual upgrades (7 labs) — design specs

All seven are **dependency-free inline SVG + vanilla JS** using the repo `.vlab` template proven in Batch 1 (`m10a/day-01`, `m13/day-04`) — the CSS block at `m10a-scaling-laws/day-01-kaplan-paradigm/lesson.html:197–224` is copied verbatim (it uses only new-shell CSS vars: `--line2`, `--panel2`, `--accent`, `--ok-soft`, `--warn-soft`, …). **No CDN, no new fonts, offline-safe, theme-aware, keyboard-reachable sliders.** Each lab is inserted into **Section 3** (`data-sec="play"`) **above** the existing fake-terminal playground; the terminal keeps its 3 `data-demo` buttons and its **completion-gate** role (untouched). Every lab carries the 7 required cards: learning question (`.vlab-q`) · labeled axes · clear change on control move · "what you should notice" · misconception note · "where this simplifies reality" (`.vlab-note`) · frontier-lab relevance.

### 5.1 m03 · day-02 · qkv — "Projection fan"

- **Learning question:** why turn one embedding into three different vectors, and what does each one do?
- **Control:** a 2-token toggle picks which token is the *query*; a slider scrubs a shared "projection strength" so you see Q/K/V move together.
- **Two panels:** (left) one embedding row → three labeled arrows (`W_Q`/`W_K`/`W_V`) → Q, K, V boxes; (right) the chosen query·each-key dot product as a small bar, with the max highlighted ("this is who it attends to").
- **Notice / misconception / simplifies:** Q and K play *asymmetric* roles (query asks, key answers); misconception = "Q, K, V are the same vector" — they are three *learned* projections; simplification = tiny `d`, integer weights, no softmax yet (that is day-03).

### 5.2 m03 · day-05 · positional — "PE waves & fingerprint"

- **Learning question:** self-attention sees a *set*, not a *sequence* — how does adding a fixed pattern restore order?
- **Two panels:** (left) an `(N positions × d dims)` PE grid colored blue→red by value; (right) the sine/cosine wave for the currently selected dimension column.
- **Controls:** a `dim` slider highlights one column and redraws its wave (slow low-`i` vs fast high-`i` frequency); a `pos` slider highlights one row to show its unique fingerprint vector.
- **Notice / misconception / simplifies:** each position gets a distinct, smooth code; misconception = "positions are just 0,1,2,… added on" (they are multi-frequency sinusoids so relative offsets are learnable); simplification = 8×8 grid, `10000` base fixed, no learned/RoPE variants.

### 5.3 m04 · day-02 · backward-pass — "Blame flows backward"

- **Learning question:** how does one loss number become a gradient for every weight?
- **One panel, two passes:** the chain `x → z1 → h → logits → loss` drawn as nodes. A **Forward** toggle animates left→right (activations light up); a **Backward** toggle sends gradient arrows right→left, each node labeled with its local gradient, and the ReLU gate visibly zeroing the units where `z1<0`.
- **Control:** a slider scrubs the chain-rule product one hop at a time so you watch `probs−y` propagate back into `dW2`, then `dh`, then `dW1`.
- **Notice / misconception / simplifies:** the ReLU gate kills gradient on dead units (`z1≤0`); misconception = "backprop is a different algorithm from forward" (it is the same graph, reversed, reusing cached activations); simplification = one sample, tiny widths, no batching.

### 5.4 m05a · day-01 · residual-connections — "Two roads for the gradient"

- **Learning question:** why does adding `x` back (`y=F(x)+x`) let 96-layer stacks train at all?
- **Control:** a **depth `N`** slider (2→96) and a **skip on/off** toggle.
- **Two bars / one panel:** the gradient reaching layer 1 after `N` layers, plotted for *plain* (`≈r^N`, vanishes) vs *residual* (`≈(r+1)^N`-ish, survives) — with the lesson's own `1.2^10≈6.2` vs `0.2^10≈1e-7` numbers as anchors. Skip-off collapses the residual bar onto the plain one.
- **Notice / misconception / simplifies:** the `+1` gives the gradient a road that never multiplies to zero; misconception = "skip connections add capacity" (their job is gradient flow / identity preservation); simplification = single scalar gain per layer, no LN interaction.

### 5.5 m05a · day-02 · layer-norm — "Recenter & rescale"

- **Learning question:** what does LayerNorm actually do to one token's numbers, and why per-token?
- **Two dot-strips on a shared number line:** (top) the raw feature values (e.g. wild `[5, 500, 5000]`); (bottom) the normalized values after `(x−μ)/(σ+ε)`.
- **Control:** a slider (or a couple of presets) reshapes the raw spread; μ and σ print live and both strips re-plot, showing the normalized strip always lands at mean 0 / std 1.
- **Notice / misconception / simplifies:** normalization is **per token across features**, not per batch (that is BatchNorm); misconception = "LN removes information" (γ,β can restore any scale it needs); simplification = γ=1,β=0, `ε` shown but tiny.

### 5.6 m05a · day-05 · causal-masking — "Who can see whom"

- **Learning question:** how does one triangle of `−inf` stop a token from reading its own future?
- **One panel:** an `N×N` attention grid. Hovering / clicking row `i` highlights the allowed cells `0..i` green and the future cells red.
- **Control:** a **diagonal-offset** toggle switches the boundary between correct (`j≤i`) and the classic off-by-one bug (`j≤i+1`) so a *leaked-future* cell lights up; a second toggle shows scores → `+M` → post-softmax (future weights become exactly 0).
- **Notice / misconception / simplifies:** the mask is added **before** softmax (so `e^{−inf}=0`); misconception = "just zero the weights after softmax" (that breaks the normalization); simplification = small `N`, integer scores, single head.

### 5.7 m05a · day-07 · text-generation — "Temperature dial"

- **Learning question:** the model outputs one fixed logit vector — so where does variety come from?
- **One panel:** a bar chart of 4–5 candidate-word probabilities.
- **Controls:** a **temperature** slider (0.1→3.0) recomputes `softmax(logits/T)` and re-heights the bars live (low T → one tall spike / greedy; high T → flat / random); a **top-p** line greys out the dropped tail as `p` changes.
- **Notice / misconception / simplifies:** temperature rescales *logits before* softmax, it does not change the model; misconception = "higher temperature = smarter" (it only raises randomness); simplification = fixed toy logits, top-k/top-p shown one at a time.

> **Reuse note:** the causal-mask grid echoes the existing `attention-heatmap.html` grid idiom; the temperature dial echoes the `softmax-scaling.html` bar-collapse idiom — both proven interactive viz already in the repo, so the inline labs stay consistent with what the strong lessons already show. Where a lesson's learning question is already served by an embedded interactive iframe (m03/day-01, day-03, day-04; m04/day-04) **no new lab is built.**

---

## 6. Preservation contract (never violate — guide §7)

For every edited lesson, these must be byte-behaviour-identical after the edit:

- `data-quest-id` on `<body>` (the localStorage key `frontier-lesson:<id>`).
- Every prev / next / hub `href` (top and sidebar nav).
- 7 sections, their `id`s and `data-sec` keys; 7 `.gotit` buttons; the progress/reset/checklist script.
- `var QS` (4 quiz entries `{q,opts,ans,fb}`), `var DEMOS` (3 keys ↔ 3 `data-demo` buttons), `var BUILD` (scroll-reveal, incl. the four interactive iframes).
- Glossary tooltip script + every `.term[data-tip]` (new terms get a `data-tip`).
- Self-contained / offline: no new external assets, CDN, or fonts. Chinese renders in the existing system font stack.
- Section wrapper tags, `id`s, and `<div class="sec-body">…</div></section>` boundaries kept intact so `_shell_migrate.py` can still extract cleanly.

**Additive-only.** No failure mode, trade-off, equation, or complexity claim is deleted; no precise language is swapped for cheerleading; no motivational fluff. The one permitted deletion per file is the old one-line `log.md` nudge, folded into the richer research-log block (as in Batch 1).

**P0 visual specifics:** the inline `.vlab` is added **above** the existing terminal playground; the terminal's 3 `data-demo` buttons remain the section's completion gate. Labs are **inline** (not iframed), so the `viz-iframe-autoresize-bug` postMessage-height concern does not apply.

---

## 7. Execution order (sequential — one file per session, no parallel lesson writing)

Per CLAUDE.md §4 and memory `capability-spiral-build`: **no subagents for parallel writing** (tone/analogy drift). Edit lessons one at a time, in this order (module order, foundational day first). Run the per-file verify gate after each before moving on.

1. `m03-attention` — day-01, day-02 **(+P0)**, day-03, day-04, day-05 **(+P0)** (5)
2. `m04-first-model-mlp` — day-01, day-02 **(+P0)**, day-03, day-04, day-05, day-06 (6)
3. `m05a-text-transformer` — day-01 **(+P0)**, day-02 **(+P0)**, day-03, day-04, day-05 **(+P0)**, day-06, day-07 **(+P0)**, day-08 (8)

P0 lessons get the Coach Layer prose **and** the inline `.vlab` in the same session.

---

## 8. Verification gate (run after each file; summarized in the report)

1. `node --check` equivalent via `/tmp/batch1_verify.js` (`vm.Script` on every inline `<script>`) — JS syntax clean.
2. jsdom headless load (`/tmp/node_modules/jsdom`, reachable via `/tmp/batch1_verify.js <path> [--p0]`) — no thrown error; asserts 7 sections / 7 got-it / 4 quiz / 3 demos / BUILD present / tooltip controller. For P0 files, `--p0` dispatches `input`/`click` and asserts the `.vlab` SVG re-renders and readouts respond.
3. `python3 sessions/lesson_audit.py <changed paths…>` (targeted — leaves `_recover_set.json` alone) → **OK**, with the six core COACH advisories (pain / ladder / interview / accept / log / bilingual) cleared for that file (except the 3 intentional Math-Ladder skips).

**Definition of done for the batch:** all 19 lessons OK with no hard failures; each has its full missing-Coach-element set added; the 7 P0 labs render and interact; nav / quest-id / quiz / playground / build / tooltips preserved everywhere; Chinese light (2–4 touches); depth ≥ before. Then write `sessions/batch_2_rollout_report.md`.

---

## 9. Risks

1. **Concurrent edits.** `sessions/` is edited by other sessions; re-read each file immediately before editing (the tooling enforces "file modified since read").
2. **P0 count is 7, higher than Batch 1's 4.** Justified: this batch is attention/transformer-heavy and the parent request explicitly named 5 mechanism visuals and said "prefer mechanism visuals over terminal walkthroughs." Three P0s (qkv, residual, layer-norm) are directive-driven elevations of auditor-P1 lessons — documented here for transparency. Guardrail: each lab stays at **one** learning question + the 7 required cards; reuse the pilot `.vlab` markup; **do not gold-plate**.
3. **Math-Ladder advisories on 3 non-formula lessons** (pytorch-version, why-pytorch, encoder-vs-decoder) will persist by design. The report lists these as intentional, not failures.
4. **Interview marker on day-05-positional / day-06-encoder-vs-decoder.** The audit did not flag these as missing an interview block; re-read before editing and only add a 🎤 block if the existing marker is not a real spoken-answer block (avoid duplication).
5. **No real-browser pixel check** (`no-real-browser-in-sandbox`). All verification is headless (jsdom + JS-syntax + hand-checked lab math); a final Light/Dim eyeball of the 7 labs would need a user screenshot if pixel fidelity matters. The SVGs reuse the `.vlab` classes proven in Batch 1.
