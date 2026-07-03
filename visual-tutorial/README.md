# Attention.Lab — Frontier Lab Visual Tutorial (Module 01)

An interactive, bilingual (中文 / English) learning lab for the **attention mechanism**, built for a
senior software engineer moving toward a frontier-lab AI Systems / Research Engineer role.

This is **Module 01** of a planned series. It is also the smoke test for the tutorial template — if the
shape works for you, Modules 02 (Multi-Head Attention) and 03 (Positional Encoding) drop into the same site.

---

## 1. What this tutorial is

A single-page, dependency-free website that teaches attention as an *interactive lab*, not a blog post.
It explains the concept at multiple levels — intuition → mental model → math → implementation → systems
trade-offs → failure modes → metrics → frontier-lab relevance — and backs each idea with a hands-on
visualization and a concrete artifact to produce.

The math shown in the live pipeline is the same worked example as the repo's
`01-transformers/architecture/attention-mechanisms.md`, and the numbers were verified numerically
(weights ≈ `[0.33, 0.67]`, output ≈ `[0.33, 1.00]`).

## 2. Who it is for

A **senior engineer**: strong in software and distributed systems, still building intuition for deep
learning internals. Chinese carries the intuition and emotional clarity; English carries the technical
terms and interview-ready phrasing (English share rises in the advanced sections).

## 3. How to open it locally

No build step, no server, no dependencies.

```bash
# from the repo root
open visual-tutorial/index.html        # macOS
# or: xdg-open visual-tutorial/index.html   (Linux)
# or just double-click index.html
```

Everything runs client-side. Progress (checklist, artifact, research log) is saved in your browser via
`localStorage`. Works fully offline.

## 4. Module list

| # | Section | What it builds |
|---|---------|----------------|
| 1 | Objective | What you'll be able to do without notes |
| 2 | Intuition 直觉 | Attention as learnable *routing*, not understanding |
| 3 | Analogy 类比 | The library (Q / K / V), routing vs. payload |
| 4 | Mental Model | Sequence as a fully-connected graph, recomputed per layer |
| 5 | The Math 技术机制 | `softmax(QKᵀ/√d_k)V`, every symbol & shape labeled |
| 6 | Interactive Lab | 5 visualizations (below) |
| 7 | Implementation | numpy from-scratch + the PyTorch fused path |
| 8 | Staff Engineer Lens | Bottlenecks, metrics, 10× scaling, design-review Qs |
| 9 | Frontier Lab Relevance | Training, serving, eval, papers |
| 10 | Common Confusions | The five misreads that break interviews |
| 11 | Mini Quiz | 5 interview-shaped questions with feedback |
| 12 | Artifact + Research Log | Produce something; reflect; save |

*Coming next:* Module 02 · Multi-Head Attention, Module 03 · Positional Encoding.

## 5. Interactive components included

All five are rendered with **D3 (v7)**, vendored locally as `d3.v7.min.js` so the page stays fully offline
(no network, no build step). The D3 layer is *additive and defensive*: if D3 ever fails to load, each
visualization falls back to a plain-DOM version automatically — the page never goes blank.

1. **Attention heatmap (D3)** — a true 2-D matrix. Hover any cell or click/arrow-key a row label; the row
   lights up, a ring marks the top key, and a `Σ = 1.00` badge shows every row is a probability distribution.
   Clicking *“it”* points at *“cat”*. Full keyboard navigation.
2. **Pipeline step-through (D3)** — animated matrices for `X → Q/K/V → Q·Kᵀ → ÷√d_k → softmax → ×V` on the
   `“cat sat”` 2-D example. Hover *or keyboard-focus* a result cell to light up the exact query row and key
   column that produced it — "score = one query · one key" made visible. Shapes labeled at every stage.
3. **√d_k "break it" lab (D3)** — twin bars (raw logits ↔ softmax weights) plus a **d_k selector**
   (1 · 4 · 16 · 64). Watch the softmax sharpen from a gentle blend into a single spike as d_k grows, then
   toggle scaling on to cancel it. At d_k = 1, scaling is a no-op — a deliberate teaching moment.
4. **O(n²) cost & memory explorer (D3)** — a log-log memory-growth chart with a **crossover marker** at
   `n = d` (where the `n×n` scores overtake the `n×d` tensor), plus sliders for `n`/`d`, a FlashAttention
   toggle, and the two area-resizing boxes with an fp16 memory gauge.
5. **KV-cache simulator (D3)** — training (parallel, full `N×N`) vs. inference (autoregressive): `Q` shrinks
   to `1×d`, each "Generate" *animates in* one cached `K`/`V` row, and a **live readout** converts the cache
   to a real 32-layer, d=4096 fp16 estimate (≈ 4.29 GB at 8k tokens) so the memory-bandwidth bottleneck lands.

Plus: scroll-driven top progress bar, sidebar section highlighting (IntersectionObserver), persisted
progress checklist, artifact checklist, an autosaving research-log box, and `prefers-reduced-motion` support.

## 6. How to extend the tutorial

- **Add a module:** copy the `<section class="module-section" id="...">` pattern in `index.html`, add a
  matching `.nav-link` in the sidebar, and (if it has a visualization) write an `init…()` function in
  `main.js` and call it from the `DOMContentLoaded` boot block. The checklist auto-derives its items from
  the nav links, so no extra wiring is needed.
- **Add a visualization:** give it a container `id` in the HTML, then add an `initD3…()` function in
  `d3-viz.js` and call it from that file's `DOMContentLoaded` block (D3 lives entirely in `d3-viz.js`; the
  shared math helpers `matmul`/`softmax` and a body-level tooltip are at the top of that file). If the same
  viz also has a plain-DOM version in `main.js`, set a `window.__ATTN_D3` flag for it in `d3-viz.js` and add
  a matching early-return guard in the `main.js` init so the two never both render.
- **Scale to many modules:** if the page grows large, split into `modules/module-0X.html` and a shared
  `data/modules.js`, as outlined in the original spec.

## 7. Known limitations / TODOs

- **Illustrative attention weights** in viz ① are hand-set to teach the pattern (e.g. “it”→“cat”); they are
  not from a trained model. The pipeline viz ②, by contrast, uses *real* live-computed math.
- **Cost-box sizing** uses a `bytes^0.25` visual scale so both boxes stay on screen across a huge range of
  `n`; the precise truth is in the numeric memory readout, not the box area.
- **No automated browser test runner** is bundled (to keep the folder dependency-free). Verification was
  done by (a) running the pipeline / scaling / KV math in Node against the verified targets, (b) a headless
  **jsdom** render harness that loads the real page and asserts every D3 viz draws, the `main.js` guards
  prevent any double-render, and the plain-DOM fallback still renders when D3 is removed, and (c) a
  four-dimension adversarial code review (correctness, math/pedagogy, integration, accessibility). The
  jsdom harness is not committed; re-add it with `npm i -D jsdom` if you extend the visualizations.
- Single-file-per-section content lives inline in `index.html`; very large future modules may warrant the
  `modules/` split above.

## 8. Suggested next artifacts for the learner

After using the lab, produce (the in-page checklist tracks these):

1. `scaled_dot_product_attention` in numpy from scratch; reproduce the lab's `“cat sat”` numbers.
2. A heatmap plot of the `n×n` attention matrix for a toy sentence.
3. A memory table for `n = 1k / 8k / 32k` confirming the 100×-per-10× attention-memory scaling.
4. A KV-cache size estimate (GB) for a 32-layer model at `n = 8k`, with one sentence on why decode is
   memory-bandwidth bound.
5. A 5-line **research log**: what surprised you, and the one experiment you'd run next.

---

*Files:* `index.html` (structure + content + D3-specific styles), `styles.css` (calm lab aesthetic),
`main.js` (plain-DOM interactivity + fallback), `d3-viz.js` (all five D3 visualizations), `d3.v7.min.js`
(vendored D3, offline), `README.md` (this file). Built from your repo's attention content; Gemini CLI
contributed visualization ideas and UX critique only — all ML explanations were authored and verified here.
