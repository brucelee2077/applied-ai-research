# Module 2 — Visual Contract

> Per-lesson **P0** visual definition. Uses `frontier-visual-builder` conventions. Only **P0** visuals
> are required by this refactor. Reconciled against the two visual surfaces that already exist in every
> lesson — do not invent a third surface:
>
> - **Playground** — a simulated Python terminal (`DEMOS`) where each button runs one step of the day's
>   computation and prints real numbers. Gates section completion after all buttons are clicked.
> - **Build-it-up** — a scroll-reveal sequence of annotated inline SVGs (`BUILD` array) that assembles
>   the idea one piece at a time. Gates completion after the last step is revealed.
> - Days 5/7/8 additionally embed the live `viz/gradient-descent.html` iframe (lr slider + Step/Run).
>
> **Design rule (all P0):** the visual must show a **quantity changing** — a loss falling, a curve
> bending, a step overshooting — never a static labeled diagram. Every SVG must render legibly on the
> dark themes (the `.build-viz` panel forces a light canvas by design — that is acceptable, not a bug).

---

| Lesson | Learning question | Visual type (surface) | Controls / interaction | Misconception prevented | Priority |
|---|---|---|---|---|---|
| **D1 Neuron** | What does one neuron actually compute, and what do `w` and `b` each do? | Build: assemble `w·x → +b → z` term-by-term with the running numbers; Playground: 3 steps (weighted sum → +bias → z) | Scroll reveals each term; playground buttons print `np.dot(w,x)`, `+b`, `z` for the anchor `x=[2,3]` | "A neuron is a black box / brain cell"; that bias is cosmetic (build shows `b` shifting `z`) | **P0** (exists — fix: note the width 2→3 jump at the layer step) |
| **D2 Activations** | Why can't we just stack linear neurons — what does the bend add? | Build: ReLU/sigmoid curves → **collapse** (`linear∘linear` = one line) → **cure** (insert ReLU) → many-layer bent shape; Playground: `relu`/`sigmoid` on a vector + the collapse demo | Scroll; playground prints activation outputs + `W1@W2` folding to one matrix | "Activation is optional decoration"; "depth alone buys power" | **P0** (exists — fix: one anchor matrix pair through all of playground/ladder/produce; scalar collapse shown first) |
| **D3 Forward pass** | How does data flow through a layer — how do shapes transform, as one matmul + activation? | Build: 2-layer net lighting up the forward pass with shape labels at each hop; Playground: push one input through, printing shape **and one numeric value** at each stage | Scroll; playground "step" advances one layer `(1,3)→(1,4)→(1,2)` | "Forward pass is a scalar for-loop"; "shapes just work out" (shows mismatch is the #1 bug) | **P0** (exists — fix: add one concrete numeric forward pass, not shapes-only) |
| **D4 Loss** | How do we turn "wrong" into one number to shrink, and why does MSE-vs-CE shape matter? | Build: prediction→error→square→mean→MSE, then CE `-log(p)`, then a falling-loss curve; Playground: MSE + CE + "loss falls" sweep; **MSE↔CE comparison table** | Scroll; playground sweeps `p_true` and prints loss falling | "Loss = accuracy"; "any loss works / MSE for everything" | **P0** (exists — add the MSE↔CE table + MSE math ladder; fix CE ordering to match criteria) |
| **D5 Gradients/backprop** | What is a gradient telling us, and how does the chain rule pass blame backward? | Build: forward, then reveal the backward pass arrow-by-arrow with local slopes multiplying; **live `gradient-descent.html` embed** (lr slider + Step/Run) | Scroll reveals each backward hop; iframe: drag lr, Step/Run to watch descent | "Gradient = the answer, jump there"; "backprop is separate magic"; (add) "GD finds the global min" | **P0** (exists — add a **numeric multi-link chain**; add a "bumpy hill / local minima" note on the descent) |
| **D6 Training loop** | How do forward/loss/gradient/update chain into a loop that makes loss fall over steps? | Playground/Build: each click runs one full iteration and drops a point onto a falling loss-vs-step curve; loop-cycle SVG + batch boxes; **batch-size trade-off table** | "run one step" / "run N steps"; weights + loss update each click | "Training is one call that solves it"; "batch = pure speed knob" | **P0** (exists — fix `0.747→0.746`; add `zero_grad` to the skeleton; add batch table; drop "Foundations done") |
| **D7 Optimizers** | Given the same landscape, how do SGD / momentum / Adam trace different paths, and why does momentum help? | Build: same loop, only the UPDATE changes → SGD step → momentum velocity → Adam per-param; **live `gradient-descent.html` embed**; Playground: SGD vs momentum on the same weight | Scroll; iframe lr slider; playground prints one step of each rule | "All optimizers are the same"; "Adam is strictly better" | **P0** (exists — fix momentum step-2 to the **real** continuation `g=−9.6, v=−24.0`; give β₂/ε defaults) |
| **D8 Learning rate** | What happens to the loss when the LR is too small / right / too big, and why is it the top knob? | Build: lr scales every step → too-small crawls → just-right → too-big explodes → warmup/decay shapes; **live `gradient-descent.html` embed** with the lr slider | Drag lr across orders of magnitude; Step/Run watches crawl vs converge vs diverge | "Bigger LR is always faster"; "LR is a fixed constant" | **P0** (exists — align playground/produce step counts; add a symbol-disambiguation note for `c` vs `v`) |
| **D9 Train/val/test** | Why does an ever-falling training loss eventually mean the model is getting worse? | Build: one dataset → 3 boxes → both losses fall → **the gap opens** (val turns up) → early-stop marker → test used once | Scroll reveals the diverging two-curve plot and the early-stop point | "Lower train loss is always better"; "one dataset is enough"; "overfitting = a bug" | **P0** for the two-curve diverge build (exists — fix best-model to **step 200**; tie overfitting to capacity). A **live** complexity/epoch slider is **P1** (justified deferral: the static two-curve build is adequate for the capstone; the Produce artifact supplies the interactive version) |

---

## Reuse / build notes

- **`viz/gradient-descent.html`** already exists and is embedded on D5/D7/D8 via the `.build-embed`
  iframe pattern. Keep it. If any embed sticks at 520px, the autoresize `postMessage` sender must be
  present in the viz page (see MEMORY: viz-iframe-autoresize-bug — copy the sender IIFE from
  `viz/broadcasting.html` if missing). Do not add new iframes for P0.
- **SVG legibility on dark themes:** the `.build-viz` panel deliberately forces a light canvas
  (`background:#F7F9FB`) so the hardcoded light-palette SVG fills read correctly. This is the existing,
  accepted convention across the module — keep it consistent; do not half-migrate some SVGs to dark.
- **Playground = the tiny worked example.** Each day's playground numbers must be the *same* numbers as
  that day's Math Ladder and Produce acceptance criteria (this is the anchor-spine rule; the current
  drift on D2/D7/D8/D9 is the fix target).

## Visual QA (checked in `/frontier-refactor-qa`)

- [ ] Every lesson's P0 visual exists and shows a changing quantity
- [ ] Playground numbers == Math-Ladder numbers == Produce acceptance numbers (per day)
- [ ] D2 uses ONE anchor matrix pair everywhere; scalar collapse shown before matrix collapse
- [ ] D3 shows at least one concrete numeric forward pass (not shapes-only)
- [ ] D5 shows a numeric multi-link chain-rule computation
- [ ] All `gradient-descent.html` embeds render and auto-size (no stuck 520px blank box)
- [ ] SVGs render legibly on dim/dark/midnight (light `.build-viz` canvas retained)
