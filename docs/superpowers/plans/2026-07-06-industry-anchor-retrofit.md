# Industry-anchor teaching principle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen the existing "Why a frontier lab cares" section in the 19 M01–M04 lessons that already have it, naming real verified systems instead of staying abstract, and add a matching teaser callout earlier in each lesson. Codify the pattern as a standing rule so future lessons inherit it.

**Architecture:** No new file types, no new `.sec` sections, no test framework — this is a content/copy change to existing self-contained HTML lesson files plus three markdown skill-guidance files. "Tests" here are structural regression checks (jsdom render-check via the repo's existing `sessions/staff_lens_audit.js`) plus grep assertions that the named fact text landed in both touchpoints.

**Tech Stack:** Plain HTML/CSS/JS lesson files (no build step), Node.js + jsdom for render verification, git for commits.

**Spec:** `docs/superpowers/specs/2026-07-06-industry-anchor-teaching-principle-design.md`

---

## Adaptation note for the executor

This codebase has no unit-test framework for lesson content — lessons are static, self-contained HTML files (see `.claude/skills/frontier-session-coach/SESSION_TEMPLATE.md`). The closest thing to "red/green" here is:
1. Before editing: note the file's current state (nothing to run — there's no failing-test step for prose).
2. After editing: run `node sessions/staff_lens_audit.js <module_prefix>` — a jsdom-based script that renders the lesson and checks structure (7 `.sec` elements, 4 quiz questions, 16 quiz options, 0 JS errors). This is the "did I break it" check.
3. Grep-assert the named fact string appears in both the new hook callout and the rewritten section 4 — this is the "did I do it" check.

Every task below follows this shape instead of TDD's red-green. Do not skip the render-check — it is the only mechanical guard against a stray unclosed tag or broken JS string.

**This repo is sometimes edited by concurrent sessions** (see project memory). Every `old_string` below was extracted directly from the live files while writing this plan, but a file can drift between plan-writing and execution time. Before running any task's Edit step, do a quick `grep -n` for a short unique substring of that `old_string` in the target file to confirm it's still there verbatim; if it isn't, re-read the current surrounding lines and adjust the block rather than forcing a stale match. Task 3 already needed one such correction during plan review — see its note below.

**Per-repo rule (from `CLAUDE.md`): write one file per session, no parallel subagents composing prose.** All 19 lessons' exact copy is pre-written below — executing this plan (whether inline or via subagent-driven-development) is mechanical application of exact text via the Edit tool, not fresh composition, so tone drift is not a risk even if subagent-driven-development dispatches one subagent per task. Tasks must still run one at a time, in the order below, each ending in its own commit.

---

## Task 0: Codify the Industry Anchor rule in the skill's teaching principles

**Files:**
- Modify: `.claude/skills/frontier-session-coach/TEACHING_PRINCIPLES.md`
- Modify: `.claude/skills/frontier-session-coach/SESSION_TEMPLATE.md`
- Modify: `.claude/skills/frontier-session-coach/SKILL.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Add the "Industry Anchor" rule to `TEACHING_PRINCIPLES.md`**

Use the Edit tool. Find this exact block (the end of the "Staff Lens" section, right before "## Language rule"):

```markdown
   (Model: `attention-lab-course/modules/04-the-math.html` — "the wrong-axis bug is the one
   a senior engineer catches in code review... the O(n²) trade-off is the one they raise in
   a design review.")
2. **At least one quiz question must be diagnostic, not recall.** Instead of "what does
   symbol X mean," ask "a teammate's code runs without errors but training stalls / the
   metric looks off — where do you look first, and why?" Recall questions are fine for the
   other three; at least one must force the reader to reason like they're debugging.

## Language rule
```

Replace it with:

```markdown
   (Model: `attention-lab-course/modules/04-the-math.html` — "the wrong-axis bug is the one
   a senior engineer catches in code review... the O(n²) trade-off is the one they raise in
   a design review.")
2. **At least one quiz question must be diagnostic, not recall.** Instead of "what does
   symbol X mean," ask "a teammate's code runs without errors but training stalls / the
   metric looks off — where do you look first, and why?" Recall questions are fine for the
   other three; at least one must force the reader to reason like they're debugging.

## Industry Anchor
*(applies to every lesson's "Frontier-lab relevance" layer — section 4's "why a frontier lab
cares" content, and its teaser in section 1)*

"A frontier lab" and "a 7-billion-parameter model" are not concrete enough. Name a real,
verifiable system:

1. **Section 4's frontier-relevance paragraph must open by naming one real system** — a
   specific model, paper, or product — with a specific number attached (a parameter count, a
   token count, a precision format, a paper's finding). Not "modern frameworks" or "a frontier
   lab"; say "DeepSeek-V3" or "Meta's Llama 3" or "Kaplan et al.'s 2020 scaling-law paper."
2. **Section 1 gets a matching one-to-two-sentence teaser callout** — same named system, same
   fact, compressed — appended at the end of section 1's body (reuse the existing
   `.callout.c-info` style, icon 🏭), so the reader meets the real-world hook before the
   mechanism, not only after it. This does not add a new `.sec` element.
3. **Facts must be well-established or paper-verified.** If you can't independently confirm a
   number against a primary source, hedge the language ("Meta has reported…", "per its
   published config…") rather than presenting it as a verbatim citation.

## Language rule
```

- [ ] **Step 2: Point `SESSION_TEMPLATE.md` at the new rule**

Use the Edit tool. Find:

```markdown
4. **机制/为什么 Mechanism & why** — the rule + frontier relevance + causal chain. Include a
   named failure-mode callout and a named trade-off callout — see Staff Lens in
   `TEACHING_PRINCIPLES.md`.
```

Replace with:

```markdown
4. **机制/为什么 Mechanism & why** — the rule + frontier relevance + causal chain. Include a
   named failure-mode callout and a named trade-off callout — see Staff Lens in
   `TEACHING_PRINCIPLES.md`. The frontier-relevance passage must name a real system — see
   Industry Anchor in `TEACHING_PRINCIPLES.md`.
```

Then find section 1's line:

```markdown
1. **是什么 What is it** — Ground Zero Rule: what/why/how-it-relates + prerequisites. Mandatory.
```

Replace with:

```markdown
1. **是什么 What is it** — Ground Zero Rule: what/why/how-it-relates + prerequisites. Mandatory.
   Ends with a short named-system teaser callout — see Industry Anchor in
   `TEACHING_PRINCIPLES.md`.
```

- [ ] **Step 3: Point `SKILL.md` at the new rule**

Use the Edit tool. Find (around line 111):

```markdown
   Include a named failure-mode callout and a named trade-off callout — see Staff Lens in
```

Read the surrounding 5 lines first with the Read tool to get exact context (this line appears once, in the section-4 contract description), then append one sentence after that paragraph: "The frontier-relevance passage must name a real system — see Industry Anchor in `TEACHING_PRINCIPLES.md`."

Also find (around line 125):

```markdown
   recall — see Staff Lens in `TEACHING_PRINCIPLES.md`.
```

Read the surrounding lines (this is the section-1 or section-6 contract description) and add a matching pointer for the section-1 teaser requirement if section 1's contract is described nearby; otherwise skip if `SKILL.md`'s section-1 description is too terse to need it — `SESSION_TEMPLATE.md`'s pointer (Step 2 above) is the source of truth either way.

- [ ] **Step 4: Add the Global rules line to `ROADMAP.md`**

Use the Edit tool. Find:

```markdown
**Staff-lens bar** *(added 2026-07-04)*: any lesson with a Mechanism/math section needs a named **failure-mode** callout and a named **trade-off** callout, plus at least one diagnostic-style quiz question — see the Staff Lens rule in `.claude/skills/frontier-session-coach/TEACHING_PRINCIPLES.md`. Model: `attention-lab-course/modules/04-the-math.html`.
```

Replace with:

```markdown
**Staff-lens bar** *(added 2026-07-04)*: any lesson with a Mechanism/math section needs a named **failure-mode** callout and a named **trade-off** callout, plus at least one diagnostic-style quiz question — see the Staff Lens rule in `.claude/skills/frontier-session-coach/TEACHING_PRINCIPLES.md`. Model: `attention-lab-course/modules/04-the-math.html`.

**Industry-anchor bar** *(added 2026-07-06)*: the "why a frontier lab cares" passage in section 4 must name one real, verifiable system (not "a frontier lab" in the abstract), with a matching teaser callout in section 1 — see the Industry Anchor rule in `.claude/skills/frontier-session-coach/TEACHING_PRINCIPLES.md`.
```

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/frontier-session-coach/TEACHING_PRINCIPLES.md .claude/skills/frontier-session-coach/SESSION_TEMPLATE.md .claude/skills/frontier-session-coach/SKILL.md ROADMAP.md
git commit -m "docs: add Industry Anchor teaching principle (frontier-session-coach)"
```

---

## Tasks 1–19: retrofit the 19 M01–M04 lessons

Each task below follows the same 5-step shape:
1. Edit — insert the section-1 hook callout (before the section's `gotit` button).
2. Edit — rewrite the section-4 frontier-relevance text (or, for the two flagged exceptions, the inline callout lead-in).
3. (Only Tasks 12 and 14) Edit — update the echoed phrase in the section-5 BUILD-step `note:` string.
4. Verify — grep assertion + jsdom render-check.
5. Commit.

For every Edit step: use the Edit tool with the exact `old_string` shown (copy it verbatim — these were extracted directly from the current files) and the exact `new_string` shown.

For every render-check step: run `node sessions/staff_lens_audit.js <module_prefix>` (module_prefix = `m01`, `m02`, `m03`, or `m04` matching the task). Expected: the printed row for that lesson's `mod/day` shows `render:ok` inside the bracket (e.g. `[fail:1 trade:1 warn:1 q:4 render:ok]`) — the `render:ok` token is what matters; the `fail`/`trade`/`warn` values are pre-existing Staff Lens results, unrelated to this change, and should be unchanged from before your edit. Do not worry about the script's overall exit code — other lessons in the module may already be listed as `gap` from the Staff Lens pass; that's expected and out of scope here.

### Task 1: M01 Day 1 — Arrays (`sessions/m01-shape-of-data/day-01-arrays/lesson.html`)

**Named system:** GPT-3's published config (96 layers, d_model 12288, 96 heads) → one FFN weight matrix, shape (12288, 49152).

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <p>The single most important thing about an array is its <span class="term" data-tip="A tuple giving the size of an array along each dimension, e.g. (2, 3) = 2 rows, 3 columns.">shape</span> — the sizes of its grid. That is today's puzzle. Keep going 👇</p>
      <button class="gotit" type="button">Got what an array is — let's go</button>
```

new_string:
```html
      <p>The single most important thing about an array is its <span class="term" data-tip="A tuple giving the size of an array along each dimension, e.g. (2, 3) = 2 rows, 3 columns.">shape</span> — the sizes of its grid. That is today's puzzle. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> GPT-3's biggest weight matrix is a rigid grid of shape <code>(12288, 49152)</code> — over 600 million numbers, one matrix, inside one model. Shape is not a toy idea.</div></div>
      <button class="gotit" type="button">Got what an array is — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>Every weight, every batch of tokens, every image inside a model is an array (a <span class="term" data-tip="The general word for an array with any number of dimensions (3-D, 4-D, ...). Used everywhere in ML.">tensor</span>). The number-one bug in machine-learning code is a <strong>shape mismatch</strong> — two arrays that don't line up. Reading shapes fluently is the skill that everything later (attention, batching, scaling) is built on. That is why Day 1 is arrays.</p>
```

new_string:
```html
      <p>GPT-3 — the 175-billion-parameter model that helped kick off the LLM era — publishes its own shape recipe: 96 layers, each one 12,288 numbers wide. One single weight matrix inside one layer has shape <code>(12288, 49152)</code> — over 600 million numbers, and dozens of matrices like it are stacked 96 deep. Every weight, every batch of tokens, every image inside a model is an array (a <span class="term" data-tip="The general word for an array with any number of dimensions (3-D, 4-D, ...). Used everywhere in ML.">tensor</span>). The number-one bug in machine-learning code is a <strong>shape mismatch</strong> — two arrays that don't line up. Reading shapes fluently is the skill that everything later (attention, batching, scaling) is built on. That is why Day 1 is arrays.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "GPT-3" sessions/m01-shape-of-data/day-01-arrays/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m01`
Expected: the `m01-shape-of-data/day-01-arrays` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m01-shape-of-data/day-01-arrays/lesson.html
git commit -m "content: name GPT-3's config as the industry anchor for M01 Day 1 (arrays)"
```

### Task 2: M01 Day 2 — Indexing & slicing (`sessions/m01-shape-of-data/day-02-indexing-slicing/lesson.html`)

**Named system:** vLLM's PagedAttention (Kwon et al. 2023) — KV cache paged like OS virtual memory.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <p>Two symbols do all the work today: the <strong>comma</strong> (separate axes) and the <strong>colon</strong> (a range, or "all"). Keep going 👇</p>
      <button class="gotit" type="button">Got the two moves — let's go</button>
```

new_string:
```html
      <p>Two symbols do all the work today: the <strong>comma</strong> (separate axes) and the <strong>colon</strong> (a range, or "all"). Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> vLLM, one of the most-used serving engines for LLMs, is built around <b>PagedAttention</b> — it slices the KV cache into fixed-size blocks the exact same way an operating system pages memory. Slicing isn't just a classroom trick.</div></div>
      <button class="gotit" type="button">Got the two moves — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>Slicing is how models move data all day: grab a training <strong>batch</strong> with <code>data[0:32]</code>, take a token window from the <span class="term" data-tip="The stored keys/values from earlier tokens that a model reuses so it does not recompute them every step. You will meet it later.">KV cache</span> with <code>cache[:, :seq_len]</code>, or pull one attention head's slice out of a big tensor. It happens millions of times per second.</p>
```

new_string:
```html
      <p>Slicing is how models move data all day, and one real system built entirely around it is <b>vLLM</b>, a widely used serving engine. Its core trick, <b>PagedAttention</b>, slices the <span class="term" data-tip="The stored keys/values from earlier tokens that a model reuses so it does not recompute them every step. You will meet it later.">KV cache</span> into small fixed-size blocks — borrowed straight from how an operating system pages memory — so it can serve many conversations at once without wasting space. The same slicing move shows up everywhere else too: grab a training <strong>batch</strong> with <code>data[0:32]</code>, or pull one attention head's slice out of a big tensor. It happens millions of times per second.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "vLLM" sessions/m01-shape-of-data/day-02-indexing-slicing/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m01`
Expected: the `m01-shape-of-data/day-02-indexing-slicing` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m01-shape-of-data/day-02-indexing-slicing/lesson.html
git commit -m "content: name vLLM/PagedAttention as the industry anchor for M01 Day 2 (indexing)"
```

### Task 3: M01 Day 3 — Broadcasting & dtypes (`sessions/m01-shape-of-data/day-03-broadcasting-dtypes/lesson.html`)

**Named system:** DeepSeek-V3 — 671B total / 37B active (MoE), matmuls trained in FP8.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>Why put them together today?</h4>
      <p>Both are about <strong>doing math efficiently at scale</strong>. Broadcasting lets one small array (like a bias) act on a huge one with no copies. The dtype decides the memory bill — and that bill is what forces the whole <span class="term" data-tip="Storing numbers in fewer bits (a smaller dtype box) to shrink a model's memory, at the cost of some precision.">quantization</span> story you'll meet later. Keep going 👇</p>
      <button class="gotit" type="button">Got the two ideas — let's go</button>
```

new_string:
```html
      <h4>Why put them together today?</h4>
      <p>Both are about <strong>doing math efficiently at scale</strong>. Broadcasting lets one small array (like a bias) act on a huge one with no copies. The dtype decides the memory bill — and that bill is what forces the whole <span class="term" data-tip="Storing numbers in fewer bits (a smaller dtype box) to shrink a model's memory, at the cost of some precision.">quantization</span> story you'll meet later. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> DeepSeek-V3 (671 billion parameters) trains almost all of its matrix multiplies in <code>FP8</code> — half the bytes of <code>bfloat16</code>, a quarter of <code>float32</code>. Bytes-per-number is not a small detail; it's a lever labs pull on purpose.</div></div>
      <button class="gotit" type="button">Got the two ideas — let's go</button>
```

**Note:** this file (`day-03-broadcasting-dtypes/`) has visibly diverged since this plan was drafted — several `lesson-redesign-*.html` and `compare*.html` siblings appeared, likely from a concurrent session. Before running this task, re-verify this exact `old_string` still matches `lesson.html` (not one of the redesign variants) with `grep -n "Why put them together today" sessions/m01-shape-of-data/day-03-broadcasting-dtypes/lesson.html` and re-read the surrounding lines; if it has moved again, update the block accordingly rather than forcing the match. Do not touch the `lesson-redesign-*.html`, `compare*.html`, `lesson-before.html`, or `log.md` siblings — those are the user's own in-progress files (per the spec's Scope section).

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>Bytes are money and speed. A 7-billion-parameter model in <code>float32</code> needs ~28 GB just for weights; in <code>bfloat16</code>, ~14 GB — the difference between fitting on one chip or not. Broadcasting is how a bias vector <code>(features,)</code> gets added to a whole batch <code>(batch, features)</code> with no copy.</p>
```

new_string:
```html
      <p>Bytes are money and speed. A 7-billion-parameter model in <code>float32</code> needs ~28 GB just for weights; in <code>bfloat16</code>, ~14 GB — the difference between fitting on one chip or not. <b>DeepSeek-V3</b> goes further: it has 671 billion total parameters (only 37 billion active per token — it's a <span class="term" data-tip="Mixture-of-experts: a model built from many specialist sub-networks, where only a few 'experts' run per token, so the active compute is much smaller than the total size.">mixture-of-experts</span>), and its designers trained almost all of its matrix multiplies in <code>FP8</code> — half of <code>bfloat16</code>'s bytes again. Broadcasting is how a bias vector <code>(features,)</code> gets added to a whole batch <code>(batch, features)</code> with no copy.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "DeepSeek-V3" sessions/m01-shape-of-data/day-03-broadcasting-dtypes/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m01`
Expected: the `m01-shape-of-data/day-03-broadcasting-dtypes` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m01-shape-of-data/day-03-broadcasting-dtypes/lesson.html
git commit -m "content: name DeepSeek-V3's FP8 training as the industry anchor for M01 Day 3 (broadcasting/dtypes)"
```

### Task 4: M01 Day 4 — Matmul & shapes (`sessions/m01-shape-of-data/day-04-matmul-and-shapes/lesson.html`)

**Named system:** Kaplan et al.'s `C≈6ND`; NVIDIA H100's published BF16 throughput.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>How does it relate to what you know?</h4>
      <p>On Day 1 you did element-wise math with <code>*</code>. Matmul is different: <code>@</code> does rows-times-columns, not element-by-element. Why care? A neural-network layer <em>is</em> a matmul: <code>y = x @ W</code> (plus a bias). Learn the shape rule and you can trace data through an entire network. Keep going 👇</p>
      <button class="gotit" type="button">Got what a matmul is — let's go</button>
```

new_string:
```html
      <h4>How does it relate to what you know?</h4>
      <p>On Day 1 you did element-wise math with <code>*</code>. Matmul is different: <code>@</code> does rows-times-columns, not element-by-element. Why care? A neural-network layer <em>is</em> a matmul: <code>y = x @ W</code> (plus a bias). Learn the shape rule and you can trace data through an entire network. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> one NVIDIA H100 chip — the workhorse behind most frontier training runs — can do roughly 1,000 trillion matmul operations a second in <code>bfloat16</code>. Almost all of that is spent on exactly the shape rule you're about to learn.</div></div>
      <button class="gotit" type="button">Got what a matmul is — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>A <span class="term" data-tip="A dense (fully-connected / linear) layer: it maps inputs to outputs by y = xW + b, a matmul plus a bias.">dense layer</span> is literally <code>y = x @ W + b</code>. Attention is matmuls too: scores <code>= Q @ Kᵀ</code>, then output <code>= scores @ V</code>. Matmul is where almost all the <span class="term" data-tip="Floating-point operations — a count of the multiply/add work a computation does. Roughly 2·m·k·n for an (m,k)@(k,n) matmul.">FLOPs</span> go, which is exactly why the compute formula <code>C ≈ 6ND</code> (that you'll meet in the scaling-law modules) is really a matmul count — and why GPUs and TPUs are, at heart, giant matmul machines.</p>
```

new_string:
```html
      <p>A <span class="term" data-tip="A dense (fully-connected / linear) layer: it maps inputs to outputs by y = xW + b, a matmul plus a bias.">dense layer</span> is literally <code>y = x @ W + b</code>. Attention is matmuls too: scores <code>= Q @ Kᵀ</code>, then output <code>= scores @ V</code>. Matmul is where almost all the <span class="term" data-tip="Floating-point operations — a count of the multiply/add work a computation does. Roughly 2·m·k·n for an (m,k)@(k,n) matmul.">FLOPs</span> go — an NVIDIA H100, the chip behind most frontier training runs, is built to do almost nothing but matmuls, at roughly 1,000 trillion of them a second in <code>bfloat16</code>. That's exactly why the compute formula <code>C ≈ 6ND</code> (Kaplan et al., 2020 — you'll meet it in the scaling-law modules) is really a matmul count — and why GPUs and TPUs are, at heart, giant matmul machines.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "H100" sessions/m01-shape-of-data/day-04-matmul-and-shapes/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m01`
Expected: the `m01-shape-of-data/day-04-matmul-and-shapes` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m01-shape-of-data/day-04-matmul-and-shapes/lesson.html
git commit -m "content: name the H100's matmul throughput as the industry anchor for M01 Day 4 (matmul)"
```

### Task 5: M01 Day 5 — Logs & exponents (`sessions/m01-shape-of-data/day-05-logs-and-exponents/lesson.html`)

**Named system:** Kaplan et al.'s scaling exponent (α≈0.05); Chinchilla's 70B/1.4T = 20 tokens/param.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>What is a power law?</h4>
      <p>A <span class="term" data-tip="A relationship y = a·x^b, where one quantity is a fixed power of another. Scaling laws are power laws.">power law</span> is a relationship <code>y = a·x^b</code> — one thing is a power of another. Scaling laws (loss vs compute) are power laws. The magic you'll use: draw a power law on <strong>log-log</strong> axes and it becomes a straight <strong>line</strong>, whose slope is the exponent <code>b</code>. Keep going 👇</p>
      <button class="gotit" type="button">Got the three words — let's go</button>
```

new_string:
```html
      <h4>What is a power law?</h4>
      <p>A <span class="term" data-tip="A relationship y = a·x^b, where one quantity is a fixed power of another. Scaling laws are power laws.">power law</span> is a relationship <code>y = a·x^b</code> — one thing is a power of another. Scaling laws (loss vs compute) are power laws. The magic you'll use: draw a power law on <strong>log-log</strong> axes and it becomes a straight <strong>line</strong>, whose slope is the exponent <code>b</code>. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> DeepMind's Chinchilla paper trained a 70-billion-parameter model on 1.4 trillion tokens — a ratio of 20 tokens per parameter they found by reading the slope of a log-log line, exactly like you're about to.</div></div>
      <button class="gotit" type="button">Got the three words — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>The Kaplan and Chinchilla <strong>scaling laws</strong> say model loss falls as a power law of compute: roughly <code>L ≈ (C/C₀)^(−α)</code>. Labs plot loss versus compute on log-log axes, fit a straight line, and read the slope <code>α</code>. That one number predicts how much better the model gets for every 10× more compute — the business case for spending millions on a training run.</p>
```

new_string:
```html
      <p>Kaplan et al.'s 2020 scaling-law paper found loss falls with compute at roughly <code>α ≈ 0.05</code> — plot loss vs. compute on log-log axes and you get a nearly straight line with that slope. Two years later, DeepMind's <b>Chinchilla</b> paper used the same log-log method to find something more useful: for compute-optimal training, tokens and parameters should grow together, at roughly <b>20 tokens for every parameter</b> — their own 70-billion-parameter model was trained on 1.4 trillion tokens (1.4T ÷ 70B ≈ 20). Labs plot loss versus compute on log-log axes, fit a straight line, and read the slope. That one number predicts how much better the model gets for every 10× more compute — the business case for spending millions on a training run.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "Chinchilla" sessions/m01-shape-of-data/day-05-logs-and-exponents/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m01`
Expected: the `m01-shape-of-data/day-05-logs-and-exponents` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m01-shape-of-data/day-05-logs-and-exponents/lesson.html
git commit -m "content: name Kaplan/Chinchilla's verified numbers as the industry anchor for M01 Day 5 (scaling laws)"
```

### Task 6: M02 Day 1 — Single neuron (`sessions/m02-the-neuron/day-01-single-neuron/lesson.html`)

**Named system:** reuses GPT-3's `(12288, 49152)` FFN matrix from Task 1 — one neuron = one column of that matrix.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>How does it relate to what you know?</h4>
      <p>Step 1 is something you can already do: <code>np.dot(w, x)</code> — the weighted sum <em>is</em> a dot product. So a neuron is just <code>activation(dot(w, x) + b)</code>. Why the activation at the end? That's the secret sauce — next lesson. Keep going 👇</p>
      <button class="gotit" type="button">Got what a neuron is — let's go</button>
```

new_string:
```html
      <h4>How does it relate to what you know?</h4>
      <p>Step 1 is something you can already do: <code>np.dot(w, x)</code> — the weighted sum <em>is</em> a dot product. So a neuron is just <code>activation(dot(w, x) + b)</code>. Why the activation at the end? That's the secret sauce — next lesson. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> that GPT-3 weight matrix from M1 Day 1, shape <code>(12288, 49152)</code>, is really 49,152 neurons like the one you're about to build — just wired to 12,288 inputs each instead of a handful.</div></div>
      <button class="gotit" type="button">Got what a neuron is — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>This is the <strong>atom</strong>. Line up many neurons that all read the same inputs, and computing them all at once is a single <strong>matmul</strong>: <code>x @ W + b</code>, then the activation — exactly M1 Day 4. Stack those layers and you have a network; a frontier model is billions of these neurons. And the <strong>weights</strong> are precisely what training adjusts.</p>
```

new_string:
```html
      <p>This is the <strong>atom</strong>. Line up many neurons that all read the same inputs, and computing them all at once is a single <strong>matmul</strong>: <code>x @ W + b</code>, then the activation — exactly M1 Day 4. Remember GPT-3's <code>(12288, 49152)</code> weight matrix from M1 Day 1? Every one of its 49,152 columns <em>is</em> one neuron just like the one you built today, wired to 12,288 inputs instead of a handful. Stack those layers and you have a network; a frontier model is billions of these neurons. And the <strong>weights</strong> are precisely what training adjusts.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "GPT-3" sessions/m02-the-neuron/day-01-single-neuron/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m02`
Expected: the `m02-the-neuron/day-01-single-neuron` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m02-the-neuron/day-01-single-neuron/lesson.html
git commit -m "content: tie M02 Day 1 (single neuron) to GPT-3's weight matrix from M01 Day 1"
```

### Task 7: M02 Day 2 — Activations (`sessions/m02-the-neuron/day-02-activations/lesson.html`)

**Named system:** SwiGLU (Llama) vs. GELU (GPT-2/3, BERT).

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <p>The activation adds <span class="term" data-tip="Not a straight line. A non-linear step between layers lets a deep network bend and combine features to learn complex patterns.">non-linearity</span> — a bend. Put a bend between each layer and the layers can no longer be squashed into one. That is the entire reason deep networks work. Two common bends: <strong>ReLU</strong> and <strong>sigmoid</strong>. Keep going 👇</p>
      <button class="gotit" type="button">Got why it exists — let's go</button>
```

new_string:
```html
      <p>The activation adds <span class="term" data-tip="Not a straight line. A non-linear step between layers lets a deep network bend and combine features to learn complex patterns.">non-linearity</span> — a bend. Put a bend between each layer and the layers can no longer be squashed into one. That is the entire reason deep networks work. Two common bends: <strong>ReLU</strong> and <strong>sigmoid</strong>. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> Llama's models use an activation called SwiGLU; GPT-2 and GPT-3 use one called GELU. Different frontier labs pick different bends — but every one of them exists for the exact reason you're about to see.</div></div>
      <button class="gotit" type="button">Got why it exists — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p><strong>ReLU</strong> is the workhorse: it's dirt cheap (just a max) and it avoids the <span class="term" data-tip="When gradients shrink toward zero as they pass back through many layers, so early layers barely learn. Sigmoid suffers from this; ReLU largely avoids it.">vanishing-gradient</span> problem that plagues sigmoid in deep stacks (you'll meet gradients in M3). Sigmoid and its cousin <span class="term" data-tip="Turns a vector of scores into probabilities that sum to 1. Used at the output of a classifier.">softmax</span> live mostly at the output, where you want probabilities.</p>
```

new_string:
```html
      <p><strong>ReLU</strong> is the classic workhorse: it's dirt cheap (just a max) and it avoids the <span class="term" data-tip="When gradients shrink toward zero as they pass back through many layers, so early layers barely learn. Sigmoid suffers from this; ReLU largely avoids it.">vanishing-gradient</span> problem that plagues sigmoid in deep stacks (you'll meet gradients in M3). Frontier labs mostly moved past plain ReLU, though: <b>Llama</b>'s models use a fancier bend called <b>SwiGLU</b>, while <b>GPT-2</b> and <b>GPT-3</b> use a smoother one called <b>GELU</b> — different labs, different bends, same reason: break the linear collapse you just saw. Sigmoid and its cousin <span class="term" data-tip="Turns a vector of scores into probabilities that sum to 1. Used at the output of a classifier.">softmax</span> live mostly at the output, where you want probabilities.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "SwiGLU" sessions/m02-the-neuron/day-02-activations/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m02`
Expected: the `m02-the-neuron/day-02-activations` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m02-the-neuron/day-02-activations/lesson.html
git commit -m "content: name SwiGLU/GELU as the industry anchor for M02 Day 2 (activations)"
```

### Task 8: M02 Day 3 — Layers & forward pass (`sessions/m02-the-neuron/day-03-layers-forward-pass/lesson.html`)

**Named system:** vLLM / TensorRT-LLM / SGLang as real serving engines running the forward pass in production.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>How does it relate to what you know?</h4>
      <p>Each layer is a line you can already write: <code>h = relu(x @ W + b)</code>. A forward pass just does that repeatedly, feeding each layer's output into the next. The one rule to respect: the shapes must <strong>chain</strong> (M1 Day 4) — each layer's output width must match the next layer's expected input width. Keep going 👇</p>
      <button class="gotit" type="button">Got layer + forward pass — let's go</button>
```

new_string:
```html
      <h4>How does it relate to what you know?</h4>
      <p>Each layer is a line you can already write: <code>h = relu(x @ W + b)</code>. A forward pass just does that repeatedly, feeding each layer's output into the next. The one rule to respect: the shapes must <strong>chain</strong> (M1 Day 4) — each layer's output width must match the next layer's expected input width. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> vLLM, TensorRT-LLM, and SGLang are real, widely used software built to do just one thing at massive scale: run the forward pass you're about to build.</div></div>
      <button class="gotit" type="button">Got layer + forward pass — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>The forward pass <em>is</em> <span class="term" data-tip="Running a trained model on new input to get an answer. The forward pass is inference. Serving = doing this for real users at scale.">inference</span> — running the model to get an answer. Every time you send a prompt to a model, the server runs a forward pass. It's the single most-served operation in all of AI. The reverse direction, which nudges the weights to make future forward passes better, is <span class="term" data-tip="Running the network backward to compute how each weight should change to reduce the error. How models learn. Covered in Module 3.">backpropagation</span> — Module 3.</p>
```

new_string:
```html
      <p>The forward pass <em>is</em> <span class="term" data-tip="Running a trained model on new input to get an answer. The forward pass is inference. Serving = doing this for real users at scale.">inference</span> — running the model to get an answer. Every time you send a prompt to a model, a server somewhere runs a forward pass — real, widely used serving engines like <b>vLLM</b>, <b>TensorRT-LLM</b>, and <b>SGLang</b> exist purely to run this one operation as fast, and as many times, as possible. It's the single most-served operation in all of AI. The reverse direction, which nudges the weights to make future forward passes better, is <span class="term" data-tip="Running the network backward to compute how each weight should change to reduce the error. How models learn. Covered in Module 3.">backpropagation</span> — Module 3.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "vLLM" sessions/m02-the-neuron/day-03-layers-forward-pass/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m02`
Expected: the `m02-the-neuron/day-03-layers-forward-pass` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m02-the-neuron/day-03-layers-forward-pass/lesson.html
git commit -m "content: name vLLM/TensorRT-LLM/SGLang as the industry anchor for M02 Day 3 (forward pass)"
```

### Task 9: M02 Day 4 — Loss (`sessions/m02-the-neuron/day-04-loss/lesson.html`)

**Named system:** cross-entropy as the literal GPT pretraining loss; Chinchilla's loss-vs-compute framing. Note: "cross-entropy" already has a term-tooltip defined earlier in this file's section 1 (line ~229) — do not add a second `<span class="term">` for it; use plain text.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      </ul>
      <p>Both boil the gap between "what the model said" and "the truth" down to one number. Keep going 👇</p>
      <button class="gotit" type="button">Got what a loss is — let's go</button>
```

new_string:
```html
      </ul>
      <p>Both boil the gap between "what the model said" and "the truth" down to one number. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> every GPT-style model — 10 weights or 175 billion — is trained to shrink the exact same cross-entropy loss you're about to compute today.</div></div>
      <button class="gotit" type="button">Got what a loss is — let's go</button>
```

This `old_string` also appears fine to match uniquely — `</ul>` immediately followed by this exact `<p>` and `<button>` combination occurs once in this file (verify with the grep in Step 3 before editing if unsure; if the Edit tool reports the string is not unique, widen the match to include the preceding `<li>` line for cross-entropy from the byte table above it).

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>Loss is <em>the</em> number. Training minimizes it. The famous <strong>scaling laws</strong> (M1 Day 5, and the scaling modules later) plot <strong>loss vs compute</strong> — the whole game is predicting how low the loss goes as you spend more. "Lower loss = a better model" is the assumption under everything.</p>
```

new_string:
```html
      <p>Loss is <em>the</em> number. Training minimizes it. Every GPT-style model ever shipped — 10 weights or 175 billion — is pretrained to minimize exactly the cross-entropy loss you built today; only the scale changes. The famous <strong>scaling laws</strong> (M1 Day 5, and Chinchilla) plot <strong>loss vs compute</strong> — the whole game is predicting how low the loss goes as you spend more. "Lower loss = a better model" is the assumption under everything.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "GPT-style model" sessions/m02-the-neuron/day-04-loss/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m02`
Expected: the `m02-the-neuron/day-04-loss` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m02-the-neuron/day-04-loss/lesson.html
git commit -m "content: tie M02 Day 4 (loss) to cross-entropy as the real GPT pretraining loss"
```

### Task 10: M02 Day 5 — Gradients & backprop (`sessions/m02-the-neuron/day-05-gradients-backprop/lesson.html`)

**Named system:** `jax.grad` / `torch.autograd`; Kaplan et al.'s "6" in `C≈6ND`.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>What is backpropagation?</h4>
      <p>A network has millions of weights. <span class="term" data-tip="The algorithm that computes the gradient of the loss for every weight, efficiently, by applying the chain rule backward from the loss to the inputs.">Backpropagation</span> is the efficient algorithm that computes the gradient for <strong>all</strong> of them in one sweep — running <em>backward</em> from the loss to the inputs, using the chain rule. Keep going 👇</p>
      <button class="gotit" type="button">Got the three words — let's go</button>
```

new_string:
```html
      <h4>What is backpropagation?</h4>
      <p>A network has millions of weights. <span class="term" data-tip="The algorithm that computes the gradient of the loss for every weight, efficiently, by applying the chain rule backward from the loss to the inputs.">Backpropagation</span> is the efficient algorithm that computes the gradient for <strong>all</strong> of them in one sweep — running <em>backward</em> from the loss to the inputs, using the chain rule. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> jax.grad and torch.autograd — the exact tools frontier labs train with — are just automated versions of the chain rule you're about to run by hand.</div></div>
      <button class="gotit" type="button">Got the three words — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>Backprop is how <strong>every</strong> neural network learns. It's the reverse of the forward pass and costs about the same (so a training step ≈ 2–3× a forward pass — the "6" in the <code>C ≈ 6ND</code> compute formula). Modern frameworks compute it automatically (<span class="term" data-tip="Automatic differentiation: the framework computes gradients for you, e.g. jax.grad. You will meet it in the JAX modules.">autodiff</span>, e.g. <code>jax.grad</code>) — but knowing it's just "chain rule, backward" demystifies the whole thing.</p>
```

new_string:
```html
      <p>Backprop is how <strong>every</strong> neural network learns. It's the reverse of the forward pass and costs about the same (so a training step ≈ 2–3× a forward pass — that's exactly where the "6" in Kaplan et al.'s <code>C ≈ 6ND</code> compute formula comes from). Every major training stack computes it automatically (<span class="term" data-tip="Automatic differentiation: the framework computes gradients for you, e.g. jax.grad. You will meet it in the JAX modules.">autodiff</span>) — <b>JAX</b>'s <code>jax.grad</code>, <b>PyTorch</b>'s <code>torch.autograd</code> — but knowing it's just "chain rule, backward" demystifies the whole thing.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "torch.autograd" sessions/m02-the-neuron/day-05-gradients-backprop/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m02`
Expected: the `m02-the-neuron/day-05-gradients-backprop` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m02-the-neuron/day-05-gradients-backprop/lesson.html
git commit -m "content: name jax.grad/torch.autograd as the industry anchor for M02 Day 5 (backprop)"
```

### Task 11: M02 Day 6 — Training loop (`sessions/m02-the-neuron/day-06-training-loop/lesson.html`)

**Named system:** Meta's reported >15T training tokens for Llama 3.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>Two words you'll see</h4>
      <p>An <span class="term" data-tip="One full pass over the entire training dataset.">epoch</span> is one full pass over all the training data. A <span class="term" data-tip="A small chunk of the dataset processed together in one step, so you update weights often without loading everything at once.">batch</span> is a chunk processed in one step. Real training loops over batches, and over epochs. Keep going 👇</p>
      <button class="gotit" type="button">Got the loop — let's go</button>
```

new_string:
```html
      <h4>Two words you'll see</h4>
      <p>An <span class="term" data-tip="One full pass over the entire training dataset.">epoch</span> is one full pass over all the training data. A <span class="term" data-tip="A small chunk of the dataset processed together in one step, so you update weights often without loading everything at once.">batch</span> is a chunk processed in one step. Real training loops over batches, and over epochs. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> Meta has reported training Llama 3's biggest model on more than 15 trillion tokens — the exact loop you're about to run, just repeated an enormous number of times.</div></div>
      <button class="gotit" type="button">Got the loop — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>This exact loop, scaled to thousands of accelerators and trillions of tokens, <em>is</em> how frontier models are trained. The whole scaling story (M1 Day 5) is "run this loop with more compute and watch the loss follow a power law." Tuning the <span class="term" data-tip="Settings you choose, not learn: learning rate, batch size, number of epochs. They strongly affect training.">hyperparameters</span> — learning rate, batch size — is much of the craft.</p>
```

new_string:
```html
      <p>This exact loop, scaled up, <em>is</em> how frontier models are trained: Meta has reported pretraining <b>Llama 3</b>'s largest model on more than <b>15 trillion tokens</b> — the very same forward → loss → backprop → update loop you just read about, just repeated an almost unimaginable number of times. The whole scaling story (M1 Day 5) is "run this loop with more compute and watch the loss follow a power law." Tuning the <span class="term" data-tip="Settings you choose, not learn: learning rate, batch size, number of epochs. They strongly affect training.">hyperparameters</span> — learning rate, batch size — is much of the craft.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "Llama 3" sessions/m02-the-neuron/day-06-training-loop/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m02`
Expected: the `m02-the-neuron/day-06-training-loop` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m02-the-neuron/day-06-training-loop/lesson.html
git commit -m "content: name Llama 3's reported 15T-token training run as the industry anchor for M02 Day 6"
```

### Task 12: M02 Day 7 — Optimizers (`sessions/m02-the-neuron/day-07-optimizers/lesson.html`)

**Named system:** ZeRO / DeepSpeed (Rajbhandari et al.) — shards Adam's optimizer state across GPUs.
**Structural note:** this file has both the `<h4>` form AND a second echo of the phrase in a section-5 BUILD-step `note:` JS string — both need updating.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      </ol>
      <p>They all move downhill. The smarter ones just choose the step more cleverly. Keep going 👇</p>
      <button class="gotit" type="button">Got it — let's meet them</button>
```

new_string:
```html
      </ol>
      <p>They all move downhill. The smarter ones just choose the step more cleverly. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> Microsoft's ZeRO, used inside the DeepSpeed library, exists to shard Adam's doubled memory cost across many GPUs — a direct fix for the "two extra numbers" you're about to meet.</div></div>
      <button class="gotit" type="button">Got it — let's meet them</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>Adam keeps <code>m</code> and <code>v</code> for <em>every</em> weight. So the <span class="term" data-tip="The extra numbers an optimizer stores per weight; for Adam that is m and v, roughly doubling the memory beyond the weights themselves.">optimizer state</span> is about <b>2× the size of the model's weights</b> — extra memory a frontier lab must pay for on every accelerator. When you reach the memory and parallelism modules (Module 10), this "Adam doubles your state" fact is a big reason huge models are split across many chips.</p>
```

new_string:
```html
      <p>Adam keeps <code>m</code> and <code>v</code> for <em>every</em> weight. So the <span class="term" data-tip="The extra numbers an optimizer stores per weight; for Adam that is m and v, roughly doubling the memory beyond the weights themselves.">optimizer state</span> is about <b>2× the size of the model's weights</b> — extra memory a frontier lab must pay for on every accelerator. Microsoft's <b>ZeRO</b> (used inside the widely-used <b>DeepSpeed</b> library) exists specifically to fix this: it shards that doubled Adam state across many GPUs instead of copying it onto every single one. When you reach the memory and parallelism modules (Module 10), this "Adam doubles your state" fact is a big reason huge models are split across many chips.</p>
```

- [ ] **Step 3: Update the section-5 BUILD-step echo**

old_string:
```html
  note:"<b>Why a frontier lab cares.</b> Adam stores <b>m</b> and <b>v</b> for every weight, so the optimizer state is about <b>2× the model's weights</b> — extra memory on every accelerator. That is a big reason huge models are split across many chips (Module 10). You now know how the step is chosen."}
```

new_string:
```html
  note:"<b>Why a frontier lab cares.</b> Adam stores <b>m</b> and <b>v</b> for every weight, so the optimizer state is about <b>2× the model's weights</b> — extra memory on every accelerator. Microsoft's ZeRO/DeepSpeed exists to shard that doubled state across GPUs. That is a big reason huge models are split across many chips (Module 10). You now know how the step is chosen."}
```

- [ ] **Step 4: Verify**

Run: `grep -c "ZeRO" sessions/m02-the-neuron/day-07-optimizers/lesson.html`
Expected: `3`

Run: `node sessions/staff_lens_audit.js m02`
Expected: the `m02-the-neuron/day-07-optimizers` row shows `render:ok`.

- [ ] **Step 5: Commit**

```bash
git add sessions/m02-the-neuron/day-07-optimizers/lesson.html
git commit -m "content: name ZeRO/DeepSpeed as the industry anchor for M02 Day 7 (optimizers)"
```

### Task 13: M02 Day 8 — Learning rate (`sessions/m02-the-neuron/day-08-learning-rate/lesson.html`)

**Named system:** GPT-3, Chinchilla, and Llama 3 all publish their exact warmup+decay schedule.
**Structural note:** this file has NO `<h4>Why a frontier lab cares</h4>` — the phrase is a bold lead-in inside an existing `<div class="callout c-info">` block. Target that inline lead-in.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>It is a choice, not something learned</h4>
      <p>The learning rate is a <span class="term" data-tip="A setting you pick before training, not a value the model learns: learning rate, batch size, number of epochs.">hyperparameter</span> — you pick it before training starts. The model never learns it for you. Because it multiplies every single update, it is the number that most decides whether a run works. Three cases matter: too big, too small, and just right. Keep going 👇</p>
      <button class="gotit" type="button">Got it — let's tune it</button>
```

new_string:
```html
      <h4>It is a choice, not something learned</h4>
      <p>The learning rate is a <span class="term" data-tip="A setting you pick before training, not a value the model learns: learning rate, batch size, number of epochs.">hyperparameter</span> — you pick it before training starts. The model never learns it for you. Because it multiplies every single update, it is the number that most decides whether a run works. Three cases matter: too big, too small, and just right. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> GPT-3, Chinchilla, and Llama 3 all publish their exact learning-rate warmup-and-decay schedule as a named part of how they were trained — not a minor detail, a headline decision.</div></div>
      <button class="gotit" type="button">Got it — let's tune it</button>
```

- [ ] **Step 2: Rewrite the inline callout lead-in in section 4**

old_string:
```html
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Why a frontier lab cares:</b> a good learning rate (and its warmup + decay schedule) is one of the highest-value knobs in a huge training run. A rate slightly too high can waste millions of dollars of compute by diverging; too low wastes it by crawling. This is a core part of the craft you will see in the scaling and JAX modules.</div></div>
```

new_string:
```html
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Why a frontier lab cares:</b> a good learning rate (and its warmup + decay schedule) is one of the highest-value knobs in a huge training run. It matters enough that <b>GPT-3</b>, <b>Chinchilla</b>, and <b>Llama 3</b> all publish their exact warmup-and-decay schedule as a named part of their training methodology, not an afterthought. A rate slightly too high can waste millions of dollars of compute by diverging; too low wastes it by crawling. This is a core part of the craft you will see in the scaling and JAX modules.</div></div>
```

- [ ] **Step 3: Verify**

Run: `grep -c "Llama 3" sessions/m02-the-neuron/day-08-learning-rate/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m02`
Expected: the `m02-the-neuron/day-08-learning-rate` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m02-the-neuron/day-08-learning-rate/lesson.html
git commit -m "content: name GPT-3/Chinchilla/Llama-3's published LR schedules as the industry anchor for M02 Day 8"
```

### Task 14: M02 Day 9 — Train/val/test (`sessions/m02-the-neuron/day-09-train-val-test/lesson.html`)

**Named system:** benchmark decontamination practice (MMLU, GSM8K) described in Llama 3 / GPT-4-class reports.
**Structural note:** no `<h4>` — inline bold lead-in inside a callout, same as Task 13. ALSO has a section-5 BUILD-step echo, same as Task 12 — both need updating.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>Why split at all?</h4>
      <p>If you only ever measure the loss on the data the model trained on, you cannot tell learning from memorizing. Held-out data is the only fair test. Keep going 👇</p>
      <button class="gotit" type="button">Got the splits — let's go</button>
```

new_string:
```html
      <h4>Why split at all?</h4>
      <p>If you only ever measure the loss on the data the model trained on, you cannot tell learning from memorizing. Held-out data is the only fair test. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> the Llama 3 and GPT-4-class technical reports both describe checking their training data for overlap with benchmarks like MMLU and GSM8K before trusting a score — the same held-out-data discipline you're about to learn.</div></div>
      <button class="gotit" type="button">Got the splits — let's go</button>
```

- [ ] **Step 2: Rewrite the inline callout lead-in in section 4**

old_string:
```html
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Why a frontier lab cares:</b> a huge model can memorize enormous amounts of data. Held-out evaluation is how a lab knows its model truly generalizes and is not just parroting its training set. Every serious result is reported on data the model never trained on. That completes the "how a model learns — and how we trust it" story.</div></div>
```

new_string:
```html
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Why a frontier lab cares:</b> a huge model can memorize enormous amounts of data. Held-out evaluation is how a lab knows its model truly generalizes and is not just parroting its training set. It matters so much that the <b>Llama 3</b> and GPT-4-class technical reports both describe explicitly checking their training data for overlap with benchmarks like <b>MMLU</b> and <b>GSM8K</b> — a practice called <span class="term" data-tip="Removing or flagging training examples that overlap with a benchmark's test set, so the reported score reflects genuine generalization rather than memorized answers.">decontamination</span> — before trusting the score. Every serious result is reported on data the model never trained on. That completes the "how a model learns — and how we trust it" story.</div></div>
```

- [ ] **Step 3: Update the section-5 BUILD-step echo**

old_string:
```html
  note:"<b>Why a frontier lab cares.</b> A huge model can memorize enormous amounts of data, so held-out evaluation is the only way to prove it truly generalizes. Every serious result is reported on data the model never trained on. You now know how a model learns — and how we trust it. Foundations done."}
```

new_string:
```html
  note:"<b>Why a frontier lab cares.</b> A huge model can memorize enormous amounts of data, so held-out evaluation is the only way to prove it truly generalizes. Real labs even check their training data against benchmarks like MMLU and GSM8K for overlap — a step called decontamination — before trusting a score. Every serious result is reported on data the model never trained on. You now know how a model learns — and how we trust it. Foundations done."}
```

- [ ] **Step 4: Verify**

Run: `grep -c "MMLU" sessions/m02-the-neuron/day-09-train-val-test/lesson.html`
Expected: `3`

Run: `node sessions/staff_lens_audit.js m02`
Expected: the `m02-the-neuron/day-09-train-val-test` row shows `render:ok`.

- [ ] **Step 5: Commit**

```bash
git add sessions/m02-the-neuron/day-09-train-val-test/lesson.html
git commit -m "content: name Llama-3/GPT-4-class decontamination practice as the industry anchor for M02 Day 9"
```

### Task 15: M03 Day 1 — Embeddings (`sessions/m03-attention/day-01-embeddings/lesson.html`)

**Named system:** Llama 3's 128,256-token vocab vs. GPT-4o's ~200k `o200k_base` tokenizer.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>How is it stored?</h4>
      <p>As a big lookup table — an <span class="term" data-tip="A matrix of shape (vocabulary size, d_model). Row i is the embedding vector for token id i.">embedding matrix</span>. Give it token id <code>i</code> and it hands back row <code>i</code> (indexing a row — M1 Day 2!). Those numbers aren't hand-set; they're <strong>learned during training</strong> (M3's loop), so meaning gradually becomes geometry. Keep going 👇</p>
      <button class="gotit" type="button">Got what an embedding is — let's go</button>
```

new_string:
```html
      <h4>How is it stored?</h4>
      <p>As a big lookup table — an <span class="term" data-tip="A matrix of shape (vocabulary size, d_model). Row i is the embedding vector for token id i.">embedding matrix</span>. Give it token id <code>i</code> and it hands back row <code>i</code> (indexing a row — M1 Day 2!). Those numbers aren't hand-set; they're <strong>learned during training</strong> (M3's loop), so meaning gradually becomes geometry. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> Llama 3's tokenizer has a 128,256-token vocabulary; GPT-4o's has roughly 200,000. Each of those is a row in a real embedding matrix, exactly like the one you're about to build.</div></div>
      <button class="gotit" type="button">Got what an embedding is — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>This is <strong>step 1 of every language model</strong>: text → tokens → embeddings, and everything after (attention, the layers you built in M2) operates on these vectors. <code>d_model</code> is a core size knob, and the embedding matrix is a big share of a model's parameters (vocab is often 50k–250k rows). Get the input representation right and the rest has something meaningful to work with.</p>
```

new_string:
```html
      <p>This is <strong>step 1 of every language model</strong>: text → tokens → embeddings, and everything after (attention, the layers you built in M2) operates on these vectors. <code>d_model</code> is a core size knob, and the embedding matrix is a big share of a model's parameters. <b>Llama 3</b>'s tokenizer has a vocabulary of <b>128,256</b> tokens; <b>GPT-4o</b>'s has roughly <b>200,000</b> — either way, that count is the number of rows in a real embedding matrix, one learned vector per possible token. Get the input representation right and the rest has something meaningful to work with.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "128,256" sessions/m03-attention/day-01-embeddings/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m03`
Expected: the `m03-attention/day-01-embeddings` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m03-attention/day-01-embeddings/lesson.html
git commit -m "content: name Llama-3/GPT-4o vocab sizes as the industry anchor for M03 Day 1 (embeddings)"
```

### Task 16: M03 Day 2 — QKV (`sessions/m03-attention/day-02-qkv/lesson.html`)

**Named system:** Llama 3 405B's 128 Query heads / 8 KV heads (per Meta's published model config).

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>How are they made?</h4>
      <p>From the same embedding <code>x</code>, using three separate <strong>learned</strong> weight matrices: <code>Q = x @ W_Q</code>, <code>K = x @ W_K</code>, <code>V = x @ W_V</code>. Three matmuls (M1 Day 4). One input, three roles. Keep going 👇</p>
      <button class="gotit" type="button">Got the three roles — let's go</button>
```

new_string:
```html
      <h4>How are they made?</h4>
      <p>From the same embedding <code>x</code>, using three separate <strong>learned</strong> weight matrices: <code>Q = x @ W_Q</code>, <code>K = x @ W_K</code>, <code>V = x @ W_V</code>. Three matmuls (M1 Day 4). One input, three roles. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> Llama 3's biggest model runs 128 separate Query heads per layer, each with its own learned projection — the same Q/K/V split you're about to build, just at industrial scale.</div></div>
      <button class="gotit" type="button">Got the three roles — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>Q/K/V is the <strong>heart of every transformer</strong>. It's what lets a model resolve "bank" from context, link a pronoun to its noun, or connect a question to its answer. The projection matrices <code>W_Q, W_K, W_V</code> are a big share of a model's parameters, and computing <code>Q @ Kᵀ</code> is one of the matmuls that dominate its FLOPs (M1 Day 4).</p>
```

new_string:
```html
      <p>Q/K/V is the <strong>heart of every transformer</strong>. It's what lets a model resolve "bank" from context, link a pronoun to its noun, or connect a question to its answer. <b>Llama 3</b>'s biggest model runs <b>128</b> separate Query heads per layer, each with its own learned <code>W_Q</code> slice, while sharing just <b>8</b> Key/Value heads among them — a memory-saving trick you'll meet on Day 4. The projection matrices are a big share of a model's parameters, and computing <code>Q @ Kᵀ</code> is one of the matmuls that dominate its FLOPs (M1 Day 4).</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "Llama 3" sessions/m03-attention/day-02-qkv/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m03`
Expected: the `m03-attention/day-02-qkv` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m03-attention/day-02-qkv/lesson.html
git commit -m "content: name Llama 3 405B's head config as the industry anchor for M03 Day 2 (QKV)"
```

### Task 17: M03 Day 3 — Attention scores (`sessions/m03-attention/day-03-attention-scores/lesson.html`)

**Named system:** FlashAttention (Dao et al., 2022); Gemini 1.5's 1–2M token context.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>Step 3 — weighted sum</h4>
      <p>Multiply those weights by the <strong>Values</strong>: <code>output = weights @ V</code>. Each word gets a blend of the other words' Values, weighted by relevance — a new, <strong>context-aware</strong> vector. The full formula: <code>softmax(Q @ Kᵀ / √d_k) @ V</code>. Keep going 👇</p>
      <button class="gotit" type="button">Got the three steps — let's go</button>
```

new_string:
```html
      <h4>Step 3 — weighted sum</h4>
      <p>Multiply those weights by the <strong>Values</strong>: <code>output = weights @ V</code>. Each word gets a blend of the other words' Values, weighted by relevance — a new, <strong>context-aware</strong> vector. The full formula: <code>softmax(Q @ Kᵀ / √d_k) @ V</code>. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> Gemini 1.5 can hold 1–2 million tokens of context. The reason that's hard: attention's score grid is seq × seq, and this exact <code>Q @ Kᵀ</code> step is what makes long context expensive.</div></div>
      <button class="gotit" type="button">Got the three steps — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>This is <strong>the</strong> transformer operation. Two facts fall straight out of it: (1) <code>Q @ Kᵀ</code> and the final <code>@ V</code> are the big matmuls that dominate attention's FLOPs — exactly what <span class="term" data-tip="A GPU-optimized way to compute attention without ever storing the full seq×seq score matrix in slow memory. You'll meet it in the kernels track.">FlashAttention</span> speeds up. (2) The score grid is <code>seq × seq</code>, so cost grows with the <strong>square</strong> of the sequence length — the reason long context is expensive and a whole research area exists to fix it.</p>
```

new_string:
```html
      <p>This is <strong>the</strong> transformer operation. Two facts fall straight out of it: (1) <code>Q @ Kᵀ</code> and the final <code>@ V</code> are the big matmuls that dominate attention's FLOPs — exactly what <span class="term" data-tip="A GPU-optimized way to compute attention without ever storing the full seq×seq score matrix in slow memory. You'll meet it in the kernels track.">FlashAttention</span> (Dao et al., 2022) speeds up. (2) The score grid is <code>seq × seq</code>, so cost grows with the <strong>square</strong> of the sequence length — the reason long context is expensive, and exactly why <b>Gemini 1.5</b> holding 1–2 million tokens of context is a genuine engineering feat, not just "make the window bigger." A whole research area exists to fight this quadratic cost.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "Gemini 1.5" sessions/m03-attention/day-03-attention-scores/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m03`
Expected: the `m03-attention/day-03-attention-scores` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m03-attention/day-03-attention-scores/lesson.html
git commit -m "content: name FlashAttention/Gemini-1.5 as the industry anchor for M03 Day 3 (attention scores)"
```

### Task 18: M03 Day 4 — Multi-head attention (`sessions/m03-attention/day-04-multihead/lesson.html`)

**Named system:** GQA paper (Ainslie et al., 2023, Google); Llama 3 / Mistral use it in production.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>Putting them back together</h4>
      <p>Run all heads, <strong>concatenate</strong> their outputs back to length <code>d_model</code>, then mix them with one final learned matrix <span class="term" data-tip="The output projection matrix in multi-head attention. It mixes the concatenated head outputs back into one d_model-sized vector.">W_O</span>. The result is one richer vector per word. Keep going 👇</p>
      <button class="gotit" type="button">Got the idea — let's go</button>
```

new_string:
```html
      <h4>Putting them back together</h4>
      <p>Run all heads, <strong>concatenate</strong> their outputs back to length <code>d_model</code>, then mix them with one final learned matrix <span class="term" data-tip="The output projection matrix in multi-head attention. It mixes the concatenated head outputs back into one d_model-sized vector.">W_O</span>. The result is one richer vector per word. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> Llama 3 and Mistral both ship Google's GQA trick in production — sharing Keys/Values across groups of heads to shrink memory, instead of giving every head its own full copy.</div></div>
      <button class="gotit" type="button">Got the idea — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>Every real transformer is multi-head — small models use 8–16 heads, big ones 32–128. Heads visibly <strong>specialize</strong> (some track syntax, some copy earlier tokens). It also sets up a big inference cost: the Keys and Values for every head must be cached, which is why later tricks like <span class="term" data-tip="Grouped-Query / Multi-Query Attention: share Keys and Values across heads to shrink the KV cache and speed up inference. Covered in the decoding modules.">GQA / MQA</span> share K/V across heads to save memory. Multi-head is the last piece of the attention block.</p>
```

new_string:
```html
      <p>Every real transformer is multi-head — small models use 8–16 heads, big ones 32–128 (<b>Llama 3</b>'s biggest runs 128). Heads visibly <strong>specialize</strong> (some track syntax, some copy earlier tokens). It also sets up a big inference cost: the Keys and Values for every head must be cached, which is why later tricks like <span class="term" data-tip="Grouped-Query / Multi-Query Attention: share Keys and Values across heads to shrink the KV cache and speed up inference. Covered in the decoding modules.">GQA / MQA</span> — GQA comes from a 2023 Google paper and now ships in production inside <b>Llama 3</b> and <b>Mistral</b> — share K/V across heads to save memory. Multi-head is the last piece of the attention block.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "Mistral" sessions/m03-attention/day-04-multihead/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m03`
Expected: the `m03-attention/day-04-multihead` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m03-attention/day-04-multihead/lesson.html
git commit -m "content: name the GQA paper/Llama-3/Mistral as the industry anchor for M03 Day 4 (multi-head)"
```

### Task 19: M04 Day 1 — MLP MNIST (`sessions/m04-first-model-mlp/day-01-mlp-mnist/lesson.html`)

**Named system:** cross-entropy loss identity between this MNIST MLP and real LLM pretraining. Note: this file already uses "cross-entropy" as plain text throughout with no term-tooltip span — match that convention, do not add a new span.

- [ ] **Step 1: Insert the section-1 hook callout**

old_string:
```html
      <h4>Train, then test</h4>
      <p>Train with the M3 loop on 60,000 images, then measure <span class="term" data-tip="The fraction of test examples the model classifies correctly. Measured on data not used in training.">accuracy</span> on 10,000 <strong>held-out</strong> images the model never trained on. Testing on unseen data is how you know it truly learned, not just memorized. Keep going 👇</p>
      <button class="gotit" type="button">Got the setup — let's go</button>
```

new_string:
```html
      <h4>Train, then test</h4>
      <p>Train with the M3 loop on 60,000 images, then measure <span class="term" data-tip="The fraction of test examples the model classifies correctly. Measured on data not used in training.">accuracy</span> on 10,000 <strong>held-out</strong> images the model never trained on. Testing on unseen data is how you know it truly learned, not just memorized. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> the cross-entropy loss you're about to use on 10 digit classes is the exact same formula a GPT-style model uses to predict its next token out of a 100,000+ word vocabulary.</div></div>
      <button class="gotit" type="button">Got the setup — let's go</button>
```

- [ ] **Step 2: Rewrite the section-4 paragraph**

old_string:
```html
      <p>This tiny classifier is the <strong>same recipe</strong> as a giant LLM: represent input as vectors, forward pass, loss, backprop, loop. Only the scale and the architecture change. And the <strong>train/test split + accuracy</strong> is your first taste of evaluation — the discipline that keeps you honest as models get bigger.</p>
```

new_string:
```html
      <p>This tiny classifier is the <strong>same recipe</strong> as a giant LLM: represent input as vectors, forward pass, loss, backprop, loop. In fact the loss function is <em>identical</em> — the cross-entropy loss you used on 10 digit classes is the exact same formula a GPT-style model uses to predict its next token out of a 100,000+ word vocabulary. Only the scale and the architecture change. And the <strong>train/test split + accuracy</strong> is your first taste of evaluation — the discipline that keeps you honest as models get bigger.</p>
```

- [ ] **Step 3: Verify**

Run: `grep -c "100,000+ word vocabulary" sessions/m04-first-model-mlp/day-01-mlp-mnist/lesson.html`
Expected: `2`

Run: `node sessions/staff_lens_audit.js m04`
Expected: the `m04-first-model-mlp/day-01-mlp-mnist` row shows `render:ok`.

- [ ] **Step 4: Commit**

```bash
git add sessions/m04-first-model-mlp/day-01-mlp-mnist/lesson.html
git commit -m "content: name the shared cross-entropy loss as the industry anchor for M04 Day 1 (MLP MNIST)"
```

---

## Task 20: Record the retrofit in ROADMAP.md

**Files:**
- Modify: `ROADMAP.md`

- [ ] **Step 1: Add the retrofit summary subsection**

Use the Edit tool. Find the line just before the Staff-lens retrofit section:

```markdown
- **Legend:** ⬜ unclaimed · 🔄 in progress · ✅ done & verified · ⛔ blocked.

### Staff-lens retrofit — ✅ COMPLETE (curriculum-wide sweep, 2026-07-04)
```

Replace with:

```markdown
- **Legend:** ⬜ unclaimed · 🔄 in progress · ✅ done & verified · ⛔ blocked.

### Industry-anchor retrofit — ✅ COMPLETE (M01–M04 only, 2026-07-06)
The Industry Anchor bar (Global rules, above) was added after the user singled out
`m01-shape-of-data/day-03-broadcasting-dtypes` as content they wanted more of: naming a real
system instead of staying abstract ("a frontier lab," "a 7B model"). Audited the curriculum
and found this "why a frontier lab cares" section already existed in exactly 19 lessons
(M01 Days 1–5, M02 Days 1–9, M03 Days 1–4, M04 Day 1) — all abstract. All 19 were retrofitted:
each got a matching section-1 teaser callout plus a section-4 rewrite naming a real, checked
system (DeepSeek-V3's FP8 training, vLLM's PagedAttention, Llama 3's published config and
15T-token training run, GPT-3's Table 2.1 config, Kaplan/Chinchilla's scaling numbers,
FlashAttention, GQA, ZeRO/DeepSpeed, and more — see the fact table in
`docs/superpowers/specs/2026-07-06-industry-anchor-teaching-principle-design.md`).

**Result: 19 of 19 targeted lessons retrofitted across 4 modules; all render-checked clean via
`sessions/staff_lens_audit.js` (7 sections, 4 quiz, 16 options, 0 JS errors, unchanged from
before the edit).**

**M05–M29 (137 lessons) are an explicit, deferred watchlist item — not scheduled.** They don't
carry this section at all today (different "Why it matters" headers); a future session should
decide whether to add the Industry Anchor pattern to them or leave those modules' existing
relevance framing as-is.

### Staff-lens retrofit — ✅ COMPLETE (curriculum-wide sweep, 2026-07-04)
```

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "docs: record the M01-M04 industry-anchor retrofit in ROADMAP.md"
```

---

## Final check (run once, after Task 20)

- [ ] Run `node sessions/staff_lens_audit.js m01`, `node sessions/staff_lens_audit.js m02`, `node sessions/staff_lens_audit.js m03`, `node sessions/staff_lens_audit.js m04` and confirm every one of the 19 target lessons' rows show `render:ok`.
- [ ] Run `grep -rl "A real one:" sessions/m01-shape-of-data sessions/m02-the-neuron sessions/m03-attention sessions/m04-first-model-mlp` and confirm exactly 19 files are listed.
- [ ] Spot-check 2–3 lessons by opening them in a browser (`open sessions/m01-shape-of-data/day-03-broadcasting-dtypes/lesson.html`) and confirming the new 🏭 callout renders correctly in section 1 and the section-4 text reads naturally.
