---
name: frontier-session-coach
description: Turn one day of the 24-week Frontier Lab curriculum into a single self-contained interactive HTML lesson that TEACHES the topic from zero — click-through sections, an interactive demo, a quiz, and progress tracking — all in one offline file. Use when the user asks to start today's study, plan a session, or generate the next curriculum day.
---

# Frontier Session Coach

You turn one day of the user's 24-week curriculum into a **single self-contained HTML lesson
page** that the user opens in a browser and **clicks through to actually learn the topic**.

The page is the lesson, not a plan. It TEACHES. The user has zero knowledge of each day's
topic when they open it — your job is to take them from zero to "I get it, and I can write it."

> ⛳ **Read `TEACHING_PRINCIPLES.md` (same folder) before writing — especially the Ground
> Zero Rule.** The most common failure is assuming the reader already knows the topic. They
> don't. Always open with "what is this, why does it exist, how does it relate to what I
> already know," and define every topic-specific term/tool/syntax on first use. Assume normal
> programming fluency (Python/NumPy), assume **zero** knowledge of the day's subject.

This skill is for self-study only. Do not include employer, day-job, or internal work
context unless the user explicitly asks.

## Inputs you read first

1. **Curriculum:** `frontier_ai_24_week_link_companion.md` (repo root) — source of truth for
   each day's reading links and practical execution. Never invent links.
2. **Ledger:** `sessions/progress.json` — tracks completed days and the `cursor` (next day).
   Create it if missing (keys: `curriculum`, `cursor`, `completed` [], `generated` []).
3. **Reference page:** `sessions/week-01/day-01-jax-immutability.html` — the canonical format.
   Copy its structure, CSS, and JS engine. Study it before generating a new day.

## Which day to generate

Default: **auto — next uncompleted day**.

1. Read `sessions/progress.json`, take `cursor` (e.g. `w03-d03`).
2. If the user named a day ("week 3 wednesday", "KV cache day"), use that.
3. Find that day in the curriculum companion; use its real links and practical task.

### Day-id scheme (consistent across both skills)

Weekday slots: `Monday=d01 · Tuesday=d02 · Wednesday=d03 · Thursday=d04 · Friday=d05`.
Multi-day blocks (e.g. "Mon–Fri") collapse to one day `d01`.

**Weekend-lab slot `d06`:** when the blueprint gives Saturday and Sunday their own distinct
topics (not folded into an existing multi-day block), those two days collapse into ONE extra
lesson, `d06`, covering both combined. Skip `d06` entirely for weeks whose blueprint already
treats Sat+Sun as a block of its own (e.g. a week structured as "Mon–Fri" + "Sat–Sun" — that
Sat–Sun block is already `d02`, no separate `d06` needed). A `d06` lesson still uses the same
HTML engine and still requires Section 1 (What is it) and the quiz, but content should lean
into the practical/capstone nature of weekend blueprint work rather than inventing a fresh
demo-heavy teaching arc — the weekday lessons already carried the concept teaching.

**Two-week blocks:** a few blueprint entries span two calendar weeks as a single unit (e.g.
"Week 17-18 — Two Weeks: Consolidation & Publishing", "Week 21-22 — Two Weeks: Formal
Verification & Refinement"). These produce exactly ONE lesson page, filed under the first of
the two week numbers (`w17-d01`, `w21-d01`). The second week number has no lesson of its own —
`sessions/index.html` should show it pointing at the same page with a "(cont. — see Week N)"
label rather than inventing distinct days for it.

Ordering: `w01-d01 → … → w01-d06 → w02-d01 → … → w24-d04` (cursor rolls over weeks; the exact
day count per week varies — 6 for a full Mon-Sun week with a weekend-lab, 5 for Mon-Fri only,
2 for a week that's just "Mon-Fri block + Sat-Sun block", 1 for a two-week block).

## Output — ONE self-contained file

Create exactly one file:

```text
sessions/week-<NN>/day-<NN>-<slug>.html
```

It must be **fully self-contained**: inline `<style>` and inline `<script>`, **no external
CSS/JS**, no `_assets/` links. The only external references allowed are the optional Google
Fonts `<link>`s (the page degrades to system fonts if they're blocked). The user must be able
to double-click the file and have everything work offline. Copy the inline CSS and the inline
JS engine verbatim from the reference page — only the lesson content changes per day.

### Two slugs, do not confuse them

- **Filename slug** — descriptive kebab-case: `day-01-jax-immutability.html`.
- **`data-quest-id`** on `<body>` — `w<NN>-d<NN>-<short>` (e.g. `w01-d01-jax`). Keys this
  day's localStorage progress; only needs to be unique and stable per day.

## Page structure — a click-through LESSON

The page is a vertical stack of teaching **sections** the learner completes one at a time. A
top progress bar fills as each section is finished; finishing all triggers a celebration.

Engine hooks the inline JS depends on (copy from reference page, do not rename):
- `<body data-quest-id="...">`
- top nav with `#bar` (role=progressbar), `#count`, `#reset`
- each section: `<section class="sec" id="sN" data-sec="<unique-name>">` with a `.sec-head`
  (badge + title + `.sec-tick`) and `.sec-body` ending in a `.gotit` button
- a final `.fin` celebration block
- the JS tracks progress by counting `.sec` elements and their `data-sec`, persists to
  localStorage, wires `.gotit` buttons, the playground, the **scroll-reveal build-up**, the quiz, and copy buttons

### Required sections, in this order

1. **是什么 What is it** — *(Ground Zero Rule lives here.)* What the topic is, why it exists,
   how it relates to something the reader already knows. State prerequisites honestly
   ("assumes basic Python/NumPy; assumes zero <topic>"). Introduce the key tool/term/library.
   End by teasing today's puzzle. **Never skip or shrink this section.**
2. **直觉 Intuition** — one vivid analogy before any code. State where the analogy breaks.
3. **动手玩 Playground** — an interactive demo where clicking teaches the mechanism (e.g. a
   simulated terminal with 2–3 buttons, or a small visual). Gate the section's `.gotit` until
   the learner has interacted. This is the centerpiece — make it genuinely illuminating.
4. **机制/为什么 Mechanism & why** — the precise rule + the reason it matters for frontier
   labs. Define every term (compiler, XLA, etc.) on first use. End with a one-line causal chain.
   Include a named failure-mode callout and a named trade-off callout — see Staff Lens in
   `TEACHING_PRINCIPLES.md`. The frontier-relevance passage must name a real system — see
   Industry Anchor in `TEACHING_PRINCIPLES.md`.
5. **Build it up (scroll-reveal visual)** — `data-sec="build"`. NOT a click-through code
   stepper (the user dislikes clicking "Next" through 20 text steps — see the
   `lesson-walkthrough-scroll-reveal` memory). Instead: a data-driven `var BUILD=[{viz, note}, …]`
   array of 5–8 steps that the fixed engine renders as scroll-reveal cards — each `viz` is an
   inline SVG/HTML **diagram** (use the `.dgram/.node/.arrow` primitives, raw `<svg>`, or a
   `.build-embed` iframe to a `sessions/viz/<name>.html` interactive) that ASSEMBLES the
   mechanism one piece at a time; each `note` is the short explanation beside it. Pieces fade in
   via IntersectionObserver (no "Next" button); reveal-all under `prefers-reduced-motion`. Gate
   the `.gotit` until all steps have scrolled into view. Prefer real diagrams over walls of code.
   Where a matching `sessions/viz/` visualization exists, embed it as the first build step.
6. **自测 Quiz** — 4 clickable multiple-choice questions with instant feedback + explanation.
   Gate `.gotit` until all answered. At least one question must be diagnostic-style, not
   recall — see Staff Lens in `TEACHING_PRINCIPLES.md`.
7. **产出 Produce** — the day's artifact (file + path + acceptance criteria). Offer two paths:
   (A) do it yourself, (B) a copy-paste prompt that triggers the right skill to build it:
   - code/experiment → `frontier-experiment-lab`
   - visualization (KV cache, MoE, scaling curves, attention) → `frontier-d3-visual-lab`
   - paper/blog day → `frontier-paper-course`
   - concept needing a full mini-course → `frontier-concept-courseware`
   Plus a one-line reflection to `sessions/week-<NN>/day-<NN>-log.md`.

Adapt the middle sections to the topic (a math day may merge playground into the worked
example; a paper day may add a "reading map" section). But section 1 (What is it) and the
quiz are mandatory, and the lesson must teach the topic in full on the page itself — never
"go read the docs" as the only teaching.

## After generating

1. Append the new day to `generated` in `sessions/progress.json`:
   ```json
   { "id":"w03-d03", "week":3, "day":"Wednesday",
     "title":"<short English title>", "page":"sessions/week-03/day-03-<slug>.html",
     "status":"generated" }
   ```
   Do NOT mark completed or advance `cursor` — that's `frontier-review-quiz`'s job.
2. **Update `sessions/index.html`** — find the matching `[id, title, null, null]` row in the `WEEKS` JS object and fill in the page path and questId:
   ```js
   // before:  ['w03-d03','The KV Cache', null, null],
   // after:   ['w03-d03','The KV Cache', 'week-03/day-03-kv-cache.html', 'w03-d03-kv'],
   ```
3. Tell the user the path and that they can open it (`open sessions/...html`), plus `open sessions/index.html` for the full map.
4. Give a 2–3 line summary. Don't paste the lesson as markdown — the page IS the lesson.

## Language

Write in **English**. The user prefers English and reads it comfortably. Use Chinese only
where it genuinely clarifies a specific nuance — rare in practice. Keep paragraphs short —
the user is overwhelmed by long blocks. Bite-sized sections + a filling progress bar is the
intended feel.

## Done definition

A day "counts" when the learner produces the artifact and writes the reflection. The page
teaches and checks understanding; the artifact + log are the proof. Grading and ledger
completion happen in `/frontier-review-quiz`.
