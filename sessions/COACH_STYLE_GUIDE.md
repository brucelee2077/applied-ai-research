# Coach Layer Style Guide

_The house style for warm, rigorous, bilingual Frontier Lab lessons in `sessions/`._

This guide governs how we upgrade an existing `lesson.html` into a **Coach Layer** lesson. It is
a refactor guide, not a from-scratch template — the lesson structure (7 sections, quiz, playground,
scroll-reveal build, produce artifact) already works. The Coach Layer makes it **warmer, more
intuitive, bilingual, math-friendly, and artifact-driven** without deleting depth or breaking
mechanics.

> **Golden rule:** the Coach Layer is **additive**. You insert small, consistent blocks into the
> existing `.sec-body` of each section. You never remove a hard concept, never turn rigor into
> motivational fluff, and never touch the interactive JavaScript (`DEMOS`, `BUILD`, `QS`), the
> navigation links, or the localStorage progress machinery.

---

## 1. Who we are writing for

Two readers share every lesson. Keep their material layered, never blended into mush.

- **Reader A — first encounter.** May be an ML beginner; English may be a second language. Needs an
  everyday analogy and plain words *before* any symbol. This reader is served by the hook, the
  intuition, the analogy, and the Chinese scaffold.
- **Reader B — Staff/Principal MLE interview prep.** Wants exact math, failure modes, trade-offs,
  and an answer they could say out loud in an interview. This reader is served by the Math Ladder,
  the Staff Lens, and the Interview-ready block.

A good Coach Layer lesson lets Reader A climb up to Reader B's material one rung at a time. Nothing
is dumbed down; the hard parts are simply reached by a warmer path.

---

## 2. Voice

Write like a brilliant friend who is rooting for the reader — not a professor, not a textbook.

- Short sentences. One idea each. Active voice. Use "you".
- Define every technical word the first time it appears, then use it freely.
- Banned phrases: "As you can see", "Obviously", "Trivially", "Recall that", "This is just",
  "It is left as an exercise". If a sentence would sound at home on an exam paper, rewrite it.
- Name the hard part out loud. Confusion is a sign of seriousness, not failure.
- After a hard idea, give a **victory lap**: tell the reader what they just unlocked.

---

## 3. The bilingual rule (light touch)

Chinese is a **learning scaffold**, not the medium. The learner reads mostly in English and prefers
it that way. Use Chinese **only** where it lowers the emotional and intuitive barrier:

- ✅ Use Chinese for: the one-line intuition, emotional orientation, and "why this is confusing".
- ❌ Do **not** translate: technical terms, formulas, variable names, code, or the interview-ready
  phrasing. Those stay in English because that is the language of the field and the interview.

**Pattern** (Chinese clause first, English technical point second, in the same block):

```html
<p><b>直觉 / Intuition:</b> KV cache 就像你边读书边做笔记，读过的不用再读一遍。
The technical point is that we cache each past token's Key and Value once, so every later decode
step reads them instead of recomputing them — turning <code>O(n²)</code> work into <code>O(n)</code>.</p>
```

Aim for **2–4 short Chinese touches per lesson**, concentrated in Section 1 (pain point) and
Section 2 (intuition). More than that starts to overwhelm the English-first reader; fewer than that
and the bilingual scaffold isn't really there. See the memory note `prefers-english-learning-content`.

---

## 4. The 15 Coach Layer elements → where they live

Every upgraded lesson should contain all 15. Most already exist in a strong lesson; the Coach Layer
adds the ones that are missing and *labels* the ones that were implicit. Map:

| # | Coach Layer element | Section | How it appears |
|---|---------------------|---------|----------------|
| 1 | Warm opening hook | Hero `.lede` | A curiosity line, not a cold definition |
| 2 | Pain point ("why confusing") | §1 What is it | **Pain-point block** (bilingual, `c-warn`) |
| 3 | Intuition-first explanation | §2 Intuition | Plain-words paragraph before any math |
| 4 | Everyday analogy | §2 Intuition | `.relate` cards (party, note cards, tire pressure…) |
| 5 | Software/systems analogy | §2 Intuition | One card or line tying it to code/systems |
| 6 | Where the analogy breaks | §2 Intuition | `c-warn` "Where it breaks" callout |
| 7 | Math Ladder | §4 How & why | **Math-Ladder block** (`c-info`, 4 rungs) |
| 8 | Tiny-number example | §4 / §3 Playground | Rung 3 of the ladder + the fake terminal |
| 9 | Common misconception | §4 / §7 | Named in a `c-warn` "Gotcha" |
| 10 | Why frontier labs care | §4 How & why | Industry anchor (real model / paper / number) |
| 11 | Staff / Research Engineer Lens | §4 How & why | **Staff-lens block** (silent failure + trade-off) |
| 12 | Interview-ready explanation | §4 How & why | **Interview block** (`c-ok`, spoken answer) |
| 13 | Produce artifact | §7 Produce | The `experiment.py` task (already present) |
| 14 | Acceptance criteria | §7 Produce | Explicit `<h4>Acceptance criteria</h4>` + list |
| 15 | 5-minute research log | §7 Produce | **Research-log block** (`c-info`, bilingual) |

If an element already exists and is good, **leave it** — do not duplicate it. The job is a complete
set, not a longer file.

---

## 5. The reusable blocks (copy these)

All blocks use only classes that already exist in **both** shells (`callout`, `c-info`, `c-warn`,
`c-ok`, `code`, `<b>`, `<h4>`). They render correctly whether the lesson is on the old top-nav shell
or the new sidebar shell. Insert each block **inside `.sec-body`, just before that section's
`.gotit` button**.

### 5.1 Pain-point block (Section 1)

```html
<div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b>
<Chinese one-line reason it feels hard>。In plain terms: <the specific confusion in English — e.g.
"people memorize the formula but can't say where the number comes from, so they freeze when the setup changes.">.</div></div>
```

### 5.2 Math Ladder block (Section 4) — the math-friendly core

Four rungs, always in this order. Never show the full formula first; climb to it.

```html
<div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — &lt;formula name&gt;</b>
<br><b>1 · In words:</b> &lt;the idea in one plain sentence, no symbols&gt;
<br><b>2 · The formula:</b> <code>&lt;formula&gt;</code> — <code>&lt;sym&gt;</code> = &lt;what it means&gt;; <code>&lt;sym&gt;</code> = &lt;what it means&gt; (every symbol labeled)
<br><b>3 · Tiny numbers:</b> &lt;plug in small, real values and show the arithmetic step by step&gt;
<br><b>4 · Sanity check:</b> &lt;one quick invariant or "does this make sense?" test&gt;</div></div>
```

Rule 8-of-CLAUDE.md math: this block *is* the "words → formula (labeled) → worked example" ladder,
plus a sanity check. Anchor the symbols back to the Section-2 analogy where you can.

### 5.3 Interview-ready block (Section 4)

A crisp answer the reader could **say out loud** in 20–30 seconds. Lead with the headline, then the
one caveat that signals depth.

```html
<div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b>
"&lt;headline claim in one sentence&gt;. &lt;the one mechanism or number that proves you understand it&gt;.
The nuance is &lt;the caveat / failure mode that separates a Hire from a Weak Hire&gt;."</div></div>
```

### 5.4 Research-log block (Section 7)

Turns the existing `log.md` nudge into a concrete 5-minute habit. Bilingual prompt, English lines.

```html
<div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b>
在关掉页面前，用自己的话写三行 — before you close the tab, write three lines in
<code>&lt;path&gt;/log.md</code>: (1) the one sentence you'd tell a teammate; (2) the number or result
that surprised you; (3) one thing you're still unsure about.</div></div>
```

Keep whatever `log.md` question the lesson already had — fold it into line (1) or leave it beside
this block.

---

## 6. Shell awareness (do not break either)

Two shells are live in `sessions/`. Know which one you are editing and match its markup exactly.

| | **Old shell** | **New shell** |
|---|---|---|
| Section wrapper | `<section class="sec" id="sN" data-sec="KEY">` | `<section class="module-section" id="sN" data-sec="KEY">` |
| Section head | `.sec-badge s-XXX` | `.sec-num s-XXX` |
| Hero lead | `<p class="lead">` | `<p class="lede">` |
| Hero title | `<h1>Main<br>Sub</h1>` | `<h1>Main<span class="sub">Sub</span></h1>` |
| Kicker | `<span class="eyebrow">` | `<span class="kicker">` |
| Nav | top `.lesson-nav` (prev/hub/next) | `#sidebar` `.nav-link` + `.side-nav` |
| Theme switch | none | Light/Dim/Dark/Midnight |
| Finale | `.fin` (block) | `.fin` (block) |

**Shell-agnostic (identical in both, safe to rely on):** `data-sec` keys
(`what,intuition,play,why,build,quiz,produce`), `.sec-body`, `.gotit` buttons, `.callout`
(`c-info/c-warn/c-ok`), `.relate` cards, `.term[data-tip]` tooltips, the `var DEMOS/BUILD/QS`
literals, and the localStorage progress script.

**Migration compatibility:** `_shell_migrate.py` migrates the old shell → new shell and re-extracts
each `.sec-body`, the hero, and the `DEMOS/BUILD/QS` literals **byte-for-byte** from the live
`lesson.html`. So Coach Layer content added inside `.sec-body` (and the hero `.lead`/`.lede`)
survives a later shell migration automatically. To stay compatible, **do not** alter a section's
wrapper tag, its `id`, its `data-sec`, or the `<div class="sec-body"> … </div></section>` boundary.

---

## 7. Hard preservation contract (never violate)

When editing an existing lesson you MUST preserve, unchanged:

- File paths and file names (`lesson.html` stays `lesson.html`).
- Every navigation `href` (prev / next / hub, top and bottom).
- `data-quest-id` on `<body>` (it is the localStorage key `frontier-lesson:<id>` — changing it
  wipes the learner's progress).
- The 7 sections, their `id`s, and their `data-sec` keys.
- The 7 `.gotit` buttons and the progress / reset / checklist script.
- Quiz mechanics: the `var QS = [...]` literal shape (`{q, opts, ans, fb}`), 4 questions.
- Playground mechanics: the `var DEMOS = {...}` literal, 3 `data-demo` buttons.
- Scroll-reveal build: the `var BUILD = [...]` literal.
- Glossary tooltip script and every `.term[data-tip]`.
- Self-contained / offline behavior: no new external assets, no new CDN links, no new fonts.
  Chinese renders in the existing system font stack.

You MAY:

- Add Coach Layer blocks inside `.sec-body` (before the `.gotit`).
- Enrich the hero `.lead`/`.lede` and the `.goal` line for warmth (keep them extractable).
- Add or rewrite `.term[data-tip]` glossary hovers for newly introduced words.
- Fix **factual/label bugs inside the pilot file itself** (e.g. a finale banner that names the wrong
  topic after a shell migration) — this is coherence, not scope creep.
- Add `<h4>Acceptance criteria</h4>` to a Produce section that lacks one.

You MUST NOT:

- Delete a failure mode, a trade-off, an equation, or a complexity claim.
- Replace precise language with cheerleading.
- Add a Coach block that duplicates content already present.
- Edit any lesson outside the agreed pilot/rollout batch.

---

## 8. Marker emoji convention (consistent, not decorative)

Every emoji signals a content type. Reuse these; do not invent new ones per lesson.

| Emoji | Means | Callout class |
|-------|-------|---------------|
| 🧭 | What this assumes / prerequisites | `c-info` |
| 😕 | Pain point — why this is confusing | `c-warn` |
| 🪜 | Math Ladder | `c-info` |
| 🧮 | A key formula | `c-ok` / `c-info` |
| ⚠️ | Failure mode / where the analogy breaks / gotcha | `c-warn` |
| ⚖️ | Trade-off | `c-info` |
| 🏭 | Real-world / frontier-lab usage | `c-info` |
| 🔗 | Causal chain / how it connects | `c-info` |
| 🎤 | Interview-ready answer | `c-ok` |
| 📓 | 5-minute research log | `c-info` |
| 📚 | Optional official docs | `c-info` |
| 📝 | Write something in `log.md` | `c-info` |

---

## 9. Definition of done (per lesson)

A Coach Layer upgrade is complete only when:

- [ ] The lesson still opens and renders (verified headless via jsdom + `node --check` on its JS).
- [ ] `lesson_audit.py` reports it **OK** with no hard failures (Chinese is not a defect).
- [ ] All 15 Coach Layer elements are present (use `enhance_lesson_checklist.md`).
- [ ] Navigation, `data-quest-id`, quiz, playground, build, and tooltips are byte-identical in
      behavior.
- [ ] The bilingual scaffold is present but light (2–4 Chinese touches, intuition/pain only).
- [ ] Technical depth is equal to or greater than before — nothing was removed.
- [ ] The learner can now *explain the concept more clearly* and is asked for a concrete artifact.

See `enhance_lesson_checklist.md` for the per-lesson checklist and `coach_layer_pilot_report.md` for
the pilot results and rollout order.
