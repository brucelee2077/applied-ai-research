# Daily Lesson Template

The daily output is **one self-contained interactive HTML lesson** that teaches the topic
from zero. Not a markdown plan, not a checklist that links external files.

Do not use a markdown template. Copy the canonical reference page:

```
sessions/week-01/day-01-jax-immutability.html
```

Everything is inline (CSS + JS). No `_assets/`, no external scripts. Double-click → it works
offline. Only the lesson content changes per day; keep the inline CSS and JS engine verbatim.

**Two navs to wire per day.** The page has a `.lesson-nav` block at the top (first child of
`<main>`, class `lesson-nav top`) and an identical one at the bottom. Both hold the same
prev / hub / next links — update **both** when you set the day's prev/next. The hub link is a
labeled button (`▦` + `Map`, `href="../index.html"`), not a bare glyph. First day: prev is a
disabled `<span>`; last day: next points back to `../index.html`.

## Sections (the teaching arc)
1. **是什么 What is it** — Ground Zero Rule: what/why/how-it-relates + prerequisites. Mandatory.
   Ends with a short named-system teaser callout — see Industry Anchor in
   `TEACHING_PRINCIPLES.md`.
2. **直觉 Intuition** — one analogy before code; say where it breaks.
3. **动手玩 Playground** — interactive demo; learn by clicking. Gate until interacted.
4. **机制/为什么 Mechanism & why** — the rule + frontier relevance + causal chain. Include a
   named failure-mode callout and a named trade-off callout — see Staff Lens in
   `TEACHING_PRINCIPLES.md`. The frontier-relevance passage must name a real system — see
   Industry Anchor in `TEACHING_PRINCIPLES.md`.
5. **逐行看 Worked example** — click-through of the real code.
6. **自测 Quiz** — 4 clickable MCQs with instant feedback. Mandatory. Gate until answered.
   At least one question must be diagnostic-style, not recall — see Staff Lens in
   `TEACHING_PRINCIPLES.md`.
7. **产出 Produce** — the artifact (path + acceptance) + copy-prompt to the right skill.

See `SKILL.md` for the full contract and `TEACHING_PRINCIPLES.md` for the Ground Zero Rule.

## Per-day "done" checklist (for the learner)
- [ ] Understood every section (progress bar full)
- [ ] Produced the artifact
- [ ] Wrote the one-line reflection log
- [ ] Ran `/frontier-review-quiz` to grade + advance `sessions/progress.json`
