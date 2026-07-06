# Enhance-Lesson Checklist

_A copy-per-lesson checklist for applying the Coach Layer. Work top to bottom. Pair with `COACH_STYLE_GUIDE.md` (blocks + rules) and `lesson_audit.py` (automated gate)._

Lesson: `__________________________/lesson.html`  ·  quest-id: `__________`  ·  shell: `old / new`

---

## Step 0 — Before you touch anything

- [ ] Read the whole `lesson.html` once. Note which shell it is (see guide §6).
- [ ] Record the four things you must not break: the `data-quest-id`, the prev/next/hub `href`s, the 3 `data-demo` keys, and the 4 quiz entries. You will re-check these at the end.
- [ ] Confirm you are editing the canonical `lesson.html`, not a `*.new.html` / `*-redesign-*.html` staging file from another workstream.
- [ ] Identify the **core formula** of the lesson — that is what the Math Ladder will teach.

---

## Step 1 — Warmth & bilingual (Reader A)

- [ ] **Hook.** Hero `.lede`/`.lead` opens with curiosity, not a cold definition. Warm it if flat.
- [ ] **Pain point (bilingual).** Add the 😕 pain-point block to §1 (guide §5.1). One Chinese clause ("为什么这里容易卡住") + the specific English confusion. Before the §1 `.gotit`.
- [ ] **Intuition line (bilingual).** In §2, add one `直觉 / Intuition:` line (Chinese first, English technical point second) if the section has no Chinese scaffold yet.
- [ ] **Everyday analogy** present in §2 `.relate` cards? (party / note cards / tire pressure / …)
- [ ] **Software or systems analogy** present (one card or line)? Add if missing.
- [ ] **Where the analogy breaks** — a `⚠️ c-warn` callout in §2? Keep or add.
- [ ] Total Chinese touches across the lesson = **2–4** (not more). Intuition/pain only.

---

## Step 2 — Math-friendliness (Reader B)

- [ ] **Math Ladder** added to §4 for the core formula (guide §5.2): 4 rungs — (1) in words, (2) formula with **every symbol labeled**, (3) tiny-number worked example, (4) sanity check. Never show the full formula first.
- [ ] Symbols in the ladder are anchored back to the §2 analogy where possible.
- [ ] **Tiny-number example** appears (ladder rung 3 and/or the §3 playground).
- [ ] **Common misconception / gotcha** is named in a `⚠️` callout (§4 or §7).

---

## Step 3 — Staff depth & interview (Reader B)

- [ ] **Why frontier labs care** — a real model / paper / number is named (§4). Keep if present.
- [ ] **Staff / Research Engineer Lens** — one *silent* failure mode (⚠️) + one *trade-off* (⚖️), each tied to a real activity (code review / design review). Keep if present; add if missing.
- [ ] **Interview-ready block** (🎤, guide §5.3) added to §4: a 20–30s spoken answer, headline + one proof + one nuance.

---

## Step 4 — Artifact (both readers)

- [ ] **Produce artifact** — a concrete `experiment.py` task with an Option A (write it) and Option B (frontier-experiment-lab prompt). Keep the exact file path.
- [ ] **Acceptance criteria** — an explicit `<h4>Acceptance criteria</h4>` + checklist. Add if the section only had prose.
- [ ] **Explain-back** — the artifact's `log.md` or `README.md` includes a 5–7 sentence explanation covering: (1) what was computed or built, (2) why the result scales that way, (3) what would break in production or research if this were ignored.
- [ ] **5-minute research log** (📓, guide §5.4) added to §7, bilingual prompt → 3 lines in `log.md`. Fold in any existing `log.md` question.

---

## Step 5 — Coherence fixes (pilot file only)

- [ ] Kicker / eyebrow names the correct module/week/day (fix leftover migration text).
- [ ] Sidebar `nav-group-label` (new shell) names the correct module.
- [ ] Finale `.fin` `<h3>` + `<p>` describe **this** lesson and point to the real next lesson.
- [ ] `<title>` and `.brand-sub` are the authoritative source for the above.

---

## Step 6 — Preservation re-check (never skip)

- [ ] `data-quest-id` unchanged.
- [ ] All prev/next/hub `href`s unchanged.
- [ ] 7 sections, 7 `data-sec` keys, 7 `.gotit` buttons — counts unchanged.
- [ ] `var DEMOS` still 3 keys matching the 3 `data-demo` buttons.
- [ ] `var QS` still 4 entries `{q, opts, ans, fb}`.
- [ ] `var BUILD` untouched.
- [ ] Glossary tooltip script + every `.term[data-tip]` intact; new terms have a `data-tip`.
- [ ] No new external assets / CDN / fonts. Still self-contained and offline.
- [ ] The section wrapper tags, `id`s, and `<div class="sec-body">…</div></section>` boundaries are intact (so `_shell_migrate.py` can still extract cleanly).

---

## Step 7 — Verify

- [ ] `node --check` passes on the lesson's extracted `<script>` blocks.
- [ ] jsdom loads the file with no thrown error and finds 7 sections / 4 quiz / 3 demos.
- [ ] `python3 sessions/lesson_audit.py <path>` → **OK** (COACH advisories are fine; hard failures are not).
- [ ] Eyeball in a browser at Light and Dim themes (new shell) or just open (old shell).

Done when every box is checked and `lesson_audit.py` shows no hard failure for this file.
