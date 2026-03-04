# Claude Instructions for this Repo

---

## 1. Who You Are Writing For

Two readers. Keep their material separate. Never mix them.

- **Reader A**: First encounter with the topic. English may not be first language. Needs clear, step-by-step explanations with no assumed knowledge.
- **Reader B**: Preparing for Staff/Principal MLE interviews. Needs precise math, failure modes, design trade-offs, and interview-grade depth.

---

## 2. File Structure

Every [Core] topic has four files. Every [Applied] topic has two. Design modules follow [Section 11](#11-design-interview-modules).

### Layer 1 — The beginner file (`topic-name.md` or `README.md`)

For Reader A. No equations, no failure modes, no complexity analysis, no interview Q&A, no comparisons assuming prior knowledge.

**The test:** could a curious 12-year-old read this file, feel good about it, and want to learn more?

**Structure:**
```
[Curiosity hook — a mystery or surprising fact]
[What you need to know first — prerequisites, max 3 items]
[Analogy — concrete, experienced thing]
[What the analogy gets right]
[The concept in plain words]
[Where the analogy breaks down — one sentence]
[Checkpoint — 2-3 questions to verify understanding]
[Victory lap — connect to something real and impressive]
[Link to the interview deep-dive file]
```

### Layer 2 — The interview deep-dive file (`topic-name-interview.md`)

For Reader B. Assumes Layer 1 intuition is clear. Contains: precise math (every symbol labeled), failure modes and edge cases, complexity analysis (exact formulas), design trade-offs, cross-field connections, 3–5 interview Q&A with four hiring levels.

**Interview Q&A format** — for each question, show all four levels:

**Q: [question text]**

---
**No Hire**
*Interviewee:* [surface-level, misses the point, or reveals a misconception]
*Interviewer:* [what the interviewer observes, what signal this gives]
*Criteria — Met:* none / *Missing:* [list]

**Weak Hire**
*Interviewee:* [correct at high level but shallow — no math, no trade-offs, no failure modes]
*Interviewer:* [what's present vs. missing, why this doesn't clear the staff bar]
*Criteria — Met:* [what's present] / *Missing:* [what's still absent]

**Hire**
*Interviewee:* [solid — correct math, at least one failure mode, a real trade-off]
*Interviewer:* [what makes this a hire, what would push it to Strong Hire]
*Criteria — Met:* [list] / *Missing:* [what would elevate to Strong Hire]

**Strong Hire**
*Interviewee:* [full depth — precise equations, multiple failure modes, connects to production or related work, offers original judgment]
*Interviewer:* [what distinguishes this from Hire, what signals staff-level thinking]
*Criteria — Met:* [full list, nothing missing]
---

**Structure:**
```
[Quick-scan summary box]
[Brief restatement — one paragraph, assumes Layer 1 was read]
[Full mathematical treatment — precise and complete]
[Visual: concept flow or relationship diagram]
[Failure modes, edge cases, hyperparameter sensitivity]
[Complexity analysis — time, memory, parameter count]
[Design trade-offs and alternatives — comparison table]
[Production and scaling considerations]
[Staff/Principal Interview Depth — Q&A]
[Key takeaways box]
```

### Layer 3 — The concept notebook (`01_topic-name.ipynb`)

Hands-on practice. Assumes Layer 1 was read. Markdown cells explain before code executes. Print shapes and intermediate values at every step. Include at least one visualization. No full concept explanations (those live in md files), no equations derived from scratch, no interview Q&A.

**The test:** can the reader run this top to bottom and understand what the code does at every step?

**Structure:**
```
[COACH session start cell]
[Title cell — topic name + one-line description]
[Setup cell — imports only]

For each concept:
  [Markdown cell — what we are about to do and why]
  [Code cell — implementation]
  [Code cell — run it, print shapes and intermediate values]
  [Markdown cell — what we just saw, what to notice]
  [Visualization cell — where relevant]

[Summary markdown cell — what was built, link to interview md]
[COACH session end cell]
```

### Experiments notebook (required for [Core])

Produces runnable evidence for interview claims. Contains: complexity benchmarks, failure mode demos, ablations, library comparison cells. No concept explanations, no new ideas, no exercises without answers.

**The test:** every cell produces output that could be shown to an interviewer as evidence.

**Structure:**
```
[COACH session start cell]
[Title cell — "Experiments: [topic name]" + one-line purpose]
[Setup cell — imports only]

For each experiment:
  [Markdown cell — what claim we are testing and why it matters]
  [Code cell — minimal setup to isolate the variable]
  [Code cell — run experiment, print or plot result]
  [Markdown cell — what output shows, one sentence for interview]

[Summary markdown cell — claims backed by evidence, link to interview md]
[COACH session end cell]
```

### File naming convention

```
attention-mechanisms.md                        ← Layer 1 (beginner)
attention-mechanisms-interview.md              ← Layer 2 (deep dive + interview prep)
01_attention_mechanisms.ipynb                  ← Layer 3 (concept notebook)
01_attention_mechanisms_experiments.ipynb      ← Experiments (required for [Core])
README.md                                      ← Layer 1 module overview (always beginner)
```

Notebooks numbered (`01_`, `02_`, ...) for reading order. `README.md` is always Layer 1 — no equations, no interview Q&A.

---

## 3. Module Coverage

A module is complete when it covers the right width (all required topics) and depth (right level per topic). Topic lists live in each module's `README.md`.

### Notebooks vs md files — no duplication

| | Notebooks (`.ipynb`) | MD files (`.md`) |
|---|---|---|
| **Purpose** | Hands-on — run code, see outputs | Explanation and reference |
| **Contains** | Working code, visualizations | Concepts, diagrams, math, interview Q&A |
| **Math** | Light — equations as context for code | Full — complete derivations |
| **Length** | One concept, one session | As long as needed |

### Depth levels

| Level | Required files | When to use |
|---|---|---|
| **[Core]** | Layer 1 md + Layer 2 interview md + concept notebook + experiments notebook | Commonly asked; requires math |
| **[Applied]** | Layer 1 md + notebook | Important for implementation; rarely deep interview focus |
| **[Awareness]** | Brief section in README only | Useful context; not expected at interview depth |
| **[Design]** | README + interview guide + 2–5 notebooks | System design case studies. See [Section 11](#11-design-interview-modules) |

When in doubt, go deeper.

**Complete [Core] notebook requires:** from-scratch implementation, key behavior visualization, library comparison cell, print statements at every step showing shapes.

**Complete [Core] experiments notebook requires:** at least one complexity benchmark with plot, one failure mode demo, one ablation with measurable output. Every cell produces output.

**Complete [Applied] notebook requires:** library-based implementation, visualization, print statements at each step.

### Coverage map in every README

```markdown
## Coverage Map

### [Subtopic group]

| Topic | Depth | Files |
|-------|-------|-------|
| Topic — one-line description | [Core] | [topic.md](./topic.md) · [topic-interview.md](./topic-interview.md) · [01_topic.ipynb](./01_topic.ipynb) · [01_topic_experiments.ipynb](./01_topic_experiments.ipynb) |
| Topic | [Applied] | [topic.md](./topic.md) · [01_topic.ipynb](./01_topic.ipynb) |
| Topic | [Awareness] | [README.md#anchor](./README.md#anchor) |
```

---

## 4. Session Workflow

### PROGRESS.md — track every module

Every module must have `PROGRESS.md` in its root — the single source of truth for what is done and what is left. Create it before writing any content. Update after completing any file.

**Format:**
```markdown
# [Module name] — Progress

## Status: In Progress

| File | Type | Status |
|------|------|--------|
| README.md | Layer 1 overview | ✅ Done |
| attention-mechanisms.md | Layer 1 | 🔄 In progress |
| 01_attention_mechanisms.ipynb | Concept notebook | ⬜ Not started |
```

**Status values:** ⬜ Not started · 🔄 In progress · ✅ Done

### File writing order

Within a module, always write in this order:

1. `PROGRESS.md` — first
2. `README.md` — module overview
3. For each topic (foundational before dependent):
   - `topic.md` → `topic-interview.md` → `topic.ipynb` → `topic_experiments.ipynb` ([Core] only)

Never write Layer 2 before Layer 1. Never write a notebook before its md files.

### One file per session — no parallelism

Write one file per session. No subagents for parallel writing (causes tone drift, analogy misalignment). Subagents are fine for **read-only tasks** (auditing, searching, validation). Exception: files in completely different modules can be written in parallel.

### Completeness check

Before marking any module as done:

- [ ] `PROGRESS.md` exists and all rows show ✅ Done
- [ ] Every [Core] topic has Layer 1 md, Layer 2 interview md, concept notebook, experiments notebook
- [ ] Every [Applied] topic has Layer 1 md and notebook
- [ ] Every [Awareness] topic has a README section
- [ ] All notebooks execute cleanly end-to-end
- [ ] Each md file follows its Layer structure from Section 2
- [ ] Module README contains up-to-date coverage map

---

## 5. Writing Style

### Language

Write for a 12-year-old encountering the topic for the first time, for whom English is not their first language.

- **Simple words.** "Use" not "utilize". "Show" not "demonstrate". If learned after age 10, find a simpler word.
- **No idioms.** "Under the hood", "out of the box" — say what you mean directly.
- **One idea per sentence.** Short sentences. Active voice. Use "you".
- **No dismissive phrases.** Banned: "As you can see", "Trivially", "Obviously", "Recall that", "It is left as an exercise", "This is just".

### Tone

Write like a brilliant friend who is rooting for the reader. Not a professor. Not a textbook. If it sounds like an exam paper, rewrite it.

### Analogies

Every new concept needs an analogy before any math. Use things a 12-year-old has experienced (physical objects, games, daily routines). Always state where the analogy breaks down.

**Analogy scaffold:** (1) Analogy — concrete, experienced thing, (2) What the analogy gets right, (3) Concept in plain words, (4) Where the analogy breaks down.

- Bad: "The attention mechanism computes a weighted average over value vectors."
- Good: "Imagine you're reading a sentence and trying to understand the word 'bank'. You look back at 'river', 'fish', 'swim' to figure out which meaning fits. Attention does this: for each word, it checks all other words and decides how much each matters. The analogy breaks down because attention looks at all words simultaneously, not one at a time."

---

## 6. Building Concepts Step by Step

### Define every word before you use it

First time a technical word appears, define it immediately. After defining, use freely.

- Bad: "The model uses an embedding to represent each word."
- Good: "The model turns each word into a list of numbers. This list is called an **embedding**."

### Only use what was already taught

Only use ideas from earlier in the same file or from prerequisite files. If you need a concept not yet taught, teach it first or add a one-line reminder.

### Introduce math in three steps

**Step 1 — Words.** Plain language description. **Step 2 — Formula.** Label every symbol. **Step 3 — Worked example.** Small real numbers, every step shown.

Every equation in Layer 2 files or notebooks must have all three steps. Layer 1 files have no equations.

### Breaking down hard equations

Five rules for complex math:

1. **Build up piece by piece.** Never show the full equation first. Start with simplest version, add one term at a time, explain why each term exists.
2. **Connect symbols to the analogy.** Anchor math symbols to the intuition from Layer 1.
3. **Show dimensions at every step.** After every equation, show the shape of each matrix/vector.
4. **Show before-and-after.** When a term fixes a problem, show what happens without it first.
5. **One new idea per equation.** If a formula introduces two concepts, split into two equations.

### Prerequisites block

Start every topic file with a prerequisites block (max 3 items, linked). If none: `**No prior knowledge needed. Start here.**`

### Checkpoints

After each major concept, add 2–3 plain questions. Frame as a tool, not a test. Include: "If you can't answer one, go back and re-read. That is completely normal."

---

## 7. Engagement and Motivation

Four rules for keeping the reader engaged:

1. **Open with a curiosity hook.** Create a "why do I need to know this?" moment before explaining anything. Never open cold with a definition.
2. **Normalize confusion.** When something is hard, say so. Frame it as a sign of seriousness, not inability.
3. **Give victory laps.** After hard concepts, tell the reader what they just unlocked.
4. **Frame progress as leveling up.** Each concept is something the reader gains, not a test they might fail.

---

## 8. Layer 2 Formatting

Layer 2 files must be detailed AND easy to scan.

### Emojis as section markers

| Emoji | Use for |
|-------|---------|
| 🎯 | Key insight or main point |
| 🧮 | Math section |
| ⚠️ | Warning, failure mode, common mistake |
| 💡 | Design insight or trade-off |
| 🔬 | Deep technical detail or proof |
| 📊 | Complexity analysis or benchmarks |
| 🏭 | Production / real-world usage |
| 🗺️ | Concept map or overview diagram |
| ✅ | Correct approach |
| ❌ | Wrong approach |

No decorative emojis. Every emoji signals a content type consistently.

### Required formatting elements

- **Quick-scan summary box:** Every Layer 2 file starts with a `> **What this file covers**` block listing 5–8 items using section marker emojis.
- **Concept flow diagrams:** Use Mermaid in md files, ASCII in notebooks. Diagrams show relationships/flows, not just computation steps. Every major concept needs at least one diagram.
- **Comparison tables:** Whenever alternatives exist, show them in a table. Do not describe alternatives only in prose.
- **Math callout blocks:** Important formulas go in labeled code blocks with the 🧮 emoji. Do not let them float in paragraphs.
- **Key takeaways box:** Every Layer 2 file ends with a numbered list of 5–10 key points. Mark critical ones with 🎯, warnings with ⚠️.

---

## 9. Code and Notebooks

### Code style

- Descriptive variable names (`forget_gate` not `f`)
- One-line comment above non-obvious lines
- Print shapes and intermediate values

### COACH system

COACH tracks XP, streaks, badges, and spaced repetition. State lives in `coach/state.json`. Every notebook must start and end with COACH cells:

**Session start cell** (first code cell):
```python
# ============================================================
# COACH — Session Start  (do not remove this cell)
# ============================================================
import sys, os
sys.path.insert(0, os.path.expanduser('~/Desktop/applied-ai-research'))
from coach.notebook_widgets import render_session_start
_SESSION = render_session_start(
    module_id="<MODULE_ID>",
    notebook_name="<NOTEBOOK_FILENAME>"
)
```

**Session end cell** (last code cell):
```python
# ============================================================
# COACH — Session End  (do not remove this cell)
# ============================================================
from coach.notebook_widgets import render_session_end
render_session_end(_SESSION)
```

**Module ID rules:** ML Design directories use directory name (`"02-visual-search"`). Foundational modules use short name (`"transformers"`, `"rnn"`). Wrong module IDs silently break XP tracking.

### Notebook conventions

- COACH start/end cells required (see above)
- Markdown cells before code — explain first, code second
- Visualizations preferred over tables of numbers

### Notebook validation

After creating or modifying any notebook, run validation:

```bash
python3 scripts/validate_notebook.py <notebook_path> <module_id>
```

**What counts as done:** exit code 0 for every notebook in the module.

**Checks:** invalid JSON, syntax errors, missing COACH cells, wrong module IDs, banned `savefig` calls, stale `../README.md` links. Does NOT check runtime errors or import failures.

**If validation fails:** fix the error, re-run on all notebooks in the module, repeat until all pass.

**Auto-generated notebooks:** `05_interviewer_perspective.ipynb` (ML Design) is generated by `generate_interviewer_notebooks.py`. Skip validation for these — validate only hand-written notebooks (`01_` through `04_`).

---

## 10. Error Handling and Retries

**The rule: try once, maybe twice, then stop.**

- First failure: one retry for transient errors
- Second failure of same kind: stop immediately
- 401 (auth error): never retry

**When stopping due to an error, always:**
1. Tell the user what failed (task, error type, message)
2. List what was completed
3. Write a resume plan with enough detail to restart in a new session

**Resume plan format:**
```
## Session stopped — here is what happened and what is left

### What failed
- Task: [name] — Error: [type and message] — File: [path if relevant]

### What was completed
- [x] Task 1 — done
- [ ] Task 2 — stopped here

### What to do next
1. [First remaining task with detail to start without re-reading]
2. [...]
```

---

## 11. Design Interview Modules

This section covers `ML Design/` and `genAI design/` directories — system design interview case studies. They follow a different content model from foundational modules.

**What applies from Sections 1–10:** Writing style (S5), concept building (S6), engagement (S7), formatting (S8), code style (S9), error handling (S10).

**What does NOT apply:** File structure (S2), depth levels (S3), file writing order (S4) — replaced by this section.

### Module categories

| Category | Directories | Governed by |
|----------|-------------|-------------|
| Foundational | `00-neural-networks/` through `10-reinforcement-learning/` | Sections 1–10 |
| ML Design | `ML Design/01-ml-design-prep/` through `ML Design/11-people-you-may-know/` | Section 11 |
| genAI Design | `genAI design/01-intro-and-framework/` through `genAI design/11-text-to-video/` | Section 11 |

### File structure — ML Design

```
README.md                              ← Beginner overview + full design reference
staff_interview_guide.md               ← Interviewer guide, 4-level calibrated answers
01_[topic]_system_design.ipynb         ← System design walkthrough
02_[component]_deep_dive.ipynb         ← Component deep-dive
03_[component2]_deep_dive.ipynb        ← Another component deep-dive
04_interview_walkthrough.ipynb         ← Mock interview simulation (candidate-facing)
05_interviewer_perspective.ipynb       ← AUTO-GENERATED (do not hand-edit)
```

### File structure — genAI Design

```
README.md                              ← Beginner overview + full design reference
INTERVIEW.md                           ← Candidate-facing interview prep (4-level answers)
staff_interview_guide.md               ← Interviewer guide (4-level calibrated answers)
01_[topic]_part_a.ipynb                ← Concept notebook part A
02_[topic]_part_b.ipynb                ← Concept notebook part B (if needed)
```

### README.md in design modules

Unlike foundational modules where README is strictly Layer 1, design module READMEs serve as **both** beginner introduction **and** comprehensive design reference, following the 7-step framework. This is intentional — design case studies are single end-to-end systems.

### The 7-step design framework

Every design module README must cover all seven steps:

1. **Clarifying Requirements** — What does the system do? Who uses it? Constraints?
2. **Frame as ML Task** — ML problem, inputs/outputs, model type?
3. **Data Preparation** — Data needed, collection, cleaning, labeling?
4. **Model Development** — Architecture, loss function, training strategy?
5. **Evaluation** — Metrics, offline vs online, how to know it works?
6. **Serving Architecture** — How to serve predictions? Latency, throughput, infrastructure?
7. **Monitoring & Infrastructure** — Drift detection, failure detection, regression monitoring?

### 4-level interview calibration (design modules)

| Level | Label | Signal |
|-------|-------|--------|
| ❌ | **No Hire** (Mid→Staff attempt) | Missing fundamentals — cannot structure a design |
| ⚠️ | **Weak Hire** (Strong Senior at Staff bar) | Correct but naive — no production awareness |
| ✅ | **Hire** (Staff Engineer) | Production-ready — handles trade-offs, failure modes, scale |
| 🌟 | **Strong Hire** (Principal Engineer) | Scope expansion, platform thinking, business impact |

### Auto-generated notebooks

`05_interviewer_perspective.ipynb` is generated by `ML Design/generate_interviewer_notebooks.py`. Never hand-edit. To update, modify the generator and re-run:
```bash
python3 "ML Design/generate_interviewer_notebooks.py"
```

### Source PDFs

PDFs at the root of `ML Design/` and `genAI design/` are reference material — not learner-facing. Do not commit copyrighted PDFs to git.

### PROGRESS.md, writing order, completeness

**PROGRESS.md:** Same format as foundational modules (Section 4). Required for every design case study.

**Writing order** (sequential, never parallel):
1. `PROGRESS.md` → 2. `README.md` → 3. `staff_interview_guide.md` → 4. `INTERVIEW.md` (genAI only) → 5. Notebooks (`01_`–`03_`) → 6. `04_interview_walkthrough.ipynb` → 7. Re-run generator for `05_` (ML Design only)

**Completeness check:**
- [ ] `PROGRESS.md` exists and all rows ✅ Done
- [ ] `README.md` covers all 7 design framework steps
- [ ] `staff_interview_guide.md` exists with 4-level calibrated answers
- [ ] `INTERVIEW.md` exists (genAI Design only) with 4-level answers
- [ ] Hand-written notebooks (`01_`–`04_`) pass validation
- [ ] `05_interviewer_perspective.ipynb` up-to-date (ML Design only)
- [ ] All notebooks execute cleanly end-to-end

**No experiments notebooks** — design modules focus on interview communication and system design reasoning.

### Coverage map for design modules

Parent-level READMEs for `ML Design/` and `genAI design/` use this format:
```markdown
## Coverage Map

| Case Study | Files |
|------------|-------|
| Visual Search | [README](./02-visual-search/README.md) · [Interview Guide](./02-visual-search/staff_interview_guide.md) · [01](./02-visual-search/01_visual_search_system_design.ipynb) · ... |
```

**Depth level:** Design modules use **[Design]** — the [Core]/[Applied]/[Awareness] taxonomy does not apply.