# Concept-Driven Lesson Shell (V9) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "concept-driven" lesson mode (V9) to the source-first compiler so a lesson body is N per-concept units (each `intro → its own inline visual → build-up`) instead of a fixed 7-section template, and rebuild Module 2 Day 2 (Activation Functions) as the first V9 lesson.

**Architecture:** V9 is an additive `mode: concept` in the existing `sessions/_compiler` toolchain. A new reusable donor (`v9-base.donor`) carries a generalized JS engine (the shipped engine is already ~90% generic — only 3 id-coupled widget engines change). `v8lib.py` gains `@@@ concept/quiz/produce` block rendering + `%%% svg/demo/quiz` widgets + a `mode=='concept'` compile path. `compile_lesson.py` branches its gate dispatch on `meta['mode']`. A new `concept_shell_gate.py` asserts V9 invariants; `reader_flow_gate.py` gets a concept-mode branch (skips the s1/s2/s4/s7 strict checks; adds a per-concept "must contain a visual" check). Shipped V8 lessons keep their donors + code path untouched.

**Tech Stack:** Python 3.11 (+ PyYAML), regex-based compiler, HTML/CSS/vanilla-JS lesson shell, pytest for gate/compiler unit tests, `node --check` for emitted-JS syntax (jsdom NOT installed — jsdom smoke is best-effort only).

**Spec:** `docs/superpowers/specs/2026-07-11-concept-driven-lesson-flow-design.md`

**Key reference anchors (read before starting):**
- `sessions/_compiler/v8lib.py` — `parse_blocks` (:42), `render_md` (:191), `render_widget` (:173), `render_viz` (:164), `render_section` (:260), `compile_html` (:316), `REGION_PATTERNS` (:299), `CONTENT_SECTIONS` (:295).
- `sessions/_compiler/compile_lesson.py` — gate calls at :56 (reader_flow) and :67 (shell_invariant), write at :71.
- `sessions/_compiler/gates/reader_flow_gate.py` — `_region_texts` mode detection (:34 `'hero' in regions`), `run()` (:51), strict checks at ~:78/:83/:99.
- `sessions/_compiler/gates/shell_invariant_gate.py` — hardcoded `==7`, `s1..s7`, DEMOS/BUILD/QS.
- `sessions/_compiler/shells/m02-day-02.donor` — CSS + shipped JS engine: progress/nav/scrollspy generic (:502-584), playground engine (:611-618), build-reveal (:635-664), quiz engine (:681-700), glossary script (:718-756), viz auto-resize (:766-785). Reusable SVGs live in `var BUILD` (:621-634); tanh SVG inline at donor :397.
- Tests live anywhere; run with explicit path (e.g. `pytest sessions/_compiler/tests/…`). `pytest.ini` puts repo root on `sys.path`.

**Global conventions for every task:**
- Work on branch `build/capability-spiral` (already current). Commit after each task.
- Determinism is sacred: same `(source.md, donor)` → byte-identical output. Never introduce time/random into the compiler.
- Never edit a shipped `lesson.html` by hand. Only edit `source.md` + compiler + donor.
- After any change touching the compiler, re-run the Day-2 compile and the V8 regression check (Task 12) before committing.

---

## File Structure

**Create:**
- `sessions/_compiler/shells/v9-base.donor` — reusable V9 shell (head/CSS reused from m02 donor + generalized JS engine + content placeholder).
- `sessions/_compiler/gates/concept_shell_gate.py` — V9 structural gate (parallel to `shell_invariant_gate.py`).
- `sessions/_compiler/tests/test_v9_widgets.py` — unit tests for the new `v8lib` widgets/blocks.
- `sessions/_compiler/tests/test_v9_compile.py` — end-to-end concept-mode compile + idempotency.
- `sessions/_compiler/tests/test_concept_gates.py` — concept_shell_gate + reader_flow concept branch (incl. broken-variant FAIL).
- `sessions/_compiler/tests/fixtures/mini_concept.md` — tiny 3-concept source used by compile/gate tests.

**Modify:**
- `sessions/_compiler/v8lib.py` — new block/widget renderers + `mode=='concept'` path in `compile_html`.
- `sessions/_compiler/gates/reader_flow_gate.py` — concept-mode branch.
- `sessions/_compiler/compile_lesson.py` — gate dispatch on `meta['mode']`.
- `sessions/m02-the-neuron/day-02-activations/source.md` — rebuilt in V9 concept mode.
- `.claude/skills/{frontier-lesson-builder,frontier-visual-evidence-builder,frontier-refactor-qa}/SKILL.md` and their twins under `frontier_lab_refactor_skills_v8/skills/…` — the V9 rules.

**Design boundaries:** the compiler stays a thin orchestrator; all rendering logic stays in `v8lib`; each gate is a standalone `run()`-plus-CLI module matching the existing pattern (`run(...) -> (ok, msgs)`, `__main__` prints + exits). Tests are pure-Python (no browser) except the optional `node --check` backstop.

---

## Task 1: Widget — `%%% svg` (inline static visual)

**Files:**
- Modify: `sessions/_compiler/v8lib.py` (add `render_svg`, register in `render_widget` :173)
- Test: `sessions/_compiler/tests/test_v9_widgets.py`

- [ ] **Step 1: Write the failing test**

```python
# sessions/_compiler/tests/test_v9_widgets.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import v8lib

def test_render_svg_wraps_in_build_viz():
    lines = ['<svg viewBox="0 0 10 10"><path d="M0 0"/></svg>']
    out = v8lib.render_widget('svg', {}, lines)
    assert 'class="build-viz"' in out          # consistent styling wrapper
    assert '<svg viewBox="0 0 10 10">' in out   # svg passed through verbatim
    assert out.count('<svg') == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest sessions/_compiler/tests/test_v9_widgets.py::test_render_svg_wraps_in_build_viz -v`
Expected: FAIL — `ValueError: unknown %%%% widget type: svg` (from `render_widget` :181)

- [ ] **Step 3: Write minimal implementation**

In `v8lib.py`, add above `render_widget`:

```python
def render_svg(lines):
    """Inline static visual: pass raw SVG through, wrapped for consistent styling."""
    svg = '\n'.join(lines).strip()
    return '<div class="build-viz">%s</div>' % svg
```

In `render_widget`, add before the final `raise`:

```python
    if typ == 'svg':        return render_svg(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest sessions/_compiler/tests/test_v9_widgets.py::test_render_svg_wraps_in_build_viz -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sessions/_compiler/v8lib.py sessions/_compiler/tests/test_v9_widgets.py
git commit -m "feat(v9): %%% svg inline visual widget"
```

---

## Task 2: Widget — `%%% demo` (inline run-demo)

One self-contained click-to-run console per concept (replaces the shared s3 playground). The button reveals `out` + `take` on click; no cross-section "saw all 3" gating.

**Files:**
- Modify: `sessions/_compiler/v8lib.py` (add `render_demo`, register in `render_widget`)
- Test: `sessions/_compiler/tests/test_v9_widgets.py`

- [ ] **Step 1: Write the failing test**

```python
def test_render_demo_emits_runnable_console():
    lines = [
        'code: relu(np.array([-3,-1,0,2,5]))',
        'out: array([0, 0, 0, 2, 5])',
        'take: ReLU zeros negatives, passes positives.',
    ]
    out = v8lib.render_widget('demo', {'id': 'relu', 'label': 'run it'}, lines)
    assert 'class="demo"' in out                       # generic selector the JS engine finds
    assert 'data-demo="relu"' in out
    assert 'class="demo-run"' in out                   # the run button
    assert 'relu(np.array([-3,-1,0,2,5]))' in out      # code shown
    assert 'array([0, 0, 0, 2, 5])' in out             # output present (revealed on run)
    assert 'ReLU zeros negatives' in out               # take-away present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest sessions/_compiler/tests/test_v9_widgets.py::test_render_demo_emits_runnable_console -v`
Expected: FAIL — unknown widget type `demo`

- [ ] **Step 3: Write minimal implementation**

Add to `v8lib.py`:

```python
def render_demo(args, lines):
    """Inline, self-contained run-demo. JS engine (v9) wires .demo-run to reveal .demo-out + .demo-take."""
    d = _kv_multiline(lines)   # see helper below
    did = args.get('id', 'demo')
    label = args.get('label', 'run it')
    code = attr_esc_text(d.get('code', ''))
    out  = attr_esc_text(d.get('out', ''))
    take = inline(d.get('take', ''))
    return ('<div class="demo" data-demo="%s">'
            '<div class="demo-code"><code>%s</code>'
            '<button class="demo-run" type="button">%s ▶</button></div>'
            '<pre class="demo-out" hidden>%s</pre>'
            '<div class="demo-take" hidden>%s</div></div>'
            % (did, code, label, out, take))
```

Add two small helpers near `_kv` (:77):

```python
def _kv_multiline(lines):
    """Parse 'key: value' lines; a value may contain colons (only split on first)."""
    d = {}
    for ln in lines:
        if ':' in ln:
            k, v = ln.split(':', 1)
            if re.fullmatch(r'\w+', k.strip()):
                d[k.strip()] = v.strip()
    return d

def attr_esc_text(s):
    """Escape HTML text content (not attributes)."""
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
```

Register in `render_widget`: `if typ == 'demo': return render_demo(args, lines)`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest sessions/_compiler/tests/test_v9_widgets.py::test_render_demo_emits_runnable_console -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sessions/_compiler/v8lib.py sessions/_compiler/tests/test_v9_widgets.py
git commit -m "feat(v9): %%% demo inline run-demo widget"
```

---

## Task 3: Widget — `%%% quiz` (authored quiz DOM)

Emit the quiz question DOM from source (replacing `var QS`). Each line is one question: `q | a:INDEX | opt | opt | opt | opt | fb`. The V9 JS engine wires click-to-answer over the emitted `.q` blocks.

**Files:**
- Modify: `sessions/_compiler/v8lib.py` (add `render_quiz`)
- Test: `sessions/_compiler/tests/test_v9_widgets.py`

- [ ] **Step 1: Write the failing test**

```python
def test_render_quiz_emits_four_q_blocks_with_answer_marker():
    lines = [
        'q: What does an activation add? | a:1 | More params | Non-linearity | Faster matmul | A bias | fb: The bend.',
        'q: No activation, ten layers = ? | a:1 | more power | one linear layer | random | sigmoid | fb: One matrix.',
        'q: ReLU(z) = ? | a:1 | 1/(1+e^-z) | max(0,z) | z^2 | -z | fb: keep positives.',
        'q: Loss plateaus, ReLUs all 0? | a:1 | sigmoid bug | dead ReLUs | OOM | converged | fb: dead units.',
    ]
    out = v8lib.render_widget('quiz', {}, lines)
    assert out.count('class="q"') == 4               # 4 question blocks
    assert out.count('class="q-opt"') == 16          # 4 options each
    assert out.count('data-correct="1"') == 4        # answer index carried per question
    assert 'The bend.' in out                        # feedback present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest sessions/_compiler/tests/test_v9_widgets.py::test_render_quiz_emits_four_q_blocks_with_answer_marker -v`
Expected: FAIL — unknown widget type `quiz`

- [ ] **Step 3: Write minimal implementation**

```python
def render_quiz(lines):
    """Author-driven quiz. One '|'-delimited question per line:
       q: TEXT | a:IDX | OPT | OPT | OPT | OPT | fb: TEXT
    Emits .q blocks the v9 engine wires click-to-answer over."""
    blocks = []
    for ln in lines:
        if not ln.strip():
            continue
        parts = [p.strip() for p in ln.split('|')]
        q = parts[0][2:].strip() if parts[0].lower().startswith('q:') else parts[0]
        ans, opts, fb = 0, [], ''
        for p in parts[1:]:
            if re.match(r'a\s*:', p, re.I):
                ans = int(re.split(r':', p, 1)[1].strip())
            elif p.lower().startswith('fb:'):
                fb = p[3:].strip()
            else:
                opts.append(p)
        optshtml = ''.join(
            '<button class="q-opt" type="button" data-opt="%d">'
            '<span class="mark"></span><span>%s</span></button>' % (i, inline(o))
            for i, o in enumerate(opts))
        blocks.append(
            '<div class="q" data-correct="%d"><div class="q-ask">%s</div>'
            '<div class="q-opts">%s</div>'
            '<div class="q-fb" data-fb="%s"></div></div>'
            % (ans, inline(q), optshtml, attr_esc(fb)))
    return '<div class="quiz">' + ''.join(blocks) + '</div>'
```

Register: `if typ == 'quiz': return render_quiz(lines)`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest sessions/_compiler/tests/test_v9_widgets.py -v` (all three widget tests)
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sessions/_compiler/v8lib.py sessions/_compiler/tests/test_v9_widgets.py
git commit -m "feat(v9): %%% quiz authored-quiz widget"
```

---

## Task 4: Block renderer — `@@@ concept`

Render a concept unit as a tracked `.module-section` (auto-numbered) with head (num/title/tag), body (rendered markdown incl. widgets), and exactly one `.gotit`.

**Files:**
- Modify: `sessions/_compiler/v8lib.py` (add `render_concept`)
- Test: `sessions/_compiler/tests/test_v9_widgets.py`

- [ ] **Step 1: Write the failing test**

```python
def test_render_concept_is_tracked_section_with_one_gotit():
    block = {'type': 'concept',
             'args': {'id': 'c3', 'num': '3', 'tag': 'Meet ReLU',
                      'title': 'ReLU — a one-way valve', 'gotit': 'Met ReLU'},
             'lines': ['Intro prose about ReLU.',
                       '%%% svg', '<svg viewBox="0 0 10 10"></svg>', '%%%',
                       'Build-up prose.']}
    out = v8lib.render_concept(block)
    assert 'class="module-section"' in out
    assert 'id="c3"' in out and 'data-sec="c3"' in out
    assert out.count('class="gotit"') == 1
    assert 'Met ReLU' in out                     # gotit label
    assert 'ReLU — a one-way valve' in out       # title
    assert '<svg viewBox="0 0 10 10">' in out    # embedded visual rendered
    assert out.count('<svg') == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest sessions/_compiler/tests/test_v9_widgets.py::test_render_concept_is_tracked_section_with_one_gotit -v`
Expected: FAIL — `AttributeError: module 'v8lib' has no attribute 'render_concept'`

- [ ] **Step 3: Write minimal implementation**

Model on `render_section` (:260). Add:

```python
def render_concept(block):
    a = block['args']
    body = render_md('\n'.join(block['lines']))
    num = a.get('num', '')
    numclass = a.get('numclass', 's-study')
    btn = '<button class="gotit" type="button">%s</button>' % a.get('gotit', 'Got it')
    return ('<section class="module-section" id="%s" data-sec="%s">\n'
            '  <div class="sec-head"><span class="sec-num %s">%s</span>'
            '<span class="sec-h">%s</span><span class="sec-tag">%s</span></div>\n'
            '  <div class="sec-body">\n      %s\n      %s\n    </div>\n</section>'
            % (a['id'], a['id'], numclass, num, a.get('title', ''), a.get('tag', ''), body, btn))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest sessions/_compiler/tests/test_v9_widgets.py::test_render_concept_is_tracked_section_with_one_gotit -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sessions/_compiler/v8lib.py sessions/_compiler/tests/test_v9_widgets.py
git commit -m "feat(v9): @@@ concept block renderer"
```

---

## Task 5: Auto-number concepts + generate sidebar/nav list

The compiler must assign concept numbers 1..N in source order and build the sidebar nav from `home + concepts + quiz + produce`. Add a pure helper so numbering is testable in isolation.

**Files:**
- Modify: `sessions/_compiler/v8lib.py` (add `concept_nav_items`)
- Test: `sessions/_compiler/tests/test_v9_widgets.py`

- [ ] **Step 1: Write the failing test**

```python
def test_concept_nav_items_number_in_order():
    blocks = [
        {'type': 'hero', 'args': {}, 'lines': []},
        {'type': 'concept', 'args': {'id': 'c1', 'title': 'The collapse'}, 'lines': []},
        {'type': 'concept', 'args': {'id': 'c2', 'title': 'The bend'}, 'lines': []},
        {'type': 'quiz', 'args': {'id': 'quiz', 'title': 'Check'}, 'lines': []},
        {'type': 'produce', 'args': {'id': 'produce', 'title': 'Produce'}, 'lines': []},
    ]
    items = v8lib.concept_nav_items(blocks)
    targets = [it['target'] for it in items]
    assert targets == ['home', 'c1', 'c2', 'quiz', 'produce']
    # concept labels are numbered; home/quiz/produce are not concept-numbered
    labels = {it['target']: it['label'] for it in items}
    assert labels['c1'].startswith('1') and 'collapse' in labels['c1'].lower()
    assert labels['c2'].startswith('2')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest sessions/_compiler/tests/test_v9_widgets.py::test_concept_nav_items_number_in_order -v`
Expected: FAIL — no attribute `concept_nav_items`

- [ ] **Step 3: Write minimal implementation**

```python
def concept_nav_items(blocks):
    """Sidebar nav items for a concept lesson: home + numbered concepts + quiz + produce."""
    items = [{'target': 'home', 'label': 'Start here'}]
    n = 0
    for b in blocks:
        if b['type'] == 'concept':
            n += 1
            items.append({'target': b['args']['id'],
                          'label': '%d · %s' % (n, b['args'].get('tag') or b['args'].get('title', ''))})
        elif b['type'] in ('quiz', 'produce'):
            items.append({'target': b['args']['id'],
                          'label': b['args'].get('tag') or b['args'].get('title', b['type'].title())})
    return items
```

Also make `render_concept` receive its number: change the concept loop in the compiler (Task 6) to set `a['num']` = running index before calling `render_concept`. (Add a one-line note; the number assignment happens in `compile_html`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest sessions/_compiler/tests/test_v9_widgets.py::test_concept_nav_items_number_in_order -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sessions/_compiler/v8lib.py sessions/_compiler/tests/test_v9_widgets.py
git commit -m "feat(v9): concept auto-numbering + sidebar nav generation"
```

---

## Task 6: Compiler — `mode=='concept'` path in `compile_html`

Assemble `hero + concepts(numbered) + quiz + produce + fin` into the donor content area, and fill the sidebar nav. Keep V8 paths untouched.

**Files:**
- Modify: `sessions/_compiler/v8lib.py` (`compile_html` :316; add a content-area marker approach for the donor)
- Create: `sessions/_compiler/tests/fixtures/mini_concept.md`, `sessions/_compiler/tests/test_v9_compile.py`

**Design note:** the V9 donor (Task 8) contains a single content placeholder `<!--V9_CONTENT-->` inside `<main id="content">…</main>` and `<!--V9_NAV-->` inside the sidebar `<nav>`. Concept mode replaces those two markers, plus `title`/`brand_sub`/`nav_prev`/`nav_next`/quest-id already handled generically. This avoids the per-section `REGION_PATTERNS` regexes (which are V8-only).

- [ ] **Step 1: Write the fixture**

```markdown
<!-- sessions/_compiler/tests/fixtures/mini_concept.md -->
---
quest_id: test-mini
mode: concept
donor: v9-base.donor
page_title: "Mini Concept Test"
module_label: "Test · Mini"
title: "Mini"
subtitle: "a tiny concept lesson"
brand_sub: "Test · Mini"
spine: "bend"
nav_prev_href: "#"
nav_prev_label: "Prev"
nav_next_href: "#"
nav_next_label: "Next"
fin_title: "Mini complete!"
fin_body: "Done."
notebook_yardstick: null
---

@@@ hero
@kicker Test · Mini
@lede Ever wonder why a bend matters? Picture two straight rulers.
@goal By the end you can explain the bend.

@@@ concept id=c1 tag="The collapse" title="Straight + straight is still straight" gotit="Got it"
Two straight rulers stacked are still one straight ruler — a bend is missing.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="collapse"><path d="M0 0"/></svg>
%%%
So depth without a bend buys nothing.

@@@ concept id=c2 tag="The bend" title="A bend between the layers" gotit="Got the bend"
Put a bend between layers and they can no longer fold flat.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="bend"><path d="M0 10 L5 0"/></svg>
%%%
That bend is the activation.

@@@ concept id=c3 tag="Meet ReLU" title="ReLU — a one-way valve" gotit="Met ReLU"
ReLU passes positives and zeroes negatives.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="relu"><path d="M0 10 L5 10 L10 0"/></svg>
%%%
%%% demo id=relu label="run it"
code: relu(np.array([-3,-1,0,2,5]))
out: array([0, 0, 0, 2, 5])
take: ReLU zeros negatives, passes positives.
%%%
One cheap max — the modern default.

@@@ quiz id=quiz tag="Check" title="Four questions" gotit="Checked"
%%% quiz
q: What does an activation add? | a:1 | params | non-linearity | speed | bias | fb: The bend.
q: Ten linear layers = ? | a:1 | more power | one linear layer | random | sigmoid | fb: One matrix.
q: ReLU(z) = ? | a:1 | 1/(1+e^-z) | max(0,z) | z^2 | -z | fb: keep positives.
q: All ReLUs output 0? | a:1 | sigmoid bug | dead ReLUs | OOM | converged | fb: dead units.
%%%

@@@ produce id=produce tag="Produce" title="Watch the collapse" gotit="Done"
Predict what `(x@W1)@W2` prints vs `x@(W1@W2)`, then run it and observe they match — until you insert a ReLU. Write it in `experiment.py` and run it.

@@@ fin
```

- [ ] **Step 2: Write the failing test**

```python
# sessions/_compiler/tests/test_v9_compile.py
import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import v8lib

def _compile_mini():
    src = open(os.path.join(HERE, 'fixtures', 'mini_concept.md'), encoding='utf-8').read()
    meta, body = v8lib.split_frontmatter(src)
    blocks = v8lib.parse_blocks(body)
    donor = open(os.path.join(HERE, '..', 'shells', 'v9-base.donor'), encoding='utf-8').read()
    return v8lib.compile_html(meta, blocks, donor), meta

def test_concept_mode_assembles_all_sections():
    html, _ = _compile_mini()
    assert html.count('class="module-section"') == 5     # 3 concepts + quiz + produce
    for cid in ('c1', 'c2', 'c3', 'quiz', 'produce'):
        assert 'id="%s"' % cid in html
    assert html.count('class="gotit"') == 5              # one per section
    assert '<!--V9_CONTENT-->' not in html               # placeholder consumed
    assert '<!--V9_NAV-->' not in html
    # nav parity: every data-target has a matching section id (or home)
    import re
    targets = set(re.findall(r'data-target="([^"]+)"', html))
    assert targets == {'home', 'c1', 'c2', 'c3', 'quiz', 'produce'}

def test_concept_mode_shows_visual_in_every_concept():
    html, _ = _compile_mini()
    import re
    for cid in ('c1', 'c2', 'c3'):
        sec = re.search(r'id="%s".*?</section>' % cid, html, re.DOTALL).group(0)
        assert ('<svg' in sec) or ('build-embed' in sec), '%s has no visual' % cid
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest sessions/_compiler/tests/test_v9_compile.py -v`
Expected: FAIL — either `v9-base.donor` missing (Task 8 not done yet) or `compile_html` has no concept path. **This task and Task 8 are interdependent; implement Task 8's donor stub first if the file is missing, then return.** (See Task 8 note.)

- [ ] **Step 4: Write minimal implementation**

In `compile_html` (:316), after computing `regions`/`secs`/`js`, add near the top:

```python
    if meta.get('mode') == 'concept':
        return _compile_concept(meta, blocks, donor)
```

Add the new function:

```python
def _compile_concept(meta, blocks, donor):
    # quest-id: the v9 donor is a NEUTRAL template carrying data-quest-id="__QUEST_ID__".
    # Substitute the source's quest_id, then verify it landed. (Do NOT compare against a
    # donor-baked id — the donor is shared across all concept lessons.)
    qid = meta['quest_id']
    donor = donor.replace('__QUEST_ID__', qid)          # substitute FIRST
    if ('data-quest-id="%s"' % qid) not in donor:
        raise RuntimeError("donor missing data-quest-id=\"__QUEST_ID__\" template token")

    bt = {b['type']: b for b in blocks}
    # content = hero + numbered concepts + quiz + produce + fin
    parts = []
    if 'hero' in bt:
        parts.append(render_hero(meta, bt['hero']))
    n = 0
    for b in blocks:
        if b['type'] == 'concept':
            n += 1
            b['args']['num'] = str(n)
            parts.append(render_concept(b))
        elif b['type'] == 'quiz':
            parts.append(render_quiz_section(b))
        elif b['type'] == 'produce':
            parts.append(render_produce_section(b))
    fin_html = render_fin(meta)
    content = '\n\n    '.join(parts)

    nav = render_sidebar_nav_items(meta, concept_nav_items(blocks))

    H = donor
    H = sub_once(r'<title>.*?</title>', '<title>%s</title>' % meta.get('page_title', ''), H, 'title')
    H = sub_once(r'<div class="brand-sub">.*?</div>',
                 '<div class="brand-sub">%s</div>' % meta.get('brand_sub', ''), H, 'brand-sub')
    H = H.replace('<!--V9_NAV-->', nav, 1)
    H = H.replace('<!--V9_CONTENT-->', content + '\n\n    ' + fin_html, 1)
    H = sub_once(REGION_PATTERNS['nav_prev'],
                 '<a class="lnav prev" href="%s"><span class="d">← Prev</span><span class="t">%s</span></a>'
                 % (meta.get('nav_prev_href', ''), meta.get('nav_prev_label', '')), H, 'nav-prev')
    H = sub_once(REGION_PATTERNS['nav_next'],
                 '<a class="lnav next" href="%s"><span class="d">Next →</span><span class="t">%s</span></a>'
                 % (meta.get('nav_next_href', ''), meta.get('nav_next_label', '')), H, 'nav-next')
    return H
```

**Quest-id decision (consistent with Task 8):** the v9 donor is a neutral template with `data-quest-id="__QUEST_ID__"`. `_compile_concept` substitutes the source `quest_id` in and asserts it landed — there is **no** donor-vs-source mismatch comparison (that only makes sense for V8's per-day donors). The substitution happens before any marker replacement, and Task 9's no-leak check confirms `__QUEST_ID__` never survives into output.

Add the two section wrappers + nav renderer (model on `render_concept` / `render_sidebar_nav`):

```python
def render_quiz_section(block):
    a = block['args']
    body = render_md('\n'.join(block['lines']))
    btn = '<button class="gotit" type="button" disabled>%s</button>' % a.get('gotit', 'answer all first')
    return ('<section class="module-section" id="%s" data-sec="%s">\n'
            '  <div class="sec-head"><span class="sec-num s-quiz">%s</span>'
            '<span class="sec-h">%s</span><span class="sec-tag">%s</span></div>\n'
            '  <div class="sec-body">\n      %s\n      %s\n    </div>\n</section>'
            % (a['id'], a['id'], a.get('num', ''), a.get('title', ''), a.get('tag', 'Quiz'), body, btn))

def render_produce_section(block):
    a = block['args']
    body = render_md('\n'.join(block['lines']))
    btn = '<button class="gotit" type="button">%s</button>' % a.get('gotit', 'Done')
    return ('<section class="module-section" id="%s" data-sec="%s">\n'
            '  <div class="sec-head"><span class="sec-num s-produce">%s</span>'
            '<span class="sec-h">%s</span><span class="sec-tag">%s</span></div>\n'
            '  <div class="sec-body">\n      %s\n      %s\n    </div>\n</section>'
            % (a['id'], a['id'], a.get('num', ''), a.get('title', ''), a.get('tag', 'Produce'), body, btn))

def render_sidebar_nav_items(meta, items):
    rows = ['      <div class="nav-group-label">%s</div>' % meta.get('module_label', '')]
    for it in items:
        rows.append('      <button class="nav-link" data-target="%s"><span class="nl-dot"></span>%s</button>'
                    % (it['target'], it['label']))
    return '\n'.join(rows)
```

**Note:** `render_hero` (:245) uses `meta['module_label']`, `meta['title']`, `meta['subtitle']` — the fixture front-matter provides all three. `@@@ hero` in concept mode uses `@kicker/@lede/@goal`; `render_hero` currently reads `@lede`/`@goal` and takes kicker from `meta['module_label']`. Keep that behavior (kicker = module_label).

- [ ] **Step 5: Run test to verify it passes** (after Task 8 donor exists)

Run: `pytest sessions/_compiler/tests/test_v9_compile.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add sessions/_compiler/v8lib.py sessions/_compiler/tests/test_v9_compile.py sessions/_compiler/tests/fixtures/mini_concept.md
git commit -m "feat(v9): concept-mode compile_html path (hero+concepts+quiz+produce+fin)"
```

---

## Task 7: Idempotency test (determinism guarantee)

**Files:**
- Test: `sessions/_compiler/tests/test_v9_compile.py`

- [ ] **Step 1: Write the test**

```python
def test_concept_compile_is_idempotent():
    a, _ = _compile_mini()
    b, _ = _compile_mini()
    assert a == b                    # deterministic
    # recompiling the OUTPUT's source must be byte-identical (no drift)
```

- [ ] **Step 2: Run**

Run: `pytest sessions/_compiler/tests/test_v9_compile.py::test_concept_compile_is_idempotent -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add sessions/_compiler/tests/test_v9_compile.py
git commit -m "test(v9): concept compile determinism"
```

---

## Task 8: Create `v9-base.donor` (shell + generalized JS engine)

**Do this BEFORE finishing Task 6's test run** (Task 6 imports the donor). Build it in two commits: first a structural stub so Task 6 compiles, then the generalized JS.

**Files:**
- Create: `sessions/_compiler/shells/v9-base.donor`

- [ ] **Step 1: Copy the V8 donor head/CSS as the base**

Start from `sessions/_compiler/shells/m02-day-02.donor`. Keep verbatim: the `<!doctype>`, `<head>` (all CSS, the theme pre-paint script at :6), the `.layout`/`.sidebar` structure, the glossary-tooltip script (donor :718-756), and the viz auto-resize script (donor :766-785). Add CSS for the new demo widget (reuse `.play`/`.play-console` styling; add `.demo`, `.demo-run`, `.demo-out`, `.demo-take` rules mirroring the existing `.play*` rules).

- [ ] **Step 2: Replace the body content region with placeholders**

In the sidebar `<nav aria-label="Sections">`, replace the 8 hardcoded `<button>`s with `<!--V9_NAV-->`.
In `<main id="content">`, replace everything from the hero through section s7 with:
```html
    <section id="home" class="hero"><!--hero rendered by compiler--></section>
    <!--V9_CONTENT-->
```
Wait — the hero is emitted by the compiler into `<!--V9_CONTENT-->` (Task 6 prepends `render_hero`). So `<main id="content">` should contain only:
```html
  <main id="content">
    <!--V9_CONTENT-->
    <footer class="site-footer">Frontier Lab · Foundations — a fully self-contained single file. Progress is saved locally in your browser (localStorage), works offline, just double-click to open.</footer>
  </main>
```
Keep the `data-quest-id` attribute on whatever element currently carries it (search the m02 donor for `data-quest-id`; it's on the element at donor :275). Because the v9 donor is **shared** across all concept lessons, it cannot bake in one lesson's id — replace `data-quest-id="wf2-d02-activations"` with **`data-quest-id="__QUEST_ID__"`**. `_compile_concept` (Task 6) substitutes the source `quest_id` for this token before any other replacement, and Task 9's no-leak check confirms the token never survives. (This is the single source of the decision; Task 6's snippet already implements it.)

- [ ] **Step 3: Commit the stub**

```bash
git add sessions/_compiler/shells/v9-base.donor
git commit -m "feat(v9): v9-base donor stub (head/CSS + content placeholders)"
```

- [ ] **Step 4: Generalize the JS engine**

Copy the shipped `<script>` (donor :495-714) into `v9-base.donor` and change ONLY the three id-coupled engines:

1. **Delete** the playground engine (`var DEMOS`, `playSec=getElementById('s3')`, the `.play-btn` loop, donor :594-618).
2. **Delete** the build-reveal engine (`var BUILD`, `buildWrap=getElementById('build')`, `buildSec=getElementById('s5')`, all reveal logic, donor :621-664).
3. **Delete** the `var QS` array and its injector (`quizWrap=getElementById('quiz')`, `quizSec=getElementById('s6')`, donor :667-700).

Add a generalized demo engine + a generalized quiz engine that operate on the compiler-emitted DOM:

```javascript
/* ===== INLINE DEMOS (generic; any number, any section) ===== */
Array.prototype.slice.call(document.querySelectorAll('.demo')).forEach(function(d){
  var run = d.querySelector('.demo-run'),
      out = d.querySelector('.demo-out'),
      take = d.querySelector('.demo-take');
  if(!run) return;
  run.addEventListener('click', function(){
    if(out) out.hidden = false;
    if(take) take.hidden = false;
    run.disabled = true; run.textContent = 'ran ✓';
  });
});

/* ===== QUIZ (generic; reads compiler-emitted .q blocks) ===== */
Array.prototype.slice.call(document.querySelectorAll('.quiz')).forEach(function(quizWrap){
  var quizSec = quizWrap.closest('.module-section');
  var qs = Array.prototype.slice.call(quizWrap.querySelectorAll('.q'));
  var answered = {};
  qs.forEach(function(q, qi){
    var correct = parseInt(q.getAttribute('data-correct'), 10);
    var fb = q.querySelector('.q-fb'), fbText = fb ? fb.getAttribute('data-fb') : '';
    var opts = Array.prototype.slice.call(q.querySelectorAll('.q-opt'));
    opts.forEach(function(o, oi){
      o.addEventListener('click', function(){
        if(answered[qi]) return; answered[qi] = true;
        opts.forEach(function(c){ c.classList.add('locked'); });
        if(oi === correct){ o.classList.add('correct'); fb.className='q-fb good show'; fb.innerHTML='✓ '+fbText; }
        else { o.classList.add('wrong'); opts[correct].classList.add('correct'); fb.className='q-fb bad show'; fb.innerHTML='The correct answer is the green one. '+fbText; }
        if(Object.keys(answered).length >= qs.length){
          var g = quizSec.querySelector('.gotit'); if(g){ g.disabled=false; g.textContent='All answered — check ✓'; }
        }
      });
    });
  });
});
```

Leave the progress/checklist/nav/scrollspy/reset/theme engines (donor :495-591, :702-713) **unchanged** — they are already generic over `.module-section` + `data-sec` + `.nav-link`.

- [ ] **Step 5: Syntax-check the emitted JS**

Compile via `v8lib` directly (NOT the `compile_lesson.py` CLI — its gate dispatch/concept branch don't exist until Tasks 10-11, and the current unbranched gates would hard-fail a concept source and refuse to write):

```bash
python3 - <<'PY'
import os, sys
C = 'sessions/_compiler'
sys.path.insert(0, C)
import v8lib
src = open(C+'/tests/fixtures/mini_concept.md', encoding='utf-8').read()
meta, body = v8lib.split_frontmatter(src)
html = v8lib.compile_html(meta, v8lib.parse_blocks(body),
                          open(C+'/shells/v9-base.donor', encoding='utf-8').read())
open('/tmp/mini.html','w',encoding='utf-8').write(html)
print('compiled', len(html), 'chars')
PY
node -e "const h=require('fs').readFileSync('/tmp/mini.html','utf8'); const m=[...h.matchAll(/<script>([\s\S]*?)<\/script>/g)]; m.forEach(s=>require('vm').compileFunction(s[1])); console.log('JS OK:', m.length, 'scripts')"
```
Expected: `compiled N chars` then `JS OK: N scripts` (no syntax error). The full CLI path is exercised later by Task 11's test.

- [ ] **Step 6: Run Task 6 + 7 tests now that the donor exists**

Run: `pytest sessions/_compiler/tests/test_v9_compile.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add sessions/_compiler/shells/v9-base.donor
git commit -m "feat(v9): generalized JS engine (generic demos + quiz; drop s3/s5/s6 coupling)"
```

---

## Task 9: New gate — `concept_shell_gate.py`

**Files:**
- Create: `sessions/_compiler/gates/concept_shell_gate.py`
- Test: `sessions/_compiler/tests/test_concept_gates.py`

- [ ] **Step 1: Write the failing test**

```python
# sessions/_compiler/tests/test_concept_gates.py
import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import v8lib, concept_shell_gate

def _compile_mini():
    src = open(os.path.join(HERE, 'fixtures', 'mini_concept.md'), encoding='utf-8').read()
    meta, body = v8lib.split_frontmatter(src)
    blocks = v8lib.parse_blocks(body)
    donor = open(os.path.join(HERE, '..', 'shells', 'v9-base.donor'), encoding='utf-8').read()
    return v8lib.compile_html(meta, blocks, donor), meta

def test_concept_shell_gate_passes_valid_lesson():
    html, meta = _compile_mini()
    ok, msgs = concept_shell_gate.run(html, meta)
    assert ok, '\n'.join(msgs)

def test_concept_shell_gate_fails_when_a_concept_has_no_visual():
    html, meta = _compile_mini()
    import re
    # strip the svg out of concept c1 to simulate a text-only concept
    broken = re.sub(r'(id="c1".*?)<div class="build-viz">.*?</div>(.*?</section>)',
                    r'\1\2', html, count=1, flags=re.DOTALL)
    ok, msgs = concept_shell_gate.run(broken, meta)
    assert not ok
    assert any('visual' in m.lower() for m in msgs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest sessions/_compiler/tests/test_concept_gates.py -k concept_shell -v`
Expected: FAIL — `ModuleNotFoundError: concept_shell_gate`

- [ ] **Step 3: Write minimal implementation**

Model the module structure on `shell_invariant_gate.py`. Core `run`:

```python
#!/usr/bin/env python3
# Concept Shell Gate (v9) — asserts concept-lesson invariants on compiled HTML.
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v8lib

def run(html, meta, donor=None):
    msgs, ok = [], [True]
    def chk(cond, label):
        msgs.append(('pass ' if cond else 'FAIL ') + label); ok[0] = ok[0] and bool(cond)

    qid = meta.get('quest_id')
    if qid:
        chk(('data-quest-id="%s"' % qid) in html, 'quest-id frozen (%s)' % qid)

    # concept sections: id/data-sec starting 'c'
    concepts = re.findall(r'<section class="module-section" id="(c\w+)"[^>]*>.*?</section>', html, re.DOTALL)
    ids = re.findall(r'<section class="module-section" id="(c\w+)"', html)
    chk(len(ids) >= 3, '>=3 concept sections (got %d)' % len(ids))

    # each concept has >=1 visual and exactly one gotit
    for cid in ids:
        sec = re.search(r'id="%s".*?</section>' % re.escape(cid), html, re.DOTALL).group(0)
        has_visual = ('<svg' in sec) or ('build-embed' in sec)
        chk(has_visual, 'concept %s has a visual' % cid)
        chk(sec.count('class="gotit"') == 1, 'concept %s has exactly one gotit' % cid)

    # exactly one quiz section with 4 questions, one produce
    chk(html.count('data-sec="quiz"') == 1, 'exactly one quiz section')
    chk(html.count('class="q"') == 4, 'quiz has 4 questions (got %d)' % html.count('class="q"'))
    chk(html.count('data-sec="produce"') == 1, 'exactly one produce section')
    # produce must reference a runnable artifact (experiment.py) so a concept lesson
    # cannot silently drop its Produce artifact. Only enforced when meta opts in.
    if meta.get('require_artifact', True):
        prod = re.search(r'data-sec="produce".*?</section>', html, re.DOTALL)
        chk(bool(prod) and 'experiment.py' in prod.group(0),
            'produce references an experiment.py artifact')

    # nav parity: every data-target (minus home) maps to a section id, and vice-versa
    targets = set(re.findall(r'data-target="([^"]+)"', html)) - {'home'}
    sec_ids = set(re.findall(r'<section class="module-section" id="([^"]+)"', html))
    chk(targets == sec_ids, 'sidebar nav parity (targets=%s sections=%s)' % (sorted(targets), sorted(sec_ids)))

    # localStorage keys, fin, no marker leakage
    chk('frontier-lesson:' in html, 'localStorage frontier-lesson:')
    chk('frontier-theme' in html, 'localStorage frontier-theme')
    chk('class="fin" id="fin"' in html, '.fin banner')
    for marker in ('<!--V9_CONTENT-->', '<!--V9_NAV-->', '__QUEST_ID__', '@@@', '%%%'):
        chk(marker not in html, 'no leaked marker %r' % marker)

    return ok[0], msgs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('lesson'); ap.add_argument('--source', required=True)
    a = ap.parse_args()
    meta, _ = v8lib.split_frontmatter(open(a.source, encoding='utf-8').read())
    ok, msgs = run(open(a.lesson, encoding='utf-8').read(), meta)
    for m in msgs: print('  ', m)
    print('\n' + ('PASS' if ok else 'FAIL')); sys.exit(0 if ok else 3)

if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest sessions/_compiler/tests/test_concept_gates.py -k concept_shell -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add sessions/_compiler/gates/concept_shell_gate.py sessions/_compiler/tests/test_concept_gates.py
git commit -m "feat(v9): concept_shell_gate — V9 structural invariants + broken-variant FAIL"
```

---

## Task 10: `reader_flow_gate.py` — concept-mode branch

**Files:**
- Modify: `sessions/_compiler/gates/reader_flow_gate.py`
- Test: `sessions/_compiler/tests/test_concept_gates.py`

- [ ] **Step 1: Write the failing test**

```python
import reader_flow_gate

def _blocks_and_meta():
    src = open(os.path.join(HERE, 'fixtures', 'mini_concept.md'), encoding='utf-8').read()
    meta, body = v8lib.split_frontmatter(src)
    return meta, v8lib.parse_blocks(body)

def test_reader_flow_concept_mode_passes():
    meta, blocks = _blocks_and_meta()
    ok, msgs = reader_flow_gate.run(meta, blocks)
    assert ok, '\n'.join(msgs)
    # must NOT emit the s1-strict failures
    assert not any('s1' in m and 'FAIL' in m for m in msgs)

def test_reader_flow_concept_fails_concept_without_visual():
    meta, blocks = _blocks_and_meta()
    # remove the %%% svg block from c1 (lines between '%%% svg' and the closing '%%%')
    for b in blocks:
        if b['type'] == 'concept' and b['args'].get('id') == 'c1':
            b['lines'] = ['Intro prose only, no visual.', 'Build-up prose only.']
    ok, msgs = reader_flow_gate.run(meta, blocks)
    assert not ok
    assert any('visual' in m.lower() and 'c1' in m for m in msgs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest sessions/_compiler/tests/test_concept_gates.py -k reader_flow -v`
Expected: FAIL — current gate lands mini source in `clean`/strict path and fails on Jargon Ladder / picture-before-vocab (or errors on missing `s1`).

- [ ] **Step 3: Write minimal implementation**

At the very top of `run()` (after the `fail/pas/warn` closures), branch:

```python
    if meta.get('mode') == 'concept':
        return _run_concept(meta, blocks, msgs, ok, fail, pas, warn)
```

Add:

```python
def _run_concept(meta, blocks, msgs, ok, fail, pas, warn):
    spine_word = (meta.get('spine') or 'bend').split(':')[0].split()[0].lower()

    # hero: human-first, no frontier-pressure
    hero = next((b for b in blocks if b['type'] == 'hero'), None)
    htxt = '\n'.join(hero['lines']).lower() if hero else ''
    hit = [x for x in v8lib.FRONTIER_TOKENS if x.lower() in htxt]
    fail('hero opens frontier-first (found %s)' % hit) if hit else pas('hero no frontier-pressure')
    curiosity = ['you', 'your', 'imagine', 'picture', '?', 'ever', 'what if', 'yesterday', 'here is', 'wonder']
    pas('hero human/curiosity opening') if any(c in htxt for c in curiosity) else fail('hero no human/curiosity anchor')

    # every concept has a visual (raw-source marker check — no tag stripping)
    concepts = [b for b in blocks if b['type'] == 'concept']
    fail('need >=3 concept units (got %d)' % len(concepts)) if len(concepts) < 3 else pas('%d concept units' % len(concepts))
    VIS = ('<svg', '%%% svg', '%%% viz', 'build-embed', 'build-viz')
    for b in concepts:
        raw = '\n'.join(b['lines'])
        cid = b['args'].get('id', '?')
        (pas('concept %s ships a visual' % cid) if any(v in raw for v in VIS)
         else fail('concept %s has NO visual (depict-on-introduction)' % cid))

    # spine word across >=3 blocks (hero + concepts)
    texts = ([htxt] + ['\n'.join(b['lines']).lower() for b in concepts])
    present = sum(1 for t in texts if spine_word in t)
    pas("spine ('%s') in %d blocks" % (spine_word, present)) if present >= 3 else fail("spine ('%s') in <3 blocks (%d)" % (spine_word, present))

    # produce = discovery-framed
    prod = next((b for b in blocks if b['type'] == 'produce'), None)
    ptxt = '\n'.join(prod['lines']).lower() if prod else ''
    cues = ['predict', 'guess', 'observe', 'notice', 'watch', 'what you should see', 'before you']
    pas('produce is discovery-framed') if any(c in ptxt for c in cues) else fail('produce not discovery-framed')

    return ok[0], msgs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest sessions/_compiler/tests/test_concept_gates.py -k reader_flow -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Verify V8 lessons still parse unchanged (regression guard)**

Run: `python3 sessions/_compiler/gates/reader_flow_gate.py sessions/m02-the-neuron/day-03-layers-forward-pass/source.md`
Expected: same output as before this task (verbatim mode → informational warnings, exit 0). If output changed, the concept branch leaked into non-concept mode — fix the guard.

- [ ] **Step 6: Commit**

```bash
git add sessions/_compiler/gates/reader_flow_gate.py sessions/_compiler/tests/test_concept_gates.py
git commit -m "feat(v9): reader_flow_gate concept-mode branch (per-concept visual check)"
```

---

## Task 11: `compile_lesson.py` — gate dispatch on `meta['mode']`

**Files:**
- Modify: `sessions/_compiler/compile_lesson.py` (:56, :67)

- [ ] **Step 1: Write the failing test**

```python
# add to test_v9_compile.py
import subprocess
def test_compile_lesson_cli_concept_mode_passes(tmp_path):
    src = os.path.join(HERE, 'fixtures', 'mini_concept.md')
    out = tmp_path / 'mini.html'
    r = subprocess.run(['python3', os.path.join(HERE, '..', 'compile_lesson.py'), src,
                        '--donor', os.path.join(HERE, '..', 'shells', 'v9-base.donor'),
                        '--out', str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'concept_shell_gate' in r.stdout.lower() or 'concept shell' in r.stdout.lower()
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest sessions/_compiler/tests/test_v9_compile.py::test_compile_lesson_cli_concept_mode_passes -v`
Expected: FAIL — the V8 `shell_invariant_gate` runs and exits 3 (asserts 7 sections / s1..s7).

- [ ] **Step 3: Write minimal implementation**

In `compile_lesson.py`, import the new gate and branch. Replace **only** the gate-running section (:55-69) with the block below. **Keep the existing write + exit-code block that follows (:71-80)** — `if not args.check_only: write(html)` then `if not sok: sys.exit(3)` — do not delete it; the new branch just sets `sok`/`smsgs` for it to consume.

```python
    concept_mode = (meta.get('mode') == 'concept')

    # -- Reader Flow Gate (source) : block write on failure --
    rok, rmsgs = reader_flow_gate.run(meta, blocks)
    log('\n-- Reader Flow Gate (source) --')
    for m in rmsgs: log('  ', m)
    if not rok:
        log('\nReader Flow Gate FAILED — nothing written.'); sys.exit(2)

    donor = open(donor_path, encoding='utf-8').read()
    html = v8lib.compile_html(meta, blocks, donor)

    if concept_mode:
        import concept_shell_gate
        sok, smsgs = concept_shell_gate.run(html, meta, donor=donor)
        log('\n-- Concept Shell Gate (output) --')
        for m in smsgs: log('  ', m)
        # notebook smoothness: 'N/A' or True = pass; a real 'FAIL'/False blocks the write.
        try:
            import notebook_smoothness_gate
            nstatus, nmsgs = notebook_smoothness_gate.run(html, meta)
            log('\n-- Notebook Smoothness Gate --')
            for m in nmsgs: log('  ', m)
            if nstatus is False or str(nstatus).upper() == 'FAIL':
                sok = False; log('   notebook smoothness FAILED')
        except Exception as e:
            log('   notebook smoothness skipped:', e)
    else:
        sok, smsgs = shell_invariant_gate.run(html, meta, donor=donor)
        log('\n-- Shell Invariant Gate (output vs donor) --')
        for m in smsgs: log('  ', m)
```

Add `sys.path.insert(0, os.path.join(HERE, 'gates'))` already exists (:26). `notebook_smoothness_gate.run(html, meta)` returns `(status, msgs)` where `status` is `'N/A'` for a null yardstick (treated as pass) — this branch fails the compile only on a genuine `FALSE`/`'FAIL'`, not on `'N/A'`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest sessions/_compiler/tests/test_v9_compile.py::test_compile_lesson_cli_concept_mode_passes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sessions/_compiler/compile_lesson.py sessions/_compiler/tests/test_v9_compile.py
git commit -m "feat(v9): gate dispatch on meta['mode'] (concept vs v8 path)"
```

---

## Task 12: V8 regression guard (shipped lessons untouched)

Prove the V9 work did not change any shipped V8 output.

**Files:**
- Test: `sessions/_compiler/tests/test_v8_regression.py`

- [ ] **Step 1: Snapshot current shipped outputs**

Run:
```bash
for d in sessions/m02-the-neuron/day-0{1,3,4,5,6,7,8,9}-* sessions/m03-attention/day-0*; do
  test -f "$d/source.md" && python3 sessions/_compiler/compile_lesson.py "$d/source.md" --check-only --quiet && echo "OK $d" || echo "SKIP/FAIL $d"
done
```
Expected: every existing lesson still `OK` (compiles, shell-invariant gate passes) — because non-concept mode is unchanged.

- [ ] **Step 2: Write a byte-identity test for one representative V8 lesson**

```python
# sessions/_compiler/tests/test_v8_regression.py
import os, sys, subprocess
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

def test_v8_day_shipped_output_unchanged(tmp_path):
    src = os.path.join(ROOT, 'sessions/m02-the-neuron/day-03-layers-forward-pass/source.md')
    shipped = os.path.join(ROOT, 'sessions/m02-the-neuron/day-03-layers-forward-pass/lesson.html')
    out = tmp_path / 'd03.html'
    r = subprocess.run(['python3', os.path.join(HERE, '..', 'compile_lesson.py'), src,
                        '--out', str(out), '--quiet'], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.read_text(encoding='utf-8') == open(shipped, encoding='utf-8').read()
```

- [ ] **Step 3: Run**

Run: `pytest sessions/_compiler/tests/test_v8_regression.py -v`
Expected: PASS (V8 recompile is byte-identical to shipped).

- [ ] **Step 4: Commit**

```bash
git add sessions/_compiler/tests/test_v8_regression.py
git commit -m "test(v9): V8 shipped lessons remain byte-identical (no regression)"
```

---

## Task 13: Author Day 2 `source.md` in V9 concept mode

Rebuild the real lesson with 7 concept units. Reuse existing SVGs (extract from `var BUILD` in the m02 donor, `'`→`"`) and the two iframes.

**Files:**
- Modify: `sessions/m02-the-neuron/day-02-activations/source.md` (full rewrite)

- [ ] **Step 1: Extract the 5 SVGs**

From `sessions/_compiler/shells/m02-day-02.donor`: `var BUILD` holds **6** SVGs; use only `BUILD[0]` (ReLU), `[1]` (sigmoid), `[2]` (collapse), `[3]` (cure) — index explicitly, do not assume the array has 4 entries. Convert their single-quote attributes to double-quote. The tanh SVG is inline at donor :397 (already double-quoted). Save these 5 for the `%%% svg` blocks.

- [ ] **Step 2: Write the front-matter + 7 concepts**

Front-matter:
```yaml
---
quest_id: wf2-d02-activations
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 2 — Activation Functions"
module_label: "Module 2 · Train"
title: "Activation Functions"
subtitle: "The Bend That Makes Depth Matter"
brand_sub: "Foundations · M2 Day 2"
spine: "bend"
nav_prev_href: "../day-01-single-neuron/lesson.html"
nav_prev_label: "A Single Neuron"
nav_next_href: "../day-03-layers-forward-pass/lesson.html"
nav_next_label: "Layers & the Forward Pass"
fin_title: "Module 2 · Day 2 complete! 🏆"
fin_body: "Nice work — you've completed <b>Activation Functions</b>.<br>Next up: <b>Layers & the Forward Pass</b>."
notebook_yardstick: 00-neural-networks/fundamentals/03_activation_functions.ipynb
---
```

Concept map (author intro → `%%% svg`/`%%% viz` → build-up, reusing the existing prose from the shipped lesson, redistributed):

| id | tag | title | visual | build-up source (from shipped) |
|----|-----|-------|--------|-------------------------------|
| c1 | The problem | Straight + straight is still straight | `%%% svg` collapse SVG | donor s1 "why it exists" + Math Ladder steps 1-3 |
| c2 | The fix | A bend between the layers | `%%% svg` cure SVG | s1 "activation adds non-linearity" + Math Ladder step 4 |
| c3 | Meet ReLU | ReLU — a one-way valve | `%%% svg` ReLU SVG + `%%% demo relu` | s2 valve card + s4 ReLU row |
| c4 | Meet sigmoid | Sigmoid — a soft dimmer | `%%% svg` sigmoid SVG + `%%% demo sigmoid` | s2 dimmer card + s4 sigmoid row |
| c5 | Meet tanh | tanh — the zero-centered cousin | `%%% svg` tanh SVG | s4 tanh paragraph |
| c6 | When bends break | Dead ReLUs and saturation | `%%% viz src=../../viz/activation-derivatives.html` | s4 staff-lens callouts (failure mode + trade-off + interview line) |
| c7 | The limit | One line can't do XOR | `%%% viz src=../../viz/xor-limit.html` | s4 XOR paragraph + frontier payoff (GELU/SwiGLU) |

Then `@@@ quiz` (the 4 existing questions, incl. the dead-ReLU diagnostic — reformatted to the `%%% quiz` `|` syntax) and `@@@ produce` (the existing Option-A/Option-B collapse-then-cure artifact, discovery-framed — keep the `predict → run → observe` cues so the reader-flow gate passes) and `@@@ fin`.

**Preserve:** the bilingual "why this trips people up" note (→ c1 or c2), the jargon glosses via `[[term||tooltip]]`, and all staff-depth content (→ c6). Keep the `%%% demo` outputs numerically identical to the shipped `var DEMOS` values.

**Kicker note:** `render_hero` uses `meta['module_label']` as the hero kicker (the `@kicker` line in the block is ignored). Set `module_label: "Module 2 · Train"` in front-matter — that string becomes both the sidebar nav-group label and the hero kicker. If you want the shipped "Module 2 · Train · Day 2" kicker exactly, set `module_label` to that; pick one and confirm it reads well in both places.

- [ ] **Step 3: Compile**

Run:
```bash
python3 sessions/_compiler/compile_lesson.py sessions/m02-the-neuron/day-02-activations/source.md
```
Expected: Reader Flow Gate PASS, Concept Shell Gate PASS, Notebook Smoothness PASS/N-A, `wrote …/lesson.html`, exit 0.
If any gate FAILs, read the messages, fix `source.md`, recompile. Repeat until all green.

- [ ] **Step 4: Syntax + structure backstop**

Run:
```bash
node -e "const h=require('fs').readFileSync('sessions/m02-the-neuron/day-02-activations/lesson.html','utf8'); [...h.matchAll(/<script>([\s\S]*?)<\/script>/g)].forEach(s=>require('vm').compileFunction(s[1])); console.log('JS OK')"
python3 sessions/lesson_audit.py m02-the-neuron 2>/dev/null | tail -5 || echo "audit not applicable"
```
Expected: `JS OK`; audit shows Day 2 healthy.

- [ ] **Step 5: Commit**

```bash
git add sessions/m02-the-neuron/day-02-activations/source.md sessions/m02-the-neuron/day-02-activations/lesson.html
git commit -m "feat(v9): rebuild m02 Day 2 Activation Functions as first concept-driven lesson"
```

---

## Task 14: Skill edits (3 skills × 2 locations = 6 files)

**Files:**
- Modify: `.claude/skills/frontier-lesson-builder/SKILL.md` + `frontier_lab_refactor_skills_v8/skills/frontier-lesson-builder/SKILL.md`
- Modify: `.claude/skills/frontier-visual-evidence-builder/SKILL.md` + twin
- Modify: `.claude/skills/frontier-refactor-qa/SKILL.md` + twin

- [ ] **Step 1: `frontier-lesson-builder` — replace fixed blueprint with V9 concept-unit model**

Replace the "Reader Flow Blueprint" section with:
```markdown
## Reader Flow Blueprint (v9 — concept-driven)

A lesson body is a sequence of **concept units**, not a fixed template. Author `source.md` as:

    hero (curiosity hook)
    → concept unit 1  (intro in plain words → its own inline visual → build-up)
    → concept unit 2  (…)
    → … as many concept units as the topic needs
    quiz (one section, all questions)
    produce (discovery artifact)
    fin

Each concept unit MUST carry its own inline visual, placed immediately after the intro,
before the build-up. Never defer a concept's picture to a later unit or to one shared
"build" section. Front-load a jargon gloss; introduce each term defined-before-use.
```
Add to the Learning Barrier Gate P0 list: `- a concept unit with no visual` and `- a concept's picture deferred to a later unit`.

- [ ] **Step 2: `frontier-visual-evidence-builder` — add the depict-in-unit rule**

Add:
```markdown
## Depict-in-unit rule (v9)

Every concept unit carries its own inline visual: a static labeled figure at minimum,
an interactive viz where the behavior is explorable. Analogy is NOT depiction — a
"valve"/"dimmer" card does not satisfy "show the ReLU curve." A lesson must not defer
a concept's picture to a later unit or to a single shared late "build" section.
The mechanical backstop is reader_flow_gate's concept-visual check + concept_shell_gate.
```

- [ ] **Step 3: `frontier-refactor-qa` — update gates to V9 invariants**

In "v8 Gates", add a V9 subsection:
```markdown
### V9 Concept Gates (mode: concept)

- Gate dispatch: compile_lesson.py branches on meta['mode']; concept mode runs
  reader_flow_gate (concept branch) + concept_shell_gate + notebook_smoothness.
- concept_shell_gate: >=3 concept units each with a visual + one gotit; one quiz
  (4 Qs) + one produce; sidebar nav parity; idempotent recompile; no marker leakage.
- reader_flow_gate concept branch: hero human-first; every concept unit ships a
  visual (P0 if not); spine across >=3 blocks; produce discovery-framed.
- "concept unit without a visual" is P0 for a fresh build.
```

- [ ] **Step 4: Apply each edit to BOTH locations, then verify parity**

Run:
```bash
for s in frontier-lesson-builder frontier-visual-evidence-builder frontier-refactor-qa; do
  diff -q ".claude/skills/$s/SKILL.md" "frontier_lab_refactor_skills_v8/skills/$s/SKILL.md" \
    && echo "PARITY OK $s" || echo "DRIFT $s"
done
```
Expected: `PARITY OK` for all three.

- [ ] **Step 5: Commit**

```bash
git add .claude/skills frontier_lab_refactor_skills_v8/skills
git commit -m "docs(v9): concept-driven rules in lesson-builder, visual-evidence, refactor-qa (both locations)"
```

---

## Task 15: Full gate run + LLM-judge + adversarial verification

**Files:** none (verification only) — record results in the session report.

- [ ] **Step 1: Run the whole test suite**

Run: `pytest sessions/_compiler/tests/ -v`
Expected: all pass (widgets, compile, idempotency, concept gates, V8 regression).

- [ ] **Step 2: Recompile Day 2 + confirm all gates green**

Run: `python3 sessions/_compiler/compile_lesson.py sessions/m02-the-neuron/day-02-activations/source.md`
Expected: exit 0, all gate blocks PASS/N-A.

- [ ] **Step 3: LLM-judge (via the local bridge)**

Write a short script that sends the compiled `lesson.html` (stripped to visible text + a list of which section each `<svg>`/iframe sits in) to the bridge (`http://localhost:11211`, model `aws:anthropic.claude-opus-4-8`, no `temperature`) with the prompt: *"For each concept in this lesson, does its picture appear inline in the same section where the concept is introduced? List any concept named without a visual, or any picture deferred to a later section. Answer PASS only if every concept shows its own picture in-place."* Record the verdict + reasons. If FAIL, fix `source.md` and re-run from Step 2. Degrade gracefully if the bridge is down (note it, rely on gates).

- [ ] **Step 4: Adversarial verification (independent agents, read-only)**

Dispatch 3-4 independent agents to confirm, each returning a structured verdict:
  1. In the compiled `lesson.html`, each of the 7 concepts (c1..c7) contains its own `<svg` or `.build-embed` iframe, in order, before the quiz. Report any concept missing a visual.
  2. V9 invariants hold and recompile is byte-identical (run compile twice, diff).
  3. Shipped m02 Days 1/3-9 and all m03 days recompile byte-identical to their committed `lesson.html` (no regression).
  4. The gates have teeth: strip one concept's visual → `concept_shell_gate` and `reader_flow_gate` both FAIL (exit non-zero).

- [ ] **Step 5: Write the session report + update memory**

Create `sessions/v9_concept_shell_report.md` (Scope, what changed, gate evidence, LLM-judge verdict, adversarial results, what's deferred: m02 Days 1/3-9 + m03 backport + m04 resume). Update the memory index note about V9.

- [ ] **Step 6: Commit**

```bash
git add sessions/v9_concept_shell_report.md
git commit -m "docs(v9): session report — concept shell shipped, Day 2 rebuilt, gates verified"
```

---

## Sequencing note

Tasks 1-5 (widgets/blocks) are independent and precede Task 6. **Task 8 (donor) must reach at least its Step 3 stub before Task 6's tests run** — build the donor stub first, then the compile path, then the donor's JS engine. Tasks 9-11 (gates + dispatch) follow the compiler. Task 12 (regression) can run any time after Task 11. Task 13 (real Day 2) needs everything before it green. Tasks 14-15 close out. Commit after every task; keep V8 output byte-identical throughout (Task 12 is the tripwire).
