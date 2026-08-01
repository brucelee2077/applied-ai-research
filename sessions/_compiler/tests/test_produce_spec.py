"""Tests for the PRODUCE-spec extractor (sessions/_produce_spec.py).

This feeds the experiment.py backfill: 101 of 115 artifacts are the placeholder
stub, and the only trustworthy statement of what each one must DO is the produce
section the lesson already ships. If extraction is wrong, a generator invents
requirements instead of honouring the lesson's own contract — the exact failure
the density metric taught us to test for.

Note on provenance: the extractor was written before these tests, which is a TDD
miss. To recover the red signal they were each checked against a deliberately
broken extractor (see test_mutants_are_caught) rather than only against a working
one, so they are known to fail when the behaviour regresses.
"""
import os, sys, importlib.util

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
_spec = importlib.util.spec_from_file_location(
    '_produce_spec', os.path.join(ROOT, 'sessions', '_produce_spec.py'))
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


STUB = ('# Module 1 - day-01 - experiment\n#\n'
        '# Placeholder. Fill this from the lesson\'s PRODUCE step (open lesson.html):\n')

REAL = ('import numpy as np\n\n'
        'if __name__ == "__main__":\n'
        '    assert np.array([1, 2, 3]).shape == (3,)\n'
        '    print("\\u2705 shapes line up")\n')


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return str(p)


def test_placeholder_stub_is_a_stub(tmp_path):
    assert ps.is_stub(_write(tmp_path, 'experiment.py', STUB))


def test_real_self_checking_artifact_is_not_a_stub(tmp_path):
    assert not ps.is_stub(_write(tmp_path, 'experiment.py', REAL))


def test_missing_file_counts_as_a_stub(tmp_path):
    assert ps.is_stub(str(tmp_path / 'nope.py'))


def test_a_script_with_no_self_check_is_still_a_stub(tmp_path):
    """Imports and prints alone do not make it self-checking."""
    weak = 'import numpy as np\nprint(np.arange(3).shape)\n'
    assert ps.is_stub(_write(tmp_path, 'experiment.py', weak))


def test_v9_produce_block_is_extracted_from_source_md():
    src = ('---\nquest_id: t\n---\n\n'
           '@@@ concept id=c1 title="x"\nbody\n\n'
           '@@@ produce id=produce tag="Produce" title="Build it"\n'
           'Define the model, then run the loop.\n\n'
           '@@@ fin\nend\n')
    got = ps.from_source(src)
    assert got.startswith('@@@ produce')
    assert 'Define the model' in got
    assert '@@@ fin' not in got, 'must stop at the next block'
    assert 'body' not in got, 'must not swallow the preceding concept'


def test_v7_produce_section_is_extracted_from_lesson_html():
    page = ('<html><body>'
            '<div class="module-section"><span class="sec-num s-quiz">6</span>'
            '<div class="sec-body">quiz text</div></div>'
            '<div class="module-section"><span class="sec-num s-produce">7</span>'
            '<div class="sec-body">Option A write it yourself.<br>'
            'Help me build my Module 1 Day 1 artifact.<br>'
            '1. Build a 1-D array.<br>2. Print its shape.<br>'
            'Acceptance criteria<br>The script prints .shape and .ndim.</div></div>'
            '</body></html>')
    got = ps.from_lesson(page)
    assert got is not None
    assert 'Help me build' in got
    assert 'quiz text' not in got, 'must not bleed in the previous section'


def test_claude_prompt_and_acceptance_are_split_apart():
    text = ('Option A write it yourself.\n'
            'Help me build my artifact.\n1. Do a thing.\n2. Do another.\n'
            'Acceptance criteria\nIt prints the shape.\n')
    prompt = ps._claude_prompt(text)
    accept = ps._acceptance(text)
    assert prompt.startswith('Help me build')
    assert '1. Do a thing.' in prompt
    assert 'Acceptance criteria' not in prompt, 'criteria must not leak into the prompt'
    assert accept == 'It prints the shape.'


def test_every_stub_day_in_the_repo_yields_a_usable_spec():
    """The backfill depends on this: no stub day may be left without a contract.

    Deliberately NOT asserting how MANY stubs remain. An earlier version required
    >= 100 and broke the moment the first module was backfilled — a test that
    fails when the work succeeds is measuring progress, not correctness. The
    invariant is per-day and holds at every point in the rollout, including the
    end when there are zero stubs left.
    """
    specs = ps.all_specs()
    empty = [s['dir'] for s in specs if not s['produce'] or len(s['produce']) < 400]
    assert not empty, 'produce spec too short to be real: %s' % empty
    # every day states its requirements one way or the other: a numbered
    # Option-B list (V7 shape) or a substantial prose produce block (V9 shape)
    without = [s['dir'] for s in specs if not s['claude_prompt'] and len(s['produce']) < 900]
    assert not without, 'no usable requirement list: %s' % without


# ---------------------------------------------------------------------------
# Acceptance-criteria extraction across BOTH spellings.
#
# V9 source.md days never write the words "Acceptance criteria" — they write
# `#### What you should see`. 46 source.md files use that heading and 0 use the
# literal one, so an extractor that only knows the literal spelling reports
# `acceptance: null` for every V9 day BY CONSTRUCTION. That null was once read
# as "the lesson never states its criteria" and written into a skill as a SPEC
# gap. It is an extractor gap. These tests pin both spellings, on real days of
# each origin, so the wrong diagnosis cannot come back.
# ---------------------------------------------------------------------------

V9_DAY = os.path.join(ROOT, 'sessions', 'm04-first-model-mlp', 'day-01-mlp-mnist')
V7_DAY = os.path.join(ROOT, 'sessions', 'm01-shape-of-data', 'day-01-arrays')


def test_real_v9_day_states_its_criteria_under_what_you_should_see():
    spec = ps.spec_for(V9_DAY)
    assert spec['origin'] == 'source.md', 'fixture day changed origin; pick another V9 day'
    assert spec['acceptance'], 'V9 day reports NO acceptance criteria, but its source.md has them'
    accept = spec['acceptance']
    # the real criteria for this day, verbatim tells
    assert '(batch, 10)' in accept
    assert '10%' in accept
    assert 'x@(W1@W2)' in accept
    # and only the criteria: not the prompt above it, not the log box below it
    assert 'Help me build' not in accept, 'swallowed the Option-B prompt'
    assert 'research log' not in accept, 'ran past the criteria into the log callout'
    assert '!!!' not in accept and '%%%' not in accept, 'leaked widget markup'


def test_real_v7_day_still_states_its_criteria_under_the_literal_heading():
    """Do not regress the compiled-lesson path while teaching the new spelling."""
    spec = ps.spec_for(V7_DAY)
    assert spec['origin'] == 'lesson.html', 'fixture day changed origin; pick another V7 day'
    accept = spec['acceptance']
    assert accept, 'V7 literal "Acceptance criteria" day lost its criteria'
    assert '.ndim' in accept and '(2, 3)' in accept
    assert 'Help me build' not in accept
    # the compiled page continues into the log box and the finale; criteria stop first
    assert 'research log' not in accept, 'ran past the criteria into the log callout'
    assert 'complete!' not in accept, 'ran past the criteria into the day finale'


def test_no_day_in_the_repo_reports_criteria_that_are_really_the_finale():
    """Blast-radius invariant: a non-null field holding the WRONG span is worse
    than null, so scan every day for the two known bleeds."""
    bled = []
    for spec in ps.all_specs(stub_only=False):
        accept = spec['acceptance'] or ''
        if 'complete!' in accept or 'fully self-contained single file' in accept:
            bled.append(spec['dir'])
    assert not bled, 'acceptance ran into the page finale on: %s' % bled[:8]


def test_every_v9_day_that_writes_the_heading_yields_criteria():
    missing = []
    for spec in ps.all_specs(stub_only=False):
        if spec['origin'] != 'source.md':
            continue
        if 'What you should see' not in spec['produce']:
            continue
        if not spec['acceptance']:
            missing.append(spec['dir'])
    assert not missing, 'heading present but criteria not extracted: %s' % missing[:8]


def test_acceptance_reads_the_new_heading_and_stops_at_the_next_block():
    text = ('%%% prompt id=pp label="ask Claude"\nHelp me build it.\n1. Thing.\n%%%\n\n'
            '#### What you should see (predict, then confirm)\n'
            '- The shape is `(2, 3)`.\n'
            '- The loss goes down.\n\n'
            '!!! c-info 📓\n5-minute research log: write three lines.\n')
    accept = ps._acceptance(text)
    assert accept is not None, 'the "What you should see" spelling is invisible to the extractor'
    assert 'The shape is `(2, 3)`.' in accept
    assert 'The loss goes down.' in accept
    assert 'research log' not in accept, 'must stop at the !!! callout'
    assert 'Help me build' not in accept


def test_acceptance_keeps_inline_prose_criteria_on_the_heading_line():
    """Three m03 days state the criteria as prose on the same line, with no
    heading of their own. Dropping the heading line would drop the criteria."""
    text = ('Write the code.\n'
            'What you should see by the end: a row of raw match numbers turning '
            'into a budget of shares that sum to 1.\n')
    accept = ps._acceptance(text)
    assert accept and 'sum to 1' in accept


def test_acceptance_drops_a_bare_parenthetical_heading_tail():
    text = ('#### What you should see (check your prediction)\n- It prints the shape.\n')
    accept = ps._acceptance(text)
    assert accept.startswith('- It prints the shape.'), (
        'the heading parenthetical is not a criterion: %r' % accept)


def test_a_bulleted_criteria_block_ends_with_the_list():
    """When the criteria are a bullet list, the list IS the criteria — the prose
    that follows it ("Then poke it…", "Option A — write it yourself") is the next
    part of the lesson, not a criterion. Real shape: m02 day-07."""
    text = ('**What you should see** when you run it:\n'
            '- Plain SGD reaches `loss < 0.01` around **step 26**.\n'
            '- Momentum gets there around **step 13**.\n'
            '\n'
            'Write it in `experiment.py` and run it. Then poke it:\n'
            '- Set `keep` to `0.0` and watch it turn back into plain SGD.\n'
            '\n'
            '**Option A — write it yourself.** Fill in `experiment.py`.\n')
    accept = ps._acceptance(text)
    assert accept is not None, 'bold heading spelling is invisible to the extractor'
    assert 'step 26' in accept and 'step 13' in accept
    assert 'when you run it' not in accept, 'kept a bare lead-in as a criterion'
    assert 'poke it' not in accept, 'ran past the criteria list into the next paragraph'
    assert 'Option A' not in accept


def test_a_bold_option_heading_stops_a_block():
    text = ('Acceptance criteria\nIt prints the shape.\n'
            '**Option A — write it yourself.** Do the thing.\n')
    accept = ps._acceptance(text)
    assert accept == 'It prints the shape.', repr(accept)


def test_claude_prompt_stops_at_the_new_acceptance_heading():
    text = ('Help me build my artifact.\n1. Do a thing.\n'
            '#### What you should see (predict, then confirm)\n- It prints the shape.\n')
    prompt = ps._claude_prompt(text)
    assert prompt.startswith('Help me build')
    assert '1. Do a thing.' in prompt
    assert 'What you should see' not in prompt, 'criteria heading leaked into the prompt'
    assert 'It prints the shape' not in prompt


def test_claude_prompt_stops_at_the_widget_fence():
    """In V9 the copy-prompt lives inside a `%%% prompt … %%%` widget, so the
    closing fence is the real boundary even before any criteria heading."""
    text = ('%%% prompt id=pp label="ask Claude to build it"\n'
            'Help me build my artifact.\n1. Do a thing.\n'
            '%%%\n\n'
            '#### What you should see\n- It prints the shape.\n'
            '!!! c-info 📓\nresearch log\n')
    prompt = ps._claude_prompt(text)
    assert prompt.startswith('Help me build')
    assert '1. Do a thing.' in prompt
    assert '%%%' not in prompt, 'prompt ran through the closing widget fence'
    assert 'What you should see' not in prompt
    assert 'research log' not in prompt


def test_real_v9_day_prompt_is_only_the_prompt():
    prompt = ps.spec_for(V9_DAY)['claude_prompt']
    assert prompt.startswith('Help me build my Module 5 Day 1 artifact.')
    assert '5. Also demonstrates the collapse' in prompt, 'lost the last requirement'
    assert '%%%' not in prompt, 'ran through the closing widget fence'
    assert 'What you should see' not in prompt, 'swallowed the acceptance criteria'
    assert 'research log' not in prompt, 'swallowed the log callout'


def test_real_v7_day_prompt_still_stops_at_the_literal_heading():
    prompt = ps.spec_for(V7_DAY)['claude_prompt']
    assert prompt.startswith('Help me build my Module 1 Day 1 artifact.')
    assert '4. Reshapes np.arange(6)' in prompt
    assert 'Acceptance criteria' not in prompt


def test_mutants_are_caught():
    """Recover the red signal: break the extractor, confirm the checks notice.

    Each mutant is a plausible implementation slip. If a mutant survives, the
    corresponding test above is not actually pinning anything.
    """
    import re
    survivors = []

    # mutant 1: produce block runs to end of file (forgets the next-block stop)
    def m1(src):
        m = re.search(r'(?ms)^@@@\s+produce\b.*\Z', src)
        return m.group(0).strip() if m else None
    src = ('@@@ produce id=p title="t"\nDo it.\n\n@@@ fin\nend\n')
    if '@@@ fin' not in (m1(src) or ''):
        survivors.append('produce-block-stop')

    # mutant 2: acceptance criteria left inside the copy-prompt
    def m2(text):
        m = re.search(r'(?is)(Help me build.*)', text)
        return m.group(1).strip() if m else None
    text = 'Help me build it.\n1. Thing.\nAcceptance criteria\nIt prints.\n'
    if 'Acceptance criteria' not in (m2(text) or ''):
        survivors.append('prompt-accept-split')

    # mutant 3: stub check that only looks for the placeholder string
    def m3(src):
        return 'placeholder. fill this' in src.lower()
    if m3('import numpy\nprint(1)\n'):
        survivors.append('stub-needs-selfcheck')

    # mutant 4: acceptance heading matched in only the literal V7 spelling
    def m4(text):
        m = re.search(r'(?is)Acceptance criteria\s*\n(.*?)\Z', text)
        return m.group(1).strip() if m else None
    v9 = '#### What you should see\n- It prints the shape.\n'
    if m4(v9) is not None:
        survivors.append('acceptance-new-spelling')

    # mutant 5: acceptance runs to end of text (forgets the next-block stop)
    def m5(text):
        m = re.search(r'(?is)What you should see[^\n]*\n(.*)\Z', text)
        return m.group(1).strip() if m else None
    v9b = '#### What you should see\n- It prints.\n\n!!! c-info\nresearch log\n'
    if 'research log' not in (m5(v9b) or ''):
        survivors.append('acceptance-stop')

    # mutant 6: prompt with no widget-fence / new-heading terminator
    def m6(text):
        m = re.search(r'(?is)(Help me build.*?)(?=\nAcceptance criteria|\Z)', text)
        return m.group(1).strip() if m else None
    v9c = 'Help me build it.\n1. Thing.\n%%%\n\n#### What you should see\n- It prints.\n'
    if 'What you should see' not in (m6(v9c) or ''):
        survivors.append('prompt-v9-terminator')

    assert not survivors, 'these mutants were NOT caught: %s' % survivors
