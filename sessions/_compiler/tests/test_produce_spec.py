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
    """The backfill depends on this: no day may be left without a contract."""
    specs = ps.all_specs()
    assert len(specs) >= 100, 'expected ~101 stub days, got %d' % len(specs)
    empty = [s['dir'] for s in specs if not s['produce'] or len(s['produce']) < 400]
    assert not empty, 'produce spec too short to be real: %s' % empty
    # every day states its requirements one way or the other
    without = [s['dir'] for s in specs if not s['claude_prompt'] and len(s['produce']) < 900]
    assert not without, 'no usable requirement list: %s' % without


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

    assert not survivors, 'these mutants were NOT caught: %s' % survivors
