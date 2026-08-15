import os, sys, shutil, subprocess
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import compile_lesson

COMPILER = os.path.join(HERE, '..', 'compile_lesson.py')
FIXTURE = os.path.join(HERE, 'fixtures', 'mini_concept.md')

SENTINEL = '<!-- PREVIOUS GOOD PAGE — MUST SURVIVE A FAILED COMPILE -->'


def _run(src, out, *extra):
    return subprocess.run(['python3', COMPILER, str(src), '--out', str(out), '--quiet', *extra],
                          capture_output=True, text=True)


# =============================================================================
# The write used to happen ABOVE the exit-3/4 checks, so a gate failure left a
# lesson.html on disk that had already been judged broken. Recovering meant
# knowing to `git checkout` the file, and a sweep over all 47 lessons could put
# a batch of them in that state at once.
# =============================================================================

def _source_that_fails_only_an_output_gate():
    """A source that PASSES the reader-flow gate but FAILS the concept shell gate.

    Picking the mutation carefully is the whole point of this file. Stripping the
    `%%% svg` blocks (the obvious break) fails reader_flow_gate with exit 2, and
    exit 2 already blocked the write before this change — so that mutation proves
    nothing about the reordering. reader_flow_gate never counts quiz questions;
    concept_shell_gate requires exactly 4. Dropping one question therefore lands
    in the narrow window this change is about: source fine, output rejected.
    """
    text = open(FIXTURE, encoding='utf-8').read()
    lines = text.split('\n')
    qs = [i for i, ln in enumerate(lines) if ln.startswith('q: ')]
    assert len(qs) == 4, 'fixture no longer has 4 quiz questions: %d' % len(qs)
    del lines[qs[-1]]
    return '\n'.join(lines)


def test_a_failing_gate_leaves_the_previous_page_untouched(tmp_path):
    src = tmp_path / 'source.md'
    src.write_text(_source_that_fails_only_an_output_gate(), encoding='utf-8')

    out = tmp_path / 'lesson.html'
    out.write_text(SENTINEL, encoding='utf-8')

    r = _run(src, out)
    assert r.returncode == 3, 'expected concept-shell-gate exit 3, got %d\n%s' % (r.returncode, r.stdout + r.stderr)
    assert out.read_text(encoding='utf-8') == SENTINEL, 'the failed compile overwrote a good page'


def test_a_failing_gate_writes_no_file_at_all_when_none_existed(tmp_path):
    src = tmp_path / 'source.md'
    src.write_text(_source_that_fails_only_an_output_gate(), encoding='utf-8')

    out = tmp_path / 'lesson.html'
    r = _run(src, out)
    assert r.returncode == 3
    assert not out.exists(), 'a failed compile created a broken page'


def test_the_reader_flow_gate_path_still_blocks_the_write(tmp_path):
    # The pre-existing exit-2 behaviour, pinned so the reorder cannot regress it.
    import re
    src = tmp_path / 'source.md'
    src.write_text(re.sub(r'(?ms)^%%%\s+svg\s*\n.*?^%%%\s*$', '',
                          open(FIXTURE, encoding='utf-8').read()), encoding='utf-8')
    out = tmp_path / 'lesson.html'
    out.write_text(SENTINEL, encoding='utf-8')
    r = _run(src, out)
    assert r.returncode == 2, r.stdout + r.stderr
    assert out.read_text(encoding='utf-8') == SENTINEL


def test_a_failing_gate_leaves_no_tmp_litter(tmp_path):
    # publish_pages classify_untracked fails CLOSED: an unrecognised path aborts
    # the publish. A stray lesson.html.tmp would do exactly that.
    src = tmp_path / 'source.md'
    src.write_text(_source_that_fails_only_an_output_gate(), encoding='utf-8')
    out = tmp_path / 'lesson.html'
    _run(src, out)
    assert list(tmp_path.glob('*.tmp')) == []


def test_a_passing_compile_still_writes(tmp_path):
    src = tmp_path / 'source.md'
    shutil.copy(FIXTURE, src)
    out = tmp_path / 'lesson.html'
    out.write_text(SENTINEL, encoding='utf-8')

    r = _run(src, out)
    assert r.returncode == 0, r.stdout + r.stderr
    body = out.read_text(encoding='utf-8')
    assert SENTINEL not in body
    assert body.lstrip().startswith('<!DOCTYPE html>')
    assert list(tmp_path.glob('*.tmp')) == []


def test_check_only_never_writes(tmp_path):
    src = tmp_path / 'source.md'
    shutil.copy(FIXTURE, src)
    out = tmp_path / 'lesson.html'
    r = _run(src, out, '--check-only')
    assert r.returncode == 0, r.stdout + r.stderr
    assert not out.exists()


# =============================================================================
# atomic_write itself
# =============================================================================

def test_atomic_write_replaces_content(tmp_path):
    p = tmp_path / 'f.html'
    p.write_text('old', encoding='utf-8')
    compile_lesson.atomic_write(str(p), 'new — 中文也要能写')
    assert p.read_text(encoding='utf-8') == 'new — 中文也要能写'
    assert list(tmp_path.glob('*.tmp')) == []


def test_atomic_write_cleans_up_its_temp_file_on_failure(tmp_path):
    p = tmp_path / 'f.html'
    p.write_text('old', encoding='utf-8')
    # An object that raises partway through being written.
    class Boom:
        def __str__(self):
            raise RuntimeError('boom')
    import pytest
    with pytest.raises(Exception):
        compile_lesson.atomic_write(str(p), Boom())     # f.write(non-str) raises
    assert p.read_text(encoding='utf-8') == 'old', 'the original was damaged'
    assert list(tmp_path.glob('*.tmp')) == [], 'temp file left behind'
