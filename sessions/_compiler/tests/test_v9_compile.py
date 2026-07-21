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
    assert html.count('class="module-section"') == 6
    for cid in ('c1','c2','c3','c4','quiz','produce'):
        assert 'id="%s"' % cid in html
    assert html.count('class="gotit"') == 6
    assert '<!--V9_CONTENT-->' not in html
    assert '<!--V9_NAV-->' not in html
    assert '__QUEST_ID__' not in html
    import re
    targets = set(re.findall(r'data-target="([^"]+)"', html))
    assert targets == {'home','c1','c2','c3','c4','quiz','produce'}

def test_concept_mode_shows_visual_in_every_concept():
    html, _ = _compile_mini()
    import re
    for cid in ('c1','c2','c3'):
        sec = re.search(r'id="%s".*?</section>' % cid, html, re.DOTALL).group(0)
        assert ('<svg' in sec) or ('build-embed' in sec), '%s has no visual' % cid

def test_concept_compile_is_idempotent():
    a, _ = _compile_mini(); b, _ = _compile_mini()
    assert a == b

def test_concept_emits_js_wiring_contract():
    """Lock the compiler-emitted DOM the generic donor JS depends on."""
    html, _ = _compile_mini()
    import re
    # every quiz .q carries a numeric data-correct and a .q-fb[data-fb]
    qs = re.findall(r'<div class="q" data-correct="(\d+)">.*?data-fb="', html, re.DOTALL)
    assert len(qs) == 4, qs
    # the demo carries the run button + hidden out/take the engine toggles
    assert 'class="demo-run"' in html
    assert 'class="demo-out"' in html
    assert 'class="demo-take"' in html


import subprocess
def test_compile_lesson_cli_concept_mode_passes(tmp_path):
    src = os.path.join(HERE, 'fixtures', 'mini_concept.md')
    out = tmp_path / 'mini.html'
    r = subprocess.run(['python3', os.path.join(HERE, '..', 'compile_lesson.py'), src,
                        '--donor', os.path.join(HERE, '..', 'shells', 'v9-base.donor'),
                        '--out', str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'concept shell' in r.stdout.lower() or 'concept_shell' in r.stdout.lower()
    assert out.exists()
