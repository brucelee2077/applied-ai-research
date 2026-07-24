import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import v8lib


def test_insight_renders_rehook_callout():
    html = v8lib.render_widget('insight', {}, [
        'This is why it **matters**: the [[net||the model]] runs but does nothing.'])
    assert 'class="takeaway"' in html          # reuses the styled .takeaway block (no donor CSS change)
    assert '💡' in html                         # the re-hook glyph, inline
    assert '<strong>matters</strong>' in html  # inline() bold ran
    assert 'class="term"' in html               # [[term||gloss]] glossary ran
    assert 'the model' in html                  # the gloss text is the tooltip


def test_insight_multiline_body_joins():
    html = v8lib.render_widget('insight', {}, [
        'Notice this:',
        'the collapse is silent — the net trains but learns nothing.'])
    assert 'Notice this' in html and 'silent' in html
    assert html.count('class="takeaway"') == 1
