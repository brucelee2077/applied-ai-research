import os, sys, subprocess
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
def test_v8_day_shipped_output_unchanged(tmp_path):
    src = os.path.join(ROOT, 'sessions/m02-the-neuron/day-03-layers-forward-pass/source.md')
    shipped = os.path.join(ROOT, 'sessions/m02-the-neuron/day-03-layers-forward-pass/lesson.html')
    out = tmp_path / 'd03.html'
    r = subprocess.run(['python3', os.path.join(HERE, '..', 'compile_lesson.py'), src, '--out', str(out), '--quiet'], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.read_text(encoding='utf-8') == open(shipped, encoding='utf-8').read()
