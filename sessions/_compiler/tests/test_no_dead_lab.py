import os, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def test_no_dead_frontier_experiment_lab_in_learner_files():
    """The /frontier-experiment-lab skill is not installed. It must not appear in any
    learner-facing or source file (lesson.html, experiment.py, source.md, .donor)."""
    out = subprocess.run(
        ['grep', '-rl', 'frontier-experiment-lab', 'sessions',
         '--include=lesson.html', '--include=experiment.py',
         '--include=source.md', '--include=*.donor'],
        cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert out == '', 'dead /frontier-experiment-lab path remains in:\n' + out
