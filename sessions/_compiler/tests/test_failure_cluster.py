import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import concept_structure_gate as g

_FM = "---\nquest_id: t\n---\n"


def _mk(concepts):
    """concepts: list of (title, has_play). Build minimal valid concept blocks
    (intro -> svg -> build-up) so only the failure-cluster advisory is under test."""
    parts = [_FM]
    for i, (title, play) in enumerate(concepts, 1):
        b = ('@@@ concept id=c%d title="%s"\n' % (i, title)
             + 'Intro prose that is comfortably longer than the forty-char minimum here.\n'
             + '%%% svg\n<svg><text x="1" y="1">x</text></svg>\n%%%\n'
             + 'Build-up prose that is comfortably longer than the forty-char minimum here.\n')
        if play:
            b += '%%% demo\ncode: 2+2\nout: 4\ntake: a takeaway line long enough to be real prose here.\n%%%\n'
        parts.append(b)
    return '\n'.join(parts)


def _cluster_warns(concepts):
    ok, msgs = g.run(_mk(concepts))
    return ok, [m for m in msgs if 'failure-mode cluster' in m]


def test_three_consecutive_failures_warn():
    ok, w = _cluster_warns([("Meet ReLU", False), ("dead ReLU trap", False),
                            ("vanishing gradient", False), ("exploding gradient", False), ("Recap", False)])
    assert w, "expected a failure-cluster warn"
    assert ok is True, "advisory must NOT fail the gate"


def test_interleaved_play_breaks_cluster():
    ok, w = _cluster_warns([("Meet ReLU", False), ("dead ReLU trap", False),
                            ("vanishing gradient", True),   # this failure unit HAS play
                            ("exploding gradient", False), ("Recap", False)])
    assert not w, "a play widget mid-run must break the cluster"


def test_two_failures_no_warn():
    ok, w = _cluster_warns([("Meet ReLU", False), ("dead ReLU trap", False),
                            ("vanishing gradient", False), ("Recap", False)])
    assert not w


def test_benign_titles_no_false_positive():
    # 'problem' is not a failure token; only 'limit' matches -> 1 consecutive, not a cluster.
    ok, w = _cluster_warns([("The problem with one line", False),
                            ("The limit of one neuron", False), ("Recap", False)])
    assert not w
