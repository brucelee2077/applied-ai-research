import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import concept_structure_gate as g

# A concept whose build-up (after its opening visual) is ONLY a %%% steps block, with no
# surrounding prose. The gate strips all %%% widgets before measuring build-up prose, so
# before the fix this FAILS "has build-up after its visual". A %%% steps block IS
# substantial build-up content, so it must satisfy the build-up floor.
_SRC = """---
quest_id: t
mode: concept
---

@@@ hero
@lede A ruler is straight; stack two and it stays straight — that is the bend puzzle.
@goal understand the bend.

@@@ concept id=c1 tag="A" title="Straight plus straight" gotit="Got it"
Intro prose for concept one, comfortably over forty characters long to clear the floor.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="a"><path d="M0 0 L10 10"/></svg>
%%%
Build-up prose for concept one, also well over forty characters, plenty of narration here.

@@@ concept id=c2 tag="B" title="The collapse" gotit="Got it"
Intro prose for concept two, again comfortably over the forty character build floor here.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="b"><path d="M0 10 L10 0"/></svg>
%%%
%%% steps
step: (x·W1)·W2
why: two matmuls back to back
step: = x·(W1·W2)
why: they collapse into one — no bend between them
%%%

@@@ concept id=c3 tag="Recap" title="Today in one page" gotit="Got it"
Build-up prose recap for concept three, also well over forty characters of real narration.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="c"><path d="M0 8 L8 2"/></svg>
%%%
The bend is what makes depth matter — come back to this any time you forget it later on.
"""


def test_steps_only_buildup_satisfies_floor():
    ok, msgs = g.run(_SRC)
    # c2's build-up is only a %%% steps widget; it must still pass the build-up floor.
    assert ok is True, 'unexpected FAIL msgs: %s' % [m for m in msgs if m.startswith('FAIL')]
    assert not any('c2 has build-up' in m and m.startswith('FAIL') for m in msgs)
