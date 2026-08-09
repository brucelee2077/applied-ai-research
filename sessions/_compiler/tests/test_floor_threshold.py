import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'gates'))
import coverage_judge as cj


def _d(*verdicts):
    return {'dimensions': [{'name': 'l%d' % i, 'verdict': v} for i, v in enumerate(verdicts)]}


# The threshold is arithmetic over the model's per-lever judgments. The model itself
# does NOT apply it reliably: 6 of 20 m02-m04 days returned FLOOR_MET from the interest
# floor while carrying >=2 WEAK levers, and the plain-language floor did the same on its
# very first run (3 WEAK -> FLOOR_MET). So it lives in code and is tested here.
def test_all_good_meets_floor():
    d = _d('GOOD', 'GOOD', 'GOOD'); d['overall'] = 'FLOOR_MET'
    assert cj._floor_from_levers(d)[0] == 'FLOOR_MET'


def test_one_weak_is_tolerated():
    d = _d('GOOD', 'WEAK', 'GOOD'); d['overall'] = 'FLOOR_MET'
    assert cj._floor_from_levers(d)[0] == 'FLOOR_MET'


def test_two_weak_is_below_floor_even_if_model_says_met():
    # The exact corpus bug: model says FLOOR_MET, two levers are WEAK.
    d = _d('WEAK', 'WEAK', 'GOOD'); d['overall'] = 'FLOOR_MET'
    verdict, weak, missing, stated = cj._floor_from_levers(d)
    assert verdict == 'BELOW_FLOOR'
    assert weak == 2 and missing == 0
    assert stated == 'FLOOR_MET'      # the disagreement stays visible


def test_any_missing_is_below_floor_even_if_model_says_met():
    d = _d('GOOD', 'GOOD', 'MISSING'); d['overall'] = 'FLOOR_MET'
    assert cj._floor_from_levers(d)[0] == 'BELOW_FLOOR'


def test_garbled_overall_fails_safe():
    d = _d('GOOD', 'GOOD'); d['overall'] = ''
    assert cj._floor_from_levers(d)[0] == 'FLOOR_MET'   # levers are clean -> pass on levers
    d2 = {'dimensions': [], 'overall': ''}
    assert cj._floor_from_levers(d2)[0] == 'BELOW_FLOOR'  # no levers + no word -> never a silent pass


def test_model_downgrade_is_respected_via_levers():
    # If the model grades levers badly, we go BELOW even when it forgot to say so.
    d = _d('MISSING', 'WEAK'); d['overall'] = 'FLOOR_MET'
    assert cj._floor_from_levers(d)[0] == 'BELOW_FLOOR'


def test_plain_language_judge_shape_on_bridge_failure():
    # Never raises, and never reports a pass when it could not run.
    out = cj.judge_plain_language_absolute('x', model='definitely-not-a-model', timeout=1)
    assert out['status'] in ('BRIDGE_UNAVAILABLE', 'PARSE_ERROR', 'OK')
    if out['status'] != 'OK':
        assert out['overall'] == 'N/A'
