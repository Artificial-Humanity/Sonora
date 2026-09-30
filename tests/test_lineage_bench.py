"""The pure half of the lineage hum bench (`scripts/lib/lineage_bench.py`).

The bench renders one voice set through each checkpoint of Sonora's warm-start chain to find
where the hum our training added (baseline bench, 2026-09-30) came in. These pin the choices
the listening result rests on: which voices, which recording each checkpoint speaks, which
side each clip is served on, and how the ratings become the pre-registered reading.
"""

import random

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import lineage_bench as lb  # noqa: E402

CKPTS = ["derisk", "vat3", "vat5", "vat6", "vat7"]


def test_spread_takes_evenly_spaced_voices_over_hnr():
    pool = {10: 1.0, 11: 2.0, 12: 3.0, 13: 4.0, 14: 5.0}
    assert lb.spread(pool, 3) == [10, 12, 14]


def test_spread_refuses_a_pool_too_small():
    with pytest.raises(SystemExit):
        lb.spread({1: 1.0, 2: 2.0}, 3)


def test_every_checkpoint_gets_its_own_recording_of_each_voice():
    clips = {1: [(w, 5.0) for w in "abcdef"], 2: [(w, 5.0) for w in "ghijk"]}
    got = lb.assign_clips(clips, CKPTS, random.Random(1), target=5.0)
    for spk in clips:
        assert sorted(got[spk]) == sorted(CKPTS)
        assert len(set(got[spk].values())) == len(CKPTS)
        assert set(got[spk].values()) <= {w for w, _s in clips[spk]}


def test_a_voices_recordings_are_the_ones_nearest_the_target_length():
    # lengths must not differ by checkpoint: more seconds is more exposure to the hum
    clips = {1: [("a", 3.6), ("b", 4.9), ("c", 5.0), ("d", 5.1), ("e", 5.2), ("f", 4.8),
                 ("g", 7.9)]}
    got = lb.assign_clips(clips, CKPTS, random.Random(1), target=5.0)
    assert set(got[1].values()) == {"b", "c", "d", "e", "f"}


def test_assign_refuses_a_voice_with_too_few_recordings():
    with pytest.raises(SystemExit, match="7"):
        lb.assign_clips({7: [("a", 5.0), ("b", 5.0)]}, CKPTS, random.Random(1), target=5.0)


def test_sides_balance_per_checkpoint_and_hold_constant_along_the_chain():
    # constant per voice: an A/B lean then cancels exactly in every step difference
    spks = list(range(8))
    sw = lb.side_swaps(spks, CKPTS, random.Random(3))
    for c in CKPTS:
        assert sum(sw[(s, c)] for s in spks) == 4
    for s in spks:
        assert len({sw[(s, c)] for c in CKPTS}) == 1


def _gaps(per_ckpt):
    """{ckpt: [gap per voice]} -> {ckpt: {voice: gap}}."""
    return {c: dict(enumerate(v)) for c, v in per_ckpt.items()}


def test_a_jump_at_one_step_is_named():
    g = _gaps({"derisk": [0] * 8, "vat3": [0] * 8, "vat5": [2] * 8, "vat6": [2] * 8,
               "vat7": [2] * 8})
    r = lb.reading(g, CKPTS)
    assert r["outcome"] == "step" and r["rises"] == [("vat3", "vat5")]


def test_hum_already_present_at_the_first_checkpoint_with_no_rise_after():
    g = _gaps({c: [2] * 8 for c in CKPTS})
    r = lb.reading(g, CKPTS)
    assert r["outcome"] == "present_at_first" and r["present_at_first"]


def test_a_step_and_hum_at_the_first_checkpoint_are_both_reported():
    g = _gaps({"derisk": [1] * 8, "vat3": [1] * 8, "vat5": [3] * 8, "vat6": [3] * 8,
               "vat7": [3] * 8})
    r = lb.reading(g, CKPTS)
    assert r["outcome"] == "step" and r["present_at_first"]


def test_a_significant_fall_is_reported():
    g = _gaps({"derisk": [0] * 8, "vat3": [2] * 8, "vat5": [0] * 8, "vat6": [2] * 8,
               "vat7": [2] * 8})
    r = lb.reading(g, CKPTS)
    assert r["falls"] == [("vat3", "vat5")]
    assert ("derisk", "vat3") in r["rises"]


def test_nothing_shown_is_inconclusive():
    # vat7 reproduces its hum, but half the voices hum from derisk on and the other half
    # only at vat7: no step, no span and no derisk level reaches significance with n = 8
    half = [1, 1, 1, 1, 0, 0, 0, 0]
    g = _gaps({"derisk": half, "vat3": half, "vat5": half, "vat6": half, "vat7": [1] * 8})
    r = lb.reading(g, CKPTS)
    assert r["rises"] == [] and not r["present_at_first"]
    assert r["outcome"] == "inconclusive"


def test_no_vat7_ratings_is_invalid():
    g = _gaps({c: [1] * 8 for c in CKPTS[:-1]})
    g["vat7"] = {}
    assert lb.reading(g, CKPTS)["outcome"] == "invalid"


def test_a_creep_with_no_single_significant_step_is_gradual():
    # each step rises on only half the voices; end to end every voice rose
    g = _gaps({"derisk": [0] * 8, "vat3": [1, 1, 1, 1, 0, 0, 0, 0],
               "vat5": [1] * 8, "vat6": [2, 2, 2, 2, 1, 1, 1, 1], "vat7": [2] * 8})
    r = lb.reading(g, CKPTS)
    assert r["rises"] == [] and r["outcome"] == "gradual"


def test_the_bench_is_invalid_when_vat7_does_not_reproduce_its_hum():
    g = _gaps({c: [0] * 8 for c in CKPTS})
    assert lb.reading(g, CKPTS)["outcome"] == "invalid"


def test_step_tests_use_only_voices_rated_at_both_ends():
    g = {"derisk": {0: 0, 1: 0}, "vat3": {0: 1}, "vat5": {0: 1, 1: 1},
         "vat6": {0: 1, 1: 1}, "vat7": {0: 1, 1: 1}}
    step = lb.reading(g, CKPTS)["steps"][0]
    assert step[:2] == ("derisk", "vat3") and step[4] == 1
