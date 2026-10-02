"""The pure half of the first-step hum bench (`scripts/lib/first_step_bench.py`).

The bench asks whether derisk-energy, the first Sonora checkpoint, already hums more than
stock `matcha_vctk` at the same voice roughness, with vat7 ep005 as the known-humming
check. These pin the choices the listening result rests on: which voices are excluded,
which side each clip is served on, and how the ratings become the pre-registered reading.
"""

import random

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import first_step_bench as fs  # noqa: E402

FAMILIES = ("stock", "derisk", "vat7")


# ---------------------------------------------------------------- exclusion

def test_vctk_voices_heard_in_an_earlier_key_are_excluded():
    # the baseline key stores a VCTK voice under `vctk_spk`; its speakers file under `vctk`
    docs = [{"items": {"item_00": {"vctk_spk": "p326", "family": "stock"},
                       "item_01": {"spk": 12, "family": "ours"}}},
            {"items": [{"vctk": "p241", "libri": 40}]}]
    assert fs.heard_vctk(docs) == {"p326", "p241"}


def test_a_libri_index_is_not_a_vctk_voice():
    assert fs.heard_vctk([{"items": {"x": {"spk": 7}}}]) == set()


# ---------------------------------------------------------------- sides

def test_a_match_serves_all_three_families_on_one_side():
    # same side within a match: a listener's A/B lean cancels exactly in every contrast
    sw = fs.side_plan(16, random.Random(1))
    assert len(sw) == 16
    for m in range(16):
        assert len({sw[m][f] for f in FAMILIES}) == 1


def test_sides_balance_within_each_family():
    sw = fs.side_plan(16, random.Random(2))
    for f in FAMILIES:
        assert sum(sw[m][f] for m in range(16)) == 8


# ---------------------------------------------------------------- the reading

def _gaps(stock, derisk, vat7):
    return {"stock": dict(enumerate(stock)), "derisk": dict(enumerate(derisk)),
            "vat7": dict(enumerate(vat7))}


FLOORS_OK = {"stock": 0.3, "derisk": 0.5, "vat7": 0.4}


def test_excess_already_at_derisk_is_the_first_step():
    g = _gaps([0] * 16, [2] * 16, [2] * 16)
    r = fs.reading(g, FLOORS_OK)
    assert r["outcome"] == "first_step"


def test_excess_at_derisk_and_more_by_vat7_is_both_steps():
    g = _gaps([0] * 16, [1] * 16, [3] * 16)
    assert fs.reading(g, FLOORS_OK)["outcome"] == "both_steps"


def test_excess_only_after_derisk_is_later():
    g = _gaps([0] * 16, [0] * 16, [2] * 16)
    assert fs.reading(g, FLOORS_OK)["outcome"] == "later"


def test_vat7_not_above_stock_is_invalid():
    g = _gaps([1] * 16, [3] * 16, [1] * 16)
    assert fs.reading(g, FLOORS_OK)["outcome"] == "invalid"


def test_nothing_shown_beyond_the_check_is_inconclusive():
    # vat7 reproduces its excess; derisk sits between it and stock voice by voice in both
    # directions, so neither step reaches p < 0.05
    g = _gaps([0] * 16, [4, -2] * 8, [2] * 16)
    r = fs.reading(g, FLOORS_OK)
    assert r["contrasts"]["vat7-stock"][1] < fs.ALPHA
    assert r["outcome"] == "inconclusive"


def test_floors_too_far_apart_make_the_stock_contrasts_unreadable():
    # a gap measured from a much higher floor has less room on a 0-5 scale
    g = _gaps([0] * 16, [2] * 16, [2] * 16)
    r = fs.reading(g, {"stock": 0.3, "derisk": 1.6, "vat7": 0.4})
    assert r["outcome"] == "invalid" and not r["floors_ok"]


def test_contrasts_use_only_matches_rated_in_both_families():
    g = _gaps([0] * 16, [2] * 16, [2] * 16)
    del g["derisk"][3]
    r = fs.reading(g, FLOORS_OK)
    assert r["contrasts"]["derisk-stock"][2] == 15
    assert r["contrasts"]["vat7-stock"][2] == 16


def test_a_significant_fall_from_derisk_to_vat7_is_reported():
    g = _gaps([0] * 16, [3] * 16, [1] * 16)
    r = fs.reading(g, FLOORS_OK)
    assert r["outcome"] == "first_step" and r["vat7_below_derisk"]


def test_any_vctk_name_is_excluded_not_only_p_names():
    assert fs.heard_vctk([{"items": {"x": {"vctk_spk": "s5"}}}]) == {"s5"}


def test_a_floor_at_or_over_four_is_invalid():
    g = _gaps([0] * 16, [2] * 16, [2] * 16)
    r = fs.reading(g, {"stock": 3.5, "derisk": 4.0, "vat7": 3.6})
    assert r["outcome"] == "invalid" and not r["floors_ok"]


def test_vat7_significantly_below_stock_is_invalid():
    g = _gaps([2] * 16, [2] * 16, [0] * 16)
    assert fs.reading(g, FLOORS_OK)["outcome"] == "invalid"
