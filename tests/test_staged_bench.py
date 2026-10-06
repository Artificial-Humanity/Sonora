"""The pure half of the staged-rebuild bench (`scripts/lib/staged_bench.py`).

Pins the pre-registered readings of checks 1-3 (Notes: Sonora/staged-rebuild-
preregistration.md), the one-sided test they rest on, which voices are excluded, and how
items are served.
"""
import random
from fractions import Fraction

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import baseline_bench as bb  # noqa: E402
import staged_bench as sb  # noqa: E402

OK = {"stock": 0.3, "R": 0.5, "derisk": 0.5, "C0": 0.4, "S1": 0.4, "S2": 0.6}


def gaps(**arms):
    return {a: dict(enumerate(v)) for a, v in arms.items()}


# ---------------------------------------------------------------- the test

def test_one_sided_p_of_sixteen_equal_rises_is_one_in_two_to_the_sixteen():
    assert sb.one_sided_p([2] * 16) == pytest.approx(1 / 65536)


def test_one_sided_p_is_never_above_the_two_sided_one():
    for d in ([1, 2, -1, 3, 0, 2, 1, -2], [1] * 5 + [-1] * 3, [0, 0, 1]):
        assert sb.one_sided_p(d) <= bb.sign_flip_p(d) + 1e-12


def test_a_fall_is_not_significant_one_sided():
    assert sb.one_sided_p([-2] * 16) == pytest.approx(1.0)


def test_more_than_twenty_units_is_refused():
    with pytest.raises(ValueError):
        sb.one_sided_p([1] * 21)


def test_an_empty_contrast_is_mean_zero_p_one():
    assert sb.test([]) == (0.0, 1.0, 0)


# ---------------------------------------------------------------- exclusion

def test_a_speakers_files_libri_voices_are_excluded_like_a_keys():
    docs = [{"items": {"item_00": {"spk": 12}, "item_01": {"vctk_spk": "p326"}}},
            {"items": [{"unit": 0, "libri": 40, "vctk": "p241"}, {"unit": 1, "libri": 7}]}]
    assert sb.heard_libri(docs) == {12, 40, 7}


# ---------------------------------------------------------------- sides

def test_a_unit_serves_every_arm_on_one_side_and_sides_balance():
    sw = sb.side_plan(20, ("S1", "S2", "R"), random.Random(3))
    assert all(len({sw[u][a] for a in ("S1", "S2", "R")}) == 1 for u in range(20))
    assert sum(sw[u]["R"] for u in range(20)) == 10


# ---------------------------------------------------------------- checkpoints

@pytest.mark.parametrize("name,ok", [
    ("checkpoint_epoch=009_step=0009209.ckpt", True),
    ("checkpoint_epoch=003_step=0003684.ckpt", False),
    ("last.ckpt", False),
    ("checkpoint_epoch=009.ckpt", False),
])
def test_only_the_tenth_epochs_step_checkpoint_counts(name, ok):
    assert sb.ten_epoch_ckpt("/x/runs/r/checkpoints/" + name) is ok


# ---------------------------------------------------------------- check 1

def test_check1_go():
    r = sb.reading(1, gaps(stock=[0] * 16, R=[2] * 16, derisk=[2] * 16), OK)
    assert r["valid"] and r["outcome"] == "go" and r["n_tests"] == 3


def test_check1_stop_when_r_is_not_shown():
    r = sb.reading(1, gaps(stock=[0] * 16, R=[0] * 16, derisk=[2] * 16), OK)
    assert r["outcome"] == "stop"


def test_check1_invalid_without_the_positive_control():
    r = sb.reading(1, gaps(stock=[0] * 16, R=[2] * 16, derisk=[0] * 16), OK)
    assert r["outcome"] == "invalid" and not r["valid"]


def test_check1_invalid_when_floors_are_far_apart():
    r = sb.reading(1, gaps(stock=[0] * 16, R=[2] * 16, derisk=[2] * 16),
                   dict(OK, derisk=1.6))
    assert r["outcome"] == "invalid" and r["floors_ok"] is False


def test_floors_exactly_one_point_apart_are_not_comparable():
    # 3/20 vs 23/20: as floats the gap computes as 0.9999999999999999 and passes.
    assert not sb.floors_ok({"stock": Fraction(3, 20), "R": Fraction(23, 20)}, ["stock", "R"])
    assert sb.floors_ok({"stock": Fraction(3, 20), "R": Fraction(22, 20)}, ["stock", "R"])


def test_a_floor_of_exactly_four_is_not_comparable():
    assert not sb.floors_ok({"stock": Fraction(4), "R": Fraction(4)}, ["stock", "R"])
    assert sb.floors_ok({"stock": Fraction(399, 100), "R": Fraction(399, 100)}, ["stock", "R"])


def test_check1_reports_r_minus_derisk_whatever_the_outcome():
    r = sb.reading(1, gaps(stock=[0] * 16, R=[1] * 16, derisk=[3] * 16), OK)
    assert r["contrasts"]["R-derisk"][0] == -2


# ---------------------------------------------------------------- check 2

def test_check2_fine_tuning_when_c0_rises():
    r = sb.reading(2, gaps(stock=[0] * 16, C0=[2] * 16, S1=[2] * 16), OK)
    assert r["outcome"] == "shown" and r["shown"] == ["fine_tuning"] and r["n_tests"] == 2


def test_check2_data_when_only_s1_rises():
    r = sb.reading(2, gaps(stock=[0] * 16, C0=[0] * 16, S1=[2] * 16), OK)
    assert r["shown"] == ["data"]


def test_check2_not_shown():
    r = sb.reading(2, gaps(stock=[0] * 16, C0=[0] * 16, S1=[0] * 16), OK)
    assert r["outcome"] == "not_shown" and r["shown"] == []


def test_check2_invalid_on_floors():
    r = sb.reading(2, gaps(stock=[0] * 16, C0=[2] * 16, S1=[2] * 16), dict(OK, S1=4.1))
    assert r["outcome"] == "invalid" and r["shown"] == []


# ---------------------------------------------------------------- check 3

def test_check3_audio_path_and_conditioning_can_both_show():
    r = sb.reading(3, gaps(S1=[0] * 16, S2=[2] * 16, R=[4] * 16), OK)
    assert r["shown"] == ["audio_path", "conditioning"] and r["n_tests"] == 4


def test_check3_valid_through_s1s_own_excess_alone():
    r = sb.reading(3, gaps(S1=[2] * 16, S2=[2] * 16, R=[2] * 16), OK)
    assert r["valid"] and r["outcome"] == "not_shown"


def test_check3_invalid_when_the_hum_is_not_visible_on_these_voices():
    r = sb.reading(3, gaps(S1=[0] * 16, S2=[0] * 16, R=[0] * 16), OK)
    assert r["outcome"] == "invalid" and r["shown"] == []


def test_contrasts_use_only_units_rated_in_both_arms():
    g = gaps(S1=[0] * 16, S2=[2] * 16, R=[2] * 16)
    del g["S2"][5]
    r = sb.reading(3, g, OK)
    assert r["contrasts"]["S2-S1"][2] == 15 and r["contrasts"]["R-S1"][2] == 16
