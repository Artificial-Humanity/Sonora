"""The pure half of the staged-rebuild bench (`scripts/lib/staged_bench.py`).

Pins the pre-registered readings of checks 1-4 (Notes: Sonora/staged-rebuild-
preregistration.md), the one-sided test they rest on, which voices are excluded, and how
items are served.
"""
import json
import random
from fractions import Fraction

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import baseline_bench as bb  # noqa: E402
import staged_bench as sb  # noqa: E402

OK = {"stock": 0.3, "R": 0.5, "derisk": 0.5, "C0": 0.4, "S1": 0.4, "S2": 0.6, "L": 0.9}


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


@pytest.mark.parametrize("arm", ["R", "C0", "S1", "S2", "L"])
@pytest.mark.parametrize("name,ok", [
    ("checkpoint_epoch=009_step=0009209.ckpt", True),
    ("checkpoint_epoch=025_step=0009229.ckpt", False),
    ("last.ckpt", False),
])
def test_an_unregistered_arms_final_is_still_the_tenth_epochs_step(arm, name, ok):
    assert arm not in sb.FINAL_CKPT
    assert sb.final_ckpt(arm, "/x/checkpoints/" + name) is ok
    assert sb.final_epochs(arm) == 10


@pytest.mark.parametrize("name,ok", [
    ("checkpoint_epoch=025_step=0009229.ckpt", True),
    ("checkpoint_epoch=009_step=0003549.ckpt", False),
    ("checkpoint_epoch=025_step=0009230.ckpt", False),
    ("checkpoint_epoch=025.ckpt", False),
])
def test_d1s_final_is_its_registered_name(name, ok):
    assert sb.final_ckpt("D1", "/x/staged_d1/runs/r/checkpoints/" + name) is ok
    assert sb.final_epochs("D1") == 26


def test_d1s_registered_final_is_26_epochs_of_its_subsample():
    import staged_data as sd
    spe = sd.steps_per_epoch(108 * sd.D1_PER_SPEAKER)
    assert sb.FINAL_CKPT["D1"] == "checkpoint_epoch=%03d_step=%07d.ckpt" % (25, 26 * spe - 1)


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


# ---------------------------------------------------------------- floors' reference arm

def test_floors_are_compared_with_the_reference_arm_given():
    f = {"stock": Fraction(0), "S1": Fraction(3, 2), "L": Fraction(2)}
    assert not sb.floors_ok(f, ["stock", "S1"])             # stock is the default reference
    assert sb.floors_ok(f, ["S1", "L"], "S1")
    assert not sb.floors_ok(dict(f, L=Fraction(5, 2)), ["S1", "L"], "S1")    # exactly 1 apart


def test_checks_1_and_2_take_stock_as_reference_and_check_3_has_no_floor_rule():
    assert [sb.CHECKS[c].get("floor_ref") for c in (1, 2, 3, 4)] == ["stock", "stock", None, "S1"]
    r = sb.reading(3, gaps(S1=[2] * 16, S2=[2] * 16, R=[2] * 16), dict(OK, S1=9))
    assert r["floors_ok"] is None and r["valid"]


# ---------------------------------------------------------------- check 4

def test_check4_arms_and_its_one_contrast():
    c = sb.CHECKS[4]
    assert c["arms"] == ("S1", "L") and c["libri_arms"] == ("S1", "L") and c["vctk_arms"] == ()
    assert c["contrasts"] == (("S1", "L"),)
    assert sb.FAMILY["L"] != sb.FAMILY["S1"] == "libri22"     # its own round-trip source


def test_check4_shown_when_s1_hums_more_than_l():
    r = sb.reading(4, gaps(S1=[2] * 16, L=[0] * 16), OK)
    assert r["valid"] and r["outcome"] == "shown" and r["shown"] == ["restoration"]


def test_check4_not_shown_when_l_hums_as_much():
    r = sb.reading(4, gaps(S1=[2] * 16, L=[2] * 16), OK)
    assert r["valid"] and r["outcome"] == "not_shown" and r["shown"] == []


def test_check4_invalid_when_s1s_own_excess_does_not_reproduce():
    r = sb.reading(4, gaps(S1=[0] * 16, L=[-2] * 16), OK)
    assert r["outcome"] == "invalid" and not r["valid"] and r["shown"] == []


def test_check4_invalid_when_ls_floor_is_a_point_or_more_from_s1s():
    r = sb.reading(4, gaps(S1=[2] * 16, L=[0] * 16),
                   {"S1": Fraction(3, 20), "L": Fraction(23, 20)})
    assert r["floors_ok"] is False and r["outcome"] == "invalid"


def test_check4_invalid_when_a_floor_is_four():
    r = sb.reading(4, gaps(S1=[2] * 16, L=[0] * 16), {"S1": Fraction(4), "L": Fraction(4)})
    assert r["outcome"] == "invalid"


def test_check4_invalid_with_fewer_than_sixteen_voices_rated_in_both_arms():
    g = gaps(S1=[2] * 17, L=[0] * 17)
    del g["L"][0]
    assert sb.reading(4, g, OK)["outcome"] == "shown"         # 16 in both
    del g["S1"][1]
    r = sb.reading(4, g, OK)                                  # 15 in both
    assert r["contrasts"]["S1-L"][2] == 15 and r["outcome"] == "invalid"


def test_check4_reports_three_p_values_two_deciding_and_ls_own_excess():
    r = sb.reading(4, gaps(S1=[2] * 16, L=[1] * 16), OK)
    assert r["n_tests"] == 3 and r["n_deciding"] == 2
    assert r["level"]["L"][0] == 1 and r["level"]["L"][1] < 0.05


def test_check4_splits_s1_minus_l_into_model_and_round_trip_descriptively():
    model = {"S1": {0: 3, 1: 4, 2: 2}, "L": {0: 2, 1: 2, 3: 5}}
    rt = {"S1": {0: 1, 1: 1, 2: 0}, "L": {0: 2, 1: 1, 3: 0}}
    d = sb.split_contrast(model, rt, "S1", "L")
    assert d == {"model": (1.5, 2), "round_trip": (-0.5, 2)}     # units 0 and 1 only


def test_qs_final_is_the_step_that_ends_its_tenth_epoch(tmp_path):
    """Audit M-4: Q has no FINAL_CKPT entry, so any epoch=009 step checkpoint passed, including one
    written mid-epoch by a mis-set SONORA_CKPT_EVERY_N_STEPS."""
    prov = tmp_path / "provenance.json"
    ck = "/x/checkpoints/checkpoint_epoch=009_step=%07d.ckpt"
    assert sb.final_ckpt("Q", ck % 5, str(prov))                    # no corpus yet: the tenth-epoch rule
    assert sb.q_final_step(str(prov)) is None
    prov.write_text(json.dumps({"steps_per_epoch": 1000}), encoding="utf-8")
    assert sb.q_final_step(str(prov)) == 9999
    assert sb.final_ckpt("Q", ck % 9999, str(prov))
    assert not sb.final_ckpt("Q", ck % 9500, str(prov))             # mid-epoch-9
    assert not sb.final_ckpt("Q", "/x/checkpoints/checkpoint_epoch=008_step=0009999.ckpt", str(prov))
    assert not sb.final_ckpt("Q", "/x/checkpoints/last.ckpt", str(prov))
    assert sb.final_ckpt("S1", ck % 9500, str(prov))                # other arms are unchanged
