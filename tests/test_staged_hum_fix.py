"""The hum-fix batch's checks A, B, C (`--check 5, 6, 7`) in the staged bench.

Pre-registered in Notes: Sonora/hum-fix-batch-preregistration.md (owner, 2026-10-12). Pins
the readings, the registered checkpoints and inputs, check C's pairing rule, and how the
tools select, render and unblind the three checks. Checks 1-4 are pinned in
`test_staged_bench.py` and `test_staged_tools.py`.
"""
from fractions import Fraction

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import staged_bench as sb  # noqa: E402
import staged_data as sd  # noqa: E402

OK = {"C0": Fraction(2, 5), "D1": Fraction(1, 2), "S1": Fraction(2, 5), "G": Fraction(1, 2),
      "F1": Fraction(3, 5)}


def gaps(**arms):
    return {a: dict(enumerate(v)) for a, v in arms.items()}


# ---------------------------------------------------------------- registration

def test_the_three_checks_and_their_arms():
    assert sb.HUM_FIX_CHECKS == (5, 6, 7)
    a, b, c = (sb.CHECKS[k] for k in (5, 6, 7))
    assert a["arms"] == ("C0", "D1") and a["vctk_arms"] == ("C0", "D1") and a["libri_arms"] == ()
    assert a["contrasts"] == (("D1", "C0"),) and a["floor_ref"] == "C0"
    assert b["arms"] == ("S1", "G") and b["libri_arms"] == ("S1", "G") and b["vctk_arms"] == ()
    assert b["contrasts"] == (("S1", "G"),) and b["floor_ref"] == "S1"
    assert c["arms"] == ("S1", "F1") and c["libri_arms"] == ("S1", "F1") and c["vctk_arms"] == ()
    assert c["contrasts"] == (("S1", "F1"),) and c["floor_ref"] == "S1"
    assert {sb.CHECKS[k]["min_units"] for k in (5, 6, 7)} == {16}


def test_each_new_arm_renders_on_its_registered_path():
    assert sb.FAMILY["D1"] == sb.FAMILY["C0"] == "vctk22"
    assert sb.FAMILY["G"] == sb.FAMILY["S1"] == "libri22"       # S1's LibriTTS-R 22k copy
    assert sb.FAMILY["F1"] not in ("libri22", "orig22", "vctk22", "libri24")


def test_the_registered_control_checkpoints():
    assert sb.CONTROL_SHA256 == {
        "C0": "d0d6b49b2e92a59e1caf43b82f0359c4c0940d7960b428caadee1091ab24b879",
        "S1": "857f62d7ac47770d94f971cf4bae80d3078e42fd028d45465fe30bd0e2a0d0d5"}
    assert sb.CONTROL_SHA256["S1"] == sb.CHECK4_S1_SHA256


def test_the_registered_input_digests():
    assert sb.INPUT_SHA256 == {
        "hnr_json": "b2f15110082e9abcc4f74454f3fb6ba1f493604f0937b956483865190abddab2",
        "vctk_hnr_from": "3bd8e2a61cdebd56dff96c7746b5418d6ee204f17b485b05fe7fb9083bd1be9f"}


@pytest.mark.parametrize("arm,good,bad", [
    ("F1", "checkpoint_epoch=009_step=0009779.ckpt", "checkpoint_epoch=009_step=0009209.ckpt"),
    ("G", "checkpoint_epoch=009_step=0009209.ckpt", "checkpoint_epoch=009_step=0009779.ckpt"),
])
def test_f1_and_g_finals_are_their_registered_names(arm, good, bad):
    assert sb.final_ckpt(arm, "/x/checkpoints/" + good)
    assert not sb.final_ckpt(arm, "/x/checkpoints/" + bad)
    assert sb.final_epochs(arm) == 10


def test_f1s_final_is_ten_epochs_of_its_train_rows():
    assert sb.FINAL_CKPT["F1"] == "checkpoint_epoch=009_step=%07d.ckpt" % (
        10 * sd.steps_per_epoch(31287) - 1)


@pytest.mark.parametrize("check,digests,why", [
    (5, {"C0": "x", "D1": None}, "C0"),
    (6, {"S1": "x", "G": None}, "S1"),
    (7, {"S1": None, "F1": None}, "S1"),
])
def test_a_control_that_is_not_the_registered_checkpoint_is_refused(check, digests, why):
    with pytest.raises(ValueError, match=why):
        sb.check_registered_ckpts(check, digests)


def test_registered_controls_pass_and_an_untrained_arm_may_be_missing():
    c0, s1 = sb.CONTROL_SHA256["C0"], sb.CONTROL_SHA256["S1"]
    sb.check_registered_ckpts(5, {"C0": c0, "D1": None})
    sb.check_registered_ckpts(5, {"C0": c0, "D1": "d1"})
    sb.check_registered_ckpts(6, {"S1": s1, "G": "g"})
    sb.check_registered_ckpts(7, {"S1": s1, "F1": "f1"})


def test_a_g_checkpoint_that_is_s1s_is_refused():
    s1 = sb.CONTROL_SHA256["S1"]
    with pytest.raises(ValueError, match="G.*S1"):
        sb.check_registered_ckpts(6, {"S1": s1, "G": s1})


def test_checks_1_to_4_have_no_hum_fix_pins():
    sb.check_registered_ckpts(2, {"stock": "a", "C0": "b", "S1": "c"})


# ---------------------------------------------------------------- exclusion

def test_an_f1_voice_in_a_check7_speakers_file_is_excluded_by_its_v7_index():
    docs = [{"items": [{"unit": 0, "libri": 40, "f1": {"libri": 4234, "f1_sid": 1}}]}]
    assert sb.heard_libri(docs) == {40, 4234}


def test_heard_libri_reads_old_files_as_before():
    docs = [{"items": {"item_00": {"spk": 12}, "item_01": {"vctk_spk": "p326"}}},
            {"items": [{"unit": 0, "libri": 40, "vctk": "p241",
                        "arms": {"S1": {"source": "x", "libri": 99}}}]}]
    assert sb.heard_libri(docs) == {12, 40}


# ---------------------------------------------------------------- check C's pairing

def test_pairs_take_the_nearest_f0_within_half_a_db_in_ascending_hnr_order():
    s1 = {10: (3.0, 100.0), 11: (5.0, 200.0)}
    f1 = {500: (3.4, 101.0, 77),       # within 0.5 dB, F0 near 100
          501: (3.0, 150.0, 78),       # HNR equal, F0 far
          502: (5.2, 210.0, 79),
          503: (5.6, 200.0, 80)}       # F0 exact, but 0.6 dB away
    assert sb.pair_f1(s1, f1) == [(10, 500), (11, 502)]


def test_the_half_db_window_is_inclusive_and_exact():
    # 5.252 - 4.752 is 0.5000000000000004 as floats; exactly 0.5 it is inside
    assert sb.pair_f1({1: (5.252, 100.0)}, {9: (4.752, 100.0, 9)}) == [(1, 9)]
    assert sb.pair_f1({1: (5.252, 100.0)}, {9: (4.751, 100.0, 9)}) == []


def test_a_used_f1_voice_is_not_reused_and_the_first_s1_voice_by_hnr_takes_it():
    s1 = {7: (4.0, 120.0), 3: (4.0, 121.0), 5: (3.9, 300.0)}
    f1 = {800: (4.0, 120.5, 1), 801: (4.1, 300.0, 2)}
    # order: 5 (3.9), then 3 (4.0, lower index), then 7 (4.0)
    assert sb.pair_f1(s1, f1) == [(5, 801), (3, 800)]


def test_f0_ties_go_to_the_smaller_hnr_gap_then_the_lower_libritts_id():
    s1 = {1: (4.0, 100.0)}
    assert sb.pair_f1(s1, {8: (4.3, 100.0, 5), 9: (4.1, 100.0, 6)}) == [(1, 9)]
    assert sb.pair_f1(s1, {8: (4.1, 100.0, 6), 9: (3.9, 100.0, 5)}) == [(1, 9)]


def test_f0_is_matched_on_the_log_scale():
    # 100 vs 150 is a ratio of 1.5; 300 vs 200 too, but 200 vs 150 is 1.33
    s1 = {1: (4.0, 200.0)}
    assert sb.pair_f1(s1, {8: (4.0, 300.0, 1), 9: (4.0, 150.0, 2)}) == [(1, 9)]


def test_an_s1_voice_with_no_candidate_is_skipped_not_replaced():
    s1 = {1: (2.0, 100.0), 2: (4.0, 100.0)}
    assert sb.pair_f1(s1, {8: (4.2, 100.0, 1)}) == [(2, 8)]


# ---------------------------------------------------------------- check A (D1 vs C0)

def test_check5_shown_when_d1_hums_more_than_c0():
    r = sb.reading(5, gaps(C0=[0] * 16, D1=[2] * 16), OK)
    assert r["valid"] and r["outcome"] == "shown" and r["shown"] == ["clips_per_speaker"]
    assert r["n_tests"] == 3 and r["n_deciding"] == 1


def test_check5_needs_no_positive_control():
    r = sb.reading(5, gaps(C0=[0] * 16, D1=[0] * 16), OK)
    assert r["valid"] and r["outcome"] == "not_shown" and r["shown"] == []


def test_check5_floors_are_compared_with_c0s():
    r = sb.reading(5, gaps(C0=[0] * 16, D1=[2] * 16),
                   {"C0": Fraction(3, 20), "D1": Fraction(23, 20)})
    assert r["floors_ok"] is False and r["outcome"] == "invalid" and r["shown"] == []


def test_check5_with_fifteen_voices_rated_in_both_arms_is_invalid():
    g = gaps(C0=[0] * 16, D1=[2] * 16)
    assert sb.reading(5, g, OK)["outcome"] == "shown"
    del g["D1"][3]
    r = sb.reading(5, g, OK)
    assert r["contrasts"]["D1-C0"][2] == 15 and r["outcome"] == "invalid"


def test_check5_reports_both_arms_excess():
    r = sb.reading(5, gaps(C0=[1] * 16, D1=[2] * 16), OK)
    assert r["level"]["C0"][0] == 1 and r["level"]["D1"][0] == 2


# ---------------------------------------------------------------- checks B and C

@pytest.mark.parametrize("check,arm,name", [(6, "G", "symbol"), (7, "F1", "dense_speakers")])
def test_shown_when_s1_hums_more(check, arm, name):
    r = sb.reading(check, gaps(S1=[2] * 16, **{arm: [0] * 16}), OK)
    assert r["valid"] and r["outcome"] == "shown" and r["shown"] == [name]
    assert r["n_tests"] == 3 and r["n_deciding"] == 2


@pytest.mark.parametrize("check,arm", [(6, "G"), (7, "F1")])
def test_not_shown_when_the_fix_hums_as_much(check, arm):
    r = sb.reading(check, gaps(S1=[2] * 16, **{arm: [2] * 16}), OK)
    assert r["valid"] and r["outcome"] == "not_shown"


@pytest.mark.parametrize("check,arm", [(6, "G"), (7, "F1")])
def test_invalid_when_s1s_own_excess_is_not_shown(check, arm):
    r = sb.reading(check, gaps(S1=[0] * 16, **{arm: [-2] * 16}), OK)
    assert r["outcome"] == "invalid" and r["shown"] == []


@pytest.mark.parametrize("check,arm", [(6, "G"), (7, "F1")])
def test_floors_are_compared_with_s1s(check, arm):
    r = sb.reading(check, gaps(S1=[2] * 16, **{arm: [0] * 16}),
                   {"S1": Fraction(3, 20), arm: Fraction(23, 20)})
    assert r["floors_ok"] is False and r["outcome"] == "invalid"
    r = sb.reading(check, gaps(S1=[2] * 16, **{arm: [0] * 16}),
                   {"S1": Fraction(4), arm: Fraction(4)})
    assert r["outcome"] == "invalid"


@pytest.mark.parametrize("check,arm", [(6, "G"), (7, "F1")])
def test_fewer_than_sixteen_voices_rated_in_both_arms_is_invalid(check, arm):
    g = gaps(S1=[2] * 16, **{arm: [0] * 16})
    del g["S1"][0]
    r = sb.reading(check, g, OK)
    assert r["contrasts"]["S1-%s" % arm][2] == 15 and r["outcome"] == "invalid"


# ---------------------------------------------------------------- the render tool's arms

def _data_cfg(name):
    """configs/data/<name>.yaml with a `defaults` base merged under it (one level, as the
    staged configs use)."""
    import yaml
    from pathlib import Path
    d = Path(__file__).resolve().parent.parent / "configs" / "data"
    c = yaml.safe_load((d / ("%s.yaml" % name)).read_text(encoding="utf-8"))
    out = {}
    for base in c.get("defaults", []):
        if base != "_self_":
            out.update(_data_cfg(base))
    out.update({k: v for k, v in c.items() if k != "defaults"})
    return out


@pytest.mark.parametrize("arm,experiment,n_spks", [
    ("D1", "staged_d1", 109), ("G", "staged_g", 247), ("F1", "staged_f1", 90)])
def test_each_new_arm_is_checked_against_its_training_datas_statistics(arm, experiment, n_spks):
    import yaml
    from pathlib import Path
    import render_ear_staged as rs
    exp = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "experiment"
                          / ("%s.yaml" % experiment)).read_text(encoding="utf-8"))
    trained_on = _data_cfg(exp["defaults"][0]["override /data"][:-len(".yaml")])
    corpus, n, use_vat = rs.EXPECT[arm]
    # render reads data_statistics from configs/data/<corpus>.yaml directly, without defaults
    own = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "data"
                        / ("%s.yaml" % corpus)).read_text(encoding="utf-8"))
    assert own["data_statistics"] == trained_on["data_statistics"]
    assert n == trained_on["n_spks"] == n_spks and use_vat is False
    assert arm in rs.NEW
