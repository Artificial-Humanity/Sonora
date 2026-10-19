"""The hum-fix batch's checks A, B, C, D (`--check 5, 6, 7, 8`) in the staged bench.

Pre-registered in Notes: Sonora/hum-fix-batch-preregistration.md (owner, 2026-10-12; check D
2026-10-19). Pins the readings, the registered checkpoints and inputs, check C's pairing
rule, and how the tools select, render and unblind the four checks. Checks 1-4 are pinned in
`test_staged_bench.py` and `test_staged_tools.py`.
"""
from fractions import Fraction

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import staged_bench as sb  # noqa: E402
import staged_data as sd  # noqa: E402

OK = {"C0": Fraction(2, 5), "D1": Fraction(1, 2), "S1": Fraction(2, 5), "G": Fraction(1, 2),
      "F1": Fraction(3, 5), "E": Fraction(1, 2)}


def gaps(**arms):
    return {a: dict(enumerate(v)) for a, v in arms.items()}


# ---------------------------------------------------------------- registration

def test_the_three_checks_and_their_arms():
    assert sb.HUM_FIX_CHECKS == (5, 6, 7, 8)
    a, b, c, d = (sb.CHECKS[k] for k in (5, 6, 7, 8))
    assert a["arms"] == ("C0", "D1") and a["vctk_arms"] == ("C0", "D1") and a["libri_arms"] == ()
    assert a["contrasts"] == (("D1", "C0"),) and a["floor_ref"] == "C0"
    assert b["arms"] == ("S1", "G") and b["libri_arms"] == ("S1", "G") and b["vctk_arms"] == ()
    assert b["contrasts"] == (("S1", "G"),) and b["floor_ref"] == "S1"
    assert c["arms"] == ("S1", "F1") and c["libri_arms"] == ("S1", "F1") and c["vctk_arms"] == ()
    assert c["contrasts"] == (("S1", "F1"),) and c["floor_ref"] == "S1"
    assert d["arms"] == ("S1", "E") and d["libri_arms"] == ("S1", "E") and d["vctk_arms"] == ()
    assert d["contrasts"] == (("S1", "E"),) and d["floor_ref"] == "S1"
    assert {sb.CHECKS[k]["min_units"] for k in (5, 6, 7, 8)} == {16}


def test_each_new_arm_renders_on_its_registered_path():
    assert sb.FAMILY["D1"] == sb.FAMILY["C0"] == "vctk22"
    assert sb.FAMILY["G"] == sb.FAMILY["E"] == sb.FAMILY["S1"] == "libri22"
    assert sb.FAMILY["F1"] not in ("libri22", "orig22", "vctk22", "libri24")


def test_the_registered_control_checkpoints():
    assert sb.CONTROL_SHA256 == {
        "C0": "d0d6b49b2e92a59e1caf43b82f0359c4c0940d7960b428caadee1091ab24b879",
        "S1": "857f62d7ac47770d94f971cf4bae80d3078e42fd028d45465fe30bd0e2a0d0d5"}
    assert sb.CONTROL_SHA256["S1"] == sb.CHECK4_S1_SHA256


def test_the_registered_input_digests():
    assert sb.INPUT_SHA256 == {
        "hnr_json": "b2f15110082e9abcc4f74454f3fb6ba1f493604f0937b956483865190abddab2",
        "vctk_hnr_from": "3bd8e2a61cdebd56dff96c7746b5418d6ee204f17b485b05fe7fb9083bd1be9f",
        "s1_hnr_json": "93a1d91eab6905a4652991c500841a509a978008a16a13c9fee7f311d620d27c"}


@pytest.mark.parametrize("arm,good,bad", [
    ("F1", "checkpoint_epoch=009_step=0009779.ckpt", "checkpoint_epoch=009_step=0009209.ckpt"),
    ("G", "checkpoint_epoch=009_step=0009209.ckpt", "checkpoint_epoch=009_step=0009779.ckpt"),
    ("E", "checkpoint_epoch=009_step=0009209.ckpt", "checkpoint_epoch=009_step=0009779.ckpt"),
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
    (8, {"S1": "x", "E": None}, "S1"),
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
    sb.check_registered_ckpts(8, {"S1": s1, "E": None})
    sb.check_registered_ckpts(8, {"S1": s1, "E": "e"})


def test_a_g_checkpoint_that_is_s1s_is_refused():
    s1 = sb.CONTROL_SHA256["S1"]
    with pytest.raises(ValueError, match="G.*S1"):
        sb.check_registered_ckpts(6, {"S1": s1, "G": s1})


@pytest.mark.parametrize("check,arm,ctrl", [
    (5, "D1", "C0"), (5, "D1", "S1"), (6, "G", "S1"), (6, "G", "C0"),
    (7, "F1", "S1"), (7, "F1", "C0"), (8, "E", "S1"), (8, "E", "C0")])
def test_a_new_arms_checkpoint_that_is_a_controls_is_refused(check, arm, ctrl):
    same = sb.CONTROL_SHA256[ctrl]
    digests = {"C0": sb.CONTROL_SHA256["C0"]} if check == 5 else {"S1": sb.CONTROL_SHA256["S1"]}
    with pytest.raises(ValueError, match="%s checkpoint is %s's" % (arm, ctrl)):
        sb.check_registered_ckpts(check, dict(digests, **{arm: same}))


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


# ---------------------------------------------------------------- checks B, C and D

@pytest.mark.parametrize("check,arm,name", [
    (6, "G", "symbol"), (7, "F1", "dense_speakers"), (8, "E", "text_frontend")])
def test_shown_when_s1_hums_more(check, arm, name):
    r = sb.reading(check, gaps(S1=[2] * 16, **{arm: [0] * 16}), OK)
    assert r["valid"] and r["outcome"] == "shown" and r["shown"] == [name]
    assert r["n_tests"] == 3 and r["n_deciding"] == 2


@pytest.mark.parametrize("check,arm", [(6, "G"), (7, "F1"), (8, "E")])
def test_not_shown_when_the_fix_hums_as_much(check, arm):
    r = sb.reading(check, gaps(S1=[2] * 16, **{arm: [2] * 16}), OK)
    assert r["valid"] and r["outcome"] == "not_shown"


@pytest.mark.parametrize("check,arm", [(6, "G"), (7, "F1"), (8, "E")])
def test_invalid_when_s1s_own_excess_is_not_shown(check, arm):
    r = sb.reading(check, gaps(S1=[0] * 16, **{arm: [-2] * 16}), OK)
    assert r["outcome"] == "invalid" and r["shown"] == []


@pytest.mark.parametrize("check,arm", [(6, "G"), (7, "F1"), (8, "E")])
def test_floors_are_compared_with_s1s(check, arm):
    r = sb.reading(check, gaps(S1=[2] * 16, **{arm: [0] * 16}),
                   {"S1": Fraction(3, 20), arm: Fraction(23, 20)})
    assert r["floors_ok"] is False and r["outcome"] == "invalid"
    r = sb.reading(check, gaps(S1=[2] * 16, **{arm: [0] * 16}),
                   {"S1": Fraction(4), arm: Fraction(4)})
    assert r["outcome"] == "invalid"


@pytest.mark.parametrize("check,arm", [(6, "G"), (7, "F1"), (8, "E")])
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
    ("D1", "staged_d1", 109), ("G", "staged_g", 247), ("F1", "staged_f1", 90),
    ("E", "staged_e", 247)])
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


# ---------------------------------------------------------------- the select / render tool

import io  # noqa: E402
import json  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import render_ear_staged as rs  # noqa: E402

R24 = "/r24/"
R22 = "/r22/"


def test_libri_r_24k_inverts_libri_r_22k_for_every_f1_subset():
    for s in sd.F1_SUBSETS:
        w = "%s/%s/19/77/19_77_000000_000000.wav" % (sd.LIBRI_R_ROOT, s)
        assert sd.libri_r_24k(sd.libri_r_22k(w)) == w
    with pytest.raises(ValueError, match="not under"):
        sd.libri_r_24k("%s/train-clean-100/19/77/19_77_000000_000000.wav" % sd.LIBRI_R_ROOT)


def test_hum_fix_checks_require_the_staged_glob_and_every_default():
    want = rs.PRIOR + [rs.ROOT + "/staged_bench/*_speakers.json"]
    for check in (5, 6, 7, 8):
        assert rs.default_prior(check) == want
        rs.require_default_prior(check, want + ["/scratch/check6_dryrun_speakers.json"])
        with pytest.raises(SystemExit, match="check %d's --prior must hold" % check):
            rs.require_default_prior(check, rs.PRIOR)


def test_hum_fix_renders_need_an_explicit_salt_and_not_the_staged_default():
    assert rs.resolve_salt(3, None) == "staged-check3-v1"
    assert rs.resolve_salt(4, "x") == "x"
    for check, salt in ((5, "hum-fix-A-v1"), (6, "hum-fix-B-v1"), (7, "hum-fix-C-v1"),
                        (8, "hum-fix-D-v1")):
        assert rs.resolve_salt(check, salt) == salt
        with pytest.raises(SystemExit, match="--salt"):
            rs.resolve_salt(check, None)
        with pytest.raises(SystemExit, match="staged-check%d-v1" % check):
            rs.resolve_salt(check, "staged-check%d-v1" % check)


def test_hum_fix_renders_refuse_any_salt_but_the_registered_one():
    for check, other in ((5, "hum-fix-B-v1"), (6, "hum-fix-C-v1"), (7, "hum-fix-A-v1"),
                         (8, "hum-fix-B-v1"), (5, "hum-fix-A-v2"), (6, "x"), (7, ""), (8, "")):
        with pytest.raises(SystemExit, match="REFUSING.*hum-fix-%s-v1" % "ABCD"[check - 5]):
            rs.resolve_salt(check, other)
    assert rs.resolve_salt(1, "anything") == "anything"


def test_select_seed_defaults_to_the_registered_one_per_check():
    assert [rs.select_seed(c, None) for c in range(1, 9)] == [20261006] * 4 + [20261012] * 4
    assert rs.select_seed(6, 7) == 7


def test_g_renders_s1s_phonemes_with_its_symbol_moved_and_no_other_arm_changes():
    assert rs.arm_phonemes("G", "pɚmˈɪɾᵻd ᵻ") == "pɚmˈɪɾᵊd ᵊ"
    assert rs.arm_phonemes("E", "pɚmˈɪɾᵊd ɪt") == "pɚmˈɪɾᵊd ɪt"
    with pytest.raises(SystemExit, match="REFUSING.*E's phonemes"):
        rs.arm_phonemes("E", "pɚmˈɪɾᵻd")
    with pytest.raises(SystemExit, match="REFUSING.*E's phonemes"):
        rs.arm_phonemes("E", "")
    for arm in ("S1", "F1", "C0", "D1", "L"):
        assert rs.arm_phonemes(arm, "pɚmˈɪɾᵻd") == "pɚmˈɪɾᵻd"


def test_f1_synthesises_with_its_own_table_index_and_every_other_arm_with_the_v7_one():
    it = {"libri": 40, "f1": {"libri": 4234, "f1_sid": 7}}
    assert rs.voice_index(it, "F1") == 7
    assert rs.voice_index(it, "S1") == rs.voice_index(it, "G") == rs.voice_index(it, "E") == 40


def test_each_hum_fix_arm_round_trips_its_own_22k_copy():
    w100 = sd.LIBRI_24K_ROOT + "/19/198/19_198_000000_000000.wav"
    w500 = sd.LIBRI_R_ROOT + "/train-other-500/19/198/19_198_000000_000000.wav"
    assert rs.rt_source_of("S1", w100) == rs.rt_source_of("G", w100) == \
        rs.rt_source_of("E", w100) == sd.to_22k(w100)
    assert rs.rt_source_of("F1", w500) == sd.libri_r_22k(w500)
    assert rs.rt_source_of("D1", "p225_001") == rs.rt_source_of("C0", "p225_001") == \
        "wav48_silence_trimmed/p225/p225_001_mic1.flac"


def test_hum_fix_keys_record_arm_voice_source_and_round_trip_source():
    w = sd.LIBRI_24K_ROOT + "/19/198/19_198_000000_000000.wav"
    w5 = sd.LIBRI_R_ROOT + "/train-other-500/88/1/88_1_000000_000000.wav"
    it = {"unit": 2, "libri": 19, "libri_hnr": 3.5, "libri_f0": 120.0,
          "partition": "train-clean-100",
          "f1": {"libri": 4234, "libritts_id": 88, "f1_sid": 7, "hnr": 3.6, "f0": 118.0,
                 "partition": "train-other-500"},
          "arms": {"S1": {"source": w}, "F1": {"source": w5}}}
    f1 = rs.truth_entry(7, "F1", it, [("A", "F1"), ("B", "F1rt")])
    assert (f1["spk"], f1["arm"], f1["source"], f1["hnr"]) == (4234, "F1", w5, 3.6)
    assert f1["rt_source"] == sd.libri_r_22k(w5) and f1["f1_sid"] == 7
    assert f1["partition"] == "train-other-500" and f1["f0"] == 118.0
    s1 = rs.truth_entry(7, "S1", it, [("A", "S1rt"), ("B", "S1")])
    assert (s1["spk"], s1["rt_source"], s1["partition"]) == (19, sd.to_22k(w), "train-clean-100")
    g = rs.truth_entry(6, "G", dict(it, arms={"G": {"source": w}}), [("A", "G"), ("B", "Grt")])
    assert g["spk"] == 19 and g["rt_source"] == sd.to_22k(w) and "f1_sid" not in g
    e = rs.truth_entry(8, "E", dict(it, arms={"E": {"source": w}}), [("A", "E"), ("B", "Ert")])
    assert e["spk"] == 19 and e["rt_source"] == sd.to_22k(w) and "f1_sid" not in e
    v = rs.truth_entry(5, "D1", {"unit": 0, "vctk": "p225", "vctk_hnr": 4.0,
                                 "arms": {"D1": {"source": "p225_001"}}},
                       [("A", "D1"), ("B", "D1rt")])
    assert v["vctk_spk"] == "p225" and "spk" not in v
    assert v["rt_source"] == "wav48_silence_trimmed/p225/p225_001_mic1.flac"


def test_render_refuses_a_speakers_file_whose_controls_are_not_registered():
    s1 = sb.CONTROL_SHA256["S1"]
    rs.require_registered_spec(6, {"ckpts": {"S1": ["/s1", s1], "G": ["/g", "g"]}})
    rs.require_registered_spec(8, {"ckpts": {"S1": ["/s1", s1], "E": ["/e", "e"]}})
    with pytest.raises(SystemExit, match="REFUSING.*S1"):
        rs.require_registered_spec(6, {"ckpts": {"S1": ["/s1", "x"], "G": ["/g", "g"]}})
    with pytest.raises(SystemExit, match="REFUSING.*G"):
        rs.require_registered_spec(6, {"ckpts": {"S1": ["/s1", s1], "G": ["/g", s1]}})
    with pytest.raises(SystemExit, match="REFUSING.*E"):
        rs.require_registered_spec(8, {"ckpts": {"S1": ["/s1", s1], "E": ["/e", s1]}})
    rs.require_registered_spec(3, {"ckpts": {"S1": ["/s1", "x"]}})


# --- synthetic worlds for `select --check 5/6/7/8`

def _wav(path, seconds, sr=24000):
    import numpy as np
    import soundfile as sf
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(sr * seconds)), sr, "PCM_16")


def _pins(tmp_path, monkeypatch, arms):
    """Stand-in checkpoints for the given arms (registered names), the controls' digests
    pinned to them, and an empty prior. -> ([--ckpt args], {arm: path})."""
    names = {"C0": "checkpoint_epoch=009_step=0013489.ckpt", "S1": "checkpoint_epoch=009_step=0009209.ckpt",
             "D1": sb.FINAL_CKPT["D1"], "G": sb.FINAL_CKPT["G"], "F1": sb.FINAL_CKPT["F1"],
             "E": sb.FINAL_CKPT["E"]}
    out, paths = [], {}
    for a in arms:
        ck = tmp_path / "ck" / a / names[a]
        ck.parent.mkdir(parents=True)
        ck.write_bytes(a.encode())
        out += ["--ckpt", "%s=%s" % (a, ck)]
        paths[a] = ck
    monkeypatch.setattr(rs.sb, "CONTROL_SHA256",
                        {a: rs.sha256(str(paths[a])) if a in paths else "-" for a in ("C0", "S1")})
    (tmp_path / "prior").mkdir()
    (tmp_path / "prior" / "p.json").write_text(json.dumps({"items": []}))
    prior = [str(tmp_path / "prior" / "*.json")]
    monkeypatch.setattr(rs, "default_prior", lambda check: prior)
    monkeypatch.setattr(rs, "ROOT", str(tmp_path / "root"))
    monkeypatch.setattr(rs, "DATA", str(tmp_path / "data"))
    return out, paths


def _pin_input(monkeypatch, name, path):
    monkeypatch.setattr(rs.sb, "INPUT_SHA256", dict(rs.sb.INPUT_SHA256, **{name: rs.sha256(path)}))


def _vctk_world(tmp_path, monkeypatch, n_voices, *, clips=3, in_d1=None, heard=()):
    """VCTK voices p301.. with `clips` 1 s mic1 recordings each in a zip, the VITS raw and
    cleaned filelists, D1's train filelist (rows `in_d1(voice, k)` is true for; default all)
    and an HNR file. Returns argv for `select --check 5`."""
    import numpy as np
    import soundfile as sf
    zpath = tmp_path / "vctk.zip"
    raw, cleaned, d1, hnr = [], [], [], {}
    with zipfile.ZipFile(zpath, "w") as z:
        for i in range(n_voices):
            spk = "p%d" % (301 + i)
            hnr[spk] = 2.0 + i / 10
            for k in range(clips):
                utt = "%s_%03d" % (spk, k)
                b = io.BytesIO()
                sf.write(b, np.zeros(48000), 48000, format="FLAC")
                z.writestr(rs.rfs.vctk_path(utt), b.getvalue())
                raw.append("DUMMY/%s/%s.wav|%d|Text %d." % (spk, utt, i, k))
                ph = "tˈɛkst %s." % "abcdefgh"[k]
                cleaned.append("DUMMY/%s/%s.wav|%d|%s" % (spk, utt, i, ph))
                if in_d1 is None or in_d1(spk, k):
                    d1.append("%s/%s/%s.wav|%d|%s" % (sd.VCTK_22K_ROOT, spk, utt, i,
                                                      sd.upstream_as_ours(ph)))
    for name, lines in (("raw.txt", raw), ("cleaned.txt", cleaned)):
        (tmp_path / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "data" / "vctk_d1").mkdir(parents=True)
    (tmp_path / "data" / "vctk_d1" / "train.txt").write_text("\n".join(d1) + "\n",
                                                             encoding="utf-8")
    vh = tmp_path / "vh.json"
    vh.write_text(json.dumps({"vctk_hnr": dict(hnr, p999=1.0)}))
    ck, _ = _pins(tmp_path, monkeypatch, ("C0",))
    _pin_input(monkeypatch, "vctk_hnr_from", str(vh))
    if heard:
        (tmp_path / "prior" / "p.json").write_text(
            json.dumps({"items": [{"vctk": h} for h in heard]}))
    return ["select", "--check", "5"] + ck + [
        "--vctk-zip", str(zpath), "--vctk-filelist", str(tmp_path / "raw.txt"),
        "--vctk-cleaned", str(tmp_path / "cleaned.txt"), "--vctk-hnr-from", str(vh),
        "--prior-optional", "--min-seconds", "0.5", "--max-seconds", "2",
        "--target-seconds", "1.0"]


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["render_ear_staged.py"] + argv)
    rs.main()


def _out(tmp_path, monkeypatch, argv, name="speakers.json", dry=True):
    out = tmp_path / "scratch" / name
    _run(monkeypatch, argv + (["--dry-run"] if dry else []) + ["--out-speakers", str(out)])
    return json.loads(out.read_text())


def test_select_check5_draws_vctk_voices_within_d1s_subsample(tmp_path, monkeypatch):
    # p301 has one D1 clip only; p302's clip 0 is outside D1 (two left: eligible)
    def in_d1(spk, k):
        return not (spk == "p301" and k > 0) and not (spk == "p302" and k == 0)
    spec = _out(tmp_path, monkeypatch, _vctk_world(tmp_path, monkeypatch, 19, in_d1=in_d1))
    assert spec["check"] == 5 and spec["dry_run"] is True and spec["eligible"] == 18
    assert spec["ckpts"]["D1"] is None and spec["ckpts"]["C0"][1] == rs.sb.CONTROL_SHA256["C0"]
    voices = [it["vctk"] for it in spec["items"]]
    assert voices == ["p%d" % i for i in range(302, 320)]           # all 18, HNR order
    p302 = spec["items"][0]["arms"]
    assert {p302["C0"]["source"], p302["D1"]["source"]} == {"p302_001", "p302_002"}
    assert set(p302["C0"]) == {"source", "seconds", "sid", "phonemes", "text"}
    assert p302["C0"]["phonemes"].startswith("tˈɛkst") and p302["C0"]["sid"] == 1
    assert spec["inputs"]["vctk_hnr_from"][1] == rs.sha256(str(tmp_path / "vh.json"))
    assert "d1_train" in spec["inputs"]


def _stand_in(tmp_path, arm, monkeypatch=None):
    """A stand-in checkpoint for a new arm, named as registered, never a control's bytes."""
    ck = tmp_path / "ck" / (arm + "x") / sb.FINAL_CKPT[arm]
    ck.parent.mkdir(parents=True)
    ck.write_bytes(("stand-in " + arm).encode())
    return ["--ckpt", "%s=%s" % (arm, ck)]


def test_check5_dry_run_items_equal_a_real_selects(tmp_path, monkeypatch):
    argv = _vctk_world(tmp_path, monkeypatch, 22)
    dry = _out(tmp_path, monkeypatch, argv, "dry.json")
    real = _out(tmp_path, monkeypatch, argv + _stand_in(tmp_path, "D1"), "real.json", dry=False)
    assert dry["dry_run"] is True and "dry_run" not in real and real["ckpts"]["D1"] is not None
    assert dry["items"] == real["items"] and len(real["items"]) == 20


def test_check6_dry_run_items_equal_a_real_selects(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 6, 24)
    (tmp_path / "prior" / "check5.json").write_text(json.dumps({"check": 5, "items": []}))
    dry = _out(tmp_path, monkeypatch, argv, "dry.json")
    real = _out(tmp_path, monkeypatch, argv + _stand_in(tmp_path, "G"), "real.json", dry=False)
    assert dry["dry_run"] is True and "dry_run" not in real and real["ckpts"]["G"] is not None
    assert dry["items"] == real["items"] and len(real["items"]) == 20


def test_check7_dry_run_items_equal_a_real_selects(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 7, 24, n_f1=30)   # writes a real check6 file
    dry = _out(tmp_path, monkeypatch, argv, "dry.json")
    real = _out(tmp_path, monkeypatch, argv + _stand_in(tmp_path, "F1"), "real.json", dry=False)
    assert dry["dry_run"] is True and "dry_run" not in real and real["ckpts"]["F1"] is not None
    assert dry["items"] == real["items"] and len(real["items"]) == 20


def test_select_refuses_a_new_arm_that_is_a_copy_of_a_control(tmp_path, monkeypatch):
    argv = _vctk_world(tmp_path, monkeypatch, 16)
    c0 = tmp_path / "ck" / "C0" / "checkpoint_epoch=009_step=0013489.ckpt"
    d1 = tmp_path / "ck" / "d1copy" / sb.FINAL_CKPT["D1"]
    d1.parent.mkdir(parents=True)
    d1.write_bytes(c0.read_bytes())
    with pytest.raises(SystemExit, match="REFUSING.*D1 checkpoint is C0's"):
        _out(tmp_path, monkeypatch, argv + ["--ckpt", "D1=%s" % d1], dry=False)


def test_select_check5_excludes_heard_vctk_voices_and_spreads_twenty(tmp_path, monkeypatch):
    argv = _vctk_world(tmp_path, monkeypatch, 30, heard=("p301", "p305"))
    spec = _out(tmp_path, monkeypatch, argv)
    assert spec["eligible"] == 28 and len(spec["items"]) == 20
    pool = {"p%d" % (301 + i): 2.0 + i / 10 for i in range(30)}
    del pool["p301"], pool["p305"]
    assert [it["vctk"] for it in spec["items"]] == rs.lb.spread(pool, 20)


def test_select_check5_with_fifteen_eligible_voices_is_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="15 eligible"):
        _out(tmp_path, monkeypatch, _vctk_world(tmp_path, monkeypatch, 15))


def test_select_check5_refuses_an_unregistered_c0(tmp_path, monkeypatch):
    argv = _vctk_world(tmp_path, monkeypatch, 16)
    monkeypatch.setattr(rs.sb, "CONTROL_SHA256", {"C0": "x", "S1": "y"})
    with pytest.raises(SystemExit, match="C0"):
        _out(tmp_path, monkeypatch, argv)


def test_select_check5_refuses_an_unregistered_hnr_input(tmp_path, monkeypatch):
    argv = _vctk_world(tmp_path, monkeypatch, 16)
    monkeypatch.setattr(rs.sb, "INPUT_SHA256", dict(rs.sb.INPUT_SHA256, vctk_hnr_from="x"))
    with pytest.raises(SystemExit, match="vctk-hnr-from"):
        _out(tmp_path, monkeypatch, argv)


def test_select_check5_refuses_a_d1_row_that_does_not_match_the_vits_filelists(tmp_path, monkeypatch):
    argv = _vctk_world(tmp_path, monkeypatch, 16)
    p = tmp_path / "data" / "vctk_d1" / "train.txt"
    p.write_text(p.read_text(encoding="utf-8").replace("|0|", "|5|", 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="D1"):
        _out(tmp_path, monkeypatch, argv)


def _libri_world(tmp_path, monkeypatch, check, n_s1, *, symbol=None, n_f1=0, f1_hnr=None,
                 f1_f0=None, g_bad=False, e_bad=None):
    """S1 voices 1..n_s1 (v7 = S1 index = LibriTTS id), each with 3 one-second train
    recordings; `symbol(spk, k)` says whether recording k carries ᵻ (default: all). F1 voices
    j = 0..n_f1-1: v7 index 1000+j, LibriTTS id 5000+j, F1 table index j, train-other-500,
    3 recordings each. HNR: S1 voice s at 2 + s/10; F1 voice j at `f1_hnr(j)` (default the
    same as S1 voice j+1). Returns argv for `select --check <check>`."""
    base = str(tmp_path / "LibriTTS_R")
    rows, s1, g, e, f1 = {}, {}, {}, {}, {}
    s1h, allh = {}, {}
    for s in range(1, n_s1 + 1):
        s1h[str(s)] = {"hnr": round(2 + s / 10, 3), "f0": 100.0 + s,
                       "partition": "train-clean-100"}
        for k in range(3):
            w = "%s%strain-clean-100/%d/77/%d_77_%06d_000000.wav" % (base, R24, s, s, k)
            _wav(w, 1.0)
            ph = "pɚmˈɪɾᵻd %d" % k if (symbol is None or symbol(s, k)) else "pɚmˈɪt %d" % k
            rows[w] = (s, ph)
            s1[w.replace(R24, R22)] = (s, ph)
            g[w.replace(R24, R22)] = (s, ph.replace("ᵻ", "ᵊ" if not g_bad else "ᵻ"))
            e[w.replace(R24, R22)] = (
                (s, "ɛspˈiːkᵊ %s." % "abc"[k]) if e_bad is None else e_bad(s, k))
    table = {"n_spks": n_f1, "libritts_id_to_index": {}, "v7_index": {}}
    for j in range(n_f1):
        lid, v = 5000 + j, 1000 + j
        table["libritts_id_to_index"][str(lid)] = j
        table["v7_index"][str(lid)] = v
        allh[str(v)] = {"hnr": f1_hnr(j) if f1_hnr else round(2 + (j + 1) / 10, 3),
                        "f0": f1_f0(j) if f1_f0 else 100.0 + j + 1,
                        "partition": "train-other-500"}
        for k in range(3):
            w = "%s%strain-other-500/%d/77/%d_77_%06d_000000.wav" % (base, R24, lid, lid, k)
            _wav(w, 1.0)
            f1[w.replace(R24, R22)] = (j, "fˈɛn %d" % k)
    tables = {("libritts_r_vat", "train_op.txt"): rows, ("libritts_r_22k", "train_op.txt"): s1,
              ("libritts_r_22k", "train_g.txt"): g, ("libritts_r_22k", "train_e.txt"): e,
              ("f1_dense", "train_op.txt"): f1}
    monkeypatch.setattr(rs, "corpus_rows", lambda corpus, name="train_op.txt": tables[(corpus, name)])
    for corpus, name in tables:                 # select records each filelist's sha256
        (tmp_path / "data" / corpus).mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / corpus / name).write_text(corpus + name)
    monkeypatch.setattr(rs.sd, "to_22k", lambda w: w.replace(R24, R22))
    monkeypatch.setattr(rs.sd, "libri_r_22k", lambda w: w.replace(R24, R22))
    monkeypatch.setattr(rs.sd, "libri_r_24k", lambda w: w.replace(R22, R24))
    arms = sb.CHECKS[check]["arms"]
    ck, _ = _pins(tmp_path, monkeypatch, ("S1",))
    (tmp_path / "data" / "f1_dense").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "f1_dense" / "speakers.json").write_text(json.dumps(table))
    sh, ah = tmp_path / "s1h.json", tmp_path / "allh.json"
    sh.write_text(json.dumps({"speakers": s1h}))
    ah.write_text(json.dumps({"speakers": dict(allh, **{"999": {"hnr": 1, "f0": 1}})}))
    _pin_input(monkeypatch, "hnr_json", str(ah))
    _pin_input(monkeypatch, "s1_hnr_json", str(sh))
    assert arms[0] == "S1"
    if check == 7:                      # check 7 is drawn after check 6
        (tmp_path / "prior" / "check6.json").write_text(json.dumps({"check": 6, "items": []}))
    elif check == 8:                    # check 8 is drawn after check 7
        (tmp_path / "prior" / "check7.json").write_text(json.dumps({"check": 7, "items": []}))
    return ["select", "--check", str(check)] + ck + [
        "--s1-hnr-json", str(sh), "--hnr-json", str(ah), "--prior-optional",
        "--min-seconds", "0.5", "--max-seconds", "2", "--target-seconds", "1.0"]


def test_the_inputs_each_hum_fix_check_must_have_registered():
    assert rs.HUM_FIX_INPUTS == {5: ("vctk_hnr_from",), 6: ("s1_hnr_json",),
                                 7: ("hnr_json", "s1_hnr_json"), 8: ("s1_hnr_json",)}


@pytest.mark.parametrize("check", [6, 7, 8])
def test_select_refuses_an_s1_hnr_file_that_is_not_the_registered_one(tmp_path, monkeypatch, check):
    argv = _libri_world(tmp_path, monkeypatch, check, 16, n_f1=16)
    sh = tmp_path / "s1h.json"
    sh.write_text(sh.read_text() + " ")             # a different file, same content
    with pytest.raises(SystemExit, match="REFUSING.*s1-hnr-json"):
        _out(tmp_path, monkeypatch, argv)


def test_select_check6_takes_only_voices_with_two_symbol_bearing_clips(tmp_path, monkeypatch):
    # voice 1: one ᵻ clip; voice 2: two
    argv = _libri_world(tmp_path, monkeypatch, 6, 18,
                        symbol=lambda s, k: not (s == 1 and k > 0) and not (s == 2 and k == 0))
    spec = _out(tmp_path, monkeypatch, argv)
    assert spec["eligible"] == 17 and [it["libri"] for it in spec["items"]] == list(range(2, 19))
    u = spec["items"][0]
    assert set(u["arms"]) == {"S1", "G"} and u["arms"]["S1"]["source"] != u["arms"]["G"]["source"]
    assert all("ᵻ" in a["phonemes"] for it in spec["items"] for a in it["arms"].values())
    assert u["libri_hnr"] == 2.2 and u["partition"] == "train-clean-100" and u["libri_f0"] == 102.0
    assert spec["inputs"]["s1_hnr_json"][1] == rs.sha256(str(tmp_path / "s1h.json"))
    assert spec["ckpts"]["G"] is None


def test_select_check6_refuses_a_g_row_that_is_not_s1s_with_the_symbol_moved(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 6, 16, g_bad=True)
    with pytest.raises(SystemExit, match="G"):
        _out(tmp_path, monkeypatch, argv)


def test_select_check7_pairs_s1_voices_with_f1_voices_by_hnr_and_f0(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 7, 18, n_f1=20)
    spec = _out(tmp_path, monkeypatch, argv)
    assert spec["eligible"] == 18 and spec["f1_eligible"] == 20 and len(spec["items"]) == 18
    for it in spec["items"]:
        s, f = it["libri"], it["f1"]
        assert f["libri"] == 1000 + s - 1 and f["libritts_id"] == 5000 + s - 1
        assert f["f1_sid"] == s - 1 and f["partition"] == "train-other-500"
        assert "/train-other-500/%d/" % f["libritts_id"] in it["arms"]["F1"]["source"]
        assert it["arms"]["F1"]["row"] == it["arms"]["F1"]["source"].replace(R24, R22)
        assert it["arms"]["F1"]["phonemes"].startswith("fˈɛn")
        assert "/train-clean-100/%d/" % s in it["arms"]["S1"]["source"]
    assert sb.heard_libri([spec]) == set(range(1, 19)) | set(range(1000, 1018))


def test_select_check7_skips_an_s1_voice_without_a_candidate(tmp_path, monkeypatch):
    # F1 voice 0 (HNR 2.1) is moved out of every window: S1 voice 1 (HNR 2.1) takes the next
    # nearest F0 within 0.5 dB, and so on up; the last S1 voice is left without a partner.
    argv = _libri_world(tmp_path, monkeypatch, 7, 17, n_f1=17,
                        f1_hnr=lambda j: 9.0 if j == 0 else round(2 + (j + 1) / 10, 3))
    spec = _out(tmp_path, monkeypatch, argv)
    assert len(spec["items"]) == 16
    assert spec["unpaired"] == [17]
    assert [it["unit"] for it in spec["items"]] == list(range(16))


def test_select_check7_with_fewer_than_sixteen_pairs_is_refused(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 7, 16, n_f1=15,
                        f1_hnr=lambda j: round(2 + (j + 1) / 10, 3))
    with pytest.raises(SystemExit, match="15 pairs"):
        _out(tmp_path, monkeypatch, argv)


def test_select_check7_excludes_check6s_voices_given_as_an_extra_prior(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 7, 20, n_f1=24)
    b = tmp_path / "scratch" / "check6_dryrun_speakers.json"
    b.parent.mkdir()
    b.write_text(json.dumps({"check": 6, "dry_run": True,
                             "items": [{"unit": 0, "libri": 3}, {"unit": 1, "libri": 4}]}))
    spec = _out(tmp_path, monkeypatch, argv + ["--prior", str(tmp_path / "prior" / "*.json"),
                                              str(b)])
    assert not {3, 4} & {it["libri"] for it in spec["items"]} and spec["eligible"] == 18


def test_select_check7_dry_run_is_reproducible(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 7, 22, n_f1=30)
    a = _out(tmp_path, monkeypatch, argv, "a.json")
    b = _out(tmp_path, monkeypatch, argv, "b.json")
    assert a["items"] == b["items"] and a["rule"] == b["rule"]


def test_select_check7_refuses_an_unregistered_s1_checkpoint(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 7, 16, n_f1=16)
    monkeypatch.setattr(rs.sb, "CONTROL_SHA256", {"C0": "x", "S1": "y"})
    with pytest.raises(SystemExit, match="S1"):
        _out(tmp_path, monkeypatch, argv)


def test_check8_dry_run_items_equal_a_real_selects_and_store_each_arms_phonemes(
        tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 8, 24, symbol=lambda s, k: k == 0)
    dry = _out(tmp_path, monkeypatch, argv, "dry.json")
    real = _out(tmp_path, monkeypatch, argv + _stand_in(tmp_path, "E"), "real.json", dry=False)
    assert dry["dry_run"] is True and "dry_run" not in real and real["ckpts"]["E"] is not None
    assert dry["items"] == real["items"] and len(real["items"]) == 20
    assert "libritts_r_22k/train_e.txt" in real["inputs"]
    for it in real["items"]:
        assert set(it["arms"]) == {"S1", "E"}
        assert it["arms"]["S1"]["source"] != it["arms"]["E"]["source"]
        assert it["arms"]["S1"]["phonemes"].startswith("pɚmˈɪ")
        assert it["arms"]["E"]["phonemes"].startswith("ɛspˈiːkᵊ")
        assert sd.G_FROM not in it["arms"]["E"]["phonemes"]


@pytest.mark.parametrize("bad_fn", [
    lambda s, k: (s + 1, "ɛspˈiːkᵊ"),          # speaker mismatch
    lambda s, k: (s, ""),                       # empty phonemes
    lambda s, k: (s, "pɚmˈɪɾᵻd"),               # unmapped ᵻ (id 176)
    lambda s, k: (s, "ɛspˈiːk~"),               # symbol outside ours_symbols
])
def test_select_check8_refuses_an_invalid_or_unmapped_e_row(tmp_path, monkeypatch, bad_fn):
    argv = _libri_world(tmp_path, monkeypatch, 8, 16, e_bad=bad_fn)
    with pytest.raises(SystemExit, match="REFUSING.*E's row"):
        _out(tmp_path, monkeypatch, argv)


# ---------------------------------------------------------------- unblind

import csv  # noqa: E402

import unblind_ear_staged as ub  # noqa: E402


def _bench(tmp_path, check, model_sev, n=16, rt=1):
    """A key + verdicts: arm `a` rated model_sev[a] over a round trip of rt[a] (int or dict)."""
    arms = sb.CHECKS[check]["arms"]
    items, rows, i = {}, [], 0
    for u in range(n):
        for a in arms:
            k = "item_%02d" % i
            i += 1
            r = rt[a] if isinstance(rt, dict) else rt
            t = {"arm": a, "unit": u, "hnr": 3.0 + u / 10, "A_label": a, "B_label": a + "rt",
                 "source": "x", "rt_source": "y"}
            if check == 7:
                t.update(spk=(40 + u if a == "S1" else 4000 + u), f0=110.0 + u,
                         partition="train-clean-100" if a == "S1" else "train-other-500")
                if a == "F1":
                    t.update(f1_sid=u, libritts_id=5000 + u)
            elif check == 5:
                t["vctk_spk"] = "p%d" % (300 + u)
            else:
                t["spk"] = 40 + u
            items[k] = t
            rows.append({"item": k, "sev_a": model_sev[a], "sev_b": r})
    key = tmp_path / "k.json"
    key.write_text(json.dumps({"check": check, "items": items}))
    _folder(tmp_path / "t", rows)
    argv = ["--check", str(check), "--key", str(key), "--test", str(tmp_path / "t")]
    if check in (5, 6, 7):              # checks 5-7: three-folder session
        _meta(tmp_path / "t", check)
        rest = [c for c in (5, 6, 7) if c != check]
        others = [_folder(tmp_path / n, [{"item": "item_%02d" % i, "sev_a": 2, "sev_b": 1}
                                         for i in range(3)], check=c)
                  for n, c in zip(("t2", "t3"), rest)]
        argv += ["--session-tests", str(tmp_path / "t")] + [str(o) for o in others]
    elif check == 8:                    # check 8: single-folder session
        _meta(tmp_path / "t", check)
    return argv


def _meta(path, check):
    """The folder's render_meta.json, which names the check it was rendered for."""
    (path / "render_meta.json").write_text(json.dumps({"check": check, "pairs": 3}))


def _folder(path, rows, check=None):
    """A test folder: items.json naming each row's item, and verdicts.csv with the rows (and,
    given a check, the render_meta.json of that check's bench)."""
    (path / "verdicts").mkdir(parents=True)
    if check is not None:
        _meta(path, check)
    (path / "items.json").write_text(json.dumps(
        {"test": path.name, "items": [{"id": r["item"], "A": "a", "B": "b"} for r in rows]}))
    with (path / "verdicts" / "verdicts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "sev_a", "sev_b"])
        w.writeheader()
        w.writerows(rows)
    return path


def test_check5_unblinds_to_shown_with_one_deciding_p(tmp_path, capsys):
    r = ub.main(_bench(tmp_path, 5, {"C0": 1, "D1": 3}))
    out = capsys.readouterr().out
    assert r["outcome"] == "shown" and r["shown"] == ["clips_per_speaker"]
    assert "OUTCOME: SHOWN" in out and "fewer clips per speaker adds hum" in out
    assert ("three p-values reported (D1 − C0, D1's excess, C0's excess), one of them deciding "
            "the reading (D1 − C0); uncorrected") in out
    assert "D1-C0" in out


def test_check5_not_shown_is_not_absent(tmp_path, capsys):
    r = ub.main(_bench(tmp_path, 5, {"C0": 2, "D1": 2}))
    out = capsys.readouterr().out
    assert r["outcome"] == "not_shown" and "Not shown is not absent" in out


def test_check5_floors_are_c0s(tmp_path, capsys):
    r = ub.main(_bench(tmp_path, 5, {"C0": 3, "D1": 4}, rt={"C0": 0, "D1": 1}))
    assert r["outcome"] == "invalid" and "within 1 point of C0's" in capsys.readouterr().out


@pytest.mark.parametrize("check,fix,text", [
    (6, "G", "the remap reduces the hum"), (7, "F1", "dense speakers reduce the hum"),
    (8, "E", "replacing op_g2p with VITS's espeak frontend reduces the hum")])
def test_checks_6_7_and_8_unblind_to_shown(tmp_path, capsys, check, fix, text):
    r = ub.main(_bench(tmp_path, check, {"S1": 3, fix: 1}))
    out = capsys.readouterr().out
    assert r["outcome"] == "shown" and "OUTCOME: SHOWN" in out and text in out
    shown = fix + "′" if fix == "G" else fix
    assert ("three p-values reported (S1's excess, S1 − %s, %s's excess), two of them deciding "
            "the reading (S1's excess and S1 − %s); uncorrected" % (shown, shown, shown)) in out


@pytest.mark.parametrize("check,fix", [(6, "G"), (7, "F1"), (8, "E")])
def test_checks_6_7_and_8_invalid_without_s1s_excess(tmp_path, capsys, check, fix):
    r = ub.main(_bench(tmp_path, check, {"S1": 1, fix: 0}))
    out = capsys.readouterr().out
    assert r["outcome"] == "invalid" and "no reading" in out and "S1's own excess" in out


@pytest.mark.parametrize("check,fix", [(5, "D1"), (6, "G"), (7, "F1"), (8, "E")])
def test_fifteen_voices_rated_in_both_arms_is_invalid(tmp_path, check, fix):
    sev = {"C0": 1, "D1": 3} if check == 5 else {"S1": 3, fix: 1}
    assert ub.main(_bench(tmp_path, check, sev, n=15))["outcome"] == "invalid"


@pytest.mark.parametrize("check,arm", [(5, "D1"), (6, "G"), (7, "F1"), (8, "E")])
def test_an_arm_with_nothing_rated_is_invalid_not_a_crash(tmp_path, capsys, check, arm):
    argv = _bench(tmp_path, check, {"C0": 1, "D1": 3} if check == 5 else {"S1": 3, arm: 1})
    p = tmp_path / "t" / "verdicts" / "verdicts.csv"
    key = json.loads((tmp_path / "k.json").read_text())["items"]
    kept = [ln for ln in p.read_text().splitlines()
            if ln.startswith("item,") or key[ln.split(",")[0]]["arm"] != arm]
    p.write_text("\n".join(kept) + "\n")
    r = ub.main(argv + ["--session-closed"])
    out = capsys.readouterr().out
    assert r["outcome"] == "invalid" and "OUTCOME: INVALID" in out
    assert "fewer than 16 voices rated in both arms" in out


def test_check7_prints_each_pairs_partition(tmp_path, capsys):
    ub.main(_bench(tmp_path, 7, {"S1": 3, "F1": 1}))
    out = capsys.readouterr().out
    assert "PAIRS" in out
    line = [ln for ln in out.splitlines() if ln.startswith("   3  ")][-1]
    assert "S1 v7 43" in line and "train-clean-100" in line
    assert "F1 v7 4003" in line and "LibriTTS 5003" in line and "train-other-500" in line


def test_the_unblind_tool_takes_checks_5_to_8_only_with_their_own_keys(tmp_path):
    argv = _bench(tmp_path, 6, {"S1": 3, "G": 1})
    argv[1] = "7"
    with pytest.raises(SystemExit, match="check 6"):
        ub.main(argv)


# --- session-wide unblinding: checks 5-7 are read only when the whole session is rated

def _unrate(path, item):
    """Blank item's sev_b in a folder's verdicts.csv."""
    p = path / "verdicts" / "verdicts.csv"
    rows = list(csv.DictReader(p.open(newline="")))
    for r in rows:
        if r["item"] == item:
            r["sev_b"] = ""
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "sev_a", "sev_b"])
        w.writeheader()
        w.writerows(rows)


@pytest.mark.parametrize("check,fix", [(5, "D1"), (6, "G"), (7, "F1")])
@pytest.mark.parametrize("folder", ["t", "t2", "t3"])
def test_an_unrated_item_in_any_of_the_three_folders_refuses_the_unblinding(
        tmp_path, capsys, check, fix, folder):
    sev = {"C0": 1, "D1": 3} if check == 5 else {"S1": 3, fix: 1}
    argv = _bench(tmp_path, check, sev)
    _unrate(tmp_path / folder, "item_01")
    with pytest.raises(SystemExit, match="REFUSING.*1 of .* items.*not fully rated.*--session-closed"):
        ub.main(argv)
    assert "OUTCOME" not in capsys.readouterr().out
    r = ub.main(argv + ["--session-closed"])             # the session is closed: left out
    assert r["outcome"] in ("shown", "not_shown", "invalid")
    assert "OUTCOME" in capsys.readouterr().out


def test_check8_is_a_single_folder_session_and_refuses_unrated_or_session_tests(
        tmp_path, capsys):
    argv = _bench(tmp_path, 8, {"S1": 3, "E": 1}, n=17)
    with pytest.raises(SystemExit, match="REFUSING.*--session-tests.*single-folder"):
        ub.main(argv + ["--session-tests", str(tmp_path / "t"), "x", "y"])
    _unrate(tmp_path / "t", "item_01")
    with pytest.raises(SystemExit, match="REFUSING.*1 of .* items.*not fully rated.*--session-closed"):
        ub.main(argv)
    assert "OUTCOME" not in capsys.readouterr().out
    r = ub.main(argv + ["--session-closed"])
    assert r["outcome"] == "shown" and "OUTCOME" in capsys.readouterr().out


def test_an_item_missing_from_the_verdicts_is_unrated(tmp_path):
    argv = _bench(tmp_path, 6, {"S1": 3, "G": 1})
    p = tmp_path / "t2" / "verdicts" / "verdicts.csv"
    p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")      # item_02 has no row
    with pytest.raises(SystemExit, match="REFUSING.*not fully rated"):
        ub.main(argv)


@pytest.mark.parametrize("check,fix", [(5, "D1"), (6, "G"), (7, "F1"), (8, "E")])
def test_a_fully_rated_session_unblinds_without_session_closed(tmp_path, check, fix):
    sev = {"C0": 1, "D1": 3} if check == 5 else {"S1": 3, fix: 1}
    assert ub.main(_bench(tmp_path, check, sev))["outcome"] == "shown"


def test_checks_5_to_7_need_their_three_folders_and_the_one_read_is_among_them(tmp_path):
    argv = _bench(tmp_path, 6, {"S1": 3, "G": 1})
    i = argv.index("--session-tests")
    with pytest.raises(SystemExit):
        ub.main(argv[:i])                                  # no --session-tests
    with pytest.raises(SystemExit, match="REFUSING.*--test"):
        ub.main(argv[:i] + ["--session-tests"] + argv[i + 2:] + [argv[-1]])   # --test is not there


def test_a_session_folder_without_items_json_is_refused(tmp_path):
    argv = _bench(tmp_path, 6, {"S1": 3, "G": 1})
    (tmp_path / "t3" / "items.json").unlink()
    with pytest.raises(SystemExit, match="REFUSING.*items.json"):
        ub.main(argv)


def _session_argv(tmp_path, check=6):
    argv = _bench(tmp_path, check, {"S1": 3, "G": 1} if check == 6 else {"S1": 3, "F1": 1})
    return argv, argv.index("--session-tests")


def test_the_three_session_folders_must_be_distinct(tmp_path):
    argv, i = _session_argv(tmp_path)
    t = str(tmp_path / "t")
    with pytest.raises(SystemExit, match="REFUSING.*distinct"):
        ub.main(argv[:i + 1] + [t, t, argv[-1]])
    # the same folder spelled another way is still the same folder
    with pytest.raises(SystemExit, match="REFUSING.*distinct"):
        ub.main(argv[:i + 1] + [t, str(tmp_path / "t2"), str(tmp_path / "x" / ".." / "t2")])


def test_a_session_folder_rendered_for_another_check_is_refused(tmp_path):
    argv, i = _session_argv(tmp_path)                     # t=6, t2=5, t3=7
    _meta(tmp_path / "t3", 6)                             # two 6s, no 7
    with pytest.raises(SystemExit, match="REFUSING.*5, 6 and 7"):
        ub.main(argv)


def test_a_session_folder_without_render_meta_is_refused(tmp_path):
    argv, i = _session_argv(tmp_path)
    (tmp_path / "t2" / "render_meta.json").unlink()
    with pytest.raises(SystemExit, match="REFUSING.*render_meta.json"):
        ub.main(argv)


def test_a_render_meta_without_a_check_is_refused(tmp_path):
    argv, i = _session_argv(tmp_path)
    (tmp_path / "t2" / "render_meta.json").write_text(json.dumps({"pairs": 3}))
    with pytest.raises(SystemExit, match="REFUSING.*5, 6 and 7"):
        ub.main(argv)


def test_the_test_folder_must_be_the_check_being_read(tmp_path):
    argv, i = _session_argv(tmp_path)                     # t=6, t2=5, t3=7
    # reading --check 6 against t2 (a check-5 folder): the three are fine, the pairing is not
    argv[argv.index("--test") + 1] = str(tmp_path / "t2")
    with pytest.raises(SystemExit, match="REFUSING.*--test.*check 5.*--check 6"):
        ub.main(argv)


def test_checks_1_to_4_take_no_session_options(tmp_path):
    argv = _bench(tmp_path, 4, {"S1": 3, "L": 1})
    with pytest.raises(SystemExit):
        ub.main(argv + ["--session-closed"])
    with pytest.raises(SystemExit):
        ub.main(argv + ["--session-tests", str(tmp_path / "t"), "x", "y"])


def test_check6_excess_text_names_g_prime(tmp_path, capsys):
    ub.main(_bench(tmp_path, 6, {"S1": 3, "G": 1}))
    out = capsys.readouterr().out
    assert ("three p-values reported (S1's excess, S1 − G′, G′'s excess), two of them deciding "
            "the reading (S1's excess and S1 − G′); uncorrected") in out


# ---------------------------------------------------------------- selection order

def test_check7_refuses_without_check6s_speakers_file_among_the_priors(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 7, 16, n_f1=16)
    (tmp_path / "prior" / "check6.json").unlink()
    with pytest.raises(SystemExit, match="check 6"):
        _out(tmp_path, monkeypatch, argv)


def test_check8_refuses_without_check7s_speakers_file_among_the_priors(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 8, 16)
    (tmp_path / "prior" / "check7.json").unlink()
    with pytest.raises(SystemExit, match="check 7"):
        _out(tmp_path, monkeypatch, argv)


def test_a_real_check7_selection_does_not_accept_check6s_dry_run(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 7, 16, n_f1=16)
    (tmp_path / "prior" / "check6.json").write_text(
        json.dumps({"check": 6, "dry_run": True, "items": []}))
    _out(tmp_path, monkeypatch, argv)                           # a dry run may use it
    f1 = tmp_path / "ck" / "F1x" / sb.FINAL_CKPT["F1"]
    f1.parent.mkdir(parents=True)
    f1.write_bytes(b"f1")
    with pytest.raises(SystemExit, match="check 6"):
        _out(tmp_path, monkeypatch, argv + ["--ckpt", "F1=%s" % f1], "real.json", dry=False)


def test_a_real_check6_selection_needs_check5s_file_and_its_dry_run_does_not(tmp_path, monkeypatch):
    argv = _libri_world(tmp_path, monkeypatch, 6, 16)
    _out(tmp_path, monkeypatch, argv)
    g = tmp_path / "ck" / "Gx" / sb.FINAL_CKPT["G"]
    g.parent.mkdir(parents=True)
    g.write_bytes(b"g")
    real = argv + ["--ckpt", "G=%s" % g]
    with pytest.raises(SystemExit, match="check 5"):
        _out(tmp_path, monkeypatch, real, "real.json", dry=False)
    (tmp_path / "prior" / "check5.json").write_text(json.dumps({"check": 5, "items": []}))
    assert len(_out(tmp_path, monkeypatch, real, "real.json", dry=False)["items"]) == 16


def test_check5_needs_no_earlier_hum_fix_file(tmp_path, monkeypatch):
    spec = _out(tmp_path, monkeypatch, _vctk_world(tmp_path, monkeypatch, 16))
    assert len(spec["items"]) == 16

