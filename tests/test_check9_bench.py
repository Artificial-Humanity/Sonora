"""Check 9 (Phase 1S Step 2, arm Q) is the pre-registered bench: its spec, its (S1, F1, Q) triplet
pairing with the 0.5 dB window widened to 1.0 dB per arm and recorded, its controls, and its reading
(six tests, three deciding, the two shown readings and the validity rule)."""
import hashlib
from fractions import Fraction

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()

import staged_bench as sb  # noqa: E402
import render_ear_staged  # noqa: E402


def _voices(n, hnr0, f00, step_hnr=0.0, step_f0=0.0):
    return {i: (round(hnr0 + i * step_hnr, 3), round(f00 + i * step_f0, 3)) for i in range(n)}


def test_the_check9_spec_is_the_registered_one():
    c = sb.CHECKS[9]
    assert c["arms"] == ("S1", "F1", "Q")
    assert c["libri_arms"] == ("S1", "F1") and c["q_arms"] == ("Q",)
    assert c["contrasts"] == (("S1", "Q"), ("F1", "Q"), ("S1", "F1"))
    assert c["floor_ref"] == "S1" and c["min_units"] == 16
    assert 9 in sb.HUM_FIX_CHECKS
    assert sb.FAMILY["Q"] == "q_22"


def test_the_f1_control_is_pinned_for_check9_only_and_is_a_sha256():
    want = sb.CHECK9_F1_SHA256
    assert len(want) == 64 and int(want, 16) >= 0
    # C0 and S1 are the hum-fix controls; F1 is trained by check 7, so it is a control only here.
    assert set(sb.CONTROL_SHA256) == {"C0", "S1"}


def test_registered_checkpoints_refuse_a_q_that_copies_a_control():
    digests = {"S1": sb.CONTROL_SHA256["S1"], "F1": sb.CHECK9_F1_SHA256,
               "Q": sb.CONTROL_SHA256["S1"]}
    with pytest.raises(ValueError, match="Q checkpoint is S1's"):
        sb.check_registered_ckpts(9, digests)


def test_check9_accepts_the_controls_and_a_fresh_q():
    digests = {"S1": sb.CONTROL_SHA256["S1"], "F1": sb.CHECK9_F1_SHA256,
               "Q": hashlib.sha256(b"q").hexdigest()}
    sb.check_registered_ckpts(9, digests)


def test_check7s_new_f1_is_not_compared_with_the_check9_control():
    digests = {"S1": sb.CONTROL_SHA256["S1"], "F1": hashlib.sha256(b"f1").hexdigest()}
    sb.check_registered_ckpts(7, digests)


def test_the_triplets_pair_within_the_strict_window_and_consume_each_voice_once():
    s1 = {1: (10.0, 120.0), 2: (11.0, 200.0)}
    f1 = {7: (10.4, 118.0, 501), 8: (10.9, 205.0, 400)}
    q = {3: (10.2, 121.0, 3), 4: (10.95, 199.0, 4)}
    out = sb.pair_q_triplets(s1, f1, q, min_triplets=2)
    assert [(t["s1"], t["f1"], t["q"]) for t in out] == [(1, 7, 3), (2, 8, 4)]
    assert all(t["widened"] == {"F1": False, "Q": False} for t in out)
    assert len({t["f1"] for t in out}) == 2 and len({t["q"] for t in out}) == 2


def test_an_arm_without_a_window_match_widens_to_one_db_and_records_it():
    s1 = {1: (10.0, 120.0), 2: (11.0, 200.0)}
    f1 = {7: (10.4, 118.0, 501), 8: (10.9, 205.0, 400)}
    q = {3: (10.7, 121.0, 3), 4: (10.95, 199.0, 4)}      # S1 1's Q is 0.7 dB away
    out = sb.pair_q_triplets(s1, f1, q, min_triplets=1)
    first = out[0]
    assert (first["s1"], first["q"]) == (1, 3)
    assert first["widened"] == {"F1": False, "Q": True}


def test_an_s1_voice_with_no_partner_within_one_db_is_skipped_and_consumes_nothing():
    s1 = {1: (10.0, 120.0), 2: (11.0, 200.0)}
    f1 = {7: (10.4, 118.0, 501), 8: (10.9, 205.0, 400)}
    q = {3: (13.0, 121.0, 3), 4: (11.4, 199.0, 4)}       # no Q within 1 dB of S1 1
    out = sb.pair_q_triplets(s1, f1, q, min_triplets=1)
    assert [t["s1"] for t in out] == [2]
    assert out[0]["q"] == 4


def test_ties_break_on_the_smaller_hnr_gap_then_the_lower_tie_id():
    s1 = {1: (10.0, 120.0)}
    f1 = {7: (10.4, 120.0, 501), 8: (10.0, 120.0, 900)}
    q = {3: (10.0, 120.0, 3)}
    out = sb.pair_q_triplets(s1, f1, q, min_triplets=1)
    assert out[0]["f1"] == 8                              # same F0: smaller HNR gap wins


def test_fewer_than_sixteen_triplets_is_refused():
    s1 = {1: (10.0, 120.0)}
    with pytest.raises(ValueError, match="registered minimum"):
        sb.pair_q_triplets(s1, {7: (10.0, 120.0, 1)}, {3: (10.0, 120.0, 3)})


def _gaps(n, s1, f1, q):
    return {"S1": {u: s1 for u in range(n)}, "F1": {u: f1 for u in range(n)},
            "Q": {u: q for u in range(n)}}


FLOORS = {"S1": Fraction(1, 2), "F1": Fraction(3, 5), "Q": Fraction(1, 2)}


def test_a_synthetic_corpus_that_hums_less_is_shown_and_valid():
    r = sb.reading(9, _gaps(20, 1.0, 0.5, 0.0), FLOORS)
    assert r["valid"] is True and r["outcome"] == "shown"
    assert "synthetic_reduces_hum" in r["shown"]
    assert "synthetic_beats_dense_real" in r["shown"]
    assert r["n_tests"] == 6 and r["n_deciding"] == 3


def test_no_difference_is_not_shown_but_still_valid():
    r = sb.reading(9, _gaps(20, 1.0, 1.0, 1.0), FLOORS)
    assert r["outcome"] == "not_shown" and r["shown"] == []


def test_fewer_than_sixteen_units_rated_in_all_three_arms_is_invalid():
    gaps = _gaps(20, 1.0, 0.5, 0.0)
    gaps["Q"] = {u: 0.0 for u in range(10)}
    r = sb.reading(9, gaps, FLOORS)
    assert r["valid"] is False and r["outcome"] == "invalid"


def test_floors_not_comparable_is_invalid():
    floors = {"S1": Fraction(1, 2), "F1": Fraction(2), "Q": Fraction(1, 2)}
    r = sb.reading(9, _gaps(20, 1.0, 0.5, 0.0), floors)
    assert r["valid"] is False


def test_s1_own_excess_must_be_positive_for_validity():
    r = sb.reading(9, _gaps(20, 0.0, 0.0, 0.0), FLOORS)
    assert r["valid"] is False


def test_require_drawn_before_needs_check8_for_check9():
    with pytest.raises(SystemExit, match="check 8"):
        render_ear_staged.require_drawn_before(9, [], dry_run=False)
    doc8 = {"check": 8, "dry_run": False}
    render_ear_staged.require_drawn_before(9, [doc8], dry_run=False)


def test_check9_reads_only_units_rated_in_all_three_arms():
    gaps = {"S1": {0: 1.0, 1: 1.0, 2: 1.0}, "F1": {0: 0.5, 1: 0.5}, "Q": {0: 0.0, 1: 0.0}}
    r = sb.reading(9, gaps, {"S1": Fraction(1, 2), "F1": Fraction(1, 2), "Q": Fraction(1, 2)})
    assert r["level"]["S1"][2] == 2 and r["contrasts"]["S1-Q"][2] == 2
    assert r["valid"] is False                            # 2 units, not 16


def test_truth_entry_names_a_q_voice_under_q_spk_never_spk():
    it = {"unit": 0, "q": {"sid": 12, "hnr": 10.0, "f0": 120.0, "widened": {"F1": False,
                                                                           "Q": True}},
          "arms": {"Q": {"source": "/x/q.wav", "phonemes": "p"}}, "libri": 5,
          "libri_hnr": 9.0, "vctk": "p225", "vctk_hnr": 1.0}
    entry = render_ear_staged.truth_entry(9, "Q", it, [(0, "Q"), (1, "S1")])
    assert entry["q_spk"] == 12 and "spk" not in entry
    assert entry["widened"] == {"F1": False, "Q": True}


def test_hum_fix_rule_names_check9_without_an_index_error():
    args = type("A", (), {"check": 9, "min_seconds": 4.0, "max_seconds": 8.0,
                          "target_seconds": 6.0, "n": 20, "seed": 20261029})()
    text = render_ear_staged.hum_fix_rule(args, sb.CHECKS[9], 20)
    assert text.startswith("check 9 (hum-fix Q): arms S1/F1/Q")
    assert "--q-hnr-json" in text and "1.0 dB" in text


def test_truth_entry_gives_s1_in_check9_its_partition_and_f0():
    it = {"unit": 0, "libri": 5, "libri_hnr": 9.0, "libri_f0": 130.0, "partition": "train-other",
          "f1": {"libri": 7, "libritts_id": 99, "f1_sid": 3, "hnr": 9.1, "f0": 128.0,
                 "partition": "train-clean-360"},
          "q": {"sid": 12, "hnr": 9.2, "f0": 131.0, "widened": {"F1": False, "Q": False}},
          "arms": {"S1": {"source": ("/data/model-training/datasets/LibriTTS_R/train-clean-100/"
                                     "19/198/19_198_000000_000000_22k.wav")}}}
    entry = render_ear_staged.truth_entry(9, "S1", it, [(0, "S1"), (1, "F1")])
    assert entry["spk"] == 5 and entry["partition"] == "train-other" and entry["f0"] == 130.0
