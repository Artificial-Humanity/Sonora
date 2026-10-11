"""The staged-rebuild bench tools' guards (`render_ear_staged.py`, `unblind_ear_staged.py`).

The render tool's torch half runs in the container; these pin what it decides on the host.
"""
import csv
import json
from fractions import Fraction
import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import render_ear_staged as rs  # noqa: E402
import staged_bench as sb  # noqa: E402
import unblind_ear_staged as ub  # noqa: E402


def test_every_arm_has_a_checkpoint_rule_and_a_render_expectation():
    arms = {a for c in sb.CHECKS.values() for a in c["arms"]}
    assert arms == set(rs.EXPECT) == set(sb.FAMILY)
    assert arms == set(rs.FIXED) | set(rs.NEW)


def test_a_new_arm_without_a_checkpoint_is_refused():
    with pytest.raises(SystemExit, match="R"):
        rs.resolve_ckpts(("stock", "R", "derisk"), [])


def test_a_new_arm_given_a_checkpoint_short_of_ten_epochs_is_refused():
    with pytest.raises(SystemExit, match="10 epochs"):
        rs.resolve_ckpts(("stock", "R", "derisk"),
                         ["R=/x/staged_r/runs/a/checkpoints/last.ckpt"])


def test_a_fixed_arm_cannot_be_overridden():
    with pytest.raises(SystemExit, match="stock"):
        rs.resolve_ckpts(("stock", "R", "derisk"),
                         ["R=/x/checkpoint_epoch=009_step=0009209.ckpt", "stock=/y.ckpt"])


def test_resolved_checkpoints_name_every_arm():
    got = rs.resolve_ckpts(("stock", "R", "derisk"),
                           ["R=/x/checkpoint_epoch=009_step=0009209.ckpt"])
    assert got == {"stock": rs.FIXED["stock"], "derisk": rs.FIXED["derisk"],
                   "R": "/x/checkpoint_epoch=009_step=0009209.ckpt"}


def test_a_prior_pattern_that_matches_nothing_is_refused(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    assert rs.load_prior([str(tmp_path / "*.json")], [], tmp_path / "out.json") == [{}]
    with pytest.raises(SystemExit, match="matches no file"):
        rs.load_prior([str(tmp_path / "*.json"), str(tmp_path / "x.json y.json")], [],
                      tmp_path / "out.json")


def test_an_optional_prior_pattern_may_match_nothing_and_the_output_is_skipped(tmp_path):
    (tmp_path / "out.json").write_text('{"me": 1}')
    (tmp_path / "a.json").write_text("{}")
    assert rs.load_prior([str(tmp_path / "a.json")],
                         [str(tmp_path / "nope_*.json"), str(tmp_path / "out.json")],
                         tmp_path / "out.json") == [{}]


def _bench(tmp_path, check, model_sev, n=16, rt=1, swap=False):
    """A key + verdicts where arm `a` is rated model_sev[a] over a round trip of rt[a] (an int,
    or a per-unit list). swap=True serves the model on B for odd units."""
    arms = sb.CHECKS[check]["arms"]
    items, rows, i = {}, [], 0
    for u in range(n):
        for a in arms:
            k = "item_%02d" % i
            i += 1
            r = rt[a] if isinstance(rt, dict) else rt
            r = r[u] if isinstance(r, list) else r
            flip = swap and u % 2
            items[k] = {"arm": a, "unit": u, "hnr": 10.0 + u,
                        "A_label": a + "rt" if flip else a, "B_label": a if flip else a + "rt",
                        "source": "x", "spk": u}
            sev_a, sev_b = (r, model_sev[a]) if flip else (model_sev[a], r)
            rows.append({"item": k, "sev_a": sev_a, "sev_b": sev_b})
    key = tmp_path / "k.json"
    key.write_text(json.dumps({"check": check, "items": items}))
    (tmp_path / "t" / "verdicts").mkdir(parents=True)
    with (tmp_path / "t" / "verdicts" / "verdicts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "sev_a", "sev_b"])
        w.writeheader()
        w.writerows(rows)
    return ["--check", str(check), "--key", str(key), "--test", str(tmp_path / "t")]


def test_check1_unblinds_to_go(tmp_path, capsys):
    r = ub.main(_bench(tmp_path, 1, {"stock": 1, "R": 3, "derisk": 3}))
    assert r["outcome"] == "go"
    assert "OUTCOME: GO" in capsys.readouterr().out


def test_check1_unblinds_to_the_same_go_with_the_model_on_either_side(tmp_path):
    sev = {"stock": 1, "R": 3, "derisk": 3}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = ub.main(_bench(tmp_path / "a", 1, sev))
    swapped = ub.main(_bench(tmp_path / "b", 1, sev, swap=True))
    assert swapped["outcome"] == plain["outcome"] == "go"
    assert swapped["level"] == plain["level"] and swapped["contrasts"] == plain["contrasts"]


def test_floors_exactly_one_point_apart_are_not_comparable(tmp_path, capsys):
    # stock's round trip: 0 on 17 units, 1 on 3 = 3/20 = 0.15; derisk's: 23/20 = 1.15.
    # A float mean computes the gap as 0.9999999999999999 and calls them comparable.
    rt = {"stock": [0] * 17 + [1] * 3, "R": [0] * 17 + [1] * 3, "derisk": [1] * 17 + [2] * 3}
    r = ub.main(_bench(tmp_path, 1, {"stock": 1, "R": 3, "derisk": 3}, n=20, rt=rt))
    assert r["floors_ok"] is False and r["outcome"] == "invalid"
    assert "floors not comparable" in capsys.readouterr().out


def test_a_key_from_another_check_is_refused(tmp_path):
    argv = _bench(tmp_path, 1, {"stock": 1, "R": 3, "derisk": 3})
    argv[1] = "2"
    with pytest.raises(SystemExit, match="check 1"):
        ub.main(argv)


def test_unrated_items_are_left_out_with_a_warning(tmp_path, capsys):
    argv = _bench(tmp_path, 3, {"S1": 1, "S2": 3, "R": 3})
    p = tmp_path / "t" / "verdicts" / "verdicts.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-1] + [lines[-1].split(",")[0] + ",,"]) + "\n")
    ub.main(argv)
    assert "not fully rated" in capsys.readouterr().out


def test_an_arm_with_nothing_rated_is_refused(tmp_path):
    argv = _bench(tmp_path, 2, {"stock": 1, "C0": 1, "S1": 3})
    p = tmp_path / "t" / "verdicts" / "verdicts.csv"
    key = json.loads((tmp_path / "k.json").read_text())["items"]
    kept = [l for l in p.read_text().splitlines()
            if l.startswith("item,") or key[l.split(",")[0]]["arm"] != "C0"]
    p.write_text("\n".join(kept) + "\n")
    with pytest.raises(SystemExit, match="no rated item"):
        ub.main(argv)


# ---------------------------------------------------------------- check 4 (arm L)

def test_a_dry_run_select_may_leave_an_untrained_arm_without_a_checkpoint():
    s1 = "/x/staged_s1/checkpoint_epoch=009_step=0009209.ckpt"
    assert rs.resolve_ckpts(("S1", "L"), ["S1=" + s1], allow_missing=True) == {"S1": s1, "L": None}
    with pytest.raises(SystemExit, match="L"):
        rs.resolve_ckpts(("S1", "L"), ["S1=" + s1])
    with pytest.raises(SystemExit, match="10 epochs"):        # a GIVEN checkpoint still counts
        rs.resolve_ckpts(("S1", "L"), ["S1=" + s1, "L=/x/last.ckpt"], allow_missing=True)


def test_a_dry_run_speakers_file_is_refused_by_render(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"check": 4, "dry_run": True, "ckpts": {"S1": ["/a", "h"], "L": None},
                             "items": []}))
    with pytest.raises(SystemExit, match="dry run"):
        rs.read_spec(str(p), 4)
    p.write_text(json.dumps({"check": 3, "ckpts": {}, "items": []}))
    with pytest.raises(SystemExit, match="check 3's"):
        rs.read_spec(str(p), 4)


def test_a_dry_run_may_not_write_where_the_real_selection_goes():
    with pytest.raises(SystemExit, match="staged_bench"):
        rs.refuse_dry_run_target(rs.ROOT + "/staged_bench/check4_speakers.json")
    rs.refuse_dry_run_target("/tmp/scratch/check4_dryrun_speakers.json")


def test_check4_requires_the_earlier_staged_speakers_files_by_default():
    base = rs.default_prior(3)
    four = rs.default_prior(4)
    assert base == [rs.ROOT + "/pitch_error/_*.json", rs.ROOT + "/eartest/_keys/*.key.json",
                    rs.ROOT + "/first_step_bench/speakers.json"]
    assert four == base + [rs.ROOT + "/staged_bench/check%d_speakers.json" % i for i in (1, 2, 3)]
    assert rs.default_prior(1) == rs.default_prior(2) == base


def test_check4_uses_every_eligible_voice_up_to_twenty_and_refuses_under_sixteen():
    pool = {s: float(s) for s in range(18)}
    assert rs.pick_voices(pool, 20, 4) == list(range(18))      # all, in HNR order
    big = {s: float(s) for s in range(30)}
    assert rs.pick_voices(big, 20, 4) == rs.lb.spread(big, 20)
    with pytest.raises(SystemExit, match="15 eligible"):
        rs.pick_voices({s: float(s) for s in range(15)}, 20, 4)
    assert rs.pick_voices(big, 20, 3) == rs.lb.spread(big, 20)  # check 3 unchanged


@pytest.mark.parametrize("r,o,ok", [("5", "5.05", True), ("5", "4.91", True), ("5", "5.1", False),
                                    ("5", "4.9", False), ("6", "5.85", False)])
def test_an_utterance_whose_original_differs_by_a_tenth_of_a_second_or_more_is_not_eligible(r, o, ok):
    # exact: as floats, 5.1 - 5.0 is 0.0999999999999996 and would pass
    assert rs.same_utterance_length(Fraction(r), Fraction(o)) is ok


def test_durations_are_read_exactly_as_frames_over_rate(tmp_path):
    import numpy as np
    import soundfile as sf
    sf.write(str(tmp_path / "a.wav"), np.zeros(24000 * 5 + 2400), 24000, "PCM_16")
    assert rs.exact_duration(str(tmp_path / "a.wav")) == Fraction(51, 10)


def test_each_libri22_arm_round_trips_its_own_corpus_copy():
    from staged_data import LIBRI_24K_ROOT, LIBRI_22K_ROOT, LIBRI_ORIG_22K_ROOT
    w = LIBRI_24K_ROOT + "/19/198/19_198_000000_000000.wav"
    assert rs.round_trip_source("S1", w) == LIBRI_22K_ROOT + "/19/198/19_198_000000_000000.wav"
    assert rs.round_trip_source("L", w) == LIBRI_ORIG_22K_ROOT + "/19/198/19_198_000000_000000.wav"


def test_check4_unblinds_to_shown(tmp_path, capsys):
    r = ub.main(_bench(tmp_path, 4, {"S1": 3, "L": 1}, rt={"S1": 1, "L": 1}))
    out = capsys.readouterr().out
    assert r["outcome"] == "shown" and r["shown"] == ["restoration"]
    assert "OUTCOME: SHOWN" in out and "restoration adds the hum" in out
    assert ("three p-values reported (S1's excess, S1 − L, L's excess), two of them deciding "
            "the reading") in out


def test_check4_unblinds_to_not_shown_and_reports_ls_excess_and_the_split(tmp_path, capsys):
    rt = {"S1": [1] * 16, "L": [1] * 8 + [2] * 8}            # floors 1 and 1.5
    r = ub.main(_bench(tmp_path, 4, {"S1": 3, "L": 4}, rt=rt))
    out = capsys.readouterr().out
    assert r["outcome"] == "not_shown" and "speakers and G2P" in out
    assert "L      n=16  model 4.00  round trip 1.50  gap +2.50" in out
    assert "model ratings alone -1.00 (n=16)" in out
    assert "round-trip ratings alone -0.50 (n=16)" in out
    assert "L − S1 on the gaps +0.50" in out


def test_check4_floors_are_compared_with_s1s(tmp_path, capsys):
    r = ub.main(_bench(tmp_path, 4, {"S1": 3, "L": 3}, rt={"S1": 0, "L": 1}))
    assert r["floors_ok"] is False and r["outcome"] == "invalid"
    assert "within 1 point of S1's" in capsys.readouterr().out


def test_check4_with_fifteen_voices_rated_in_both_arms_is_invalid(tmp_path):
    r = ub.main(_bench(tmp_path, 4, {"S1": 3, "L": 1}, n=15, rt={"S1": 1, "L": 1}))
    assert r["outcome"] == "invalid"
