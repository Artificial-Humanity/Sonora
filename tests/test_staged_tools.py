"""The staged-rebuild bench tools' guards (`render_ear_staged.py`, `unblind_ear_staged.py`).

The render tool's torch half runs in the container; these pin what it decides on the host.
"""
import csv
import json
from fractions import Fraction
from pathlib import Path

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


def test_d1_resolves_only_to_its_registered_26_epoch_final():
    good = "/x/staged_d1/runs/a/checkpoints/checkpoint_epoch=025_step=0009229.ckpt"
    assert rs.resolve_ckpts(("D1",), ["D1=" + good]) == {"D1": good}
    with pytest.raises(SystemExit, match="26 epochs"):
        rs.resolve_ckpts(("D1",), ["D1=/x/checkpoint_epoch=009_step=0003549.ckpt"])


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


def _clips(items):
    """The key's `clips`: an opaque name per side of each item, and the label it carries."""
    out = {}
    for k, t in items.items():
        out["a_" + k], out["b_" + k] = {"label": t["A_label"]}, {"label": t["B_label"]}
    return out


def _serve(test_dir, items):
    """The test folder's items.json, naming the opaque clip on each side of each item."""
    (test_dir / "items.json").write_text(json.dumps(
        {"test": test_dir.name, "items": [{"id": k, "A": "a_" + k, "B": "b_" + k} for k in items]}))


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
    key.write_text(json.dumps({"check": check, "items": items, "clips": _clips(items)}))
    (tmp_path / "t" / "verdicts").mkdir(parents=True)
    _serve(tmp_path / "t", items)
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


# ------------------------------------------------- check 4: review fixes (provenance, refusals)

S1_REGISTERED = "857f62d7ac47770d94f971cf4bae80d3078e42fd028d45465fe30bd0e2a0d0d5"


def test_the_registered_s1_checkpoint_sha_is_pinned_next_to_check_4():
    assert sb.CHECK4_S1_SHA256 == S1_REGISTERED


def _world(tmp_path, monkeypatch, n_voices, *, extra=None):
    """Synthetic LibriTTS-R / S1 / L corpora for `select --check 4`: `n_voices` voices of two
    1 s recordings each (24 kHz), their S1 and L filelist rows, originals of the same length,
    an HNR file, an empty prior and a stand-in S1 checkpoint whose sha is registered.
    `extra(spk, k, files)` may change one recording. Returns argv for `select`."""
    import numpy as np
    import soundfile as sf
    r24, o24 = tmp_path / "r24", tmp_path / "o24"
    rows, s1_rows, l_rows = {}, {}, {}
    for spk in range(1, n_voices + 1):
        for k in range(2):
            name = "%d_%d.wav" % (spk, k)
            w = str(r24 / str(spk) / name)
            o = str(o24 / str(spk) / name)
            files = {"r24": 24000, "o24": 24000}
            if extra:
                files = extra(spk, k, files) or files
            for base, key in ((w, "r24"), (o, "o24")):
                if files.get(key):
                    Path(base).parent.mkdir(parents=True, exist_ok=True)
                    sf.write(base, np.zeros(files[key]), 24000, "PCM_16")
            rows[w] = (spk, "p%d" % k)
            if files.get("s1", True):
                s1_rows[w.replace("/r24/", "/r22/")] = (spk, "p%d" % k)
            if files.get("l", True):
                l_rows[w.replace("/o24/", "/o22/").replace("/r24/", "/o22/")] = (spk, "p%d" % k)
    tables = {"libritts_r_vat": rows, "libritts_r_22k": s1_rows, "libritts_22k": l_rows}
    monkeypatch.setattr(rs, "corpus_rows", lambda corpus: tables[corpus])
    monkeypatch.setattr(rs.sd, "to_22k", lambda w: w.replace("/r24/", "/r22/"))
    monkeypatch.setattr(rs.sd, "to_l22k", lambda w: w.replace("/r24/", "/o22/"))
    monkeypatch.setattr(rs.sd, "to_original", lambda w: w.replace("/r24/", "/o24/"))
    hnr = tmp_path / "hnr.json"
    hnr.write_text(json.dumps({"speakers": {str(s): {"hnr": 5.0 + s}
                                            for s in range(1, n_voices + 3)}}))
    (tmp_path / "prior").mkdir()
    (tmp_path / "prior" / "p.json").write_text(json.dumps({"items": []}))
    for name in ("vl.txt", "vc.txt", "vh.json"):
        (tmp_path / name).write_text("x")
    ck = tmp_path / "checkpoint_epoch=009_step=0009209.ckpt"
    ck.write_bytes(b"s1")
    monkeypatch.setattr(rs.sb, "CHECK4_S1_SHA256", rs.sha256(str(ck)))
    monkeypatch.setattr(rs, "ROOT", str(tmp_path / "root"))
    prior = [str(tmp_path / "prior" / "*.json")]
    monkeypatch.setattr(rs, "default_prior", lambda check: prior)
    return ["select", "--check", "4", "--ckpt", "S1=" + str(ck),
            "--hnr-json", str(hnr), "--vctk-filelist", str(tmp_path / "vl.txt"),
            "--vctk-cleaned", str(tmp_path / "vc.txt"),
            "--vctk-hnr-from", str(tmp_path / "vh.json"), "--prior-optional",
            "--min-seconds", "0.5", "--max-seconds", "2"]


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["render_ear_staged.py"] + argv)
    rs.main()


def _select(tmp_path, monkeypatch, n_voices, *, dry=True, extra=None, more=()):
    argv = _world(tmp_path, monkeypatch, n_voices, extra=extra)
    out = tmp_path / "scratch" / "speakers.json"
    argv += ["--dry-run"] if dry else []
    if not dry:       # a real selection also needs L's checkpoint
        ck = tmp_path / "l" / "checkpoint_epoch=009_step=0009209.ckpt"
        ck.parent.mkdir()
        ck.write_bytes(b"l")
        argv += ["--ckpt", "L=" + str(ck)]
    _run(monkeypatch, argv + list(more) + ["--out-speakers", str(out)])
    return json.loads(out.read_text())


def test_select_check4_dry_run_marks_the_file_and_uses_the_filters(tmp_path, monkeypatch):
    spec = _select(tmp_path, monkeypatch, 17)
    assert spec["dry_run"] is True and spec["check"] == 4
    assert spec["eligible"] == 17 and len(spec["items"]) == 17
    assert spec["ckpts"]["L"] is None and spec["ckpts"]["S1"][1] == rs.sb.CHECK4_S1_SHA256
    assert "; 17 units;" in spec["rule"] and "checks 3-4" in spec["rule"]


def test_a_real_select_check4_has_no_dry_run_key(tmp_path, monkeypatch):
    spec = _select(tmp_path, monkeypatch, 16, dry=False)
    assert "dry_run" not in spec and len(spec["items"]) == 16


def test_select_refuses_more_units_than_the_exact_test_can_read_before_anything_else(
        tmp_path, monkeypatch):
    """Audit 2026-10-31, M-3: --n 24 used to be drawn, rendered and rated, then crash."""
    with pytest.raises(SystemExit, match="at most 20"):
        _select(tmp_path, monkeypatch, 17, more=["--n", "24"])
    assert not (tmp_path / "scratch" / "speakers.json").exists()


@pytest.mark.parametrize("n,used", [(16, 16), (20, 20), (22, 20)])
def test_select_check4_boundaries_accept_sixteen_and_use_all_twenty(tmp_path, monkeypatch, n, used):
    spec = _select(tmp_path, monkeypatch, n)
    assert spec["eligible"] == n and len(spec["items"]) == used


def test_select_check4_with_fifteen_eligible_voices_is_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="15 eligible"):
        _select(tmp_path, monkeypatch, 15)


def test_select_check4_drops_recordings_missing_from_either_filelist_or_the_originals(
        tmp_path, monkeypatch):
    def extra(spk, k, files):
        if spk == 1 and k == 0:
            return dict(files, s1=False)          # not in S1's train filelist
        if spk == 2 and k == 1:
            return dict(files, l=False)           # not in L's train filelist
        if spk == 3 and k == 0:
            return dict(files, o24=0)             # no original recording
        if spk == 4 and k == 1:
            return dict(files, o24=26400)         # 1.1 s against 1.0 s: drift of 0.1 s
        if spk == 5 and k == 1:
            return dict(files, o24=25920)         # 1.08 s: within 0.1 s, eligible
    spec = _select(tmp_path, monkeypatch, 20, extra=extra)
    voices = {it["libri"] for it in spec["items"]}
    assert voices == set(range(5, 21))
    assert spec["eligible"] == 16
    assert spec["items"][0]["arms"]["S1"]["phonemes"] in ("p0", "p1")


def test_select_check4_excludes_the_voices_heard_in_the_prior(tmp_path, monkeypatch):
    argv = _world(tmp_path, monkeypatch, 20)
    (tmp_path / "prior" / "p.json").write_text(json.dumps({"items": [{"libri": 3}, {"spk": 4}]}))
    out = tmp_path / "scratch" / "s.json"
    _run(monkeypatch, argv + ["--dry-run", "--out-speakers", str(out)])
    spec = json.loads(out.read_text())
    assert spec["eligible"] == 18
    assert not {3, 4} & {it["libri"] for it in spec["items"]}


def test_select_check4_reads_the_original_length_exactly(tmp_path, monkeypatch):
    # 5.1 s against 5.0 s is 0.0999999999999996 as floats; exact_duration refuses it.
    seen = []
    real = rs.exact_duration
    monkeypatch.setattr(rs, "exact_duration", lambda p: seen.append(p) or real(p))
    _select(tmp_path, monkeypatch, 16)
    assert any("/o24/" in p for p in seen) and any("/r24/" in p for p in seen)


def test_select_check4_refuses_an_s1_checkpoint_that_is_not_the_registered_one(
        tmp_path, monkeypatch):
    argv = _world(tmp_path, monkeypatch, 16)
    monkeypatch.setattr(rs.sb, "CHECK4_S1_SHA256", S1_REGISTERED)
    out = tmp_path / "scratch" / "s.json"
    with pytest.raises(SystemExit, match="registered"):
        _run(monkeypatch, argv + ["--dry-run", "--out-speakers", str(out)])
    lck = tmp_path / "l" / "checkpoint_epoch=009_step=0009209.ckpt"
    lck.parent.mkdir()
    lck.write_bytes(b"l")
    with pytest.raises(SystemExit, match="registered"):
        _run(monkeypatch, argv + ["--ckpt", "L=" + str(lck), "--out-speakers", str(out)])
    assert not out.exists()


def test_the_check4_rule_counts_the_units_and_names_the_spread_checks():
    class A:
        check, n, min_seconds, max_seconds, max_gap, target_seconds, seed = \
            4, 20, 4.0, 8.0, 0.25, 6.0, 7
    c = sb.CHECKS[4]
    text = rs.speakers_rule(A, c, 17)
    assert text.startswith("check 4: arms S1/L; 17 units; ")
    assert "(checks 3-4, target 6.0s)" in text and "(check 3," not in text


def test_the_rule_text_of_checks_1_to_3_is_unchanged():
    class A:
        check, n, min_seconds, max_seconds, max_gap, target_seconds, seed = \
            3, 20, 4.0, 8.0, 0.25, 6.0, 7
    c = sb.CHECKS[3]
    assert rs.speakers_rule(A, c, 20) == (
        "check 3: arms S1/S2/R; 20 units; clips 4.0-8.0s; VCTK/LibriTTS-R matched within 0.25 dB "
        "HNR (checks 1-2) or LibriTTS-R voices spread over HNR (check 3, target 6.0s); a "
        "distinct training recording per arm; seed 7.")


def test_an_explicit_prior_for_check_4_must_hold_every_default_entry():
    full = rs.default_prior(4)
    rs.require_default_prior(4, full + ["/extra/*.json"])
    rs.require_default_prior(3, ["/just/this.json"])           # checks 1-3 unchanged
    with pytest.raises(SystemExit, match="check3_speakers.json") as e:
        rs.require_default_prior(4, [p for p in full if "check3" not in p])
    assert "check1_speakers.json" not in str(e.value)
    with pytest.raises(SystemExit, match="first_step_bench"):
        rs.require_default_prior(4, ["/x/*.json"])


def test_main_refuses_a_short_explicit_prior_for_check_4(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="every default"):
        _run(monkeypatch, ["select", "--check", "4", "--dry-run", "--prior", "/x/*.json",
                           "--out-speakers", str(tmp_path / "o.json")])


def test_a_dry_run_may_not_write_anywhere_under_the_bench_data_root():
    with pytest.raises(SystemExit, match="under"):
        rs.refuse_dry_run_target(rs.ROOT + "/eartest/_keys/x.key.json")
    with pytest.raises(SystemExit, match="under"):
        rs.refuse_dry_run_target(rs.ROOT + "/anything.json")


def test_a_dry_run_may_not_write_a_file_a_prior_pattern_matches(tmp_path):
    out = tmp_path / "prior" / "x_speakers.json"
    with pytest.raises(SystemExit, match="prior"):
        rs.refuse_dry_run_target(out, [str(tmp_path / "prior" / "*_speakers.json")])
    rs.refuse_dry_run_target(out, [str(tmp_path / "other" / "*.json")])
    rs.refuse_dry_run_target("/tmp/scratch/s.json", [rs.ROOT + "/pitch_error/_*.json"])


def test_a_relative_prior_pattern_is_resolved_against_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / "prior").mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="prior"):
        rs.refuse_dry_run_target(tmp_path / "prior" / "x.json", ["prior/*.json"])
    with pytest.raises(SystemExit, match="prior"):
        rs.refuse_dry_run_target("prior/x.json", ["prior/*.json"])
    rs.refuse_dry_run_target(tmp_path / "other" / "x.json", ["prior/*.json"])


def test_a_prior_pattern_star_does_not_cross_a_directory_as_glob_does_not(tmp_path):
    pat = str(tmp_path / "scratch" / "*.json")
    rs.refuse_dry_run_target(tmp_path / "scratch" / "sub" / "x.json", [pat])
    with pytest.raises(SystemExit, match="prior"):
        rs.refuse_dry_run_target(tmp_path / "scratch" / "x.json", [pat])


def test_check4_keys_record_the_file_each_arm_round_tripped_and_checks_1_to_3_do_not():
    from staged_data import LIBRI_22K_ROOT, LIBRI_ORIG_22K_ROOT, LIBRI_24K_ROOT
    w = LIBRI_24K_ROOT + "/19/198/19_198_000000_000000.wav"
    it = {"unit": 3, "libri": 19, "libri_hnr": 9.5,
          "arms": {"S1": {"source": w}, "L": {"source": w}}}
    sides = [("A", "S1rt"), ("B", "S1")]
    s1 = rs.truth_entry(4, "S1", it, sides)
    assert s1["rt_source"] == LIBRI_22K_ROOT + "/19/198/19_198_000000_000000.wav"
    sides = [("A", "L"), ("B", "Lrt")]
    l_ = rs.truth_entry(4, "L", it, sides)
    assert l_["rt_source"] == LIBRI_ORIG_22K_ROOT + "/19/198/19_198_000000_000000.wav"
    assert l_["source"] == w and l_["kind"] == "Lrt_vs_L" and l_["spk"] == 19
    old = rs.truth_entry(3, "S1", it, [("A", "S1"), ("B", "S1rt")])
    assert "rt_source" not in old
    assert list(old) == ["spk", "kind", "arm", "unit", "hnr", "A_label", "B_label", "source"]


def test_check4_with_an_arm_nothing_rated_prints_the_invalid_outcome(tmp_path, capsys):
    argv = _bench(tmp_path, 4, {"S1": 3, "L": 1}, rt={"S1": 1, "L": 1})
    p = tmp_path / "t" / "verdicts" / "verdicts.csv"
    key = json.loads((tmp_path / "k.json").read_text())["items"]
    kept = [l for l in p.read_text().splitlines()
            if l.startswith("item,") or key[l.split(",")[0]]["arm"] != "L"]
    p.write_text("\n".join(kept) + "\n")
    r = ub.main(argv)
    out = capsys.readouterr().out
    assert r["outcome"] == "invalid"
    assert "OUTCOME: INVALID" in out and "fewer than 16 voices" in out
    assert "no reading" in out and "owner decides" in out
