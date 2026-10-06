"""The staged-rebuild bench tools' guards (`render_ear_staged.py`, `unblind_ear_staged.py`).

The render tool's torch half runs in the container; these pin what it decides on the host.
"""
import csv
import json
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


def _bench(tmp_path, check, model_sev, n=16):
    """A key + verdicts where arm `a` is rated model_sev[a] over a round trip of 1."""
    arms = sb.CHECKS[check]["arms"]
    items, rows, i = {}, [], 0
    for u in range(n):
        for a in arms:
            k = "item_%02d" % i
            i += 1
            items[k] = {"arm": a, "unit": u, "hnr": 10.0 + u, "A_label": a,
                        "B_label": a + "rt", "source": "x", "spk": u}
            rows.append({"item": k, "sev_a": model_sev[a], "sev_b": 1})
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
