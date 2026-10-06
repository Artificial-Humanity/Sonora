"""The staged-rebuild bench tools' guards (`render_ear_staged.py`, `unblind_ear_staged.py`).

The render tool's torch half runs in the container; these pin what it decides on the host.
"""
import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import render_ear_staged as rs  # noqa: E402
import staged_bench as sb  # noqa: E402


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
