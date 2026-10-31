"""`ear_bench.Bench.write`'s guards (audit 2026-10-31, M-5). `ear_bench` imports torch and the
Matcha stack at module scope and the host has neither, so they are stubbed here and the real
module is loaded from its file under a private name; nothing leaks into `sys.modules`."""
import csv
import importlib.util
import json
import sys
from unittest import mock

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()


@pytest.fixture
def eb(monkeypatch):
    for name in ("torch", "matcha", "matcha.delivery", "matcha.cli", "matcha.text",
                 "matcha.utils", "matcha.utils.utils"):
        monkeypatch.setitem(sys.modules, name, mock.MagicMock())
    spec = importlib.util.spec_from_file_location("ear_bench_stubbed", SCRIPTS / "ear_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bench(eb, out):
    b = object.__new__(eb.Bench)
    b.out, b.salt, b.sample_rate, b.key, b.written, b.seed = out, "s", 24000, {}, 0, 1
    return b


def _verdicts(out, rows):
    (out / "verdicts").mkdir(parents=True, exist_ok=True)
    with (out / "verdicts" / "verdicts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "choice", "sev_a", "sev_b"])
        w.writeheader()
        w.writerows(rows)


def _item(i, a, b):
    return {"id": i, "A": a, "B": b}


def test_an_item_with_only_a_severity_is_judged_and_its_sides_cannot_move(eb, tmp_path):
    """A defects-scale item with one slider set has no derived `choice`; the guard counted
    `choice` alone and let its A/B be re-pointed."""
    out = tmp_path / "t"
    out.mkdir()
    b = _bench(eb, out)
    b.write("t", [], [_item("item_00", "x", "y")], {}, key_out=str(tmp_path / "k.json"))
    _verdicts(out, [{"item": "item_00", "choice": "", "sev_a": "2", "sev_b": ""}])
    with pytest.raises(SystemExit, match="already-judged pairs would change sides"):
        b.write("t", [], [_item("item_00", "y", "x")], {}, key_out=str(tmp_path / "k.json"))


def test_the_same_assignment_may_be_regenerated_over_judged_items(eb, tmp_path):
    out = tmp_path / "t"
    out.mkdir()
    b = _bench(eb, out)
    served = [_item("item_00", "x", "y")]
    b.write("t", [], served, {}, key_out=str(tmp_path / "k.json"))
    _verdicts(out, [{"item": "item_00", "choice": "", "sev_a": "2", "sev_b": "1"}])
    b.write("t", [], served, {}, key_out=str(tmp_path / "k.json"))


def test_a_key_inside_the_served_tree_is_refused_before_items_json_is_written(eb, tmp_path):
    out = tmp_path / "t"
    out.mkdir()
    with pytest.raises(SystemExit, match="inside the served tree"):
        _bench(eb, out).write("t", [], [_item("item_00", "x", "y")], {},
                              key_out=str(out / "key.json"))
    assert not (out / "items.json").exists(), "items.json was left in the served tree"


def test_a_key_outside_the_tree_is_written_beside_items_json(eb, tmp_path):
    """The positive control for the refusal above."""
    out = tmp_path / "t"
    out.mkdir()
    _bench(eb, out).write("t", [], [_item("item_00", "x", "y")], {}, key_out=str(tmp_path / "k.json"))
    assert json.loads((out / "items.json").read_text())["items"][0]["id"] == "item_00"
    assert (tmp_path / "k.json").is_file()
