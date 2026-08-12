"""QC-M4: the confusion matrix that "decides pipeline 1.0" understated and misreported.

Two defects in the same path, and both are the shape this repo keeps closing: the number
looked like an answer.

  * **It excluded the ear's strongest rejections.** Dropped/reroll rows were skipped
    wholesale. The reason was sound — their stored `score` is the one they held BEFORE the
    rejection, so reading it as a keep pollutes the cells — but skipping the row throws
    the VERDICT away with the stale number, and drops are precisely what the
    "instrument keep / ear reject" false-negative cell exists to count. `cmd_sweep` had
    read status-as-verdict since 2026-08-08; the fix never reached here.

  * **It scored one campaign and printed the header of many.** `--campaign-dir` is
    `action="append"`, the sweep path pools, and this path read `campaign_dir[0]` while
    printing "{N} pooled: <every dir>". A single-campaign matrix quoted as multi-campaign
    evidence — "a check whose summary is wrong is worse than no check".

Everything here runs against tmp_path; nothing touches /data.
"""

import csv
import json
import pathlib
import sys

import pytest
from scripts_layout import SCRIPTS  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
SYNTH = SCRIPTS
SCRIPTS.on_path()
gc = pytest.importorskip("gate_calibration")

RATINGS_HEADER = ["campaign", "id", "engine", "score", "note", "status", "link"]


def _campaign(root, name, verdicts):
    """verdicts: {id: instrument_keep_bool}."""
    d = root / name
    d.mkdir(parents=True)
    with (d / "qc_verdicts.jsonl").open("w", encoding="utf-8") as fh:
        for cid, keep in verdicts.items():
            fh.write(json.dumps({"id": cid, "keep": keep}) + "\n")
    return d


def _ratings(root, rows):
    """rows: [(id, score, status)]."""
    p = root / "ratings.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RATINGS_HEADER)
        w.writeheader()
        for cid, score, status in rows:
            w.writerow({"campaign": "c", "id": cid, "engine": "qwen", "score": score,
                        "note": "", "status": status, "link": ""})
    return p


def _run(monkeypatch, dirs, ratings):
    argv = ["gate_calibration"]
    for d in dirs:
        argv += ["--campaign-dir", str(d)]
    argv += ["--ratings", str(ratings)]
    monkeypatch.setattr(sys, "argv", argv)
    gc.main()


def test_a_dropped_clip_counts_as_the_rejection_it_is(tmp_path, monkeypatch, capsys):
    """THE REGRESSION. The instrument certified both clips; the ear kept one and DROPPED
    the other. That drop is a false negative and it is the whole point of the table.

    Skipping the row emptied the cell of exactly the clips it exists to count, so the
    false-negative rate was systematically understated — while still printing as a result.
    """
    d = _campaign(tmp_path, "camp", {"good": True, "bad": True})
    # `bad` still carries the 5 it was given before the owner dropped it — which is why
    # the number cannot be trusted and the STATUS has to be.
    ratings = _ratings(tmp_path, [("good", "5", "keep"), ("bad", "5", "dropped")])

    _run(monkeypatch, [d], ratings)
    out = capsys.readouterr().out

    assert "ear rejections counted: 1" in out
    row = next(l for l in out.splitlines() if l.strip().startswith("instrument keep"))
    assert row.split()[2:] == ["1", "1"], f"the false-negative cell is empty: {row!r}"


def test_a_retake_is_a_rejection_in_the_matrix_too(tmp_path, monkeypatch, capsys):
    """QC-M3 reaching this path: Retake writes score "0" and status `reroll`. Before the
    marker fix the row parsed to None and never entered the table at all."""
    d = _campaign(tmp_path, "camp", {"a": True, "b": True})
    ratings = _ratings(tmp_path, [("a", "5", "keep"), ("b", "0", "reroll")])

    _run(monkeypatch, [d], ratings)
    out = capsys.readouterr().out
    assert "rated    : 2 of 2" in out
    assert "ear rejections counted: 1" in out


def test_a_machine_folded_row_is_still_not_an_ear_verdict(tmp_path, monkeypatch, capsys):
    """The exclusion that must SURVIVE the fix. A folded row carries a fabricated score
    the ear never gave; counting it as a keep is what polluted the cells originally."""
    d = _campaign(tmp_path, "camp", {"a": True, "folded": True})
    p = tmp_path / "ratings.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RATINGS_HEADER)
        w.writeheader()
        w.writerow({"campaign": "c", "id": "a", "engine": "q", "score": "5",
                    "note": "", "status": "keep", "link": ""})
        w.writerow({"campaign": "c", "id": "folded", "engine": "q", "score": "4",
                    "note": "folded: staged unheard", "status": "keep", "link": ""})

    _run(monkeypatch, [d], p)
    out = capsys.readouterr().out
    assert "1 machine-folded rows" in out
    assert "rated    : 1 of 2" in out


def test_pooling_scores_every_campaign_it_names(tmp_path, monkeypatch, capsys):
    """The header printed "2 pooled: camp-a, camp-b" while the matrix covered camp-a
    alone. Either the header or the reading had to change; the sweep path already pools,
    so this one does now too."""
    a = _campaign(tmp_path, "camp-a", {"a1": True})
    b = _campaign(tmp_path, "camp-b", {"b1": True})
    ratings = _ratings(tmp_path, [("a1", "5", "keep"), ("b1", "x", "dropped")])

    _run(monkeypatch, [a, b], ratings)
    out = capsys.readouterr().out

    assert "2 pooled: camp-a, camp-b" in out
    assert "verdicts : qc_verdicts.jsonl (2 clips)" in out, (
        "the second campaign's verdicts were never read")
    assert "rated    : 2 of 2" in out


def test_one_campaign_still_names_itself(tmp_path, monkeypatch, capsys):
    d = _campaign(tmp_path, "camp", {"a": True})
    ratings = _ratings(tmp_path, [("a", "5", "keep")])
    _run(monkeypatch, [d], ratings)
    out = capsys.readouterr().out
    assert "pooled" not in out
    assert str(d) in out
