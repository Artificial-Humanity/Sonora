"""C-M5: the last three writes that could lose state they had already read.

These are not the same defect as an un-flushed file. Every one of them is a
READ-MODIFY-WRITE across a gap: the snapshot is taken at startup, the work takes minutes,
and the snapshot is written back at the end. Whatever else touched the file in between is
gone — and gone silently, because the write itself succeeds.

The three files are the ones that cannot be regenerated:

  * `staging_log.json`     — the only record of which clips have been staged. Lose an
                             entry and those clips are staged twice.
  * `reader_profiles.json` — the propagation SSOT. Lose it and every (reader, title)
                             becomes unconfirmed, which does not merely cost a re-run:
                             `stage_pool` then refuses to stage anything and the auditor
                             is re-queued for pairs already heard.
  * `metadata.jsonl`       — the dataset index. A truncated one is a corpus that lost its
                             tail without saying so.

Everything else these scripts write is derived output (bank.json, qc_measures.jsonl, the
reports) where re-running IS the repair, so it is deliberately left alone.

Nothing here touches the real files: every case runs against tmp_path.
"""

import csv
import json
import os
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SYNTH = REPO / "scripts" / "synthesis"
sys.path.insert(0, str(SYNTH))
sys.path.insert(0, str(REPO))

synth_common = pytest.importorskip("synth_common")

RATINGS_HEADER = ["campaign", "id", "engine", "status", "score", "note", "link",
                  "gender", "age", "accent", "delivery"]


def _write_ratings(path, rows=()):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RATINGS_HEADER)
        w.writeheader()
        w.writerows(rows)


def _rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- staging_log ----------------------------------------------------------------------


@pytest.fixture()
def staging(tmp_path, monkeypatch):
    """A minimal but real campaign: 4 pooled clips, one ear-confirmed (reader, title)."""
    sp = pytest.importorskip("stage_pool")
    datasets = tmp_path / "datasets"
    cdir = datasets / "camp"
    cdir.mkdir(parents=True)
    with (cdir / "librivox_manifest.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(4):
            fh.write(json.dumps({
                "id": f"clip{i}", "book": "A Book", "reader": "A Reader",
                "section": 1, "sentence_index": i, "wav": f"clip{i}.wav",
                "seconds": 8.0, "librivox_url": "https://librivox.org/a-book/",
            }) + "\n")

    ratings = datasets / "ratings.csv"
    _write_ratings(ratings)
    profiles = datasets / "reader_profiles.json"
    profiles.write_text(json.dumps({
        "A Reader": {"titles": {"A Book": {"gender": "female", "age": "adult",
                                           "accent": "american", "clips_seen": 3}},
                     "hint": {}}}), encoding="utf-8")

    monkeypatch.setattr(sp, "DATASETS", datasets)
    monkeypatch.setattr(sp, "RATINGS", ratings)
    monkeypatch.setattr(sp, "PROFILES", profiles)
    monkeypatch.setattr(sp, "LEDGER", datasets / "books_ledger.json")
    return sp, cdir, ratings


def _stage(sp, monkeypatch, n=2):
    monkeypatch.setattr(sys, "argv",
                        ["stage_pool", "--campaign", "camp", "--stage", str(n), "--apply"])
    return sp.main()


def test_a_concurrent_staging_run_is_not_forgotten(staging, monkeypatch):
    """THE REGRESSION. `log` is read at startup and was written back at the end, so a run
    that finished in between vanished from the record — and its clips, no longer listed as
    staged, would be staged a second time. `write_json_atomic` does not help: the write
    was never torn, it was stale."""
    sp, cdir, _ratings = staging
    log_path = cdir / "staging_log.json"
    real_flagged = sp.qc_flagged

    def flagged_then_race(campaign_dir):
        # Fires immediately after stage_pool reads the log, before it writes it.
        synth_common.update_json(log_path, lambda cur: cur.setdefault("runs", []).append(
            {"run": 1, "date": "2026-01-01", "count": 1, "kind": "stage",
             "range": None, "ids": ["other-run-clip"]}))
        monkeypatch.setattr(sp, "qc_flagged", real_flagged)
        return real_flagged(campaign_dir)

    monkeypatch.setattr(sp, "qc_flagged", flagged_then_race)
    assert _stage(sp, monkeypatch) == 0

    log = json.loads(log_path.read_text(encoding="utf-8"))
    ids = {i for run in log["runs"] for i in run["ids"]}
    assert "other-run-clip" in ids, "the concurrent run's entry was overwritten"
    assert {"clip0", "clip1"} <= ids
    # And the run numbers stay dense over what is actually in the file.
    assert [r["run"] for r in log["runs"]] == [1, 2]


def test_the_log_counts_what_was_written_not_what_was_picked(staging, monkeypatch):
    """`count` was `len(take)` — the clips this run PICKED. When the ratings transaction
    finds them already present (a lost log, or a run that got there first) nothing is
    written and the entry claimed a staging run that did not happen. It is still recorded,
    on purpose: the ids are staged either way, and omitting them leaves the next run
    picking the same take forever. Recording it truthfully is the difference."""
    sp, cdir, ratings = staging
    log_path = cdir / "staging_log.json"
    assert _stage(sp, monkeypatch) == 0
    assert len(_rows(ratings)) == 2

    log_path.unlink()                       # the log is lost; ratings.csv is not
    assert _stage(sp, monkeypatch) == 0

    run = json.loads(log_path.read_text(encoding="utf-8"))["runs"][-1]
    assert run["count"] == 0
    assert run["already_in_ratings"] == 2
    assert run["ids"] == ["clip0", "clip1"]  # so they are never picked a third time


def test_a_normal_run_records_no_reconciliation_field(staging, monkeypatch):
    """The marker must only appear when the two numbers actually disagree, or it stops
    meaning anything."""
    sp, cdir, _ = staging
    assert _stage(sp, monkeypatch) == 0
    run = json.loads((cdir / "staging_log.json").read_text(encoding="utf-8"))["runs"][-1]
    assert run["count"] == 2
    assert "already_in_ratings" not in run


# --- reader_profiles.json -------------------------------------------------------------


def test_learn_does_not_drop_a_reader_another_run_had_just_written(tmp_path, monkeypatch):
    """`--learn` is per-reader and incremental, so writing back the whole-file snapshot
    read at startup is how one run silently deletes another's readers. The merge now
    happens against the file's CURRENT contents, under the lock."""
    rp = pytest.importorskip("reader_profile")
    datasets = tmp_path
    ratings = datasets / "ratings.csv"
    _write_ratings(ratings, [
        {"campaign": "book-camp", "id": "clip0", "engine": "librivox", "status": "keep",
         "score": "5", "note": "", "link": "", "gender": "female", "age": "adult",
         "accent": "american", "delivery": "Neutral"},
    ])
    profiles = datasets / "reader_profiles.json"
    profiles.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rp, "PROFILES", profiles)
    monkeypatch.setattr(rp, "clip_meta", lambda: {"clip0": ("A Reader", "A Book")})

    real_learn = rp.learn

    def learn_then_race(rows, meta):
        # Another --learn run commits a different reader while this one is thinking.
        synth_common.update_json(
            profiles, lambda cur: cur.__setitem__("Other Reader", {"titles": {}, "hint": {}}))
        return real_learn(rows, meta)

    monkeypatch.setattr(rp, "learn", learn_then_race)
    monkeypatch.setattr(sys, "argv", ["reader_profile", "--learn", "--ratings", str(ratings)])
    assert rp.main() == 0

    written = json.loads(profiles.read_text(encoding="utf-8"))
    assert "Other Reader" in written, "the concurrent run's reader was overwritten"
    assert written["A Reader"]["titles"]["A Book"]["gender"] == "female"


def test_learn_then_apply_still_sees_what_it_just_learned(tmp_path, monkeypatch):
    """`--learn --apply` in one invocation used to work because the merge mutated the
    in-memory dict. Moving the merge under the lock has to keep that: --apply must
    propagate from the profiles this run just wrote, not from the pre-run snapshot."""
    rp = pytest.importorskip("reader_profile")
    ratings = tmp_path / "ratings.csv"
    _write_ratings(ratings, [
        {"campaign": "book-camp", "id": "heard", "engine": "librivox", "status": "keep",
         "score": "5", "note": "", "link": "", "gender": "female", "age": "adult",
         "accent": "american", "delivery": ""},
        {"campaign": "book-camp", "id": "blank", "engine": "librivox", "status": "keep",
         "score": "", "note": "", "link": "", "gender": "", "age": "", "accent": "",
         "delivery": ""},
    ])
    profiles = tmp_path / "reader_profiles.json"
    profiles.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rp, "PROFILES", profiles)
    monkeypatch.setattr(rp, "clip_meta",
                        lambda: {"heard": ("A Reader", "A Book"),
                                 "blank": ("A Reader", "A Book")})
    monkeypatch.setattr(sys, "argv",
                        ["reader_profile", "--learn", "--apply", "--ratings", str(ratings)])
    assert rp.main() == 0

    filled = {r["id"]: r for r in _rows(ratings)}["blank"]
    assert filled["gender"] == "female" and filled["accent"] == "american"


# --- metadata.jsonl -------------------------------------------------------------------


@pytest.fixture()
def meta_file(tmp_path):
    path = tmp_path / "metadata.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps({"id": f"c{i}", "engine": "qwen"}) + "\n")
    return path


def _publish(pt, monkeypatch, path, apply=True):
    argv = ["publish_tier", "--meta", str(path)] + (["--apply"] if apply else [])
    monkeypatch.setattr(sys, "argv", argv)
    return pt.main()


def test_a_failed_backfill_leaves_the_dataset_index_whole(meta_file, monkeypatch, capsys):
    """metadata.jsonl was rewritten in place with a plain `open(..., "w")`, which
    truncates FIRST. A crash between truncate and the last row leaves a shorter valid
    JSONL — a corpus that lost its tail and says nothing about it. With tmp + replace the
    interrupted run leaves the original exactly as it was."""
    pt = pytest.importorskip("publish_tier")
    before = meta_file.read_text(encoding="utf-8")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(synth_common.os, "replace", boom)
    with pytest.raises(OSError):
        _publish(pt, monkeypatch, meta_file)

    assert meta_file.read_text(encoding="utf-8") == before
    leftovers = [p.name for p in meta_file.parent.iterdir()
                 if p.name.startswith(".") and p.name.endswith(".tmp.jsonl")]
    assert not leftovers, f"temp file left behind: {leftovers}"


def test_the_backfill_lands(meta_file, monkeypatch):
    pt = pytest.importorskip("publish_tier")
    assert _publish(pt, monkeypatch, meta_file) is None
    rows = [json.loads(l) for l in meta_file.read_text(encoding="utf-8").splitlines() if l]
    assert len(rows) == 3
    assert all(r["engine_license"] == "Apache-2.0" and r["publish"] is True for r in rows)


def test_the_second_run_does_not_call_a_stale_backup_this_runs_original(
        meta_file, monkeypatch, capsys):
    """The `.bak` is only ever taken on the first run, and that is correct — `--apply` only
    fills blanks, so there is nothing about an earlier run to undo and the pristine
    original is the artefact worth keeping. What was wrong is that every run printed
    "original kept at …", so run two named a file that is NOT its own pre-state as though
    it were an undo point."""
    pt = pytest.importorskip("publish_tier")
    _publish(pt, monkeypatch, meta_file)
    assert "original kept at" in capsys.readouterr().out

    with meta_file.open("a", encoding="utf-8") as fh:      # a later render is appended
        fh.write(json.dumps({"id": "c9", "engine": "qwen"}) + "\n")
    _publish(pt, monkeypatch, meta_file)
    out = capsys.readouterr().out
    assert "PRE-FIRST-RUN" in out
    assert "not this run's pre-state" in out

    bak = meta_file.with_suffix(meta_file.suffix + ".pre-publish-tier.bak")
    assert "engine_license" not in bak.read_text(encoding="utf-8")
