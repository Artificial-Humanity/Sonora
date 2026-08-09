"""One implementation of "safely rewrite ratings.csv" (C-M6).

`ratings.csv` is the ear-verdict SSOT and the Dataset Auditions app is a LIVE WRITER, so
a script that reads it, thinks and writes back can erase an edit committed in between.
That happened on 2026-07-26 to an owner-set accent value, and the response was six scripts
each growing their own mtime stamp — in four flavours, the widest of which stamps before
serializing ~1,500 rows and never re-checks.

Nothing here touches the real file: every case runs against a tmp_path copy.
"""

import csv
import os
import pathlib
import sys

import pytest

SYNTH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "synthesis"
sys.path.insert(0, str(SYNTH))

synth_common = pytest.importorskip("synth_common")

HEADER = ["id", "status", "score", "note"]


@pytest.fixture()
def ratings(tmp_path):
    path = tmp_path / "ratings.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows([{"id": "a", "status": "unaudited", "score": "", "note": ""},
                     {"id": "b", "status": "keep", "score": "5", "note": "good"}])
    return path


def _rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_a_clean_exit_writes(ratings):
    with synth_common.ratings_transaction(ratings, backup=False) as (_hdr, rows):
        rows[0]["status"] = "keep"
    assert _rows(ratings)[0]["status"] == "keep"


def test_a_raise_writes_nothing(ratings):
    with pytest.raises(RuntimeError):
        with synth_common.ratings_transaction(ratings, backup=False) as (_hdr, rows):
            rows[0]["status"] = "keep"
            raise RuntimeError("boom")
    assert _rows(ratings)[0]["status"] == "unaudited"


def test_a_concurrent_edit_aborts_rather_than_erasing_it(ratings):
    """The 2026-07-26 sequence exactly: the app commits between our read and our write."""
    with pytest.raises(SystemExit, match="changed while this was running"):
        with synth_common.ratings_transaction(ratings, backup=False) as (_hdr, rows):
            rows[0]["status"] = "keep"
            # the app, which takes no lock
            live = _rows(ratings)
            live[1]["note"] = "owner set accent"
            with ratings.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=HEADER)
                w.writeheader()
                w.writerows(live)
            os.utime(ratings, ns=(0, 10**9))
    assert _rows(ratings)[1]["note"] == "owner set accent", "the app's edit was erased"


def test_dry_run_reads_but_never_writes(ratings):
    """Every converted script has a report-only mode. Without this each would have to
    rebuild the read itself — the duplication being removed — or write a byte-identical
    file, taking a backup and moving the mtime for a run that was supposed to change
    nothing."""
    before = ratings.stat().st_mtime_ns
    with synth_common.ratings_transaction(ratings, backup=False, dry_run=True) as (_h, rows):
        assert len(rows) == 2, "a dry run must still see the data"
        rows[0]["status"] = "keep"
    assert _rows(ratings)[0]["status"] == "unaudited"
    assert ratings.stat().st_mtime_ns == before


def test_dry_run_sentinel_leaves_without_writing(ratings):
    """`DryRun` is for a script that only discovers mid-transaction that it has nothing
    to do. It must not escape to the caller as an error."""
    with synth_common.ratings_transaction(ratings, backup=False) as (_hdr, rows):
        rows[0]["status"] = "keep"
        raise synth_common.DryRun
    assert _rows(ratings)[0]["status"] == "unaudited"


def test_a_backup_is_taken_once_per_day_per_tag(ratings):
    with synth_common.ratings_transaction(ratings, tag="t") as (_hdr, rows):
        rows[0]["status"] = "keep"
    baks = list(ratings.parent.glob("ratings.csv.bak-*-t"))
    assert len(baks) == 1
    with synth_common.ratings_transaction(ratings, tag="t") as (_hdr, rows):
        rows[0]["status"] = "drop"
    assert len(list(ratings.parent.glob("ratings.csv.bak-*-t"))) == 1, "clobbered the backup"


def test_requeue_reroll_uses_the_shared_transaction():
    src = (SYNTH / "requeue_reroll.py").read_text(encoding="utf-8")
    assert "synth_common.ratings_transaction" in src
    assert "def write_rows(" not in src, "still carries its own writer"
    assert "st_mtime_ns" not in src, "still carries its own mtime flavour"


# --- all six writers converted ---------------------------------------------------------


@pytest.mark.parametrize("script", [
    "pick_audit_subset.py", "stage_pool.py", "seed_delivery.py",
    "reader_profile.py", "sweep_dropped.py", "requeue_reroll.py",
])
def test_every_ratings_writer_uses_the_shared_transaction(script):
    """Six scripts each grew an mtime stamp after the 2026-07-26 loss, in four flavours.
    They worked; they were each their own. The interesting casualty of converting them is
    `pick_audit_subset`'s five-attempt retry loop: with an flock serialising our own
    scripts, the only writer left to race is a human in a browser, and retrying is a way
    to eventually win a race we should not be running."""
    src = (SYNTH / script).read_text(encoding="utf-8")
    assert "synth_common.ratings_transaction" in src, f"{script} not converted"
    assert "st_mtime" not in src, f"{script} still carries its own mtime flavour"
    assert "mkstemp" not in src, f"{script} still carries its own writer"


def test_the_conflict_gate_blocks_a_hint_on_a_disputed_title():
    """C-M7. The gate was the truthiness of the whole title entry, but a title whose ear
    pass disagreed with ITSELF still produces one — `learn()` writes `{attr}_CONFLICT`
    instead of `{attr}` and `clips_seen` is set either way. So the hint fired and wrote a
    cross-title guess into the one title we have positive evidence is inconsistent, then
    marked it machine-written so it thereafter looked settled."""
    rp = pytest.importorskip("reader_profile")
    assert rp._title_is_settled({}, "gender") is False
    assert rp._title_is_settled({"gender": "F", "clips_seen": 4}, "gender") is True
    assert rp._title_is_settled(
        {"gender_CONFLICT": {"F": 2, "M": 2}, "clips_seen": 4}, "gender") is False
    # A conflict on a DIFFERENT attribute must not block this one — that title's gender
    # evidence can still be clean.
    assert rp._title_is_settled({"age_CONFLICT": {"a": 1}, "clips_seen": 4}, "gender") is True


# --- QC-M5: the app was the one writer outside the lock -------------------------------


def _app_module(tmp_path, monkeypatch):
    """Import the audition app against a synthetic ratings dir."""
    import importlib
    monkeypatch.setenv("AUDITION_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDITION_RATINGS_DIR", str(tmp_path))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "audition"))
    sys.modules.pop("app.main", None)
    return importlib.import_module("app.main")


def test_the_app_and_the_scripts_take_the_same_lock(tmp_path, monkeypatch):  # noqa: D401
    """QC-M5. The scripts guard themselves against the app (mtime re-check inside their
    own lock), but nothing guarded the reverse: a `stage_pool` or `pick_audit_subset`
    commit landing inside the app's read -> `os.replace` window was silently overwritten.
    Milliseconds wide, and the loss — a whole staging run's rows — is silent.

    The two cannot share code: the app runs in a `python:3.12-slim` container with only
    fastapi and uvicorn, so `synth_common` is not importable there. They share the
    filesystem instead, which means the lock PATH is the contract and this is the test
    that holds it.
    """
    app = _app_module(tmp_path, monkeypatch)
    assert app._lock._path == app.RATINGS_CSV
    # Byte-identical to `synth_common._exclusive`'s sidecar convention.
    src = (SYNTH / "synth_common.py").read_text(encoding="utf-8")
    assert 'lock_path = f"{path}.lock"' in src
    assert 'f"{self._path}.lock"' in (
        pathlib.Path(app.__file__).read_text(encoding="utf-8"))


def test_the_app_lock_actually_excludes_a_script(tmp_path, monkeypatch):
    """Behavior, not path-matching: while the app holds its lock, a script transaction on
    the same file must block rather than proceed."""
    app = _app_module(tmp_path, monkeypatch)
    path = tmp_path / "ratings.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerow({"id": "a", "status": "unaudited", "score": "", "note": ""})

    import subprocess

    # A separate PROCESS, because that is the whole point: a threading.Lock cannot see
    # one, and every script that writes this file is one.
    probe = (
        "import fcntl,sys\n"
        "fh = open(sys.argv[1] + '.lock', 'a+')\n"
        "try:\n"
        "    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    print('FREE')\n"
        "except BlockingIOError:\n"
        "    print('HELD')\n"
    )

    def probe_lock():
        return subprocess.run([sys.executable, "-c", probe, str(path)],
                              capture_output=True, text=True, timeout=30).stdout.strip()

    with app._lock:
        assert probe_lock() == "HELD", "a script could have written inside the app's window"
    # ...and it is released afterwards, or the app deadlocks every script on the box.
    assert probe_lock() == "FREE"


def test_the_app_lock_is_reusable_and_released_on_error(tmp_path, monkeypatch):
    """11 call sites enter it in sequence; a raise inside one must not wedge the rest."""
    app = _app_module(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        with app._lock:
            raise ValueError("boom")
    with app._lock:            # would hang forever if the release leaked
        pass
