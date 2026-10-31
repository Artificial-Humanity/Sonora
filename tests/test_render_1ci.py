"""Phase 1S Step 1c-i's planner: the banks it writes and what it refuses.

`scripts/tools/render_1ci.py` writes one campaign dir per (anchor, engine) cell for
`synth_bank.sh`. These tests run `plan` and `render` against a fake text bank and fake anchor
clips (the registered hashes are swapped for the fakes'), so they need neither /data nor a GPU.
"""
import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from scripts_layout import REPO, SCRIPTS

SCRIPTS.on_path()

import book_ingest  # noqa: E402
import check_bank  # noqa: E402
import probe_text_bank  # noqa: E402
import ref_select  # noqa: E402
import render_1ci as r1  # noqa: E402

N = probe_text_bank.BANK_SIZE
CELLS = ["M4-s3_chatterbox", "M4-s3_zonos", "F6-s2_chatterbox", "F6-s2_zonos"]


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _text_rows():
    return [{"id": "1ci-%04d" % i, "text": "This is narration line number %d of the bank." % i}
            for i in range(1, N + 1)]


def _write_text(root, rows, meta_sha=None):
    d = root / "text"
    d.mkdir(exist_ok=True)
    (d / "lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sha = probe_text_bank.canonical_sha256(rows)
    (d / "meta.json").write_text(json.dumps({"sha256": meta_sha or sha}))
    return d, sha


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fake inputs wired into render_1ci, a fixed commit and a clean tree."""
    rows = _text_rows()
    tdir, sha = _write_text(tmp_path, rows)
    monkeypatch.setattr(r1, "TEXT_BANK", str(tdir / "lines.jsonl"))
    monkeypatch.setattr(r1, "TEXT_META", str(tdir / "meta.json"))
    monkeypatch.setattr(r1, "TEXT_SHA256", sha)
    adir = tmp_path / "anchors"
    (adir / "clips").mkdir(parents=True)
    pins, manifest = [], []
    for aid, gender, blob in (("M4-s3", "Male", b"male clip"), ("F6-s2", "Female", b"female clip")):
        (adir / "clips" / (aid + ".wav")).write_bytes(blob)
        pins.append((aid, _sha(blob)))
        manifest.append({"id": aid, "file": "clips/%s.wav" % aid, "gender": gender,
                         "text": "The anchor %s says this." % aid})
    # an anchor that is not locked must not be picked up
    manifest.append({"id": "F2-s1", "file": "clips/F2-s1.wav", "gender": "Female", "text": "x"})
    (adir / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    monkeypatch.setattr(r1, "ANCHOR_DIR", str(adir))
    monkeypatch.setattr(r1, "ANCHORS", tuple(pins))
    monkeypatch.setattr(ref_select, "REF_FILE_ROOT", str(tmp_path) + "/")
    zonos = tmp_path / "zonos-checkout"                 # engine_pin refuses an unreadable one
    zonos.mkdir()
    _g(tmp_path, "init", "-q", str(zonos))
    (zonos / "model.py").write_text("v0\n")
    _g(zonos, "add", "-A")
    _g(zonos, "commit", "-q", "-m", "v0")
    monkeypatch.setattr(r1, "ZONOS_SOURCE", str(zonos))
    git = {"commit": "c0ffee", "tree_clean": True}
    monkeypatch.setattr(r1, "tool_commit", lambda: dict(git))
    return {"root": tmp_path, "out": tmp_path / "out", "rows": rows, "adir": adir, "git": git}


def _plan(env, *extra):
    return r1.main(["plan", "--out-root", str(env["out"]), *extra])


def _bank(env, cell):
    return json.loads((env["out"] / cell / "bank.json").read_text())


def _meta(env, cell):
    return json.loads((env["out"] / cell / "plan_meta.json").read_text())


# --- plan ---------------------------------------------------------------------------------

def test_plan_writes_one_separate_campaign_dir_per_cell(env):
    _plan(env)
    assert sorted(p.name for p in env["out"].iterdir()) == sorted(CELLS)
    for cell in CELLS:
        assert sorted(p.name for p in (env["out"] / cell).iterdir()) == ["bank.json",
                                                                        "plan_meta.json"]
    banks = [_bank(env, c) for c in CELLS]
    assert len({b["campaign"] for b in banks}) == 4
    ids = [ln["id"] for b in banks for ln in b["lines"]]
    assert len(ids) == len(set(ids)) == 4 * N, "a clip id repeats across cells"


def test_each_cell_renders_into_its_own_qc_campaign_dir(env):
    """synth_bank.sh gates `dirname "$OUT"`: OUT must be <cell>/audio, never shared."""
    _plan(env)
    outs = [r1.render_command(str(env["out"]), c, c.split("_")[1])[2] for c in CELLS]
    assert [os.path.dirname(o) for o in outs] == [str(env["out"] / c) for c in CELLS]
    assert all(os.path.basename(o) == "audio" for o in outs)


def test_bank_lines_are_the_text_bank_in_order_with_a_1_based_render_index(env):
    _plan(env)
    for cell in CELLS:
        lines = _bank(env, cell)["lines"]
        assert [ln["render_index"] for ln in lines] == list(range(1, N + 1))
        assert [ln["text_id"] for ln in lines] == [r["id"] for r in env["rows"]]
        assert [ln["text"] for ln in lines] == [r["text"] for r in env["rows"]]
        assert all(ln["engine"] == cell.split("_")[1] for ln in lines)


def test_no_bank_carries_intended_labels(env):
    """No `intended` V/A/T, so qc_verdict counts the campaign as undirected and no EIV runs."""
    _plan(env)
    for cell in CELLS:
        assert not any("intended" in ln for ln in _bank(env, cell)["lines"])


# The values each engine's lines must carry, pinned here rather than derived: Zonos's are the lead
# resident's decision (render_1ci.ZONOS_DIRECTION_DECISION), Chatterbox's are build_direction's
# Neutral-lane defaults (0.5 capped to NARRATION_MAX_EXAGGERATION, cfg_weight 0.5).
PINNED_DIRECTION = {
    "zonos": {"pitch_std": 28.0, "speaking_rate": 14.0, "emotion": None},
    "chatterbox": {"exaggeration": 0.4, "cfg_weight": 0.5},
}


@pytest.mark.parametrize("engine", ["chatterbox", "zonos"])
def test_direction_is_build_directions_neutral_payload_on_every_line(env, engine):
    _plan(env)
    tag = dict(r1.DIRECTION_TAGS[engine], engine=engine)
    want = book_ingest.build_direction(tag, "any text", lane="Neutral")[1]
    assert {k: want[k] for k in PINNED_DIRECTION[engine]} == PINNED_DIRECTION[engine]
    for k, v in PINNED_DIRECTION[engine].items():
        assert type(want[k]) is type(v), (k, want[k])
    for anchor in ("M4-s3", "F6-s2"):
        cell = "%s_%s" % (anchor, engine)
        assert all(ln["direction"] == want for ln in _bank(env, cell)["lines"])
        meta = _meta(env, cell)["direction"]
        assert meta["direction"] == want and meta["lane"] == "Neutral"
        assert meta["tag"] == tag
    assert "28.0" in _meta(env, "M4-s3_zonos")["direction"]["decision"]


def _skill_row(engine, lane):
    """{column: cell} for `lane`'s row in the narration table of the engine's director skill."""
    text = (REPO / "scripts" / "assets" / "director_skills" / ("%s.md" % engine)).read_text(
        encoding="utf-8")
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        cells = [c.strip().strip("`") for c in ln.strip().strip("|").split("|")]
        if cells and cells[0] == "lane" and "pitch_std" in cells:
            header = cells
            for row in lines[i + 2:]:
                vals = [c.strip() for c in row.strip().strip("|").split("|")]
                if not row.startswith("|"):
                    break
                if vals[0] == lane:
                    return dict(zip(header, vals))
    raise AssertionError("no %s row in %s.md's lane table" % (lane, engine))


def _span(cell):
    lo, hi = re.fullmatch(r"\**(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\**", cell).groups()
    return float(lo), float(hi)


def test_the_zonos_direction_is_inside_zonos_mds_neutral_row():
    row = _skill_row("zonos", "Neutral")
    d = r1.DIRECTION_TAGS["zonos"]
    lo, hi = _span(row["pitch_std"])
    assert lo <= d["pitch_std"] <= hi, (d["pitch_std"], row)
    lo, hi = _span(row["speaking_rate"])
    assert lo <= d["speaking_rate"] <= hi, (d["speaking_rate"], row)
    assert "omit" in row["emotion"].lower() and d["emotion"] is None


def test_a_direction_build_direction_would_clamp_refuses(env, monkeypatch):
    monkeypatch.setitem(r1.DIRECTION_TAGS, "zonos", dict(r1.DIRECTION_TAGS["zonos"], pitch_std=10.0))
    with pytest.raises(SystemExit, match="changed the zonos direction"):
        _plan(env)
    assert not env["out"].exists()


def test_a_direction_that_varies_by_line_refuses(env, monkeypatch):
    real = book_ingest.build_direction

    def varying(tag, text, **kw):
        engine, d = real(tag, text, **kw)
        return engine, dict(d, design=text)
    monkeypatch.setattr(book_ingest, "build_direction", varying)
    with pytest.raises(SystemExit, match="more than one direction"):
        _plan(env)
    assert not env["out"].exists()


def test_seeds_follow_the_rule_and_are_the_same_in_every_cell_and_every_plan(env):
    _plan(env)
    again = env["root"] / "again"
    r1.main(["plan", "--out-root", str(again)])
    for cell in CELLS:
        lines = _bank(env, cell)["lines"]
        assert [ln["seed"] for ln in lines] == [r1.SEED_BASE + i for i in range(1, N + 1)]
        assert json.loads((again / cell / "bank.json").read_text()) == _bank(env, cell)
    assert _meta(env, CELLS[0])["seeds"]["rule"] == r1.SEED_RULE


def test_every_line_pins_its_cells_anchor_by_file(env):
    _plan(env)
    for cell in CELLS:
        anchor = cell.split("_")[0]
        pin = dict(r1.ANCHORS)[anchor]
        refs = {json.dumps(ln["ref_file"], sort_keys=True) for ln in _bank(env, cell)["lines"]}
        assert len(refs) == 1
        ref = json.loads(refs.pop())
        assert ref["path"] == str(env["adir"] / "clips" / (anchor + ".wav"))
        assert ref["sha256"] == pin and ref["id"] == anchor
        assert ref["text"] == "The anchor %s says this." % anchor
        assert ref["gender"] == ("Male" if anchor == "M4-s3" else "Female")
        assert not any(ln.get("ref_id") for ln in _bank(env, cell)["lines"])


def test_the_banks_pass_check_bank(env):
    _plan(env)
    for cell in CELLS:
        path = str(env["out"] / cell / "bank.json")
        assert check_bank.check(path) == []
        assert check_bank.check_refs(path) == []
        assert check_bank.main(["check_bank.py", path]) == 0


def test_plan_meta_records_the_provenance(env):
    _plan(env)
    meta = _meta(env, "F6-s2_zonos")
    assert meta["commit"] == "c0ffee" and meta["tree_clean"] is True
    assert meta["text"]["sha256"] == r1.TEXT_SHA256 and meta["text"]["lines"] == N
    assert meta["anchor"]["id"] == "F6-s2" and meta["anchor"]["sha256"] == dict(r1.ANCHORS)["F6-s2"]
    assert meta["synth_bank"]["env"] == {"SYNTH_BANK_PROBE": "1"}
    assert meta["engine_pin"]["source_checkout"] == r1.ZONOS_SOURCE
    assert "not available" in meta["batched_run"]
    assert "leading-gap" in meta["synth_bank"]["known_gap"]
    assert meta["render_affecting_paths"] == list(r1.RENDER_AFFECTING_PATHS)
    assert _meta(env, "F6-s2_chatterbox")["engine_pin"]["wheel"] == "chatterbox-tts==0.1.7"


# --- what plan refuses --------------------------------------------------------------------

def test_a_text_bank_that_does_not_hash_to_the_registered_sha_refuses(env):
    rows = env["rows"][:]
    rows[149] = dict(rows[149], text=rows[149]["text"] + " Changed.")
    _write_text(env["root"], rows)
    with pytest.raises(SystemExit, match="canonical sha256"):
        _plan(env)
    assert not env["out"].exists()


def test_a_meta_json_recording_another_sha_refuses(env):
    _write_text(env["root"], env["rows"], meta_sha="0" * 64)
    with pytest.raises(SystemExit, match="records sha256"):
        _plan(env)


def test_a_text_bank_of_the_wrong_size_refuses(env, monkeypatch):
    rows = env["rows"][:-1]
    _d, sha = _write_text(env["root"], rows)
    monkeypatch.setattr(r1, "TEXT_SHA256", sha)
    with pytest.raises(SystemExit, match="299 lines"):
        _plan(env)


@pytest.mark.parametrize("anchor", ["M4-s3", "F6-s2"])
def test_an_anchor_clip_that_does_not_match_its_pin_refuses(env, anchor):
    (env["adir"] / "clips" / (anchor + ".wav")).write_bytes(b"another take")
    with pytest.raises(SystemExit, match="not the pinned"):
        _plan(env)
    assert not env["out"].exists()


def test_a_missing_anchor_row_refuses(env, monkeypatch):
    monkeypatch.setattr(r1, "ANCHORS", (("M9-s9", "0" * 64),) + r1.ANCHORS[1:])
    with pytest.raises(SystemExit, match="no row M9-s9"):
        _plan(env)


def test_a_dirty_tree_refuses_unless_allowed_and_is_recorded(env):
    env["git"]["tree_clean"] = False
    with pytest.raises(SystemExit, match="uncommitted"):
        _plan(env)
    _plan(env, "--allow-dirty")
    assert _meta(env, CELLS[0])["tree_clean"] is False


def test_replanning_the_same_inputs_is_allowed_but_a_different_bank_is_not(env, monkeypatch):
    _plan(env)
    _plan(env)
    monkeypatch.setattr(r1, "SEED_BASE", r1.SEED_BASE + 1)
    with pytest.raises(SystemExit, match="differs from this plan"):
        _plan(env)


def test_a_cell_with_renders_from_another_commit_is_not_replanned(env):
    _plan(env)
    audio = env["out"] / CELLS[0] / "audio"
    audio.mkdir()
    (audio / "chatterbox_manifest.jsonl").write_text("{}\n")
    _plan(env)                                   # same commit: fine
    env["git"]["commit"] = "deadbeef"
    with pytest.raises(SystemExit, match="another commit") as exc:
        _plan(env)
    msg = str(exc.value)
    assert "c0ffee" in msg and "deadbeef" in msg
    assert "render %s" % CELLS[0] in msg and "new --out-root" in msg, "no way out named"


def test_a_cell_with_renders_is_not_replanned_under_a_changed_engine_pin(env, monkeypatch):
    """Audit M-6: plan rewrote plan_meta.engine_pin unconditionally, so the render launch gate then
    passed a half-rendered cell on different engine code."""
    _plan(env)
    audio = env["out"] / "M4-s3_zonos" / "audio"
    audio.mkdir()
    (audio / "zonos_manifest.jsonl").write_text("{}\n")
    before = (env["out"] / "M4-s3_zonos" / "plan_meta.json").read_text()
    real = r1.engine_pin
    monkeypatch.setattr(r1, "engine_pin", lambda e: dict(real(e), commit="moved") if e == "zonos" else real(e))
    with pytest.raises(SystemExit, match="engine pin"):
        _plan(env)
    assert (env["out"] / "M4-s3_zonos" / "plan_meta.json").read_text() == before
    # A cell with no renders may be re-planned under the new pin:
    (audio / "zonos_manifest.jsonl").unlink()
    _plan(env)
    assert _meta(env, "M4-s3_zonos")["engine_pin"]["commit"] == "moved"


# --- render -------------------------------------------------------------------------------

class Launches(list):
    """The synth_bank.sh launches recorded by `no_exec`, and what the fake does on each: exit
    with `code`, and (when `write`) append a manifest row for every bank id not in `drop`."""
    code = 0
    write = True
    drop = ()


def _write_rows(audio, engine, ids):
    audio.mkdir(parents=True, exist_ok=True)
    with open(audio / ("%s_manifest.jsonl" % engine), "a") as fh:
        for i in ids:
            fh.write(json.dumps({"id": i, "engine": engine}) + "\n")


@pytest.fixture
def no_exec(monkeypatch):
    """Records each synth_bank.sh launch instead of running it; anything else runs. By default
    the fake renders the whole bank (one manifest row per id) and exits 0."""
    calls, real = Launches(), r1.subprocess.run

    def fake_run(argv, **kw):
        if argv[0] != r1.SYNTH_BANK:
            return real(argv, **kw)
        calls.append((argv, kw))
        if calls.write:
            ids = [ln["id"] for ln in json.loads(open(argv[1]).read())["lines"]]
            _write_rows(Path(argv[2]), argv[3],
                        [i for i in ids if i not in calls.drop])

        class Done:
            returncode = calls.code
        return Done()
    monkeypatch.setattr(r1.subprocess, "run", fake_run)
    return calls


def test_render_prints_the_probe_command_and_runs_nothing(env, no_exec, capsys):
    _plan(env)
    assert r1.main(["render", "M4-s3_zonos", "--out-root", str(env["out"])]) == 0
    out = capsys.readouterr().out
    cdir = env["out"] / "M4-s3_zonos"
    assert ("SYNTH_BANK_PROBE=1 %s %s %s zonos" % (r1.SYNTH_BANK, cdir / "bank.json",
                                                    cdir / "audio")) in out
    assert "printed only" in out
    assert no_exec == []


def test_render_refuses_when_the_plan_no_longer_holds(env, no_exec):
    _plan(env)
    args = ["render", "F6-s2_chatterbox", "--out-root", str(env["out"])]
    (env["adir"] / "clips" / "F6-s2.wav").write_bytes(b"swapped")
    with pytest.raises(SystemExit, match="not the pinned"):
        r1.main(args)


def test_render_refuses_an_edited_bank(env, no_exec):
    _plan(env)
    path = env["out"] / "M4-s3_chatterbox" / "bank.json"
    bank = json.loads(path.read_text())
    bank["lines"][0]["direction"]["exaggeration"] = 0.7
    path.write_text(json.dumps(bank))
    with pytest.raises(SystemExit, match="is not the bank"):
        r1.main(["render", "M4-s3_chatterbox", "--out-root", str(env["out"])])


def test_render_refuses_another_commit_and_a_dirty_tree(env, no_exec):
    _plan(env)
    args = ["render", "M4-s3_chatterbox", "--out-root", str(env["out"])]
    env["git"]["tree_clean"] = False
    with pytest.raises(SystemExit, match="uncommitted"):
        r1.main(args)
    assert r1.main(args + ["--allow-dirty"]) == 0
    env["git"].update(commit="deadbeef", tree_clean=True)     # a commit git does not have
    with pytest.raises(SystemExit, match="git cannot"):
        r1.main(args)


def test_render_refuses_an_unknown_cell_and_an_unplanned_one(env, no_exec):
    with pytest.raises(SystemExit, match="no cell"):
        r1.main(["render", "M4-s3_qwen", "--out-root", str(env["out"])])
    with pytest.raises(SystemExit, match="Run `plan` first"):
        r1.main(["render", "M4-s3_zonos", "--out-root", str(env["out"])])


def test_render_run_refuses_outside_data_or_without_the_venv(env, no_exec, monkeypatch):
    _plan(env)
    args = ["render", "M4-s3_zonos", "--out-root", str(env["out"]), "--run"]
    with pytest.raises(SystemExit, match="blockers"):
        r1.main(args)
    assert no_exec == []


def test_render_run_executes_synth_bank_in_probe_mode_and_logs_it(env, no_exec, monkeypatch):
    _plan(env)
    venv = env["root"] / "python"
    venv.write_text("#!/bin/sh\n")
    venv.chmod(0o755)
    monkeypatch.setattr(r1, "DATA_ROOT", str(env["root"]) + "/")
    monkeypatch.setattr(r1, "VENV_PYTHON", str(venv))
    assert r1.main(["render", "M4-s3_zonos", "--out-root", str(env["out"]), "--run"]) == 0
    (argv, kw), = no_exec
    cdir = env["out"] / "M4-s3_zonos"
    assert argv == [r1.SYNTH_BANK, str(cdir / "bank.json"), str(cdir / "audio"), "zonos"]
    assert kw["env"]["SYNTH_BANK_PROBE"] == "1"
    log = [json.loads(x) for x in (cdir / "render_log.jsonl").read_text().splitlines()]
    assert log[0]["exit_code"] == 0
    assert log[0]["plan_commit"] == log[0]["head_commit"] == "c0ffee"
    assert log[0]["command"].startswith("SYNTH_BANK_PROBE=1 ")
    assert (log[0]["rendered"], log[0]["expected"], log[0]["missing"]) == (N, N, [])


# --- a partly rendered cell ---------------------------------------------------------------

def _log(env, cell):
    return [json.loads(x) for x in
            (env["out"] / cell / "render_log.jsonl").read_text().splitlines()]


def _run(env, cell):
    return r1.main(["render", cell, "--out-root", str(env["out"]), "--run"])


def test_render_run_with_missing_rows_exits_non_zero_naming_them(env, no_exec, monkeypatch,
                                                                 capsys):
    """A renderer's CASTING FAILED skips the clip and exits 0; synth_bank.sh then exits 0."""
    _plan(env)
    _run_ready(env, monkeypatch)
    cell = "F6-s2_chatterbox"
    no_exec.drop = ("%s-017" % cell, "%s-250" % cell)
    assert _run(env, cell) == r1.PARTIAL_EXIT != 0
    err = capsys.readouterr().err
    assert "PARTIAL RENDER: %s is missing 2 of %d clips: %s-017, %s-250." % (cell, N, cell, cell) \
        in err
    assert "synth_bank.sh exited 0" in err and "Re-run `render %s --run`" % cell in err
    log, = _log(env, cell)
    assert log["exit_code"] == 0
    assert (log["rendered"], log["expected"]) == (N - 2, N)
    assert log["missing"] == ["%s-017" % cell, "%s-250" % cell]


def test_many_missing_ids_are_listed_first_n_plus_a_count(env, no_exec, monkeypatch, capsys):
    _plan(env)
    _run_ready(env, monkeypatch)
    cell = "M4-s3_zonos"
    no_exec.drop = tuple("%s-%03d" % (cell, i) for i in range(1, 26))
    assert _run(env, cell) == r1.PARTIAL_EXIT
    err = capsys.readouterr().err
    assert "missing 25 of %d clips: %s-001, " % (N, cell) in err
    assert "%s-010 and 15 more." % cell in err and "%s-011" % cell not in err
    assert len(_log(env, cell)[0]["missing"]) == 25


def test_a_resumed_run_that_completes_the_cell_exits_zero(env, no_exec, monkeypatch):
    _plan(env)
    _run_ready(env, monkeypatch)
    cell = "M4-s3_chatterbox"
    no_exec.drop = ("%s-100" % cell,)
    assert _run(env, cell) == r1.PARTIAL_EXIT
    no_exec.drop = ()                       # the re-run renders the clip that failed
    assert _run(env, cell) == 0
    assert [x["rendered"] for x in _log(env, cell)] == [N - 1, N]


def test_a_failing_synth_bank_keeps_its_exit_status(env, no_exec, monkeypatch, capsys):
    _plan(env)
    _run_ready(env, monkeypatch)
    no_exec.code = 1
    assert _run(env, "F6-s2_zonos") == 1                       # every row written
    no_exec.write = False                                       # and with every row missing
    assert _run(env, "M4-s3_zonos") == 1
    err = capsys.readouterr().err
    assert "synth_bank.sh exited 1." in err and "missing %d of %d" % (N, N) in err
    assert "synth_bank.sh exited 0" not in err
    assert _log(env, "M4-s3_zonos")[0]["rendered"] == 0


def test_a_missing_clip_with_a_wav_but_no_row_says_a_rerun_will_not_fix_it(
        env, no_exec, monkeypatch, capsys):
    _plan(env)
    _run_ready(env, monkeypatch)
    cell = "F6-s2_zonos"
    no_exec.drop = ("%s-005" % cell,)
    (env["out"] / cell / "audio").mkdir(parents=True)
    (env["out"] / cell / "audio" / ("%s-005.wav" % cell)).write_bytes(b"orphan")
    assert _run(env, cell) == r1.PARTIAL_EXIT
    assert "have a wav but no manifest row (%s-005)" % cell in capsys.readouterr().err
    assert _log(env, cell)[0]["wav_without_row"] == ["%s-005" % cell]


def test_status_reports_the_counts_and_launches_nothing(env, no_exec, capsys):
    _plan(env)
    cell = "M4-s3_zonos"
    args = ["status", cell, "--out-root", str(env["out"])]
    assert r1.main(args) == r1.PARTIAL_EXIT                    # nothing rendered yet
    assert "%s: 0 of %d bank ids" % (cell, N) in capsys.readouterr().out
    audio = env["out"] / cell / "audio"
    _write_rows(audio, "zonos", ["%s-%03d" % (cell, i) for i in range(1, N)])
    with open(audio / "zonos_manifest.jsonl", "a") as fh:
        fh.write('{"id": "cut sho\n')                          # a write cut short
    assert r1.main(args) == r1.PARTIAL_EXIT
    out, err = capsys.readouterr()
    assert "%s: %d of %d" % (cell, N - 1, N) in out and "1 line(s)" in out
    assert "missing 1 of %d clips: %s-%03d." % (N, cell, N) in err
    _write_rows(audio, "zonos", ["%s-001" % cell])             # a reroll: N rows, N - 1 ids
    assert r1.main(args) == r1.PARTIAL_EXIT
    assert "%s: %d of %d" % (cell, N - 1, N) in capsys.readouterr().out
    _write_rows(audio, "zonos", ["%s-%03d" % (cell, N)] * 2)
    assert r1.main(args) == 0
    assert no_exec == []
    with pytest.raises(SystemExit, match="Run `plan` first"):
        r1.main(["status", cell, "--out-root", str(env["root"] / "nowhere")])


def test_render_without_run_prints_the_counts_of_a_started_cell(env, no_exec, capsys):
    _plan(env)
    cell = "F6-s2_chatterbox"
    _write_rows(env["out"] / cell / "audio", "chatterbox", ["%s-001" % cell])
    assert r1.main(["render", cell, "--out-root", str(env["out"])]) == 0
    out = capsys.readouterr()
    assert "%s: 1 of %d bank ids" % (cell, N) in out.out and "PARTIAL RENDER" in out.err
    assert no_exec == []


# --- the engine pin -----------------------------------------------------------------------

def test_render_refuses_a_changed_chatterbox_pin_naming_both(env, no_exec, tmp_path, monkeypatch):
    pins = tmp_path / "container_env.sh"
    pins.write_text('SONORA_PIN_CHATTERBOX="chatterbox-tts==0.1.7"\n')
    monkeypatch.setattr(r1, "CONTAINER_ENV", str(pins))
    _plan(env)
    args = ["render", "M4-s3_chatterbox", "--out-root", str(env["out"])]
    assert r1.main(args) == 0                                   # unchanged: passes
    pins.write_text('SONORA_PIN_CHATTERBOX="chatterbox-tts==0.1.8"\n')
    with pytest.raises(SystemExit, match="engine pin") as exc:
        r1.main(args)
    assert "chatterbox-tts==0.1.7" in str(exc.value) and "chatterbox-tts==0.1.8" in str(exc.value)
    assert no_exec == []


def test_render_refuses_a_moved_zonos_checkout_naming_both(env, no_exec, tmp_path, monkeypatch):
    zonos = tmp_path / "zonos-src"
    zonos.mkdir()
    _g(tmp_path, "init", "-q", str(zonos))
    (zonos / "a.py").write_text("v0\n")
    _g(zonos, "add", "-A")
    _g(zonos, "commit", "-q", "-m", "v0")
    planned = _g(zonos, "rev-parse", "HEAD")
    monkeypatch.setattr(r1, "ZONOS_SOURCE", str(zonos))
    _plan(env)
    args = ["render", "F6-s2_zonos", "--out-root", str(env["out"])]
    assert r1.main(args) == 0                                   # unchanged: passes
    (zonos / "a.py").write_text("v1\n")
    _g(zonos, "commit", "-q", "-am", "v1")
    moved = _g(zonos, "rev-parse", "HEAD")
    with pytest.raises(SystemExit, match="engine pin") as exc:
        r1.main(args)
    assert planned in str(exc.value) and moved in str(exc.value)
    assert no_exec == []


# --- render at a HEAD other than the plan commit ------------------------------------------

def _g(repo, *argv):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           "-c", "commit.gpgsign=false", *argv], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repo holding every RENDER_AFFECTING_PATHS file and one unrelated file, wired in as
    the checkout render_1ci compares commits in. Returns (path, commit-making function)."""
    root = tmp_path / "repo"
    for rel in list(r1.RENDER_AFFECTING_PATHS) + ["docs/unrelated.md"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("v0 %s\n" % rel)
    _g(tmp_path, "init", "-q", str(root))
    _g(root, "add", "-A")
    _g(root, "commit", "-q", "-m", "plan")
    monkeypatch.setattr(r1, "_SONORA_REPO", str(root))

    def commit(rel, text):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
        _g(root, "add", "-A")
        _g(root, "commit", "-q", "-m", "change " + rel)
        return _g(root, "rev-parse", "HEAD")
    return root, commit


def _run_ready(env, monkeypatch):
    venv = env["root"] / "python"
    venv.write_text("#!/bin/sh\n")
    venv.chmod(0o755)
    monkeypatch.setattr(r1, "DATA_ROOT", str(env["root"]) + "/")
    monkeypatch.setattr(r1, "VENV_PYTHON", str(venv))


def test_render_at_a_head_that_changed_only_unrelated_files_is_allowed_and_logs_both(
        env, repo, no_exec, monkeypatch, capsys):
    root, commit = repo
    plan_commit = _g(root, "rev-parse", "HEAD")
    env["git"]["commit"] = plan_commit
    _plan(env)
    head = commit("docs/unrelated.md", "pulled from main\n")
    env["git"]["commit"] = head
    _run_ready(env, monkeypatch)
    assert r1.main(["render", "F6-s2_zonos", "--out-root", str(env["out"]), "--run"]) == 0
    assert "no render-affecting file differs" in capsys.readouterr().out
    assert len(no_exec) == 1
    log = [json.loads(x) for x in
           (env["out"] / "F6-s2_zonos" / "render_log.jsonl").read_text().splitlines()]
    assert log[0]["plan_commit"] == plan_commit and log[0]["head_commit"] == head


@pytest.mark.parametrize("rel", ["scripts/lib/ref_select.py", "scripts/stages/synth_bank.sh",
                                 "matcha/delivery.py", "scripts/container_env.sh"])
def test_render_at_a_head_that_changed_a_render_affecting_file_refuses_naming_it(
        env, repo, no_exec, rel):
    root, commit = repo
    plan_commit = _g(root, "rev-parse", "HEAD")
    env["git"]["commit"] = plan_commit
    _plan(env)
    commit("docs/unrelated.md", "pulled\n")
    env["git"]["commit"] = commit(rel, "changed\n")
    with pytest.raises(SystemExit, match="render-affecting files differ") as exc:
        r1.main(["render", "M4-s3_chatterbox", "--out-root", str(env["out"])])
    msg = str(exc.value)
    assert plan_commit in msg and rel in msg and "docs/unrelated.md" not in msg
    assert no_exec == []


def test_render_refuses_when_a_listed_path_is_not_tracked(env, repo, no_exec, monkeypatch):
    """A misspelt entry matches nothing, and `git diff --quiet` would call it unchanged."""
    root, commit = repo
    env["git"]["commit"] = _g(root, "rev-parse", "HEAD")
    _plan(env)
    env["git"]["commit"] = commit("docs/unrelated.md", "pulled\n")
    monkeypatch.setattr(r1, "RENDER_AFFECTING_PATHS",
                        r1.RENDER_AFFECTING_PATHS + ("scripts/lib/ref_selct.py",))
    with pytest.raises(SystemExit, match="does not track: scripts/lib/ref_selct.py"):
        r1.main(["render", "M4-s3_chatterbox", "--out-root", str(env["out"])])


def test_render_refuses_a_plan_commit_git_cannot_compare(env, repo, no_exec):
    """HEAD is listable but the plan commit is unknown: `git diff --quiet` exits 128, which must
    be refused, never read as "unchanged"."""
    root, _commit = repo
    env["git"]["commit"] = "0123456789abcdef0123456789abcdef01234567"   # not in this repo
    _plan(env)
    env["git"]["commit"] = _g(root, "rev-parse", "HEAD")
    with pytest.raises(SystemExit, match="git cannot compare the plan commit 0123456789abcdef"):
        r1.main(["render", "M4-s3_chatterbox", "--out-root", str(env["out"])])
    assert no_exec == []


def test_render_refuses_when_quiet_and_name_only_disagree(env, repo, no_exec, monkeypatch):
    """`--quiet` says the render-affecting files differ, `--name-only` names none: refused."""
    root, commit = repo
    env["git"]["commit"] = _g(root, "rev-parse", "HEAD")
    _plan(env)
    env["git"]["commit"] = commit("scripts/lib/ref_select.py", "changed\n")
    real = r1._git

    def no_names(*argv):
        if argv[0] == "diff" and "--name-only" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return real(*argv)
    monkeypatch.setattr(r1, "_git", no_names)
    with pytest.raises(SystemExit, match="--name-only names none of them"):
        r1.main(["render", "M4-s3_chatterbox", "--out-root", str(env["out"])])
    assert no_exec == []


def test_every_render_affecting_path_is_tracked_here():
    tracked = set(subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True,
                                 text=True, check=True).stdout.split())
    missing = [p for p in r1.RENDER_AFFECTING_PATHS if p not in tracked]
    assert not missing, missing
    assert len(r1.RENDER_AFFECTING_PATHS) == len(set(r1.RENDER_AFFECTING_PATHS))


def _module_level_imports(path):
    """Module names imported by statements that run at import time (not inside a function)."""
    names, todo = [], list(ast.parse(path.read_text(encoding="utf-8")).body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
        todo += [c for c in ast.iter_child_nodes(node) if isinstance(c, ast.stmt)]
    return names


def _repo_file(module):
    """The tracked file a module name resolves to on render_1ci's search path, or None."""
    parts = module.split(".")
    for base in (REPO / "scripts" / "lib", REPO / "scripts" / "stages", REPO):
        f = base.joinpath(*parts).with_suffix(".py")
        if f.is_file():
            return f
    return None


def test_the_render_affecting_paths_cover_every_repo_module_they_import():
    listed = set(r1.RENDER_AFFECTING_PATHS)
    seen, todo, unlisted = set(), [REPO / p for p in listed if p.endswith(".py")], []
    assert len(todo) >= 10, "the enumeration found too few files to mean anything"
    while todo:
        f = todo.pop()
        rel = f.relative_to(REPO).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        if rel not in listed:
            if rel not in r1.NOT_RENDER_AFFECTING:
                unlisted.append(rel)
            continue                      # an excluded module's own imports are not followed
        for module in _module_level_imports(f):
            hit = _repo_file(module)
            if hit is not None:
                todo.append(hit)
            if "." in module:              # a package's __init__ runs too
                init = _repo_file(module.split(".")[0] + ".__init__")
                if init is not None:
                    todo.append(init)
    assert not unlisted, "imported by a render-affecting file but in neither list: %s" % unlisted
    assert {"scripts/lib/schemas.py", "matcha/delivery.py"} <= seen


def test_every_script_synth_bank_names_is_classified():
    src = "".join((REPO / rel).read_text(encoding="utf-8")
                  for rel in ("scripts/stages/synth_bank.sh", "scripts/container_env.sh"))
    named = set(re.findall(r"scripts/(?:stages|lib|tools)?/?[A-Za-z0-9_]+\.(?:py|sh)", src))
    named = {n.replace("scripts//", "scripts/") for n in named}
    assert len(named) >= 10, named
    unclassified = sorted(n for n in named
                          if n not in r1.RENDER_AFFECTING_PATHS and n not in r1.NOT_RENDER_AFFECTING)
    assert not unclassified, unclassified


# --- the rest -----------------------------------------------------------------------------

def test_batched_says_it_is_not_available(capsys):
    assert r1.main(["batched"]) == 1
    assert "not available" in capsys.readouterr().out


def test_the_zonos_source_is_the_one_synth_bank_runs():
    """plan_meta's zonos pin is this checkout's commit, so it must be the one on PYTHONPATH."""
    src = (REPO / "scripts" / "stages" / "synth_bank.sh").read_text(encoding="utf-8")
    assert "PYTHONPATH=%s python /sonora/scripts/stages/synth_zonos.py" % r1.ZONOS_SOURCE in src


def test_the_registered_constants_are_the_spec_values():
    assert r1.TEXT_SHA256 == "4022c8e3b241ee15c4615ef1817dea45df8154e206c2f8f0f997d5a2b89ebee1"
    assert dict(r1.ANCHORS) == {
        "M4-s3": "f3398bb813d784a2a8ae65daf7feb1827de013ca512fa24f1d40e6c966e4fa52",
        "F6-s2": "11977e770851d2acdb5516cb743fc2b0e1f73d23bac5ce603a4f64fcd5574209"}
    assert r1.ENGINES == ("chatterbox", "zonos") and r1.LANE == "Neutral"


@pytest.mark.skipif(not os.path.isfile(r1.TEXT_BANK), reason="needs the real 1c-i text bank")
def test_the_real_inputs_verify():
    """Standing check on the real /data inputs where they exist."""
    assert len(r1.load_text()) == N
    assert [a["id"] for a in r1.load_anchors()] == ["M4-s3", "F6-s2"]


def test_plan_refuses_an_unreadable_zonos_checkout(env, tmp_path, monkeypatch):
    """An unreadable checkout would record commit None at plan and at render, and None == None
    would pass the launch check, so it is refused."""
    monkeypatch.setattr(r1, "ZONOS_SOURCE", str(tmp_path / "no-zonos-checkout"))
    with pytest.raises(SystemExit, match="cannot read the Zonos source checkout"):
        _plan(env)


def test_plan_refuses_a_dirty_zonos_checkout(env, monkeypatch):
    (Path(r1.ZONOS_SOURCE) / "model.py").write_text("edited\n")
    with pytest.raises(SystemExit, match="uncommitted changes"):
        _plan(env)
