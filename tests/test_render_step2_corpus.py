"""Phase 1S Step 2c's corpus is the pre-registered one: a 90-speaker table of the registered kinds and
counts, a speaker lock that refuses short genders and unrated anchors, bank seeds and directions from
the plan, dataset-mode rendering that never sets the probe flag, and a corpus build that refuses any
speaker under the QC floor and resamples to 22.05 kHz."""
import json
import math

import numpy as np
import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()

import render_step2_corpus as s2c  # noqa: E402
import render_1cii as r2  # noqa: E402


def _ref(i, gender):
    return {"path": "/x/%s%02d.wav" % (gender[0], i), "sha256": "0" * 64, "text": "t",
            "id": "%s%02d" % (gender[0], i), "gender": gender}


def _table():
    males = [_ref(i, "Male") for i in range(42)]
    females = [_ref(i, "Female") for i in range(43)]
    locked = {"M4-s3": _ref(0, "Male"), "F6-s2": _ref(0, "Female")}
    return s2c.build_speakers(males, females, locked)


def test_the_table_has_90_speakers_in_the_registered_order_and_kinds():
    t = _table()
    assert [s["index"] for s in t] == list(range(90))
    assert t[0]["engine"] == "qwen_base" and t[0]["gender"] == "Male"
    assert t[1]["engine"] == "qwen_base" and t[1]["gender"] == "Female"
    assert all(t[i]["engine"] == "qwen_base" and t[i]["gender"] == "Male" for i in range(2, 44))
    assert all(t[i]["engine"] == "qwen_base" and t[i]["gender"] == "Female" for i in range(44, 87))
    assert t[87]["engine"] == "qwen_custom" and t[87]["speaker"] == "ryan"
    assert t[88]["engine"] == "qwen_custom" and t[88]["speaker"] == "aiden"
    assert t[87]["instruct"] == t[88]["instruct"] == "Calm and measured."
    assert t[89]["engine"] == "orpheus" and t[89]["voice"] == "jess"


def test_the_table_refuses_a_wrong_anchor_count():
    with pytest.raises(SystemExit):
        s2c.build_speakers([_ref(i, "Male") for i in range(41)],
                           [_ref(i, "Female") for i in range(43)],
                           {"M4-s3": _ref(0, "Male"), "F6-s2": _ref(0, "Female")})


def test_the_lock_keeps_core_hum_free_anchors_in_plan_order():
    admitted = ["a", "b", "c", "d"]
    verdicts = {"a": {"zone": "core", "hum": "0"}, "b": {"zone": "niche", "hum": "0"},
                "c": {"zone": "core", "hum": "2"}, "d": {"zone": "core", "hum": ""}}
    assert s2c._admitted_passing(admitted, verdicts) == ["a", "d"]


def test_split_is_deterministic_disjoint_and_sized():
    ids = ["x%03d" % i for i in range(320)]
    train, val = s2c.split_rows(ids)
    assert len(val) == 10 and len(train) == 310
    assert set(train).isdisjoint(val) and set(train) | set(val) == set(ids)
    assert s2c.split_rows(ids) == (train, val)
    with pytest.raises(ValueError):
        s2c.split_rows(ids[:5])


def test_bank_lines_take_the_plan_seed_rule_and_the_engine_directions():
    t = _table()
    rows = [{"id": "2b-%04d" % k, "text": "Line %d." % k, "role": "primary",
             "speaker_line_index": k} for k in range(1, 4)]
    lines, _ = s2c.bank_lines(t[0], rows)
    assert [l["seed"] for l in lines] == [s2c.SEED_BASE + 1000 * 0 + k for k in (1, 2, 3)]
    assert all(l["direction"] == {"design": ""} for l in lines)   # build_direction, as 1c-ii records
    assert all("ref_file" in l for l in lines)
    lines_c, _ = s2c.bank_lines(t[87], rows)
    assert all(l["direction"].get("speaker") == "ryan" for l in lines_c)
    assert all("ref_file" not in l for l in lines_c)
    lines_o, _ = s2c.bank_lines(t[89], rows)
    assert all(l["direction"] == {"voice": "jess", "render_text": l["text"]} for l in lines_o)


def test_the_render_argv_is_the_dataset_bank_call_not_a_probe():
    t = _table()
    argv = s2c.render_argv("/data/model-training/datasets/q_synth_raw", t[89])
    assert argv[0] == r2.SYNTH_BANK and argv[-1] == "orpheus"
    assert argv[1].endswith("spk_89/bank.json") and argv[2].endswith("spk_89/audio")
    assert "SYNTH_BANK_PROBE" not in " ".join(argv)


def test_qc_passing_keeps_only_hard_pass_clips(tmp_path):
    spk = tmp_path / "spk_00"
    (spk / "audio").mkdir(parents=True)
    rows = [{"id": "a", "wav": "a.wav", "text": "A."}, {"id": "b", "wav": "b.wav", "text": "B."}]
    (spk / "audio" / "qwen_base_manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (spk / "audio" / "a.wav").write_bytes(b"x")
    (spk / "qc_measures.jsonl").write_text(
        json.dumps({"id": "a", "hard_pass": True}) + "\n"
        + json.dumps({"id": "b", "hard_pass": False}) + "\n", encoding="utf-8")
    kept = s2c.qc_passing(str(spk), "qwen_base")
    assert [k[0] for k in kept] == ["a"]


def test_build_corpus_refuses_any_speaker_under_the_qc_floor(tmp_path, capsys):
    t = _table()
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "speakers.json").write_text(json.dumps({"speakers": t}), encoding="utf-8")
    for s in t:
        d = raw / s["name"]
        (d / "audio").mkdir(parents=True)
        (d / "audio" / ("%s_manifest.jsonl" % s["engine"])).write_text("", encoding="utf-8")
        (d / "qc_measures.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        s2c.cmd_build_corpus(type("A", (), {"raw_root": str(raw), "out_dir": str(tmp_path / "o")})())
    err = capsys.readouterr().err
    assert "under 310" in err and "spk_00=0" in err


def test_resample_to_gives_22050_pcm16(tmp_path):
    import soundfile as sf
    src = tmp_path / "in.wav"
    x = 0.2 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000)
    sf.write(str(src), x.astype("float32"), 24000, "PCM_16")
    dst = tmp_path / "out" / "o.wav"
    clipped = s2c.resample_to(str(src), str(dst))
    info = sf.info(str(dst))
    assert info.samplerate == 22050 and info.subtype == "PCM_16"
    assert clipped == 0
    assert math.isclose(info.frames / info.samplerate, 1.0, rel_tol=0.01)


def test_a_non_integer_hum_is_a_refusal_naming_the_id(capsys):
    with pytest.raises(SystemExit):
        s2c._admitted_passing(["x1"], {"x1": {"zone": "core", "hum": "0.0"}})
    assert "x1" in capsys.readouterr().err


def test_a_passing_row_with_no_wav_is_a_refusal(tmp_path, capsys):
    spk = tmp_path / "spk_00"
    (spk / "audio").mkdir(parents=True)
    (spk / "audio" / "qwen_base_manifest.jsonl").write_text(
        json.dumps({"id": "a", "wav": "missing.wav", "text": "A."}) + "\n", encoding="utf-8")
    (spk / "qc_measures.jsonl").write_text(json.dumps({"id": "a", "hard_pass": True}) + "\n",
                                           encoding="utf-8")
    with pytest.raises(SystemExit):
        s2c.qc_passing(str(spk), "qwen_base")
    assert "missing.wav" in capsys.readouterr().err


def test_build_corpus_refuses_the_locked_five_table(tmp_path, capsys):
    five = [s for s in _table() if s["index"] in s2c.LOCKED_SPEAKERS]
    (tmp_path / "speakers_locked5.json").write_text(json.dumps({"speakers": five}), encoding="utf-8")
    with pytest.raises(SystemExit):
        s2c.cmd_build_corpus(type("A", (), {"raw_root": str(tmp_path), "out_dir": str(tmp_path / "o")})())
    assert "no speakers.json" in capsys.readouterr().err


def test_plan_refuses_a_text_bank_that_is_not_the_frozen_one(tmp_path, capsys):
    (tmp_path / "meta.json").write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    args = type("A", (), {"text_dir": str(tmp_path), "out_root": str(tmp_path), "speakers": None,
                          "include_reserve": False, "allow_dirty": True})()
    with pytest.raises(SystemExit):
        s2c.cmd_plan(args)
    assert "frozen bank" in capsys.readouterr().err


def test_the_bank_hash_matches_the_plan_and_a_change_is_detected():
    t = _table()
    rows = [{"id": "2b-0001", "text": "Line.", "role": "primary", "speaker_line_index": 1}]
    bank = {"version": s2c.VERSION, "campaign": "c", "lines": bank_lines_for(t[0], rows)}
    h = s2c._file_sha_of_json(bank)
    assert h == s2c._file_sha_of_json(json.loads(json.dumps(bank)))
    bank["lines"][0]["seed"] += 1
    assert h != s2c._file_sha_of_json(bank)


def bank_lines_for(speaker, rows):
    return s2c.bank_lines(speaker, rows)[0]


def test_locked_five_lock_writes_exactly_the_five_speakers(tmp_path, monkeypatch):
    anchors = tmp_path / "anchors"
    anchors.mkdir()
    (anchors / "clips").mkdir()
    locked = tmp_path / "locked"
    (locked / "clips").mkdir(parents=True)
    rows = [{"id": "M4-s3", "file": "clips/M4-s3.wav", "text": "M.", "gender": "Male"},
            {"id": "F6-s2", "file": "clips/F6-s2.wav", "text": "F.", "gender": "Female"}]
    (locked / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                           encoding="utf-8")
    (locked / "clips" / "M4-s3.wav").write_bytes(b"m")
    (locked / "clips" / "F6-s2.wav").write_bytes(b"f")
    out = tmp_path / "out"
    out.mkdir()
    args = type("A", (), {"anchors_dir": str(anchors), "locked_dir": str(locked),
                          "out_root": str(out), "locked_five_only": True})()
    s2c.cmd_lock_speakers(args)
    written = json.loads((out / "speakers_locked5.json").read_text())
    assert [s["index"] for s in written["speakers"]] == list(s2c.LOCKED_SPEAKERS)
    assert not (out / "speakers.json").exists()


def test_the_render_env_is_the_probe_flag_so_register_audition_is_skipped():
    import render_1ci as r1
    assert s2c.r1.PROBE_ENV == {"SYNTH_BANK_PROBE": "1"}
    assert r1.PROBE_ENV == {"SYNTH_BANK_PROBE": "1"}


def test_a_360_plan_grows_to_400_with_reserve_and_nothing_else(tmp_path):
    t = _table()
    rows = [{"id": "2b-%04d" % k, "text": "L%d." % k, "role": "primary",
             "speaker_line_index": k} for k in range(1, 361)]
    primary = s2c.bank_lines(t[0], rows)[0]
    cdir = tmp_path / "spk_00"
    cdir.mkdir()
    bank360 = {"version": "v", "campaign": "c", "lines": primary}
    (cdir / "bank.json").write_text(json.dumps(bank360), encoding="utf-8")
    reserve = [{"id": "2b-%04d" % k, "text": "R%d." % k, "role": "reserve",
                "speaker_line_index": k} for k in range(361, 401)]
    bank400 = {"version": "v", "campaign": "c",
               "lines": s2c.bank_lines(t[0], rows + reserve)[0]}
    s2c._refuse_existing(str(cdir), bank400, {"include_reserve": True})   # no SystemExit
    # Not a growth without the flag, and not a changed primary line:
    with pytest.raises(SystemExit):
        s2c._refuse_existing(str(cdir), bank400, {"include_reserve": False})
    changed = json.loads(json.dumps(bank400))
    changed["lines"][0]["seed"] += 1
    with pytest.raises(SystemExit):
        s2c._refuse_existing(str(cdir), changed, {"include_reserve": True})


def test_load_speakers_refuses_unknown_and_duplicate_indices(tmp_path, capsys):
    (tmp_path / "speakers_locked5.json").write_text(
        json.dumps({"speakers": [s for s in _table() if s["index"] in s2c.LOCKED_SPEAKERS]}),
        encoding="utf-8")
    assert [s["index"] for s in s2c.load_speakers(str(tmp_path), [0, 89])] == [0, 89]
    with pytest.raises(SystemExit):
        s2c.load_speakers(str(tmp_path), [2])
    assert "not in the loaded table" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        s2c.load_speakers(str(tmp_path), [0, 0])
