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
