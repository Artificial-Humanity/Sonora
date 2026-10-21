"""Phase 1S Step 1c-0's ladder is the pre-registered one, its F0 measure reads the whole range or
says it cannot, its renderer imports in the qwen container, and a resume cannot mix two renders."""
import argparse
import ast
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("render_qwen_1c0",
                                               REPO / "scripts/tools/render_qwen_1c0.py")
rq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq)

# The pre-registration's text, copied here so a drift in either is a failure.
PREREG_M4 = ("gender: Male. pitch: Medium male pitch, stable. speed: Steady, unhurried. volume: "
             "Moderate conversational. age: Middle-aged adult. clarity: High clarity with distinct "
             "pronunciation. fluency: Highly fluent. accent: American English. texture: Clear and "
             "natural. emotion: Calm, even. tone: Even, narratorial. personality: Composed and "
             "unobtrusive.")
# § 1c-0's rung table, verbatim: rung | `pitch:` value | seeds (sentences 1-3).
PREREG_RUNGS = """\
| M1 | `pitch: Extremely low male pitch, stable.` | 9111–9113 |
| M2 | `pitch: Very low male pitch, stable.` | 9121–9123 |
| M3 | `pitch: Low male pitch, stable.` | 9131–9133 |
| M4 | `pitch: Medium male pitch, stable.` | 9141–9143 |
| M5 | `pitch: High male pitch, stable.` | 9151–9153 |
| M6 | `pitch: Very high male pitch, stable.` | 9161–9163 |
| M7 | `pitch: Extremely high male pitch, stable.` | 9171–9173 |
| F1 | `pitch: Extremely low female pitch, stable.` | 9211–9213 |
| F2 | `pitch: Very low female pitch, stable.` | 9221–9223 |
| F3 | `pitch: Low female pitch, stable.` | 9231–9233 |
| F4 | `pitch: Medium female pitch, stable.` | 9241–9243 |
| F5 | `pitch: High female pitch, stable.` | 9251–9253 |
| F6 | `pitch: Very high female pitch, stable.` | 9261–9263 |
| F7 | `pitch: Extremely high female pitch, stable.` | 9271–9273 |
"""
# § Niche personas' table, verbatim: persona | age band | gender | pitch | age | texture | seeds.
PREREG_PERSONAS = """\
| P1 | Child girl | Female | High child pitch, stable | Child, about eight years old | Light and clear | 9311–9313 |
| P2 | Young girl | Female | High, stable | Young teenager, about thirteen | Light and natural | 9321–9323 |
| P3 | Child boy | Male | High child pitch, stable | Child, about nine years old | Light and clear | 9331–9333 |
| P4 | Man (large) | Male | Very low male pitch, stable | Adult | Deep, full and chest-resonant | 9341–9343 |
| P5 | Elderly man | Male | Low male pitch, stable | Elderly, about seventy-five | Slightly hoarse, with light age strain | 9351–9353 |
| P6 | Elderly woman | Female | Low female pitch, stable | Elderly, about seventy-five | Slightly hoarse, with light age strain | 9361–9363 |
"""


def _keys(instruct):
    return dict(p.split(": ", 1) for p in instruct.rstrip(".").split(". "))


def _table(text):
    return [[c.strip() for c in line.strip("|").split("|")] for line in text.splitlines()]


def _seeds(cell):
    lo, hi = (int(s) for s in cell.split("–"))
    return list(range(lo, hi + 1))


def test_sixty_items_in_listening_order_with_the_registered_seeds():
    items = rq.ladder()
    assert len(items) == 60
    assert [i["id"] for i in items[:3]] == ["M1-s1", "M1-s2", "M1-s3"]
    assert [i["id"] for i in items[21:24]] == ["F1-s1", "F1-s2", "F1-s3"]
    assert [i["id"] for i in items[-3:]] == ["P6-s1", "P6-s2", "P6-s3"]
    assert len({i["id"] for i in items}) == 60 and len({i["seed"] for i in items}) == 60
    seeds = {i["id"]: i["seed"] for i in items}
    assert (seeds["M1-s1"], seeds["M4-s2"], seeds["F7-s3"], seeds["P3-s1"]) == (9111, 9142, 9273, 9331)


def test_the_middle_rung_is_the_registered_instruct():
    assert next(i for i in rq.ladder() if i["id"] == "M4-s1")["instruct"] == PREREG_M4


def test_every_rung_is_the_spec_tables_pitch_and_seeds():
    rows = _table(PREREG_RUNGS)
    assert len(rows) == 14 and sum(r[0].startswith("F") for r in rows) == 7
    by_id = {i["id"]: i for i in rq.ladder()}
    for rung, pitch, seeds in rows:
        got = [by_id["%s-s%d" % (rung, s)] for s in (1, 2, 3)]
        assert ["pitch: %s." % _keys(i["instruct"])["pitch"] for i in got] == [pitch.strip("`")] * 3
        assert [i["seed"] for i in got] == _seeds(seeds)


def test_every_persona_is_the_spec_tables_row():
    rows = _table(PREREG_PERSONAS)
    assert len(rows) == 6
    by_id = {i["id"]: i for i in rq.ladder()}
    for pid, band, gender, pitch, age, texture, seeds in rows:
        got = [by_id["%s-s%d" % (pid, s)] for s in (1, 2, 3)]
        for i in got:
            k = _keys(i["instruct"])
            assert (k["gender"], k["pitch"], k["age"], k["texture"]) == (gender, pitch, age, texture)
            # The taxonomy has no size band: "Man (large)" is band Man, size goes in the label.
            assert i["band"] == band.replace(" (large)", "")
            assert i["label"].startswith(pid + " ")
        assert [i["seed"] for i in got] == _seeds(seeds)
    assert by_id["P4-s1"]["label"] == "P4 large man"


def test_the_ladder_moves_pitch_and_nothing_else():
    for g in ("M", "F"):
        rungs = [_keys(i["instruct"]) for i in rq.ladder() if i["id"].startswith(g) and i["sentence"] == 1]
        assert len(rungs) == 7
        assert len({r["pitch"] for r in rungs}) == 7
        for r in rungs:
            assert {k: v for k, v in r.items() if k != "pitch"} == \
                {k: v for k, v in rungs[0].items() if k != "pitch"}
    pitches = [_keys(i["instruct"])["pitch"] for i in rq.ladder() if i["id"].startswith("M") and i["sentence"] == 1]
    assert pitches[0] == "Extremely low male pitch, stable" and pitches[-1] == "Extremely high male pitch, stable"


def test_personas_change_only_gender_pitch_age_texture():
    fixed = {k: v for k, v in rq.FIXED.items() if k not in ("age", "texture")}
    for i in rq.ladder():
        if i["group"] != "niche personas":
            continue
        k = _keys(i["instruct"])
        assert {key: k[key] for key in fixed} == fixed
    bands = [i["band"] for i in rq.ladder() if i["group"] == "niche personas" and i["sentence"] == 1]
    assert bands == ["Child girl", "Young girl", "Child boy", "Man", "Elderly man", "Elderly woman"]


def test_every_item_reads_the_same_three_sentences():
    for i in rq.ladder():
        assert i["text"] == rq.SENTENCES[i["sentence"] - 1]
    assert len(set(rq.SENTENCES)) == 3


def test_build_instruct_refuses_a_missing_or_unknown_key():
    keys = dict(rq.FIXED, gender="Male", pitch="Low male pitch, stable")
    assert rq.build_instruct(keys).startswith("gender: Male. pitch: Low male pitch, stable. ")
    with pytest.raises(ValueError):
        rq.build_instruct({k: v for k, v in keys.items() if k != "tone"})
    with pytest.raises(ValueError):
        rq.build_instruct(dict(keys, colour="blue"))


def test_render_refuses_any_qwen_tts_but_the_studied_release():
    assert rq.require_qwen_tts(lambda _pkg: "0.1.1") == "0.1.1"
    with pytest.raises(SystemExit, match="REFUSING"):
        rq.require_qwen_tts(lambda _pkg: "0.1.2")


# --- F0 -------------------------------------------------------------------------------------

SR = 24000


def _voice(f0, sr=SR, seconds=1.5):
    t = np.arange(int(sr * seconds)) / sr
    return 0.3 * sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3, 4)), sr


def _stack(f0, seconds=1.5, amps=None):
    """A periodic voice: harmonics at 1/k² up to 0.45 sr, or the given amplitudes."""
    t = np.arange(int(SR * seconds)) / SR
    amps = amps or [1.0 / k ** 2 for k in range(1, 400) if k * f0 < 0.45 * SR]
    return 0.3 * sum(a * np.sin(2 * np.pi * f0 * k * t) for k, a in enumerate(amps, 1)) / sum(amps)


@pytest.mark.parametrize("f0", [90.0, 220.0, 520.0])
def test_f0_profile_finds_a_voice_across_the_whole_band(f0):
    x, sr = _voice(f0)
    got = rq.f0_profile(x, sr)
    assert got is not None and abs(got["f0_median"] - f0) / f0 < 0.03, got
    assert got["f0_edge_frac"] == 0 and not got["f0_above_ceiling"], got
    assert got["f0_band"] == ("high" if f0 > 350 else "low")


@pytest.mark.parametrize("f0", [60.0, 70.0, 80.0, 400.0, 650.0])
def test_f0_profile_reads_a_low_or_high_harmonic_stack_unpinned(f0):
    """A flat 50-700 Hz search reads every stack here below 80 Hz as ~706 Hz (its 34-sample top
    lag); the two-pass measure must read each within 3% and off both search edges."""
    got = rq.f0_profile(_stack(f0), SR)
    assert abs(got["f0_median"] - f0) / f0 < 0.03, got
    assert got["f0_edge_frac"] == 0 and not got["f0_above_ceiling"], got


def test_a_flat_wide_search_pins_a_low_voice_to_its_top_edge():
    """Positive control for the two passes: one 50-700 Hz search reads 70 Hz as ~706 Hz."""
    from periodicity import frames
    f0, _h = frames(_stack(70.0), SR, 50.0, 700.0, 0.01, 0.3)
    assert abs(float(np.median(f0)) - SR / int(SR / 700.0)) < 1.0


def test_a_low_glide_reads_its_range_in_semitones():
    t = np.arange(int(SR * 1.5)) / SR
    phase = 2 * np.pi * np.cumsum(70.0 + 40.0 * t / 1.5) / SR
    x = 0.3 * sum(np.sin(k * phase) / k ** 2 for k in range(1, 90))
    got = rq.f0_profile(x, SR)
    assert got["f0_band"] == "low" and got["f0_edge_frac"] == 0, got
    assert abs(got["f0_median"] - 90.0) / 90.0 < 0.03, got
    assert 74.0 < got["f0_p10"] < 78.0 and 102.0 < got["f0_p90"] < 108.0, got
    assert 5.0 < got["f0_excursion_st"] < 7.0, got
    assert got["f0_excursion"] == pytest.approx(got["f0_p90"] - got["f0_p10"])


def test_the_repo_default_band_would_miss_a_child_voice():
    """Positive control for the high band: at measure_speaker_hnr's 350 Hz ceiling, a 520 Hz voice
    is not measured as 520 Hz, which is why this probe re-measures up to 700 Hz."""
    from periodicity import frames
    x, sr = _voice(520.0)
    f0, _h = frames(np.asarray(x, dtype=np.float64), sr, 55.0, 350.0, 0.01, 0.3)
    assert not len(f0) or abs(float(np.median(f0)) - 520.0) / 520.0 > 0.1


def test_a_dominant_second_harmonic_is_not_read_an_octave_up():
    """The high band alone reads this 220 Hz voice at ~440 Hz (positive control); its HNR there is
    within `HNR_MARGIN_DB` of the low band's, so the low band's 220 Hz stands."""
    x = _stack(220.0, amps=[0.15, 1.0, 0.1, 0.5, 0.05, 0.2])
    high = rq._read_band(x, SR, rq.HIGH_BAND)
    assert abs(float(np.median(high["f0"])) - 440.0) / 440.0 < 0.03
    got = rq.f0_profile(x, SR)
    assert got["f0_band"] == "low" and abs(got["f0_median"] - 220.0) / 220.0 < 0.03, got


@pytest.mark.parametrize("f0", [720.0, 800.0, 1000.0])
def test_a_voice_above_the_ceiling_is_flagged(f0):
    """Just above 700 Hz the high band pins to its top edge; further up it locks to a subharmonic
    off the edges and only the ceiling probe sees it. Either way the row must warn."""
    got = rq.f0_profile(_stack(f0), SR)
    assert got["f0_edge_frac"] > 0.5 or got["f0_above_ceiling"], got
    if f0 >= 800:
        assert got["f0_above_ceiling"], got


def test_hnr_rises_with_f0_for_a_perfectly_periodic_voice():
    """Why each row carries `hnr_note`: the same perfect periodicity reads >5 dB apart."""
    low, high = rq.f0_profile(_stack(70.0), SR), rq.f0_profile(_stack(650.0), SR)
    assert high["hnr"] - low["hnr"] > 5.0, (low["hnr"], high["hnr"])


def test_f0_profile_is_none_on_silence():
    assert rq.f0_profile(np.zeros(24000), 24000) is None


# --- render's imports -----------------------------------------------------------------------

LIB = REPO / "scripts" / "lib"
HEAVY = ("matcha", "torch", "hydra", "lightning", "monotonic_align")


def _heavy_after(code):
    """Modules from HEAVY in sys.modules after `code` runs in a fresh interpreter."""
    probe = ("import sys; sys.path[:0] = [%r, %r]\n%s\n"
             "print(sorted(m for m in %r if m in sys.modules))" % (str(REPO), str(LIB), code, HEAVY))
    got = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout.strip().replace("'", '"'))


def test_render_imports_only_dependency_light_lib_modules():
    tree = ast.parse((REPO / "scripts/tools/render_qwen_1c0.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cmd_render")
    names = {a.name.split(".")[0] for n in ast.walk(fn) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(fn) if isinstance(n, ast.ImportFrom)}
    assert {m for m in names if (LIB / (m + ".py")).exists()} == {"loudness", "synth_common"}
    assert _heavy_after("import loudness, synth_common") == []
    assert _heavy_after("import matcha.delivery") == ["matcha"]     # the probe can see one


def test_ear_bench_re_exports_the_one_match_loudness():
    """Both import spellings of ear_bench hand back loudness.match_loudness itself. Its torch and
    Matcha imports are stubbed: this checks the re-export, not the bench."""
    stubs = ("import types\n"
             "for name, attrs in {'torch': {'device': lambda *a: None},"
             " 'matcha.cli': dict.fromkeys(['detect_lane', 'load_matcha', 'load_vocoder_24k',"
             " 'process_text_for_lane', 'to_waveform']), 'matcha.text': {'text_to_sequence': None},"
             " 'matcha.utils': {}, 'matcha.utils.utils': {'intersperse': None}}.items():\n"
             "    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m\n"
             "sys.path.insert(0, %r)\n"
             "import ear_bench, lib.ear_bench, loudness, lib.loudness\n"
             "assert ear_bench.match_loudness is loudness.match_loudness\n"
             "assert lib.ear_bench.match_loudness is lib.loudness.match_loudness\n"
             "assert 'def match_loudness' not in open(ear_bench.__file__).read()\n"
             % str(REPO / "scripts"))
    _heavy_after(stubs)


# --- render resume --------------------------------------------------------------------------

def _model(tmp_path):
    d = tmp_path / "model"
    (d / "speech_tokenizer" / "sub").mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"weights")
    (d / "speech_tokenizer" / "config.json").write_text("{}")
    (d / "speech_tokenizer" / "sub" / "model.bin").write_bytes(b"tok")
    return d


def _rendered(out, commit, weights, items=None, wav=lambda it: b"RIFF"):
    """An --out a render left: meta, a manifest row per item and its (placeholder) wav."""
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "render_meta.json").write_text(json.dumps(
        {"commit": commit, "weights_sha256": weights, "qwen_tts": "0.1.1"}))
    rows = []
    for it in items if items is not None else rq.ladder():
        (out / "clips" / (it["id"] + ".wav")).write_bytes(wav(it))
        rows.append(dict(it, file="clips/%s.wav" % it["id"], sr=SR, commit=commit))
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


@pytest.fixture
def render_env(tmp_path, monkeypatch):
    """cmd_render against a fake model dir. torch and qwen_tts are made unimportable, so a render
    that got as far as loading the model fails loudly instead of loading one."""
    model = _model(tmp_path)
    monkeypatch.setattr(rq, "MODEL_DIR", str(model))
    monkeypatch.setattr(rq, "require_qwen_tts", lambda _v: "0.1.1")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "qwen_tts", None)
    return tmp_path / "out", rq.weights_sha256(str(model)), model


def _render(out, commit="abc123"):
    rq.cmd_render(argparse.Namespace(out=str(out), commit=commit))


def test_weights_hash_covers_every_speech_tokenizer_file(tmp_path):
    w = rq.weights_sha256(str(_model(tmp_path)))
    assert sorted(w) == ["model.safetensors", "speech_tokenizer/config.json",
                         "speech_tokenizer/sub/model.bin"]
    empty = tmp_path / "empty"
    (empty / "speech_tokenizer").mkdir(parents=True)
    (empty / "model.safetensors").write_bytes(b"w")
    with pytest.raises(SystemExit, match="REFUSING"):
        rq.weights_sha256(str(empty))


def test_a_fresh_render_writes_its_meta_before_loading_the_model(render_env):
    out, weights, _model_dir = render_env
    with pytest.raises(ImportError):
        _render(out)
    meta = json.loads((out / "render_meta.json").read_text())
    assert (meta["commit"], meta["weights_sha256"], meta["qwen_tts"]) == ("abc123", weights, "0.1.1")
    assert (meta["batch_size"], meta["items"], meta["lufs"]) == (1, 60, rq.LUFS)


def test_a_finished_render_exits_before_loading_the_model(render_env, capsys):
    out, weights, _model_dir = render_env
    _rendered(out, "abc123", weights)
    _render(out)
    assert "the model was not loaded" in capsys.readouterr().out


@pytest.mark.parametrize("change, match", [
    (lambda r: r.update(instruct=r["instruct"].replace("Calm", "Bright")), "instruct"),
    (lambda r: r.update(seed=r["seed"] + 1), "seed"),
    (lambda r: r.update(text="Another sentence."), "text"),
    (lambda r: r.update(commit="0ld"), "commit"),
])
def test_resume_refuses_a_done_row_that_is_not_its_ladder_item(render_env, change, match):
    out, weights, _model_dir = render_env
    rows = _rendered(out, "abc123", weights, items=rq.ladder()[:10])
    change(rows[4])
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="REFUSING: manifest row %s .*%s" % (rows[4]["id"], match)):
        _render(out)


def test_resume_refuses_a_missing_wav_or_a_repeated_id(render_env):
    out, weights, _model_dir = render_env
    rows = _rendered(out, "abc123", weights, items=rq.ladder()[:10])
    (out / "clips" / "M1-s2.wav").unlink()
    with pytest.raises(SystemExit, match="REFUSING: manifest row M1-s2 has no wav"):
        _render(out)
    _rendered(out, "abc123", weights, items=rq.ladder()[:10])
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + rows[:1]))
    with pytest.raises(SystemExit, match="REFUSING: manifest id 'M1-s1' is repeated"):
        _render(out)


@pytest.mark.parametrize("what", ["commit", "weights_sha256", "qwen_tts"])
def test_resume_refuses_another_renders_meta(render_env, what):
    out, weights, model = render_env
    _rendered(out, "abc123", weights, items=rq.ladder()[:10])
    commit = "abc123"
    if what == "commit":
        commit = "def456"
    elif what == "weights_sha256":
        (model / "speech_tokenizer" / "sub" / "model.bin").write_bytes(b"retrained")
    else:
        meta = json.loads((out / "render_meta.json").read_text())
        (out / "render_meta.json").write_text(json.dumps(dict(meta, qwen_tts="0.1.2")))
    with pytest.raises(SystemExit, match="REFUSING: resuming .*%s" % what):
        _render(out, commit=commit)


def test_resume_drops_only_a_truncated_final_line(tmp_path, capsys):
    out = tmp_path / "out"
    rows = _rendered(out, "abc123", {"w": "x"}, items=rq.ladder()[:5])
    meta = {"commit": "abc123", "weights_sha256": {"w": "x"}, "qwen_tts": "0.1.1"}
    whole = (out / "manifest.jsonl").read_text()
    (out / "manifest.jsonl").write_text(whole + json.dumps(rows[0])[:40])
    assert rq.resume_state(str(out), meta) == {r["id"] for r in rows}
    assert "truncated final manifest line" in capsys.readouterr().out
    assert (out / "manifest.jsonl").read_text() == whole
    (out / "manifest.jsonl").write_text(whole[:30] + "\n" + whole)
    with pytest.raises(SystemExit, match="REFUSING: .* line 1 is not JSON"):
        rq.resume_state(str(out), meta)


# --- measure --------------------------------------------------------------------------------

def _clip_f0(it):
    return 520.0 if it["id"].startswith("P1") else 210.0 if it["id"].startswith("F") else 110.0


def _measure_dir(tmp_path):
    import soundfile as sf
    out = tmp_path / "probe"
    _rendered(out, "abc123", {"w": "x"})
    for it in rq.ladder():
        sf.write(str(out / "clips" / (it["id"] + ".wav")), _stack(_clip_f0(it), 1.0).astype("float32"), SR)
    return out


def test_measure_writes_a_row_per_ladder_clip(tmp_path, capsys):
    out = _measure_dir(tmp_path)
    rq.cmd_measure(argparse.Namespace(out=str(out)))
    rows = [json.loads(line) for line in (out / "measures.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [it["id"] for it in rq.ladder()]
    by_id = {r["id"]: r for r in rows}
    for it in rq.ladder():
        r, f0 = by_id[it["id"]], _clip_f0(it)
        assert abs(r["f0_median"] - f0) / f0 < 0.03, r
        assert r["hnr_note"] == rq.HNR_NOTE
        if f0 > 400:
            assert r["f0_band"] == "high" and r["cpp"] is None and r["h1h2"] is None, r
            assert "cepstral band" in r["phonation_null_reason"], r
        else:
            assert r["f0_band"] == "low" and r["cpp"] is not None and r["phonation_null_reason"] is None, r
        assert r["phonation_f0_band_hz"] == [60.0, 400.0]
    assert "60 clips measured; 0 without a voiced frame; 3 without a phonation measure" \
        in capsys.readouterr().out


def test_measure_refuses_a_missing_manifest_a_wrong_id_set_or_a_missing_wav(tmp_path):
    with pytest.raises(SystemExit, match="REFUSING: no manifest"):
        rq.cmd_measure(argparse.Namespace(out=str(tmp_path / "nothing")))
    out = _measure_dir(tmp_path)
    manifest = out / "manifest.jsonl"
    lines = manifest.read_text().splitlines(keepends=True)
    manifest.write_text("".join(lines[:-1]))
    with pytest.raises(SystemExit, match=r"REFUSING: 59 of 60 .*P6-s3"):
        rq.cmd_measure(argparse.Namespace(out=str(out)))
    manifest.write_text("".join(lines[:-1] + lines[:1]))
    with pytest.raises(SystemExit, match="REFUSING: manifest id 'M1-s1' is repeated"):
        rq.cmd_measure(argparse.Namespace(out=str(out)))
    stranger = dict(json.loads(lines[-1]), id="Q9-s1", file="clips/Q9-s1.wav")
    manifest.write_text("".join(lines[:-1]) + json.dumps(stranger) + "\n")
    with pytest.raises(SystemExit, match="REFUSING: manifest id 'Q9-s1' is not a ladder id"):
        rq.cmd_measure(argparse.Namespace(out=str(out)))
    manifest.write_text("".join(lines))
    (out / "clips" / "F3-s2.wav").unlink()
    with pytest.raises(SystemExit, match="REFUSING: manifest row F3-s2 has no wav"):
        rq.cmd_measure(argparse.Namespace(out=str(out)))
