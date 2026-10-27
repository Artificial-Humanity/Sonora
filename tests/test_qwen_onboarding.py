"""Step 1c-ii onboarding: synth_qwen_base.py and synth_qwen_custom.py.

Both renderers load a Qwen checkpoint through qwen_load and need a GPU, so these
tests import them with torch and qwen_load replaced by stubs that record every
call, and run `main()` on a small bank (the test_renderer_pins_and_timing
pattern). What is checked is each renderer's own logic: the clone prompt is
built once per reference and the seed lands after the cache lookup, a casting
failure loses one clip rather than the bank, an unlisted CustomVoice speaker is
skipped before any generate call, and the manifest rows carry the licence,
tokenizer dtype and timing fields the measures read.

The RELAY and channel-veto checks cover the two wiring traps the onboarding
design named: prefix matching in the audition app (both siblings start with
"qwen") and capability vetoes for engines that hold no production share yet.
"""
import ast
import hashlib
import importlib
import json
import pathlib
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from scripts_layout import SCRIPTS

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS.on_path()

import ref_select  # noqa: E402

SR = 24000


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


@pytest.fixture
def calls():
    return []


@pytest.fixture
def fake_torch(monkeypatch, calls):
    torch = _module("torch",
                    manual_seed=lambda s: calls.append(("seed", s)),
                    cuda=types.SimpleNamespace(synchronize=lambda: calls.append(("sync",))))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _fresh(name):
    """A fresh import of the renderer against the stubs; dropped again after the test."""
    sys.modules.pop(name, None)
    try:
        yield importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def qwen_base(monkeypatch, fake_torch, calls):
    """synth_qwen_base imported against a stub Base model that records its calls."""

    class Model:
        def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
            calls.append(("prompt", ref_audio, ref_text, x_vector_only_mode))
            return ("prompt-for", ref_audio)

        def generate_voice_clone(self, text, language, voice_clone_prompt):
            calls.append(("clone", text, language, voice_clone_prompt))
            return [np.zeros(2 * SR, dtype=np.float32)], SR     # 2.0 s

    def load_qwen(model_dir):
        calls.append(("load", model_dir))
        return Model()

    monkeypatch.setitem(sys.modules, "qwen_load",
                        _module("qwen_load", TOKENIZER_DTYPE="float32", load_qwen=load_qwen))
    yield from _fresh("synth_qwen_base")


@pytest.fixture
def qwen_custom(monkeypatch, fake_torch, calls):
    """synth_qwen_custom imported against a stub CustomVoice model."""

    class Model:
        def generate_custom_voice(self, text, speaker, language, instruct):
            calls.append(("custom", text, speaker, language, instruct))
            return [np.zeros(3 * SR, dtype=np.float32)], SR     # 3.0 s

    def load_qwen(model_dir):
        calls.append(("load", model_dir))
        return Model()

    monkeypatch.setitem(sys.modules, "qwen_load",
                        _module("qwen_load", TOKENIZER_DTYPE="float32", load_qwen=load_qwen))
    yield from _fresh("synth_qwen_custom")


@pytest.fixture
def anchor(tmp_path, monkeypatch):
    """A real wav under a REF_FILE_ROOT pointed at tmp_path, and its ref_file spec."""
    monkeypatch.setattr(ref_select, "REF_FILE_ROOT", str(tmp_path) + "/")
    path = tmp_path / "anchors" / "M4-s3.wav"
    path.parent.mkdir()
    sf.write(str(path), np.zeros(SR * 6, dtype=np.float32), SR)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": sha, "text": "The anchor reads this.",
            "id": "M4-s3", "gender": "Male"}


def _clock(monkeypatch, mod, *ticks):
    it = iter(ticks)
    monkeypatch.setattr(mod, "time", types.SimpleNamespace(monotonic=lambda: next(it)))


def _run(mod, monkeypatch, tmp_path, lines):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"version": "t1", "campaign": "t", "lines": lines}))
    out = tmp_path / "audio"
    monkeypatch.setattr(sys, "argv", [mod.__file__, "--bank", str(bank), "--out", str(out)])
    mod.main()
    engine = lines[0]["engine"]
    manifest = out / ("%s_manifest.jsonl" % engine)
    rows = [json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
    return out, rows


def _base_line(**kw):
    line = {"id": "b1", "engine": "qwen_base", "seed": 1000001,
            "text": "This is a narration line long enough to render.",
            "direction": {"design": "middle-aged man, gravelled"}}
    line.update(kw)
    return line


def _custom_line(**kw):
    line = {"id": "c1", "engine": "qwen_custom", "seed": 1000001,
            "text": "This is a narration line long enough to render.",
            "direction": {"speaker": "ryan", "instruct": "Calm and measured."}}
    line.update(kw)
    return line


# --- qwen_base: prompt caching, seeding order, provenance ----------------------

def test_the_clone_prompt_is_built_once_per_reference_and_seeded_after_the_cache(
        qwen_base, anchor, calls, monkeypatch, tmp_path):
    # clip 1 (cache miss): t0, ref_cond end, t1, render end. clip 2 (hit): t1, end.
    _clock(monkeypatch, qwen_base, 10.0, 12.0, 12.0, 15.0, 20.0, 24.0)
    lines = [_base_line(ref_file=anchor),
             _base_line(id="b2", seed=1000002, ref_file=anchor)]
    out, rows = _run(qwen_base, monkeypatch, tmp_path, lines)
    assert (out / "b1.wav").exists() and (out / "b2.wav").exists()
    assert [c[0] for c in calls] == ["load", "prompt", "sync", "seed", "clone",
                                     "sync", "seed", "clone", "sync"], calls
    prompt = ("prompt-for", anchor["path"])
    assert ("prompt", anchor["path"], anchor["text"], False) in calls
    assert calls[4] == ("clone", lines[0]["text"], "English", prompt)
    assert calls[7] == ("clone", lines[1]["text"], "English", prompt)
    assert (rows[0]["prompt_cached"], rows[1]["prompt_cached"]) == (False, True)
    assert rows[0]["ref_condition_seconds"] == 2.0
    assert rows[1]["ref_condition_seconds"] == 0.0
    assert rows[0]["render_seconds"] == 3.0 and rows[1]["render_seconds"] == 4.0
    for row in rows:
        assert row["engine_license"] == "Apache-2.0 (Qwen3-TTS-12Hz-1.7B-Base)"
        assert row["speech_tokenizer_dtype"] == "float32"
        assert row["ref"]["path"] == anchor["path"]
        assert row["ref"]["sha256"] == anchor["sha256"]
        assert row["intended_gender"] == "Male"


def test_a_base_casting_failure_loses_one_clip_not_the_bank(
        qwen_base, anchor, calls, monkeypatch, tmp_path, capsys):
    _clock(monkeypatch, qwen_base, 0.0, 1.0, 1.0, 2.0)
    lines = [_base_line(ref_file=dict(anchor, sha256="0" * 64)),
             _base_line(id="b2", seed=1000002, ref_file=anchor)]
    _out, rows = _run(qwen_base, monkeypatch, tmp_path, lines)
    assert [r["id"] for r in rows] == ["b2"]
    assert "CASTING FAILED" in capsys.readouterr().out
    assert [c[0] for c in calls].count("clone") == 1


def test_no_qwen_base_jobs_spends_no_model_load(qwen_base, calls, monkeypatch, tmp_path):
    line = _base_line(engine="qwen")
    _run(qwen_base, monkeypatch, tmp_path, [line])
    assert calls == [], calls


# --- qwen_custom: speaker guard, canonical case, provenance -------------------

def test_custom_voice_renders_with_the_canonical_speaker(
        qwen_custom, calls, monkeypatch, tmp_path):
    _clock(monkeypatch, qwen_custom, 5.0, 9.0, 20.0, 26.0)
    lines = [_custom_line(direction={"speaker": "Ryan", "instruct": "Calm."}),
             _custom_line(id="c2", seed=1000002,
                          direction={"speaker": "vivian", "instruct": ""})]
    out, rows = _run(qwen_custom, monkeypatch, tmp_path, lines)
    assert (out / "c1.wav").exists() and (out / "c2.wav").exists()
    assert [c[0] for c in calls] == ["load", "seed", "custom", "sync",
                                     "seed", "custom", "sync"], calls
    assert calls[2] == ("custom", lines[0]["text"], "ryan", "English", "Calm.")
    # an empty instruct reaches the shipped API as None (its own "no instruction")
    assert calls[5] == ("custom", lines[1]["text"], "vivian", "English", None)
    assert rows[0]["applied_speaker"] == "ryan"
    assert (rows[0]["intended_gender"], rows[1]["intended_gender"]) == ("Male", "Female")
    assert rows[0]["render_seconds"] == 4.0 and rows[1]["render_seconds"] == 6.0
    assert rows[0]["ref_condition_seconds"] == 0.0
    assert rows[0]["engine_license"] == "Apache-2.0 (Qwen3-TTS-12Hz-1.7B-CustomVoice)"
    assert rows[0]["speech_tokenizer_dtype"] == "float32"


def test_an_unlisted_custom_speaker_is_skipped_before_any_generate_call(
        qwen_custom, calls, monkeypatch, tmp_path, capsys):
    _clock(monkeypatch, qwen_custom, 0.0, 1.0)
    lines = [_custom_line(direction={"speaker": "julia", "instruct": "Calm."}),
             _custom_line(id="c2", seed=1000002)]
    _out, rows = _run(qwen_custom, monkeypatch, tmp_path, lines)
    assert [r["id"] for r in rows] == ["c2"]
    assert "SKIP (unknown speaker" in capsys.readouterr().out
    assert [c[0] for c in calls].count("custom") == 1


def test_no_qwen_custom_jobs_spends_no_model_load(qwen_custom, calls, monkeypatch,
                                                   tmp_path):
    _run(qwen_custom, monkeypatch, tmp_path, [_custom_line(engine="qwen")])
    assert calls == [], calls


# --- wiring traps: RELAY prefix order and the capability veto -----------------

def _relay():
    src = (REPO / "audition" / "app" / "main.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "RELAY":
            return ast.literal_eval(node.value)
    raise AssertionError("no RELAY assignment in audition/app/main.py")


def test_the_qwen_siblings_resolve_to_their_own_relay_rows_not_qwens():
    """_annotate_relay returns the FIRST key that prefixes the engine name, so
    table order decides: below "qwen", both siblings would inherit
    VoiceDesign's row — instruct "yes" for a clone with no instruct slot,
    scoring it for direction it never received."""
    relay = _relay()
    keys = list(relay)
    assert keys.index("qwen_base") < keys.index("qwen")
    assert keys.index("qwen_custom") < keys.index("qwen")

    def resolved(engine):
        for key, val in relay.items():
            if engine.startswith(key):
                return key, val
        return None, None

    key, row = resolved("qwen_base")
    assert key == "qwen_base" and row["instruct"] == "no" and row["design"] == "casting"
    key, row = resolved("qwen_custom")
    assert key == "qwen_custom" and row["instruct"] == "yes"
    key, _row = resolved("qwen")
    assert key == "qwen"


def test_the_channel_veto_separates_the_siblings():
    """The onboarding ruling's two examples, as executable checks: the clone is
    vetoed where a delivery must be described, the preset engine where a
    reference identity is required."""
    kept, dropped = ref_select.route_engines_for({"prose"}, ["qwen_base", "qwen_custom"])
    assert kept == ["qwen_custom"] and [e for e, _ in dropped] == ["qwen_base"]
    kept, dropped = ref_select.route_engines_for({"reference"}, ["qwen_base", "qwen_custom"])
    assert kept == ["qwen_base"] and [e for e, _ in dropped] == ["qwen_custom"]
