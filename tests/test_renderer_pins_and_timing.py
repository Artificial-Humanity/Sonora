"""synth_chatterbox.py and synth_zonos.py: the `ref_file` pin and the throughput fields.

Both renderers import their engine at module scope and need a GPU, so these tests import them with
torch, the engine packages and (for chatterbox) perth replaced by stubs that record every call, and
run `main()` on a small bank. What is checked is the renderer's own logic: which reference it uses,
that a `ref_file` is sha256-checked before any engine call, the order of seed, conditioning and
generate, and the timing fields in each manifest row. Existing pins (`ref_id`) and cast lines are run
too, to show they reach the engine with the same arguments as before.
"""
import hashlib
import importlib
import json
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from scripts_layout import SCRIPTS

SCRIPTS.on_path()

import ref_select  # noqa: E402

SR_CBX, SR_ZON = 24000, 44100


class FakeTensor:
    """Enough of a torch tensor for the renderers' `.squeeze().float().cpu().numpy()` chains."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def squeeze(self, *_a):
        return FakeTensor(self.arr.squeeze())

    def unsqueeze(self, *_a):
        return self

    def float(self):
        return self

    def bfloat16(self):
        return self

    def cpu(self):
        return self

    def to(self, *_a, **_k):
        return self

    def numpy(self):
        return self.arr


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
                    from_numpy=lambda a: FakeTensor(a),
                    cuda=types.SimpleNamespace(synchronize=lambda: calls.append(("sync",))))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _fresh(name):
    """A fresh import of the renderer against the stubs; dropped again after the test, so no
    later test imports a module bound to them."""
    sys.modules.pop(name, None)
    try:
        yield importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def chatterbox(monkeypatch, fake_torch, calls):
    """synth_chatterbox imported against a stub ChatterboxTTS that records its calls."""

    class Model:
        sr = SR_CBX

        def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
            calls.append(("prepare", wav_fpath, exaggeration))

        def generate(self, text, **kw):
            calls.append(("generate", text, kw))
            return FakeTensor(np.zeros((1, 2 * SR_CBX)))      # 2.0 s

    class ChatterboxTTS:
        @classmethod
        def from_local(cls, ckpt_dir, device):
            calls.append(("load", ckpt_dir))
            return Model()

    monkeypatch.setitem(sys.modules, "perth", _module("perth", PerthImplicitWatermarker=object))
    monkeypatch.setitem(sys.modules, "chatterbox", _module("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.tts", _module("chatterbox.tts",
                                                               ChatterboxTTS=ChatterboxTTS))
    yield from _fresh("synth_chatterbox")


@pytest.fixture
def zonos(monkeypatch, fake_torch, calls):
    """synth_zonos imported against stub Zonos modules that record their calls."""

    class Autoencoder:
        sampling_rate = SR_ZON

        def decode(self, codes):
            calls.append(("decode", codes))
            return FakeTensor(np.zeros(SR_ZON * 3))             # 3.0 s

    class Model:
        autoencoder = Autoencoder()
        device = "cpu"

        def prepare_conditioning(self, cond):
            calls.append(("prepare", cond))
            return ("prepared", cond["text"])

        def generate(self, conditioning):
            calls.append(("generate", conditioning))
            return "codes"

    class Zonos:
        @classmethod
        def from_local(cls, config, weights, device):
            calls.append(("load", config))
            return Model()

    class SpeakerEmbeddingLDA:
        def __init__(self, device):
            pass

        def __call__(self, wav, sr):
            calls.append(("embed", sr))
            return None, FakeTensor(np.zeros(128))

    def make_cond_dict(**kw):
        calls.append(("cond", kw))
        return kw

    monkeypatch.setitem(sys.modules, "zonos", _module("zonos"))
    monkeypatch.setitem(sys.modules, "zonos.model", _module("zonos.model", Zonos=Zonos))
    monkeypatch.setitem(sys.modules, "zonos.conditioning",
                        _module("zonos.conditioning", make_cond_dict=make_cond_dict))
    monkeypatch.setitem(sys.modules, "zonos.speaker_cloning",
                        _module("zonos.speaker_cloning", SpeakerEmbeddingLDA=SpeakerEmbeddingLDA))
    yield from _fresh("synth_zonos")


@pytest.fixture
def anchor(tmp_path, monkeypatch):
    """A real wav under a REF_FILE_ROOT pointed at tmp_path, and its ref_file spec."""
    monkeypatch.setattr(ref_select, "REF_FILE_ROOT", str(tmp_path) + "/")
    path = tmp_path / "anchors" / "M4-s3.wav"
    path.parent.mkdir()
    sf.write(str(path), np.zeros(SR_CBX * 6, dtype=np.float32), SR_CBX)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": sha, "text": "The anchor reads this.", "id": "M4-s3",
            "gender": "Male"}


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


def _cbx_line(**kw):
    line = {"id": "c1", "engine": "chatterbox", "seed": 1000001,
            "text": "This is a narration line long enough to render.",
            "direction": {"design": "", "exaggeration": 0.4, "cfg_weight": 0.5}}
    line.update(kw)
    return line


def _zon_line(**kw):
    line = {"id": "z1", "engine": "zonos", "seed": 1000001,
            "text": "This is a narration line long enough to render.",
            "direction": {"design": "", "emotion": None, "pitch_std": 45.0,
                          "speaking_rate": 15.0}}
    line.update(kw)
    return line


# --- the ref_file pin ---------------------------------------------------------------------

def test_chatterbox_renders_a_ref_file_pin_and_records_its_path_and_sha(
        chatterbox, anchor, calls, monkeypatch, tmp_path):
    _clock(monkeypatch, chatterbox, 10.0, 10.5, 13.0)
    out, (row,) = _run(chatterbox, monkeypatch, tmp_path, [_cbx_line(ref_file=anchor)])
    assert (out / "c1.wav").exists()
    assert ("prepare", anchor["path"], 0.4) in calls
    assert row["ref"]["path"] == anchor["path"] and row["ref"]["sha256"] == anchor["sha256"]
    assert row["ref"]["pinned_by"] == "file" and row["ref"]["ref_text"] == anchor["text"]
    assert row["intended_gender"] == "Male"
    # pinned: never damped, the direction's exaggeration is what ran
    assert row["applied_exaggeration"] == 0.4 and row["bright_ref_damped"] is False


def test_zonos_renders_a_ref_file_pin_and_records_its_path_and_sha(
        zonos, anchor, calls, monkeypatch, tmp_path):
    _clock(monkeypatch, zonos, 1.0, 2.0, 7.0)
    out, (row,) = _run(zonos, monkeypatch, tmp_path, [_zon_line(ref_file=anchor)])
    assert (out / "z1.wav").exists()
    assert ("embed", SR_CBX) in calls, "the speaker embedding was not taken from the anchor"
    assert row["ref"]["path"] == anchor["path"] and row["ref"]["sha256"] == anchor["sha256"]
    assert row["intended_gender"] == "Male"


@pytest.mark.parametrize("which", ["chatterbox", "zonos"])
def test_a_ref_file_whose_sha_differs_is_refused_before_any_engine_call(
        which, request, anchor, calls, monkeypatch, tmp_path, capsys):
    mod = request.getfixturevalue(which)
    line = (_cbx_line if which == "chatterbox" else _zon_line)(ref_file=dict(anchor,
                                                                             sha256="0" * 64))
    out, rows = _run(mod, monkeypatch, tmp_path, [line])
    assert rows == [] and not list(out.glob("*.wav"))
    assert "CASTING FAILED" in capsys.readouterr().out
    assert [c[0] for c in calls] == ["load"], calls


@pytest.mark.parametrize("which", ["chatterbox", "zonos"])
def test_a_missing_or_relative_ref_file_is_refused(which, request, anchor, calls, monkeypatch,
                                                   tmp_path, capsys):
    mod = request.getfixturevalue(which)
    make = _cbx_line if which == "chatterbox" else _zon_line
    lines = [make(id="a", ref_file=dict(anchor, path=anchor["path"] + ".gone")),
             make(id="b", ref_file=dict(anchor, path="anchors/M4-s3.wav"))]
    _out, rows = _run(mod, monkeypatch, tmp_path, lines)
    assert rows == []
    assert capsys.readouterr().out.count("CASTING FAILED") == 2


def test_pinned_reference_file_checks_the_sha_on_every_call(anchor, tmp_path):
    job = {"ref_file": anchor}
    assert ref_select.pinned_reference_file(job)[2]["sha256"] == anchor["sha256"]
    with open(anchor["path"], "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(ValueError, match="has sha256"):
        ref_select.pinned_reference_file(job)
    with pytest.raises(ValueError, match="both ref_id"):
        ref_select.pinned_reference_file(dict(job, ref_id="pool_1"))


# --- the throughput fields ----------------------------------------------------------------

def test_chatterbox_times_conditioning_and_generate_separately(chatterbox, anchor, calls,
                                                               monkeypatch, tmp_path):
    _clock(monkeypatch, chatterbox, 10.0, 10.5, 13.0)
    _out, (row,) = _run(chatterbox, monkeypatch, tmp_path, [_cbx_line(ref_file=anchor)])
    assert row["ref_condition_seconds"] == 0.5
    assert row["render_seconds"] == 2.5
    assert row["audio_seconds"] == 2.0
    assert row["rtf"] == 1.25
    # seed, then conditioning, then generate WITHOUT audio_prompt_path: the order
    # generate(audio_prompt_path=...) ran them in itself, with the same arguments
    kinds = [c[0] for c in calls]
    assert kinds == ["load", "seed", "prepare", "sync", "generate"]
    gen = calls[kinds.index("generate")]
    assert gen[2] == {"exaggeration": 0.4, "cfg_weight": 0.5}


def test_zonos_times_conditioning_and_generate_separately(zonos, anchor, calls, monkeypatch,
                                                          tmp_path):
    _clock(monkeypatch, zonos, 1.0, 2.0, 7.0)
    _out, (row,) = _run(zonos, monkeypatch, tmp_path, [_zon_line(ref_file=anchor)])
    assert row["ref_condition_seconds"] == 1.0
    assert row["render_seconds"] == 5.0
    assert row["audio_seconds"] == 3.0
    assert row["rtf"] == round(5.0 / 3.0, 4)
    kinds = [c[0] for c in calls]
    assert kinds == ["load", "seed", "embed", "cond", "prepare", "sync", "generate", "decode"]


def test_render_timing_handles_an_empty_clip():
    import synth_common
    assert synth_common.render_timing(0.1, 0.2, 0.0)["rtf"] is None


# --- existing banks -----------------------------------------------------------------------

def _pool_meta():
    return {"id": "pool_1", "register": "neutral_narration", "engine": "libritts-r",
            "gender": "Female", "score": 0.0, "pinned": True, "ref_excursion_hz": 300.0}


def test_a_ref_id_pin_still_resolves_from_the_pool_undamped(chatterbox, calls, monkeypatch,
                                                            tmp_path):
    seen = []
    monkeypatch.setattr(chatterbox, "pinned_reference",
                        lambda rid: seen.append(rid) or ("/pool/p1.wav", "pool text", _pool_meta()))
    monkeypatch.setattr(chatterbox, "pinned_reference_file",
                        lambda job: pytest.fail("ref_file path taken for a ref_id line"))
    _clock(monkeypatch, chatterbox, 0.0, 1.0, 2.0)
    line = _cbx_line(ref_id="pool_1", direction={"design": "", "exaggeration": 0.5,
                                                 "cfg_weight": 0.5})
    _out, (row,) = _run(chatterbox, monkeypatch, tmp_path, [line])
    assert seen == ["pool_1"]
    assert ("prepare", "/pool/p1.wav", 0.5) in calls
    assert row["applied_exaggeration"] == 0.5 and row["bright_ref_damped"] is False
    assert row["ref"]["id"] == "pool_1" and row["ref_id"] == "pool_1"


def test_a_cast_line_still_goes_through_select_reference_and_damping(chatterbox, calls,
                                                                     monkeypatch, tmp_path):
    seen = []

    def select(design, intended, used, **kw):
        seen.append((design, intended, kw))
        meta = dict(_pool_meta(), pinned=False)
        return "/pool/p2.wav", "pool text", meta
    monkeypatch.setattr(chatterbox, "select_reference", select)
    _clock(monkeypatch, chatterbox, 0.0, 1.0, 2.0)
    line = _cbx_line(intended={"V": 0.1}, direction={"design": "a woman", "exaggeration": 0.5,
                                                     "cfg_weight": 0.5})
    _out, (row,) = _run(chatterbox, monkeypatch, tmp_path, [line])
    assert len(seen) == 1 and "max_excursion" in seen[0][2] and "exclude" in seen[0][2]
    want, damped = ref_select.bright_ref_exaggeration(0.5, 300.0)
    assert damped and want < 0.5, "the fixture must be one the live policy damps"
    assert row["applied_exaggeration"] == want and row["bright_ref_damped"] is True
    assert ("prepare", "/pool/p2.wav", want) in calls


def test_zonos_ref_id_and_cast_lines_reach_the_engine_as_before(zonos, calls, monkeypatch,
                                                                tmp_path):
    ref = tmp_path / "pool.wav"
    sf.write(str(ref), np.zeros(16000, dtype=np.float32), 16000)
    monkeypatch.setattr(zonos, "pinned_reference",
                        lambda rid: (str(ref), "pool text", _pool_meta()))
    monkeypatch.setattr(zonos, "select_reference",
                        lambda *a, **k: (str(ref), "pool text", dict(_pool_meta(), pinned=False)))
    _clock(monkeypatch, zonos, *range(6))
    lines = [_zon_line(id="z1", ref_id="pool_1"),
             _zon_line(id="z2", direction={"design": "", "emotion": None, "pitch_std": 30.0,
                                           "speaking_rate": 20.0})]
    _out, rows = _run(zonos, monkeypatch, tmp_path, lines)
    assert [r["id"] for r in rows] == ["z1", "z2"]
    conds = [c[1] for c in calls if c[0] == "cond"]
    assert conds[0]["pitch_std"] == 45.0 and conds[0]["speaking_rate"] == 15.0
    assert "emotion" in conds[0]["unconditional_keys"]
    assert conds[1]["speaking_rate"] == zonos.MAX_SPEAKING_RATE        # the clamp still bites
    assert rows[1]["applied_speaking_rate"] == zonos.MAX_SPEAKING_RATE
