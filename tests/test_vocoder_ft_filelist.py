"""build_vocoder_ft_filelist.py's hard invariants, on tiny real wavs.

The tool chooses what the vocoder learns to produce, so two of its rules are not
preferences: synthetic audio never enters, and a held-out speaker is absent WHOLE. Both
were held only by the code as written until this file (review of Sonya/vocoder-gta-finetune).
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "build_vocoder_ft_filelist", os.path.join(REPO, "scripts/tools/build_vocoder_ft_filelist.py"))


@pytest.fixture
def tool(tmp_path, monkeypatch):
    mod = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(mod)
    real, synth = tmp_path / "real", tmp_path / "synth"
    monkeypatch.setattr(mod, "REAL_ROOTS", (str(real) + "/",))
    monkeypatch.setattr(mod, "SYNTHETIC_ROOTS", (str(synth) + "/",))
    return mod, real, synth, tmp_path


def wav(path, seconds=1.5, subtype="PCM_16"):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(24000 * seconds), dtype="float32"), 24000, subtype)
    return str(path)


def corpus(tmp_path, rows):
    p = tmp_path / "train_op.txt"
    p.write_text("\n".join("%s|%d|a|0,0,0" % (w, s) for w, s in rows) + "\n")
    return str(p)


def key(tmp_path, spks):
    p = tmp_path / "key.json"
    p.write_text(json.dumps({"items": {"item_%02d" % i: {"spk": s}
                                       for i, s in enumerate(spks)}}))
    return str(p)


def run(mod, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["build_vocoder_ft_filelist.py", *argv])
    mod.main()


def listed(out, name):
    return [l.split("|")[0] for l in (out / name).read_text().splitlines() if l]


def test_synthetic_rows_are_excluded_and_held_out_speakers_are_absent(tool, monkeypatch):
    mod, real, synth, tmp = tool
    rows = [(wav(real / ("s%d_%d.wav" % (s, i))), s) for s in (1, 2, 3) for i in range(3)]
    rows += [(wav(synth / "fake_0.wav"), 1), (wav(synth / "fake_1.wav"), 2)]
    out = tmp / "out"
    run(mod, monkeypatch, "--corpus", corpus(tmp, rows), "--exclude-key", key(tmp, [3]),
        "--out", str(out), "--val-clips", "1")
    chosen = listed(out, "ft_train_op.txt") + listed(out, "ft_val_op.txt")
    assert chosen, "the positive control: something must be chosen"
    assert not [w for w in chosen if str(synth) in w], "a synthetic clip entered"
    assert not [w for w in chosen if "/s3_" in w], "a held-out speaker's clip entered"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["synthetic_rows_excluded"] == 2
    assert manifest["held_out_speakers"] == [3]
    # hifi-gan's list is the same clips, without `.wav`
    assert listed(out, "hifigan_train.txt") == [w[:-4] for w in listed(out, "ft_train_op.txt")]


def test_an_unclassified_root_refuses(tool, monkeypatch):
    mod, real, synth, tmp = tool
    rows = [(wav(real / "s1_0.wav"), 1), (wav(tmp / "elsewhere" / "x.wav"), 2)]
    with pytest.raises(SystemExit, match="neither real nor synthetic"):
        run(mod, monkeypatch, "--corpus", corpus(tmp, rows), "--out", str(tmp / "out"))


def test_a_key_without_items_refuses(tool, monkeypatch):
    mod, real, synth, tmp = tool
    empty = tmp / "empty.json"
    empty.write_text(json.dumps({"clips": {}}))
    with pytest.raises(SystemExit, match="no `items`"):
        run(mod, monkeypatch, "--corpus", corpus(tmp, [(wav(real / "a.wav"), 1)]),
            "--exclude-key", str(empty), "--out", str(tmp / "out"))


def test_two_clips_with_one_stem_refuse(tool, monkeypatch):
    mod, real, synth, tmp = tool
    rows = [(wav(real / "d1" / "same.wav"), 1), (wav(real / "d2" / "same.wav"), 2)]
    with pytest.raises(SystemExit, match="share a file stem"):
        run(mod, monkeypatch, "--corpus", corpus(tmp, rows), "--out", str(tmp / "out"),
            "--val-clips", "0")


def test_a_non_pcm16_file_refuses(tool, monkeypatch):
    mod, real, synth, tmp = tool
    rows = [(wav(real / "f.wav", subtype="FLOAT"), 1)]
    with pytest.raises(SystemExit, match="PCM_16"):
        run(mod, monkeypatch, "--corpus", corpus(tmp, rows), "--out", str(tmp / "out"))
