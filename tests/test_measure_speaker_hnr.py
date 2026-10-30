"""measure_speaker_hnr.py on a 3-field, no-VAT corpus (data/q_synth_22k) measures each clip on its
24 kHz source under the corpus's raw_root, and refuses what it cannot measure exactly: a 22.05 kHz
or missing source, a mixed field count, or a 3-field corpus with no raw_root. Its 4-field (VAT)
behaviour is unchanged."""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("measure_speaker_hnr",
                                               REPO / "scripts/tools/measure_speaker_hnr.py")
mh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mh)


def _tone(path, rate, seconds=1.2, f0=120.0):
    t = np.arange(int(rate * seconds)) / rate
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    x = 0.5 * x / np.abs(x).max() + 0.01 * np.random.default_rng(1).standard_normal(len(t))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype("float32"), rate, "PCM_16")


def _layout(tmp_path, rate=24000, provenance=True):
    """A raw root with two speakers' 24 kHz sources and a 3-field corpus pointing at 22.05 kHz
    copies under corpus/wavs/. Returns (corpus dir, raw root)."""
    raw = tmp_path / "q_synth_raw"
    corpus = tmp_path / "q_synth_22k"
    lines = []
    for spk, sid in (("spk_00", 0), ("spk_01", 1)):
        for cid in ("%s-001" % spk, "%s-002" % spk):
            _tone(raw / spk / "audio" / (cid + ".wav"), rate, f0=110.0 + 20 * sid)
            copy = corpus / "wavs" / spk / (cid + ".wav")
            lines.append("%s|%d|h%s" % (copy, sid, cid))
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "train_op.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if provenance:
        (corpus / "provenance.json").write_text(json.dumps({"raw_root": str(raw)}),
                                                encoding="utf-8")
    return corpus, raw


def _run(monkeypatch, tmp_path, corpus, *extra):
    out = tmp_path / "hnr.json"
    argv = ["measure_speaker_hnr.py", "--corpus", str(corpus), "--min-rows", "1",
            "--clips-per-speaker", "2", "--seed", "20261029", "--json", str(out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    mh.main()
    return json.loads(out.read_text())


def test_a_3field_corpus_is_measured_on_its_24khz_sources(tmp_path, monkeypatch):
    corpus, raw = _layout(tmp_path)
    doc = _run(monkeypatch, tmp_path, corpus)
    assert set(doc["speakers"]) == {"0", "1"}
    assert doc["raw_root"] == str(raw) and doc["dataset"] is None
    assert all(v["clips"] == 2 for v in doc["speakers"].values())


def test_the_raw_root_flag_overrides_the_provenance(tmp_path, monkeypatch):
    corpus, raw = _layout(tmp_path, provenance=False)
    doc = _run(monkeypatch, tmp_path, corpus, "--raw-24k-root", str(raw))
    assert doc["raw_root"] == str(raw) and set(doc["speakers"]) == {"0", "1"}


def test_a_22khz_source_is_refused(tmp_path, monkeypatch):
    corpus, raw = _layout(tmp_path)
    _tone(raw / "spk_00" / "audio" / "spk_00-001.wav", 22050)     # overwrite with 22.05 kHz
    with pytest.raises(SystemExit, match="22050 Hz"):
        _run(monkeypatch, tmp_path, corpus)


def test_a_missing_24khz_source_is_refused(tmp_path, monkeypatch):
    corpus, raw = _layout(tmp_path)
    (raw / "spk_01" / "audio" / "spk_01-002.wav").unlink()
    with pytest.raises(SystemExit, match="missing"):
        _run(monkeypatch, tmp_path, corpus)


def test_a_3field_corpus_with_no_raw_root_is_refused(tmp_path, monkeypatch):
    corpus, _raw = _layout(tmp_path, provenance=False)
    with pytest.raises(SystemExit, match="raw_root"):
        _run(monkeypatch, tmp_path, corpus)


def test_mixed_field_counts_are_refused(tmp_path, monkeypatch):
    corpus, _raw = _layout(tmp_path)
    fp = corpus / "train_op.txt"
    fp.write_text(fp.read_text() + "x|0|a|" + ",".join(["0.0"] * mh.VAT_DIM) + "\n",
                  encoding="utf-8")
    with pytest.raises(SystemExit, match="mixes field counts"):
        _run(monkeypatch, tmp_path, corpus)


def test_raw_24khz_flag_is_refused_on_a_vat_corpus(tmp_path, monkeypatch):
    vat = tmp_path / "vat"
    vat.mkdir()
    (vat / "train_op.txt").write_text("/x/a.wav|0|p|" + ",".join(["0.0"] * mh.VAT_DIM) + "\n",
                                      encoding="utf-8")
    with pytest.raises(SystemExit, match="4-field"):
        _run(monkeypatch, tmp_path, vat, "--raw-24k-root", str(tmp_path))


def test_a_missing_source_on_an_unsampled_row_is_still_refused(tmp_path, monkeypatch):
    corpus, raw = _layout(tmp_path)
    (raw / "spk_01" / "audio" / "spk_01-002.wav").unlink()
    out = tmp_path / "hnr.json"
    argv = ["measure_speaker_hnr.py", "--corpus", str(corpus), "--min-rows", "1",
            "--clips-per-speaker", "1", "--seed", "20261029", "--json", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="missing"):
        mh.main()
    assert not out.exists()


def test_dataset_is_refused_on_a_3field_corpus(tmp_path, monkeypatch):
    corpus, _raw = _layout(tmp_path)
    with pytest.raises(SystemExit, match="3-field"):
        _run(monkeypatch, tmp_path, corpus, "--dataset", "LibriTTS_R")


def test_an_empty_filelist_is_refused_by_name(tmp_path, monkeypatch):
    corpus = tmp_path / "empty"
    corpus.mkdir()
    (corpus / "train_op.txt").write_text("\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="is empty"):
        _run(monkeypatch, tmp_path, corpus)
