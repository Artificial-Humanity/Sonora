"""`prep_staged_corpora.py`'s `d1`, `g` and `f1` writers (the hum-fix batch,
Notes: Sonora/hum-fix-batch-preregistration.md), on tiny fixtures under tmp_path: `DATA` and
the LibriTTS-R root are monkeypatched, so nothing under /data is read or written."""
import argparse
import json

import numpy as np
import pytest
import soundfile as sf

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import prep_staged_corpora as prep  # noqa: E402
from lib import staged_data as sd  # noqa: E402


class _Serial:
    """A stand-in for ProcessPoolExecutor: same `map`, no processes."""
    def __init__(self, *_a, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def map(self, fn, it, chunksize=1):
        return map(fn, it)


def _world(tmp_path, monkeypatch):
    data, libri = tmp_path / "data", tmp_path / "LibriTTS_R"
    monkeypatch.setattr(prep, "DATA", str(data))
    monkeypatch.setattr(sd, "LIBRI_R_ROOT", str(libri))
    monkeypatch.setattr(prep, "ProcessPoolExecutor", _Serial)
    return data, libri


# ------------------------------------------------------------------------------- d1 and g

def test_d1_writes_what_the_helper_produces_and_copies_val(tmp_path, monkeypatch):
    data, _ = _world(tmp_path, monkeypatch)
    src = data / "vctk_22k"
    src.mkdir(parents=True)
    train = ["/x/vctk_22k/%s/%s_%03d.wav|%d|t%d" % (v, v, i, k, i)
             for k, v in enumerate(("p225", "p226")) for i in range(sd.D1_PER_SPEAKER + 5)]
    (src / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (src / "val.txt").write_text("/x/vctk_22k/p225/p225_900.wav|0|v\n", encoding="utf-8")
    prep.d1(argparse.Namespace())
    out = data / "vctk_d1"
    want = sd.d1_subsample(train)
    assert (out / "train.txt").read_text(encoding="utf-8") == "\n".join(want) + "\n"
    assert (out / "val.txt").read_bytes() == (src / "val.txt").read_bytes()
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["rows"] == {"train": len(want), "val": 1} and prov["speakers"] == 2
    assert len(want) == 2 * sd.D1_PER_SPEAKER


def test_g_writes_g_line_of_every_row_beside_s1s(tmp_path, monkeypatch):
    data, _ = _world(tmp_path, monkeypatch)
    d = data / "libritts_r_22k"
    d.mkdir(parents=True)
    rows = {"train": ["/x/a.wav|3|pɚmˈɪɾᵻd ᵻ", "/x/b.wav|4|ɪt"], "val": ["/x/c.wav|3|ᵻ"]}
    for split, ls in rows.items():
        (d / ("%s_op.txt" % split)).write_text("\n".join(ls) + "\n", encoding="utf-8")
    prep.g(argparse.Namespace())
    for split, ls in rows.items():
        want = [sd.g_line(l)[0] for l in ls]
        assert (d / ("%s_g.txt" % split)).read_text(encoding="utf-8") == "\n".join(want) + "\n"
        assert sd.G_FROM not in (d / ("%s_g.txt" % split)).read_text(encoding="utf-8")
    prov = json.loads((d / "provenance_g.json").read_text(encoding="utf-8"))
    assert prov["rows"] == {"train": 2, "val": 1}
    assert prov["replaced"] == {"train": 2, "val": 1}


# ------------------------------------------------------------------------------------ f1

SOURCES = (("libritts_r_vat", "train-clean-100"),
           ("_derived_train_clean_360", "train-clean-360"),
           ("_derived_train_other_500", "train-other-500"))


def _f1_world(tmp_path, monkeypatch):
    """Dense speakers 100 (clean-100, no measures) and 200 (clean-360); speaker 300
    (other-500) has 2 train rows only, so it is not dense. Dense = 3 rows here."""
    data, libri = _world(tmp_path, monkeypatch)
    real = sd.f1_dense_speakers
    monkeypatch.setattr(sd, "f1_dense_speakers", lambda w: real(w, min_rows=3))
    (data / "libritts_r_full_vat_v7").mkdir(parents=True)
    (data / "libritts_r_full_vat_v7" / "speakers.json").write_text(
        json.dumps({"libritts_id_to_index": {"100": 0, "200": 1, "300": 2}}))
    # per source: (speaker, [(seconds, ...)]) for train and val
    spec = {
        "libritts_r_vat": {"train": [(100, 5.0)] * 3, "val": [(100, 5.0)]},
        "_derived_train_clean_360": {
            # 1 <= s <= 16 inclusive: 0.99 and 16.01 go, 1.0 and 16.0 stay
            "train": [(200, 0.99), (200, 1.0), (200, 16.0), (200, 16.01), (200, 5.0)],
            "val": [(200, 5.0), (200, 20.0), (200, 0.5)]},
        "_derived_train_other_500": {"train": [(300, 5.0)] * 2, "val": [(300, 5.0)]}}
    sr, wave = 24000, 0.1 * np.sin(2 * np.pi * 150 * np.arange(2400) / 24000)
    for (name, subset), idx in zip(SOURCES, (5, 6, 7)):
        d = data / name
        d.mkdir(parents=True, exist_ok=True)
        spk = {"libritts_r_vat": 100, "_derived_train_clean_360": 200,
               "_derived_train_other_500": 300}[name]
        (d / "speakers.json").write_text(json.dumps({"libritts_id_to_index": {str(spk): idx}}))
        measures, n = [], 0
        for split in ("train", "val"):
            lines = []
            for sp, sec in spec[name][split]:
                n += 1
                w = libri / subset / str(sp) / "1" / ("%d_1_%06d_000000.wav" % (sp, n))
                w.parent.mkdir(parents=True, exist_ok=True)
                sf.write(w, wave, sr)
                lines.append("%s|%d|ab|%s" % (w, idx, ",".join(["0"] * 4)))
                measures.append(json.dumps({"wav": str(w), "seconds": sec}))
            (d / ("%s_op.txt" % split)).write_text("\n".join(lines) + "\n", encoding="utf-8")
        if name != "libritts_r_vat":
            (d / "measures.jsonl").write_text("\n".join(measures) + "\n", encoding="utf-8")
    monkeypatch.setattr(prep, "F1_EXPECT", {"speakers": 2, "train": 6, "val": 2})
    return data, libri


ARGS = argparse.Namespace(workers=1)


def test_f1_keeps_one_to_sixteen_seconds_inclusive_and_sources_val_the_same_way(
        tmp_path, monkeypatch):
    data, libri = _f1_world(tmp_path, monkeypatch)
    prep.f1(ARGS)
    out = data / "f1_dense"
    train = (out / "train_op.txt").read_text(encoding="utf-8").splitlines()
    val = (out / "val_op.txt").read_text(encoding="utf-8").splitlines()
    assert len(train) == 6 and len(val) == 2
    by_spk = lambda ls: sorted(int(l.split("|")[1]) for l in ls)      # noqa: E731
    assert by_spk(train) == [0, 0, 0, 1, 1, 1]       # F1 indices of 100 and 200; 300 absent
    assert by_spk(val) == [0, 1]                     # 300's val row left out; 200's 5 s row kept
    # speaker 200's kept train rows are the 1.0, 16.0 and 5.0 s ones, not 0.99 or 16.01
    kept200 = sorted(l.split("|")[0] for l in train if l.split("|")[1] == "1")
    assert [p.split("_")[-2] for p in kept200] == ["000002", "000003", "000005"]
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["filtered_out"]["_derived_train_clean_360/train"] == 2
    assert prov["filtered_out"]["_derived_train_clean_360/val"] == 2
    assert json.loads((out / "speakers.json").read_text())["libritts_id_to_index"] == {
        "100": 0, "200": 1}
    for l in train + val:                            # every listed wav was written
        assert sf.info(l.split("|")[0]).samplerate == sd.TARGET_SR
        assert l.split("|")[0].startswith(str(libri))


def test_f1_refuses_a_count_mismatch_before_writing_anything(tmp_path, monkeypatch):
    data, libri = _f1_world(tmp_path, monkeypatch)
    monkeypatch.setattr(prep, "F1_EXPECT", {"speakers": 2, "train": 7, "val": 2})
    with pytest.raises(SystemExit, match="REFUSING"):
        prep.f1(ARGS)
    assert not (data / "f1_dense").exists()
    assert not list(libri.glob("*_22k"))             # no audio copy either


def test_f1_writes_no_filelist_if_a_copy_fails_midway(tmp_path, monkeypatch):
    data, _ = _f1_world(tmp_path, monkeypatch)
    calls = []

    def boom(src):
        calls.append(src)
        if len(calls) == 3:
            raise RuntimeError("disk full")
        return prep._f1_job_real(src)
    monkeypatch.setattr(prep, "_f1_job_real", prep._f1_job, raising=False)
    monkeypatch.setattr(prep, "_f1_job", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        prep.f1(ARGS)
    out = data / "f1_dense"
    for name in ("train_op.txt", "val_op.txt", "speakers.json", "provenance.json"):
        assert not (out / name).exists()


# ------------------------------------------------------------------------------------ e

def _e_world(tmp_path, monkeypatch):
    data, libri = _world(tmp_path, monkeypatch)
    r22 = libri / "train-clean-100_22k"
    r24 = libri / "train-clean-100"
    monkeypatch.setattr(sd, "LIBRI_22K_ROOT", str(r22))
    monkeypatch.setattr(sd, "LIBRI_24K_ROOT", str(r24))
    monkeypatch.setattr(prep, "E_EXPECT", {"train": 2, "val": 1})
    monkeypatch.setattr(prep, "_espeak_versions", lambda: dict(prep.E_VERSIONS))
    d = data / "libritts_r_22k"
    d.mkdir(parents=True)
    spec = {
        "train": [("1/2/1_2_000001_000000", 3, "pɚmˈɪɾᵻd", "Permitted.", "pɚmˈɪɾᵻd."),
                  ("1/2/1_2_000002_000000", 4, "ɪt", "It", "ɪt")],
        "val": [("1/2/1_2_000003_000000", 3, "kˈæt", "Cats", "kˈæts")]}
    espeak_map = {}
    for split, items in spec.items():
        lines = []
        for rel, spk, op_ph, text, es_ph in items:
            tpath = r24 / (rel + ".normalized.txt")
            tpath.parent.mkdir(parents=True, exist_ok=True)
            tpath.write_text(text + "\n", encoding="utf-8")
            lines.append("%s/%s.wav|%d|%s" % (r22, rel, spk, op_ph))
            espeak_map[text] = es_ph
        (d / ("%s_op.txt" % split)).write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(prep, "_espeak_phonemize", lambda texts, workers=1: [espeak_map[t] for t in texts])
    return d, r24, espeak_map


def test_e_writes_remapped_espeak_phonemes_beside_s1s(tmp_path, monkeypatch):
    d, _r24, _es = _e_world(tmp_path, monkeypatch)
    prep.e(ARGS)
    train = (d / "train_e.txt").read_text(encoding="utf-8").splitlines()
    val = (d / "val_e.txt").read_text(encoding="utf-8").splitlines()
    assert [l.split("|")[2] for l in train] == ["pɚmˈɪɾᵊd.", "ɪt"]
    assert [l.split("|")[2] for l in val] == ["kˈæts"]
    assert sd.G_FROM not in (d / "train_e.txt").read_text(encoding="utf-8")
    assert sd.G_FROM not in (d / "val_e.txt").read_text(encoding="utf-8")
    prov = json.loads((d / "provenance_e.json").read_text(encoding="utf-8"))
    assert prov["rows"] == {"train": 2, "val": 1}
    assert prov["differ_from_s1"] == {"train": 1, "val": 1}
    assert prov["differ_from_g"] == {"train": 1, "val": 1}
    assert prov["phonemizer_version"] == "3.4.0" and prov["espeak_ng_version"] == "1.52.0"
    assert prov["outputs"] == {
        "train": [str(d / "train_e.txt"), prep.sha256(d / "train_e.txt")],
        "val": [str(d / "val_e.txt"), prep.sha256(d / "val_e.txt")]}


def test_e_refuses_missing_transcript_or_count_or_version_mismatch_before_writing(tmp_path, monkeypatch):
    d, r24, es = _e_world(tmp_path, monkeypatch)
    monkeypatch.setattr(prep, "_espeak_versions", lambda: {"phonemizer": "3.3.0", "espeak_ng": "1.52.0"})
    with pytest.raises(SystemExit, match="REFUSING.*requires"):
        prep.e(ARGS)
    assert not (d / "train_e.txt").exists()
    monkeypatch.setattr(prep, "_espeak_versions", lambda: dict(prep.E_VERSIONS))
    monkeypatch.setattr(prep, "E_EXPECT", {"train": 3, "val": 1})
    with pytest.raises(SystemExit, match="REFUSING"):
        prep.e(ARGS)
    assert not (d / "train_e.txt").exists()
    monkeypatch.setattr(prep, "E_EXPECT", {"train": 2, "val": 1})
    (r24 / "1/2/1_2_000001_000000.normalized.txt").unlink()
    with pytest.raises(SystemExit, match="REFUSING"):
        prep.e(ARGS)
    assert not (d / "train_e.txt").exists()
    (r24 / "1/2/1_2_000001_000000.normalized.txt").write_text("Permitted.\n", encoding="utf-8")
    es["Permitted."] = ""
    with pytest.raises(SystemExit, match="REFUSING"):
        prep.e(ARGS)
    assert not (d / "train_e.txt").exists()
    es["Permitted."] = "pɚm'ɪɾᵻd"
    with pytest.raises(SystemExit, match="REFUSING.*id 176"):
        prep.e(ARGS)
    assert not (d / "train_e.txt").exists()


def test_espeak_phonemize_matches_english_cleaners2(monkeypatch):
    from matcha.text import cleaners
    seen = []

    class _FakeEspeak:
        def phonemize(self, texts, strip=True, njobs=1):
            assert strip is True
            seen.append((list(texts), njobs))
            return ["[%s]   ˈoʊk" % t for t in texts]

    monkeypatch.setattr(cleaners, "_get_espeak", lambda: _FakeEspeak())
    sents = ["Hello, Mr. Smith!", "Permitted [by] Drs. Café -- really?"]
    expected = [cleaners.english_cleaners2(s) for s in sents]
    got = prep._espeak_phonemize(sents, workers=4)
    assert got == expected
    assert seen[-1] == ([s[0][0] for s in seen[:-1]], 4)
