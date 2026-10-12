"""`measure_speaker_hnr.py`'s speaker choice, and `--speakers-from` (the hum-fix batch,
Notes: Sonora/hum-fix-batch-preregistration.md: the 106 S1 speakers without an HNR entry are
measured by name, not drawn)."""
import json
import random

import numpy as np
import pytest
import soundfile as sf

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import measure_speaker_hnr as mh  # noqa: E402
from matcha.delivery import VAT_DIM  # noqa: E402


def _by_spk():
    return {3: ["a"] * 120, 5: ["b"] * 99, 8: ["c"] * 300, 9: ["d"] * 100, 11: ["e"] * 4}


def test_without_a_list_the_draw_is_unchanged():
    by = _by_spk()
    r1, r2 = random.Random(7), random.Random(7)
    got = mh.choose_speakers(by, 100, 2, r1)
    eligible = sorted(s for s, w in by.items() if len(w) >= 100)        # the old code, verbatim
    want = sorted(r2.sample(eligible, min(2, len(eligible))))
    assert got == want
    assert r1.random() == r2.random()      # same rng state after: the clip draws are unchanged


def test_a_list_measures_exactly_its_speakers_and_does_not_draw():
    by, rng = _by_spk(), random.Random(7)
    assert mh.choose_speakers(by, 1, 400, rng, listed=[11, 3, 5]) == [3, 5, 11]
    assert rng.random() == random.Random(7).random()


def test_a_listed_speaker_without_enough_rows_is_refused_not_skipped():
    with pytest.raises(SystemExit, match="11"):
        mh.choose_speakers(_by_spk(), 5, 400, random.Random(7), listed=[3, 11])
    with pytest.raises(SystemExit, match="42"):
        mh.choose_speakers(_by_spk(), 1, 400, random.Random(7), listed=[3, 42])


def test_the_speakers_file_is_whitespace_separated_integers(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("# the 106\n3 5\n11\n", encoding="utf-8")
    assert mh.read_speakers_file(p) == [3, 5, 11]
    for bad, why in (("3\n3\n", "twice"), ("# none\n", "no speaker"), ("3 x\n", "x")):
        p.write_text(bad, encoding="utf-8")
        with pytest.raises(SystemExit, match=why):
            mh.read_speakers_file(p)


def _corpus(tmp_path, rows_per_spk):
    root = tmp_path / "datasets" / "LibriTTS_R" / "train-clean-100"
    sr, t = 24000, np.arange(24000) / 24000
    wave = 0.5 * sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 6))
    lines = []
    for spk, n in rows_per_spk.items():
        for i in range(n):
            w = root / str(spk) / "1" / ("%d_1_%06d_000000.wav" % (spk, i))
            w.parent.mkdir(parents=True, exist_ok=True)
            sf.write(w, wave, sr)
            lines.append("%s|%d|ab|%s" % (w, spk, ",".join(["0"] * VAT_DIM)))
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "train_op.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return c


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["measure_speaker_hnr.py"] + argv)
    mh.main()


def test_speakers_from_end_to_end_writes_exactly_the_listed_speakers(tmp_path, monkeypatch):
    c = _corpus(tmp_path, {3: 2, 5: 7, 8: 1})
    lst = tmp_path / "s.txt"
    lst.write_text("8\n5\n", encoding="utf-8")
    out = tmp_path / "o.json"
    _run(monkeypatch, ["--corpus", str(c), "--min-rows", "1", "--clips-per-speaker", "5",
                       "--seed", "20261012", "--speakers-from", str(lst), "--json", str(out)])
    j = json.loads(out.read_text())
    assert sorted(j["speakers"]) == ["5", "8"]
    assert j["speakers"]["5"]["clips"] == 5 and j["speakers"]["8"]["clips"] == 1
    assert j["speakers_from"]["speakers"] == [5, 8]


def test_without_speakers_from_the_output_does_not_name_a_list(tmp_path, monkeypatch):
    c = _corpus(tmp_path, {3: 2, 5: 2})
    out = tmp_path / "o.json"
    _run(monkeypatch, ["--corpus", str(c), "--min-rows", "1", "--json", str(out)])
    j = json.loads(out.read_text())
    assert sorted(j["speakers"]) == ["3", "5"] and "speakers_from" not in j


def test_the_output_records_every_option_that_defines_the_measurement(tmp_path, monkeypatch):
    c = _corpus(tmp_path, {3: 2, 5: 2})
    out = tmp_path / "o.json"
    _run(monkeypatch, ["--corpus", str(c), "--min-rows", "1", "--rms-floor", "0.02",
                       "--periodicity", "0.4", "--seed", "99", "--dataset", "LibriTTS_R",
                       "--speakers-sampled", "7", "--json", str(out)])
    j = json.loads(out.read_text())
    assert j["rms_floor"] == 0.02 and j["periodicity"] == 0.4
    assert j["seed"] == 99 and j["dataset"] == "LibriTTS_R" and j["speakers_sampled"] == 7
    assert j["filelist"] == str(c / "train_op.txt")
    assert "speakers_from" not in j       # seed is top-level now, not only under a list
