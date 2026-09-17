"""`measure_hifi_libritts_overlap.py` decides how much of Hi-Fi TTS the v8 merge deletes, so
the case that has to be impossible is a SILENT one: reporting "no overlap" while the side
being measured was never read.

That was not hypothetical. The first version scored its positive control on the LibriTTS-R
side only. Renaming the Hi-Fi TTS transcripts to a suffix the walker skips produced `none`,
both controls green, exit 0 — the "empty enumeration is a vacuous pass" shape, on the corpus
proposed for deletion. Every refusal below was reproduced by hand against the real corpora
before it was written as a test.

⚠ THESE FIXTURES ARE TINY TREES, NOT THE CORPORA. The real run takes minutes and needs 49 GB
on `/data`; the behaviour under test is the walking and the refusing, which a handful of
files exercises exactly as well. The one thing a fixture cannot check is the real measured
verdict, and that belongs in the notes, not here.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "scripts" / "tools" / "measure_hifi_libritts_overlap.py"

# Long enough to yield 8-grams on both sides; the shared half is what makes the pair real.
SHARED = ("the quick brown fox jumps over the lazy dog and then walks away slowly "
          "through the tall wet grass beside the river in the early morning light")
OTHER = ("a completely different sentence about machinery and steam pressure gauges "
         "that shares no phrasing whatsoever with the passage above it in this file")


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree(tmp_path, hifi_text=SHARED, lib_text=SHARED, suffix=".normalized.txt",
          speaker="99", second_reader=False):
    """A two-corpus fixture: one shared reader, one book of one section, one chapter."""
    hifi, lib = tmp_path / "hifi", tmp_path / "lib"
    rows = []
    for i in range(3):
        stem = "title_01_reader_%04d" % i
        _write(hifi / speaker / "bookA" / (stem + suffix), hifi_text)
        rows.append({"id": stem, "speaker": speaker, "book": "bookA", "source_duration": 2.0})
        _write(lib / "train-clean-100" / speaker / "chap1" / ("%s_chap1_%03d.normalized.txt" % (speaker, i)),
               lib_text)
        (lib / "train-clean-100" / speaker / "chap1" / ("%s_chap1_%03d.wav" % (speaker, i))).write_bytes(
            _silent_wav())
    if second_reader:
        for i in range(3):
            stem = "other_01_reader_%04d" % i
            _write(hifi / "77" / "bookB" / (stem + ".normalized.txt"), OTHER)
            rows.append({"id": stem, "speaker": "77", "book": "bookB", "source_duration": 2.0})
            _write(lib / "train-clean-100" / "77" / "chap9" / ("77_chap9_%03d.normalized.txt" % i), OTHER)
            (lib / "train-clean-100" / "77" / "chap9" / ("77_chap9_%03d.wav" % i)).write_bytes(_silent_wav())
    (hifi / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return hifi, lib


def _silent_wav():
    """A 44-byte header for 1 s of 24 kHz mono PCM. `libritts_cost` reads the header only."""
    import struct
    n = 24000 * 2
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
            + b"data" + struct.pack("<I", n) + b"\0" * n)


def _run(hifi, lib, *extra):
    return subprocess.run(
        [sys.executable, str(TOOL), "--hifi-root", str(hifi), "--libritts-root", str(lib),
         "--subsets", "train-clean-100", *extra],
        capture_output=True, text=True, cwd=REPO)


def test_an_intact_duplicate_pair_is_flagged(tmp_path):
    """⚠ THE POSITIVE CONTROL FOR EVERY REFUSAL BELOW. Without it, a tool that refused
    everything would pass this file."""
    r = _run(*_tree(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "none" not in r.stdout.split("SAME RECORDING")[1].split("DROP")[0], r.stdout
    assert "bookA" not in r.stdout or "title_01_reader" in r.stdout, r.stdout


def test_unreadable_hifi_transcripts_refuse_instead_of_reporting_no_overlap(tmp_path):
    """The measured regression: wrong suffix -> `none`, both controls green, exit 0."""
    hifi, lib = _tree(tmp_path, suffix=".txt")
    r = _run(hifi, lib)
    assert r.returncode != 0, ("a silent 'no overlap' is the failure this tool exists to "
                               "avoid", r.stdout)
    assert "bookA" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr, r.stderr


def test_an_unknown_subset_is_refused_and_names_what_exists(tmp_path):
    hifi, lib = _tree(tmp_path)
    r = _run(hifi, lib, "--subsets", "train-clean-1OO")
    assert r.returncode != 0, r.stdout
    assert "train-clean-100" in r.stderr, ("the refusal must name what does exist", r.stderr)
    assert "Traceback" not in r.stderr, r.stderr


def test_no_shared_reader_is_refused_with_both_counts(tmp_path):
    hifi, lib = _tree(tmp_path)
    (lib / "train-clean-100" / "99").rename(lib / "train-clean-100" / "1234")
    r = _run(hifi, lib)
    assert r.returncode != 0, r.stdout
    assert "no reader appears in both corpora" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr, r.stderr


def test_one_shared_reader_says_the_cross_reader_floor_is_empty(tmp_path):
    """⚠ A floor computed over zero pairs is not a floor, and it prints as 0.000000 either
    way. The count is what distinguishes them."""
    r = _run(*_tree(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "best of 0 cross-reader pairs" in r.stdout, r.stdout
    assert "NO CROSS-READER PAIRS EXIST" in r.stdout, r.stdout


def test_a_second_reader_restores_a_real_floor(tmp_path):
    r = _run(*_tree(tmp_path, second_reader=True))
    assert r.returncode == 0, r.stderr
    assert "best of 0 cross-reader pairs" not in r.stdout, r.stdout
    assert "NO CROSS-READER PAIRS EXIST" not in r.stdout, r.stdout


def test_a_flagged_section_missing_from_the_manifest_is_refused(tmp_path):
    """⚠ `0 clips 0.00 h` reads as 'cheap to drop' rather than as 'not measured'."""
    hifi, lib = _tree(tmp_path)
    (hifi / "manifest.jsonl").write_text("", encoding="utf-8")
    r = _run(hifi, lib)
    assert r.returncode != 0, r.stdout
    assert "absent from" in r.stderr, r.stderr


def test_a_stem_with_no_utterance_index_is_refused(tmp_path):
    """The section is the comparison unit; guessing it changes what is compared."""
    hifi, lib = _tree(tmp_path)
    for p in sorted((hifi / "99" / "bookA").glob("*.normalized.txt")):
        p.rename(p.with_name("nosuffixhere.normalized.txt"))
        break
    r = _run(hifi, lib)
    assert r.returncode != 0, r.stdout
    assert "utterance index" in r.stderr, r.stderr


@pytest.mark.parametrize("n", [4, 8, 12])
def test_the_shingle_size_is_a_flag_because_it_is_load_bearing(tmp_path, n):
    """n and the threshold decide the verdict jointly; a hardcoded n cannot be swept."""
    r = _run(*_tree(tmp_path), "--shingle", str(n))
    assert r.returncode == 0, r.stderr
    assert ("n=%d" % n) in r.stdout, r.stdout
