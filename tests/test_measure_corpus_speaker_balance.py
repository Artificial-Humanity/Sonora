"""Cases for `scripts/tools/measure_corpus_speaker_balance.py`.

Every case here is a failure the tool would otherwise report as a measurement. The ones
that matter most are the refusals: a balance report over an empty or half-read corpus looks
exactly like a balance report, and the number it prints — a concentration figure — is the
one the rung-4 argument rests on.

The fixture is deliberately lopsided in the same direction as the real corpus (one speaker
holding most of the rows) so the concentration assertions have something to find. A uniform
fixture would pass an implementation that sorted the wrong way.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "scripts" / "tools" / "measure_corpus_speaker_balance.py"

sys.path.insert(0, str(REPO))
from matcha.delivery import VAT_DIM  # noqa: E402

_VEC = ",".join(["0"] * VAT_DIM)


def _row(path, spk, vec=_VEC):
    return "%s|%d|fˈoʊniːmz|%s" % (path, spk, vec)


def _corpus(tmp_path, rows, name="train_op.txt"):
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    (d / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return d


def _run(*args):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True)


# One speaker with 6 rows, one with 3, two with 1 — 11 rows over 4 speakers, so the top
# speaker alone clears half and `half_the_rows` must be 1 rather than 2.
_LOPSIDED = (
    [_row("/d/sets/LibriTTS_R/train-other-500/900/c/%d.wav" % i, 900) for i in range(6)]
    + [_row("/d/sets/LibriTTS_R/train-clean-100/901/c/%d.wav" % i, 901) for i in range(3)]
    + [_row("/d/sets/emilia/wavs/a.wav", 902), _row("/d/sets/emilia/wavs/b.wav", 903)]
)


def test_the_concentration_counts_the_fewest_speakers_that_clear_half(tmp_path):
    """POSITIVE CONTROL: the one speaker holding 6 of 11 rows is half the corpus alone."""
    d = _corpus(tmp_path, _LOPSIDED)
    out = tmp_path / "r.json"
    r = _run("--corpus", str(d), "--json", str(out))
    assert r.returncode == 0, r.stderr
    rep = json.loads(out.read_text())
    assert rep["rows"] == 11 and rep["speakers"] == 4
    assert rep["half_the_rows"] == 1
    assert rep["rows_per_speaker"]["max"] == 6
    # Nearest rank, not floor: n=4, p50 -> ceil(2) = rank 2 of sorted [1,1,3,6] = 1.
    # The floor form returned index 2 (= 3) and this assertion used to lock that in.
    assert rep["rows_per_speaker"]["p50"] == 1
    assert rep["rows_per_speaker"]["p25"] == 1
    assert rep["rows_per_speaker"]["p99"] == 6


def test_the_partition_key_is_taken_below_the_shared_root(tmp_path):
    d = _corpus(tmp_path, _LOPSIDED)
    out = tmp_path / "r.json"
    assert _run("--corpus", str(d), "--json", str(out)).returncode == 0
    rep = json.loads(out.read_text())
    assert rep["root"] == "/d/sets"
    assert rep["partitions"] == {"LibriTTS_R": 9, "emilia": 2}


def test_depth_two_splits_libritts_into_its_own_quality_partitions(tmp_path):
    """The clean/other split is the whole point of --depth 2 and must survive it."""
    d = _corpus(tmp_path, _LOPSIDED)
    out = tmp_path / "r.json"
    assert _run("--corpus", str(d), "--depth", "2", "--json", str(out)).returncode == 0
    rep = json.loads(out.read_text())
    assert rep["partitions"] == {
        "LibriTTS_R/train-other-500": 6,
        "LibriTTS_R/train-clean-100": 3,
        "emilia/wavs": 2,
    }


def test_a_depth_that_names_speakers_rather_than_partitions_refuses(tmp_path):
    """The real scenario: enough speaker directories that the DEFAULT --max-groups trips.

    This case used to pass `--max-groups 2` against a fixture yielding three groups, so it
    was a --max-groups test wearing a depth test's name and the documented failure — depth 3
    on a real corpus producing thousands of speaker groups — was never exercised.
    """
    rows = [_row("/d/sets/L/tr-other/%d/c/0.wav" % (900 + i), 900 + i) for i in range(60)]
    rows.append(_row("/d/sets/emilia/wavs/a.wav", 999))
    d = _corpus(tmp_path, rows)
    r = _run("--corpus", str(d), "--depth", "3")
    assert r.returncode == 2
    assert "produces 61 groups" in r.stderr
    assert "per-speaker split" in r.stderr
    # The remedy is searched, so it must name a depth that actually fits.
    assert "--depth 2 produces 2." in r.stderr


def test_the_max_groups_remedy_is_searched_not_hardcoded_to_depth_one(tmp_path):
    """Refusing AT depth 1 used to advise "--depth 1 produces 60" — itself."""
    rows = [_row("/d/sets/%d/c/0.wav" % (900 + i), 900 + i) for i in range(60)]
    d = _corpus(tmp_path, rows)
    r = _run("--corpus", str(d), "--depth", "1")
    assert r.returncode == 2
    assert "--depth 1 produces 60." not in r.stderr
    assert "No shallower depth" in r.stderr


@pytest.mark.parametrize("depth", ["0", "-1"])
def test_a_depth_below_one_refuses_instead_of_printing_a_nameless_group(tmp_path, depth):
    """--depth 0 keyed every row on "" and printed one unnamed group holding 100%."""
    d = _corpus(tmp_path, _LOPSIDED)
    r = _run("--corpus", str(d), "--depth", depth)
    assert r.returncode == 2
    assert "not a partition depth" in r.stderr


def test_a_single_row_with_no_directory_refuses_like_two_of_them_do(tmp_path):
    """The n=1 escape: `root` started empty and the emptiness check was inside the loop."""
    d = _corpus(tmp_path, [_row("only.wav", 900)])
    r = _run("--corpus", str(d))
    assert r.returncode == 2
    assert "share no directory prefix" in r.stderr


def test_every_row_sitting_in_the_root_refuses_rather_than_printing_an_empty_table(tmp_path):
    d = _corpus(tmp_path, [_row("/d/sets/a.wav", 900), _row("/d/sets/b.wav", 901)])
    r = _run("--corpus", str(d))
    assert r.returncode == 2
    assert "no path component left" in r.stderr


def test_a_split_that_excludes_speakers_says_so_in_the_header(tmp_path):
    """v7's val filelist holds 53 speakers train does not; quoting one AS the corpus is the bug."""
    d = _corpus(tmp_path, _LOPSIDED)
    _corpus(tmp_path, [_row("/d/sets/LibriTTS_R/train-clean-100/950/c/0.wav", 950)],
            name="val_op.txt")
    r = _run("--corpus", str(d), "--split", "train")
    assert r.returncode == 0, r.stderr
    assert "1 further speakers appear ONLY in the filelist this split excludes" in r.stdout
    assert _run("--corpus", str(d), "--split", "both").stdout.count("further speakers") == 0


def test_the_row_control_survives_python_dash_O(tmp_path):
    """The control was two asserts, and `python -O` removed both of them silently."""
    d = _corpus(tmp_path, _LOPSIDED)
    r = subprocess.run([sys.executable, "-O", str(TOOL), "--corpus", str(d)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    bad = _corpus(tmp_path, [_row("/d/sets/a/b.wav", 900), "/d/sets/a/c.wav|901"])
    r = subprocess.run([sys.executable, "-O", str(TOOL), "--corpus", str(bad)],
                       capture_output=True, text=True)
    assert r.returncode == 2, "the -O run accepted a malformed row"


def test_an_empty_corpus_refuses_rather_than_reporting_a_tidy_zero(tmp_path):
    d = _corpus(tmp_path, [""])
    r = _run("--corpus", str(d))
    assert r.returncode == 3
    assert "0 rows" in r.stderr


def test_a_missing_filelist_refuses_and_names_what_it_wanted(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    r = _run("--corpus", str(d))
    assert r.returncode == 2
    assert "train_op.txt" in r.stderr


def test_a_short_row_refuses_and_names_the_line(tmp_path):
    d = _corpus(tmp_path, [_LOPSIDED[0], "/d/sets/a/b/c.wav|900"])
    r = _run("--corpus", str(d))
    assert r.returncode == 2
    # ⚠ `"4" in r.stderr` used to stand here and the tmp path alone satisfies it.
    assert "line 2" in r.stderr
    assert "states 2 `|`-separated fields" in r.stderr


def test_a_non_numeric_speaker_refuses(tmp_path):
    d = _corpus(tmp_path, ["/d/sets/a/b/c.wav|spk900|f|%s" % _VEC])
    r = _run("--corpus", str(d))
    assert r.returncode == 2
    assert "not an index" in r.stderr


def test_a_filelist_of_the_wrong_conditioning_width_refuses(tmp_path):
    """A 3-wide vat filelist parses cleanly here and means a different model (#holdout trap)."""
    d = _corpus(tmp_path, [_row("/d/sets/a/b/c.wav", 900, vec="0,0,0")])
    r = _run("--corpus", str(d))
    assert r.returncode == 2
    assert "VAT_DIM as %d" % VAT_DIM in r.stderr


def test_paths_sharing_no_root_refuse_rather_than_keying_on_the_dataset_root(tmp_path):
    d = _corpus(tmp_path, [_row("/alpha/x/a.wav", 900), _row("/beta/y/b.wav", 901)])
    r = _run("--corpus", str(d))
    assert r.returncode == 2
    assert "share no directory prefix" in r.stderr


def test_a_named_speaker_reports_its_rank_and_partitions(tmp_path):
    d = _corpus(tmp_path, _LOPSIDED)
    out = tmp_path / "r.json"
    r = _run("--corpus", str(d), "--speakers", "900,901", "--depth", "2", "--json", str(out))
    assert r.returncode == 0, r.stderr
    named = json.loads(out.read_text())["named_speakers"]
    assert named["900"] == {"rows": 6, "rank": 1, "of": 4,
                            "partitions": ["LibriTTS_R/train-other-500"]}
    assert named["901"]["rank"] == 2


def test_an_unknown_named_speaker_refuses_and_states_the_range(tmp_path):
    d = _corpus(tmp_path, _LOPSIDED)
    r = _run("--corpus", str(d), "--speakers", "12345")
    assert r.returncode == 2
    assert "indices run 900 to 903" in r.stderr
    # ⚠ It used to refuse AFTER the whole report had gone to stdout, so stdout and the
    # exit code disagreed about whether the run had happened.
    assert r.stdout == "", "a refused run printed a measurement anyway"


@pytest.mark.parametrize("split", ["train", "val", "both"])
def test_every_split_reads_the_filelist_it_names(tmp_path, split):
    d = _corpus(tmp_path, _LOPSIDED)
    # ⚠ Two rows from ONE directory make that directory the shared root and leave every
    # row unkeyed, which the tool now refuses. The val fixture has to span two.
    _corpus(tmp_path, [_LOPSIDED[0], _LOPSIDED[9]], name="val_op.txt")
    out = tmp_path / "r.json"
    r = _run("--corpus", str(d), "--split", split, "--json", str(out))
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["rows"] == {"train": 11, "val": 2, "both": 13}[split]
