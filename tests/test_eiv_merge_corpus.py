"""The first test that RUNS `scripts/tools/eiv_merge_corpus.py`.

It was reached only by the repo-wide "the module compiles" sweep until the v7 pass, when a
raw file written by the CURRENT `eiv_score.py` met this reader for the first time and the
merge refused: 303,638 of 333,989 clips read as carrying a different head set. Nothing was
mislabelled — `main()`'s ragged check caught it — but the two halves of one file format had
disagreed for as long as both existed, and no test held them together.

The disagreement is worth naming because it is not a typo. `load_raw` promises
`wav -> {head: score}` and strips exactly one key, `wav`. The writer later grew a second
non-score key, `wav_mtime`, as the per-row resume stamp that replaced a whole-file mtime
check (#56) — a good change, made in the file that writes the format, with no reason for its
author to look at the file that reads it. **A format with two owners needs a test that fails
when only one of them moves**, which is `test_non_head_keys_covers_every_key_the_writer_adds`
below: it reads the writer's row literal by AST, so a THIRD bookkeeping key is a red test
here rather than a refused corpus build hours into a run.

Both directions, per this suite's standing rule: the bookkeeping key is accepted AND a
genuinely mixed head set is still refused. A ragged guard that only ever sees clean input is
indistinguishable from `pass`, and one that only ever sees dirty input is indistinguishable
from `return 1` — the v7 pass met the second failure mode in production, which is the whole
reason this file exists.
"""

import ast
import json

import pytest

from scripts_layout import SCRIPTS  # noqa: E402

SCRIPTS.on_path()

import eiv_merge_corpus  # noqa: E402

HEADS = ("Valence", "Arousal", "Soft_vs._Harsh")
SPEAKERS = ("100", "200")


def _row(spk, i, *, mtime=None, drop=None):
    """One raw row in the writer's shape: wav, optional bookkeeping, then heads."""
    row = {"wav": f"/corpus/{spk}/sess/{spk}_{i}.wav"}
    if mtime is not None:
        row["wav_mtime"] = mtime
    for h in HEADS:
        if h != drop:
            row[h] = round(0.1 * i + 0.01 * len(h) + (0.5 if spk == "200" else 0.0), 6)
    return row


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture()
def eiv(tmp_path, monkeypatch):
    """A self-contained eiv_scores/ directory, so nothing here reads or writes /data."""
    monkeypatch.setattr(eiv_merge_corpus, "EIV", tmp_path)
    monkeypatch.setattr(eiv_merge_corpus, "RAW", ("base.jsonl",))
    monkeypatch.setattr(eiv_merge_corpus, "SPEC", tmp_path / "valence_combo_v1.json")

    base = [_row(s, i) for s in SPEAKERS for i in range(3)]
    _write_jsonl(tmp_path / "base.jsonl", base)
    (tmp_path / "valence_combo_v1.json").write_text(
        json.dumps({"heads": list(HEADS), "weights": [0.6, 0.3, 0.1]}), encoding="utf-8")
    # main() reports the recomputation against this file; it only needs shared keys.
    (tmp_path / "corpus_valence_combo.json").write_text(
        json.dumps({r["wav"]: 0.1 * i for i, r in enumerate(base)}), encoding="utf-8")
    return tmp_path


def _run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["eiv_merge_corpus.py", *argv])
    return eiv_merge_corpus.main()


# --------------------------------------------------------------------------------------
# The defect the v7 pass hit.
# --------------------------------------------------------------------------------------

def test_load_raw_strips_the_writers_bookkeeping_keys(eiv):
    """`wav_mtime` is not a head, so it must not reach the head map."""
    _write_jsonl(eiv / "add.jsonl", [_row("300", i, mtime=1723000000 + i) for i in range(3)])
    raw = eiv_merge_corpus.load_raw(["base.jsonl", "add.jsonl"])

    assert len(raw) == 9, "the fixture's own population changed"
    for wav, heads in raw.items():
        assert set(heads) == set(HEADS), f"{wav} carried {sorted(set(heads) - set(HEADS))}"


def test_a_stamped_add_file_merges_instead_of_being_refused(eiv, monkeypatch, capsys):
    """End-to-end: the exact shape that refused on the v7 pass now completes."""
    _write_jsonl(eiv / "add.jsonl", [_row("300", i, mtime=1723000000 + i) for i in range(3)])

    assert _run(monkeypatch, "--add", "add.jsonl", "--suffix", "_t", "--apply") == 0
    out = capsys.readouterr().out
    assert "refusing" not in out, out

    soft = json.loads((eiv / "corpus_soft_t.json").read_text())
    combo = json.loads((eiv / "corpus_valence_combo_t.json").read_text())
    assert len(soft) == len(combo) == 9
    assert all("wav_mtime" not in k for k in combo)


def test_the_ragged_guard_still_refuses_a_genuinely_mixed_head_set(eiv, monkeypatch, capsys):
    """The other direction. Stripping bookkeeping must not blind the check to a real hole.

    This is the guard that turned a silently-dropped weighted head into a refusal, so a fix
    that made it stop firing would be worse than the defect it was fixing.
    """
    _write_jsonl(eiv / "add.jsonl",
                 [_row("300", i, mtime=1723000000 + i, drop="Arousal" if i == 1 else None)
                  for i in range(3)])

    assert _run(monkeypatch, "--add", "add.jsonl", "--suffix", "_t") == 1
    assert "carry a different head set" in capsys.readouterr().out
    assert not list(eiv.glob("corpus_soft_t.json")), "a refused run must write nothing"


# --------------------------------------------------------------------------------------
# The pair, held together.
# --------------------------------------------------------------------------------------

def test_non_head_keys_covers_every_key_the_writer_adds():
    """AST over eiv_score.py's row literal: a new bookkeeping key fails HERE.

    Text-scanning for `wav_mtime` would pass on the comment that mentions it, so this walks
    the writer's actual `row = {...}` assignment. It asserts the walk found something before
    it asserts anything about what it found — an enumeration that quietly matches nothing is
    a vacuous pass, and that is the failure mode a guard like this dies of.
    """
    tree = ast.parse(SCRIPTS.src("eiv_score.py"))
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and any(isinstance(t, ast.Name) and t.id == "row" for t in node.targets)
    ]
    assert literals, (
        "no `row = {...}` literal found in eiv_score.py — the walk is broken, not the writer; "
        "this test cannot pass by finding nothing")

    written = {k.value for lit in literals for k in lit.keys
               if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    assert "wav" in written, f"the row literal no longer carries 'wav': {sorted(written)}"

    bookkeeping = written - {"wav"}
    unhandled = bookkeeping - set(eiv_merge_corpus.NON_HEAD_KEYS)
    assert not unhandled, (
        f"eiv_score.py writes non-score key(s) {sorted(unhandled)} that eiv_merge_corpus "
        f"would read as heads. Add them to NON_HEAD_KEYS — otherwise load_raw's ragged "
        f"check refuses every file written after the change, which is how this test's "
        f"subject was discovered in the first place.")


def test_non_head_keys_is_not_a_wildcard():
    """`wav` stays a positional strip, not a member — load_raw needs it as the KEY."""
    assert "wav" not in eiv_merge_corpus.NON_HEAD_KEYS
    assert eiv_merge_corpus.NON_HEAD_KEYS, "an empty tuple would silently restore the defect"
