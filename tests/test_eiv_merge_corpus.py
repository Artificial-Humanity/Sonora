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

def keys_written_to_row(source):
    """Every constant string key written into a local named `row`, by ANY idiom.

    ⚠⚠ THE FIRST VERSION OF THIS WALK COLLECTED ONLY `row = {...}` DICT LITERALS, AND THE
    IDIOM IT COULD NOT SEE WAS THE NEXT LINE OF THE FILE IT GUARDS (#368). `eiv_score.py`
    writes the stamp in a literal and then `row.update({...})` for the heads, so a future
    bookkeeping key added by `update()` or by `row["k"] = …` was invisible: the walk returned
    the same two keys, `unhandled` came out empty, and the guard passed while the reader it
    protects read the new key as a thirteenth head.

    That is the same defect the guard exists to catch, one level up — an enumeration whose
    population is narrower than its subject. So this is a FUNCTION, separately exercised
    against known-answer fixtures below, rather than a walk inlined in the assertion where
    the only input it ever sees is a file that happens to use the idiom it handles.
    """
    tree = ast.parse(source)
    keys, saw_row, unknown = set(), False, []
    for node in ast.walk(tree):
        # row |= {...}
        if (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "row"):
            saw_row = True
            if isinstance(node.op, ast.BitOr) and isinstance(node.value, ast.Dict):
                keys |= {k.value for k in node.value.keys
                         if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            else:
                unknown.append(f"line {node.lineno}: augmented assignment to `row`")
        # row.<method>(...) — setdefault understood, anything else REFUSED rather than skipped
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "row"):
            saw_row = True
            if node.func.attr == "setdefault":
                if node.args and isinstance(node.args[0], ast.Constant) \
                        and isinstance(node.args[0].value, str):
                    keys.add(node.args[0].value)
            elif node.func.attr not in ("update", "get", "keys", "items", "values"):
                unknown.append(f"line {node.lineno}: row.{node.func.attr}(...)")
        # row = {...}
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == "row" for t in node.targets)):
            saw_row = True
            keys |= {k.value for k in node.value.keys
                     if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        # row["k"] = ...
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                        and t.value.id == "row" and isinstance(t.slice, ast.Constant)
                        and isinstance(t.slice.value, str)):
                    saw_row = True
                    keys.add(t.slice.value)
        # row.update({...})
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "row"):
            saw_row = True
            for arg in node.args:
                if isinstance(arg, ast.Dict):
                    keys |= {k.value for k in arg.keys
                             if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return keys, saw_row, unknown


@pytest.mark.parametrize("src,expected", [
    ('row = {"wav": w, "wav_mtime": m}', {"wav", "wav_mtime"}),
    ('row = {"wav": w}\nrow["wav_dur"] = d', {"wav", "wav_dur"}),
    ('row = {"wav": w}\nrow.update({"wav_dur": d})', {"wav", "wav_dur"}),
    ('row = {"wav": w}\nrow.update({name: v for name in heads})', {"wav"}),
    ('other = {"wav": w}', set()),
    ('row = {"wav": w}\nrow |= {"wav_dur": d}', {"wav", "wav_dur"}),
    ('row = {"wav": w}\nrow.setdefault("wav_dur", d)', {"wav", "wav_dur"}),
])
def test_the_walk_sees_every_idiom_that_writes_a_row_key(src, expected):
    """Known-answer fixtures through the same code path the real guard uses.

    Rows two and three are the mutations that shipped green before #368 — measured by the
    reviewer as leaving `written` unchanged. Row four is the comprehension the real writer
    uses for the heads: it has no constant keys and must contribute none, or the guard would
    demand that every HEAD be declared non-head. Row five keeps the walk anchored to `row`.
    """
    keys, _, _ = keys_written_to_row(src)
    assert keys == expected


def test_non_head_keys_covers_every_key_the_writer_adds():
    """The pairing guard: a bookkeeping key the writer adds must be known to the reader.

    It asserts the walk found something before it asserts anything about what it found — an
    enumeration that quietly matches nothing is a vacuous pass, and that is the failure mode
    a guard like this dies of.
    """
    written, saw_row, unknown = keys_written_to_row(SCRIPTS.src("eiv_score.py"))
    assert not unknown, (
        f"eiv_score.py mutates `row` in a way this walk does not model: {unknown}. "
        f"An enumeration of IDIOMS silently omits the one nobody listed — which is how "
        f"#374 was reached after #368 fixed the two idioms it named. Teach the walk the "
        f"new idiom, or the key it writes is invisible to the pairing guard.")
    assert saw_row, (
        "no writes to a local named `row` found in eiv_score.py — the walk is broken, not "
        "the writer; this test cannot pass by finding nothing")
    assert "wav" in written, f"the row no longer carries 'wav': {sorted(written)}"

    unhandled = written - {"wav"} - set(eiv_merge_corpus.NON_HEAD_KEYS)
    assert not unhandled, (
        f"eiv_score.py writes non-score key(s) {sorted(unhandled)} that eiv_merge_corpus "
        f"would read as heads. Add them to NON_HEAD_KEYS — otherwise load_raw's ragged "
        f"check refuses every file written after the change, which is how this test's "
        f"subject was discovered in the first place.")


def test_non_head_keys_is_not_a_wildcard():
    """`wav` stays a positional strip, not a member — load_raw needs it as the KEY."""
    assert "wav" not in eiv_merge_corpus.NON_HEAD_KEYS
    assert eiv_merge_corpus.NON_HEAD_KEYS, "an empty tuple would silently restore the defect"


@pytest.mark.parametrize("src", [
    'row = {"wav": w}\nrow.merge_from(other)',
    'row = {"wav": w}\nrow += other',
])
def test_an_unmodelled_mutation_of_row_is_REFUSED_not_skipped(src):
    """The residual #368 left, and the reason it recurred (#374).

    Handling three idioms and skipping the rest is still an enumeration — the next writer to
    use a fourth gets silence, exactly as `row.update({...})` got silence before. So the walk
    now REPORTS what it could not model, and the real guard asserts that report is empty. A
    new idiom is a red test naming the line, instead of a key that disappears.
    """
    _, saw_row, unknown = keys_written_to_row(src)
    assert saw_row
    assert unknown, "an unmodelled mutation of `row` must be reported, not silently skipped"
