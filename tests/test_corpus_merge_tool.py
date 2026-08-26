"""⚠ #312. The first test that RUNS `scripts/tools/merge_libritts_full_corpus.py`.

Until this file existed, the tool was reached by exactly two repo-wide sweeps — "the module
compiles" and "its in-repo asset paths resolve" — and by nothing that executed it. That
mattered more than usual here: **the byte-identity guard was wrong twice in three commits**
(a hardcoded `True`, then a comparison that was `True` by construction with an unreachable
`die()`), and both times the only thing that caught it was a reviewer hand-building a fixture
and firing it. A guard whose sole verifier is the review loop stops being checked the moment
the branch merges.

So the point of this file is not coverage. It is that every refusal in that tool is exercised
**in both directions** — the bad input is refused AND the good input is accepted — because a
guard that only ever sees failing input is indistinguishable from one that refuses everything,
and a guard that only ever sees passing input is indistinguishable from `return True`.

The fixture is built under `tmp_path` with directory names that the licence manifest already
declares (`libritts_r_emilia_expressive_vat_v6`, `_derived_train_clean_360`,
`libritts_r_full_vat_v7`, `LibriTTS-R`), because `classify_path` matches on any path
COMPONENT. That is not a trick to get past the wall — it is the wall working as designed on a
directory whose name says what it holds.
"""

import json
import pathlib
import subprocess

import pytest

from scripts_layout import SCRIPTS  # noqa: E402

SCRIPTS.on_path()

import derive_vat_corpus  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "scripts" / "tools" / "merge_libritts_full_corpus.py"
PY = REPO / ".venv" / "bin" / "python"

# The SAME function the tool imports, so the fixture cannot drift from the rule under test.
_in_val = derive_vat_corpus._in_val

# The VAT block is ONE pipe-field of 8 comma-separated values (3 V/A/T + the 5-wide one-hot
# delivery block), matching a real v6 row. Getting this wrong makes every row an 11-field
# parse error, which is how the tool's own field check earned its keep on the first run.
VAT = ",".join(["0.0"] * 8)


def _row(wav_dir, name, spk):
    return f"{wav_dir}/{name}.wav|{spk}|h ə l oʊ|{VAT}"


def _corpus(root, dirname, speakers, n_clips, wav_dir, tag):
    """A minimal but REAL corpus directory: rows split by the shipped hash rule.

    ⚠ The split is not hand-assigned. `_in_val` is imported from the module the tool itself
    imports, so the fixture cannot drift from the rule under test — which is the entire
    subject of the split-agreement pre-flight (#300).
    """
    d = root / dirname
    d.mkdir(parents=True, exist_ok=True)
    train, val = [], []
    for i in range(n_clips):
        for idx, sid in enumerate(speakers):
            name = f"{sid}_{tag}_{i:04d}_0000"
            r = _row(wav_dir, name, idx)
            (val if _in_val(r) else train).append(r)
    # ⚠ A corpus with an EMPTY SIDE is not a corpus, and a fixture that quietly produces one
    # makes every assertion about that side vacuously true — the empty-enumeration trap, one
    # layer out. VAL_FRACTION is 0.03, so `n_clips` has to be large enough that the hash
    # actually lands some rows in val; at 24 rows it did not, and `val_op.txt did not grow`
    # was a fixture defect wearing the costume of a tool defect. Floor both sides here so the
    # next person who shrinks the fixture finds out immediately.
    assert train, "fixture produced no train rows"
    assert val, (f"fixture produced no val rows from {n_clips * len(speakers)} clips — "
                 f"raise n_clips, or this file's val assertions are vacuous")
    for name, part in (("train_op.txt", train), ("val_op.txt", val)):
        (d / name).write_text("\n".join(part) + "\n", encoding="utf-8")
    (d / "speakers.json").write_text(json.dumps({
        "n_spks": len(speakers),
        "libritts_id_to_index": {sid: i for i, sid in enumerate(speakers)},
    }, indent=2), encoding="utf-8")
    return d, train, val


@pytest.fixture
def world(tmp_path):
    wav = tmp_path / "LibriTTS-R" / "wavs"
    wav.mkdir(parents=True)
    base, btrain, bval = _corpus(tmp_path, "libritts_r_emilia_expressive_vat_v6",
                                 ["100", "101", "102"], 60, wav, "base")
    add, _, _ = _corpus(tmp_path, "_derived_train_clean_360", ["900", "901"], 60, wav, "add")
    out = tmp_path / "libritts_r_full_vat_v7"
    return dict(tmp=tmp_path, base=base, add=add, out=out, btrain=btrain, bval=bval)


def run(*args):
    # ⚠ #322. SKIP, NOT ERROR, WHEN THE REPO VENV IS ABSENT. This shells out to
    # `.venv/bin/python` per the run-mode rule, and a checkout that has never had `uv venv`
    # run in it — every CI runner, **including the review lane's own worktree** — got a bare
    # `FileNotFoundError` and 12 of 13 ERRORs here rather than a skip. `test_gate_scripts.py`
    # already carried exactly this guard, with a comment naming that same population; I wrote
    # a second subprocess harness beside it and did not carry the lesson across. "The
    # interpreter these tools are specified to run under is not on this machine" is the same
    # class of fact as "the corpus is not on this machine", which this suite treats as a skip.
    if not PY.exists():
        pytest.skip(f"{PY} is absent — this checkout has no repo venv, so the tool cannot be "
                    f"run as specified (run-mode rule, AGENTS.md).")
    return subprocess.run([str(PY), str(TOOL), *map(str, args)],
                          capture_output=True, text=True, cwd=REPO)


def merge(w, *extra):
    return run("--base", w["base"], "--add", w["add"], "--out", w["out"], *extra)


# --------------------------------------------------------------------------- happy path

def test_merge_appends_and_reports_byte_identity(world):
    """The POSITIVE control for every refusal below: good input is accepted."""
    r = merge(world)
    assert r.returncode == 0, r.stdout + r.stderr
    out = world["out"]
    for n in ("train_op.txt", "val_op.txt", "speakers.json", "derivation_report.json"):
        assert (out / n).exists(), f"{n} missing\n{r.stdout}{r.stderr}"

    # Speakers appended, never renumbered: the base keeps indices 0..2, the add gets 3..4.
    spk = json.loads((out / "speakers.json").read_text())
    assert spk["n_spks"] == 5
    m = spk["libritts_id_to_index"]
    assert [m["100"], m["101"], m["102"]] == [0, 1, 2]
    assert sorted([m["900"], m["901"]]) == [3, 4]

    rep = json.loads((out / "derivation_report.json").read_text())
    assert all(v["identical"] for v in rep["base_rows_byte_identical"].values())


def test_byte_identity_is_a_real_prefix_of_the_written_file(world):
    """⚠ The #296 shape, twice over: a check that is True BY CONSTRUCTION.

    Both earlier versions of the guard would pass this test's happy path. What they could not
    do is what is asserted here — that the bytes on disk in `--out` genuinely OPEN with the
    bytes on disk in `--base`, compared outside the tool, by this test, from the files.
    """
    assert merge(world).returncode == 0
    for n in ("train_op.txt", "val_op.txt"):
        base_bytes = (world["base"] / n).read_bytes()
        out_bytes = (world["out"] / n).read_bytes()
        assert out_bytes[:len(base_bytes)] == base_bytes, f"{n} is not a byte-prefix"
        assert len(out_bytes) > len(base_bytes), f"{n} did not grow — nothing was appended"


# --------------------------------------------------------------------------- #296 / #311

def test_dropped_interior_row_is_caught_by_byte_identity(world):
    """The case the guard exists for, and the one the comment's first half names."""
    p = world["base"] / "train_op.txt"
    rows = p.read_text().splitlines()
    p.write_text("\n".join(rows[:-1]) + "\n\n" + rows[-1] + "\n", encoding="utf-8")
    r = merge(world)
    assert r.returncode != 0
    assert "BYTE-IDENTITY FAILED" in r.stdout + r.stderr
    # ...and it cleaned up after itself.
    assert not (world["out"] / "train_op.txt").exists()


def test_base_without_trailing_newline_is_ACCEPTED(world):
    """⚠ #311. The comment beside the guard claimed, under this repo's strongest evidential
    label, that this case "came out different on disk". It does not, and the guard is RIGHT
    to pass it: `"\\n".join(part) + "\\n"` restores the newline, so the leading bytes still
    reproduce. This test is here so the claim can never be re-added without going red —
    a false "measured" is worth a test precisely because it invites someone to "fix" correct
    code."""
    p = world["base"] / "train_op.txt"
    p.write_text(p.read_text().rstrip("\n"), encoding="utf-8")
    r = merge(world)
    assert r.returncode == 0, r.stdout + r.stderr


# --------------------------------------------------------------------------- #295 / #306

def test_out_equal_to_base_is_refused(world):
    r = run("--base", world["base"], "--add", world["add"], "--out", world["base"])
    assert r.returncode != 0
    assert "same directory as --base" in r.stdout + r.stderr


def test_out_equal_to_an_add_is_refused_even_under_force(world):
    """⚠ #306. `--force` is exactly the flag that removes the mask: an `--add` always holds a
    `train_op.txt`, so the "--out already holds a corpus" guard hid this at default settings.
    The regression to fear is someone reinstating the base-only check, which this catches
    only because `--force` is passed."""
    before = (world["add"] / "train_op.txt").read_bytes()
    r = run("--base", world["base"], "--add", world["add"], "--out", world["add"], "--force")
    assert r.returncode != 0
    assert "same directory as --add" in r.stdout + r.stderr
    assert (world["add"] / "train_op.txt").read_bytes() == before, "the input was modified"


def test_out_via_symlink_to_base_is_refused(world):
    """The realpath half. A symlink is the spelling the guard was written to survive."""
    link = world["tmp"] / "libritts_r_full_vat_v7_link"
    link.symlink_to(world["base"])
    r = run("--base", world["base"], "--add", world["add"], "--out", link)
    assert r.returncode != 0
    assert "same directory as --base" in r.stdout + r.stderr


# --------------------------------------------------------------------------- #299

def test_base_without_the_libritts_namespace_refuses_cleanly(world):
    """⚠ #299. The failure mode was a raw `KeyError` — and, worse, one that arrived AFTER a
    per-directory summary reading like the append had succeeded. Both halves are asserted:
    a named refusal, and nothing printed that claims progress."""
    p = world["base"] / "speakers.json"
    spk = json.loads(p.read_text())
    spk["expressive_id_to_index"] = spk.pop("libritts_id_to_index")
    p.write_text(json.dumps(spk), encoding="utf-8")
    r = merge(world)
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert "Traceback" not in out and "KeyError" not in out, out
    assert "has no libritts_id_to_index map" in out
    assert "speakers ->" not in out, "printed a successful-looking append summary before dying"


# --------------------------------------------------------------------------- #300

def test_a_row_on_the_wrong_side_of_the_split_is_refused(world):
    """⚠ #300. Simulates an input derived under a different SPLIT_SALT/VAL_FRACTION by moving
    one row across. Nothing in the tool or in training would otherwise notice, and the result
    is val rows sitting in train — a second silent way for the split to leak."""
    tp = world["add"] / "train_op.txt"
    vp = world["add"] / "val_op.txt"
    rows = tp.read_text().splitlines()
    moved, rest = rows[0], rows[1:]
    tp.write_text("\n".join(rest) + "\n", encoding="utf-8")
    vp.write_text(vp.read_text() + moved + "\n", encoding="utf-8")
    r = merge(world)
    assert r.returncode != 0
    assert "land on the other side under the shared hash split" in r.stdout + r.stderr


def test_a_base_row_on_the_wrong_side_of_the_split_is_refused(world):
    """⚠ THE BASE SIDE, WHICH HAD NO COVERAGE AT ALL.

    `check_split_agrees` is called four times — twice on the base, twice per `--add` — and
    the test above only ever mutated an `--add`. So deleting both base-side calls left the
    suite green, and the base is the half that carries 41,937 already-shipped rows: a base
    whose split rule had moved would pool two rules into v7 and put val rows in train, which
    is the whole defect #300 exists to prevent. Found by an adversarial pass on my own fix,
    2026-08-26 — the mutation battery for #300 had one arm and looked like two.
    """
    tp = world["base"] / "train_op.txt"
    vp = world["base"] / "val_op.txt"
    rows = tp.read_text().splitlines()
    moved, rest = rows[0], rows[1:]
    tp.write_text("\n".join(rest) + "\n", encoding="utf-8")
    vp.write_text(vp.read_text() + moved + "\n", encoding="utf-8")
    r = merge(world)
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "land on the other side under the shared hash split" in out
    assert str(world["base"]) in out, "the refusal does not say WHICH input was wrong"
    assert not (world["out"] / "train_op.txt").exists(), "wrote output despite refusing"


def test_the_split_preflight_passes_on_an_untouched_fixture(world):
    """POSITIVE CONTROL — otherwise a pre-flight that refused everything would satisfy both
    tests above.

    ⚠ This asserts rc 0 AND that the base's row counts survived the merge intact. It used to
    assert only that the "safe to grow" line was printed, which is NOT a control for the
    check: that print sits after `check_split_agrees` returns, so gutting the check to
    `wrong = []` left the line — and this test — green. An assertion about a print is an
    assertion about a print.
    """
    r = merge(world)
    assert r.returncode == 0
    assert "hash split reproduces the base's own train/val exactly" in r.stdout
    # The real control: the base's own rows are still classified correctly on the way out.
    for name, side in (("train_op.txt", False), ("val_op.txt", True)):
        for row in (world["out"] / name).read_text().splitlines():
            assert _in_val(row) == side, f"{name}: merged output disagrees with the hash rule"


# --------------------------------------------------------------------------- #307

def test_a_late_refusal_under_force_leaves_no_stale_report(world, tmp_path):
    """⚠ #307. `speakers.json` and `derivation_report.json` are written LAST, so on a fresh
    `--out` a late refusal never created them. Under `--force` over a POPULATED `--out` they
    belonged to the PREVIOUS corpus and survived the cleanup — leaving a directory whose
    report described rows that were no longer on disk, under a message reading "left with no
    corpus". The stale report is the defect; a clean failure or a clean corpus are both fine.
    """
    assert merge(world).returncode == 0                       # populate --out for real
    assert (world["out"] / "derivation_report.json").exists()

    # Now make the licence wall refuse at the end: rows pointing at an undeclared audio root.
    # `classify_path` passes the DIRECTORY (its name is declared) and `enforce` refuses the
    # ROWS — which is exactly the late-failure window this issue is about.
    bad = tmp_path / "SomeUndeclaredCorpus" / "wavs"
    bad.mkdir(parents=True)
    tp = world["add"] / "train_op.txt"
    tp.write_text("\n".join(r.replace(str(world["tmp"] / "LibriTTS-R" / "wavs"), str(bad))
                            for r in tp.read_text().splitlines()) + "\n", encoding="utf-8")

    r = merge(world, "--force")
    assert r.returncode != 0
    assert "License wall" in r.stdout + r.stderr
    leftovers = [n for n in ("train_op.txt", "val_op.txt", "speakers.json",
                             "derivation_report.json")
                 if (world["out"] / n).exists()]
    assert not leftovers, f"stale corpus artifacts survived a refusal: {leftovers}"


def test_an_early_abort_under_force_leaves_the_previous_corpus_INTACT(world):
    """⚠ #325. THE THIRD ABORT CASE, and the one two successive docstrings denied.

    Every pre-flight refusal — the licence classify, the base/`--add` structure checks, the
    split agreement — fires before a single byte is written. So over a populated `--out` under
    `--force`, an early abort leaves the previous corpus exactly where it was: intact and
    byte-identical. That is the BEST of the three outcomes, which is precisely why an absolute
    ("writes nothing", then "leaves no corpus behind") kept being written over it — both were
    generalised from the one path the author happened to be looking at.

    Asserted here so the invariant in the module docstring is checked rather than described.
    """
    assert merge(world).returncode == 0
    before = {n: (world["out"] / n).read_bytes()
              for n in ("train_op.txt", "val_op.txt", "speakers.json",
                        "derivation_report.json")}

    # An EARLY refusal: move a base row across the split, which the pre-flight catches before
    # any write. (The late-refusal counterpart is the licence test above.)
    tp = world["base"] / "train_op.txt"
    vp = world["base"] / "val_op.txt"
    rows = tp.read_text().splitlines()
    tp.write_text("\n".join(rows[1:]) + "\n", encoding="utf-8")
    vp.write_text(vp.read_text() + rows[0] + "\n", encoding="utf-8")

    r = merge(world, "--force")
    assert r.returncode != 0
    assert "land on the other side under the shared hash split" in r.stdout + r.stderr
    for n, blob in before.items():
        assert (world["out"] / n).exists(), f"{n} was destroyed by an abort that never wrote"
        assert (world["out"] / n).read_bytes() == blob, f"{n} changed despite an early abort"


def test_populated_out_without_force_is_refused(world):
    assert merge(world).returncode == 0
    r = merge(world)
    assert r.returncode != 0
    assert "already holds" in r.stdout + r.stderr
