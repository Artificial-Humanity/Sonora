"""⚠ #312. The first test that RUNS `scripts/tools/merge_libritts_full_corpus.py`.

Until this file existed, the tool was reached by exactly two repo-wide sweeps — "the module
compiles" and "its in-repo asset paths resolve" — and by nothing that executed it. That
mattered more than usual here: **the byte-identity guard was wrong twice in three commits**
(a hardcoded `True`, then a comparison that was `True` by construction with an unreachable
`die()`), and both times the only thing that caught it was a reviewer hand-building a fixture
and firing it. A guard whose sole verifier is the review loop stops being checked the moment
the branch merges.

The standard this file holds itself to is that a refusal it covers is exercised **in both
directions** — the bad input is refused AND the good input is accepted — because a guard that
only ever sees failing input is indistinguishable from one that refuses everything, and a
guard that only ever sees passing input is indistinguishable from `return True`.

⚠⚠ **THIS PARAGRAPH USED TO SAY "EVERY REFUSAL IN THAT TOOL", AND THAT WAS FALSE — MEASURED
8 OF 25 (#335).** It was written to retire #312, and it overclaimed in the direction that
stops anyone checking: a maintainer meets it first, concludes the refusal surface is covered,
and hand-checks nothing. What went unwatched was the tool's own strongest claim about itself —
the speaker-collision check, which its comment calls *THE CHECK THAT MAKES THE APPEND LEGAL*.
A mutation battery with a working control measured `clash = sorted(set(local) & known)` ->
`clash = []` as **14 passed**. That is now covered, and mutating it fails 2 tests.

⚠ **So the count is in a LEDGER at the bottom of this file rather than in this sentence.**
Prose asserting its own coverage is what failed; `test_every_refusal_in_the_tool_is_classified`
enumerates the tool's `die()` sites by AST and refuses to pass while any of them is unlisted.
It does not claim to VERIFY coverage — a matcher cannot, and two attempts at one proved it in
both directions (`speakers.json` matched as a refusal phrase when it is a filename; the
namespace assertion was missed because the asserted text spans a `%s`). What it guarantees is
that the number cannot drift silently again: add a refusal to the tool and this file goes red
until someone writes down which kind it is.

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


# ------------------------------------------------------- #335 — the speaker-collision guard
#
# ⚠⚠ THE TOOL'S STRONGEST CLAIM ABOUT ITSELF WAS WATCHED BY NOTHING. Its own comment calls
# the clash check "THE CHECK THAT MAKES THE APPEND LEGAL", and a mutation battery with a
# working control measured `clash = sorted(set(local) & known)` -> `clash = []` as **14
# passed** against the shipped suite. Three other refusals mutated green the same way.
#
# ⚠ AND THE FAILURE IS SILENT IN EVERY OTHER INSTRUMENT, which is why it earned a test rather
# than a note. Re-adding a speaker leaves v6's rows untouched, so byte-identity passes; it
# appends a fresh dense index, so contiguity passes; it increments the counter in step, so
# `n_spks` passes. Nothing goes red. It surfaces only as one person holding two rows of
# `spk_emb.weight`, each trained on half their audio — a voice that is quietly worse.

def test_a_speaker_already_in_the_base_is_refused(world):
    """The base/add collision. `101` is in the fixture base; a donor carrying it is refused.

    ⚠ THE DONOR NAMES HERE ARE THE REAL v7 ONES (`_derived_train_clean_360`,
    `_derived_train_other_500`), and that is not decoration. The first draft invented
    `_derived_other_500`, and the licence pre-flight refused it before the clash check was
    ever reached — so the test went RED while asserting a non-zero exit that it did get, for
    entirely the wrong reason. It was caught only because these assertions check the refusal
    MESSAGE. An exit-code-only test would have passed here and watched nothing.
    """
    collide, _, _ = _corpus(world["tmp"], "_derived_train_other_500", ["101", "902"], 60,
                            world["tmp"] / "LibriTTS-R" / "wavs", "clash")
    r = run("--base", world["base"], "--add", collide, "--out", world["out"])
    assert r.returncode != 0, "a speaker already in the base was APPENDED — one person now "\
                              "holds two embedding rows:\n" + r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "ALREADY in the base" in out, out
    assert "101" in out, "the refusal must name the colliding id, or it cannot be acted on"
    # ⚠ "Nothing written" is part of the refusal's own promise, so it is asserted, not assumed.
    assert not world["out"].exists(), "a refused merge left an output directory behind"


def test_a_speaker_shared_between_two_ADDS_is_refused(world):
    """⚠ THE SECOND COLLISION, AND IT IS NOT THE SAME TEST. The base/add case is caught by
    the initial `known`; this one is caught only because `known.add(sid)` accumulates ACROSS
    `--add` directories inside the loop. A rewrite that built `known` once from the base — an
    entirely natural-looking simplification — would keep the test above green and lose this.
    """
    wav = world["tmp"] / "LibriTTS-R" / "wavs"
    # `world["add"]` is `_derived_train_clean_360` holding 900/901; this donor re-uses 901.
    # Neither collides with the BASE, so only the accumulating `known` can catch it.
    b, _, _ = _corpus(world["tmp"], "_derived_train_other_500", ["901", "902"], 60, wav, "b")
    r = run("--base", world["base"], "--add", world["add"], "--add", b, "--out", world["out"])
    assert r.returncode != 0, "two donors shared speaker 901 and the merge was ACCEPTED:\n" \
                              + r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "ALREADY in the base" in out, out
    assert "901" in out, out


def test_the_other_direction_every_speaker_gets_exactly_one_dense_index(world):
    """⚠ THE POSITIVE HALF, without which the two refusals above are indistinguishable from a
    guard that refuses everything — this file's own stated standard.

    It asserts the PROPERTY the guard exists to protect rather than just an exit code: the
    merged map holds every speaker from both inputs, once each, over a contiguous index space
    starting at 0. A duplicated append would show up here as a count mismatch even if some
    future refusal string changed.
    """
    wav = world["tmp"] / "LibriTTS-R" / "wavs"
    extra, _, _ = _corpus(world["tmp"], "_derived_train_other_500", ["902", "903"], 60, wav, "x")
    r = run("--base", world["base"], "--add", world["add"], "--add", extra,
            "--out", world["out"])
    assert r.returncode == 0, r.stdout + r.stderr

    spk = json.loads((world["out"] / "speakers.json").read_text())
    m = spk["libritts_id_to_index"]
    expected = ["100", "101", "102", "900", "901", "902", "903"]
    assert sorted(m) == expected, f"speaker set changed across the merge: {sorted(m)}"
    assert len(set(m.values())) == len(m), f"a speaker id shares an index with another: {m}"
    assert sorted(m.values()) == list(range(len(expected))), \
        f"index space is not dense from 0: {sorted(m.values())}"
    assert spk["n_spks"] == len(expected), f"n_spks={spk['n_spks']} vs {len(expected)} speakers"


def test_the_licence_preflight_refuses_an_undeclared_donor(world):
    """⚠ COVERED BECAUSE IT CAUGHT ME, not because it was next on a list. The first draft of
    the collision tests above invented a donor directory name (`_derived_other_500`), and this
    refusal fired first — so the test went red while asserting a non-zero exit it *did* get,
    for entirely the wrong reason. It survived only because those assertions check the refusal
    message. Recorded here so the pre-flight ordering is pinned rather than rediscovered.
    """
    d, _, _ = _corpus(world["tmp"], "undeclared_donor_dir", ["900", "901"], 60,
                      world["tmp"] / "LibriTTS-R" / "wavs", "u")
    r = run("--base", world["base"], "--add", d, "--out", world["out"])
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "matches no declared dataset" in out, out
    assert not world["out"].exists(), "a licence refusal is a PRE-flight; it must write nothing"


# ------------------------------------------------------------------- #335 — the refusal ledger
#
# ⚠⚠ WHAT THIS IS FOR: the module docstring above claimed every refusal was exercised, and it
# was 8 of 25. A false coverage claim is worse than no claim, because it is the first thing a
# maintainer reads and it tells them to stop looking. The fix is not a better sentence — it is
# that the sentence can no longer drift away from the tool without something going red.
#
# ⚠ WHAT THIS IS *NOT*: a coverage verifier. Two attempts at matching refusal text against
# assertions failed in BOTH directions, which is why this is a hand-classified list:
#   * FALSE POSITIVE — `%s has no speakers.json` "matched" because the tests mention
#     `speakers.json` as a FILENAME, which is not an assertion about that refusal at all;
#   * FALSE NEGATIVE — `test_base_without_the_libritts_namespace_refuses_cleanly` asserts
#     "has no libritts_id_to_index map", which spans a `%s` in the template, so no matcher
#     comparing against source text can see it.
# A guard that reports coverage it cannot see would be #312's defect wearing #335's costume.
# This one asserts CLASSIFICATION, which it can check exactly.

# Refusals with a test that fires them. The value names it, so a claim here is falsifiable by
# reading one function rather than trusting this comment.
COVERED_REFUSALS = {
    "rows land on the other side under the shared hash split":
        "test_a_row_on_the_wrong_side_of_the_split_is_refused (+ the _base_ variant)",
    "--out is the same directory as --base":
        "test_out_equal_to_base_is_refused, test_out_via_symlink_to_base_is_refused",
    "--out is the same directory as --add":
        "test_out_equal_to_an_add_is_refused_even_under_force",
    "already holds": "test_populated_out_without_force_is_refused",
    "matches no declared dataset": "test_the_licence_preflight_refuses_an_undeclared_donor",
    "base %s has no %s map": "test_base_without_the_libritts_namespace_refuses_cleanly",
    "are ALREADY in the base":
        "test_a_speaker_already_in_the_base_is_refused, test_a_speaker_shared_between_two_ADDS_is_refused",
    "BYTE-IDENTITY FAILED": "test_dropped_interior_row_is_caught_by_byte_identity",
}

# Refusals with NO test, written down so the gap is a fact on the record rather than an
# absence nobody can see. ⚠ Being listed here is not permission — it is a queue.
UNCOVERED_REFUSALS = {
    "is it a corpus directory": "base/add missing train_op/val_op",
    "has no speakers.json": "base/add missing the map file",
    "two speakers on index": "donor's own map is already broken",
    "nothing to add": "no --add passed",
    "is NON-COMMERCIAL": "the NC licence branch — the other half of the wall",
    "base n_spks=%s but its maps hold": "base counter disagrees with its map",
    "base index space is not contiguous": "base has a hole",
    "%s has no %s map — this script appends LibriTTS-lane": "the --add namespace check; #335 "
        "notes the base assertion would match this string too, but only the base path runs",
    "carries more than one namespace": "multi-namespace donor",
    "row has %d fields": "row parse — field count",
    "speaker field %r is not an index": "row parse — non-integer speaker",
    "which its own speakers.json does not define": "row parse — undefined index",
    "PREFIX PROOF FAILED": "#336 — filed as unreachable, so a test needs the reachability "
        "question answered first",
    "merged map holds %d indices but the counter says": "post-merge counter disagreement",
    "merged index space is not contiguous": "post-merge hole",
    "merged corpus has an EMPTY val split": "post-merge empty side",
    "rows disagree about the VAT vector width": "post-merge VAT width",
    "would NOT be a legal warm-start donor": "the byte-identity refusal's second message",
}


def _die_messages():
    """Every `die(...)` message in the tool, by AST — not by regex over the text.

    ⚠ #335 counted 25 by regex and had to subtract three: the `def die`, one occurrence inside
    a COMMENT, and a multi-line duplicate. An AST walk counts call sites and cannot make any
    of those three mistakes. It also joins EVERY literal chunk of the message, because the
    first version read only the first chunk and lost any phrase after a `%s`.
    """
    import ast
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "die" and n.args:
            lits = [c.value for c in ast.walk(n.args[0])
                    if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            out.append((n.lineno, " ".join(" ".join(lits).split())))
    return out


def test_every_refusal_in_the_tool_is_classified():
    """The ledger's invariant: no `die()` may be neither covered nor recorded as uncovered."""
    messages = _die_messages()

    # ⚠ FLOOR FIRST. If the AST walk stopped finding call sites, every assertion below would
    # pass over an empty list — the vacuous pass, in the file whose subject is a false claim
    # of coverage.
    assert len(messages) >= 20, (
        f"only {len(messages)} die() call sites found in {TOOL.name}; the walk is broken, "
        f"not the tool")

    known = {**COVERED_REFUSALS, **UNCOVERED_REFUSALS}
    unclassified = [(ln, m) for ln, m in messages
                    if not any(k in m for k in known)]
    assert not unclassified, (
        "refusal(s) in the tool that this file has never classified:\n"
        + "\n".join(f"  L{ln}: {m[:88]}" for ln, m in unclassified)
        + "\n\n⚠ A new refusal is not a problem; an UNRECORDED one is. Add it to "
          "COVERED_REFUSALS with the test that fires it, or to UNCOVERED_REFUSALS with what "
          "it guards. #335 was exactly this drift: the docstring said every refusal was "
          "exercised while 17 of 25 were not, and the claim is what stopped anyone checking.")

    # ⚠ AND THE OTHER DIRECTION, which is the one that rots quietly. A key left behind after
    # its refusal is deleted or reworded makes this ledger describe a tool that no longer
    # exists — and it would keep passing forever, since the check above only looks for
    # unclassified messages.
    joined = "\n".join(m for _, m in messages)
    stale = [k for k in known if k not in joined]
    assert not stale, (
        "ledger key(s) matching no refusal in the tool any more:\n"
        + "\n".join(f"  {k!r}  ({known[k]})" for k in stale)
        + "\n\nThe refusal was reworded or removed. Update the key, or drop the entry.")
