"""`scripts/lib/schemas.py` — the validated loaders, and the degrade they exist to stop.

The defect that motivated this module is worth stating precisely, because it is the shape
this repo keeps paying for rather than a parsing bug:

    `book_ingest._lexicon()` was `except Exception: return []`. An unreadable, renamed or
    malformed `register_lexicon.json` produced an empty list, and then TWO things failed in
    the same direction with nothing on stdout:

      * the director's prompt said "`register` MUST be copied EXACTLY from this controlled
        lexicon:" followed by nothing at all; and
      * the off-lexicon guard read `if REGISTER_LEXICON and ...`, so the empty list SWITCHED
        THAT GUARD OFF — every invented register passed through unreported.

    The one circumstance that emptied the vocabulary was also the one that disabled the check
    on the consequence. A green run and a correct run were indistinguishable.

These tests pin the refusal, not the wording.
"""

import io
import json
import tokenize
from pathlib import Path

import pytest

from scripts_layout import SCRIPTS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# ⚠ `SCRIPTS.on_path()`, NOT a hand-rolled `sys.path.insert`. `test_stage_coverage.py`
# forbids the latter, and for a measured reason: the insert mutates the PROCESS-GLOBAL path,
# so a module that inserts a bucket leaves it there for every module collected after it. A
# test that only passes because an alphabetically earlier file happened to run first is the
# failure #26 step 3 actually hit.
SCRIPTS.on_path()
import schemas  # noqa: E402


def code_only(path):
    """Source with comments AND string literals removed — so a test reads code, not prose.

    ⚠ WRITTEN AFTER FAILING FOR EXACTLY THIS REASON. The first version stripped `#` comments
    only, and `book_ingest._lexicon()`'s docstring QUOTES the old guard (`if REGISTER_LEXICON
    and ...`) in order to explain the defect. The test asserting that guard's absence found
    the sentence describing its absence and failed against correct code. Docstrings are
    strings, not comments, and a line-based strip cannot see them.
    """
    # ⚠ F-STRINGS ARE NOT `STRING` TOKENS ON PYTHON 3.12+. They tokenize as FSTRING_START /
    # FSTRING_MIDDLE / FSTRING_END, so a strip that names only `STRING` leaves every
    # f-string's TEXT in the "code" — which failed a test against correct code, because a
    # director prompt built with an f-string mentions the very identifiers being asserted
    # absent. Named defensively: these members do not exist on older interpreters.
    drop = {tokenize.COMMENT, tokenize.STRING}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        if hasattr(tokenize, name):
            drop.add(getattr(tokenize, name))
    out = []
    src = Path(path).read_text(encoding="utf-8")
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in drop:
            continue
        out.append(tok.string)
    # ⚠ Joined with SPACES, not newlines. Newlines make every multi-token phrase
    # (`not in REGISTER_LEXICON`) unsearchable, which failed against correct code.
    return " ".join(out)


# --- the register lexicon ----------------------------------------------------------------

def test_the_real_asset_loads():
    lex = schemas.load_register_lexicon()
    assert len(lex) > 0 and lex == sorted(lex) == sorted(set(lex))


@pytest.mark.parametrize("bad,why", [
    ({"lexicon": []}, "empty"),
    ({"lexicon": ["b", "a"]}, "unsorted"),
    ({"lexicon": ["a", "a"]}, "duplicated"),
    ({"lexicon": ["Not Snake"]}, "not snake_case"),
    ({}, "missing the key entirely"),
])
def test_a_broken_lexicon_is_refused(tmp_path, bad, why):
    """⚠ `SchemaError`, NOT `Exception` — and the wide spelling is what hid issue #94.

    Pydantic converts a `ValueError` raised in a validator into `ValidationError`, so every
    invariant here surfaced as a type the module never promised. `pytest.raises(Exception)`
    passed either way, and the message assertion passed too because pydantic preserves the
    text. The empty case — the one the validator calls the one that matters — was the worst
    of them. Pinning the type is the whole assertion.
    """
    p = tmp_path / "register_lexicon.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(schemas.SchemaError) as e:
        schemas.load_register_lexicon(str(p))
    assert "lexicon" in str(e.value).lower(), why


def test_the_empty_case_says_why_it_matters(tmp_path):
    """⚠ The message has to explain that empty is not "weaker" but "absent".

    A reader who thinks an empty vocabulary is a degraded vocabulary will restore the
    `except: return []` the first time this refusal is inconvenient.
    """
    # Through the LOADER, because that is the boundary that converts (#94). Constructing the
    # model directly still raises pydantic's own type, and asserting on that would pin the
    # very behaviour the fix routes around.
    with pytest.raises(schemas.SchemaError) as e:
        _lex = tmp_path / "register_lexicon.json"
        _lex.write_text(json.dumps({"lexicon": []}), encoding="utf-8")
        schemas.load_register_lexicon(str(_lex))
    msg = str(e.value)
    assert "guard" in msg and "prompt" in msg


def test_missing_and_malformed_files_are_named(tmp_path):
    """`FileNotFoundError` and `JSONDecodeError` both say WHICH file and what it was for —
    the bare `json.load(...)["lexicon"]` this replaced said neither."""
    with pytest.raises(schemas.SchemaError) as e:
        schemas.load_register_lexicon(str(tmp_path / "nope.json"))
    assert "nope.json" in str(e.value)

    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(schemas.SchemaError) as e:
        schemas.load_register_lexicon(str(p))
    assert "bad.json" in str(e.value)


# --- the delivery lane split: NOT THIS MODULE'S JOB --------------------------------------
#
# ⚠ Three tests here drove `schemas.validate_delivery_label`, which was REMOVED (issue #95).
# `matcha.delivery.check_assignable` already owns that rule and is wired at the real write
# sites; a second implementation in `schemas.py` was the "second literal" its own docstring
# warns about. The rule is still tested — by `tests/test_delivery_channel.py` against the
# function that production actually calls, which is where a test of it belongs.


# --- no second copy of any vocabulary -----------------------------------------------------

def test_schemas_does_not_re_spell_any_controlled_vocabulary():
    """⚠ THE FILE MOST LIKELY TO BECOME THE SECOND LITERAL IS THIS ONE.

    Issue #28 was one word — "unknown" — spelled twice from two independent literals. A
    module whose job is validating vocabularies is exactly where a convenience copy gets
    pasted, and the copy would drift silently because both sides would still "work".
    ⚠ `matcha/delivery.py`'s ORDER IS THE WIRE FORMAT, so a copy that drifted by one entry
    would reinterpret every filelist and every checkpoint written at that `vat_dim`.
    """
    from matcha import delivery
    src = (REPO / "scripts" / "lib" / "schemas.py").read_text(encoding="utf-8")
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    for lane in delivery.DELIVERY_LANES:
        assert f'"{lane}"' not in code and f"'{lane}'" not in code, \
            f"schemas.py spells the lane {lane!r} as a literal instead of importing it"


def test_the_adopting_readers_no_longer_swallow_failures():
    """Both production readers must go through the validated loader.

    ⚠ Checked as CODE. `book_ingest`'s docstring quotes the old `except Exception: return []`
    to explain the defect, so a naive text search finds the very string it is asserting the
    absence of — the same substring-over-the-whole-file mistake this suite has caught twice
    elsewhere today.
    """
    import ast

    # ⚠ Read as a TREE, not as text. With string literals stripped the old pattern
    # (`json.load(...)["lexicon"]`) is unsearchable, and with them kept the docstrings
    # quoting it match. Only the AST distinguishes "the code does this" from "the prose
    # mentions this".
    def loader_calls(path):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        return {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}

    for name in ("book_ingest.py", "make_teacher_ab_bank.py"):
        calls = loader_calls(REPO / "scripts" / "lib" / name)
        assert "schemas.load_register_lexicon" in calls, \
            f"{name} does not go through the validated loader"


def test_the_off_lexicon_guard_is_no_longer_short_circuited():
    """⚠ `if REGISTER_LEXICON and ...` read as defensiveness and WAS the off switch: the only
    thing that could empty the lexicon was the loader's swallow, so the one circumstance that
    emptied the vocabulary also disabled the check on the consequence."""
    code = code_only(REPO / "scripts" / "lib" / "book_ingest.py")
    assert "REGISTER_LEXICON and" not in code, "the short-circuit is back"
    assert "not in REGISTER_LEXICON" in code, "the off-lexicon check is gone entirely"


def test_no_caller_wraps_the_validated_loader_in_a_swallow():
    """⚠⚠ THE TEST THE MUTATION BATTERY SAID WAS MISSING, AND IT WAS THE IMPORTANT ONE.

    Re-introducing the exact original defect —

        try:
            return schemas.load_register_lexicon()
        except Exception:
            return []

    — passed all fourteen tests written before this one. Every one of them checked that the
    validated loader is CALLED, and wrapping a call still calls it. The refusal was intact and
    the swallow was back on top of it, which is precisely the arrangement the module exists to
    remove: `load_register_lexicon` raising into a bare `except` is no better than the
    `json.load` it replaced, and reads as more careful.

    So this asserts on STRUCTURE: no call to a `schemas.load_*` loader may sit inside a `try`.
    ⚠ That is deliberately stricter than "no bare except". A narrow
    `except SchemaError: return []` degrades exactly as silently, and the whole point is that
    an empty vocabulary is not a degraded vocabulary — it is the absence of the mechanism.
    """
    import ast

    for name in ("book_ingest.py", "make_teacher_ab_bank.py"):
        tree = ast.parse((REPO / "scripts" / "lib" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    fn = ast.unparse(inner.func)
                    assert not fn.startswith("schemas.load_"), (
                        f"{name}: {fn}() is inside a try/except. A validated loader that is "
                        f"caught is a swallow with extra steps — let it raise."
                    )


def test_no_source_directory_would_swallow_a_new_file():
    """⚠⚠ `.gitignore` HAD A BARE `lib/`, AND IT ATE NEW SOURCE FILES WITHOUT A WORD.

    It comes from the standard Python template, where it means the top-level setuptools build
    directory. Unanchored it also matched `scripts/lib/` — ten tracked modules — so every NEW
    file added there was skipped by `git add -A` with no message, no `git status` entry and no
    failure. Existing files stayed tracked, which is exactly what kept it hidden: the rule only
    ever ate the NEXT one.

    Measured twice in one day: `scripts/lib/find_changeset.py` was written, imported, reviewed
    and deleted having existed in no commit; `scripts/lib/schemas.py` then vanished from a
    guard that enumerates via `git ls-files`, which read as the guard being broken.

    ⚠ THIS PROBES A HYPOTHETICAL NEW FILE, and the first version did not — it walked the files
    that exist, every one of which is already tracked, and `git check-ignore` reports nothing
    for a tracked path unless `--no-index` is given. So it passed against the reinstated bare
    `lib/`: a test for "new files get swallowed" that could only ever see old ones.
    """
    import subprocess

    roots = sorted({str(p.parent.relative_to(REPO))
                    for d in ("scripts", "tests", "matcha", "configs", "workflow")
                    if (REPO / d).is_dir()
                    for p in (REPO / d).rglob("*.py")
                    if "__pycache__" not in p.parts})
    roots += [d for d in ("scripts", "tests", "matcha", "workflow") if (REPO / d).is_dir()]
    assert roots, "found no source directories — is the walk working?"

    probes = [f"{r}/_probe_new_file.py" for r in sorted(set(roots))]
    proc = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"],
                          cwd=REPO, text=True, input="\n".join(probes), capture_output=True)
    swallowed = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert not swallowed, (
        "a NEW file added to these source directories would be silently skipped by "
        "`git add`:\n  " + "\n  ".join(swallowed[:20])
    )


# --- intended V/A/T, validated at the WRITER ----------------------------------------------

def test_an_absent_axis_stays_absent_and_is_not_neutralised():
    """⚠ `tag.get("valence", 0.0)` TURNED SILENCE INTO A NEUTRAL LABEL.

    Both writers in `book_ingest` defaulted a missing axis to 0.0, so "the director said
    nothing" and "the director said neutral" arrived identical. `qc_verdict.intended_labels`
    goes to real trouble to keep those apart — "an absent axis ... is not collected, because
    there is nothing wrong with it" — and the writer's default made that branch UNREACHABLE
    from this path: every row arrived with all three axes present and numeric.
    """
    out = schemas.intended_vat({"valence": 0.5})
    assert out["V"] == 0.5
    assert out["A"] is None and out["T"] is None, "a missing axis must not become 0.0"


def test_a_numeric_string_is_a_label_and_a_word_is_not():
    """Issue #58's rule, now applied where the value is WRITTEN rather than where it is read."""
    assert schemas.intended_vat({"valence": "0.7"})["V"] == 0.7
    with pytest.raises(schemas.SchemaError):
        schemas.intended_vat({"valence": "very sad"})


def test_out_of_range_is_refused_because_it_would_be_clamped_silently():
    with pytest.raises(schemas.SchemaError) as e:
        schemas.intended_vat({"valence": 3.0})
    assert "clamped" in str(e.value)


def test_lenient_mode_records_absence_rather_than_failing():
    """The casting retry loop must not abort a whole run on one bad axis; the manifest write
    is where it becomes fatal."""
    assert schemas.intended_vat({"valence": "very sad"}, strict=False)["V"] is None


def test_the_coercion_rule_has_exactly_one_definition():
    """⚠ `qc_verdict.coerce_axis` must DELEGATE, not keep a second copy.

    Two implementations of "what counts as a number" drift invisibly: both return floats for
    every easy case, so only an odd input reveals the disagreement, and by then it is in a
    manifest. Checked structurally — the function body must be a single call.
    """
    import ast

    tree = ast.parse((REPO / "scripts" / "stages" / "qc_verdict.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "coerce_axis")
    body = [n for n in fn.body if not (isinstance(n, ast.Expr)
                                       and isinstance(n.value, ast.Constant))]  # drop docstring
    assert len(body) == 1 and isinstance(body[0], ast.Return), \
        "qc_verdict.coerce_axis has grown its own logic again"
    assert ast.unparse(body[0].value).startswith("schemas.coerce_axis")


def test_no_module_keeps_its_own_idea_of_what_counts_as_a_number():
    """⚠ THE SAME RULE AS ABOVE, BUT NOT SCOPED TO ONE FILE — which is why it kept happening.

    `test_the_coercion_rule_has_exactly_one_definition` pins `qc_verdict.coerce_axis` and is
    blind to every other module, so it passed while a THIRD copy went into `ref_select._n`
    (#104) and stayed passing while a FOURTH sat in `make_v3d_bank.complete_vat` (#113) —
    eight lines above a comment congratulating that same commit for removing this exact
    class of drift.

    The shape is always `isinstance(x, (int, float))`, and it is always wrong in the same
    direction: it reads the numeric string `"0.7"` as not-a-number, which is the one input
    issue #58 ruled on. `schemas.coerce_axis` is the definition; everything else asks it.

    ⚠ AST, not a text scan — three of the comments explaining this defect quote the offending
    expression verbatim, and a text scan matches its own explanation.
    """
    import ast

    # ⚠ EXEMPTIONS ARE PER SITE AND CARRY A REASON, because the syntactic shape is shared by
    # checks that are NOT this rule. **Measured at `1d9c184^`, over every file this scan
    # reads: eleven hits, FIVE of them axes** (`schemas.py:198`, `schemas.py:282`,
    # `scm.py:26`, `make_narration_bank.py:216`, `make_v3d_bank.py:78`).
    #
    # ⚠ ONE SCOPING, STATED WHOLE. The first version of this comment said "six hits, one
    # axis" — from a truncated first look at the output. The correction said "eleven hits,
    # three axes", which pairs a count that INCLUDES `schemas.py` with a count that
    # EXCLUDES it, and contradicts the note four lines below establishing `fmt_axis` as a
    # second axis reader in that same file. Nine and three is the other consistent pair,
    # for the scoping that skips the file. Issue #119, twice: the first correction was
    # made by re-running the instrument and still mis-stated, because I re-ran it under
    # two scopings and quoted one number from each.
    #
    # A blanket ban would have forced durations and
    # engine parameters through a function about V/A/T, which is a worse error than the one
    # being fixed. Explicit so that adding to it is a decision rather than a slip, the same
    # shape `RECURSIVE_BY_DESIGN` and the doc-claims exemptions use.
    #
    # ⚠ NO FILE-LEVEL ENTRY FOR `schemas.py` (issue #118). It had one, and the file is not
    # single-purpose: `fmt_axis` was a SECOND reader inside it, disagreeing with the
    # definition on `"0.7"`, invisible to this scan, and beyond the staleness assertion
    # below — which only covers entries carrying a line. The guard was switched off for
    # precisely the file most likely to grow the next copy. `fmt_axis` now delegates, so
    # only the definition itself needs naming, and it is named BY LINE like everything else.
    EXEMPT = {
        # the single definition itself
        ("scripts/lib/schemas.py", 198): "coerce_axis: the definition",
        # NOT axes: an 8-float zonos emotion vector, and engine parameter ranges
        ("scripts/lib/book_ingest.py", 1036): "_in_range: engine params (pitch_std, cfg_weight)",
        ("scripts/lib/book_ingest.py", 1046): "zonos emotion vector element, not an axis",
        ("scripts/lib/book_ingest.py", 1050): "_clamp: generic engine-param clamp",
        ("scripts/lib/book_ingest.py", 1069): "_l1: zonos emotion vector, not an axis",
        # NOT an axis: a duration, with its own documented None-vs-0.0 reasoning
        ("scripts/lib/synth_common.py", 598): "book duration in seconds, not an axis",
        # NOT an axis: an ASR word error rate, guarded so a missing measure does not
        # TypeError the whole registration
        ("scripts/stages/register_audition.py", 405): "asr_wer, a measure not a label",
    }
    # Kept as a mechanism rather than deleted: a genuinely single-purpose file could earn
    # one. Nothing does today, and `schemas.py` proved why the bar is high.
    EXEMPT_FILES = {f for f, ln in EXEMPT if ln is None}

    offenders = []
    seen = set()
    for path in sorted((REPO / "scripts").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel in EXEMPT_FILES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "isinstance" and len(node.args) == 2):
                continue
            types = node.args[1]
            names = {e.id for e in getattr(types, "elts", [])
                     if isinstance(e, ast.Name)} or (
                        {types.id} if isinstance(types, ast.Name) else set())
            if not {"int", "float"} <= names:
                continue
            if (rel, node.lineno) in EXEMPT:
                seen.add((rel, node.lineno))
                continue
            offenders.append(f"  {rel}:{node.lineno}  {ast.unparse(node)}")

    # ⚠ AN EXEMPTION WHOSE SITE HAS MOVED IS A HIDING PLACE WAITING FOR A NEW TENANT. Line
    # numbers drift, so a stale entry silently exempts whatever now sits on that line.
    stale = {k for k in EXEMPT if k[1] is not None} - seen
    assert not stale, (
        f"these exemptions no longer point at an `isinstance(x, (int, float))`: "
        f"{sorted(stale)}. Re-point or delete them — an exemption that matches nothing "
        f"exempts nothing, and one that has drifted exempts the wrong thing.")

    assert not offenders, (
        "these read a number by hand instead of asking `schemas.coerce_axis`, which is the "
        "single definition. Each disagrees with it on a numeric string — issue #58's input, "
        "and the reason #104 and #113 exist:\n" + "\n".join(offenders))


def test_the_writers_no_longer_default_an_axis_to_zero():
    """The literal that caused it, gone from both sites. Read as code — the new comments
    quote the old expression to explain it."""
    code = code_only(REPO / "scripts" / "lib" / "book_ingest.py")
    # ⚠ The `.replace("intended_vat", "")` that used to wrap this was DEAD: "valence" is not a
    # substring of "intended_vat", so it removed nothing and the assertion was already the
    # stronger one. Worse than harmless to leave — it read as load-bearing, so the next editor
    # would have had to work out what it defended against before touching the line. Raised as a
    # nit by the 2026-08-17 review, which declined to spend a tracker record on it.
    assert "valence" not in code, \
        "book_ingest still reaches for a raw axis instead of schemas.intended_vat()"


# --- the two shapes of an absent axis (issues #90, #91, #92) ------------------------------
#
# Both HIGH findings of the 2026-08-17 review came from ONE change: `intended_vat` writes all
# three keys always, using `None` for an absent axis. That is right for the manifest and wrong
# for a label dict, and the difference had to become two functions.

def test_a_label_dict_omits_an_absent_axis_rather_than_passing_none():
    """⚠ THE PRE-EXISTING GUARD WAS RIGHT; THE VALUE SHAPE WAS WRONG (#90).

    `book_ingest.casting_pass` formats each axis with `:+.2f` under `if k in labels` — a guard
    written for exactly the axis a director never produced. Making every key always present
    satisfied `in`, and `format(None, '+.2f')` raises. The run died.
    """
    tag = {"valence": 0.4, "register": "wry"}
    labels = schemas.intended_labels(tag)
    assert labels == {"V": 0.4}, "an absent axis must not appear at all"
    # The real consumer's idiom, verbatim.
    rendered = ", ".join(f"{k}={labels[k]:+.2f}" for k in ("V", "A", "T") if k in labels)
    assert rendered == "V=+0.40"


def test_the_manifest_still_records_an_absent_axis_as_an_explicit_null():
    """The other half: dropping the key here would make "not asked" and "asked, no answer"
    identical, which is the distinction `qc_verdict` goes to real trouble to preserve."""
    got = schemas.intended_vat({"valence": 0.4, "register": "wry"})
    assert got == {"V": 0.4, "A": None, "T": None}
    assert set(got) == {"V", "A", "T"}, "the manifest keeps all three keys"


def test_the_two_shapes_agree_on_every_axis_they_both_report():
    """They must differ only in whether an absent axis is present, never in its VALUE."""
    for tag in ({"valence": 0.4}, {"valence": 0.4, "arousal": -0.2, "tension": 0.0},
                {"valence": "0.7", "arousal": "junk"}, {}):
        vat = schemas.intended_vat(tag, strict=False)
        lab = schemas.intended_labels(tag)
        assert lab == {k: v for k, v in vat.items() if v is not None}, tag


def test_a_progress_line_prints_an_absent_axis_instead_of_dying(capsys):
    """⚠ #91's ordering is why this is not cosmetic: the crash sat AFTER the checkpoint was
    written, flushed and fsynced, so the offending row was already durable. Resume skipped it
    and died on the next one — a run made exactly one chunk of progress per restart."""
    i = schemas.intended_vat({"valence": 0.4})
    line = (f"V={schemas.fmt_axis(i['V'])} A={schemas.fmt_axis(i['A'])} "
            f"T={schemas.fmt_axis(i['T'])}")
    assert "+0.4" in line
    assert "None" not in line, "a placeholder, not the word None"


def test_fmt_axis_refuses_a_bool():
    """`True` is an `int` in Python and `format(True, '+.1f')` renders `+1.0` — a plausible
    number from a value that is not one. Every axis reader in this module refuses bools for
    the same reason."""
    assert schemas.fmt_axis(True) == schemas.fmt_axis(None)
    assert schemas.fmt_axis(0.0) == "+0.0"


def test_the_removed_delivery_validator_has_not_come_back():
    """⚠ #95: `matcha.delivery.check_assignable` owns this rule. A re-export or a thin wrapper
    here restores the fork with an extra layer over it."""
    assert not hasattr(schemas, "validate_delivery_label"), (
        "schemas.py is re-implementing the delivery rule again — import check_assignable from "
        "matcha.delivery instead")
