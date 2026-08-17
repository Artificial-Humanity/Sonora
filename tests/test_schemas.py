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
    out = []
    src = Path(path).read_text(encoding="utf-8")
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
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
    p = tmp_path / "register_lexicon.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(Exception) as e:
        schemas.load_register_lexicon(str(p))
    assert "lexicon" in str(e.value).lower(), why


def test_the_empty_case_says_why_it_matters():
    """⚠ The message has to explain that empty is not "weaker" but "absent".

    A reader who thinks an empty vocabulary is a degraded vocabulary will restore the
    `except: return []` the first time this refusal is inconvenient.
    """
    with pytest.raises(Exception) as e:
        schemas.RegisterLexicon(lexicon=[])
    msg = str(e.value)
    assert "guard" in msg and "prompt" in msg


def test_missing_and_malformed_files_are_named(tmp_path):
    """`FileNotFoundError` and `JSONDecodeError` both say WHICH file and what it was for —
    the bare `json.load(...)["lexicon"]` this replaced said neither."""
    with pytest.raises(Exception) as e:
        schemas.load_register_lexicon(str(tmp_path / "nope.json"))
    assert "nope.json" in str(e.value)

    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(Exception) as e:
        schemas.load_register_lexicon(str(p))
    assert "bad.json" in str(e.value)


# --- the delivery lane split -------------------------------------------------------------

def test_producing_and_reading_are_different_vocabularies():
    """⚠ `matcha/delivery.py` states the rule and nothing enforced it: a NEW row may only be
    labelled with an ACTIVE lane, while readers stay on the full vocabulary because v5's
    filelists and ep019's weights carry the retired channel and must keep decoding."""
    from matcha import delivery
    retired = delivery.RETIRED_LANES[0]

    assert schemas.validate_delivery_label(retired, producing=False) == retired
    with pytest.raises(Exception) as e:
        schemas.validate_delivery_label(retired, producing=True)
    assert "RETIRED" in str(e.value), "the message must distinguish retired from unknown"


def test_unknown_is_accepted_on_both_sides():
    """The blank cell is a legitimate label: `seed_delivery.py` leaves the ear's cases blank
    rather than guessing, and every LibriTTS clip predates the axis."""
    from matcha import delivery
    for producing in (True, False):
        assert schemas.validate_delivery_label(
            delivery.DELIVERY_UNKNOWN, producing=producing) == delivery.DELIVERY_UNKNOWN


def test_a_typo_is_refused_on_both_sides():
    for producing in (True, False):
        with pytest.raises(Exception):
            schemas.validate_delivery_label("Netural", producing=producing)


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
