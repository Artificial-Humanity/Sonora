"""`scm.validate` certifies a RATIFIED wire format, which is not the tolerant question.

WHY THIS FILE EXISTS (2026-08-19)
---------------------------------
`scm.py` had no test at all — `grep -rn "import scm" tests/` returned nothing, and its only
consumer reads `/data`. That is how the following went in and back out again inside two days.

Issue #117 correctly observed that only ONE of the module's three vat readers had been routed
through `schemas.coerce_axis`, so `validate` began reporting "no schema violations" for a
numeric string that `verify_vat` and `render_inline` still crashed on. The fix routed all
three through `coerce_axis`. That removed the inconsistency and introduced a worse one:

    notes/markup-schema-brief.md, RATIFIED v0.1 — "Field semantics":
        `utterance.vat` | continuous [−1,1] each
    and, under "Numbers vs symbols":
        the sidecar keeps VAT **continuous** (it IS the contract); binning to symbols
        happens only in the quantizer's Gemma prompt and the inline header — presentation,
        not storage.

So a JSON string is a schema violation, and `validate` — whose entire job is conformance —
was blessing one. **`coerce_axis` and `validate` ask different questions.** `coerce_axis`:
"is this a usable number for a LABEL", tolerant, for director output. `validate`: "does this
conform to the contract", strict, for storage.

⚠ Found by sweeping the docs after the change rather than by anything failing. The comment
beside the code was consistent; a ratified brief three directories away was not.
"""

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()

import scm  # noqa: E402
import schemas  # noqa: E402

LEX = ["neutral"]


def _obj(vat):
    return {"scm": "0.1", "utterance": {"register": "neutral", "vat": vat, "style": []}}


def _vat_errors(vat):
    return [e for e in scm.validate(_obj(vat), LEX) if e.startswith("vat.")]


def test_a_conformant_sidecar_validates():
    assert not _vat_errors({"V": 0.2, "A": 0.1, "T": 0.0})


def test_a_numeric_STRING_is_a_schema_violation():
    """⚠ THE ONE THE BRIEF SETTLES. `coerce_axis("0.7")` is 0.7 — that is #58's ruling about
    a director's LABEL, and it does not license a string in ratified storage."""
    errs = _vat_errors({"V": "0.2", "A": 0.1, "T": 0.0})
    assert errs and "continuous" in errs[0]
    assert schemas.coerce_axis("0.2") == 0.2, "and coerce_axis still reads it, for labels"


@pytest.mark.parametrize("bad", [None, True, False, "warm", [], {}])
def test_a_non_number_is_a_schema_violation(bad):
    assert _vat_errors({"V": bad, "A": 0.1, "T": 0.0})


def test_out_of_range_is_reported_separately_from_wrong_type():
    """Two different repairs — re-emit vs re-clamp — so they must not share a message."""
    typ = _vat_errors({"V": "0.2", "A": 0.1, "T": 0.0})[0]
    rng = _vat_errors({"V": 1.5, "A": 0.1, "T": 0.0})[0]
    assert "continuous" in typ and "out of" in rng and typ != rng


def test_the_bounds_come_from_schemas_not_from_a_literal():
    """The half of the #113 fix that was right: `[-1, 1]` is one constant, not two."""
    assert (schemas.AXIS_MIN, schemas.AXIS_MAX) == (-1.0, 1.0)
    assert not _vat_errors({"V": schemas.AXIS_MIN, "A": schemas.AXIS_MAX, "T": 0.0})


def test_the_TOLERANT_readers_still_do_not_crash_on_an_unvalidated_string():
    """⚠ #117's actual complaint, and it must stay fixed. `validate` refusing the string is
    what stops it being CERTIFIED; these two are called on unvalidated input by `tag_spike`,
    so they must degrade rather than raise."""
    obj = _obj({"V": "0.2", "A": 0.1, "T": 0.0})
    ok, flags = scm.verify_vat(obj, {"V": 0.2, "A": 0.1, "T": 0.0})
    assert ok and not flags
    assert "V+0.20" in scm.render_inline(obj)


def test_the_tolerance_is_not_restated_anywhere_in_the_module():
    """⚠ THE FIRST VERSION OF THIS TEST WAS ITSELF A SECOND LITERAL (issue #133).

    It was named `..._is_not_restated_in_this_file` and its body was `assert scm.VAT_TOL ==
    0.35` — a restatement, in a test asserting there are none. It also missed the one that
    mattered: `scm.py`'s own module docstring gave the tolerance as `±0.35` five lines above
    the constant, and nothing could see it, because `test_doc_claims.docs()` builds its file
    set from `notes/*.md`, the root README and `configs/data/*.yaml` — no `.py` at all.

    So the assertion is now about ABSENCE, and it names no number of its own.
    """
    src = SCRIPTS.src("scm.py")
    body = src.split("VAT_TOL", 1)[0]      # everything above the definition, i.e. the docstring
    assert str(scm.VAT_TOL) not in body, (
        f"scm.py's docstring restates the tolerance ({scm.VAT_TOL}) above the constant that "
        f"defines it. Point at `VAT_TOL` instead; a value in two places drifts, and the "
        f"doc-claims gate cannot read .py files.")
    assert "±0.35" not in src and "0.35" not in src.replace("VAT_TOL = 0.35", "", 1), \
        "the tolerance appears in scm.py somewhere other than its definition"
