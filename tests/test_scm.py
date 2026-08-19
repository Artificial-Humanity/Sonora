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


def restatements(src, value):
    """How many times `value` appears in `src` beyond its own definition.

    Extracted so the rule can be EXERCISED rather than only applied — a check that only ever
    runs against the real file cannot be shown to still work after the value changes, which
    is precisely how v2 of this test went vacuous (issue #135).
    """
    definition = f"VAT_TOL = {value}"
    if definition not in src:
        raise AssertionError(f"no `{definition}` in the source")
    return src.replace(definition, "", 1).count(value)


def test_the_tolerance_is_not_restated_anywhere_in_the_module():
    """⚠ THIS TEST HAS NOW BEEN WRONG TWICE, IN OPPOSITE DIRECTIONS.

    v1 was `assert scm.VAT_TOL == 0.35` under the name `..._is_not_restated_in_this_file` —
    a restatement, in a test asserting there are none (issue #133).

    v2 replaced it with `"±0.35" not in src` plus a `replace("VAT_TOL = 0.35", "", 1)`, which
    put the literal back THREE times while its docstring and the brief both claimed it named
    none — and went **vacuous the moment the constant is amended**: at `VAT_TOL = 0.4` the
    replace matches nothing and the search string `0.35` is absent from a file that now
    restates `0.4` everywhere. A guard a value change silently disarms is §5b's whole
    subject (issue #135).

    v3 derives everything from `scm.VAT_TOL`, so amending the constant re-aims the test.
    """
    assert restatements(SCRIPTS.src("scm.py"), str(scm.VAT_TOL)) == 0, (
        f"{scm.VAT_TOL} appears in scm.py somewhere other than its definition. A value in "
        f"two places drifts, and the doc-claims gate cannot read .py files, so nothing else "
        f"would catch it.")


@pytest.mark.parametrize("value", ["0.35", "0.4", "0.25", "0.5"])
def test_the_restatement_rule_still_bites_at_any_value(value):
    """⚠ THE LIVENESS CHECK, and the thing #135 was actually about.

    The failure mode is not "the assertion is wrong" but "the assertion stops applying". So
    the rule is run against synthetic sources at several values: a clean one must pass and a
    restating one must be caught, whatever the constant happens to be.
    """
    clean = f"VAT_TOL = {value}  # the only mention\ndef f():\n    return VAT_TOL\n"
    dirty = f'"""docstring says within ±{value}."""\nVAT_TOL = {value}\n'
    assert restatements(clean, value) == 0
    assert restatements(dirty, value) == 1


def test_the_restatement_rule_refuses_a_source_with_no_definition():
    """A rule that silently passes when it cannot find its anchor is disarmed, not clean."""
    with pytest.raises(AssertionError, match="no `VAT_TOL"):
        restatements("nothing here\n", "0.35")


# --- the messages, one per repair (issues #132, #137, #138) ----------------------------
#
# ⚠ #138: the #132 split added three messages and no assertion that could tell them apart,
# so collapsing them back into one would have kept the suite green. Each repair is a
# different action for the operator, which is the whole reason they are separate strings.

@pytest.mark.parametrize("vat,expect", [
    ({"A": 0.1, "T": 0.0},                      "is missing"),
    ({"V": None, "A": 0.1, "T": 0.0},           "is missing"),
    ({"V": "0.7", "A": 0.1, "T": 0.0},          "not a number"),
    ({"V": True, "A": 0.1, "T": 0.0},           "not a number"),
    ({"V": float("nan"), "A": 0.1, "T": 0.0},   "not a finite number"),
    ({"V": float("inf"), "A": 0.1, "T": 0.0},   "not a finite number"),
    ({"V": 1.5, "A": 0.1, "T": 0.0},            "out of"),
])
def test_each_repair_gets_its_own_message(vat, expect):
    errs = _vat_errors(vat)
    assert len(errs) == 1 and expect in errs[0], errs


def test_the_four_messages_are_all_distinct():
    """Collapsing any two back together must fail here, which is what #138 asked for."""
    msgs = {_vat_errors(v)[0] for v in (
        {"A": 0.1, "T": 0.0},
        {"V": "0.7", "A": 0.1, "T": 0.0},
        {"V": float("nan"), "A": 0.1, "T": 0.0},
        {"V": 1.5, "A": 0.1, "T": 0.0})}
    assert len(msgs) == 4, msgs


def test_a_wrong_type_and_an_out_of_range_value_both_name_the_value():
    """#137: the range message printed no value while the two beside it did."""
    assert "'0.7'" in _vat_errors({"V": "0.7", "A": 0.1, "T": 0.0})[0]
    assert "1.5" in _vat_errors({"V": 1.5, "A": 0.1, "T": 0.0})[0]
    assert "nan" in _vat_errors({"V": float("nan"), "A": 0.1, "T": 0.0})[0]
