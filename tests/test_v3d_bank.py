"""`make_v3d_bank` — the module that could not be tested, now that it can be.

WHY THIS FILE EXISTS (2026-08-19, issue #124)
---------------------------------------------
`scripts/tools/make_v3d_bank.py` was the only file in `scripts/tools/` without a `__main__`
guard (measured with `grep -L`). Its whole body ran at import: two reads under `/data`, one
write. So nothing in it could be collected by the suite, and the cost is on the record —
**every one of this branch's findings against that file (#107, #112, #113, #116, #123) was
filed "unverified, cannot be executed", and three of them were regressions introduced by
the fix to the previous one.**

That is the argument for the guard, and this file is what the guard buys. Each test below
pins a defect that was found by reading because there was no other way to find it.
"""

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()

import make_v3d_bank as v3d  # noqa: E402
import ref_select  # noqa: E402


def test_the_module_imports_without_touching_the_data_volume():
    """⚠ THE POINT OF #124. If this ever regresses, every test below stops existing —
    an import that reads `/data` fails collection on any machine without the mount, and a
    collection error aborts the whole session rather than failing one module."""
    assert callable(v3d.main), "the body must live in main(), not at module scope"


# --- the row gate -------------------------------------------------------------------

def test_a_fully_labelled_row_is_rankable():
    assert v3d.rankable_vat({"V": 0.2, "A": 0.1, "T": 0.0})


def test_a_PARTIALLY_labelled_row_is_rankable(): # issue #112
    """It used to require all three, so a row with V and no A/T was thrown out — and the
    message told the operator to re-direct it, a 31B inference for a row that ranks fine."""
    assert v3d.rankable_vat({"V": 0.2, "A": None, "T": None})


def test_a_numeric_string_axis_is_rankable():  # issue #113
    """#58's ruling: `"0.7"` IS a label. The old `isinstance` test said otherwise."""
    assert v3d.rankable_vat({"V": "0.7", "A": None, "T": None})


def test_a_LONG_FORM_axis_block_is_rankable():  # issue #123
    """The scorer became alias-aware while this gate stayed short-key-only, so a
    `{"valence": ...}` block was dropped with a message saying it labelled nothing."""
    assert v3d.rankable_vat({"valence": 0.7})
    assert v3d.rankable_vat({"arousal": -0.3})
    assert v3d.rankable_vat({"tension": 0.5})


def test_a_row_labelling_nothing_is_NOT_rankable():
    """Still dropped, and for the reason that is true: it scores 0.0 against every
    candidate, so they all tie and ENGINE_PREF picks the reference by itself."""
    assert not v3d.rankable_vat({"V": None, "A": None, "T": None})
    assert not v3d.rankable_vat({})
    assert not v3d.rankable_vat({"V": True, "A": False})   # bools are not axis values


@pytest.mark.parametrize("row", [
    {"V": 0.2, "A": None, "T": None},
    {"valence": 0.7},
    {"V": "0.7"},
    {"V": None, "A": None, "T": None},
    {},
])
def test_the_gate_and_the_scorer_never_disagree(row):
    """⚠ THE #123 INVARIANT, as a property rather than as three examples.

    A row the gate keeps must be one the scorer can actually rank, and a row it drops must
    be one the scorer would tie. Both now ask `ref_select._vat`, so this holds by
    construction — which is the point: the previous version held only while two functions
    happened to be edited in the same pass.
    """
    normalised = ref_select._vat(row)
    has_a_number = any(v is not None for v in normalised.values())
    assert v3d.rankable_vat(row) is has_a_number


def test_the_scorer_survives_every_shape_the_gate_admits():
    """⚠ #116 IN ONE ASSERTION. The gate was loosened to admit a numeric string while the
    scorer still got the row RAW, so `"0.7" - 0.5` was a TypeError that ended the build."""
    keep = {"V": 0.5, "A": 0.1, "T": 0.0}
    for row in ({"V": "0.7", "A": 0.1, "T": 0.0}, {"valence": 0.7}, {"V": True, "A": 0.1}):
        if not v3d.rankable_vat(row):
            continue
        d = ref_select._vat_distance(ref_select._vat(row), ref_select._vat(keep))
        assert isinstance(d, float)
