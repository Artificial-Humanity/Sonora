"""`ref_select`'s axis ranking, held to what its docstring promises.

WHY THIS FILE EXISTS (2026-08-18, issues #103 and #104)
------------------------------------------------------
`_vat` and `select_reference` had **no test of their return value**. `test_campaign_tooling`
checks `select_reference`'s call sites as source *text* and never runs the ranking, so two
defects went in together and neither was catchable:

* `_vat` returned `0.0` for an unknown axis while its docstring said an unknown axis
  "contributes nothing to the distance". The distance is `(want[a] - kv[a]) ** 2`, so `0.0`
  contributes `kv[a] ** 2` — the largest term for the most strongly-labelled candidates — and
  minimising it is the definition of preferring a neutral reference. Measured on the 436-clip
  pool, the top ten for a V-only tag averaged |A|+|T| of 0.252 against 0.219 for the pool at
  large: **flatter than average on the axes nobody had labelled.**
* `_n` re-implemented "what counts as a number" as `isinstance(c, (int, float))`, the exact
  test issue #58 was filed against, so the numeric string `"0.7"` became no-information.

⚠ **THE POINT OF THIS FILE IS THAT REVERTING EITHER ONE TURNS IT RED.** A guard that only
asserts the current shape would pass on the old code too; `test_reverting_the_neutral_default_
would_be_caught` and `test_reverting_the_isinstance_test_would_be_caught` are written as
falsifiers, against the specific expression that was there before.
"""

import math

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()

import ref_select  # noqa: E402
import schemas  # noqa: E402


# --------------------------------------------------------------------------- _vat

def test_an_absent_axis_is_none_not_zero():
    assert ref_select._vat({}) == {"V": None, "A": None, "T": None}


def test_a_present_but_null_axis_is_none():
    """The trap from #92: `.get(k, 0)` does not fire on a present key holding null."""
    assert ref_select._vat({"V": None, "A": None, "T": None}) == {"V": None, "A": None, "T": None}


def test_a_numeric_string_is_a_label():
    """Issue #58's ruling, which `_n`'s isinstance test contradicted (#104)."""
    assert ref_select._vat({"valence": "0.7"})["V"] == 0.7
    assert ref_select._vat({"tension": " -0.4 "})["T"] == -0.4


def test_a_non_numeric_string_carries_no_number():
    assert ref_select._vat({"valence": "very sad"})["V"] is None


def test_booleans_are_not_axis_values():
    assert ref_select._vat({"valence": True})["V"] is None
    assert ref_select._vat({"valence": False})["V"] is None


def test_the_alias_chain_falls_through_rather_than_first_key_wins():
    """An unreadable `V` must still let `valence` answer — the isinstance version did this."""
    assert ref_select._vat({"V": "very sad", "valence": 0.6})["V"] == 0.6
    assert ref_select._vat({"energy": -0.3})["A"] == -0.3


@pytest.mark.parametrize("raw", [0.7, "0.7", "very sad", None, True, 2, {"x": 1}, []])
def test_vat_agrees_with_the_single_definition(raw):
    """⚠ THE ANTI-DRIFT ASSERTION. `schemas.coerce_axis` is the one definition (#58, #104);
    this pins `_vat` to it for every input rather than to a copy of its current behaviour."""
    assert ref_select._vat({"valence": raw})["V"] == schemas.coerce_axis(raw)


def test_reverting_the_isinstance_test_would_be_caught():
    """Falsifier for #104. The pre-fix `_n` was:

        if isinstance(c, (int, float)) and not isinstance(c, bool): return float(c)
        return 0.0

    which answers 0.0 here. Nothing else in this file distinguishes the two on this input.
    """
    assert ref_select._vat({"valence": "0.7"})["V"] != 0.0


# ------------------------------------------------------------------- _vat_distance

FULL = {"V": 0.8, "A": 0.8, "T": 0.65}


def test_all_three_axes_is_exactly_the_old_euclidean_formula():
    """⚠ THE CALIBRATION GUARD. `score` is summed with ENGINE_PREF (0.2) and AGE_WEIGHT, so a
    change of scale here silently reweights the whole ranking. The rescale is identity when
    all three axes are present, and this pins that."""
    for want, kv in [({"V": 0.4, "A": -0.2, "T": 0.1}, {"V": 0.1, "A": 0.3, "T": -0.2}),
                     ({"V": -1.0, "A": 1.0, "T": 0.0}, {"V": 1.0, "A": -1.0, "T": 0.0}),
                     (FULL, FULL)]:
        old = math.sqrt(sum((want[a] - kv[a]) ** 2 for a in "VAT"))
        assert ref_select._vat_distance(want, kv) == pytest.approx(old, abs=1e-12)


def test_no_shared_axis_drops_the_term_entirely():
    """"Contributes nothing to the distance" — now true rather than claimed."""
    assert ref_select._vat_distance({"V": None, "A": None, "T": None}, FULL) == 0.0


def test_an_unlabelled_axis_does_not_pull_toward_neutral():
    """The #103 defect, as one assertion.

    Target: V=0.8, A and T unlabelled. Candidate X is strongly aroused and tense; candidate Y
    is flat on both. They are identical on the ONE axis anyone labelled, so the ranking must
    not prefer either — and must certainly not prefer the flat one.
    """
    want = {"V": 0.8, "A": None, "T": None}
    strong = {"V": 0.8, "A": 0.8, "T": 0.65}
    flat = {"V": 0.8, "A": 0.0, "T": 0.0}
    assert ref_select._vat_distance(want, strong) == ref_select._vat_distance(want, flat)


def test_reverting_the_neutral_default_would_be_caught():
    """Falsifier for #103, at the level the bug actually lived: `_vat` feeding the distance.

    With `_vat` returning 0.0 for an unknown axis and a plain three-axis sum, `strong` scores
    1.033 and `flat` scores 0.0 — the flat clip wins outright. Written through `_vat` rather
    than against `_vat_distance` alone so that reverting EITHER function turns it red.
    """
    want = ref_select._vat({"valence": 0.8})
    strong = ref_select._vat({"valence": 0.8, "arousal": 0.8, "tension": 0.65})
    flat = ref_select._vat({"valence": 0.8, "arousal": 0.0, "tension": 0.0})
    assert ref_select._vat_distance(want, strong) == ref_select._vat_distance(want, flat)
    assert ref_select._vat_distance(want, flat) == 0.0


def test_a_partly_labelled_target_still_ranks_on_what_it_labelled():
    """Skipping unknown axes must not make every candidate tie — V still discriminates."""
    want = {"V": 0.8, "A": None, "T": None}
    near = {"V": 0.75, "A": -0.9, "T": 0.9}
    far = {"V": -0.5, "A": 0.8, "T": 0.65}
    assert ref_select._vat_distance(want, near) < ref_select._vat_distance(want, far)


def test_the_rescale_does_not_advantage_a_sparsely_labelled_candidate():
    """⚠ WHY THE `* 3 / len(axes)` IS THERE. A raw sum over shared axes would let a candidate
    that labelled fewer axes score lower for free, which is a bias toward unlabelled data."""
    want = {"V": 0.5, "A": 0.5, "T": 0.5}
    one_axis = {"V": 0.0, "A": None, "T": None}      # 0.5 off on its one axis
    three_axis = {"V": 0.0, "A": 0.0, "T": 0.0}      # 0.5 off on each of three
    assert ref_select._vat_distance(want, one_axis) == pytest.approx(
        ref_select._vat_distance(want, three_axis))


def test_vat_refuses_a_record_instead_of_reading_it_as_all_absent():
    """⚠ THE #107 GUARD. A keeps record keeps its axes under `intended_vat`; `_vat` reads
    `V`/`valence` at the top level. Handed the record it used to answer "no axes at all",
    which is not an error anyone sees — the caller then ranks every candidate against the
    same constant and still produces output. Refusing is the only way that surfaces."""
    record = {"id": "x", "intended_vat": {"V": 0.8, "A": 0.1, "T": 0.0}, "file": "a.wav"}
    with pytest.raises(TypeError, match="record, not an axis dict"):
        ref_select._vat(record)
    # the thing it is telling you to write must of course still work
    assert ref_select._vat(record["intended_vat"]) == {"V": 0.8, "A": 0.1, "T": 0.0}
