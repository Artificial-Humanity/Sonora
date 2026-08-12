"""The A-channel cell-mean table must give every design level its own column heading.

Regression guard for the `{e:+.0f}` defect: on the energy design A ∈ {-1, 0, +0.5, +1}
that specifier rounds +0.5 to `+0`, so two adjacent cell-mean columns printed the
IDENTICAL heading `A=+0`. The numbers underneath stayed correct and became
unattributable — a reader cannot tell which column is which, and this section exists
to make the A-channel statistics auditable.

Asserted on the RENDERED HEADINGS rather than on the format string, because the defect
is a collision between two levels and not the presence of any particular specifier: a
future `+.1f` would also be wrong (noisy on the three integers) and a future `+.2g`
would be fine, and only the output distinguishes them.
"""
import importlib.util
import io
import os
import re
from contextlib import redirect_stdout

import pytest
from scripts_layout import SCRIPTS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = str(SCRIPTS / "derive_a_channel_stats.py")

# The design the defect lands on. +0.5 is the level `+.0f` collapses onto 0.
LEVELS = [-1.0, 0.0, 0.5, 1.0]


def _load():
    spec = importlib.util.spec_from_file_location("derive_a_channel_stats", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    """A balanced synthetic design: 2 lanes x 4 levels x 2 texts x 3 repeats.

    Keys are `probe()`'s own — lane / energy / text / lufs. It sys.exits on any absent
    (lane, energy) cell, so the design must be complete rather than merely plausible.
    """
    out = []
    # `unknown` is REQUIRED: section 8 compares every named lane against it and
    # sys.exits without it. A synthetic design has to be complete, not plausible.
    for lane in ("unknown", "Neutral", "Newscaster"):
        for a in LEVELS:
            for text in range(2):
                for rep in range(3):
                    out.append({"lane": lane, "energy": a, "text": f"t{text}",
                                "lufs": -20.0 + a + rep * 0.1})
    return out


def test_every_design_level_gets_its_own_heading():
    mod = _load()
    if not hasattr(mod, "probe"):
        pytest.skip("derive_a_channel_stats.probe() is gone — update this guard")

    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.probe(_rows())

    headings = re.findall(r"A=[-+][\d.]+", buf.getvalue())
    # ⚠ NOT a skip. A guard that opts out when it cannot find its subject reports the same
    # green as one that checked — the failure mode AGENTS.md §5b exists for. If the table
    # moved, this must go red so somebody re-points it.
    assert headings, (
        "no `A=` column headings in probe() output — the cell-mean table moved, and this "
        "guard now checks nothing. Re-point it rather than deleting it.")

    # The whole point: as many DISTINCT headings as there are levels rendered.
    assert len(set(headings)) == len(headings), (
        "two design levels rendered the SAME column heading, so the cell-mean columns "
        f"are unattributable: {headings}"
    )


def _rows_all_level():
    """Every named lane sits EXACTLY on `unknown` at A = 0 — perfect agreement.

    Deliberately the pathological input for the direction partition: `d == 0.0` puts a lane
    in neither `quieter` nor `louder`, and the first guard for that made `level` a VETO,
    so this design routed into the branch headed "THE NAMED LANES DO NOT AGREE IN
    DIRECTION" — the defect the guard was added to fix, one branch to the left.
    """
    out = []
    for lane in ("unknown", "Neutral", "Newscaster"):
        for a in LEVELS:
            for text in range(2):
                for rep in range(3):
                    # No lane offset at all: identical loudness in every lane.
                    out.append({"lane": lane, "energy": a, "text": f"t{text}",
                                "lufs": -20.0 + a + rep * 0.1})
    return out


def test_perfect_agreement_is_not_reported_as_disagreement():
    """`not level` was a veto where a partition was wanted. Agreement is "exactly one of
    the three directions is occupied" — and level is a direction, not a disqualification."""
    mod = _load()
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.probe(_rows_all_level())
    out = buf.getvalue()

    assert "DO NOT AGREE IN DIRECTION" not in out, (
        "every named lane measured EXACTLY on the reference — the most complete agreement "
        "the design admits — and it was reported as disagreement")
    assert "sits EXACTLY on `unknown` at A = 0" in out, \
        "the all-level case needs its own words; `all the same direction, by 0.00 dB` reads as a bug"
    assert "0.00 to 0.00 dB" not in out


def test_a_lane_at_zero_is_counted_and_named_when_others_differ():
    """The mixed case: the tally reads as exhaustive, so the three counts must sum."""
    mod = _load()
    rows = []
    off = {"unknown": 0.0, "Neutral": -0.5, "Newscaster": 0.0}
    for lane, o in off.items():
        for a in LEVELS:
            for text in range(2):
                for rep in range(3):
                    rows.append({"lane": lane, "energy": a, "text": f"t{text}",
                                 "lufs": -20.0 + o + a + rep * 0.1})
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.probe(rows)
    out = buf.getvalue()

    line = next(l for l in out.splitlines() if "DO NOT AGREE IN DIRECTION" in l)
    m = re.search(r"(\d+) quieter, (\d+) louder, (\d+) level \(of (\d+)\)", line)
    assert m, f"the tally is no longer in the form this guard reads: {line}"
    q, lo, lv, total = (int(x) for x in m.groups())
    assert q + lo + lv == total, \
        f"the tally reads as exhaustive and does not sum: {q}+{lo}+{lv} != {total}"
    assert "level: Newscaster" in out, "a lane sitting on the reference must be NAMED"
