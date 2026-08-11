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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "derive_a_channel_stats.py")

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
