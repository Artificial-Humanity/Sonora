"""#388. `measure_delivery_confound.py` promises exit 2 for "a malformed row" and delivered
exit 1 with a traceback for three of them — a non-numeric channel, a two-hot block, a
fractional label. The two the docstring specifically advertised as refused by the validating
reader were among them. Nothing was mis-measured; the CODE was the wrong one, and the exit
code is the interface `_refuse`'s own comment says a caller keys on.

Each refusal here is checked in both directions in the sense that file's own standard asks
for: the malformed shapes exit 2 with an `ABORT:` line and no traceback, and a well-formed
crossed corpus exits 0 and reports the confound — so a change that made the reader refuse
everything would fail the control, not pass by accident.

The fixture rows are derived from `matcha.delivery`, not typed: the delivery block is the
last `len(DELIVERY_LANES)` channels of a `VAT_DIM`-wide vector, and restating either width
here would be the second copy the tool itself refuses to keep.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from matcha.delivery import DELIVERY_LANES, VAT_DIM

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "scripts" / "tools" / "measure_delivery_confound.py"
N_LANES = len(DELIVERY_LANES)
PROSODY = ["0"] * (VAT_DIM - N_LANES)


def _block(*values):
    """A delivery block padded to N_LANES, as the comma-joined string a filelist carries."""
    lanes = list(values) + ["0"] * (N_LANES - len(values))
    return ",".join(PROSODY + lanes)


def _run(tmp_path, rows):
    (tmp_path / "train_op.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(TOOL), "--corpus", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO,
    )


@pytest.mark.parametrize("label, block", [
    ("non-numeric", _block("x")),
    ("two-hot", _block("1", "1")),
    ("fractional", _block("0.5")),
])
def test_malformed_row_exits_2_without_a_traceback(tmp_path, label, block):
    r = _run(tmp_path, [f"a.wav|s1|p|{block}"])
    assert r.returncode == 2, (label, r.returncode, r.stderr)
    assert "ABORT:" in r.stderr, (label, r.stderr)
    assert "Traceback" not in r.stderr, (label, r.stderr)
    assert ":1" in r.stderr, ("the refusal must name the row", label, r.stderr)


def test_two_hot_refusal_keeps_the_readers_reason(tmp_path):
    r = _run(tmp_path, [f"a.wav|s1|p|{_block('1', '1')}"])
    assert r.returncode == 2
    assert "not one-hot" in r.stderr, r.stderr


def test_crossed_corpus_is_measured_positive_control(tmp_path):
    rows = [f"a.wav|s1|p|{_block('1')}", f"b.wav|s1|p|{_block('0', '1')}"]
    r = _run(tmp_path, rows)
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert "Traceback" not in r.stderr
