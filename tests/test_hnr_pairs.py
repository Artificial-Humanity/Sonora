"""`hnr_pairs.matched_pairs` edge cases (audit 2026-10-31, M-9)."""
from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import hnr_pairs  # noqa: E402

SPK = {1: {"f0": 100.0, "hnr": 5.0, "partition": "p"},
       2: {"f0": 101.0, "hnr": 9.0, "partition": "p"},
       3: {"f0": 102.0, "hnr": 9.5, "partition": "p"}}


def _pairs(rows):
    return hnr_pairs.matched_pairs(SPK, rows.__getitem__, max_f0_gap=5.0, hnr_lo=1.0, hnr_hi=10.0,
                                   row_ratio=2.0, match_partition=True)


def test_a_speaker_with_no_rows_is_skipped_not_a_division_by_zero():
    pairs = _pairs({1: 0, 2: 100, 3: 100})
    assert all(1 not in (r, c) for r, c, *_ in pairs), "the zero-row speaker was paired"


def test_speakers_within_the_row_ratio_still_pair():
    """The positive control: the guard must not disarm the row-ratio rule."""
    assert len(_pairs({1: 100, 2: 150, 3: 100})) == 1
