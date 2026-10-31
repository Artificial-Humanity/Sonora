"""`loudness.match_pair_loudness`: a peak-bound side must not leave its pair 12 dB apart
(audit 2026-10-31, M-1). numpy + pyloudnorm only."""
import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
np = pytest.importorskip("numpy")
pyln = pytest.importorskip("pyloudnorm")
import loudness  # noqa: E402

SR = 24000


def _tone(amp=0.1, seconds=3.0):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype("float64")


def _impulses():
    """Sparse impulses at peak 0.9: loud on the peak meter, quiet on the loudness meter, so
    reaching -23 LUFS would need a gain that clips."""
    x = np.zeros(3 * SR)
    x[::SR // 4] = 0.9
    return x


def _lufs(x):
    return pyln.Meter(SR).integrated_loudness(x.astype("float64"))


def test_the_impulse_clip_cannot_reach_the_target_alone():
    """The precondition: without it the tests below prove nothing."""
    y, bound = loudness.match_loudness(_impulses(), SR, -23.0)
    assert bound and _lufs(y) < -26.0


def test_a_peak_bound_side_sets_the_level_for_both_sides():
    a, b, bound = loudness.match_pair_loudness(_tone(), _impulses(), SR, -23.0)
    assert bound == 1
    assert abs(_lufs(a) - _lufs(b)) < 0.1, (_lufs(a), _lufs(b))
    assert _lufs(a) < -26.0                       # both are at the quieter side's level


def test_an_unbound_pair_is_matched_to_the_target_and_reports_none_bound():
    a, b, bound = loudness.match_pair_loudness(_tone(0.1), _tone(0.3), SR, -23.0)
    assert bound == 0
    assert _lufs(a) == pytest.approx(-23.0, abs=0.05) and _lufs(b) == pytest.approx(-23.0, abs=0.05)


def test_a_clip_shorter_than_the_measurement_block_is_a_refusal_not_a_traceback():
    with pytest.raises(SystemExit, match="too short"):
        loudness.match_loudness(_tone(seconds=0.2), SR, -23.0)
