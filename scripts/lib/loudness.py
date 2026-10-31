"""Loudness matching, numpy and pyloudnorm only.

⚠ DEPENDENCY-LIGHT ON PURPOSE. `match_loudness` lived in `ear_bench`, which imports torch and
the whole Matcha stack at module scope, so a renderer in an engine's own container (one that
installs no Matcha) could not level-match without importing a model it does not run.
`ear_bench.match_loudness` is this function, re-exported, so every bench keeps one
implementation.
"""


def match_loudness(x, sr, target_lufs):
    """Level-match to `target_lufs`, peak-limited. Returns (audio, ceiling_bound).

    ⚠⚠ LOUDNESS IS THE CONFOUND THAT RUNS BACKWARDS. The teacher-portfolio comparison had
    to be redone because a level difference was read as a quality difference in the wrong
    direction. Any bench putting two differently-produced signals side by side matches
    level first, or it measures gain.
    """
    import numpy as np
    import pyloudnorm
    try:
        loud = pyloudnorm.Meter(sr).integrated_loudness(x.astype("float64"))
    except ValueError as e:                  # shorter than the 0.4 s measurement block
        raise SystemExit("REFUSING: a clip is too short to measure its loudness (%s)." % e)
    if not np.isfinite(loud):
        raise SystemExit("REFUSING: a clip has no measurable loudness.")
    gain = 10.0 ** ((target_lufs - loud) / 20.0)
    peak = float(np.max(np.abs(x))) or 1.0
    g = min(gain, 0.99 / peak)
    return (x * g).astype("float32"), g < gain


def match_pair_loudness(a, b, sr, target_lufs):
    """Level-match two sides of one pair to the SAME loudness. Returns (a, b, bound_sides).

    ⚠ A PEAK-BOUND SIDE BREAKS THE MATCH, SO IT SETS THE LEVEL FOR BOTH. `match_loudness`
    caps a side at the peak ceiling, and a sparse-impulse clip (one vocoder pop) can then land
    12 dB under target while its partner reaches it: the pair differs in gain, the confound
    this module exists to remove. When either side is bound, both are re-matched to the
    quieter achieved loudness, which both can reach. `bound_sides` counts the sides that hit
    the ceiling at the requested target (0 if the match was exact).
    """
    import pyloudnorm
    x, bx = match_loudness(a, sr, target_lufs)
    y, by = match_loudness(b, sr, target_lufs)
    if not (bx or by):
        return x, y, 0
    meter = pyloudnorm.Meter(sr)
    achieved = min(meter.integrated_loudness(v.astype("float64")) for v in (x, y))
    x, _ = match_loudness(a, sr, achieved)
    y, _ = match_loudness(b, sr, achieved)
    return x, y, int(bx) + int(by)
