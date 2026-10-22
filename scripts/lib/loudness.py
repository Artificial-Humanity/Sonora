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
    loud = pyloudnorm.Meter(sr).integrated_loudness(x.astype("float64"))
    if not np.isfinite(loud):
        raise SystemExit("REFUSING: a clip has no measurable loudness.")
    gain = 10.0 ** ((target_lufs - loud) / 20.0)
    peak = float(np.max(np.abs(x))) or 1.0
    g = min(gain, 0.99 / peak)
    return (x * g).astype("float32"), g < gain
