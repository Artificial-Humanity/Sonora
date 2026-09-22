"""Frame-level pitch and harmonic-to-noise ratio, from one autocorrelation pass.

⚠⚠ ONE IMPLEMENTATION ON PURPOSE. `measure_speaker_f0.py` and `measure_harmonicity.py`
both need to decide which frames are voiced, and they must agree exactly: an F0 axis and
an HNR axis built from different voicing rules are not comparable, and the whole 2026-09
hum result is a correlation between those two axes. Both now call this.

The estimator is autocorrelation over a 40 ms window at a 10 ms hop. The mean is removed
per frame, frames below `rms_floor` are dropped as silence, and the peak is searched
between the period bounds implied by `fmax` and `fmin`. A frame whose normalised peak is
below `periodicity` is dropped as unvoiced.

HNR falls out of the same peak. For a normalised autocorrelation maximum r at the pitch
period, the harmonic fraction of the frame's energy is r and the noise fraction is 1 - r,
so HNR_dB = 10 log10(r / (1 - r)). A perfectly periodic frame gives r -> 1 and HNR -> +inf.

⚠ r IS CLAMPED BELOW 1 BEFORE THE LOG. A near-constant frame can produce r at or above 1
through floating-point error, and one infinity poisons every mean computed downstream.
"""

import math

import numpy as np


def frames(x, sr, fmin, fmax, rms_floor, periodicity):
    """Returns (f0 Hz, hnr dB) arrays over the voiced frames only. Empty when none are."""
    w, hop = int(0.040 * sr), int(0.010 * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    f0, hnr = [], []
    for s in range(0, len(x) - w, hop):
        fr = x[s:s + w].astype(np.float64)
        fr = fr - fr.mean()
        if np.sqrt(np.mean(fr ** 2)) < rms_floor:
            continue
        ac = np.correlate(fr, fr, "full")[w - 1:]
        ac = ac / (ac[0] or 1.0)
        seg = ac[lo:hi]
        if not len(seg):
            continue
        k = int(np.argmax(seg))
        r = float(seg[k])
        if r < periodicity:
            continue
        f0.append(sr / (lo + k))
        r = min(max(r, 1e-6), 1.0 - 1e-6)
        hnr.append(10.0 * math.log10(r / (1.0 - r)))
    return np.asarray(f0), np.asarray(hnr)
