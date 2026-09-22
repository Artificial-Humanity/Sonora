"""Detecting a DELAYED COPY of the voice summed with itself — the "layering" defect.

⚠⚠ EIGHT DETECTORS HAVE ALREADY FAILED ON THIS DEFECT CLASS. `reverb-artifact-diagnosis`
records whole-clip `env_ac`, a local CPP dip, an emphasis-subharmonic delta and an ASR
insertion rate among them, on the teacher engines. Two lessons from that wreckage are built
into this module rather than rediscovered:

  1. THE DEFECT IS LOCAL. It appears on individual syllables and every whole-clip mean
     averages it away — one probe scored a clip the owner heard as split at 0.019, the
     CLEANEST in its set. So this reports a TAIL statistic across frames, never a mean.
  2. ENERGY-DECAY MEASURES ARE THE WRONG PHYSICS. C50 and envelope autocorrelation look
     for reverberation, and a breathy voice reads as reverberant on them — C50 falsely
     condemned a clean speaker. A summed delayed copy is not reverberation: it is a COMB
     FILTER, and its signature is periodic ripple in the MAGNITUDE SPECTRUM.

THE OWNER'S OBSERVATION THAT MAKES THIS WORTH TRYING (2026-09-22): the layering and the
hum may be one defect. "When the two voices are extremely close, it creates an almost
buzz/hum effect." That is exactly right physically. Summing a signal with a copy delayed
by tau puts notches every 1/tau Hz across the spectrum. At tau = 50 ms the ear resolves two
events and calls it layering; at tau of a few milliseconds the notch spacing lands in the
formant region and the same mechanism is heard as a metallic buzz. One cause, two
descriptions depending on the delay.

⚠ AND IT EXPLAINS WHY THE PERIODICITY INSTRUMENTS WERE BLIND. A comb-filtered voice is not
LESS periodic — the delayed copy adds a correlation peak, so autocorrelation HNR reads it
as equally or MORE harmonic. Every mel-domain and HNR measurement in this investigation
could have been correct and still missed this.

⚠⚠ AN ECHO AT AN EXACT MULTIPLE OF THE PITCH PERIOD IS INVISIBLE HERE, and no choice of
parameters changes that. A copy delayed by exactly one pitch period is, to a cepstrum,
indistinguishable from the voice being periodic — the energy lands on a rahmonic that must
be blanked. The detector therefore has blind stripes at lag = k/F0. For a 120 Hz voice
those sit every 8.3 ms. Report it as a floor on sensitivity, never as a clean reading.

⚠ THE PITCH QUEFRENCY MUST BE EXCLUDED, and that is the whole difficulty. The cepstrum's
largest peak is at 1/F0 by construction, and for a 100 Hz voice that is 10 ms — squarely
inside the delay range of interest. A detector that does not exclude the pitch peak and its
multiples reports pitch strength and calls it layering.
"""

import numpy as np


def _frames(x, sr, win, hop):
    n = int(win * sr)
    h = int(hop * sr)
    for s in range(0, max(0, len(x) - n), h):
        yield x[s:s + n]


def _running_median(a, w):
    """Median filter that ignores NaNs, so the blanked pitch bands do not drag the trend."""
    n = len(a)
    h = w // 2
    out = np.empty(n)
    for i in range(n):
        seg = a[max(0, i - h):min(n, i + h + 1)]
        seg = seg[~np.isnan(seg)]
        out[i] = np.median(seg) if len(seg) else np.nan
    return out


def echo_strength(x, sr, fmin=55.0, fmax=350.0, lag_lo=0.004, lag_hi=0.060,
                  win=0.150, hop=0.050, rms_floor=0.01, pitch_guard=0.15,
                  n_harmonics=4, pct=90):
    """Per-clip strength of a summed delayed copy, as a TAIL over frames.

    ⚠⚠ THE WINDOW MUST BE LONGER THAN THE LONGEST DELAY SOUGHT, and the first version of
    this was not: a 40 ms window cannot contain a 50 ms echo, so every frame was discarded
    by the length guard and the whole clip scored a tidy 0.00 that looked like a clean
    reading. The self-test caught it because it refuses on a lag it inserted and did not
    get back. 150 ms is still one syllable, so the locality the previous eight detectors
    lost to whole-clip averaging is kept.

    Returns `(strength, lag_seconds, n_frames)`. `strength` is the `pct`-th percentile of
    the per-frame off-pitch cepstral peak, normalised by that frame's cepstral spread, so
    it is comparable across clips of different level and voice.

    `pitch_guard` blanks a +/- fraction around the pitch quefrency and its first
    `n_harmonics` multiples. ⚠ SUBMULTIPLES MATTER TOO: half the pitch period is where a
    period-doubling voice puts energy, and it is not a delayed copy.
    """
    # ⚠ 150 ms CONTAINS EVERY LAG SOUGHT AND STAYS LOCAL. An earlier version derived the
    # window from `lag_hi` (5x) because a 45 ms echo was being missed and spectral
    # resolution looked like the reason. It was not — the miss was an over-wide rahmonic
    # guard, fixed below — and the longer window was strictly worse, losing the 5.5 ms lag
    # to the pitch region. The wrong explanation is recorded here because it was in this
    # file as a confident comment for three edits.
    q_lo, q_hi = int(lag_lo * sr), int(lag_hi * sr)
    best, lags = [], []
    for fr in _frames(np.asarray(x, dtype=np.float64), sr, win, hop):
        fr = fr - fr.mean()
        if np.sqrt(np.mean(fr ** 2)) < rms_floor:
            continue
        w = fr * np.hanning(len(fr))
        spec = np.abs(np.fft.rfft(w, n=2 * len(w)))
        logspec = np.log(spec + 1e-10)
        # real cepstrum; index k is a quefrency of k/sr seconds
        # ⚠ KEEP THE WHOLE irfft. Slicing to len(logspec) throws away the half of the
        # quefrency axis the long delays live on, and index k means k/sr seconds only if
        # the axis is intact.
        cep = np.fft.irfft(logspec)
        if len(cep) <= q_hi:
            continue
        # pitch quefrency from the cepstrum's own strongest peak in the plausible band
        p_lo, p_hi = int(sr / fmax), int(sr / fmin)
        if p_hi >= len(cep):
            continue
        pk = p_lo + int(np.argmax(cep[p_lo:p_hi]))
        # ⚠⚠ THE TREND IS COMPUTED OVER THE WHOLE LOW-QUEFRENCY RANGE, THEN THE PEAK IS
        # SOUGHT IN A SUB-BAND. Detrending only `cep[q_lo:q_hi]` gave the median filter no
        # context below q_lo, so it could not track the envelope's steep edge and left a
        # large residual at the very start of the band. The detector then reported "2.3 ms"
        # for echoes inserted at 5.5, 19 and 45 ms — the strength responded correctly while
        # the lag was an edge artifact, which is exactly the kind of half-working number
        # that gets believed.
        full = cep[:q_hi].astype(np.float64).copy()
        n_m = max(n_harmonics, int(q_hi / max(pk, 1)) + 1)
        # ⚠⚠ THE GUARD IS AN ABSOLUTE WIDTH, NOT A PERCENTAGE OF THE RAHMONIC. A
        # proportional guard grows with m: at +/-15% the fifth rahmonic of a 7.9 ms period
        # blanks 33.5-45.3 ms, which swallowed a real 45 ms echo whole and reported it as
        # clean. A rahmonic peak is about as wide in quefrency wherever it sits, so the
        # guard is a fraction of ONE pitch period applied at every multiple.
        half = max(1, int(pitch_guard * pk))
        for m in list(range(1, n_m + 1)) + [0.5]:
            c0 = int(pk * m)
            full[max(0, c0 - half):max(0, c0 + half + 1)] = np.nan
        trend = _running_median(full, max(3, int(0.008 * sr) | 1))
        resid = (full - trend)[q_lo:]
        if np.all(np.isnan(resid)):
            continue
        spread = np.nanstd(resid)
        if not np.isfinite(spread) or spread <= 0:
            continue
        i = int(np.nanargmax(resid))
        best.append(float(np.nanmax(resid)) / spread)
        lags.append((q_lo + i) / sr)
    if not best:
        return 0.0, 0.0, 0
    best = np.asarray(best)
    k = int(np.argsort(best)[int(pct / 100.0 * (len(best) - 1))])
    return float(np.percentile(best, pct)), float(lags[k]), len(best)


def add_echo(x, sr, lag_s, gain):
    """A clean positive control: the signal plus a delayed, attenuated copy of itself."""
    d = int(lag_s * sr)
    y = np.asarray(x, dtype=np.float64).copy()
    y[d:] += gain * np.asarray(x, dtype=np.float64)[:len(x) - d]
    return y / (np.max(np.abs(y)) or 1.0)
