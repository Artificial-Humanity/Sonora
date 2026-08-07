"""Silero VAD via onnxruntime — speech boundaries, NOT content removal.

WHY THIS EXISTS. `qc_gate.py` measured effective speech with
`librosa.effects.split(top_db=35)`, a bare energy gate. It cannot tell a quiet consonant
from a pause, it is fooled by room tone, and it has no notion of *speech* — so a hum, a
page turn or a breath all read as signal. Silero is a trained speech detector and gets all
of those right.

⚠ THAT ARGUMENT IS ABOUT CORRECTNESS, NOT MAGNITUDE, and the two are not the same. Where
it has been measured, the win is decisive for SILENCES — the pause gate found 20 of 21
ear-confirmed stalls the energy gate never flagged — and small for TOTAL SPEECH DURATION.
Sampled 2026-08-07 over 150 clips from four campaigns: mean VAD/energy ratio 1.008, median
delta -0.06 s, and at the owner's 4 s floor exactly one clip in 150 changes side. So do not
assume swapping the instrument moves a duration threshold; on this material it does not.
The two questions this module answers have very different sensitivities to which detector
answers them.

⚠ SCOPE — owner decision 2026-07-29. This is for BOUNDARIES ONLY: leading/trailing trim,
effective speech duration, and locating over-long internal silences. It must NEVER be used
to strip lip-smacks, breaths or mouth noise. Breathiness is a tension marker in the VAT
work, `hushed`/`breathy` are register-lexicon terms, and Orpheus ships `<sigh>`/`<gasp>`
tags because those are wanted. A de-breathed corpus would teach Sonora to perform without
breathing, which is a large part of what makes synthesis sound synthetic.

No new heavy dependency: onnxruntime is already required for DNSMOS. Model at
/data/toolchain/silero/silero_vad.onnx (2.3 MB), mirroring the dnsmos layout.

The immediate payoff is the Zonos failure mode from revisit-v1 — "no sound for the first
2.5 seconds", "3.5 second pause between 'patience.' and 'We asked for time.'" Those are
exactly what `long_silences()` reports, and the energy gate never flagged either.
"""
import numpy as np

MODEL_PATH = "/data/toolchain/silero/silero_vad.onnx"
SR = 16000
WINDOW = 512          # samples per frame at 16 kHz — the size Silero expects
CONTEXT = 64          # v5 prepends the previous 64 samples; omit it and every
                      # probability collapses to ~0 on real speech (measured 2026-07-29)
_sess = None


def _session():
    global _sess
    if _sess is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _sess = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    return _sess


def speech_probs(wav16k):
    """Per-frame speech probability. `wav16k` must be mono float32 at 16 kHz."""
    sess = _session()
    x = np.asarray(wav16k, dtype=np.float32)
    n = len(x) // WINDOW
    if n == 0:
        return np.zeros(0, dtype=np.float32), WINDOW / SR
    # Silero v5 carries its own recurrent state between frames; it must be threaded
    # through in order, and reset per clip or one file's tail biases the next one's head.
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT, dtype=np.float32)
    sr = np.array(SR, dtype=np.int64)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        chunk = x[i * WINDOW:(i + 1) * WINDOW]
        frame = np.concatenate([context, chunk]).reshape(1, -1)
        p, state = sess.run(None, {"input": frame, "state": state, "sr": sr})
        out[i] = float(p[0][0])
        context = chunk[-CONTEXT:]
    return out, WINDOW / SR


def speech_intervals(wav16k, threshold=0.5, min_speech_s=0.10, min_gap_s=0.10):
    """[(start_s, end_s), ...] of speech. Short gaps are bridged, not split.

    `min_gap_s` matters: without it every inter-word pause becomes a boundary and the
    "speech duration" undercounts badly. Bridging keeps a clip whole.
    """
    p, dt = speech_probs(wav16k)
    if not len(p):
        return []
    on = p >= threshold
    spans, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append([start, i])
            start = None
    if start is not None:
        spans.append([start, len(on)])
    merged = []
    for s in spans:
        if merged and (s[0] - merged[-1][1]) * dt <= min_gap_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a * dt, b * dt) for a, b in merged if (b - a) * dt >= min_speech_s]


def speech_duration(wav16k, **kw):
    """Total seconds of actual speech."""
    return float(sum(e - s for s, e in speech_intervals(wav16k, **kw)))


def long_silences(wav16k, min_s=1.0, **kw):
    """Internal silences >= min_s, as (start_s, end_s). Leading/trailing excluded.

    Catches the Zonos emotion-vector instability directly: dropped words and long gaps
    mid-utterance. Leading/trailing are excluded because those are a trim question, not a
    performance defect — see `trim_bounds`.
    """
    iv = speech_intervals(wav16k, **kw)
    return [(iv[i][1], iv[i + 1][0]) for i in range(len(iv) - 1)
            if iv[i + 1][0] - iv[i][1] >= min_s]


def trim_bounds(wav16k, pad_s=0.10, **kw):
    """(start_s, end_s) of speech with a little padding, or None if no speech found.

    Padding is deliberate: cutting hard to the VAD boundary clips plosive onsets and
    breath tails, which is content removal by the back door.
    """
    iv = speech_intervals(wav16k, **kw)
    if not iv:
        return None
    return (max(0.0, iv[0][0] - pad_s), iv[-1][1] + pad_s)
