# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "soundfile", "scipy"]
# ///
"""Detector for the two "reverb" artifacts the owner keeps hearing (2026-07-28).

WHY THIS EXISTS. The owner reported a reverb-like defect across Chatterbox, Zonos
and Orpheus auditions and noted it was "not pervasive to a particular model". It is
in fact TWO different defects that sound similar, and `qc_gate.py` catches neither:
of the 23 revisit-v1 clips the owner flagged, 20 passed every existing gate. DNSMOS
barely moves (bak 4.10 defective vs 4.21 clean) because neither defect is noise.

  ROOM     genuine reverberation — low C50, shallow decay after speech offsets.
           Orpheus/`tara`, 16 of 16 clips. Sounds like "a small enclosed room with
           hard floors".
  DOUBLE   a second voice layered over the first — the OPPOSITE acoustic signature
           (C50 *higher*, decay *steeper*), detected as envelope self-similarity at
           a fixed short lag. Chatterbox and Zonos. Sounds "ghostly", "multi-voice".

Because the signatures are opposed, a single "reverb score" gets DOUBLE backwards.
That is the main thing this file exists to prevent.

⚠ KNOWN BLIND SPOT (owner, 2026-07-29). The DOUBLE detector below MISSES the Chatterbox
voice-split entirely. The owner hears the voice break into a layered double on individual
EMPHASISED syllables — "how long", "make", "oh, the third door!" — and a one-second break
inside a seven-second clip averages away to nothing in any whole-clip statistic. On the
three clips where the owner confirmed splitting by ear, env_ac returned 0.022 / 0.020 /
0.012: its three CLEANEST scores in the probe. A clean run of this script is NOT evidence
that a Chatterbox bank is free of splitting.

SEVEN measurement approaches have now been tried against this defect family and SEVEN have
failed. The full list, so nobody spends another day rediscovering them:

  1. whole-clip envelope autocorrelation (env_ac) — scored the owner's three confirmed
     splits 0.022/0.020/0.012, its CLEANEST numbers. Whole-clip stats cannot see a
     one-second local event.
  2. local CPP dip — localized correctly (worst dip inside the owner's window on 4 of 4)
     but creaky phrase-ends dip just as deep. No separation.
  3. emphasis-subharmonic delta — separated all 6 probe clips, then collapsed to 20%
     precision on 40 labelled clips, where the top scorers were owner-marked clean.
  4. median reference F0 as a casting proxy — refuted by blind test: a 227 Hz reference
     split and a 335 Hz one did not.
  5. per-voice C50 — identified Orpheus `tara` correctly and condemned `zoe` falsely;
     a breathy voice reads as reverberant.
  6. spectral rolloff, for the "electronic / phone call" precursor — BACKWARDS: those
     clips carry MORE high-frequency energy than clean ones, not less.
  7. envelope roughness (30-300 Hz AM), for the "crackle at loudness edges" description —
     BACKWARDS: the smoothest voice (`jess`) scored the HIGHEST roughness, and within
     `mia`/`leah` clean and defective ranges overlap completely.
  8. ASR insertion rate, for VibeVoice ad-libbing (`qc_gate.align_ops`) — all SIX
     stress-v1 clips with owner-confirmed invented content scored ins_rate 0.00, while the
     only non-zero insertions landed on clips with no ad-lib. Structural, not a threshold:
     VV's liberties are NON-LEXICAL (crowd cheer, piano, reverb, a second actor), and ASR
     is trained to suppress exactly that. Do not retry with a bigger ASR model.

Only ONE measurable quantity has survived contact with ear-verified labels: reference pitch
EXCURSION (p90-p10), and it is a casting RISK FACTOR — it predicts which references are
dangerous — not a detector of whether a given clip is defective. See
synth_chatterbox.MAX_REF_EXCURSION.

If you attempt an eighth, two rules. The target is a LOCAL event at pitch/loudness peaks
on a continuum (clean -> "electronic"/crackle -> layered), so a whole-clip statistic is
disqualified before you start. And you need ear-verified labels collected while listening
for splitting SPECIFICALLY — the revisit-v1 "clean" notes mean "not flagged" by an audit
that was not listening for it, and are not valid negatives.

WHAT IT IS AND IS NOT. This is a **triage instrument, not a gate**. Validated against
the owner's 60 revisit-v1 notes, the doubling flag reaches 86% recall at 60% precision;
the room flag does not survive an absolute threshold at all — Chatterbox and Zonos clips
sit as low as C50 7.7 dB with nothing audibly wrong, while Orpheus room clips reach 16.8.
So ROOM is scored **within engine**, as an outlier against that engine's own cohort, and
even then it is a "listen to this one first" signal. Nothing here should auto-drop a clip;
`--rank` exists so the owner's ear reaches the worst candidates first.

Know the per-clip ROOM score's blind spot: it cannot see a defect that dominates its own
cohort. On revisit-v1 it flags only 2 of `tara`'s 16 reverberant clips, because with 16 of
20 affected the Orpheus median is itself reverberant and there is nothing to be an outlier
against. That failure mode is exactly why the run ends with a **per-engine C50 table** —
across cohorts the same defect is unmissable (orpheus 12.7 dB vs chatterbox 16.5, zonos
16.0). Read the table first and the per-clip flags second.

The durable fixes are upstream and already applied — MAX_REF_F0 in synth_chatterbox.py,
MAX_SPEAKING_RATE in synth_zonos.py, and the `tara` ban in director_skills/orpheus.md.
This script is the net under them, for defects those three do not anticipate.

Usage:
    .venv/bin/python scripts/synthesis/qc_artifacts.py --dir <campaign_dir> [--rank] [--json out.jsonl]
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt

# Envelope self-similarity above this reads as a layered second voice. Absolute
# (not within-engine) because it validated that way: 6 of 7 owner-flagged doubles
# clear it, against 4 false positives in 53 clean clips.
DOUBLE_AC = 0.10
# Within-engine robust z-score past which a clip is called reverberant for its cohort.
ROOM_Z = 1.0


def load(path):
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), sr


def _frames(x, sr, win_ms=25.0, hop_ms=10.0):
    w, h = int(sr * win_ms / 1000), int(sr * hop_ms / 1000)
    n = 1 + max(0, (len(x) - w) // h)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return x[idx] * np.hanning(w)[None, :], h / sr


def _energy_db(x, sr):
    """Band-limited frame energy — 300-4000 Hz, where the decay is cleanest."""
    sos = butter(4, [300 / (sr / 2), min(4000, sr / 2 - 1) / (sr / 2)],
                 btype="band", output="sos")
    f, dt = _frames(sosfilt(sos, x), sr)
    return 10 * np.log10(np.maximum((f ** 2).mean(axis=1), 1e-12)), dt


def decay_metrics(x, sr):
    """(median free-decay slope dB/s, median C50-like early/late ratio dB).

    Every speech offset is a free decay. Reverberation makes them shallow and
    smooth; a dry recording falls off a cliff. Measuring the offsets directly
    avoids needing a reference or an impulse.
    """
    e, dt = _energy_db(x, sr)
    if len(e) < 30:
        return None, None
    floor, top = np.percentile(e, 5), np.percentile(e, 95)
    if top - floor < 12:              # no dynamic range: nothing to measure
        return None, None
    hi, lo = top - 6.0, floor + 6.0   # start from a loud frame, stop above the noise
    slopes, c50, i, span = [], [], 1, int(0.30 / dt)
    while i < len(e) - 3:
        if e[i] > hi and e[i + 1] < e[i]:
            j = i
            while j + 1 < len(e) and j - i < span and e[j + 1] < e[j] and e[j + 1] > lo:
                j += 1
            if (j - i) >= 4:          # a real decay, not one frame of jitter
                seg = e[i:j + 1]
                slopes.append(np.polyfit(np.arange(len(seg)) * dt, seg, 1)[0])
                k = min(int(0.05 / dt), j - i)
                early = np.sum(10 ** (seg[:k + 1] / 10))
                late = np.sum(10 ** (seg[k + 1:] / 10))
                if late > 0:
                    c50.append(10 * np.log10(early / late))
                i = j
        i += 1
    if not slopes:
        return None, None
    return float(np.median(slopes)), (float(np.median(c50)) if c50 else None)


def doubling(x, sr):
    """(envelope-autocorrelation peak prominence, its lag in ms).

    A second offset copy of the same voice makes the amplitude envelope self-similar
    at the offset delay. The envelope carries no F0 structure, so this does not fire
    on the periodicity of the voice itself. Prominence is measured against a fitted
    trend because the autocorrelation falls off smoothly and a raw peak would just
    rediscover that slope.
    """
    e, dt = _energy_db(x, sr)
    if len(e) < 60:
        return None, None
    e = e - e.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    if ac[0] <= 0:
        return None, None
    ac = ac / ac[0]
    lo, hi = int(0.03 / dt), min(int(0.30 / dt), len(ac) - 1)
    if hi - lo < 6:
        return None, None
    seg = ac[lo:hi]
    t = np.arange(len(seg))
    resid = seg - np.polyval(np.polyfit(t, seg, 2), t)
    k = int(np.argmax(resid))
    return float(resid[k]), float((lo + k) * dt * 1000)


def engine_of(path, manifests):
    return manifests.get(os.path.basename(path), {}).get("engine", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="campaign dir (or its audio/ subdir)")
    ap.add_argument("--json", default=None, help="write per-clip rows here")
    ap.add_argument("--rank", action="store_true", help="print worst-first, not by name")
    args = ap.parse_args()

    root = args.dir
    audio = os.path.join(root, "audio")
    if os.path.isdir(audio):
        root = audio
    # C-M9. The recursive glob swept `_dropped/` and `_superseded/` as well, so clips the
    # ear had ALREADY REJECTED were folded into the cohort statistics — and ROOM is scored
    # as a robust z against the cohort's own median/MAD, which means the rejects were
    # helping define what "normal" looks like for the clips still under judgement. On a
    # cohort quarantined precisely because it was reverberant, that shifts the median
    # toward the defect and hides the next one.
    #
    # `_pre_loudnorm` was already excluded here by substring, which is the same rule; the
    # other two quarantine directories were simply missed when the drop-quarantine
    # convention landed on 2026-07-31.
    EXCLUDED_DIRS = ("_pre_loudnorm", "_dropped", "_superseded")
    wavs = sorted(p for p in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
                  if not any(d in p.split(os.sep) for d in EXCLUDED_DIRS))
    if not wavs:
        sys.exit(f"no wavs under {root}")

    # engine per clip, so ROOM can be judged against the right cohort.
    #
    # C-M9, second half: this glob is NOT recursive while the wav glob above is, so a
    # campaign laid out as <campaign>/<engine-out>/ — which is the layout synth_bank.sh
    # writes and the QC gate assumes — found no manifests at all. `engine_of` then fell
    # back to "?" for every clip, and the whole campaign became ONE cohort named "?".
    # The per-engine comparison is the entire point of this tool, and it was silently
    # not happening; a "?" row looks like missing metadata rather than a broken join.
    manifests = {}
    for mf in glob.glob(os.path.join(root, "**", "*_manifest.jsonl"), recursive=True):
        if any(d in mf.split(os.sep) for d in EXCLUDED_DIRS):
            continue
        for line in open(mf, encoding="utf-8"):
            try:
                r = json.loads(line)
                manifests[r.get("wav", "")] = r
            except Exception:
                pass

    rows = []
    for p in wavs:
        x, sr = load(p)
        slope, c50 = decay_metrics(x, sr)
        ac, lag = doubling(x, sr)
        rows.append({
            "file": os.path.basename(p), "engine": engine_of(p, manifests),
            "dur": round(len(x) / sr, 2),
            "decay_db_s": None if slope is None else round(slope, 1),
            "c50_db": None if c50 is None else round(c50, 1),
            "env_ac": None if ac is None else round(ac, 3),
            "env_lag_ms": None if lag is None else round(lag),
        })

    # ROOM is only meaningful relative to the engine's own cohort — see the module
    # docstring. Median/MAD rather than mean/sd so a couple of bad clips do not
    # define "normal" for the rest.
    for eng in {r["engine"] for r in rows}:
        coh = [r for r in rows if r["engine"] == eng and r["c50_db"] is not None]
        if len(coh) < 4:
            continue
        v = [r["c50_db"] for r in coh]
        med = st.median(v)
        mad = st.median([abs(a - med) for a in v]) or 1e-6
        for r in coh:
            r["room_z"] = round((med - r["c50_db"]) / (1.4826 * mad), 2)

    for r in rows:
        flags = []
        if r.get("env_ac") is not None and r["env_ac"] > DOUBLE_AC:
            flags.append("DOUBLE")
        if r.get("room_z", 0) > ROOM_Z:
            flags.append("ROOM")
        r["flags"] = flags
        r["severity"] = max(r.get("env_ac", 0) / DOUBLE_AC, r.get("room_z", 0) / ROOM_Z)

    out = sorted(rows, key=lambda r: -r["severity"]) if args.rank else rows
    print(f"{'file':34s} {'engine':11s} {'c50':>7s} {'room_z':>7s} {'env_ac':>7s} "
          f"{'lag':>5s}  flags")
    for r in out:
        print(f"{r['file']:34s} {r['engine']:11s} {str(r['c50_db']):>7s} "
              f"{str(r.get('room_z', '-')):>7s} {str(r['env_ac']):>7s} "
              f"{str(r['env_lag_ms']):>5s}  {','.join(r['flags'])}")

    n = sum(1 for r in rows if r["flags"])
    print(f"\n{n}/{len(rows)} clip(s) flagged for a listen. "
          f"Triage only — see the module docstring before treating these as verdicts.")

    # The per-clip ROOM score is blind to a defect that dominates its own cohort:
    # `tara` reverberates in 16 of 20 Orpheus clips, so the Orpheus median is itself
    # reverberant and almost nothing stands out against it. That case is only visible
    # ACROSS cohorts, which is what this table is for. On revisit-v1 it is unmissable —
    # orpheus 12.7 dB against chatterbox 16.5 and zonos 16.0.
    coh = {}
    for r in rows:
        if r["c50_db"] is not None:
            coh.setdefault(r["engine"], []).append(r["c50_db"])
    if len(coh) > 1:
        print("\nper-engine C50 median vs the OTHER engines — a cohort >2 dB low is "
              "reverberant AS A COHORT:")
        # Each engine is scored against everyone else, not against the campaign as a
        # whole: a third of revisit-v1 IS the reverberant cohort, so including it in
        # the baseline drags the baseline down toward it and hides the gap (orpheus
        # reads -1.6 dB against the campaign, -3.4 dB against its peers).
        for eng, v in sorted(coh.items(), key=lambda kv: st.median(kv[1])):
            m = st.median(v)
            others = [x for e2, vs in coh.items() if e2 != eng for x in vs]
            if not others:
                continue
            delta = m - st.median(others)
            mark = "  <-- reverberant" if delta < -2 else ""
            print(f"  {eng:12s} n={len(v):3d}  C50 {m:5.1f} dB  ({delta:+.1f} vs peers){mark}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
