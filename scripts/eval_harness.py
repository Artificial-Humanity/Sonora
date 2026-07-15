"""Standing eval harness: controllability, identity leakage, intelligibility.

The objective half of the listen->iterate loop (north star §7) and the gate
for every directability experiment: run it over a directory of rendered WAVs
described by a JSONL manifest, get pass/fail against pre-registered
thresholds. Reusable across channels (energy, duration, f0, ...) and across
Actor checkpoints — nothing here knows how the audio was made.

Manifest: one JSON object per line, keys:
    wav       path (relative to the manifest's directory or absolute)
    text      reference transcript (for WER)
    group     sweep id — rows sharing a group form one controllability sweep
    requested {"<channel>": value} — the control value(s) asked of the render
    baseline  true on the sweep's neutral row (identity-drift reference)

Metrics per group:
    controllability  Spearman rho(requested, produced) per channel; produced
                     measures: energy->LUFS loudness, duration->seconds,
                     f0->pyin median Hz
    identity         ECAPA-TDNN cosine drift from the baseline row across the
                     sweep; with --speaker-refs A.wav B.wav the drift is
                     normalized by the real inter-speaker gap -> leakage ratio
    wer              faster-whisper WER per row; guardrail = worst row vs
                     baseline row delta

Pre-registered thresholds (north star §7; override via flags only with a
written reason): rho >= 0.9, leakage <= 0.2, WER delta <= +0.10.

Usage:
    python scripts/eval_harness.py manifest.jsonl [--speaker-refs A.wav B.wav]
        [--skip-identity] [--skip-wer] [--report out.json]

Eval-only deps (NOT in requirements.txt — keep the training image lean):
    uv pip install soundfile pyloudnorm librosa faster-whisper speechbrain
"""

import argparse
import json
import math
import os
import sys

import numpy as np

RHO_MIN = 0.9
LEAKAGE_MAX = 0.2
WER_DELTA_MAX = 0.10

MEASURES = {"energy": "loudness_lufs", "duration": "seconds", "f0": "f0_median_hz"}


def load_wav(path):
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def loudness_lufs(wav, sr):
    try:
        import pyloudnorm
        meter = pyloudnorm.Meter(sr)
        return float(meter.integrated_loudness(wav))
    except Exception:  # short clip or missing dep: log-RMS fallback
        rms = float(np.sqrt(np.mean(wav**2)))
        return 20 * math.log10(max(rms, 1e-9))


def f0_median_hz(wav, sr):
    import librosa

    f0, _, _ = librosa.pyin(wav, fmin=65, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else float("nan")


def _ranks(x):
    # Tie-averaged ranks. Positional tie-breaking (plain argsort-of-argsort)
    # is a trap here: a sweep whose produced values are all IDENTICAL (e.g. a
    # zero-init FiLM model ignoring the control) would get ranks 0..n-1 in
    # manifest order and score a spurious rho=1.0 against the sorted requested
    # values. Averaged ties make that case zero-variance -> nan -> FAIL.
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        tied = x == v
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    if len(ra) < 2 or np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def norm_words(text):
    import re

    return re.sub(r"[^a-z' ]", "", text.lower().replace("-", " ")).split()


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[-1, -1]) / max(len(r), 1)


class Whisper:
    def __init__(self):
        from faster_whisper import WhisperModel

        self.model = WhisperModel("base.en", device="cpu", compute_type="int8")

    def transcribe(self, wav):
        segs, _ = self.model.transcribe(wav, beam_size=5)
        return " ".join(s.text.strip() for s in segs).strip()


class Ecapa:
    def __init__(self):
        import torch  # noqa: F401
        from speechbrain.inference.speaker import EncoderClassifier

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.cache/sonora/spkrec-ecapa-voxceleb"),
        )

    def embed(self, wav, sr):
        import torch
        import torchaudio

        t = torch.from_numpy(wav).unsqueeze(0)
        if sr != 16000:
            t = torchaudio.functional.resample(t, sr, 16000)
        emb = self.model.encode_batch(t).squeeze()
        return emb.detach().cpu().numpy()


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--speaker-refs", nargs=2, metavar=("A.wav", "B.wav"),
                    help="two different real speakers; enables the leakage "
                         "ratio (identity drift / inter-speaker gap)")
    ap.add_argument("--skip-identity", action="store_true")
    ap.add_argument("--skip-wer", action="store_true")
    ap.add_argument("--report", default=None)
    ap.add_argument("--rho-min", type=float, default=RHO_MIN)
    ap.add_argument("--leakage-max", type=float, default=LEAKAGE_MAX)
    ap.add_argument("--wer-delta-max", type=float, default=WER_DELTA_MAX)
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args.manifest))
    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    for r in rows:
        if not os.path.isabs(r["wav"]):
            r["wav"] = os.path.join(base_dir, r["wav"])

    whisper = None if args.skip_wer else Whisper()
    ecapa = None if args.skip_identity else Ecapa()

    inter_speaker_gap = None
    if ecapa and args.speaker_refs:
        embs = []
        for p in args.speaker_refs:
            w, sr = load_wav(p)
            embs.append(ecapa.embed(w, sr))
        inter_speaker_gap = 1.0 - cos(embs[0], embs[1])

    # Per-row measurements.
    for r in rows:
        wav, sr = load_wav(r["wav"])
        r["_measures"] = {
            "loudness_lufs": loudness_lufs(wav, sr),
            "seconds": len(wav) / sr,
        }
        if any(ch == "f0" for ch in r.get("requested", {})):
            r["_measures"]["f0_median_hz"] = f0_median_hz(wav, sr)
        if whisper:
            hyp = whisper.transcribe(wav)
            r["_wer"] = wer(r["text"], hyp)
            r["_transcript"] = hyp
        if ecapa:
            r["_emb"] = ecapa.embed(wav, sr)

    groups = {}
    for r in rows:
        groups.setdefault(r.get("group", "default"), []).append(r)

    report = {"groups": {}, "inter_speaker_gap": inter_speaker_gap,
              "thresholds": {"rho_min": args.rho_min,
                             "leakage_max": args.leakage_max,
                             "wer_delta_max": args.wer_delta_max}}
    failed = []
    for name, grp in sorted(groups.items()):
        g = {"n": len(grp)}
        channels = sorted({ch for r in grp for ch in r.get("requested", {})})
        g["controllability"] = {}
        for ch in channels:
            pts = [(r["requested"][ch], r["_measures"].get(MEASURES.get(ch, "seconds")))
                   for r in grp if ch in r.get("requested", {})]
            pts = [(a, b) for a, b in pts if b is not None and not math.isnan(b)]
            if len(pts) >= 3:
                rho = spearman([p[0] for p in pts], [p[1] for p in pts])
                produced = [p[1] for p in pts]
                g["controllability"][ch] = {
                    "spearman_rho": rho,
                    "produced_range": [min(produced), max(produced)],
                    "pass": bool(abs(rho) >= args.rho_min),
                }
                if not g["controllability"][ch]["pass"]:
                    failed.append(f"{name}:{ch} rho={rho:.3f}")
        base = next((r for r in grp if r.get("baseline")), grp[0])
        if whisper:
            worst = max(r["_wer"] for r in grp)
            delta = worst - base["_wer"]
            g["wer"] = {"baseline": base["_wer"], "worst": worst,
                        "delta": delta,
                        "pass": bool(delta <= args.wer_delta_max)}
            if not g["wer"]["pass"]:
                failed.append(f"{name}: WER delta {delta:.3f}")
        if ecapa and len(grp) > 1:
            drift = max(1.0 - cos(base["_emb"], r["_emb"])
                        for r in grp if r is not base)
            g["identity"] = {"max_drift": drift}
            if inter_speaker_gap:
                leak = drift / inter_speaker_gap
                g["identity"]["leakage_ratio"] = leak
                g["identity"]["pass"] = bool(leak <= args.leakage_max)
                if not g["identity"]["pass"]:
                    failed.append(f"{name}: leakage {leak:.3f}")
        report["groups"][name] = g

    for r in rows:
        r.pop("_emb", None)
    report["rows"] = [{k: v for k, v in r.items() if k != "_emb"} for r in rows]
    out = json.dumps(report, indent=2, default=float)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"report -> {args.report}")

    for name, g in report["groups"].items():
        print(f"[{name}] n={g['n']}")
        for ch, c in g.get("controllability", {}).items():
            print(f"  {ch}: rho={c['spearman_rho']:.3f} "
                  f"range=[{c['produced_range'][0]:.2f}, "
                  f"{c['produced_range'][1]:.2f}] "
                  f"{'PASS' if c['pass'] else 'FAIL'}")
        if "wer" in g:
            print(f"  wer: base={g['wer']['baseline']:.3f} "
                  f"worst={g['wer']['worst']:.3f} "
                  f"{'PASS' if g['wer']['pass'] else 'FAIL'}")
        if "identity" in g:
            ident = g["identity"]
            leak = ident.get("leakage_ratio")
            print(f"  identity: drift={ident['max_drift']:.4f}"
                  + (f" leakage={leak:.3f} "
                     f"{'PASS' if ident['pass'] else 'FAIL'}" if leak is not None
                     else " (no speaker refs -> drift only)"))
    if failed:
        print("\nFAILED:", "; ".join(failed))
    else:
        print("\nALL GATES PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
