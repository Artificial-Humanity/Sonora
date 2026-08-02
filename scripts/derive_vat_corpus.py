# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy", "soundfile", "pyloudnorm", "librosa>=0.10", "numba>=0.60",
#   "unidecode", "pyyaml", "ai-edge-litert", "ml_dtypes",
# ]
# # NOT openphonemizer and NOT torch. Both were listed here until 2026-08-01 and
# # neither is imported anywhere on this path. "OpenPhonemizer" names the upstream
# # PROJECT (NeuralVox/OpenPhonemizer, Clear BSD) whose two artefacts we vendor —
# # the 275k espeak-IPA dictionary and the ForwardTransformer OOV checkpoint — not
# # its PyPI package, whose runtime we never call. The checkpoint ships here as a
# # TFLite graph run through ai-edge-litert (ml_dtypes is its undeclared runtime
# # dep), so nothing on this path needs torch at all. That is the point of the
# # espeak-free lane; listing torch here contradicted north star §8.3.
# ///
"""VAT corpus derivation over LibriTTS-R (dataset-landscape.md §Strategy).

HOW TO RUN IT (owner decision 2026-08-01): the repo **.venv**, not `uv run`.

    .venv/bin/python scripts/derive_vat_corpus.py --out data/libritts_r_vat_v3 ...

uv still MANAGES that venv (`uv pip install --python .venv/bin/python`) — nothing here
introduces bare pip. `uv run` binds to the repo pyproject.toml instead of the inline
PEP 723 block below, which is what made four launch attempts fail on 2026-08-01, each
with a different missing module. The inline block stays as a record of what is needed.

THE "sys.executable" BLOCKER WAS A MISDIAGNOSIS — resolved 2026-08-01, kept here as a
warning about the evidence. The symptom was real:

    File ".../uv/python/cpython-3.11.15-.../multiprocessing/pool.py"
    ModuleNotFoundError: No module named 'numpy'

and it was read as "the pool relaunched the BASE interpreter, so workers lost the venv's
site-packages; fix with mp.set_executable(sys.executable)". Every step of that is wrong:

  * A venv does not copy the stdlib. `multiprocessing/pool.py` ALWAYS resolves under
    sys.base_prefix, in a healthy venv as much as a broken one. That frame proves
    nothing about which interpreter is running, and it is the only "evidence" there was.
  * This pool uses **fork**, not spawn — no set_start_method call exists anywhere in the
    file. Fork inherits the parent's memory outright; sys.executable is never consulted,
    so the proposed fix was a no-op against a mechanism not in use.
  * sys.executable already resolves to .venv/bin/python. The venv was simply BARE: numpy
    had not been installed into it yet. The import failed in the worker because it would
    have failed anywhere.

Verified after installing the deps: `mp.Pool(4).imap_unordered(measure_clip, ...)`
measures 24/24 clips under .venv, start method `fork`, sys.executable inside the venv.

Fork is safe HERE specifically because nothing touches the GPU before the pool — the
espeak-free G2P is a TFLite/CPU graph. That is a property of this script, not a general
licence; fork-after-GPU still wedges gfx1151 ([[gfx1151-rocm-diffusion]]).

THE ACTUAL BLOCKER was the next line down, and it was a stale path. OpenPhonemizerG2P
resolved its assets to <repo>/../Reference/models/litert-community/Matcha-TTS — the
umbrella layout deleted 2026-07-22 — so __init__ died opening g2p_dict.txt.gz. The
assets are at /data/models/litert-community/Matcha-TTS; op_g2p.py now probes for the
dictionary instead of hard-coding one location. Full lane re-verified end to end:
274,927 dictionary entries loaded, neural OOV fallback resolving unknown words, 0 misses.

WHAT ALREADY WORKS: the logic is sound at MAX_SECONDS=22. A run on 2026-08-01 measured
all 33,232 utterances, applied the soft-json and passed the independence gate at
corr(T,A) = -0.059, failing only at the final G2P relabel.

COVERAGE GAP CLOSED 2026-08-01 (owner: "let's re-run EIV first"). The symptom was
`soft-json: 30351 scores, 1094 kept clips uncovered` — the utterances the 16 -> 22 s cap
admitted postdate the labelling run. Two things the original note got wrong:

  * It read as a TENSION problem. **Valence had the identical gap** — all 1,094 were
    missing from corpus_valence_combo.json too, so v3 would have derived them with the
    whole V channel absent, not just the tension blend's fourth component.
  * A gap is not closed by appending. The combo is dot(weights, per-speaker z(head)), so
    growing a speaker's population moves clips that were ALREADY scored; appending only
    the new rows recreates the same two-inconsistent-halves defect one level up.

So: all 1,094 scored over the same 12 heads (scripts/eiv_score.sh), then every derived
value rebuilt from the immutable raw scores by scripts/eiv_merge_corpus.py. Pass the _v2
files to --valence-json / --soft-json; the originals cover 30,351 and are superseded.

    --valence-json /data/model-training/sonora/eiv_scores/corpus_valence_combo_v2.json
    --soft-json    /data/model-training/sonora/eiv_scores/corpus_soft_v2.json

Verified: 31,445 in-band clips, 0 uncovered by either file. Worth knowing before reading
v3 labels — the newly admitted long clips sit 0.086 BELOW the corpus valence mean (about
0.18 sd), while their softness mean is identical to the corpus (+0.320 both). Whatever is
different about long utterances shows up in V, not in T. That is consistent with the
already-recorded duration/loudness finding at the head of this file, and it means the A
and V channels move together with length while T does not.


Walks a local LibriTTS-R subset, measures per-clip features, derives V/A/T
labels, phonemizes transcripts through the espeak-free op-G2P lane, remaps
speakers to contiguous ids, and emits `path|spk|ipa|v,a,t` filelists (the
`load_vat` datamodule format) plus a speaker map and a derivation report.

Label derivation v1 (owner-approved 2026-07-16; briefs:
vat-channels.md):
    A (arousal slot) = per-speaker z-score of integrated loudness (LUFS),
        clamped to [-1, 1] at 2 sigma — unchanged from v0, validated at
        ρ ≈ 1.000 in the §7 de-risk. Per-speaker normalization is load-
        bearing: global normalization would bake mic gain / speaker identity
        into the energy label — leakage by construction.
    T (tension slot) = phonation tension, pressed(+) <-> breathy(-):
        equal-weight sum of per-speaker z-scored voiced-frame measures
        — alpha ratio (1-5 kHz vs 50 Hz-1 kHz energy, dB; pressed voices are
        top-heavy), CPP (cepstral peak prominence; breathiness lowers it),
        and -(H1-H2) (first-harmonic dominance; breathy voices are
        H1-heavy) — re-z-scored per speaker, clamped at 2 sigma.
        v2 (2026-07-17, after the within-voice audit failed the + end):
        --soft-json adds -z(EIV Soft_vs._Harsh) as a fourth component — the
        acoustic trio owns the breathy end (audit-validated), EIV harshness
        (JL d'=0.65, angry/assertive=harsh) repairs the pressed end.
    V (valence slot) = EIV pseudo-labels via --valence-json: raw scores from
        the JL-calibrated 9-head combo (valence_combo_v1.json, LOSO CV
        d'=0.88 — the single EIV Valence head failed calibration at d'=0.23);
        per-speaker z-scored + clamped here, same recipe. 0.0 when absent.
    Independence gate (pre-registered): pooled per-speaker |corr| < 0.3
        between any two channels' labels, else residualize T on A and
        re-check. Verdict recorded in the derivation report.
Per-clip raw measures are dumped to measures.jsonl so calibration and later
labeling passes (EIV, Emilia mining) reuse them without re-measuring.

License wall: emitted audio paths carry the LibriTTS_R component and pass
the manifest as `libritts_r` (permissive). Run from the Sonora repo root:
    python scripts/derive_vat_corpus.py [--root DIR] [--out DIR] [--workers N]
        [--valence-json eiv_valence.json]
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DEFAULT_ROOT = "/data/model-training/datasets/LibriTTS_R/train-clean-100"
MIN_SECONDS = 1.0
# Raised 16.0 -> 22.0 (owner, 2026-08-01) so the base corpus stops disagreeing with the
# teacher corpus about length. Our synthesized clips average 12.5 s against LibriTTS-R's
# 5.9 s, and ~20% of them exceeded the old ceiling — the two halves of the corpus were
# built to different length policies and only one was written down. No rationale for 16.0
# was ever recorded; this note exists so 22.0 does not repeat that.
#
# What 22.0 buys, measured on a 5,000-utterance sample of train-clean-100: LibriTTS-R has
# 4.2% of utterances over 16 s and 1.8% over 20 s, with a corpus maximum of 30.6 s. So this
# recovers roughly 790 utterances (~2.4%), and almost nothing exists past 25 s at any
# setting. It is an arc-diversity change, NOT a long-form-capability change: Matcha is
# non-autoregressive (flow-matching decoder, MAS at training only), so inference length is
# bounded by memory rather than by anything learned about length. Long-form reading is a
# cross-chunk continuity problem, not a clip-length one.
#
# ⚠ TWO CONSEQUENCES, both load-bearing:
#   1. All three VAT channels are PER-SPEAKER z-scores clamped at 2 sigma, so changing the
#      population changes every existing label. This is a corpus version bump (v2 -> v3)
#      and a retrain, not a config tweak — the vat3 checkpoint was fitted to v2 labels.
#   2. Duration correlates with integrated loudness at r = -0.385 (longer utterances are
#      quieter; 12-16 s clips sit 1.28 LU below the sub-12 s body, ~0.69 sigma). Since
#      A = per-speaker z(LUFS), longer utterances carry systematically lower arousal.
#      TESTED AND REJECTED (2026-08-01): the obvious explanation — that long reads hold
#      more pause and pause drags integrated loudness down — is WRONG. Measured on 600
#      clips (300 <8 s, 300 >=12 s), gating loudness to speech frames only left the
#      correlation unchanged (-0.415 whole-file vs -0.430 speech-only) and the short/long
#      gap identical at -1.46 LU; speech fraction differs by just 3.3% (0.979 vs 0.946).
#      Long utterances are genuinely quieter WHILE SPEAKING. That is plausibly real
#      acoustics — intensity declination across a long utterance, and LibriTTS's short
#      segments being disproportionately exclamations and dialogue — in which case A is
#      reporting a true difference rather than a length artifact. Do not "fix" this
#      without first establishing which it is; a speech-gated meter is not the answer.
MAX_SECONDS = 22.0
SAMPLE_RATE = 24000
VAL_FRACTION = 0.03
SEED = 1234


def find_clips(root):
    """Yields (wav_path, normalized_text, speaker_id) for every utterance."""
    for spk in sorted(os.listdir(root)):
        spk_dir = os.path.join(root, spk)
        if not os.path.isdir(spk_dir):
            continue
        for chapter in sorted(os.listdir(spk_dir)):
            ch_dir = os.path.join(spk_dir, chapter)
            if not os.path.isdir(ch_dir):
                continue
            for name in sorted(os.listdir(ch_dir)):
                if not name.endswith(".wav"):
                    continue
                txt = os.path.join(ch_dir, name[:-4] + ".normalized.txt")
                if not os.path.exists(txt):
                    continue
                with open(txt, encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    yield os.path.join(ch_dir, name), text, spk



FRAME = 2048   # 85 ms @ 24 kHz — long enough for cepstral f0 down to ~60 Hz
HOP = 512
F0_MIN, F0_MAX = 60.0, 400.0


def phonation_measures(wav, sr):
    """Voiced-frame phonation measures for the tension composite
    (vat-channels.md): alpha ratio, CPP, H1-H2. numpy only.
    Returns None when too few usable voiced frames."""
    import numpy as np

    if len(wav) < FRAME * 2:
        return None
    n_frames = 1 + (len(wav) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = wav[idx] * np.hanning(FRAME)[None, :]

    # Speech-active frames: within 30 dB of the clip's loudest frame.
    frame_rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    active = 20 * np.log10(frame_rms) > 20 * np.log10(frame_rms.max()) - 30
    if active.sum() < 10:
        return None
    mag = np.abs(np.fft.rfft(frames[active], axis=1))
    log_mag = 20 * np.log10(mag + 1e-9)

    # Real cepstrum; f0 search band in quefrency.
    cep = np.fft.irfft(log_mag, axis=1)
    q_lo, q_hi = int(sr / F0_MAX), int(sr / F0_MIN)
    band = cep[:, q_lo:q_hi]
    peak_rel = band.argmax(axis=1)
    peak_q = peak_rel + q_lo
    # CPP: peak height above a linear regression of the cepstrum across the
    # band, evaluated at the peak quefrency (Hillenbrand-style, simplified).
    x = np.arange(q_lo, q_hi, dtype=np.float64)
    xm, ym = x.mean(), band.mean(axis=1)
    slope = ((x - xm)[None, :] * (band - ym[:, None])).sum(axis=1) / ((x - xm) ** 2).sum()
    baseline = ym + slope * (peak_q - xm)
    cpp = band[np.arange(len(band)), peak_rel] - baseline

    # Voiced = clearly periodic frames; keep the most-periodic half above a
    # floor so every clip contributes comparable frame counts.
    order = np.argsort(cpp)[::-1]
    voiced = order[: max(int(len(order) * 0.5), 10)]
    voiced = voiced[cpp[voiced] > 0]
    if len(voiced) < 10:
        return None

    # H1, H2 from the frame spectrum at the cepstral f0 (nearest-bin + local
    # 3-bin max to tolerate the 11.7 Hz grid — weak-label precision).
    f0 = sr / peak_q[voiced].astype(np.float64)
    binf = FRAME / sr
    n_bins = log_mag.shape[1]

    def harm_amp(mult):
        b = np.clip(np.round(f0 * mult * binf).astype(int), 1, n_bins - 2)
        neigh = np.stack([log_mag[voiced, b - 1], log_mag[voiced, b],
                          log_mag[voiced, b + 1]], axis=1)
        return neigh.max(axis=1)

    h1h2 = float((harm_amp(1) - harm_amp(2)).mean())

    # Alpha ratio over the voiced frames' mean power spectrum.
    freqs = np.fft.rfftfreq(FRAME, 1 / sr)
    power = (mag[voiced] ** 2).mean(axis=0)
    lo = power[(freqs >= 50) & (freqs < 1000)].sum()
    hi = power[(freqs >= 1000) & (freqs < 5000)].sum()
    alpha = 10 * float(np.log10((hi + 1e-12) / (lo + 1e-12)))

    return {"alpha_db": alpha, "cpp": float(cpp[voiced].mean()), "h1h2": h1h2}


def measure_clip(args):
    """Worker: (wav_path,) -> dict of acoustic measures, or None to skip."""
    wav_path = args
    import numpy as np
    import soundfile as sf

    try:
        wav, sr = sf.read(wav_path, dtype="float32")
    except Exception:
        return wav_path, None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    seconds = len(wav) / sr
    if sr != SAMPLE_RATE or not MIN_SECONDS <= seconds <= MAX_SECONDS:
        return wav_path, None
    try:
        import pyloudnorm

        lufs = float(pyloudnorm.Meter(sr).integrated_loudness(wav))
    except Exception:
        rms = float(np.sqrt(np.mean(wav**2)))
        lufs = 20 * float(np.log10(max(rms, 1e-9)))
    if not np.isfinite(lufs) or lufs < -60:
        return wav_path, None  # silence / broken clip
    phon = phonation_measures(wav, sr)
    if phon is None:
        return wav_path, None  # not enough voiced speech to label tension
    return wav_path, {"seconds": seconds, "lufs": lufs, **phon}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="data/libritts_r_vat")
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    ap.add_argument("--no-neural-oov", action="store_true")
    ap.add_argument("--valence-json", default=None,
                    help="JSON {wav_path: raw_valence} from the EIV labeling "
                         "run; z-scored per speaker here. Absent -> V=0.")
    ap.add_argument("--soft-json", default=None,
                    help="JSON {wav_path: raw EIV Soft_vs._Harsh}; enables "
                         "the tension-v2 blend (-z(soft) fourth component).")
    ap.add_argument("--reuse-from", default=None,
                    help="existing derivation dir: reuse its kept clips, "
                         "phonemes (filelists) and measures.jsonl — relabel "
                         "only, no audio measuring or G2P.")
    ap.add_argument("--allow-uncovered", action="store_true",
                    help="proceed when the valence/soft JSONs do not cover "
                         "every kept clip. Uncovered clips get V=0.0, which "
                         "is indistinguishable from a genuine at-speaker-mean "
                         "label — this is how 1,094 mislabeled clips shipped "
                         "in an earlier pass. Off by default.")
    ap.add_argument("--allow-gate-fail", action="store_true",
                    help="write filelists even when the independence gate "
                         "fails. Off by default.")
    args = ap.parse_args()

    ipa_cache = None
    if args.reuse_from:
        with open(os.path.join(args.reuse_from, "speakers.json"), encoding="utf-8") as f:
            idx_to_id = {v: k for k, v in json.load(f)["libritts_id_to_index"].items()}
        measured = {}
        with open(os.path.join(args.reuse_from, "measures.jsonl"), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                measured[d.pop("wav")] = d
        kept, ipa_cache = [], {}
        for name in ("train_op.txt", "val_op.txt"):
            with open(os.path.join(args.reuse_from, name), encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    p, sidx, ipa, _vat = line.strip().split("|")
                    kept.append((p, None, idx_to_id[int(sidx)]))
                    ipa_cache[p] = ipa
        clips = kept
        print(f"reused {len(kept)} clips + measures + phonemes from {args.reuse_from}")
    else:
        clips = list(find_clips(args.root))
        print(f"found {len(clips)} utterances with transcripts under {args.root}")

        with mp.Pool(args.workers) as pool:
            measured = dict(pool.imap_unordered(measure_clip, [c[0] for c in clips], chunksize=64))
        kept = [(p, t, s) for p, t, s in clips if measured.get(p)]
        print(f"kept {len(kept)} after duration/rate/loudness filters "
              f"({len(clips) - len(kept)} dropped)")

    # Per-speaker z-scores (v1): A from LUFS; T from the phonation composite
    # (z(alpha) + z(cpp) - z(h1h2), re-z-scored); V from --valence-json when
    # provided. All clamped at 2 sigma.
    import numpy as np

    uncovered = {}

    valence_raw = {}
    if args.valence_json:
        with open(args.valence_json, encoding="utf-8") as f:
            valence_raw = json.load(f)
        missing = sum(1 for p, _, _ in kept if p not in valence_raw)
        print(f"valence-json: {len(valence_raw)} scores, {missing} kept clips uncovered")
        if missing:
            uncovered["valence"] = missing

    speakers = sorted({s for _, _, s in kept}, key=int)
    spk_index = {s: i for i, s in enumerate(speakers)}

    def per_spk_z(values_by_path):
        """{path: raw} -> {path: per-speaker z}, using kept's speaker map."""
        groups = {}
        for p, _, s in kept:
            if p in values_by_path:
                groups.setdefault(s, []).append(values_by_path[p])
        stats = {s: (float(np.mean(v)), float(np.std(v) + 1e-6)) for s, v in groups.items()}
        return {p: (values_by_path[p] - stats[s][0]) / stats[s][1]
                for p, _, s in kept if p in values_by_path}

    soft_raw = {}
    if args.soft_json:
        with open(args.soft_json, encoding="utf-8") as f:
            soft_raw = json.load(f)
        missing = sum(1 for p, _, _ in kept if p not in soft_raw)
        print(f"soft-json: {len(soft_raw)} scores, {missing} kept clips uncovered")
        if missing:
            uncovered["soft"] = missing

    if uncovered and not args.allow_uncovered:
        detail = ", ".join(f"{k}: {n} clips" for k, n in sorted(uncovered.items()))
        sys.exit(
            f"ABORT: label sources do not cover every kept clip ({detail}).\n"
            "  Uncovered clips would be written V=0.0, which reads as a real "
            "at-speaker-mean label and is invisible to every downstream gate.\n"
            "  Re-run the EIV pass over the missing clips, or pass "
            "--allow-uncovered if the zeros are genuinely intended."
        )

    lufs_z = per_spk_z({p: measured[p]["lufs"] for p, _, _ in kept})
    t_raw = {}
    for name, sign in (("alpha_db", 1.0), ("cpp", 1.0), ("h1h2", -1.0)):
        z = per_spk_z({p: measured[p][name] for p, _, _ in kept})
        for p, v in z.items():
            t_raw[p] = t_raw.get(p, 0.0) + sign * v
    if soft_raw:  # tension v2: EIV harshness repairs the pressed (+) end
        z = per_spk_z({p: soft_raw[p] for p, _, _ in kept if p in soft_raw})
        for p, v in z.items():
            t_raw[p] = t_raw.get(p, 0.0) - v
    tension_z = per_spk_z(t_raw)
    valence_z = per_spk_z(valence_raw) if valence_raw else {}

    def clamp2(z):
        return max(-1.0, min(1.0, z / 2.0))

    # A single NaN anywhere in a raw score poisons its speaker's mean/std, and
    # clamp2 turns the resulting NaN into a clean-looking 1.0 (min(1.0, nan)
    # returns 1.0 in Python, because the comparison is False). The whole
    # channel for that speaker would saturate at the extreme and format as
    # "1.0000". LUFS is isfinite-guarded upstream; the EIV JSONs are not.
    nonfinite = {
        name: sorted(p for p, v in table.items() if not np.isfinite(v))
        for name, table in (("A (lufs)", lufs_z), ("T (tension)", tension_z),
                            ("V (valence)", valence_z))
        if table
    }
    nonfinite = {k: v for k, v in nonfinite.items() if v}
    if nonfinite:
        lines = "\n".join(
            f"  {ch}: {len(paths)} non-finite z-scores, e.g. {paths[:3]}"
            for ch, paths in sorted(nonfinite.items())
        )
        sys.exit("ABORT: non-finite label values before clamping:\n" + lines
                 + "\n  These would clamp to a clean 1.0 and saturate the "
                   "channel. Fix the raw scores.")

    # Pre-registered independence gate: pooled per-speaker-z correlations.
    def corr(a, b):
        common = [p for p, _, _ in kept if p in a and p in b]
        if len(common) < 100:
            return None
        return float(np.corrcoef([a[p] for p in common], [b[p] for p in common])[0, 1])

    corr_ta = corr(tension_z, lufs_z)
    corr_tv = corr(tension_z, valence_z) if valence_z else None
    corr_va = corr(valence_z, lufs_z) if valence_z else None
    gate_ok = all(c is None or abs(c) < 0.3 for c in (corr_ta, corr_tv, corr_va))

    def _fmt(c):
        # corr() returns None on small populations; the old +.3f crashed there.
        return "n/a" if c is None else f"{c:+.3f}"

    print(f"independence gate: corr(T,A)={_fmt(corr_ta)}"
          + (f" corr(T,V)={_fmt(corr_tv)} corr(V,A)={_fmt(corr_va)}" if valence_z else "")
          + f" -> {'PASS (|r|<0.3)' if gate_ok else 'FAIL — residualize before training'}")
    if not gate_ok and not args.allow_gate_fail:
        sys.exit(
            "ABORT: independence gate failed. The channels are entangled, so "
            "a model trained on these labels cannot be steered on one axis "
            "without moving another.\n"
            "  Residualize before training, or pass --allow-gate-fail to write "
            "the filelists anyway."
        )

    # Phonemize (espeak-free lane) and assemble rows.
    from matcha.text.op_g2p import OpenPhonemizerG2P
    from matcha.data.license_wall import enforce as license_check  # noqa: F401

    def label(p):
        v = clamp2(valence_z[p]) if p in valence_z else 0.0
        return f"{v:.4f},{clamp2(lufs_z[p]):.4f},{clamp2(tension_z[p]):.4f}"

    if ipa_cache is not None:
        rows = [f"{p}|{spk_index[s]}|{ipa_cache[p]}|{label(p)}" for p, _, s in kept]
        print(f"relabeled {len(rows)} rows (phonemes reused)")
    else:
        g2p = OpenPhonemizerG2P(use_neural_oov=not args.no_neural_oov)
        rows, bad_vocab = [], 0
        for i, (p, text, s) in enumerate(kept):
            ipa = g2p.phonemize(text)
            if g2p.validate(ipa):
                bad_vocab += 1
                continue
            rows.append(f"{p}|{spk_index[s]}|{ipa}|{label(p)}")
            if (i + 1) % 5000 == 0:
                print(f"  phonemized {i + 1}/{len(kept)}")
        print(f"phonemized {len(rows)} rows ({bad_vocab} dropped for vocab violations)")
        s_ = g2p.stats
        total = sum(s_.values())
        print(f"G2P: dict {100 * s_['dict_hits'] / max(total, 1):.2f}% | "
              f"contractions {s_['contraction_hits']} | "
              f"neural {s_['neural_hits']} | "
              f"apostrophe fallback {s_['apostrophe_fallbacks']} | "
              f"unresolved {s_['oov_misses']}")
        if s_["oov_misses"]:
            # Unresolved words are written as bare letters, which validate()
            # cannot catch (a-z are in the vocab), so they do NOT show up in
            # the bad_vocab drop count above.
            sys.exit(
                f"ABORT: {s_['oov_misses']} words were unresolved and would "
                f"ship as literal letters, e.g. {sorted(g2p.oov_words)[:10]}"
            )

    # Canonical order before shuffling, so the train/val split depends only on
    # SEED and the clip set — not on how the rows happened to be assembled.
    # --reuse-from builds them train-then-val while a fresh run uses
    # find_clips() order, so the same seed used to produce two different
    # permutations and any val-loss comparison across label versions (the
    # whole point of a relabel) was contaminated.
    rows.sort()
    random.seed(SEED)
    random.shuffle(rows)
    n_val = max(int(len(rows) * VAL_FRACTION), 1)
    os.makedirs(args.out, exist_ok=True)
    for name, part in (("val_op.txt", rows[:n_val]), ("train_op.txt", rows[n_val:])):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            f.write("\n".join(part) + "\n")
        print(f"wrote {len(part)} rows -> {os.path.join(args.out, name)}")

    with open(os.path.join(args.out, "speakers.json"), "w", encoding="utf-8") as f:
        json.dump({"n_spks": len(speakers), "libritts_id_to_index": spk_index}, f, indent=2)
    # Raw per-clip measures: calibration + later labeling passes reuse these.
    with open(os.path.join(args.out, "measures.jsonl"), "w", encoding="utf-8") as f:
        for p, _, _ in kept:
            f.write(json.dumps({"wav": p, **measured[p]}) + "\n")
    report = {
        "root": args.root,
        "derivation": (("v2" if soft_raw else "v1")
                       + ": A=per-speaker LUFS z@2sigma (v0, validated); "
                       "T=phonation composite z(alpha)+z(cpp)-z(h1h2)"
                       + ("-z(EIV soft)" if soft_raw else "")
                       + ", re-z@2sigma (pressed+ / breathy-); "
                       + ("V=EIV combo (valence_combo_v1) per-speaker z@2sigma"
                          if valence_raw else "V=0 (EIV pass pending)")),
        "utterances_found": len(clips),
        "kept": len(rows),
        "n_spks": len(speakers),
        "seconds_total": float(sum(measured[p]["seconds"] for p, _, _ in kept)),
        "filters": {"min_s": MIN_SECONDS, "max_s": MAX_SECONDS, "sample_rate": SAMPLE_RATE},
        "independence_gate": {"threshold": 0.3, "corr_TA": corr_ta,
                              "corr_TV": corr_tv, "corr_VA": corr_va,
                              "pass": gate_ok},
        "license": "LibriTTS-R CC-BY-4.0 (see configs/data_licenses.yaml)",
    }
    with open(os.path.join(args.out, "derivation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"hours: {report['seconds_total'] / 3600:.1f} | speakers: {len(speakers)}")
    print("report ->", os.path.join(args.out, "derivation_report.json"))


if __name__ == "__main__":
    main()
