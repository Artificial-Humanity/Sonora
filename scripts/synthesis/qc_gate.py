# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy", "librosa>=0.10", "numba>=0.60", "soundfile", "pyloudnorm",
#   "onnxruntime", "faster-whisper",
# ]
# ///
# NB both the interpreter ceiling and the explicit numba pin are load-bearing. The
# host default is python 3.14, which numba does not support; left to itself the
# resolver backsolves to numba 0.53.1 / llvmlite 0.36, finds no wheel, tries to build
# from source and dies — so the gate never measures anything. Verified resolution:
# python 3.12.13 + librosa 0.11.0 + numba 0.66.0.
"""Synthesis QC gate, stage 1 of 2 (measures + hard gates).

⚠ This file had NO PEP 723 header until 2026-07-31, so `uv run qc_gate.py` started
with an empty environment and died on `import librosa`. That is most of why the gate
was only ever run by hand: it did not work the way every other script here works.

Walks a campaign directory (engine subdirs with *_manifest.jsonl + wavs) and
for every clip computes:
  - duration-vs-text sanity  (catches Dia improvised tails / truncation / collapse)
  - DNSMOS P.835 (local ONNX; sig/bak/ovr)  (catches noise collapapse, artifacts)
  - phonation composite inputs (alpha/CPP/H1H2 via derive_vat_corpus) + LUFS
Emits <campaign>/qc_measures.jsonl (manifest row + measures + gate flags) and
<campaign>/qc_filelist.txt for scripts/eiv_score.py. Verdict merge (intended-
vs-measured label check, keeps list, owner audit sample) is qc_verdict.py's
job once EIV scores exist — this script only measures and hard-gates.

Deps: numpy, librosa, soundfile, pyloudnorm, onnxruntime, faster-whisper.
DNSMOS model: /data/toolchain/dnsmos/sig_bak_ovr.onnx (Microsoft DNS-Challenge).
"""
import argparse
import glob
import json
import os
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derive_vat_corpus import phonation_measures  # noqa: E402
import silero_vad  # noqa: E402
import synth_common  # noqa: E402
from synth_common import edge_loss  # noqa: E402

DNSMOS_ONNX = "/data/toolchain/dnsmos/sig_bak_ovr.onnx"
DNSMOS_SR = 16000
DNSMOS_LEN = int(9.01 * DNSMOS_SR)
# Microsoft dnsmos_local.py polynomial mappings (P.835)
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

TARGET_SR = 24000
CHARS_PER_SEC_FAST = 26.0   # duration below text_len/26 s => truncated/collapsed (26 not 22: a fast drawl nicked the pilot's arrogance take)
CHARS_PER_SEC_SLOW = 5.0    # duration above text_len/5 + 2 s => improvised tail
# Owner-audit finding 2026-07-17: DNSMOS cannot separate "expressive" from
# "broken" in its 2.0-2.6 band (a great giddy clip scored 2.25; white-noise
# collapse scored 2.05). ASR fidelity is the primary structural gate now:
# transcribe and compare to the script. Catches collapse, wordless output,
# half-empty files, and improvised tails in one instrument.
ASR_MAX_WER = 0.35


def _words(s):
    return [w for w in "".join(c.lower() if c.isalnum() or c.isspace() else " "
                              for c in s).split() if w]


def align_ops(ref, hyp):
    """Levenshtein over words, returning the OPERATION BREAKDOWN, not just the total.

    Aggregate WER hides a distinction worth keeping: a clip that mangles words and a clip
    that ADDS an invented phrase can score the same WER, but only the second one is
    unusable for force-alignment, because its audio contains speech its text does not.
    Insertions isolate that. Deletions isolate the opposite failure — words never spoken —
    and those DO gate today via the aggregate (st_07_singsong_whi_DIA, 0.8 s of speech in
    a 9.8 s file, scored WER 0.67 almost entirely on deletions).

    ⚠ MEASURED FAILURE 2026-07-29 — this does NOT detect VibeVoice ad-libbing, and was
    built to. Calibrated against stress-v1, where the owner's notes name the six clips
    containing invented content: ALL SIX scored ins_rate 0.00, while the only non-zero
    insertions fell on clips with no ad-lib at all. The cause is structural, not a
    threshold problem — VibeVoice's liberties are NON-LEXICAL (a crowd cheering behind the
    line, a piano, reverb, a second actor overlapping), and ASR is trained to suppress
    exactly that and emit a clean transcript. No ASR-derived metric can see content ASR is
    designed to discard. Do not retry this with a better model or a lower threshold.

    Kept because it is free and strictly more informative than scalar WER for the lexical
    cases it does cover. It is instrumentation, NOT a gate for ad-libbing; the control for
    that is routing (`ref_select.NARRATION_ONLY`) plus the owner's ear.
    """
    r, h = _words(ref), _words(hyp)
    if not r:
        return {"sub": 0, "ins": 0, "dele": 0, "n_ref": 0, "wer": 0.0, "ins_rate": 0.0}
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1,
                          d[i-1, j-1] + (r[i-1] != h[j-1]))
    # Backtrace to attribute the distance to substitutions / insertions / deletions.
    i, j, sub, ins, dele = len(r), len(h), 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + (r[i-1] != h[j-1]):
            sub += (r[i-1] != h[j-1])
            i, j = i - 1, j - 1
        elif j > 0 and d[i, j] == d[i, j-1] + 1:
            ins += 1          # hyp has a word the reference does not — an ad-lib
            j -= 1
        else:
            dele += 1         # reference word never spoken
            i -= 1
    return {"sub": sub, "ins": ins, "dele": dele, "n_ref": len(r),
            "wer": float(sub + ins + dele) / len(r), "ins_rate": float(ins) / len(r)}


def wer(ref, hyp):
    return align_ops(ref, hyp)["wer"]


# Owner ear vs instrument, 2026-07-31: the-return_nar_0050_doc_MOS lost its final 19
# words of 139 and the auditor confirmed "cuts off after 'thinks himself a failure'" —
# but it scored WER 0.24 and sailed through ASR_MAX_WER. A global error rate cannot see
# a tail truncation: 19 missing words out of 139 is a small error rate and a totally
# unusable clip. Ask WHERE the render stopped, not how many words differ.
# The 30 s ceiling is named in director_skills/zonos.md as "a hard cap and a training
# limit", and nothing enforced it anywhere. Measured 2026-07-31: one 840-char quoted
# monologue produced a 61.3 s clip on qwen and 64.2 s after re-cast — both a COMPLETE,
# correct read, so every CONTENT gate passed them. At hop 256 that is ~5,750 mel frames,
# and the datamodule has no duration filter, so one such clip pads its whole training
# batch. Length is not a content defect, which is why it needs a gate of its own.
MAX_CLIP_SECONDS = 30.0

TAIL_LOST_MAX = 0.05        # >5% of the passage unspoken at the end
TAIL_WORDS_MIN = 3          # ...and at least 3 real words, so one dropped "the" is not a gate

# Internal dead air. WER cannot see this: a clip can transcribe perfectly and still
# stall mid-sentence, and both clips the ear dropped from newscaster-v1 passed every
# other gate (WER 0.00 and 0.045, DNSMOS 3.3) while the owner's notes on both said
# "long pause". silero_vad.py was written for exactly this failure and was wired
# into nothing until 2026-08-02.
#
# Calibrated against 369 already-audited clips (newscaster-v1, delivery-v1-narration,
# revisit-v1): 303 keeps, and 21 drops whose ear note names a pause. Sweeping the
# worst internal silence against those labels:
#   1.40 s -> catches 20 of 21 (95%), flags 3.3% of keeps
#   2.50 s -> catches 15 of 21 (71%), flags 0.0% of keeps
# The single clip 1.40 s misses has no measurable pause at all, so 95% is the
# practical ceiling for this instrument.
#
# Two levels, because the cost of being wrong is asymmetric. Past PAUSE_HARD_MAX the
# defect is unambiguous (the observed cases run 3-8 s, usually with a rushed
# resumption after) and zero good clips reach it, so it is a hard gate. Between the
# two the clip is suspicious but a long pause can be legitimately dramatic, so it
# only earns an audition — the ear decides. A relative threshold (pause / duration)
# was tested and matched, not beat, the absolute one; absolute is easier to reason
# about at the console.
PAUSE_HARD_MAX = 2.5        # longest internal silence a clip may contain
PAUSE_FLAG_MAX = 1.4        # ...above this it is advisory: queue it for the ear
PAUSE_MIN_GAP = 0.30        # silences shorter than this are ordinary phrasing

# ------------------------------------------------------ UNCALIBRATED (C-M4, 2026-08-07)
#
# `head_ok` and `speech_ok` are MEASURED here and gate nothing. That is the whole point:
# a gate on a guessed threshold either passes truncated clips or rejects good ones, and
# neither failure announces itself. `tail_ok` and the two pause bands were each calibrated
# against a few hundred already-audited clips before they gated anything, and these get
# the same treatment — `gate_calibration.py --sweep` produces the same table.
#
# Until then both live in `advisories`, NOT in `gates`. `hard_pass` is `all(gates.values())`
# and `stage_pool.qc_flagged` reads `gates` too, so putting an uncalibrated measure there
# would silently make it a hard gate at whatever number happened to be typed.
#
# WHAT IS ACTUALLY MISSING IS DIFFERENT FOR THE TWO:
#
#   speech_ok — NEITHER is missing any more; what is left is one policy decision.
#     The number is the owner's: 4 s of speech ([[min-clip-length-4s]], 2026-07-25, the
#     keep-rate cliff is exactly there). The instrument was the open question, because
#     `librivox_align` enforces the same floor with `librosa.effects.split` and says in as
#     many words that it does so to match qc_gate, "because two different measures of one
#     owner rule is one measure too many" — so a unilateral switch here would re-open the
#     divergence that comment closed, by an unmeasured amount.
#
#     MEASURED 2026-08-07, 150 clips sampled across delivery-v1-narration, librivox-v2,
#     newscaster-v1 and revisit-v1: the two instruments agree far more closely than the
#     argument for Silero assumed. Mean VAD/energy ratio 1.008 — the energy gate slightly
#     UNDER-counts on this material rather than over-counting, median delta -0.06 s,
#     p10..p90 -0.34..+0.11 s. At the 4 s floor both reject the same 4.0% of the sample,
#     and exactly ONE clip in 150 (rev_04_tenderness_ORP: energy 4.93 s, VAD 3.84 s)
#     changes side. The floor therefore transfers between instruments essentially intact,
#     and the earlier claim in this comment — that Silero would tighten it by an unknown
#     amount — was written before the sweep and is wrong.
#
#     So the remaining question is not measurement, it is admission policy: should 4 s be
#     a HARD gate that keeps a clip out of a bank, or stay the audition note it is today
#     in `register_audition`? Turning it on rejects ~4% of clips that currently pass QC,
#     which is a corpus decision and the owner's. One line when they answer.
#
#   head_ok — genuinely blocked, and harder than filed. Mirroring TAIL_LOST_MAX is the
#     obvious guess and it is a guess: ASR is worse at the first word than the last, so
#     the same fraction does not mean the same thing at each end. Worse, there is nothing
#     to calibrate AGAINST — searching every drop note in ratings.csv for a phrase naming
#     a late start returns nothing, because no auditor has ever been asked to listen for
#     it. `register_audition` now says so on any clip missing >= 3 opening words; those
#     notes are the evidence a threshold has to come from, and `gate_calibration.py
#     --sweep head_lost_frac` reads them back.
SPEECH_MIN_SECONDS = None   # 4.0 s, pending the owner's hard-gate-or-advisory call
HEAD_LOST_MAX = None        # do NOT copy TAIL_LOST_MAX here without the sweep
HEAD_WORDS_MIN = 3          # ...and one dropped "the" is not a truncation at either end


def speech_seconds_energy(wav, sr):
    """Effective speech by the energy gate — the measure every consumer uses today.

    Kept under its own name because it is about to have a sibling and the two disagree.
    `librosa.effects.split` has no notion of speech: room tone, a page turn, a breath and
    a hum all read as signal, so it OVER-counts, and the amount depends on the recording.
    """
    return float(sum(e - s for s, e in librosa.effects.split(wav, top_db=35))) / sr


class DNSMOS:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def score(self, wav24):
        wav = librosa.resample(wav24, orig_sr=TARGET_SR, target_sr=DNSMOS_SR)
        if len(wav) < DNSMOS_LEN:
            wav = np.tile(wav, int(np.ceil(DNSMOS_LEN / len(wav))))
        hop = DNSMOS_SR  # 1 s
        raws = []
        for start in range(0, max(len(wav) - DNSMOS_LEN, 0) + 1, hop):
            seg = wav[start:start + DNSMOS_LEN]
            if len(seg) < DNSMOS_LEN:
                break
            out = self.sess.run(None, {self.inp: seg[None, :].astype(np.float32)})[0][0]
            raws.append(out)
        sig, bak, ovr = np.mean(raws, axis=0)
        return {"dnsmos_sig": float(P_SIG(sig)), "dnsmos_bak": float(P_BAK(bak)),
                "dnsmos_ovr": float(P_OVR(ovr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--dnsmos-min", type=float, default=2.5,
                    help="advisory quality floor (owner-audit 2026-07-17: the old 3.3 "
                         "mining bar is register-biased against expressive clips; "
                         "structural failures are the ASR gate's job now)")
    args = ap.parse_args()

    import pyloudnorm
    from faster_whisper import WhisperModel
    meter = pyloudnorm.Meter(TARGET_SR)
    dnsmos = DNSMOS(DNSMOS_ONNX)
    asr = WhisperModel("small.en", device="cpu", compute_type="int8")

    # A manifest APPENDS one record per render, so a clip rerolled three times has
    # three records — all pointing at the one wav currently on disk. Scoring each of
    # them measured the same audio repeatedly and inflated every gate count (416
    # records over 301 wavs on delivery-v1-narration, 2026-07-31). Last record wins:
    # it is the take that actually exists.
    jobs = {}
    # Two layouts exist and both are ours. `synth_bank.sh` writes engine subdirs
    # (`<campaign>/audio/qwen_manifest.jsonl`); the hand-authored banks — newscaster-v1
    # is the live example — write manifests and wavs flat at the campaign root. Only
    # the nested pattern was globbed, so pointing the gate at a flat campaign found
    # nothing. That used to mean a silent 0/0 pass; since the empty-glob fix below it
    # means a hard QC-GATE-FAIL on a campaign that is perfectly well formed. Either
    # way the layout, not the audio, decided the outcome. `eng_dir` is just
    # dirname(manifest), and wav paths are resolved against it, so the flat case needs
    # no other special-casing.
    manifests = sorted(
        glob.glob(os.path.join(args.campaign_dir, "*", "*_manifest.jsonl"))
        + glob.glob(os.path.join(args.campaign_dir, "*_manifest.jsonl")))
    for mpath in manifests:
        eng_dir = os.path.dirname(mpath)
        for line in open(mpath, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                jobs[(eng_dir, rec.get("id"))] = rec
    print(f"{len(jobs)} unique clips across "
          f"{len(set(k[0] for k in jobs))} manifest dir(s)", flush=True)

    # An empty glob used to print "0/0 hard-pass", write an empty measures
    # file, print QC-GATE-DONE and exit 0 — so a caller that passed a slightly
    # wrong campaign dir (one trailing slash was enough) got a *passing*
    # mandatory gate over nothing, and the batch went to the ear ungated.
    # A gate with nothing to measure has not passed; it has not run.
    if not manifests:
        sys.exit(f"QC-GATE-FAIL: no *_manifest.jsonl in {args.campaign_dir} "
                 f"or {args.campaign_dir}/*/ — nothing to gate. Check the "
                 f"campaign dir (a trailing slash breaks the glob).")
    if not jobs:
        sys.exit(f"QC-GATE-FAIL: {len(manifests)} manifest(s) found but no "
                 f"records in them — nothing to gate.")

    rows = []
    for (eng_dir, _cid), row in sorted(jobs.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        wav_path = os.path.join(eng_dir, row["wav"])
        quarantined = False
        if not os.path.isfile(wav_path):
            # Quarantined (_dropped/_superseded) or hand-moved clips still have a
            # manifest record. Skipping keeps a whole campaign's gate run from dying
            # on a clip that was deliberately swept aside.
            alt = next((p for p in (os.path.join(eng_dir, d, row["wav"])
                                    for d in ("_dropped", "_superseded"))
                        if os.path.isfile(p)), None)
            if alt:
                # Measured for the record, but it must never re-enter keeps:
                # these were deliberately quarantined, and qc_filelist.txt is
                # a downstream consumer that would have taken them back.
                wav_path = alt
                quarantined = True
            else:
                print(f"{row.get('id','?'):26s} SKIP (wav not found)", flush=True)
                continue
        wav, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        dur = len(wav) / TARGET_SR
        if len(wav) == 0:
            # A zero-length wav used to divide by zero inside DNSMOS and kill
            # the whole campaign run.
            print(f"{row.get('id','?'):26s} FAIL (empty wav)", flush=True)
            row.update({"wav_abs": wav_path, "duration": 0.0, "speech_dur": 0.0,
                        "speech_dur_vad": 0.0,
                        "quarantined": quarantined, "empty_wav": True,
                        "gates": {"nonempty_ok": False}, "advisories": {},
                        "hard_pass": False})
            rows.append(row)
            continue
        n_chars = len(row["text"])

        # effective speech duration (pilot: a 12 s file held 4 s of speech)
        speech_dur = speech_seconds_energy(wav, TARGET_SR)

        # Internal dead air, measured with a trained speech detector rather than the
        # energy gate above — which cannot tell a quiet consonant from a pause.
        # Leading/trailing silence is excluded by long_silences(): that is a trim
        # question, not a performance defect.
        wav16 = librosa.resample(wav, orig_sr=TARGET_SR, target_sr=16000)
        sils = silero_vad.long_silences(wav16, min_s=PAUSE_MIN_GAP)
        worst_pause = max((b - a for a, b in sils), default=0.0)
        # The same detector, asked the OTHER question it was written for. Measured beside
        # the energy figure rather than in place of it: the two disagree, the owner's 4 s
        # floor was set against the energy one, and `librivox_align` enforces that floor
        # with the same instrument on purpose. Recording both is what makes the switch a
        # measurement instead of a guess (C-M4).
        speech_dur_vad = silero_vad.speech_duration(wav16)

        gates = {}
        gates["pause_ok"] = worst_pause <= PAUSE_HARD_MAX
        gates["duration_ok"] = (n_chars / CHARS_PER_SEC_FAST) <= speech_dur <= (n_chars / CHARS_PER_SEC_SLOW + 2.0)
        segs, _ = asr.transcribe(wav_path, language="en")
        hyp = " ".join(s.text for s in segs)
        asr_wer = wer(row["text"], hyp)
        gates["asr_ok"] = asr_wer <= ASR_MAX_WER
        edges = edge_loss(row["text"], hyp)
        tl_frac, tl_words = edges["tail_frac"], edges["tail_words"]
        gates["tail_ok"] = not (tl_frac > TAIL_LOST_MAX and tl_words >= TAIL_WORDS_MIN)
        gates["length_ok"] = dur <= MAX_CLIP_SECONDS

        # Uncalibrated — recorded, never gated. `None` is not "passed": it is "no
        # threshold has been set", and every consumer has to be able to tell those apart.
        advisories = {
            "head_ok": None if HEAD_LOST_MAX is None else not (
                edges["head_frac"] > HEAD_LOST_MAX
                and edges["head_words"] >= HEAD_WORDS_MIN),
            "speech_ok": None if SPEECH_MIN_SECONDS is None else (
                speech_dur_vad >= SPEECH_MIN_SECONDS),
        }
        scores = dnsmos.score(wav)
        # DNSMOS demoted to advisory quality tier (register-biased);
        # collapse detection now belongs to the ASR gate.
        gates["dnsmos_ok"] = scores["dnsmos_ovr"] >= args.dnsmos_min
        try:
            phon = phonation_measures(wav, TARGET_SR)
        except Exception:
            phon, gates["measures_ok"] = None, False
        else:
            gates["measures_ok"] = phon is not None
        lufs = float(meter.integrated_loudness(wav)) if len(wav) > TARGET_SR // 2 else None

        row.update({"wav_abs": wav_path, "duration": dur,
                    "speech_dur": speech_dur,
                    "speech_dur_vad": round(speech_dur_vad, 3),
                    "asr_wer": asr_wer,
                    "worst_pause": round(worst_pause, 3),
                    "pause_count": len(sils),
                    "tail_lost_frac": tl_frac, "tail_words_lost": tl_words,
                    "head_lost_frac": edges["head_frac"],
                    "head_words_lost": edges["head_words"],
                    "asr_hyp": hyp.strip(), **scores,
                    "lufs": lufs, "phonation": phon, "gates": gates,
                    "advisories": advisories,
                    "quarantined": quarantined,
                    "hard_pass": all(gates.values()) and not quarantined})
        rows.append(row)
        print(f"{row['id']:26s} {dur:5.1f}s speech={speech_dur:4.1f}s"
              + f"/vad={speech_dur_vad:4.1f}s "
              + (f"pause={worst_pause:4.1f}s " if worst_pause >= PAUSE_FLAG_MAX else "")
              + (f"head-lost={edges['head_words']}w " if edges["head_words"] else "")
              + f"wer={asr_wer:.2f} ovr={scores['dnsmos_ovr']:.2f} "
              + f"pass={row['hard_pass']}", flush=True)

    out = os.path.join(args.campaign_dir, "qc_measures.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.campaign_dir, "qc_filelist.txt"), "w", encoding="utf-8") as f:
        for r in rows:
            if r["hard_pass"]:
                f.write(r["wav_abs"] + "\n")
    n_pass = sum(r["hard_pass"] for r in rows)
    n_quar = sum(1 for r in rows if r.get("quarantined"))
    print(f"{n_pass}/{len(rows)} hard-pass -> {out}"
          + (f" ({n_quar} quarantined, excluded from keeps)" if n_quar else ""))

    # Say out loud what is measured but NOT gated, so "the gate passed" is never read as
    # "nothing is wrong with the head or the length of speech". C-M4 leaves these open on
    # purpose; silence about them is how an uncalibrated gate becomes an assumed one.
    n_head = sum(1 for r in rows if (r.get("head_words_lost") or 0) >= HEAD_WORDS_MIN)
    n_short = sum(1 for r in rows if r.get("speech_dur_vad") is not None
                  and r["speech_dur_vad"] < 4.0)
    uncal = [k for k, v in (rows[0].get("advisories") or {}).items() if v is None] \
        if rows else []
    if uncal:
        print(f"  NOT GATED (no calibrated threshold yet): {', '.join(sorted(uncal))}"
              f" — {n_head} clip(s) drop >= {HEAD_WORDS_MIN} words off the HEAD, "
              f"{n_short} hold < 4.0 s of speech by the VAD measure.")
        print("  Set them with:  gate_calibration.py --sweep head_lost_frac "
              "--campaign-dir <dir>   (and --sweep speech_dur_vad)")
    if not rows:
        sys.exit("QC-GATE-FAIL: every manifest record was skipped (no wav "
                 "found for any of them) — nothing was measured.")
    if n_pass == 0:
        # Not a verdict on the clips: a batch where nothing passes is far more
        # often a broken run (wrong refs, a dead engine, a bad threshold) than
        # a genuinely worthless bank, and it must not register silently.
        sys.exit(f"QC-GATE-FAIL: 0 of {len(rows)} clips passed. Inspect "
                 f"{out} before registering anything for audition.")
    print("QC-GATE-DONE")


if __name__ == "__main__":
    main()
