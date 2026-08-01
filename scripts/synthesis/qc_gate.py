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
from derive_vat_corpus import phonation_measures  # noqa: E402

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


def tail_lost(ref, hyp):
    """Fraction of the passage left unspoken after the last word ASR could align."""
    import difflib
    import re as _re
    norm = lambda s: _re.sub(r"[^a-z0-9 ]", "", s.lower().replace("’", "'")).split()
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0, 0
    blocks = [b for b in difflib.SequenceMatcher(a=r, b=h, autojunk=False)
              .get_matching_blocks() if b.size]
    last = (blocks[-1].a + blocks[-1].size - 1) if blocks else -1
    lost = len(r) - 1 - last
    return lost / len(r), lost


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
    for mpath in sorted(glob.glob(os.path.join(args.campaign_dir, "*", "*_manifest.jsonl"))):
        eng_dir = os.path.dirname(mpath)
        for line in open(mpath, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                jobs[(eng_dir, rec.get("id"))] = rec
    print(f"{len(jobs)} unique clips across "
          f"{len(set(k[0] for k in jobs))} manifest dir(s)", flush=True)

    rows = []
    for (eng_dir, _cid), row in sorted(jobs.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        wav_path = os.path.join(eng_dir, row["wav"])
        if not os.path.isfile(wav_path):
            # Quarantined (_dropped/_superseded) or hand-moved clips still have a
            # manifest record. Skipping keeps a whole campaign's gate run from dying
            # on a clip that was deliberately swept aside.
            alt = next((p for p in (os.path.join(eng_dir, d, row["wav"])
                                    for d in ("_dropped", "_superseded"))
                        if os.path.isfile(p)), None)
            if alt:
                wav_path = alt
            else:
                print(f"{row.get('id','?'):26s} SKIP (wav not found)", flush=True)
                continue
        wav, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        dur = len(wav) / TARGET_SR
        n_chars = len(row["text"])

        # effective speech duration (pilot: a 12 s file held 4 s of speech)
        intervals = librosa.effects.split(wav, top_db=35)
        speech_dur = float(sum(e - s for s, e in intervals)) / TARGET_SR

        gates = {}
        gates["duration_ok"] = (n_chars / CHARS_PER_SEC_FAST) <= speech_dur <= (n_chars / CHARS_PER_SEC_SLOW + 2.0)
        segs, _ = asr.transcribe(wav_path, language="en")
        hyp = " ".join(s.text for s in segs)
        asr_wer = wer(row["text"], hyp)
        gates["asr_ok"] = asr_wer <= ASR_MAX_WER
        tl_frac, tl_words = tail_lost(row["text"], hyp)
        gates["tail_ok"] = not (tl_frac > TAIL_LOST_MAX and tl_words >= TAIL_WORDS_MIN)
        gates["length_ok"] = dur <= MAX_CLIP_SECONDS
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
                    "speech_dur": speech_dur, "asr_wer": asr_wer,
                    "tail_lost_frac": tl_frac, "tail_words_lost": tl_words,
                    "asr_hyp": hyp.strip(), **scores,
                    "lufs": lufs, "phonation": phon, "gates": gates,
                    "hard_pass": all(gates.values())})
        rows.append(row)
        print(f"{row['id']:26s} {dur:5.1f}s speech={speech_dur:4.1f}s "
              f"wer={asr_wer:.2f} ovr={scores['dnsmos_ovr']:.2f} "
              f"pass={row['hard_pass']}", flush=True)

    out = os.path.join(args.campaign_dir, "qc_measures.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.campaign_dir, "qc_filelist.txt"), "w", encoding="utf-8") as f:
        for r in rows:
            if r["hard_pass"]:
                f.write(r["wav_abs"] + "\n")
    n_pass = sum(r["hard_pass"] for r in rows)
    print(f"{n_pass}/{len(rows)} hard-pass -> {out}")
    print("QC-GATE-DONE")


if __name__ == "__main__":
    main()
