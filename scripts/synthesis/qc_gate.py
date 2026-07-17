"""Synthesis QC gate, stage 1 of 2 (measures + hard gates).

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


def wer(ref, hyp):
    r = [w for w in "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in ref).split() if w]
    h = [w for w in "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in hyp).split() if w]
    if not r:
        return 0.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1,
                          d[i-1, j-1] + (r[i-1] != h[j-1]))
    return float(d[len(r), len(h)]) / len(r)


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

    rows = []
    for mpath in sorted(glob.glob(os.path.join(args.campaign_dir, "*", "*_manifest.jsonl"))):
        eng_dir = os.path.dirname(mpath)
        for line in open(mpath, encoding="utf-8"):
            row = json.loads(line)
            wav_path = os.path.join(eng_dir, row["wav"])
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
