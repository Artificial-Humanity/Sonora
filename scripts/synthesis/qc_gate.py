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

Deps: numpy, librosa, soundfile, pyloudnorm, onnxruntime.
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
CHARS_PER_SEC_FAST = 22.0   # duration below text_len/22 s => truncated/collapsed
CHARS_PER_SEC_SLOW = 5.0    # duration above text_len/5 + 2 s => improvised tail


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
    ap.add_argument("--dnsmos-min", type=float, default=3.3,
                    help="OVR floor; same bar the Emilia mining used")
    args = ap.parse_args()

    import pyloudnorm
    meter = pyloudnorm.Meter(TARGET_SR)
    dnsmos = DNSMOS(DNSMOS_ONNX)

    rows = []
    for mpath in sorted(glob.glob(os.path.join(args.campaign_dir, "*", "*_manifest.jsonl"))):
        eng_dir = os.path.dirname(mpath)
        for line in open(mpath, encoding="utf-8"):
            row = json.loads(line)
            wav_path = os.path.join(eng_dir, row["wav"])
            wav, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
            dur = len(wav) / TARGET_SR
            n_chars = len(row["text"])

            gates = {}
            gates["duration_ok"] = (n_chars / CHARS_PER_SEC_FAST) <= dur <= (n_chars / CHARS_PER_SEC_SLOW + 2.0)
            scores = dnsmos.score(wav)
            gates["dnsmos_ok"] = scores["dnsmos_ovr"] >= args.dnsmos_min
            try:
                phon = phonation_measures(wav, TARGET_SR)
            except Exception:
                phon, gates["measures_ok"] = None, False
            else:
                gates["measures_ok"] = phon is not None
            lufs = float(meter.integrated_loudness(wav)) if len(wav) > TARGET_SR // 2 else None

            row.update({"wav_abs": wav_path, "duration": dur, **scores,
                        "lufs": lufs, "phonation": phon, "gates": gates,
                        "hard_pass": all(gates.values())})
            rows.append(row)
            print(f"{row['id']:26s} {dur:5.1f}s ovr={scores['dnsmos_ovr']:.2f} "
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
