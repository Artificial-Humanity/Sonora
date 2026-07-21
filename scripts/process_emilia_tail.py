"""Emilia tail integration, stage 1: transcode + ASR cross-check.

First of the pre-registered tail steps (emilia-mining-plan.md; STATE.md
"ASR/24k/labels/filelists"): turn the mined keeps (mp3 + YODAS caption text)
into corpus-ready audio with a per-clip transcript-quality signal.

Per keep (manifest.json in --keeps):
    * decode mp3 -> 24 kHz mono wav in --out/wavs/ (the corpus rate)
    * faster-whisper base.en (CPU int8) transcript of the same audio
    * WER between the YODAS caption and the ASR transcript -> asr.jsonl

No clip is dropped here: the WER is recorded, and the drop threshold is a
merge-time decision (stage 2, labels/filelists) so the filter can be tuned
without re-running ASR. YODAS captions are the canonical text per the
force-align-first rule; ASR is the cross-check, not the replacement — a high
WER means "caption unreliable, exclude or re-transcribe", never "silently
swap in the ASR text".

Fully CPU (multiprocessing, one whisper per worker) — safe alongside GPU
work. Resumable: clips already in asr.jsonl are skipped.

Usage:
    uv run --with librosa --with soundfile --with faster-whisper \
        python scripts/process_emilia_tail.py \
        --keeps /data/model-training/datasets/emilia_kept \
        --out /data/model-training/datasets/emilia_kept_24k \
        [--workers 12]
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys

SR_OUT = 24000
SR_ASR = 16000

_model = None


def _init_worker():
    global _model
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from faster_whisper import WhisperModel

    _model = WhisperModel("base.en", device="cpu", compute_type="int8",
                          cpu_threads=1)


def _norm_words(text):
    return re.sub(r"[^a-z' ]", "", text.lower().replace("-", " ")).split()


def _wer(ref, hyp):
    import numpy as np

    r, h = _norm_words(ref), _norm_words(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[-1, -1]) / max(len(r), 1)


def _process(job):
    src, wav_out, text = job
    import librosa
    import soundfile as sf

    try:
        y, _ = librosa.load(src, sr=SR_OUT, mono=True)
        sf.write(wav_out, y, SR_OUT)
        y16 = librosa.resample(y, orig_sr=SR_OUT, target_sr=SR_ASR)
        segs, info = _model.transcribe(y16, beam_size=5)
        hyp = " ".join(s.text.strip() for s in segs).strip()
        return {"file": os.path.basename(src), "wav": os.path.basename(wav_out),
                "seconds": round(len(y) / SR_OUT, 2),
                "text": text, "asr": hyp, "wer": round(_wer(text, hyp), 4)}
    except Exception as e:  # keep going; a broken mp3 is a data point too
        return {"file": os.path.basename(src), "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keeps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.keeps, "manifest.json"),
                              encoding="utf-8"))
    keeps = manifest["keeps"]
    wav_dir = os.path.join(args.out, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    asr_path = os.path.join(args.out, "asr.jsonl")

    done = set()
    if os.path.exists(asr_path):
        with open(asr_path, encoding="utf-8") as f:
            done = {json.loads(l)["file"] for l in f if l.strip()}
        print(f"resuming: {len(done)} already processed")

    jobs = []
    for k in keeps:
        if k["file"] in done:
            continue
        jobs.append((os.path.join(args.keeps, k["file"]),
                     os.path.join(wav_dir, k["file"][:-4] + ".wav"),
                     k["text"]))
    print(f"{len(jobs)} clips to process ({len(keeps)} keeps total)")

    n_ok = n_err = 0
    with open(asr_path, "a", encoding="utf-8") as fout, \
            mp.get_context("spawn").Pool(args.workers,
                                         initializer=_init_worker) as pool:
        for row in pool.imap_unordered(_process, jobs, chunksize=8):
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_err += 1 if "error" in row else 0
            n_ok += 0 if "error" in row else 1
            total = n_ok + n_err
            if total % 250 == 0 or total == len(jobs):
                print(f"  {total}/{len(jobs)} (errors: {n_err})", flush=True)
    print(f"done -> {asr_path} ({n_ok} ok, {n_err} errors)")
    sys.exit(1 if n_err and not n_ok else 0)


if __name__ == "__main__":
    main()
