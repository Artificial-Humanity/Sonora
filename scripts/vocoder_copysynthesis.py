"""Copy-synthesis gate for the 24 kHz vocoder fine-tune.

Ground-truth mel -> vocoder -> waveform, compared against the original
recording (sample-rate-24khz-decision.md consequence #2): a bad vocoder must
be caught here, standalone, before the de-risk conditioning run depends on
it. Renders N validation clips through a given checkpoint and reports
per-clip mel-reconstruction error (mel of the vocoded output vs the input
mel — the vocoder's own training metric) plus faster-whisper WER vs the
LibriTTS transcript, and writes A/B wav pairs for listening.

Usage:
    python scripts/vocoder_copysynthesis.py \
        --checkpoint /data/model-training/vocoder/cp_hifigan_24k/g_02505000 \
        [--n 8] [--out /data/model-training/vocoder/copysynth]
"""

import argparse
import json
import os
import sys

HIFIGAN = os.environ.get("SONORA_HIFIGAN_DIR", "/data/model-training/vocoder/hifi-gan")
WAVS_DIR = "/data/model-training/datasets/LibriTTS_R/train-clean-100"
VAL_LIST = "/data/model-training/vocoder/filelist_val.txt"

sys.path.insert(0, HIFIGAN)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import soundfile as sf
import torch

from env import AttrDict
from meldataset import mel_spectrogram
from models import Generator


def wer(ref, hyp):
    import re

    def norm(t):
        return re.sub(r"[^a-z' ]", "", t.lower().replace("-", " ")).split()

    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[-1, -1]) / max(len(r), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=os.path.join(HIFIGAN, "config_24k_80band.json"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="/data/model-training/vocoder/copysynth")
    ap.add_argument("--skip-asr", action="store_true")
    args = ap.parse_args()

    h = AttrDict(json.load(open(args.config)))
    g = Generator(h)
    g.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["generator"])
    g.eval()
    g.remove_weight_norm()

    whisper = None
    if not args.skip_asr:
        from faster_whisper import WhisperModel

        whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

    with open(VAL_LIST, encoding="utf-8") as f:
        rels = [line.strip() for line in f if line.strip()][: args.n]
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for rel in rels:
        wav_path = os.path.join(WAVS_DIR, rel + ".wav")
        txt_path = os.path.join(WAVS_DIR, rel + ".normalized.txt")
        wav, sr = sf.read(wav_path, dtype="float32")
        assert sr == h.sampling_rate, (sr, h.sampling_rate)
        x = torch.from_numpy(wav).unsqueeze(0)
        mel = mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                              h.win_size, h.fmin, h.fmax, center=False)
        with torch.no_grad():
            y = g(mel).squeeze().numpy()
        mel_y = mel_spectrogram(torch.from_numpy(y).unsqueeze(0), h.n_fft, h.num_mels,
                                h.sampling_rate, h.hop_size, h.win_size, h.fmin,
                                h.fmax, center=False)
        n = min(mel.shape[-1], mel_y.shape[-1])
        mel_err = float(torch.mean(torch.abs(mel[..., :n] - mel_y[..., :n])))
        base = rel.replace("/", "_")
        sf.write(os.path.join(args.out, f"{base}_ref.wav"), wav, sr)
        sf.write(os.path.join(args.out, f"{base}_voc.wav"), y, sr)
        row = {"clip": rel, "mel_l1": mel_err}
        if whisper:
            ref_text = open(txt_path, encoding="utf-8").read().strip()
            segs, _ = whisper.transcribe(y, beam_size=5)
            hyp = " ".join(s.text.strip() for s in segs).strip()
            row["wer"] = wer(ref_text, hyp)
            row["transcript"] = hyp
        rows.append(row)
        print(row)

    mean_mel = float(np.mean([r["mel_l1"] for r in rows]))
    report = {"checkpoint": args.checkpoint, "mean_mel_l1": mean_mel, "rows": rows}
    if whisper:
        report["mean_wer"] = float(np.mean([r["wer"] for r in rows]))
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nmean mel L1: {mean_mel:.4f}"
          + (f" | mean WER: {report['mean_wer']:.3f}" if whisper else ""))
    print("A/B pairs + report ->", args.out)


if __name__ == "__main__":
    main()
