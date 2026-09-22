#!/usr/bin/env python3
"""Render the acoustic model's own mels, aligned to real audio, for the vocoder fine-tune.

The vocoder learned on mels computed from real recordings. The acoustic model does not make
those — it makes mels with its own texture — and the ceiling bench (2026-09-22) left open
how much of the hum is the vocoder's response to that difference. hifi-gan's fine-tuning
mode answers it: give the vocoder the MODEL's mel as input and the REAL audio as target.

⚠⚠ THE MEL MUST LINE UP WITH THE AUDIO, SAMPLE FOR SAMPLE. hifi-gan crops a random run of
mel frames and the audio at `frame * hop`, so a mel even one frame longer or shorter than
the recording trains the vocoder against the wrong 256 samples, everywhere, with no error.
`aligned_synth.aligned_generate` keeps the recording's own length through MAS, and this
tool REFUSES any clip where frames != samples // hop instead of trusting that.

⚠ THE SETTINGS ARE THE EAR BENCH'S, NOT A GUESS. 10 ODE steps, temperature 0.667, guidance
1.0 — the values `ear_bench.py` renders with, so the vocoder learns the mels the ceiling
bench heard. Change them here and the fine-tune answers a different question.

⚠⚠ THE SAVED MEL IS DENORMALISED WITH THE CONFIG'S STATISTICS, NOT THE MODEL'S BUFFERS.
The model works in corpus-normalised units; hifi-gan's input is a log-mel of the raw
signal. This tool loads with a bare `load_state_dict`, which skips the Lightning hook that
corrects stale buffers, and the first smoke run here is how the stale buffers were found:
every mel came out 1.09 log units below the real mel of its own clip (matcha/mel_stats.py).
The config is what the dataset below normalises with, so it is the only consistent choice.

Resumable: a clip whose .npy exists is skipped, so a killed run restarts where it stopped.

Usage (container paths — see run_in_rocm.sh):
    scripts/stages/run_in_rocm.sh scripts/stages/render_aligned_mels.py \\
        --filelist /data/model-training/vocoder/vocoder_ft/vat7_ep005/ft_train_op.txt \\
        --model-config <run>/.hydra/config.yaml --ckpt <run>/checkpoints/<ckpt> \\
        --out /data/model-training/vocoder/vocoder_ft/vat7_ep005/mels
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

_REPO = os.environ.get("SONORA_REPO", "/sonora")
for _p in (_REPO, os.path.join(_REPO, "scripts", "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hydra.utils import instantiate  # noqa: E402

from aligned_synth import aligned_generate, build_dataset  # noqa: E402
from matcha.data.license_wall import enforce  # noqa: E402
from matcha.data.text_mel_datamodule import TextMelBatchCollate  # noqa: E402
from matcha.utils.model import denormalize  # noqa: E402

# The ear bench's render settings. Imported values would be better, but ear_bench pulls in
# the G2P front end and its LiteRT dependency, which this container does not install.
N_TIMESTEPS = 10
TEMPERATURE = 0.667
GUIDANCE = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    enforce([args.filelist])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.model_config)
    hop = int(cfg.data.hop_length)
    collate = TextMelBatchCollate(cfg.data.n_spks)
    ds = build_dataset(cfg, args.filelist)

    model = instantiate(cfg.model)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit("REFUSING: checkpoint does not match this config "
                         "(missing=%d unexpected=%d)." % (len(missing), len(unexpected)))
    model = model.to(device).eval()
    mel_mean = float(cfg.data.data_statistics.mel_mean)
    mel_std = float(cfg.data.data_statistics.mel_std)
    print("  denormalising with the config's mean %.4f std %.4f (buffers: %.4f / %.4f)"
          % (mel_mean, mel_std, float(model.mel_mean), float(model.mel_std)), flush=True)

    os.makedirs(args.out, exist_ok=True)
    meta = {"filelist": args.filelist, "ckpt": args.ckpt, "model_config": args.model_config,
            "n_timesteps": N_TIMESTEPS, "temperature": TEMPERATURE, "guidance": GUIDANCE,
            "hop": hop, "alignment": "MAS against the true mel",
            "mel_mean": mel_mean, "mel_std": mel_std}
    meta_path = os.path.join(args.out, "render_meta.json")
    if os.path.exists(meta_path):
        prev = json.load(open(meta_path, encoding="utf-8"))
        if {k: prev.get(k) for k in meta} != meta:
            raise SystemExit("REFUSING: %s was rendered with different settings:\n  %s\n"
                             "A resumed directory would mix two sets of mels."
                             % (args.out, prev))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    def stem(i):
        return os.path.splitext(os.path.basename(ds.filepaths_and_text[i][0]))[0]

    todo = [i for i in range(len(ds))
            if not os.path.exists(os.path.join(args.out, stem(i) + ".npy"))]
    todo = todo[: args.limit or None]
    # Sorted by length, so a batch pads a little rather than a lot. The sort reads each
    # clip's header once; it is the cheap part of this job.
    samples = {i: sf.info(ds.filepaths_and_text[i][0]).frames for i in todo}
    todo.sort(key=lambda i: samples[i])
    print("  %d of %d clips to render, batch %d, on %s"
          % (len(todo), len(ds), args.batch, device.type), flush=True)

    t0, done = time.time(), 0
    for b in range(0, len(todo), args.batch):
        idx = todo[b:b + args.batch]
        batch = collate([ds[i] for i in idx])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        torch.manual_seed(zlib.crc32(stem(idx[0]).encode()))
        gen, _y, _m = aligned_generate(model, batch, N_TIMESTEPS, TEMPERATURE,
                                       guidance=GUIDANCE)
        mel = denormalize(gen, mel_mean, mel_std).float().cpu().numpy()
        for j, i in enumerate(idx):
            n = int(batch["y_lengths"][j])
            if n != samples[i] // hop:
                raise SystemExit("REFUSING: %s has %d samples, so %d frames at hop %d, and "
                                 "the mel has %d. hifi-gan would pair every frame with the "
                                 "wrong audio." % (ds.filepaths_and_text[i][0], samples[i],
                                                   samples[i] // hop, hop, n))
            path = os.path.join(args.out, stem(i) + ".npy")
            np.save(path + ".tmp.npy", mel[j, :, :n])
            os.replace(path + ".tmp.npy", path)
        done += len(idx)
        if done % (args.batch * 25) < args.batch or done == len(todo):
            rate = done / (time.time() - t0)
            print("    %d/%d  %.1f clips/s  eta %.0f min"
                  % (done, len(todo), rate, (len(todo) - done) / rate / 60), flush=True)


if __name__ == "__main__":
    main()
