"""Render a VAT-conditioned sweep for the eval harness (north star §7).

The generation half of the derisk gate: load an acoustic checkpoint (CPU),
render each selected validation clip at several values of ONE VAT channel
(all else neutral), vocode with the 24 kHz HiFi-GAN fine-tune, and write the
WAVs plus the JSONL manifest scripts/eval_harness.py consumes. Nothing here
computes a verdict — that's the harness's job.

Clips come from the pre-phonemized VAT val filelist (path|spk|phonemes|v,a,t),
so no G2P runs here; the WER reference text is the LibriTTS .normalized.txt
sibling of the source wav. Also writes speaker_refs.txt (two real wavs from
two different selected speakers) for the harness's leakage normalization, and
render_meta.json (checkpoint/vocoder/settings) for the gate postprocessor.

Fully CPU — safe to run while a GPU training run owns the card.

Usage:
    python scripts/render_vat_sweep.py \
        --checkpoint <matcha .ckpt> --out <dir> \
        [--channel energy] [--values -1,-0.5,0,0.5,1] [--n-clips 4]
"""

import argparse
import json
import os
import sys

HIFIGAN = os.environ.get("SONORA_HIFIGAN_DIR", "/data/model-training/vocoder/hifi-gan")
sys.path.insert(0, HIFIGAN)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import soundfile as sf
import torch

# energy rides the arousal slot in the v0 derisk corpus (V=T=0 labels;
# configs/experiment/derisk_energy.yaml).
CHANNEL_SLOT = {"valence": 0, "energy": 1, "tension": 2}


def load_acoustic(path):
    from matcha.models.matcha_tts import MatchaTTS

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    model = MatchaTTS(**hp)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def load_vocoder(ckpt_path, config_path):
    from env import AttrDict
    from models import Generator

    h = AttrDict(json.load(open(config_path)))
    g = Generator(h)
    g.load_state_dict(torch.load(ckpt_path, map_location="cpu")["generator"])
    g.eval()
    g.remove_weight_norm()
    return g, h


def pick_clips(filelist, n_clips, min_chars=60, max_chars=200):
    """First n qualifying rows with pairwise-distinct speakers, deterministic."""
    rows, seen_spk = [], set()
    with open(filelist, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 4:
                continue
            wav_path, spk, phonemes, _vat = parts
            if not (min_chars <= len(phonemes) <= max_chars):
                continue
            if spk in seen_spk:
                continue
            txt_path = wav_path.rsplit(".", 1)[0] + ".normalized.txt"
            if not os.path.exists(txt_path):
                continue
            seen_spk.add(spk)
            rows.append({"wav": wav_path, "spk": int(spk), "phonemes": phonemes,
                         "text": open(txt_path, encoding="utf-8").read().strip()})
            if len(rows) == n_clips:
                break
    if len(rows) < n_clips:
        raise SystemExit(f"only {len(rows)} qualifying clips in {filelist}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocoder",
                    default="/data/model-training/vocoder/cp_hifigan_24k/g_02510000")
    ap.add_argument("--vocoder-config",
                    default=os.path.join(HIFIGAN, "config_24k_80band.json"))
    ap.add_argument("--val-filelist", default="data/libritts_r_vat/val_op.txt")
    ap.add_argument("--channel", default="energy", choices=sorted(CHANNEL_SLOT))
    ap.add_argument("--values", default="-1,-0.5,0,0.5,1",
                    help="comma-separated sweep; must include 0 (the baseline row)")
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--n-timesteps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.667)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234,
                    help="re-seeded before every render so sweep rows differ only in vat")
    args = ap.parse_args()

    values = [float(v) for v in args.values.split(",")]
    if 0.0 not in values:
        raise SystemExit("--values must include 0 (baseline row)")

    from matcha.text import text_to_sequence
    from matcha.utils.utils import intersperse

    model = load_acoustic(args.checkpoint)
    vocoder, h = load_vocoder(args.vocoder, args.vocoder_config)
    clips = pick_clips(args.val_filelist, args.n_clips)
    os.makedirs(args.out, exist_ok=True)

    slot = CHANNEL_SLOT[args.channel]
    manifest = []
    for clip in clips:
        seq, _ = text_to_sequence(clip["phonemes"], ["no_cleaners"])
        x = torch.tensor(intersperse(seq, 0), dtype=torch.long)[None]
        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long)
        spks = torch.tensor([clip["spk"]], dtype=torch.long)
        group = os.path.basename(clip["wav"]).rsplit(".", 1)[0]
        for v in values:
            vat = torch.zeros(1, 3)
            vat[0, slot] = v
            torch.manual_seed(args.seed)
            with torch.no_grad():
                out = model.synthesise(x, x_lengths, n_timesteps=args.n_timesteps,
                                       temperature=args.temperature, spks=spks,
                                       length_scale=args.length_scale, vat=vat)
                wav = vocoder(out["mel"]).squeeze().numpy()
            fname = f"{group}_{args.channel}{v:+.2f}.wav"
            sf.write(os.path.join(args.out, fname), wav, h.sampling_rate)
            manifest.append({"wav": fname, "text": clip["text"], "group": group,
                             "requested": {args.channel: v}, "baseline": v == 0.0})
            print(f"rendered {fname} ({len(wav) / h.sampling_rate:.1f}s)")

    with open(os.path.join(args.out, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    # Two real recordings from two different swept speakers -> the harness's
    # inter-speaker gap (leakage denominator).
    with open(os.path.join(args.out, "speaker_refs.txt"), "w", encoding="utf-8") as f:
        f.write(clips[0]["wav"] + "\n" + clips[1]["wav"] + "\n")
    with open(os.path.join(args.out, "render_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "vocoder": args.vocoder,
                   "channel": args.channel, "values": values,
                   "clips": [c["wav"] for c in clips],
                   "n_timesteps": args.n_timesteps, "temperature": args.temperature,
                   "length_scale": args.length_scale, "seed": args.seed}, f, indent=2)
    print(f"\n{len(manifest)} renders + manifest -> {args.out}")


if __name__ == "__main__":
    main()
