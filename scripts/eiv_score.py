"""Score audio with LAION Empathic-Insight-Voice heads (CC-BY-4.0).

The corpus-brief (option B) labeler: EmoWhisper encoder states (16 kHz, 30 s
cap, last_hidden_state padded/truncated to 1500x768, flattened) -> one tiny
MLP per attribute head. Head architecture is inferred from each checkpoint's
tensor shapes, so Small/Large suites both load. Recipe follows the published
EIV inference example verbatim.

Used for: the valence labeling pass (--heads Valence over the corpus ->
derive_vat_corpus.py --valence-json), and the tension recalibration
(Soft_vs._Harsh / Distress vs. the phonation composite;
tension-definition-brief.md calibration status).

Inputs: wav paths, dirs, or `path|...` filelists. Output: JSONL rows
{"wav": ..., "<head>": score, ...}.

Usage:
    python scripts/eiv_score.py --out scores.jsonl \
        --inputs data/libritts_r_vat_v1/train_op.txt data/libritts_r_vat_v1/val_op.txt \
        [--heads "Valence,Arousal,Distress,Soft_vs._Harsh"] [--batch-size 8]
        [--limit N] [--sample N]
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

ENCODER_DEFAULT = "/data/reference/models/mkrausio/EmoWhisper-AnS-Small-v0.1"
HEADS_DIR_DEFAULT = "/data/reference/models/laion/Empathic-Insight-Voice-Large"
SR = 16000
MAX_SECONDS = 30.0
SEQ_LEN = 1500
EMBED_DIM = 768


class Head(nn.Module):
    """FullEmbeddingMLP with dims inferred from the checkpoint."""

    def __init__(self, state_dict):
        super().__init__()
        proj_out, proj_in = state_dict["proj.weight"].shape
        self.proj = nn.Linear(proj_in, proj_out)
        idxs = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith("mlp.")})
        layers, prev = [], proj_out
        seq = {}
        for i in idxs:
            out_dim, in_dim = state_dict[f"mlp.{i}.weight"].shape
            seq[i] = nn.Linear(in_dim, out_dim)
            prev = out_dim
        # Rebuild the Sequential with ReLU/Dropout placeholders at the gaps so
        # module indices match the checkpoint keys.
        max_i = max(idxs)
        mods = []
        for i in range(max_i + 1):
            mods.append(seq.get(i, nn.ReLU() if i % 3 != 1 else nn.Dropout(0.0)))
        self.mlp = nn.Sequential(*mods)
        sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.load_state_dict(sd)

    def forward(self, emb):  # emb: (B, 1500, 768)
        return self.mlp(self.proj(emb.flatten(1)))


def collect_wavs(inputs):
    wavs = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _, names in os.walk(item):
                wavs += [os.path.join(root, n) for n in sorted(names) if n.endswith(".wav")]
        elif item.endswith(".txt"):
            with open(item, encoding="utf-8") as f:
                wavs += [line.split("|")[0] for line in f if line.strip()]
        else:
            wavs.append(item)
    seen, out = set(), []
    for w in wavs:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heads", default="Valence,Arousal,Distress,Soft_vs._Harsh")
    ap.add_argument("--heads-dir", default=HEADS_DIR_DEFAULT)
    ap.add_argument("--encoder", default=ENCODER_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N wavs only")
    ap.add_argument("--sample", type=int, default=0, help="random N wavs (seed 1234)")
    args = ap.parse_args()

    import librosa
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    wavs = collect_wavs(args.inputs)
    if args.sample:
        random.seed(1234)
        wavs = random.sample(wavs, min(args.sample, len(wavs)))
    if args.limit:
        wavs = wavs[: args.limit]
    print(f"{len(wavs)} wavs to score")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    processor = WhisperProcessor.from_pretrained(args.encoder)
    encoder = WhisperForConditionalGeneration.from_pretrained(args.encoder) \
        .get_encoder().to(device).eval()

    heads = {}
    for name in args.heads.split(","):
        path = os.path.join(args.heads_dir, f"model_{name}_best.pth")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        heads[name] = Head(sd).to(device).eval()
        print(f"head {name}: proj {tuple(sd['proj.weight'].shape)}")

    done = set()
    if os.path.exists(args.out):  # resumable
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["wav"] for line in f if line.strip()}
        print(f"resuming: {len(done)} already scored")

    max_samples = int(MAX_SECONDS * SR)
    todo = [w for w in wavs if w not in done]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fout:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            audio = []
            for w in batch:
                y, _ = librosa.load(w, sr=SR, mono=True)
                audio.append(y[:max_samples])
            feats = processor(audio, sampling_rate=SR, return_tensors="pt").input_features
            with torch.no_grad():
                emb = encoder(input_features=feats.to(device)).last_hidden_state
                if emb.shape[1] < SEQ_LEN:
                    emb = torch.nn.functional.pad(emb, (0, 0, 0, SEQ_LEN - emb.shape[1]))
                elif emb.shape[1] > SEQ_LEN:
                    emb = emb[:, :SEQ_LEN]
                scores = {name: h(emb).squeeze(-1).float().cpu().tolist()
                          for name, h in heads.items()}
            for i, w in enumerate(batch):
                row = {"wav": w}
                row.update({name: round(scores[name][i], 6) for name in heads})
                fout.write(json.dumps(row) + "\n")
            fout.flush()
            n = start + len(batch)
            if n % (args.batch_size * 25) < args.batch_size or n == len(todo):
                print(f"  scored {n}/{len(todo)}")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
