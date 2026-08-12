"""Score audio with LAION Empathic-Insight-Voice heads (CC-BY-4.0).

The corpus-brief (option B) labeler: EmoWhisper encoder states (16 kHz, 30 s
cap, last_hidden_state padded/truncated to 1500x768, flattened) -> one tiny
MLP per attribute head. Head architecture is inferred from each checkpoint's
tensor shapes, so Small/Large suites both load. Recipe follows the published
EIV inference example verbatim.

Used for: the valence labeling pass (--heads Valence over the corpus ->
derive_vat_corpus.py --valence-json), and the tension recalibration
(Soft_vs._Harsh / Distress vs. the phonation composite;
vat-channels.md calibration status).

Inputs: wav paths, dirs, or `path|...` filelists. Output: JSONL rows
{"wav": ..., "wav_mtime": ..., "<head>": score, ...} — `wav_mtime` is the mtime
of the file that was actually scored, so a consumer can tell a score of THIS
take from a score of the one before it (qc_verdict.py's reroll guard).

Usage:
    python scripts/stages/eiv_score.py --out scores.jsonl \
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

ENCODER_DEFAULT = "/data/models/mkrausio/EmoWhisper-AnS-Small-v0.1"
HEADS_DIR_DEFAULT = "/data/models/laion/Empathic-Insight-Voice-Large"
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
                wavs += [os.path.join(root, n) for n in sorted(names)
                         if n.endswith((".wav", ".mp3"))]
        elif item.endswith(".txt"):
            with open(item, encoding="utf-8") as f:
                wavs += [line.strip().split("|")[0] for line in f if line.strip()]
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

    # Resume is BY PATH, which is why every row carries the mtime of the wav it scored.
    # A clip re-rendered in place keeps its path, so the resume skips it and its old
    # scores stand for a take that no longer exists — the loudnorm reroll defect's shape
    # (a sidecar keyed on a path that was rewritten underneath it). qc_verdict.py used to
    # notice that by comparing wavs against the mtime of this whole FILE, which every
    # append refreshed, so one newly-scored clip silenced the check for all of them
    # (issue #56). A per-row stamp is a fact about the clip; a file mtime is a fact about
    # the batch.
    #
    # A stamped row whose wav has moved on is dropped from `done`, i.e. RE-SCORED — which
    # is the fix rather than the announcement. Rows written before the stamp existed
    # resume exactly as they always did; nothing mass re-scores.
    # ⚠ LAST ROW WINS, AND THE COUNT IS OF CLIPS, NOT ROWS. The output file is APPEND-ONLY,
    # so this feature's own success used to poison its announcement: a re-rendered clip is
    # dropped from `done`, re-scored, and a SECOND row for the same path is appended. Every
    # later run then saw both — row 1 stale (`rescored += 1`), row 2 fresh (`done.add`) —
    # so `done` came out right while `rescored` counted a repair that had already happened,
    # for the rest of the file's life. The run announced "N re-rendered -> re-scoring" and
    # re-scored nothing.
    state = {}
    if os.path.exists(args.out):  # resumable
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                stamp = row.get("wav_mtime")
                # A row with no stamp predates the stamp and resumes as it always did.
                state[row["wav"]] = not (
                    stamp is not None and os.path.isfile(row["wav"])
                    and os.path.getmtime(row["wav"]) > stamp + 1e-6)
    # ⚠ SCOPED TO THIS RUN'S INPUTS, because the arrow in the message is a promise about
    # THIS RUN. `state` covers the whole output file while `todo` is drawn from `wavs`, so a
    # run under `--limit`, `--sample`, or pointed at one engine directory while `--out`
    # accumulates across several, announced "N re-rendered -> re-scoring" about clips it
    # would never open: 500 rows, 50 stale, `--limit 10` => "50 re-rendered -> re-scoring"
    # and zero of them re-scored. The 50 stay stale and the next narrow run says the same
    # thing. Same failure the rows-vs-clips fix above was written about, reached through the
    # input set instead of through duplicate rows.
    wanted = set(wavs)
    done = {w for w, fresh in state.items() if fresh}
    stale = {w for w, fresh in state.items() if not fresh}
    rescored = len(stale & wanted)
    if os.path.exists(args.out):
        # ⚠ EVERY NUMBER IN THIS LINE IS RUN-SCOPED, AND THE FILE-WIDE ONES SAY SO. Scoping
        # `rescored` and leaving `len(done)` file-wide put two different scopes in one
        # sentence with nothing marking the seam: `--limit 10` on a 500-row output read
        # "resuming: 490 already scored" and then scored ten clips. The reader has no way to
        # tell which half is about this run, and the argument for scoping the second clause
        # — the arrow is a promise about THIS RUN — applies verbatim to the first.
        skipped = len(done & wanted)
        msg = f"resuming: {skipped} of this run's {len(wanted)} input(s) already scored"
        if rescored:
            msg += f"; {rescored} re-rendered since they were scored -> re-scoring"
        # The file-wide figures are real facts about the corpus and worth keeping — in their
        # own clause, so they cannot be read as promises about this run. Without them a
        # narrow run looks like it cleared everything.
        if len(state) > len(wanted):
            msg += (f"  [file-wide: {len(done)} scored, {len(stale)} stale of "
                    f"{len(state)}; {len(stale) - rescored} stale clip(s) are outside this "
                    f"run's inputs and stay stale until a run covers them]")
        print(msg)

    max_samples = int(MAX_SECONDS * SR)
    todo = [w for w in wavs if w not in done]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fout:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            audio, mtimes = [], []
            for w in batch:
                # Read BEFORE the audio, so a wav rewritten mid-run stamps OLD and is
                # re-scored next pass rather than carrying a stamp that vouches for
                # content it never saw.
                mtimes.append(os.path.getmtime(w))
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
                row = {"wav": w, "wav_mtime": mtimes[i]}
                row.update({name: round(scores[name][i], 6) for name in heads})
                fout.write(json.dumps(row) + "\n")
            fout.flush()
            n = start + len(batch)
            if n % (args.batch_size * 25) < args.batch_size or n == len(todo):
                print(f"  scored {n}/{len(todo)}")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
