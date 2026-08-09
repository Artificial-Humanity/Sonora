#!/usr/bin/env python3
"""score_holdout — teacher-forced loss for every checkpoint against a never-trained set.

THE INSTRUMENT (quality-gap-plan.md § Phase 0a)
-----------------------------------------------
`loss/val_epoch` has never been a generalization measure here. Splits were re-drawn per
corpus version while every run warm-started from the previous version, so 93-97% of v3c's
val clips had been trained on under an earlier corpus. This scores checkpoints against
LibriTTS-R **dev-clean** instead — same corpus family, same recording conditions, same
license entry, 40 speakers disjoint from the corpus's 247, and no checkpoint in the
lineage has ever seen a frame of it. That makes six weeks of runs comparable for the
first time, retroactively and without training anything.

WHAT THE NUMBER IS, AND WHAT IT IS NOT
--------------------------------------
It is a RELATIVE instrument: same clips, same phonemes, same VAT labels, same paired
noise draws, one model swapped. Differences between checkpoints are real. The absolute
magnitude is not comparable to any `loss/val_epoch` in MLflow, for three reasons, all of
them deliberate:

  1. **Speaker conditioning is wrong by construction.** `n_spks` is 247 and dev-clean's
     speakers are none of them, so the derive assigns contiguous ids 0..39 that land on
     arbitrary trained speaker embeddings. Every checkpoint is handed the same wrong
     voice for the same clip, so the mismatch is a constant floor that cancels in the
     comparison — but it *is* a floor, and it is large. If the checkpoints fail to
     separate, that floor is the first suspect, and the fix is a nearest-voice mapping
     (ECAPA over the 247 vs the 40) rather than a bigger sample.
  2. **Normalisation is the v3c training corpus's**, not dev-clean's own. Scoring-only
     means no `data_statistics` re-measure: normalise with anything else and the numbers
     compare to nothing. Checkpoints from the v2 lineage were trained under mel constants
     that differ by 0.0203 / 0.0024 — under 1% of a std, and shared by every checkpoint
     scored here.
  3. `diff_loss` is masked as of 2026-08-06 (E-M5). Nothing logged before that commit is
     on this scale.

THE NOISE FLOOR, MEASURED
-------------------------
Two identical invocations (200 clips, 2 draws, 2 checkpoints) agree on every aggregate to
better than 1e-4, but NOT per clip: `prior` drifts by 2e-7, `diff` by 4e-5, and `dur` by
up to **7e-3 on 58% of clips**. The outlier is structural rather than alarming — GPU
reductions are not bit-reproducible on ROCm, and `dur_loss` reads that noise through
MAS's discrete argmax, so a near-tied alignment path flips and the duration term steps.
Chasing it with `use_deterministic_algorithms` is not worth it: the aggregates are what
this instrument reports, and they are stable to four decimals over 200 clips and better
over 5,463. Do NOT read a per-clip `dur` difference below ~1e-2 as a real effect.

Usage (in the ROCm container — see score_holdout.sh for the wrapper):

    python scripts/score_holdout.py \
        --filelist data/libritts_r_holdout_devclean/holdout.txt \
        --model-config <run>/.hydra/config.yaml \
        --ckpt vat3c_init=/data/.../warmstart/vat3c_init.ckpt \
        --ckpt vat3c_ep099=/data/.../checkpoint_epoch=099.ckpt \
        --assert-disjoint-from data/libritts_r_vat_v3c/train_op.txt \
                               data/libritts_r_vat_v3c/val_op.txt \
        --out /data/model-training/sonora/holdout_eval/per_clip.csv
"""

import argparse
import csv
import json
import os
import sys
import time
import zlib

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.environ.get("SONORA_REPO", "/sonora"))

from hydra.utils import instantiate  # noqa: E402

from matcha.data.license_wall import enforce  # noqa: E402
from matcha.data.text_mel_datamodule import TextMelBatchCollate, TextMelDataset  # noqa: E402


def parse_ckpt(spec):
    if "=" not in spec:
        raise SystemExit(f"--ckpt wants name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    if not os.path.exists(path):
        raise SystemExit(f"no such checkpoint: {path}")
    return name, path


def assert_disjoint(holdout, corpus_filelists):
    """The instrument's whole claim is that no checkpoint has seen these clips.

    Asserted on wav basenames, which are globally unique in LibriTTS-R
    (`{speaker}_{chapter}_{utt}_{seg}`), so this catches a corpus that quietly grew into
    the holdout as well as an outright path collision. A gate that cannot fail is not a
    gate: any overlap at all stops the run.
    """

    def basenames(path):
        with open(path, encoding="utf-8") as f:
            return {os.path.basename(line.split("|")[0]) for line in f if line.strip()}

    held = basenames(holdout)
    if not corpus_filelists:
        # An omitted `--assert-disjoint-from` used to mean "no check ran", silently
        # (TR-M3). The report then carried numbers that LOOK like holdout numbers and
        # carry none of the guarantee, which is worse than not scoring: a contaminated
        # comparison reads as a result. The check is the reason the instrument means
        # anything, so it is not optional — name the corpora, or say out loud that you
        # are deliberately scoring without the guarantee.
        raise SystemExit(
            "REFUSING to score with no contamination check.\n"
            "  Pass --assert-disjoint-from <corpus filelist> [...] — normally every\n"
            "  train/val filelist of the corpus the checkpoints were trained on.\n"
            "  The holdout's entire claim is that no checkpoint has seen these clips;\n"
            "  unchecked, this run would produce numbers with that claim's shape and\n"
            "  none of its content."
        )
    for corpus in corpus_filelists:
        overlap = held & basenames(corpus)
        if overlap:
            raise SystemExit(
                f"CONTAMINATED: {len(overlap)} holdout clips also appear in {corpus}\n"
                f"  e.g. {sorted(overlap)[:3]}\n"
                "  The holdout is only worth scoring while this set is empty."
            )
        print(f"  disjoint from {corpus}", flush=True)
    return len(held)


def build_model(cfg, ckpt_path, device, allow_partial):
    model = instantiate(cfg.model)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        msg = (f"{os.path.basename(ckpt_path)}: missing={len(missing)} "
               f"unexpected={len(unexpected)} -- architecture does not match this config")
        if not allow_partial:
            raise SystemExit(f"  !! {msg}\n     (--allow-partial to score anyway)")
        print(f"  [warn] {msg}", flush=True)
    return model.to(device).eval()


def build_dataset(cfg, filelist):
    d = cfg.data
    # `data.vat_dim` / `data.load_vat` did not exist before 2026-08-04 — the vat3 run's
    # own config has neither, and it is exactly the kind of config this instrument is for
    # (scoring an old checkpoint under the constants it actually trained with). Defaulting
    # is safe rather than lax: every VAT-era filelist carries a 3-vector, and a filelist
    # that does not will fail the parse loudly. Reading them off the MODEL is the tell —
    # `model.vat_dim` is the single source of truth (STATE.md, delivery-channel seams).
    vat_dim = d.get("vat_dim", cfg.model.get("vat_dim", 3))
    load_vat = d.get("load_vat", vat_dim > 0)
    return TextMelDataset(
        filelist_path=filelist,
        n_spks=d.n_spks,
        cleaners=d.cleaners,
        add_blank=d.add_blank,
        n_fft=d.n_fft,
        n_mels=d.n_feats,
        sample_rate=d.sample_rate,
        hop_length=d.hop_length,
        win_length=d.win_length,
        f_min=d.f_min,
        f_max=d.f_max,
        data_parameters=OmegaConf.to_container(d.data_statistics, resolve=True),
        seed=1234,
        load_durations=d.load_durations,
        load_vat=load_vat,
        vat_dim=vat_dim,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--model-config", required=True,
                    help="a run's .hydra/config.yaml — supplies BOTH the model "
                         "architecture and the mel normalisation constants")
    ap.add_argument("--ckpt", action="append", required=True, metavar="name=path")
    ap.add_argument("--assert-disjoint-from", nargs="*", default=[],
                    help="corpus filelists the holdout must share no clip with")
    ap.add_argument("--samples", type=int, default=4,
                    help="paired flow-matching draws per clip")
    ap.add_argument("--limit", type=int, default=0, help="first N clips (0 = all)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="score a checkpoint whose keys do not match the config")
    ap.add_argument("--out", required=True, help="per-clip CSV; a .json report sits beside it")
    args = ap.parse_args()

    enforce([args.filelist])

    print("holdout:", flush=True)
    n_held = assert_disjoint(args.filelist, args.assert_disjoint_from)
    print(f"  {n_held} clips", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.model_config)
    collate = TextMelBatchCollate(cfg.data.n_spks)
    ds = build_dataset(cfg, args.filelist)
    idx = list(range(len(ds)))
    if args.limit:
        idx = idx[: args.limit]
    print(f"  scoring {len(idx)} of {len(ds)} clips x {args.samples} draws "
          f"on {device.type}", flush=True)

    # CHECKPOINTS ARE THE INNER LOOP, deliberately. Building a datapoint means reading the
    # wav and computing its mel on CPU, and that dominates: with the checkpoint outer, an
    # 8.7 h holdout is re-melled once per checkpoint and the sweep costs ~10 h. Held here,
    # each clip is prepared once and handed to every model, which is also the stronger
    # guarantee — the models are compared on the same tensor object, not on two
    # independently recomputed ones. The whole lineage is ~86 MB of weights apiece, so
    # resident models cost far less VRAM than the redundant work costs wall clock.
    loaded = []
    for spec in args.ckpt:
        name, path = parse_ckpt(spec)
        print(f"  loading {name}", flush=True)
        loaded.append((name, path, build_model(cfg, path, device, args.allow_partial)))

    rows, summary = [], {}
    acc_all = {name: {"dur": 0.0, "prior": 0.0, "diff": 0.0} for name, _, _ in loaded}
    t0 = time.time()
    for n, i in enumerate(idx):
        dp = ds[i]
        clip = os.path.basename(ds.filepaths_and_text[i][0])
        batch = collate([dp])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        for name, _, model in loaded:
            acc = {"dur": 0.0, "prior": 0.0, "diff": 0.0}
            for s in range(args.samples):
                # PAIRED: the same seed for the same (clip, draw) across every
                # checkpoint, so both see identical timesteps and noise. This is the
                # variance term that buried the phoneme-fix effect in the aggregate.
                # NOT builtin hash() — str hashing is salted per process (PYTHONHASHSEED),
                # so that version paired correctly within a run and drifted between runs:
                # two invocations of the same 200 clips agreed on dur/prior to the digit
                # and disagreed on diff by 7%, because only diff reads the noise. crc32 is
                # stable, so a checkpoint scored next month is comparable to one scored
                # today without rescoring the whole lineage.
                torch.manual_seed(zlib.crc32(f"{clip}|{s}".encode()))
                with torch.no_grad():
                    dur, prior, diff, *_ = model(
                        x=batch["x"], x_lengths=batch["x_lengths"],
                        y=batch["y"], y_lengths=batch["y_lengths"],
                        spks=batch["spks"], out_size=None,
                        durations=batch.get("durations"), vat=batch.get("vat"),
                    )
                acc["dur"] += float(dur)
                acc["prior"] += float(prior)
                acc["diff"] += float(diff)
            row = {"clip": clip, "ckpt": name,
                   **{k: v / args.samples for k, v in acc.items()}}
            row["total"] = row["dur"] + row["prior"] + row["diff"]
            rows.append(row)
            for k in acc_all[name]:
                acc_all[name][k] += row[k]
        if (n + 1) % 500 == 0:
            print(f"    {n+1}/{len(idx)}  {(n+1)/(time.time()-t0):.1f} clips/s", flush=True)

    print("", flush=True)
    for name, path, _ in loaded:
        summary[name] = {k: v / len(idx) for k, v in acc_all[name].items()}
        summary[name]["total"] = sum(summary[name].values())
        summary[name]["ckpt_path"] = path
        print(f"  === {name} ===  dur {summary[name]['dur']:.4f}  "
              f"prior {summary[name]['prior']:.4f}  diff {summary[name]['diff']:.4f}  "
              f"total {summary[name]['total']:.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "ckpt", "dur", "prior", "diff", "total"])
        w.writeheader()
        w.writerows(rows)
    report = {
        "filelist": args.filelist,
        "model_config": args.model_config,
        "clips": len(idx),
        "samples": args.samples,
        "data_statistics": OmegaConf.to_container(cfg.data.data_statistics, resolve=True),
        "checkpoints": summary,
    }
    report_path = os.path.splitext(args.out)[0] + ".json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  wrote {len(rows)} rows -> {args.out}\n  report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
