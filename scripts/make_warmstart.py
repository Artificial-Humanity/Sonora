"""Builds the warm-start checkpoint for the §7 de-risk run.

Lightning's `ckpt_path` resume needs an exact-shape checkpoint, but the
de-risk model differs from every existing checkpoint (247-speaker embedding
table + FiLM/VAT tensors). This script instantiates the model from the
hydra experiment config, loads a donor checkpoint with strict=False (the
multi-speaker `matcha_vctk` — Phase-0 is single-speaker and shape-
incompatible), reports exactly which tensors warm vs fresh, and saves a
resumable Lightning checkpoint (epoch 0, fresh optimizer).

Usage (from the Sonora repo root):
    python scripts/make_warmstart.py \
        --experiment derisk_energy \
        --donor /data/model-training/sonora/warmstart/matcha_vctk.ckpt \
        --out /data/model-training/sonora/warmstart/derisk_energy_init.ckpt
Then train with:
    python -m matcha.train experiment=derisk_energy ckpt_path=<out>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="derisk_energy")
    ap.add_argument("--donor", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={args.experiment}"])
    model = instantiate(cfg.model)

    donor = torch.load(args.donor, map_location="cpu", weights_only=False)
    # strict=False tolerates missing/unexpected KEYS but not shape mismatches
    # (the 109->247 speaker table) — drop mismatched tensors first.
    model_sd = model.state_dict()
    donor_sd = {}
    shape_dropped = []
    for k, v in donor["state_dict"].items():
        if k in model_sd and model_sd[k].shape != v.shape:
            shape_dropped.append(f"{k} {tuple(v.shape)}->{tuple(model_sd[k].shape)}")
        else:
            donor_sd[k] = v
    if shape_dropped:
        print("shape-mismatched (fresh):", shape_dropped)
    missing, unexpected = model.load_state_dict(donor_sd, strict=False)
    fresh = sorted(missing)
    skipped = sorted(unexpected)
    print(f"warm tensors : {len(donor['state_dict']) - len(skipped)}")
    print(f"fresh tensors: {len(fresh)} (expected: FiLM/vat_trunk + spk_emb)")
    for name in fresh:
        if "film" not in name and "vat_trunk" not in name and "spk_emb" not in name:
            raise SystemExit(f"UNEXPECTED fresh tensor (architecture drift?): {name}")
    if skipped:
        print(f"donor-only tensors skipped: {skipped}")

    trainer = Trainer(logger=False, enable_checkpointing=False, accelerator="cpu", devices=1)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
