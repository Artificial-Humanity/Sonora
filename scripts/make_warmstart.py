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


# Tensors whose CHANNEL axis may grow, and the axis it grows on. Nothing else is widened.
#
# Only the VAT trunk's input convolution qualifies, and it qualifies for a reason that is
# written down rather than assumed: contract v2 makes CHANNEL POSITION the wire format
# (ARCHITECTURE §1 — "reordering is the one edit that must never happen"). Channels 0..2
# are V/A/T and mean in the 8-wide model exactly what they meant in the 3-wide one, and
# the appended delivery block is all-zero for `unknown`. So copying the donor's weights
# into the leading slice and zeroing the rest is not an approximation of the old model —
# it IS the old model, extended with channels that contribute nothing until trained.
#
# WHY THIS EXISTS (2026-08-07). It did not, and dropping the tensor is not a small loss:
# `vat_trunk.net.0.weight` going (256,3,1) -> (256,8,1) was discarded and randomly
# re-initialised, which throws away everything vat3-24k ep099 learned about mapping V/A/T
# into the FiLM trunk — the entire conditioning pathway, on a run whose stated purpose is
# to change the WIDTH and nothing else. notes/todo.md §1 asserted make_warmstart "already
# applies" this treatment. It did not.
#
# The speaker table is the counter-example and the reason this is an allowlist rather than
# a rule: `spk_emb.weight` also grows (109 -> 247 rows from the vctk donor), and copying
# the first 109 rows would be WRONG — speaker ids do not correspond across corpora, so
# row i is a different person. Growth is only safe where position carries a contract.
_WIDENABLE = {"vat_trunk.net.0.weight": 1}


def _widen(name, donor_t, model_t):
    """Donor tensor extended to the model's shape, or None if it must not be widened."""
    axis = next((a for suffix, a in _WIDENABLE.items() if name.endswith(suffix)), None)
    if axis is None or donor_t.dim() != model_t.dim():
        return None
    # Every axis but the channel axis must match exactly, and the channel axis must GROW.
    # A shrink would silently truncate a trained channel, which is a different edit.
    for a in range(donor_t.dim()):
        if a != axis and donor_t.shape[a] != model_t.shape[a]:
            return None
    if donor_t.shape[axis] >= model_t.shape[axis]:
        return None
    out = torch.zeros_like(model_t)
    out.narrow(axis, 0, donor_t.shape[axis]).copy_(donor_t)
    return out


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
    shape_dropped, widened = [], []
    for k, v in donor["state_dict"].items():
        if k in model_sd and model_sd[k].shape != v.shape:
            grown = _widen(k, v, model_sd[k])
            if grown is None:
                shape_dropped.append(f"{k} {tuple(v.shape)}->{tuple(model_sd[k].shape)}")
            else:
                donor_sd[k] = grown
                widened.append(f"{k} {tuple(v.shape)}->{tuple(model_sd[k].shape)}")
        else:
            donor_sd[k] = v
    if widened:
        print("widened (donor channels kept, new channels zero):", widened)
    if shape_dropped:
        print("shape-mismatched (fresh):", shape_dropped)
    missing, unexpected = model.load_state_dict(donor_sd, strict=False)
    fresh = sorted(missing)
    skipped = sorted(unexpected)
    print(f"warm tensors : {len(donor['state_dict']) - len(skipped)} "
          f"({len(widened)} widened)")
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
