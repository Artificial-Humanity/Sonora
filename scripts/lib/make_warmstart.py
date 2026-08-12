"""Builds the warm-start checkpoint for the §7 de-risk run.

Lightning's `ckpt_path` resume needs an exact-shape checkpoint, but the
de-risk model differs from every existing checkpoint (247-speaker embedding
table + FiLM/VAT tensors). This script instantiates the model from the
hydra experiment config, loads a donor checkpoint with strict=False (the
multi-speaker `matcha_vctk` — Phase-0 is single-speaker and shape-
incompatible), reports exactly which tensors warm vs fresh, and saves a
resumable Lightning checkpoint (epoch 0, fresh optimizer).

Usage (from the Sonora repo root):
    python scripts/lib/make_warmstart.py \
        --experiment derisk_energy \
        --donor /data/model-training/sonora/warmstart/matcha_vctk.ckpt \
        --out /data/model-training/sonora/warmstart/derisk_energy_init.ckpt
Then train with:
    python -m matcha.train experiment=derisk_energy ckpt_path=<out>
"""

import argparse
import collections
import json
import os
import sys

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

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
# `spk_emb.weight` is the case that shows why this is an allowlist with a POLICY rather
# than a rule. It also grows — 109 -> 247 rows from the vctk donor, and 247 -> 2,655 when
# Emilia's 2,408 speakers merge — but the two growths are not the same operation:
#
#   * vctk -> LibriTTS: different corpora, so row `i` is a DIFFERENT PERSON. Copying the
#     leading rows is silently wrong and the script must refuse.
#   * LibriTTS v4 -> a merged corpus: the same LibriTTS speakers keep their indices and the
#     new ones append, so rows 0..246 keep their meaning exactly.
#
# Nothing in a checkpoint says which of those it is. So widening the speaker table requires
# the donor's `speakers.json` on the command line and PROVES the prefix property before
# touching anything (`--donor-speakers`); absent that proof the table stays fresh, which
# costs relearning and risks nothing.
#
# The new rows also want a different filling from the VAT trunk's. A zero CHANNEL means
# "contributes nothing", which is exactly right for an unknown delivery lane. A zero
# speaker EMBEDDING is not a neutral speaker — it is one specific point that every new
# speaker would start from identically, which is a poor init for something that must be
# learned. New speaker rows therefore keep the model's own fresh initialisation.
_WIDENABLE = {
    "vat_trunk.net.0.weight": (1, "zero"),
    "spk_emb.weight": (0, "model_init"),
}


def _index_map(speakers):
    """Every `*_id_to_index` block in a speakers.json, as ONE id -> index map.

    A merged corpus keeps its sources in separate namespaces — v5 has
    `libritts_id_to_index` (247) beside `emilia_id_to_index` (2,253) — because the ids come
    from different datasets and nothing guarantees they cannot look alike. Reading only the
    LibriTTS block here would have made the prefix check technically pass and its report a
    lie: "247 donor speakers keep their index; 0 appended" on a corpus that appends 2,253.
    A check whose summary is wrong is worse than no check, because it is quoted.

    Refuses on a cross-namespace collision rather than letting one block win, since that is
    two speakers sharing an embedding row — the exact confusion the prefix proof exists to
    prevent, arriving from the other side.

    TR-L2: that refusal used to fire only on DIFFERENT indices, so two namespaces assigning
    the same id string to the SAME index collapsed into one dict entry and the check passed
    with two speakers on one embedding row — the identical outcome, reached quietly. The id
    is refused now regardless of index: a shared id across namespaces is the defect itself,
    and whether the two happen to agree on a row number says nothing about whether they are
    the same voice. Unreachable through `merge_emilia_corpus` (its own collision abort
    refuses shared ids outright); the gap is against a hand-assembled `speakers.json`, which
    is exactly the input a proof should not assume was produced correctly.
    """
    merged, source = {}, {}
    for key, block in sorted((speakers or {}).items()):
        if not key.endswith("_id_to_index") or not isinstance(block, dict):
            continue
        for sid, idx in block.items():
            if sid in merged and source[sid] != key:
                raise SystemExit(
                    f"ABORT: speaker id {sid!r} appears in both {source[sid]} and {key} "
                    f"(indices {merged[sid]} and {idx}). Namespaces exist because ids from "
                    f"different datasets can look alike; one id in two of them is two "
                    f"speakers sharing an embedding row, whether or not the indices match.")
            merged[sid], source[sid] = idx, key
    return merged


def _speaker_prefix_ok(donor_speakers, model_speakers):
    """Is the donor's speaker index map a PREFIX of the model's? -> (ok, detail)."""
    if not donor_speakers or not model_speakers:
        return False, "no speaker map supplied (--donor-speakers)"
    old, new = _index_map(donor_speakers), _index_map(model_speakers)
    if not old or not new:
        return False, "a speakers.json has no `*_id_to_index` block"
    dup = [i for i, n in collections.Counter(new.values()).items() if n > 1]
    if dup:
        return False, (f"{len(dup)} embedding row(s) are claimed by more than one speaker, "
                       f"e.g. index {dup[0]}")
    moved = [s for s, i in old.items() if new.get(s) != i]
    if moved:
        return False, (f"{len(moved)} speaker(s) changed index, e.g. "
                       f"{moved[0]}: {old[moved[0]]} -> {new.get(moved[0])}")
    if len(new) < len(old):
        return False, f"the new map has FEWER speakers ({len(new)} < {len(old)})"
    return True, f"{len(old)} donor speakers keep their index; {len(new) - len(old)} appended"


def _policy(name):
    """Which filling the new slice of `name` gets — reported, so the log cannot lie."""
    hit = next((v for suffix, v in _WIDENABLE.items() if name.endswith(suffix)), None)
    return hit[1] if hit else "n/a"


def _widen(name, donor_t, model_t, allow_speakers=False):
    """Donor tensor extended to the model's shape, or None if it must not be widened."""
    hit = next((v for suffix, v in _WIDENABLE.items() if name.endswith(suffix)), None)
    if hit is None or donor_t.dim() != model_t.dim():
        return None
    axis, policy = hit
    if policy == "model_init" and not allow_speakers:
        return None          # no proof of the prefix property — refuse, do not guess
    # Every axis but the growing one must match exactly, and it must GROW. A shrink would
    # silently truncate a trained channel or speaker, which is a different edit entirely.
    for a in range(donor_t.dim()):
        if a != axis and donor_t.shape[a] != model_t.shape[a]:
            return None
    if donor_t.shape[axis] >= model_t.shape[axis]:
        return None
    out = torch.zeros_like(model_t) if policy == "zero" else model_t.clone()
    out.narrow(axis, 0, donor_t.shape[axis]).copy_(donor_t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="derisk_energy")
    ap.add_argument("--donor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--donor-speakers", default=None,
                    help="the donor corpus's speakers.json. Supplying it PERMITS the "
                         "speaker table to be widened, and only if this map is a prefix of "
                         "the new corpus's — same ids at the same indices, new ones "
                         "appended. Without it the table stays fresh: a checkpoint does "
                         "not record which corpus its rows belong to, and copying rows "
                         "across corpora points every speaker at someone else.")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="extra hydra override, repeatable. MUST match the overrides the "
                         "run itself will use: this script builds the model to decide "
                         "which tensors warm, so a warm start composed from a different "
                         "config is a different model. Getting it wrong surfaced as a "
                         "state_dict shape error 2026-08-08 — the loud failure; the quiet "
                         "one is a config that happens to agree on shapes and not on "
                         "meaning.")
    args = ap.parse_args()

    config_dir = os.path.join(_SONORA_REPO, "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train.yaml",
                      overrides=[f"experiment={args.experiment}", *args.override])
    if args.override:
        print("overrides: " + " ".join(args.override))
    model = instantiate(cfg.model)

    # The proof, before anything is copied. The new corpus's map sits beside the filelist
    # the experiment already names, so there is nothing to pass for that half.
    allow_speakers = False
    if args.donor_speakers:
        new_path = os.path.join(os.path.dirname(cfg.data.train_filelist_path), "speakers.json")
        try:
            with open(args.donor_speakers, encoding="utf-8") as f:
                donor_spk = json.load(f)
            with open(new_path, encoding="utf-8") as f:
                new_spk = json.load(f)
        except OSError as exc:
            raise SystemExit(f"!! --donor-speakers: {exc}")
        allow_speakers, detail = _speaker_prefix_ok(donor_spk, new_spk)
        print(f"speaker map: {'PREFIX OK' if allow_speakers else 'REFUSED'} — {detail}")
        if not allow_speakers:
            raise SystemExit(
                "!! the donor's speaker indices are not preserved in the new corpus, so "
                "widening\n   spk_emb.weight would point trained rows at different "
                "people. Re-derive the\n   corpus preserving the index map, or drop "
                "--donor-speakers to start the table fresh.")

    donor = torch.load(args.donor, map_location="cpu", weights_only=False)
    # strict=False tolerates missing/unexpected KEYS but not shape mismatches
    # (the 109->247 speaker table) — drop mismatched tensors first.
    model_sd = model.state_dict()
    donor_sd = {}
    shape_dropped, widened = [], []
    for k, v in donor["state_dict"].items():
        if k in model_sd and model_sd[k].shape != v.shape:
            grown = _widen(k, v, model_sd[k], allow_speakers=allow_speakers)
            if grown is None:
                shape_dropped.append(f"{k} {tuple(v.shape)}->{tuple(model_sd[k].shape)}")
            else:
                donor_sd[k] = grown
                widened.append((k, f"{k} {tuple(v.shape)}->{tuple(model_sd[k].shape)}"))
        else:
            donor_sd[k] = v
    if widened:
        # The filling differs per tensor, so the line says which one each got — the old
        # wording ("new channels zero") was printed for spk_emb too, where it is false.
        print("widened — donor slice kept, and the rest:")
        for name, detail in widened:
            print(f"  {detail}   new: {_policy(name)}")
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
