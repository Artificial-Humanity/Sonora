"""The license wall: refuses to train on non-permissive or undeclared data.

Enforces the corpus bar of the north star's Load-Bearing Constraints (§8) in code,
so non-permissive sources
(Expresso, original-subset Emilia) can never silently enter the production
corpus. Every dataset reachable from a training filelist must be declared in
configs/data_licenses.yaml.

Two ways to fail, and they are different findings:
  * `class: blocked` — declared, recognised, and refused. Names the dataset.
  * undeclared       — no entry matches the path at all. Names the path.

⚠⚠ THERE IS NO ESCAPE HATCH, AND THERE USED TO BE (removed 2026-09-09, owner).
`SONORA_LICENSE_WALL=derisk` permitted class-`nc` data behind a taint banner.
The `nc` class was retired the same day — the 2026-08-01 fully-Apache-2.0
posture sets the bar at unrestricted open redistribution, which CC-BY-NC does
not clear — so the hatch permitted nothing, and a dormant permission is the one
route restricted material could take back into a lineage.

⚠ THE CANON WAS RIGHT BEFORE THIS CODE WAS. The north star's §8 table has read
"Corpus bar = unrestricted open redistribution — NC licences do not clear it"
since 2026-08-20, while this module went on permitting NC data behind the hatch
and citing "§8.2" — a section number that the 2026-08-20 rewrite removed. A
stale citation is how a rule and its enforcement drift apart without either
looking wrong: the doc could not be checked against the code because the code
pointed at a doc section that no longer existed.

⚠ THAT WAS THE *LICENCE* HATCH ONLY, and this docstring used to conflate two
unrelated things by calling it "the §7 de-risk escape hatch". NORTH-STAR §7
DE-RISK EXPERIMENTS ARE UNAFFECTED and continue exactly as before — the
single-channel energy run, the identity-at-init playbook, all of it. They
simply cannot be run on non-permissive data. If you came here from a §7
reference looking for the flag, it was never yours: it gated licences.
"""

import os

import yaml

_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "data_licenses.yaml",
)

_manifest_cache = None
_publish_cache = None


class LicenseWallError(RuntimeError):
    pass


def _manifest():
    global _manifest_cache
    if _manifest_cache is None:
        with open(_MANIFEST_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)["datasets"]
        _manifest_cache = {}
        for name, entry in raw.items():
            for d in entry["dirs"]:
                _manifest_cache[d.lower()] = (name, entry["class"], entry["license"])
        # ⚠ SEPARATE MAP, SEPARATE AXIS. `publish` is orthogonal to `class` and must not be
        # folded into it: the crossed delivery bank is CC-BY-4.0 and genuinely `permissive`
        # — the licence is satisfied — while shipping a model trained on it is forbidden for
        # a reason licences do not speak to (owner ruling 12, 2026-09-09: ~20 cloned real
        # LibriTTS-R voices at tens of clips each). Encoding "do not publish" as a licence
        # class would make the manifest state something false about the licence.
        global _publish_cache
        _publish_cache = {}
        for name, entry in raw.items():
            policy = entry.get("publish", "allowed")
            for d in entry["dirs"]:
                _publish_cache[d.lower()] = (name, policy, entry.get("publish_reason", ""))
    return _manifest_cache


def _candidates(component):
    """The names a single path component may be known by.

    ⚠ HUGGING FACE CACHES RENAME THE DATASET, AND A DECLARATION THAT DOES NOT ACCOUNT FOR
    IT IS DECORATIVE. Measured 2026-09-09: `expresso` was declared here for weeks while the
    only copy on disk sat at `/data/huggingface/datasets/ylacombe___expresso`, whose
    components are `huggingface`, `datasets`, `ylacombe___expresso` — none of them
    `expresso`. `classify_path` returned None for every file in it.

    ⚠ It still REFUSED that data, via the undeclared branch, so this was never a hole
    through which NC audio could pass. It is worse in a quieter way: the entry looked like
    it was doing something, and the refusal named a path instead of a dataset.

    Both cache layouts, because they differ and both exist on this machine:
      * datasets cache: `org___name`      (three underscores)
      * hub cache:      `datasets--org--name` / `models--org--name`
    Only the NAME half is offered — an org is not a dataset, and matching one would let a
    publisher's whole namespace inherit one entry's class.
    """
    yield component
    if "___" in component:
        yield component.rsplit("___", 1)[-1]
    parts = component.split("--")
    if len(parts) == 3 and parts[0] in ("datasets", "models"):
        yield parts[2]


def classify_path(path):
    """Returns (dataset_name, class, license) or None if no component matches."""
    table = _manifest()
    for component in os.path.normpath(path).split(os.sep):
        for name in _candidates(component):
            hit = table.get(name.lower())
            if hit:
                return hit
    return None


def enforce(filelist_paths):
    """Validates filelists + every audio path inside them against the manifest.

    Raises LicenseWallError on `class: blocked` data and on paths that match no declared
    dataset. ⚠ There is no mode parameter and no environment variable: the `derisk` hatch
    was removed 2026-09-09 with the `nc` class it existed to permit. See the module
    docstring — §7 de-risk EXPERIMENTS are a different thing and are unaffected.
    """
    blocked_hits, unknown = [], []
    for filelist in filelist_paths:
        seen_dirs = set()
        rows = []
        with open(filelist, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(line.split("|")[0])
        for audio_path in rows:
            d = os.path.dirname(audio_path)
            if d in seen_dirs:
                continue
            seen_dirs.add(d)
        for p in [filelist, *seen_dirs]:
            hit = classify_path(p)
            if hit is None:
                unknown.append(p)
            elif hit[1] != "permissive":
                blocked_hits.append((p, hit[0], hit[2]))
    if unknown:
        raise LicenseWallError(
            "License wall: undeclared dataset path(s) in training filelists — "
            f"declare them in configs/data_licenses.yaml first: {sorted(set(unknown))[:5]}"
        )
    if blocked_hits:
        detail = "; ".join(f"{p} -> {name} ({lic})" for p, name, lic in blocked_hits[:5])
        raise LicenseWallError(
            f"License wall: non-permissive data in training filelists: {detail}. "
            "The corpus bar is UNRESTRICTED OPEN REDISTRIBUTION (north star § 8 "
            "Load-Bearing Constraints, owner 2026-08-01, ruled again 2026-09-09). There is "
            "no override: the `nc` class and "
            "SONORA_LICENSE_WALL=derisk were both retired on 2026-09-09. If you believe "
            "this dataset clears the bar, change its entry in configs/data_licenses.yaml "
            "and say why in the commit — that is a licence decision, not a run setting."
        )


# --- the publish wall: a licence is not the only reason an artifact must not ship ---------
#
# Owner ruling 12, answered 2026-09-09. The crossed delivery bank exists to answer ONE
# question — can delivery be separated from speaker identity — and nothing trained on it
# ships. A positive result licenses a REBUILD of a publishable bank, not the publication of
# this one. Publishing on attribution alone, and publishing behind an identifiability audit,
# were both offered to the owner and refused.
#
# ⚠ HELD HERE RATHER THAN IN A NOTE, on this repo's standing lesson to prefer the shape that
# BREAKS when an assumption expires. A rule written only in prose is one an agent reads in a
# year if at all, and this one bites at exactly the moment a result is good and someone is
# eager to ship — which is when prose loses.
#
# ⚠ WHAT ARMS IT FOR A CORPUS THAT DOES NOT EXIST YET. The bank has not been built, so no
# entry names it today and this guard has nothing to fire on in the live manifest. It is
# armed anyway, by the wall above: `enforce` REFUSES AN UNDECLARED CORPUS AT TRAINING TIME,
# so the bank cannot be trained on until someone adds a manifest entry for it — and that is
# the moment they choose `publish:`. The two guards close the loop on each other.
#
# ⚠ UNDECLARED IS ALLOWED HERE, DELIBERATELY, and that is the opposite of `enforce`'s
# treatment. Measured 2026-09-09: `data/hi-fi_en-US_female/` and `data/filelists/` (VCTK)
# classify as undeclared, and both predate the wall. Refusing them would break the export of
# old checkpoints to buy no safety — anything trained SINCE the wall is necessarily declared.


def lineage_filelists(ckpt):
    """The corpus filelists a Lightning checkpoint was trained on.

    ⚠ VERIFIED AGAINST A REAL CHECKPOINT (2026-09-09), not inferred from the Lightning docs:
    `datamodule_hyper_parameters` really is written, and really does carry these two keys.
    Read out of `vat7_finetune/.../checkpoint_epoch=000_step=0003505.ckpt` — a 263 MB file —
    without torch, by unzipping `archive/data.pkl` and reading the literal strings, which is
    also why nothing here had to be executed to check it. The values were
    `data/libritts_r_full_vat_v7/{train_op,val_op}.txt`: repo-relative corpus directories,
    exactly what `classify_path` wants.

    ⚠ A checkpoint with no datamodule hparams yields NOTHING, and the caller must treat that
    as "unknown lineage", never as "clean".
    """
    dm = ckpt.get("datamodule_hyper_parameters") or {}
    return [dm[k] for k in ("train_filelist_path", "valid_filelist_path")
            if dm.get(k)]


def refuse_unpublishable(filelist_paths, what="this artifact"):
    """Refuse to export/publish an artifact whose corpus is marked `publish: forbidden`.

    Takes the filelists rather than the checkpoint so it can be exercised without torch.
    """
    _manifest()
    bad = []
    for p in filelist_paths:
        for component in os.path.normpath(p).split(os.sep):
            for nm in _candidates(component):
                hit = _publish_cache.get(nm.lower())
                if hit and hit[1] != "allowed":
                    bad.append((p, hit[0], hit[1], hit[2]))
                    break
            else:
                continue
            break
    if bad:
        detail = "; ".join(f"{p} -> {name} (publish: {policy})" for p, name, policy, _ in bad)
        reason = next((r for *_, r in bad if r), "")
        raise LicenseWallError(
            f"Publish wall: {what} was trained on a corpus that must not ship: {detail}. "
            + (f"{reason.strip()} " if reason else "")
            + "This is NOT a licence refusal — the corpus may be perfectly well licensed. "
            "It is a restriction on shipping a model built from it. Changing it is an owner "
            "decision recorded in configs/data_licenses.yaml, not a run setting, and there "
            "is no override flag."
        )
# The clean-holdout wall, as a gate rather than a naming convention (TR-M3).
#
# `data/libritts_r_holdout_devclean/README.md` said it plainly: "the wall will not stop a
# training run pointed here. The naming is the guard." What actually stopped it was an
# ACCIDENT — the 3-wide `holdout.txt` refused an 8-wide config on the vat_dim seam — and
# that accident is gone: `holdout_8w.txt` was correctly built for scoring at the current
# width, and it is the first holdout file a current config would happily TRAIN on.
#
# One epoch permanently destroys the lineage's only never-trained instrument, and there is
# no second dev-clean to cut a new one from. Every rung of the quality-gap plan is gated on
# holdout deltas, so the cost is not one run — it is the ability to measure any future run.
# Grep-verified at the time of writing that no config or script references a holdout as a
# training path: this is exposure, not an incident.
#
# Deliberately a substring test on the PATH, not a manifest lookup. It has to hold for a
# holdout that does not exist yet, and the naming convention is already the thing every
# other guard leans on. `score_holdout.py` is unaffected — it constructs `TextMelDataset`
# directly and never comes through the datamodule, which is the seam that makes this
# cheap: the sanctioned reader keeps working, and only the training path refuses.
#
# Val is refused too. A holdout used as a val set selects checkpoints on it, which is
# fitting to it by a slower route.
HOLDOUT_MARKER = "holdout"


def refuse_holdout(filelist_paths):
    hits = [p for p in filelist_paths if HOLDOUT_MARKER in str(p).lower()]
    if hits:
        raise RuntimeError(
            "REFUSING to train: a filelist path names a HOLDOUT.\n"
            + "".join(f"  {p}\n" for p in hits)
            + "  The clean holdout is the lineage's only never-trained measurement and\n"
              "  there is no second dev-clean to replace it. One epoch over it and every\n"
              "  cross-rung comparison the plan depends on becomes meaningless.\n"
              "  To SCORE a checkpoint on it, use scripts/stages/score_holdout.py, which reads\n"
              "  the filelist without ever putting it in front of an optimizer."
        )
