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

# The keys an entry may carry. ⚠ CHECKED AT LOAD (#422): `publish` is read with a default of
# `allowed`, so before this a misspelled key — `publsh:`, `publish_policy:`, a mis-indented
# `publish:` — disarmed the wall with no error, no warning and no failing test. The VALUE side
# always failed closed (`!= "allowed"` refuses `forbiden`); the KEY side is what fails open,
# and the whole guard rests on one future entry being spelled right.
_ENTRY_KEYS = {"dirs", "license", "class", "publish", "publish_reason"}
_PUBLISH_VALUES = {"allowed", "forbidden"}

# Top-level checkpoint key carrying the filelists of every EARLIER training stage. Written by
# `on_save_checkpoint`, read back by `on_load_checkpoint` and by `lineage_filelists` (#421).
LINEAGE_KEY = "sonora_lineage"

# The marker a warm start records when its donor's corpus was never written down (#425).
# ⚠ DELIBERATELY NOT A PATH. It travels as an ordinary member of the lineage list, so every
# carrier that already copies that list carries it for free — and a reader that has never
# heard of it still fails CLOSED, because `refuse_unpublishable` cannot open it and returns
# it as unread, which every caller already treats as "not a clean result".
LINEAGE_UNKNOWN = "<UNKNOWN: pre-wall donor, ancestry unrecorded>"


class LicenseWallError(RuntimeError):
    pass


def _read_manifest():
    with open(_MANIFEST_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)["datasets"]
    for name, entry in raw.items():
        unknown = set(entry) - _ENTRY_KEYS
        if unknown:
            raise LicenseWallError(
                f"data_licenses.yaml: entry {name!r} has unknown key(s) {sorted(unknown)}; "
                f"the keys are {sorted(_ENTRY_KEYS)}. A misspelled `publish:` would read "
                "as `allowed` and disarm the publish wall, so this is refused at load."
            )
        policy = entry.get("publish", "allowed")
        if policy not in _PUBLISH_VALUES:
            raise LicenseWallError(
                f"data_licenses.yaml: entry {name!r} has publish: {policy!r}; it must be "
                f"one of {sorted(_PUBLISH_VALUES)}."
            )
    return raw


def _manifest():
    global _manifest_cache
    if _manifest_cache is None:
        _manifest_cache = {}
        for name, entry in _read_manifest().items():
            for d in entry["dirs"]:
                _manifest_cache[d.lower()] = (name, entry["class"], entry["license"])
    return _manifest_cache


def _publish():
    """The `publish` map, built on its own (#424): it used to be filled only inside
    `_manifest()`'s cold-cache branch, so a test that injected `_manifest_cache` alone left
    this one `None` and `refuse_unpublishable` died with an AttributeError instead of either
    answer. Each map now fills itself from the yaml when it is missing.

    ⚠ SEPARATE MAP, SEPARATE AXIS. `publish` is orthogonal to `class` and must not be
    folded into it. A corpus can be perfectly licensed and still be one nothing trained
    on it may ship, for a reason licences do not speak to. Encoding "do not publish" as a
    licence class would make the manifest state something false about the licence.

    The standing user is the private-lineage firewall in `docs/audiobook-corpus-policy.md`,
    whose rules are absolute and had no code behind them before this axis existed: nothing
    leaves the machine, and a private branch never warm-starts into the public lineage.
    That is `publish: forbidden` plus warm-start lineage propagation (#421).
    """
    global _publish_cache
    if _publish_cache is None:
        _publish_cache = {}
        for name, entry in _read_manifest().items():
            policy = entry.get("publish", "allowed")
            for d in entry["dirs"]:
                _publish_cache[d.lower()] = (name, policy, entry.get("publish_reason", ""))
    return _publish_cache


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


def _lookup(path, table):
    for component in os.path.normpath(path).split(os.sep):
        for name in _candidates(component):
            hit = table.get(name.lower())
            if hit:
                return hit
    return None


def classify_path(path):
    """Returns (dataset_name, class, license) or None if no component matches."""
    return _lookup(path, _manifest())


def publish_policy(path):
    """Returns (dataset_name, publish, publish_reason) or None if no component matches."""
    return _lookup(path, _publish())


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
# ⚠⚠ THE CROSSED DELIVERY BANK IS NOT WHAT THIS GUARDS ANY MORE, AND THAT REVERSED.
# Owner ruling 12 was answered 2026-09-09 as "diagnostic only — nothing trained on the bank
# ships", and this block stated that as current for the life of the wall. The owner REVISED
# it on 2026-09-10: the bank is PUBLISHABLE and models trained on it may ship. They revisited
# on two grounds — the restriction was never a licence matter (the bank is CC-BY-4.0 and the
# licence is satisfied), and diagnostic-only was stricter than their standing position on
# cloning real people, which is "not forbidden and not off the table, just not a focus".
# Diagnostic-only had hardened that into a prohibition.
#
# ⚠ NOTHING HERE WAS UNBUILT, WHICH IS WHY THE REVISION IS EASY TO MISS. The `publish` axis,
# `refuse_unpublishable` and the export and promotion call sites are unchanged. What changed
# is only that the bank is not marked `publish: forbidden` — and the mechanism has a STRONGER
# standing user than the one it was built for: the `docs/audiobook-corpus-policy.md` firewall,
# whose rules are described as absolute and had no code behind them at all.
#
# ⚠ ONE CONSIDERATION IS RECORDED RATHER THAN SETTLED, per the owner. The argument against
# relaxing was that non-commercial intent has no bearing on likeness: a freely distributed
# model reproducing ~20 identifiable LibriVox volunteers is arguably MORE exposed than a
# private one, and consent to a CC-BY recording is not consent to a voice clone. Copyright
# and likeness are separate regimes. The owner's call stands; this is here so the trade is
# legible if it is ever revisited, not to relitigate it.
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
    """Every corpus filelist a Lightning checkpoint descends from.

    ⚠ VERIFIED AGAINST A REAL CHECKPOINT (2026-09-09), not inferred from the Lightning docs:
    `datamodule_hyper_parameters` really is written, and really does carry these two keys.
    Read out of `vat7_finetune/.../checkpoint_epoch=000_step=0003505.ckpt` — a 263 MB file —
    without torch, by unzipping `archive/data.pkl` and reading the literal strings, which is
    also why nothing here had to be executed to check it. The values were
    `data/libritts_r_full_vat_v7/{train_op,val_op}.txt`: repo-relative corpus directories,
    exactly what `classify_path` wants.

    ⚠ THOSE KEYS NAME THE LAST STAGE ONLY (#421). This lineage is warm-started stage over
    stage, and `make_warmstart.py` writes the init with no datamodule attached, so the
    donor's corpus did not travel: a fine-tune of a bank-trained checkpoint exported with
    only the fine-tune corpus in view, while the ~20 cloned voices sat in its weights. The
    ancestors now ride under `LINEAGE_KEY`: `make_warmstart.py` sets it from the donor, and
    the module's `on_load_checkpoint`/`on_save_checkpoint` carry it through every later
    save. The rule the gap breaks is "a descendant WAS trained on this — at an earlier stage
    — so the ancestors count", and it holds for whatever is marked `publish: forbidden`. It
    was written when the crossed bank was the example; ruling 12 was relaxed on 2026-09-10 and
    the bank is publishable, but the lineage requirement is the firewall's too and is not
    specific to either corpus.

    ⚠ A checkpoint with no datamodule hparams and no lineage key yields NOTHING, and the
    caller must treat that as "unknown lineage", never as "clean". This function stays
    honest about what is RECORDED rather than converting the gap itself: an export has to
    tell "nothing recorded at all" from "warm-started from something unrecorded", and
    `lineage_gaps` gives those two different sentences. The conversion belongs at the point
    a lineage is ADOPTED into a descendant, which is `carried_lineage` — used by
    `make_warmstart.py` and by the module's `on_load_checkpoint` (#425).
    """
    dm = ckpt.get("datamodule_hyper_parameters") or {}
    own = [dm[k] for k in ("train_filelist_path", "valid_filelist_path") if dm.get(k)]
    out = []
    for p in [*(ckpt.get(LINEAGE_KEY) or []), *own]:
        if p not in out:
            out.append(p)
    return out


def carried_lineage(donor):
    """The lineage a warm-start init must record, given the donor checkpoint it starts from.

    ⚠ AN UNRECORDED ANCESTRY MUST NOT TRAVEL AS AN EMPTY LIST (#425). `make_warmstart.py`
    wrote `lineage_filelists(donor)` straight through, so a donor carrying neither
    `datamodule_hyper_parameters` nor `LINEAGE_KEY` — a pre-wall checkpoint, whose corpus was
    never written down — produced `sonora_lineage: []`. The fine-tune loaded that, saved its
    OWN corpus beside it, and from there every descendant read as a complete record:
    `check_publishable.py` printed the fine-tune corpus and exited 0 on "no `publish:
    forbidden` corpus anywhere in the lineage", which is a claim about ancestry the record
    could not support. The invariant stated in three docstrings — empty is UNKNOWN, never
    clean — held for exactly one generation, because the UNKNOWN lived only in a print.

    ⚠ A VALUE IN THE LIST, NOT A SECOND KEY BESIDE IT, and that is the whole design. See
    `LINEAGE_UNKNOWN`: it fails closed for a reader that does not know about it, where a
    boolean flag would fail open the first time someone forgot to check it.
    """
    return lineage_filelists(donor) or [LINEAGE_UNKNOWN]


def lineage_gaps(lineage, unread=()):
    """Every reason this lineage is not a complete record. Empty list = complete.

    The callers print these and treat a non-empty list as "not a clean result":
    `check_publishable.py` exits 2 on it, and the export prints it (#426). Before this, an
    empty lineage produced NO message on the export path at all — `refuse_unpublishable([])`
    returns `[]`, so the loop that reports unread filelists had nothing to print and the log
    of a checkpoint with nothing to check was identical to one whose lineage was read and
    cleared. AGENTS.md §5b: a tool's failure is easily mistaken for its negative result, so
    assert the instrument ran before believing it.

    `unread` is `refuse_unpublishable`'s return. `LINEAGE_UNKNOWN` lands in it as well — the
    fail-closed property that makes the marker safe — so it is filtered out here and reported
    once, from the lineage, in the wording that fits it rather than as a file that would not
    open.
    """
    gaps = []
    if not lineage:
        gaps.append(
            "lineage UNKNOWN: the checkpoint carries no datamodule hparams and no lineage "
            "key (pre-wall?). Nothing was checked, so this is not a clean result.")
    if LINEAGE_UNKNOWN in lineage:
        gaps.append(
            "lineage UNKNOWN for an ancestor: this was warm-started from a pre-wall donor "
            "whose corpus was never recorded. The stages below it were checked; what the "
            "donor trained on is unknown, so this is not a clean result.")
    gaps.extend(
        f"could not open {p}: its audio paths were NOT classified, so this filelist is "
        "UNKNOWN on that half and this is not a clean result"
        for p in unread if p != LINEAGE_UNKNOWN)
    return gaps


def refuse_unpublishable(filelist_paths, what="this artifact", root=None):
    """Refuse to export/publish an artifact whose corpus is marked `publish: forbidden`.

    Takes the filelists rather than the checkpoint so it can be exercised without torch.

    ⚠ THE AUDIO INSIDE THE FILELIST IS CLASSIFIED TOO, exactly as `enforce` does (#420).
    Every corpus since v5 is a filelists-only directory under `merged_vat_corpora` whose
    audio lives elsewhere; that is how the expressive bank joined, and it is how a crossed
    bank would join. Walking the filelist PATH alone let that configuration through — the
    guard fired only on a corpus trained from its own directory, which this lineage has
    never done. So each filelist is opened and every audio directory in it is looked up on
    its own components.

    A filelist that cannot be read — repo-relative paths are resolved against `root` — is
    RETURNED, not treated as clean: its path components are still checked, its audio is
    not, and the caller must say so. Returns the list of filelists it could not open.
    """
    bad, unread = [], []
    for p in filelist_paths:
        checked = [p]
        resolved = p if root is None or os.path.isabs(p) else os.path.join(root, p)
        try:
            with open(resolved, encoding="utf-8") as f:
                seen = set()
                for line in f:
                    line = line.strip()
                    if line:
                        seen.add(os.path.dirname(line.split("|")[0]))
            checked.extend(sorted(seen))
        except OSError:
            unread.append(p)
        for q in checked:
            hit = publish_policy(q)
            if hit and hit[1] != "allowed":
                bad.append((q, hit[0], hit[1], hit[2]))
    if bad:
        detail = "; ".join(f"{p} -> {name} (publish: {policy})" for p, name, policy, _ in bad[:5])
        reason = next((r for *_, r in bad if r), "")
        raise LicenseWallError(
            f"Publish wall: {what} was trained on a corpus that must not ship: {detail}. "
            + (f"{reason.strip()} " if reason else "")
            + "This is NOT a licence refusal — the corpus may be perfectly well licensed. "
            "It is a restriction on shipping a model built from it. Changing it is an owner "
            "decision recorded in configs/data_licenses.yaml, not a run setting, and there "
            "is no override flag."
        )
    return unread


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
