"""Quarantine a campaign's dropped clips into `audio/_dropped/`, rewriting their links.

Standing convention (owner 2026-07-31): a dropped clip is MOVED aside, not deleted,
and its ratings.csv link is rewritten to follow it. Keeping the row and the audio
preserves why something was rejected — a delete leaves a verdict pointing at nothing
and the next campaign relearns the same lesson.

Three guards, each for a way this can go wrong:

  * SHARED WAVS. One wav can be referenced by rows in more than one campaign. Moving
    it because campaign A dropped it breaks campaign B's link, where the clip may be
    a keep. Any wav referenced outside the target campaign is refused.
  * PROTECTED CORPORA. `audit-*` campaigns and licensed source audio are never swept;
    those trees are inputs we do not own the layout of. The licensed half was prose only
    until 2026-08-09 (QC-L3) — the guard is now the licence manifest itself, which is
    already the SSOT for which directories hold source corpora.
  * ALREADY-MOVED CLIPS. A row can be `dropped` while its wav has gone elsewhere —
    `_superseded/`, most often, when a clip was re-cast under a new id and retired
    rather than judged. There is nothing to quarantine and its link is already dead;
    it is reported, not moved, because rewriting it would point at a file that this
    script never put there.

    .venv/bin/python scripts/tools/sweep_dropped.py --campaign <name>
    .venv/bin/python scripts/tools/sweep_dropped.py --campaign <name> --apply
"""
import argparse
import os
import pathlib
import shutil
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
import synth_common  # noqa: E402

DATASETS = pathlib.Path("/data/model-training/datasets")
RATINGS = DATASETS / "sonora-expressive-registers" / "ratings.csv"
PROTECTED_PREFIXES = ("audit-",)
LICENSES_YAML = (pathlib.Path(__file__).resolve().parents[2]
                 / "configs" / "data_licenses.yaml")


def licensed_corpus_dirs():
    """Every directory name declared in the licence manifest, lowercased.

    QC-L3: the docstring promised that "licensed source audio" is never swept, and
    `PROTECTED_PREFIXES = ("audit-",)` was the entire guard — the rest rested on the
    accident that no licensed corpus happens to be named like a campaign. A guard that
    exists only in prose is the shape this repo keeps finding; here it would move audio
    out of a source tree we do not own the layout of, which no rewrite undoes.

    `configs/data_licenses.yaml` is already the SSOT for "which directories hold source
    corpora" — the training wall refuses any path it cannot classify there — so the set
    is read from it rather than restated. A second hand-maintained list of corpus names
    would be exactly the fork the wall exists to prevent.

    Parsed directly instead of via `matcha.data.license_wall`, which imports torch. If the
    manifest cannot be read the sweep REFUSES rather than proceeding unprotected: the
    whole point is that this guard is not allowed to be silently absent.
    """
    import yaml
    with open(LICENSES_YAML, encoding="utf-8") as f:
        raw = yaml.safe_load(f)["datasets"]
    return {d.lower() for entry in raw.values() for d in entry["dirs"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.campaign.startswith(PROTECTED_PREFIXES):
        print(f"refusing: {args.campaign} is a protected corpus")
        return 2
    if args.campaign.lower() in licensed_corpus_dirs():
        print(f"refusing: {args.campaign} is a declared source corpus "
              f"(configs/data_licenses.yaml) — we do not own its layout")
        return 2

    audio = DATASETS / args.campaign / "audio"

    # C-M6: one shared implementation (synth_common.ratings_transaction) instead of this
    # script's own read/stamp/write flavour. The flock serialises our scripts against each
    # other, which an mtime stamp cannot do; the mtime re-check inside it is what catches
    # the audition app, which takes no lock.
    with synth_common.ratings_transaction(RATINGS, tag="sweep",
                                          dry_run=not args.apply) as (_hdr, rows):
        return _sweep(args, rows, audio)


def _sweep(args, rows, audio):
    # Which wavs are spoken for by some OTHER campaign?
    elsewhere = {os.path.basename(r["link"]) for r in rows
                 if r["campaign"] != args.campaign and r["link"]}

    move, blocked = [], []
    for r in rows:
        if r["campaign"] != args.campaign or r["status"] != "dropped":
            continue
        name = os.path.basename(r["link"]) or f"{r['id']}.wav"
        # FOLLOW THE LINK (QC-L3). This looked only in `<campaign>/audio/<basename>`, so
        # a campaign registered from an engine-out subdir — which is exactly what
        # `synth_bank.sh --audio-dir` produces — reported EVERY drop as "already moved
        # elsewhere". Wrong reason, no sweep, and a report that reads as "nothing to do".
        #
        # The link already says where the wav is: it is written relative to the ratings
        # directory by `register_audition._build_row`, and the audition app resolves it
        # the same way to serve the audio. Reading it here means one answer to "where is
        # this clip" instead of two. Falls back to the conventional layout for a row with
        # no link at all.
        src = (RATINGS.parent / r["link"]).resolve() if r["link"] else audio / name
        if name in elsewhere:
            blocked.append((r["id"], "wav is also referenced by another campaign"))
        elif not src.is_file():
            blocked.append((r["id"], f"wav not found at {src} (already moved elsewhere)"))
        else:
            move.append((r, name, src))

    print(f"{len(move)} to quarantine, {len(blocked)} skipped\n")
    for r, name, _src in move:
        print(f"  MOVE  {r['id'][:44]:46s} {name}")
        print(f"        why: {(r['note'] or '')[:88]}")
    for cid, why in blocked:
        print(f"  SKIP  {cid[:44]:46s} {why}")

    if not args.apply:
        print("\nreport only — pass --apply")
        return 0
    if not move:
        # Nothing to do, and nothing to write. The sentinel leaves the transaction
        # without touching the file rather than rewriting it byte-identically.
        raise synth_common.DryRun

    # The moves happen INSIDE the transaction, so the link rewrite that describes them
    # either lands with them or the whole thing aborts before either. Previously a
    # concurrent app edit meant "wavs were moved, re-run to rewrite the links" — a state
    # where the csv points at files that are no longer there.
    # Quarantine BESIDE the clip, not at a path assembled from the campaign name. The
    # rewrite used to be the literal `../{campaign}/audio/_dropped/{name}`, which is only
    # correct for the conventional layout and silently wrong for the engine-out one — it
    # would have pointed the app at a file that was never put there. `relpath` against the
    # ratings dir reproduces the conventional link byte-for-byte and stays right for the
    # rest.
    dests = set()
    for r, name, src in move:
        dropped_dir = src.parent / "_dropped"
        dropped_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dropped_dir / name))
        r["link"] = os.path.relpath(dropped_dir / name, RATINGS.parent)
        dests.add(dropped_dir)

    print(f"\nquarantined {len(move)} clip(s) -> "
          + ", ".join(str(d) for d in sorted(dests)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
