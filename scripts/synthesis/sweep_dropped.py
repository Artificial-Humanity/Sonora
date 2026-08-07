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
    those trees are inputs we do not own the layout of.
  * ALREADY-MOVED CLIPS. A row can be `dropped` while its wav has gone elsewhere —
    `_superseded/`, most often, when a clip was re-cast under a new id and retired
    rather than judged. There is nothing to quarantine and its link is already dead;
    it is reported, not moved, because rewriting it would point at a file that this
    script never put there.

    .venv/bin/python scripts/synthesis/sweep_dropped.py --campaign <name>
    .venv/bin/python scripts/synthesis/sweep_dropped.py --campaign <name> --apply
"""
import argparse
import os
import pathlib
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_common  # noqa: E402

DATASETS = pathlib.Path("/data/model-training/datasets")
RATINGS = DATASETS / "sonora-expressive-registers" / "ratings.csv"
PROTECTED_PREFIXES = ("audit-",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.campaign.startswith(PROTECTED_PREFIXES):
        print(f"refusing: {args.campaign} is a protected corpus")
        return 2

    campaign_dir = DATASETS / args.campaign
    audio = campaign_dir / "audio"
    dropped_dir = audio / "_dropped"

    # C-M6: one shared implementation (synth_common.ratings_transaction) instead of this
    # script's own read/stamp/write flavour. The flock serialises our scripts against each
    # other, which an mtime stamp cannot do; the mtime re-check inside it is what catches
    # the audition app, which takes no lock.
    with synth_common.ratings_transaction(RATINGS, tag="sweep",
                                          dry_run=not args.apply) as (_hdr, rows):
        return _sweep(args, rows, audio, dropped_dir)


def _sweep(args, rows, audio, dropped_dir):
    # Which wavs are spoken for by some OTHER campaign?
    elsewhere = {os.path.basename(r["link"]) for r in rows
                 if r["campaign"] != args.campaign and r["link"]}

    move, blocked = [], []
    for r in rows:
        if r["campaign"] != args.campaign or r["status"] != "dropped":
            continue
        name = os.path.basename(r["link"]) or f"{r['id']}.wav"
        src = audio / name
        if name in elsewhere:
            blocked.append((r["id"], "wav is also referenced by another campaign"))
        elif not src.is_file():
            blocked.append((r["id"], "wav is not in audio/ (already moved elsewhere)"))
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
    dropped_dir.mkdir(parents=True, exist_ok=True)
    for r, name, src in move:
        shutil.move(str(src), str(dropped_dir / name))
        r["link"] = f"../{args.campaign}/audio/_dropped/{name}"

    print(f"\nquarantined {len(move)} clip(s) -> {dropped_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
