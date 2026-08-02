# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""seed_delivery — pre-fill the Dialogue/Neutral axis so the ear pass is a binary.

The `delivery` column is a MIX-BALANCE axis (owner 2026-07-28): it exists so the
training set does not over-concentrate in one speaking mode. Its values are
Dialogue, Documentary, Neutral, Newscaster, Speech (the last added 2026-07-29:
public address — rally, sermon, toast; a mode of address, not an emotional colour,
which is why the oratory registers alone weren't enough).

For balance, coverage is what matters, and it stood at 10 of 730 keeps. But the
values do not all come from the same place:

  * Dialogue vs Neutral is a property of the TEXT — is this a character speaking,
    or is it narration? That is knowable without listening.
  * Newscaster vs Documentary is a property of the RENDER. An engine can deliver
    narration in a broadcast voice, and 9 of the 10 existing tags are VibeVoice,
    which is exactly that happening.
  * Speech is a property of BOTH — the text must address an audience, and the
    render must project. This seeder does not attempt it.

So this script seeds ONLY the text axis. What remains for the owner is the fast
binary "does this actually sound like a broadcast read?" rather than a multi-way
judgement on 700+ clips. It never emits Newscaster or Documentary — those are
promotions the ear makes from a seeded Neutral — and never Speech, which the ear
promotes from a seeded Dialogue.

SIGNALS, in order of authority:

1. **Book lane filenames.** `_nar_` / `_dia_` are written by book_ingest's SCM quote
   detection, which already resolved speaker attribution against the source text.
   That is a stronger signal than anything inferable later. 164 rows, clean split.

2. **The register label**, for authored campaigns. It is curated and present on all
   820 rows, so no text join is needed and `audit-markup-v0` (whose ids are `mk_`-
   prefixed) is covered like everything else.

NOT SEEDED, deliberately:
  * The embodiment registers. Their texts are narration with an embedded quote
    ("Then Marcus jabbed his stick and growled, \"This net is mine tonight.\"") —
    genuinely both modes in one clip, so a guess would be noise in a balance axis.
  * audit-tension-v2 / audit-valence-v1 — instrument-calibration sweeps over
    LibriTTS speakers, not expressive-corpus material.
  * audit-emilia-keeps-v1 — real mined audio, and per notes/emilia-mining-and-verdict.md
    its keeps are "news broadcasts, technical webinars, and similar", so this is
    likely where the real Newscaster/Documentary mass lives. Ear-only by nature.

Existing values are NEVER overwritten — the owner's 10 hand-set tags win.

Usage:
    .venv/bin/python scripts/synthesis/seed_delivery.py --ratings <ratings.csv>            # dry run
    .venv/bin/python scripts/synthesis/seed_delivery.py --ratings <ratings.csv> --apply
"""
import argparse
import csv
import datetime
import pathlib
import shutil
import sys
from collections import Counter

# Third-person prose registers: the narrator is speaking, not a character.
NARRATION_REGISTERS = {"neutral_narration", "narrative", "narration_low_western"}

# Narrator voicing a character mid-sentence. Both modes at once — leave to the ear.
EMBODIMENT_REGISTERS = {"embodiment", "narrative embodiment", "neutral embodiment"}

# Not expressive-corpus material; seeding them would pollute the balance counts.
SKIP_CAMPAIGNS = {"audit-tension-v2", "audit-valence-v1", "audit-emilia-keeps-v1"}


def classify(row):
    """Return 'Dialogue' | 'Neutral' | None (leave for the ear)."""
    cid, campaign, register = row["id"], row["campaign"], row["register"]

    if campaign in SKIP_CAMPAIGNS:
        return None

    # Signal 1 — the book lane already resolved this against the source text.
    if "_nar_" in cid:
        return "Neutral"
    if "_dia_" in cid:
        return "Dialogue"

    # Signal 2 — the curated register label.
    if register in EMBODIMENT_REGISTERS:
        return None
    if register in NARRATION_REGISTERS:
        return "Neutral"
    # An unrecognized register is NOT evidence of Dialogue. Falling through to
    # it mass-mislabeled anything outside the curated vocabulary — and the
    # librivox real-audio campaigns are exactly that shape (blank register, no
    # _nar_/_dia_ id, not in SKIP_CAMPAIGNS), so the documented "safe to
    # repeat" re-run would have stamped audiobook narration and Dickens's
    # Speeches as Dialogue. Unknown means unknown; leave it for the ear.
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
    args = ap.parse_args()

    path = pathlib.Path(args.ratings)
    if not path.is_file():
        sys.exit(f"no such file: {path}")

    # The audition app is a live writer: every rating/tag click does a full
    # read-modify-write of this file. If it commits between our read and our
    # write, our stale copy silently ERASES whatever was just tagged — which is
    # exactly how an owner-set accent value was lost on 2026-07-26. Stamp the
    # mtime now and refuse to write if it moved.
    mtime_at_read = path.stat().st_mtime_ns

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames, rows = reader.fieldnames, list(reader)

    if "delivery" not in fieldnames:
        sys.exit("ratings.csv has no `delivery` column")

    seeded, kept, left = Counter(), 0, Counter()
    for row in rows:
        if row.get("delivery", "").strip():
            kept += 1                       # owner-set; never touch
            continue
        verdict = classify(row)
        if verdict is None:
            left[row["campaign"] if row["campaign"] in SKIP_CAMPAIGNS else "embodiment"] += 1
            continue
        if args.apply:
            row["delivery"] = verdict
        seeded[verdict] += 1

    if args.apply:
        if path.stat().st_mtime_ns != mtime_at_read:
            sys.exit(
                "ABORTED: ratings.csv changed while this was running — the audition app\n"
                "  committed an edit. Writing now would erase it. Nothing was modified;\n"
                "  re-run (this only fills blank cells, so it is safe to repeat)."
            )
        stamp = datetime.datetime.now().strftime("%Y%m%d")
        backup = path.with_name(f"{path.name}.bak-{stamp}-seeddelivery")
        if not backup.exists():
            shutil.copy2(path, backup)
        # Write-and-rename so a reader never observes a half-written file, matching
        # what the app itself does.
        tmp = path.with_suffix(".csv.seedtmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)

    verb = "seeded" if args.apply else "would seed"
    print(f"{verb}: " + ", ".join(f"{v} {k}" for k, v in sorted(seeded.items())) + f"  (total {sum(seeded.values())})")
    print(f"preserved {kept} owner-set value(s)")
    for k, v in sorted(left.items()):
        print(f"left blank for the ear: {v:4d}  {k}")
    if args.apply:
        print(f"backup: {backup.name}")


if __name__ == "__main__":
    main()
