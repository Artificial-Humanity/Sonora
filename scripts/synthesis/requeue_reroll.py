"""Put re-rendered clips back in front of the ear, and retire ids that no longer exist.

A reroll PRESERVES the clip id (ratings.csv keys on id, and the audit trail should
follow the passage, not the take). The consequence is that after re-rendering, the
row still carries round-1's verdict — `status=reroll`, the old score, the old note —
so the audition app does not show it and register_audition skips it as "already
present". The new audio exists and nobody is ever asked to listen to it.

This resets those rows to `unaudited` so they land back in `todo`.

THE MISSING-WAV GUARD is the point of this script, not a nicety. Round 1 re-cast a
zonos line to qwen; the id carries the engine, so the replacement rendered under a
NEW id while the old row stayed live pointing at a wav that had been moved to
`_superseded/`. The app served a dead link and the ear wrote "Play will not start."
Requeueing on trust would reproduce that exactly. So an id is only requeued if its
wav is on disk, and `--retire` is the explicit way to close out the other case.

    .venv/bin/python scripts/synthesis/requeue_reroll.py --bank <bank.json>
    .venv/bin/python scripts/synthesis/requeue_reroll.py --bank <bank.json> --apply
    .venv/bin/python scripts/synthesis/requeue_reroll.py --retire ID --superseded-by ID --apply
"""
import argparse
import csv
import json
import os
import pathlib
import shutil
import tempfile

RATINGS = pathlib.Path(
    "/data/model-training/datasets/sonora-expressive-registers/ratings.csv")

# Deliberately says nothing about WHICH WAY the clip was changed. The ear's round-1
# note ("too rapid") is preserved in the bank's `reroll_of.note`, so the audit trail
# is intact without priming the listener toward the answer we are hoping for. The
# open question here — does chatterbox at exaggeration 0.35 read *subdued*? — is
# precisely the one a note saying "now slower" would suppress.
REQUEUE_NOTE = "[re-rendered; judge fresh]"


def load_rows():
    with RATINGS.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames or [], list(rd)


def write_rows(hdr, rows, before):
    if RATINGS.stat().st_mtime_ns != before:
        print("ABORT: ratings.csv changed under us (live writer)")
        return 1
    fd, tmp = tempfile.mkstemp(dir=str(RATINGS.parent), suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    if RATINGS.stat().st_mtime_ns != before:
        os.unlink(tmp)
        print("ABORT: ratings.csv changed during write")
        return 1
    shutil.copymode(str(RATINGS), tmp)
    os.replace(tmp, str(RATINGS))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", help="bank json whose lines were re-rendered")
    ap.add_argument("--retire", action="append", default=[],
                    help="id whose clip no longer exists under that id")
    ap.add_argument("--superseded-by", action="append", default=[],
                    help="the id that replaced it (parallel to --retire)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if len(args.retire) != len(args.superseded_by):
        ap.error("--retire and --superseded-by must be given in pairs")

    before = RATINGS.stat().st_mtime_ns
    hdr, rows = load_rows()
    by_id = {r["id"]: r for r in rows}

    requeue, blocked = [], []
    if args.bank:
        bank = json.load(open(args.bank))
        audio = pathlib.Path(args.bank).parent / "audio"
        for line in bank["lines"]:
            cid = line["id"]
            row = by_id.get(cid)
            if row is None:
                blocked.append((cid, "no ratings row — register_audition never saw it"))
                continue
            if not (audio / f"{cid}.wav").is_file():
                blocked.append((cid, "wav missing on disk — would serve a dead link"))
                continue
            requeue.append((cid, row, line.get("reroll_of", {}).get("remedy", "")))

    retire = []
    for cid, repl in zip(args.retire, args.superseded_by):
        row = by_id.get(cid)
        if row is None:
            blocked.append((cid, "no ratings row to retire"))
        elif repl not in by_id:
            blocked.append((cid, f"replacement {repl} is not in ratings.csv"))
        else:
            retire.append((cid, row, repl, by_id[repl]["status"]))

    print(f"requeue {len(requeue)}, retire {len(retire)}, blocked {len(blocked)}\n")
    for cid, row, remedy in requeue:
        print(f"  REQUEUE {cid[:40]:42s} {row['status']}/{row['score'] or '-'} "
              f"-> unaudited   ({remedy})")
    for cid, row, repl, rstat in retire:
        print(f"  RETIRE  {cid[:40]:42s} {row['status']} -> dropped "
              f"(superseded by {repl}, which is a {rstat})")
    for cid, why in blocked:
        print(f"  BLOCKED {cid[:40]:42s} {why}")

    if not args.apply:
        print("\nreport only — pass --apply")
        return 0

    for cid, row, _ in requeue:
        row["status"], row["score"], row["note"] = "unaudited", "", REQUEUE_NOTE
    for cid, row, repl, _ in retire:
        # `dropped` rather than a new status word: it is terminal, so
        # register_audition will not re-queue this id, and the app already knows
        # it. The note carries the fact that this is a bookkeeping retirement and
        # not an ear verdict on the audio.
        row["status"], row["score"] = "dropped", "x"
        row["note"] = (f"Retired, not judged: re-cast in round 1 and re-rendered as "
                       f"{repl}. The wav under this id was superseded before the ear "
                       f"heard it, which is why it would not play.")

    rc = write_rows(hdr, rows, before)
    if rc == 0:
        print(f"\nwrote {len(requeue)} requeued + {len(retire)} retired -> {RATINGS}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
