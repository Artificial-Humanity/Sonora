"""queue_head_audit — put late-starting clips in front of the ear, so head_ok can exist.

C-M4 left `head_ok` measured and ungated on purpose: mirroring `TAIL_LOST_MAX` is a guess,
ASR is measurably worse at the first word than the last, and there is nothing to calibrate
AGAINST — searching every drop note in `ratings.csv` for a phrase naming a late start
returns nothing, because no auditor has ever been pointed at one.

That is the deadlock this script breaks. The threshold needs ear labels; the ear labels
need a queue; the queue was gated on the threshold. **A threshold is only needed to
REJECT a clip. Queueing one costs a listen.**

WHAT IT QUEUES, and why it is a short list rather than all 214 clips that lose a word:
clips that drop >= `HEAD_WORDS_FLAG` opening words **and passed every gate**. A clip that
already failed a gate is going to the ear anyway and its drop note will say why; the
interesting population is the one the instrument called clean. Measured 2026-08-07 over
3,189 clips in 14 campaigns: 19 clips drop >= 3 opening words and **8 of those passed
every gate**, up to 25% of the passage.

Rows that predate the head measure — which is all of them, it shipped 2026-08-07 — are
recomputed from `text`/`asr_hyp`, the same fallback `gate_calibration.py` uses.

A VERDICT IS NEVER DISCARDED. Four of the eight already carry an ear verdict, two of them
a `keep`. Status goes to `unaudited` so the clip lands in the app's todo queue, and the
existing score and note are preserved in place — the auditor is being asked a question
nobody asked before ("does it start late?"), not told their answer was wrong.

Run:
  .venv/bin/python scripts/tools/queue_head_audit.py            # report only
  .venv/bin/python scripts/tools/queue_head_audit.py --apply
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

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

DATA_ROOT = Path(os.environ.get(
    "AUDITION_DATA_ROOT", "/data/model-training/datasets")).resolve()
RATINGS = Path(os.environ.get(
    "AUDITION_RATINGS_DIR", str(DATA_ROOT / "sonora-expressive-registers"))) / "ratings.csv"

# The dedup key is a STABLE prefix, not the formatted note. Keying on the whole sentence
# re-queued a clip whenever a measure changed underneath it: the 2026-08-08 hyphen fix
# moved the reference token count, so the percentage in an already-written note no longer
# matched the one being generated, and a second `--apply` appended the line again.
#
# QC-M6: the wording itself now comes from `synth_common.head_note`, shared with
# `register_audition`. This file kept writing "first N words … never spoken" for a day
# after `683c43f` retired that sentence for asserting a conclusion the evidence
# contradicts — the rewording reached one of the instrument's two writers. The old mark
# stays in the dedup set so any sheet that DOES carry it is not queued a second time; no
# row on the live sheet does, checked 2026-08-09.
NOTE_MARK = synth_common.HEAD_NOTE_MARK
LEGACY_NOTE_MARKS = ("QC: starts late",)
NOTE_TAIL = (" This clip PASSED every gate; head loss is measured but not gated, and "
             "your note is what sets the threshold (C-M4).")


def candidates():
    """-> [(campaign, row)] for gate-passing clips that drop >= HEAD_WORDS_FLAG words."""
    out = []
    for path in sorted(glob.glob(str(DATA_ROOT / "**" / "qc_measures.jsonl"),
                                 recursive=True)):
        campaign = os.path.relpath(os.path.dirname(path), DATA_ROOT)
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("id") or not synth_common.head_flagged(row):
                continue
            # The population of interest is what the instrument called CLEAN. A clip that
            # failed a gate already has an ear pass coming and a note that names why.
            if not row.get("hard_pass"):
                continue
            if row.get("head_words_lost") is None and row.get("text") and row.get("asr_hyp"):
                edges = synth_common.edge_loss(row["text"], row["asr_hyp"])
                row["head_words_lost"] = edges["head_words"]
                row["head_lost_frac"] = edges["head_frac"]
            out.append((campaign, row))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true",
                    help="write ratings.csv (default is report-only)")
    args = ap.parse_args()

    found = candidates()
    by_id = {r["id"]: (c, r) for c, r in found}
    print(f"{len(found)} clip(s) drop >= {synth_common.HEAD_WORDS_FLAG} opening words "
          f"AND passed every gate\n")

    queued, pooled = [], []

    def mutate(rows):
        seen = set()
        changed = 0
        for row in rows:
            if row["id"] not in by_id:
                continue
            seen.add(row["id"])
            campaign, measures = by_id[row["id"]]
            words = measures.get("head_words_lost") or 0
            note = synth_common.head_note(
                words, measures.get("head_lost_frac") or 0) + NOTE_TAIL
            was = row.get("status") or ""
            # Preserve the verdict. `score` is untouched, and an existing note is kept
            # ahead of ours — the auditor's own words outrank a generated line.
            existing = (row.get("note") or "").strip()
            if any(m in existing for m in (NOTE_MARK, *LEGACY_NOTE_MARKS)):
                continue                     # already queued by an earlier run
            row["note"] = f"{existing} | {note}".strip(" |") if existing else note
            row["status"] = "unaudited"
            changed += 1
            queued.append((campaign, row["id"], was, row.get("score") or "-", words))
        for clip_id, (campaign, measures) in sorted(by_id.items()):
            if clip_id not in seen:
                pooled.append((campaign, clip_id, measures.get("head_words_lost") or 0))
        return changed

    with synth_common.ratings_transaction(RATINGS, tag="head-audit",
                                          dry_run=not args.apply) as (_hdr, rows):
        n = mutate(rows)

    if queued:
        verb = "queued" if args.apply else "WOULD queue"
        print(f"{verb} {len(queued)} clip(s) already in ratings.csv:")
        for campaign, clip_id, was, score, words in sorted(queued):
            print(f"  {campaign:26s} {clip_id:36s} {was:10s} -> unaudited  "
                  f"(score {score}, {words}w lost)")
    if pooled:
        print(f"\n{len(pooled)} clip(s) have NO ratings.csv row — they are still in the "
              f"POOL, not the corpus.\nNothing to queue: `stage_pool.qc_flagged` now "
              f"treats a late start as owed-an-ear, so they\nenter unaudited when the "
              f"pool is staged rather than folding as silent keeps.")
        for campaign, clip_id, words in sorted(pooled):
            print(f"  {campaign:26s} {clip_id:36s} {words}w lost")
    if not args.apply:
        print(f"\n(report only — {n} row(s) would change; re-run with --apply)")


if __name__ == "__main__":
    main()
