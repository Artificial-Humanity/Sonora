"""Reconcile `direction.design` (and `intended_gender`) with the reference that was cast.

WHAT WENT WRONG. The old narration builder cast per line and then pinned one reference
per (book, engine). Those are two different decisions and they were allowed to disagree:
the pin took the FIRST line's voice while `design` kept each line's own casting intent.
On 10 clips the two point at opposite genders — `"middle-aged woman, steady and measured"`
against a male reference.

WHY THE AUDIO IS FINE ANYWAY, and why this is a correction rather than a falsification.
For chatterbox and zonos `voice_design` is a CASTING string, not a prompt — book_ingest
says so at the RELAY table: *"it never reaches the model, it selects the reference clip."*
Neither engine has a prose channel (`ENGINE_CHANNELS`: chatterbox = coarse+reference,
zonos = numeric+reference), so this text was never transmitted to anything. The voice came
from the pinned reference, which is why the ear heard the right gender on all 10 and
confirmed every pre-filled dropdown was correct.

So `design` is not a record of what we sent. It is a DESCRIPTION of the cast voice that
disagrees with the voice actually cast. Rewriting the gender word to match the reference
makes the description true; leaving it would keep a wrong label in the manifests that any
later analysis of direction-vs-outcome would read as real.

The reference is the authority here, not the ear — the ear confirms it, but the reference
is what physically produced the audio. This refuses to touch any line where the two
disagree, because that would be a different and much more interesting problem.

The builder no longer produces this: casting, pinning and design are now one pass that
freezes a single design per (book, engine) from the group's MAJORITY gender intent.

    .venv/bin/python scripts/synthesis/fix_design_gender.py
    .venv/bin/python scripts/synthesis/fix_design_gender.py --apply
"""
import argparse
import csv
import glob
import json
import os
import pathlib
import re
import shutil
import tempfile

CAMPAIGN = pathlib.Path("/data/model-training/datasets/delivery-v1-narration-r2")
POOL = pathlib.Path("/data/model-training/datasets/reference-pool-v2/metadata.jsonl")
RATINGS = pathlib.Path(
    "/data/model-training/datasets/sonora-expressive-registers/ratings.csv")

# Gender-bearing nouns/pronouns that may appear in a casting string, and their
# counterpart. Ordered longest-first so "gentleman" is not matched by "man".
SWAP = [("gentleman", "lady"), ("woman", "man"), ("lady", "gentleman"),
        ("girl", "boy"), ("boy", "girl"), ("man", "woman"),
        ("female", "male"), ("male", "female"),
        ("she", "he"), ("her", "his"), ("his", "her"), ("he", "she")]
FEM = re.compile(r"\b(woman|female|she|her|lady|girl)\b", re.I)
MAS = re.compile(r"\b(man|male|he|his|him|gentleman|boy)\b", re.I)


def design_gender(d):
    """Female/Male if the string is unambiguous, else None (leave it alone)."""
    if not d:
        return None
    f, m = bool(FEM.search(d)), bool(MAS.search(d))
    return "Female" if f and not m else "Male" if m and not f else None


def reword(design, target):
    """Swap the gender word(s) toward `target`, preserving everything else."""
    out = design
    for a, b in SWAP:
        src, dst = (a, b) if target == "Male" else (b, a)
        if design_gender(src) == ("Female" if target == "Male" else "Male"):
            out = re.sub(rf"\b{src}\b", dst, out, flags=re.IGNORECASE)
    return out


def load_pool():
    pool = {}
    for line in open(POOL, encoding="utf-8"):
        j = json.loads(line)
        pool[j.get("id") or j.get("ref_id")] = j
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    pool = load_pool()
    ear = {r["id"]: r["gender"] for r in csv.DictReader(open(RATINGS, encoding="utf-8"))
           if r["campaign"] == CAMPAIGN.name and r["gender"]}

    targets, conflicts = {}, []
    for bank in sorted(CAMPAIGN.glob("bank*.json")):
        for line in json.load(open(bank, encoding="utf-8"))["lines"]:
            cid, rid = line["id"], line.get("ref_id")
            ref_g = (pool.get(rid) or {}).get("gender")
            cur = design_gender((line.get("direction") or {}).get("design"))
            if not ref_g or not cur or cur == ref_g:
                continue
            # The ear is the cross-check, not the authority. If it ever disagreed
            # with the reference, the story would not be a stale label and a blind
            # rewrite would bury it.
            if cid in ear and ear[cid] != ref_g:
                conflicts.append((cid, cur, ref_g, ear[cid]))
                continue
            targets[cid] = ref_g

    print(f"{len(targets)} clip(s) to correct, {len(conflicts)} conflict(s)\n")
    for cid, cur, ref_g, e in conflicts:
        print(f"  CONFLICT {cid[:38]:40s} design={cur} ref={ref_g} ear={e} — SKIPPED")

    edits = 0
    files = sorted(CAMPAIGN.glob("bank*.json"))
    jsonls = sorted(glob.glob(str(CAMPAIGN / "audio" / "*_manifest.jsonl"))) + \
        [str(CAMPAIGN / "qc_measures.jsonl")]

    def patch(rec):
        nonlocal edits
        cid = rec.get("id")
        if cid not in targets:
            return False
        tgt, changed = targets[cid], False
        d = rec.get("direction") or {}
        if d.get("design") and design_gender(d["design"]) != tgt:
            was = d["design"]
            d["design"] = reword(was, tgt)
            rec["direction"] = d
            rec.setdefault("corrections", {})["design"] = {
                "was": was, "reason": "disagreed with the pinned reference's gender"}
            changed = True
        if rec.get("intended_gender") and rec["intended_gender"] != tgt:
            rec.setdefault("corrections", {})["intended_gender"] = {
                "was": rec["intended_gender"],
                "reason": "disagreed with the pinned reference's gender"}
            rec["intended_gender"] = tgt
            changed = True
        edits += changed
        return changed

    plan = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        n = sum(patch(l) for l in d["lines"])
        if n:
            plan.append((f, d, n, "json"))
    for f in jsonls:
        if not os.path.isfile(f):
            continue
        recs = [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
        n = sum(patch(r) for r in recs)
        if n:
            plan.append((pathlib.Path(f), recs, n, "jsonl"))

    for f, _, n, _ in plan:
        print(f"  {n:3d} record(s)  {f.name}")
    for cid, tgt in sorted(targets.items()):
        print(f"    -> {cid[:40]:42s} design gender := {tgt}")

    if not args.apply:
        print("\nreport only — pass --apply")
        return 0

    for f, payload, _, kind in plan:
        fd, tmp = tempfile.mkstemp(dir=str(f.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if kind == "json":
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            else:
                for r in payload:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        shutil.copymode(str(f), tmp)
        os.replace(tmp, str(f))
    print(f"\ncorrected {edits} record(s) across {len(plan)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
