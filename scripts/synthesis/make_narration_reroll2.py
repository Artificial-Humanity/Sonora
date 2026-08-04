"""Reroll round 2 — remedies driven by what the EAR said, not by a blanket retry.

The audition of delivery-v1-narration-r2 (2026-08-03) left 7 non-keeps, and they
are four different defects. Rerolling them identically would have re-rolled the
same dice on three of them.

  chatterbox x3  "too rapid of a rate" / "a bit too hurried of a pace"
                 -> exaggeration 0.35. chatterbox.md prescribes 0.5 for Neutral and
                    warns "do NOT drift lower; low exag reads subdued/ironic", but
                    that was marked provisional and rested on a DIALOGUE record.
                    This is the first narration audition and it says 0.5 is too
                    fast. 0.35 deliberately tests the warning: if these come back
                    subdued, the floor is real and sits between 0.35 and 0.5.
  chatterbox x1  "references to illustrations not visible here"
                 -> DROPPED, not rerolled. The passage narrates figures that do not
                    exist in audio; no engine or setting fixes that. Now caught at
                    mint by book_ingest.references_the_invisible.
  orpheus    x1  '"Madison, Wis." was pronounced exactly how it is spelled'
                 -> text expanded. ARCHITECTURE §1: digits/abbreviations are the
                    CALLER's job. Not a model limitation — the engine read exactly
                    what we gave it.
  zonos      x1  '"He cut off" came out very jumbled'
                 -> reseed. Known stochastic defect, and the standing rule for those
                    is reroll, don't re-cast.

    .venv/bin/python scripts/synthesis/make_narration_reroll2.py [--apply]
"""
import argparse
import csv
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import book_ingest as bi  # noqa: E402

CAMPAIGN = pathlib.Path("/data/model-training/datasets/delivery-v1-narration-r2")
AUDIO = CAMPAIGN / "audio"
SUPERSEDED = AUDIO / "_superseded"
RATINGS = pathlib.Path(
    "/data/model-training/datasets/sonora-expressive-registers/ratings.csv")

NARRATION_EXAGGERATION = 0.35
SEED_BUMP = 7000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    bank = json.load(open(CAMPAIGN / "bank.json"))
    by_id = {l["id"]: l for l in bank["lines"]}
    verdicts = {r["id"]: r for r in csv.DictReader(open(RATINGS))
                if r["campaign"] == CAMPAIGN.name}

    # A line that was RE-CAST in round 1 still has its old row in ratings.csv, and
    # the old wav is in _superseded, so the app served nothing and the ear wrote
    # "Play will not start." That is my bookkeeping, not a defect in the clip — the
    # replacement rendered under a new id (the id carries the engine) and passed.
    # Rerolling the retired id would put two clips on one passage.
    # Read the DIRECTORY, not the bank: a re-cast line is written to a reroll bank,
    # so bank.json never learns about it and a bank-only check silently finds
    # nothing. The filesystem is the one place that knows what actually exists.
    on_disk_stems = {p.stem.rsplit("_", 1)[0] for p in AUDIO.glob("*.wav")}

    todo = [(i, r) for i, r in verdicts.items()
            if r["status"] in ("reroll", "dropped") and i in by_id]
    lines, dropped, phantom, notes = [], [], [], []
    for cid, r in sorted(todo):
        src = by_id[cid]
        engine = src["engine"]

        stem = cid.rsplit("_", 1)[0]
        if not (AUDIO / f"{cid}.wav").is_file() and stem in on_disk_stems:
            phantom.append((cid, "re-cast in round 1; its replacement already passed"))
            continue

        if bi.references_the_invisible(src["text"]):
            dropped.append((cid, "narrates figures/plates that do not exist in audio"))
            continue

        line = dict(src, seed=src["seed"] + SEED_BUMP)
        why = []

        text = bi.normalize_speakable(src["text"])
        if text != src["text"]:
            line["text"] = text
            # `text` is NOT what orpheus and dia are handed — they render from
            # `direction.render_text`. Rewriting only `text` is what made the first
            # attempt at this fix fail silently: the bank, the manifest and the WER
            # reference all said "Wisconsin" while the model was still given "Wis."
            # and duly said it. Same transformation, both copies; the abbreviation
            # patterns are word-anchored so engine markup passes through untouched.
            d = dict(line["direction"])
            if d.get("render_text"):
                d["render_text"] = bi.normalize_speakable(d["render_text"])
                line["direction"] = d
            why.append("expanded abbreviations")

        if engine == "chatterbox":
            d = dict(line["direction"])
            was = d.get("exaggeration")
            d["exaggeration"] = NARRATION_EXAGGERATION
            line["direction"] = d
            why.append(f"exaggeration {was} -> {NARRATION_EXAGGERATION}")

        if not why:
            why.append("reseed only (stochastic defect)")
        line["reroll_of"] = {"round": 2, "status": r["status"],
                             "note": r["note"][:200], "remedy": "; ".join(why)}
        lines.append(line)
        notes.append((cid, engine, "; ".join(why), r["note"][:70]))

    print(f"{len(todo)} ear non-keeps -> {len(lines)} reroll(s), "
          f"{len(dropped)} dropped, {len(phantom)} phantom\n")
    for cid, engine, why, note in notes:
        print(f"  {cid[:40]:42s} {engine:11s} {why}")
        print(f"{'':44s} ear: {note}")
    for cid, why in dropped:
        print(f"  DROP    {cid[:40]:42s} {why}")
    for cid, why in phantom:
        print(f"  PHANTOM {cid[:40]:42s} {why}")

    if not args.apply:
        print("\nreport only — pass --apply")
        return 0

    SUPERSEDED.mkdir(parents=True, exist_ok=True)
    moved = 0
    for line in lines:
        wav = AUDIO / f"{line['id']}.wav"
        if wav.is_file():
            shutil.move(str(wav), str(SUPERSEDED / wav.name))
            moved += 1
    print(f"\nsuperseded {moved} wav(s)")

    out = CAMPAIGN / "bank-reroll2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"version": f"{bank['version']}-reroll2",
                   "campaign": bank["campaign"],
                   "note": ("Round-2 reroll, remedies per the ear's own diagnosis: "
                            "chatterbox pace -> exaggeration 0.35 (testing "
                            "chatterbox.md's warning against the first narration "
                            "audition), orpheus -> abbreviations expanded, zonos -> "
                            "reseed. One passage dropped: it narrates figures that "
                            "do not exist in audio."),
                   "license_note": bank.get("license_note", ""),
                   "lines": lines}, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(lines)} lines -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
