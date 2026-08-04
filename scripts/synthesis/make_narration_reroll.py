"""Reroll the zonos QC failures at shorter text — the length hypothesis, tested.

delivery-v1-narration-r2 measured, 2026-08-03:

    text length   zonos        other four engines
    <150 chars     6/6  100%    13/13  100%
    150-225       11/12  92%    34/35   97%
    225+          15/20  75%    54/55   98%

All 38 zonos lines carried `emotion: null`, so this is NOT the 2026-07-30
conditioning defect returning — that one is fixed and stayed fixed. Length is a
separate axis the emotion work never touched, and the skill file's only length
guard (`phonemes / rate < 30 s`) computes to ~18 s for a 300-character line and
never fires. Mean text length was near-identical between zonos and the others
(232 vs 223), so allocation is not confounding it, and within the 225+ bucket
every pinned reference degrades — it is not a bad-reference story either.

This re-windows each failed line to the longest COMPLETE-SENTENCE prefix under
the ceiling and re-renders it on the same engine, same pinned voice, new seed.
Same text, same book, same position — only the window shortens, so a pass here
is evidence about length rather than about a different clip.

Ids are PRESERVED and the old wavs move to `audio/_superseded/`: ratings.csv
keys on id, and qc_gate already knows to measure a superseded clip for the
record without letting it re-enter keeps.

    .venv/bin/python scripts/synthesis/make_narration_reroll.py            # report
    .venv/bin/python scripts/synthesis/make_narration_reroll.py --apply
"""
import argparse
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import book_ingest as bi  # noqa: E402

CAMPAIGN = pathlib.Path("/data/model-training/datasets/delivery-v1-narration-r2")
AUDIO = CAMPAIGN / "audio"
SUPERSEDED = AUDIO / "_superseded"

# The ceiling. 225 is where the measured zonos pass rate falls off (92% below,
# 75% above); the boundary is soft — the longest passing clip was 352 chars and
# the shortest failure 184 — so this is a risk budget, not a cliff.
MAX_CHARS = 225
SEED_BUMP = 3000
# Where a line goes when it is too long for zonos and has no cut point. qwen is
# the richest instruct slot and measured 37/38 here, flat across every length
# bucket — the property zonos lacks.
RECAST_ENGINE = "qwen"


def shorten(text):
    """Longest complete-sentence prefix under the ceiling.

    Completeness is the gate, not length (owner 2026-07-28): a short but WHOLE
    utterance is ratable, an incomplete one is not, because prosody lives in the
    arc of a finished utterance. So this drops whole sentences off the end rather
    than truncating, and refuses to return a fragment.
    """
    sents = bi.split_sentences(text)
    out = []
    for s in sents:
        cand = " ".join(out + [s])
        if out and len(cand) > MAX_CHARS:
            break
        out.append(s)
    kept = " ".join(out).strip()
    if not kept or not bi.is_complete_utterance(kept):
        return None
    if len(kept) < bi.MIN_CLIP_CHARS:      # 4 s owner floor, gates input text
        return None
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    measures = [json.loads(l) for l in open(CAMPAIGN / "qc_measures.jsonl")]
    failed = [m for m in measures if not m["hard_pass"] and m["engine"] == "zonos"]
    bank = json.load(open(CAMPAIGN / "bank.json"))
    by_id = {l["id"]: l for l in bank["lines"]}

    print(f"{len(failed)} zonos hard failures to reroll (ceiling {MAX_CHARS} chars)\n")
    lines, recast = [], []
    for m in sorted(failed, key=lambda m: m["id"]):
        src = by_id[m["id"]]
        was = (f"was pause={m['worst_pause']:.1f} wer={m['asr_wer']:.2f}")
        n = len(src["text"])
        prior = {"chars": n, "worst_pause": m["worst_pause"],
                 "asr_wer": round(m["asr_wer"], 3),
                 "failed_gates": [k for k, v in m["gates"].items() if not v]}

        # Already inside the budget: length is not the story, so shortening would
        # not be the remedy — and "reroll, don't re-cast" is the standing rule for
        # a known stochastic defect. New seed, same everything else.
        if n <= MAX_CHARS:
            line = dict(src, seed=src["seed"] + SEED_BUMP, reroll_of=prior)
            lines.append(line)
            print(f"  {m['id'][:42]:44s} {n:4d} chars, already under the ceiling "
                  f"-> reseed only   {was}")
            continue

        short = shorten(src["text"])
        if short is not None and len(short) < n:
            line = dict(src, text=short, seed=src["seed"] + SEED_BUMP, reroll_of=prior)
            lines.append(line)
            print(f"  {m['id'][:42]:44s} {n:4d} -> {len(short):3d} chars   {was}")
            continue

        # One long sentence with nowhere to cut — Darwin's specialty. Shortening
        # cannot help and re-rendering on zonos would just re-roll a 75% bet, so
        # this RE-CASTS to the engine the same measurement says is flat across
        # length (the other four ran 54/55 above 225 chars). New id, because the
        # id carries the engine.
        recast.append((m, src, prior))
        print(f"  {m['id'][:42]:44s} {n:4d} chars, one sentence — no cut point "
              f"-> re-cast to {RECAST_ENGINE}   {was}")

    for m, src, prior in recast:
        labels = {k: src["intended"][k] for k in ("V", "A", "T")}
        labels["register"] = src.get("register", "")
        cast = bi.casting_pass(src["text"], RECAST_ENGINE, labels=labels)
        if cast is None:
            print(f"  !! {src['id']}: casting_pass failed for {RECAST_ENGINE}")
            continue
        tag = {"engine": RECAST_ENGINE, **labels, **cast}
        eng, direction = bi.build_direction(tag, src["text"],
                                            lane=src["intended_delivery"])
        new_id = src["id"].rsplit("_", 1)[0] + f"_{eng[:3].upper()}"
        lines.append(dict(src, id=new_id, engine=eng, direction=direction,
                          ref_id=None, seed=src["seed"] + SEED_BUMP,
                          reroll_of=dict(prior, recast_from=src["id"])))
        print(f"     -> {new_id}")

    if not args.apply:
        print("\nreport only — pass --apply to write the bank and supersede the old wavs")
        return 0

    SUPERSEDED.mkdir(parents=True, exist_ok=True)
    moved = 0
    for line in lines:
        # A RE-CAST line carries its NEW id (the id encodes the engine), so the
        # wav to retire is the original's, not this line's. Superseding by
        # `line['id']` silently left the failed zonos clip live on disk while its
        # replacement rendered under a different name — two clips for one slot,
        # the failed one still eligible to register.
        old_id = line.get("reroll_of", {}).get("recast_from", line["id"])
        wav = AUDIO / f"{old_id}.wav"
        if wav.is_file():
            shutil.move(str(wav), str(SUPERSEDED / wav.name))
            moved += 1
    print(f"\nmoved {moved} superseded wav(s) -> {SUPERSEDED}")

    out = CAMPAIGN / "bank-reroll.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"version": f"{bank['version']}-reroll1",
                   "campaign": bank["campaign"],
                   "note": (f"Reroll of {len(lines)} zonos QC failures at <= {MAX_CHARS} "
                            "chars, testing the measured length effect (zonos 75% over "
                            "225 chars vs 98% for the other four engines; all lines had "
                            "emotion:null, so this is not the conditioning defect). Same "
                            "engine, same pinned voice, same book position, new seed."),
                   "license_note": bank.get("license_note", ""),
                   "lines": lines}, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(lines)} lines -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
