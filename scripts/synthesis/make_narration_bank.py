"""Build delivery-v1-narration round 2 — the last two open lanes.

Closes Neutral (+81) and Documentary (+35), the only gaps left in the ratified
50/30/8/6/6 mix (delivery-mix-campaign.md). Text comes from the five books
ingested 2026-08-02; labels (V/A/T + register) are inherited from their book banks,
which pass 1 emits engine-agnostically on purpose.

SINCE 2026-08-11 THE TWO LANES ARE ONE. `Documentary` was retired into `Neutral`
(matcha.delivery.RETIRED_LANES) and the +35 it was sized for is now part of Neutral's
gap, so this round's lane map and targets are Neutral-only — see LANES and TARGETS.

WHAT MAKES THIS ROUND DIFFERENT, and why it is worth stating: the book banks route
220 of 220 narration lines to zonos, because pass 1 lets the director pick an engine
per line and it has a strong opinion about narration. Rendering that as-is would
make both lanes a single engine's timbre. Allocation here comes from
`ref_select.ENGINE_MIX_BY_LANE` instead — measured per lane, floored so no engine
starves, capped so no lane collapses to one voice — and pass 2 is re-run for
whichever engine each line actually lands on. A qwen line must be cast in qwen's
language, not handed zonos's numbers.

This is also the end-to-end test the zonos tier decision was waiting on. Every
prior zonos narration bank had `emotion: null` hand-patched after the fact; here it
has to come out of `casting_pass` -> `_l1(None)` -> `build_direction` on its own.

    .venv/bin/python scripts/synthesis/make_narration_bank.py            # report
    .venv/bin/python scripts/synthesis/make_narration_bank.py --apply

Then render with `synth_bank.sh` (which runs the mandatory QC gate), and queue with
`pick_audit_subset.py`.
"""
import argparse
import collections
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import book_ingest as bi  # noqa: E402
import ref_select  # noqa: E402

BOOKS = pathlib.Path("/data/model-training/datasets/book-prose")
CAMPAIGN = "delivery-v1-narration-r2"
OUT_DEFAULT = pathlib.Path("/data/model-training/datasets") / CAMPAIGN

# Lane per book. EVERY BOOK IS NEUTRAL SINCE 2026-08-11 (issue #12).
# `voyage-of-the-beagle` was this round's one Documentary pick, on the reading that
# Documentary is a RENDER style over narration text so the text only has to INVITE it —
# third-person natural history does, first-person recollection does not (owner,
# 2026-08-01: autobiography is "more narrative", i.e. Neutral). That reading did not
# survive the ear: Documentary was RETIRED into Neutral (owner, 2026-08-10), and every one
# of the 154 Documentary rows in ratings.csv was relabelled Neutral, not Newscaster. So
# this book's lane follows the rows, and the invitation the prose extends is one no label
# now records.
LANES = {
    "voyage-of-the-beagle":    "Neutral",       # was Documentary until the retirement
    "conan-stories":           "Neutral",
    "up-from-slavery":         "Neutral",       # autobiography
    "franklin-autobiography":  "Neutral",       # autobiography
    "walden":                  "Neutral",       # first-person recollection
}

# Renders PER BOOK, sized from the measured gaps at ~90% keep with a little headroom.
#
# The retired lane's target FOLDED IN rather than vanishing: the round wanted 45 lines for
# the Documentary gap (+35) and 96 for the Neutral one (+81), and retiring the lane changed
# where those clips are LABELLED, not whether the corpus wants them. The whole round is
# already rendered and audited (delivery-v1-narration-r2 closed 2026-08-04), so these
# numbers exist to REPRODUCE it, not to size a new campaign.
#
# PER BOOK RATHER THAN PER LANE, because that is how the target is consumed (issue #44).
# The selection below divides the lane total across the lane's books — `each = want //
# len(books)` — and `len(books)` went from 1 (Documentary) and 4 (Neutral) to 5 when
# `voyage-of-the-beagle` moved into Neutral. A single `{"Neutral": 141}` therefore did NOT
# reproduce the round, in two ways, both silent:
#   * 141 // 5 == 28 and 28 * 5 == 140 — the floor division loses a line the two-lane
#     split did not lose, so the comment's own headline number was never reached;
#   * every book's share changed. beagle 45 -> 28 and the other four 24 -> 28, which is a
#     materially different corpus from the one the ear actually closed.
# Verified against the shipped bank: 141 lines, beagle 45, conan/up-from-slavery/
# franklin/walden 24 each. These numbers ARE that split.
#
# ⚠ THE CASTING CANNOT BE REPRODUCED, and no arithmetic here can fix that. Engines are
# allocated per LANE (`allocate_engines(n, lane=lane)`), and the closed round dealt two
# lanes under two different measured mixes — Documentary's 45 and Neutral's 96, giving
# e.g. orpheus 4 + 14 and moss_vg 2 + 10. A re-run deals ONE lane of 141 under Neutral's
# mix alone, so the engine split, and with it every per-(book, engine) frozen voice, comes
# out different. `ENGINE_MIX_BY_LANE` carries no Documentary mix any more and
# `ref_select.mix_for_lane` refuses the lane outright, so the old deal is not recoverable
# even in principle. What re-runs reproduces is the TEXT SELECTION; the audio it casts is
# a new draw. The rendered round on disk stays the artifact of record.
PER_BOOK = {
    "voyage-of-the-beagle":    45,      # the round's Documentary target, now Neutral
    "conan-stories":           24,
    "up-from-slavery":         24,
    "franklin-autobiography":  24,
    "walden":                  24,
}

# Lane totals, DERIVED from the per-book split rather than restated beside it — a second
# number that has to agree with a first is how the 141/140 disagreement got in. Only
# `allocate_engines` reads this, and it is handed the count actually selected.
TARGETS = {lane: sum(n for slug, n in PER_BOOK.items() if LANES[slug] == lane)
           for lane in dict.fromkeys(LANES.values())}

# References must be narration-shaped. A dialogue reference drags the read toward
# performance, which is the opposite of what these lanes want.
REF_REGISTERS = {"neutral_narration", "narrative", "formal_narrative"}

SEED_BASE = 8200


def narration_lines(slug):
    path = BOOKS / slug / f"{slug}_bank.json"
    bank = json.load(open(path, encoding="utf-8"))
    return [l for l in bank["lines"] if l.get("chunk_type") == "narration"]


def spread(items, n):
    """Even spread across the book rather than the first n — a book's opening pages
    are not representative of its prose, and the ingest already sampled evenly."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, help="cap lines per book (smoke tests)")
    args = ap.parse_args()

    # --- pick the text -------------------------------------------------------
    per_lane = collections.defaultdict(list)
    for slug, lane in LANES.items():
        lines = narration_lines(slug)
        for l in lines:
            l["_book"] = slug
        per_lane[lane].append((slug, lines))

    selected = []
    for lane, books in per_lane.items():
        # Sized per book (PER_BOOK), not by dividing a lane total — see the table's
        # comment for why the division did not reproduce the closed round.
        # INTERLEAVE the books round-robin before the engines are dealt over this
        # list. Concatenating them instead put each book in one contiguous block,
        # and since the engine queue is also blocked by engine, every book came out
        # as a SINGLE engine (conan 24/24 qwen, up-from-slavery 24/24 chatterbox,
        # franklin 24/24 zonos — measured, first build). That confounds engine with
        # text perfectly: if pulpy adventure prose keeps at 100% and dense 19th-c
        # philosophical prose at 70%, nothing in the result can say which caused it.
        # It is the same confound that made moss_vg's newscaster 20/20 unreadable.
        picked = [[(lane, slug, l)
                   for l in spread(lines, min(PER_BOOK[slug],
                                              args.limit or PER_BOOK[slug]))]
                  for slug, lines in books]
        for idx in range(max(len(p) for p in picked)):
            selected.extend(p[idx] for p in picked if idx < len(p))
        got = sum(1 for s in selected if s[0] == lane)
        print(f"{lane:12s} {got:3d} lines from {len(books)} book(s), interleaved")

    # --- allocate engines PER LANE, not per line -----------------------------
    plan = {}
    for lane in TARGETS:
        n = sum(1 for s in selected if s[0] == lane)
        plan[lane] = ref_select.allocate_engines(n, lane=lane)
        print(f"  {lane:12s} {plan[lane]}")

    # Deal engines out within a lane. `selected` is already interleaved across books
    # (above), so a blocked engine queue now lands each engine across ALL the lane's
    # books rather than one book each — which is the point: the frozen-voice
    # discipline is per (book, engine), and a book that only ever sees one engine
    # gives the lane no within-book comparison between engines.
    queues = {lane: [e for e, k in sorted(plan[lane].items(), key=lambda kv: -kv[1])
                     for _ in range(k)] for lane in plan}
    for q in queues.values():
        q.reverse()

    # --- pass 1: cast every line ------------------------------------------
    # Casting runs BEFORE any voice is pinned, because pinning needs to see the
    # whole group's intent. First-line-wins was the r2 defect: up-from-slavery's
    # opening zonos line asked for "middle-aged woman" and pinned a female
    # reference, then five consecutive lines asking for a man were overridden by
    # the freeze — a male first-person autobiography narrated by a woman, against
    # the standing discipline to bias narrator casting male-ward. The ear kept all
    # five, because the audio was fine; only the design/reference disagreement
    # exposed it.
    off_register = {k["id"] for k in ref_select._load_pool()
                    if k.get("register") not in REF_REGISTERS}
    cast_rows, skipped = [], collections.Counter()
    for i, (lane, slug, src) in enumerate(selected):
        engine = queues[lane].pop() if queues[lane] else None
        if engine is None:
            skipped["no engine left in lane"] += 1
            continue
        labels = {"V": src["intended"]["V"], "A": src["intended"]["A"],
                  "T": src["intended"]["T"], "register": src.get("register", "")}
        cast = bi.casting_pass(src["text"], engine, labels=labels)
        if cast is None:
            skipped[f"{engine}: casting_pass failed"] += 1
            continue
        cast_rows.append({"i": i, "lane": lane, "slug": slug, "src": src,
                          "engine": engine, "labels": labels, "cast": cast})

    # --- pass 2: freeze one voice per (book, engine) -------------------------
    # The frozen design is chosen from the group's MAJORITY gender intent, and then
    # every line in the group carries that one design. For zonos and chatterbox the
    # design's only reader is ref_select, so once a reference is pinned a per-line
    # design is not direction at all — it is a casting call that was never honoured,
    # and leaving it in place is what let the manifest claim a woman while a man
    # spoke. One voice per book means one casting call per book.
    groups = collections.defaultdict(list)
    for r in cast_rows:
        groups[(r["slug"], r["engine"])].append(r)

    pinned, frozen_design, personas, taken = {}, {}, {}, set()
    for key, rows in groups.items():
        slug, engine = key
        designs = [r["cast"].get("voice_design", "") for r in rows]
        if engine in ("qwen", "moss_vg"):
            personas[key] = rows[0]["cast"].get("instruct", "")
            continue
        if engine not in ("zonos", "chatterbox"):
            continue
        votes = collections.Counter(ref_select.design_gender(d) for d in designs)
        want = votes.most_common(1)[0][0]
        design = next(d for d in designs if ref_select.design_gender(d) == want)
        frozen_design[key] = design
        same_book = {v for (b, _e), v in pinned.items() if b == slug}
        try:
            _wav, _text, meta = ref_select.select_reference(
                design, rows[0]["labels"], used=taken,
                exclude=off_register | same_book)
            rid = meta.get("id") if isinstance(meta, dict) else None
        except LookupError:
            rid = None
        if not rid:
            # Distinctness ACROSS books is a preference, not a requirement — making
            # it a requirement is what exhausted the 16-clip pool mid-build before
            # the real-speech pool landed. Reusing costs a repeated voice, not the round.
            rid = next((v for (b, e2), v in pinned.items() if e2 == engine), None)
            if not rid:
                skipped[f"{engine}: no narration reference available at all"] += 1
                continue
            skipped[f"{engine}: reused a pinned ref (pool exhausted)"] += 1
        pinned[key] = rid
        taken.add(rid)
        if len(votes) > 1:
            print(f"  {slug}|{engine}: intent split {dict(votes)} -> pinned {want}")

    # --- pass 3: emit -------------------------------------------------------
    lines_out = []
    for r in cast_rows:
        key, engine, lane, src = (r["slug"], r["engine"]), r["engine"], r["lane"], r["src"]
        cast = dict(r["cast"])
        if engine in ("qwen", "moss_vg"):
            cast["instruct"] = personas.get(key, cast.get("instruct", ""))
        elif key in frozen_design:
            cast["voice_design"] = frozen_design[key]
        elif engine in ("zonos", "chatterbox"):
            skipped[f"{engine}: group lost its reference"] += 1
            continue

        tag = {"engine": engine, **r["labels"], **cast}
        eng, direction = bi.build_direction(tag, src["text"], lane=lane)

        kept, dropped = ref_select.route_engines(
            cast.get("voice_design", ""), [eng], lane)
        if not kept:
            skipped[f"routed out: {dropped}"] += 1
            continue

        lines_out.append({
            "id": f"{r['slug']}_r2_{r['i']:04d}_{eng[:3].upper()}",
            "engine": eng,
            "register": src.get("register", "neutral_narration"),
            "chunk_type": "narration",
            "intended": src["intended"],
            "seed": SEED_BASE + r["i"],
            "text": src["text"],
            "direction": direction,
            "intended_delivery": lane,
            "book": r["slug"],
            "ref_id": pinned.get(key),
            "source_ref": src.get("source_ref", {}),
        })

    # --- report --------------------------------------------------------------
    by = collections.Counter((l["intended_delivery"], l["engine"]) for l in lines_out)
    print(f"\n{len(lines_out)} lines built" + (f", {sum(skipped.values())} skipped" if skipped else ""))
    for k, v in sorted(by.items()):
        print(f"  {k[0]:12s} {k[1]:11s} {v:3d}")
    for reason, n in skipped.items():
        print(f"  !! {n} x {reason}")

    z = [l for l in lines_out if l["engine"] == "zonos"]
    if z:
        nul = sum(1 for l in z if l["direction"].get("emotion") is None)
        print(f"\nzonos: {nul}/{len(z)} emotion null "
              f"({'PASS — director path clean' if nul == len(z) else 'CHECK the rest'})")
    print(f"pinned refs: {len(pinned)}   frozen personas: {len(personas)}")

    if not args.apply:
        print("\nreport only — pass --apply to write")
        return 0

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = {
        "version": f"{CAMPAIGN}-1",
        "campaign": CAMPAIGN,
        "note": ("Round 2 of the narration push: Neutral +81 / Documentary +35, the last "
                 "two open lanes — rebuilt as one Neutral lane after the Documentary "
                 "retirement (2026-08-10). Engine allocation from ref_select.ENGINE_MIX_BY_LANE "
                 "(measured per lane, floored, capped) rather than pass-1's per-line "
                 "choice, which routed 220/220 narration lines to zonos. Pass 2 re-run "
                 "per assigned engine. One frozen voice per (book, engine). This is also "
                 "the end-to-end director-path test for zonos emotion=null."),
        "license_note": ("Text: Standard Ebooks CC0. Synthetic audio from Apache/MIT "
                         "teacher models. Director: Gemma 4 (Apache-2.0)."),
        "lines": lines_out,
    }
    path = out / "bank.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(lines_out)} lines -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
