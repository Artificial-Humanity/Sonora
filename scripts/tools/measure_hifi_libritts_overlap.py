# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = []
# # Text only. No torch, no audio, no G2P — this reads `.normalized.txt` and the
# # converter's `manifest.jsonl`, so it runs on any interpreter in minutes.
# ///
"""Which Hi-Fi TTS recordings are the same LibriVox recordings as LibriTTS-R chapters.

HOW TO RUN IT: the repo **.venv**, not `uv run` (AGENTS.md §3).

    .venv/bin/python scripts/tools/measure_hifi_libritts_overlap.py
    .venv/bin/python scripts/tools/measure_hifi_libritts_overlap.py --out overlap.json

WHY THIS EXISTS. `convert_hifi_tts.py` deliberately does not deduplicate, and says why —
four of Hi-Fi TTS's ten readers are already in LibriTTS-R under the same LibriVox reader
ids, its exact-transcript probe is a FLOOR rather than a measurement, and the overlap it
bounds from below is a recording-level fact. The v8 merge has to act on it. This is the
measurement it acts on.

⚠⚠ THE COMPARISON UNIT IS THE LIBRIVOX SECTION, NOT THE BOOK, AND THAT IS THE WHOLE POINT.
A Hi-Fi TTS `book` directory is a whole LibriVox TITLE — book 7670 holds twelve sections —
while a LibriTTS-R `chapter` is ONE section. Scoring title against section answers "does
this title contain duplicated material", which is true of a title 96% of whose audio is
clean. Measured here: in each of the two overlapping titles exactly ONE section matches and
every other section scores 0.0000, so dropping sections costs 323 clips where dropping
titles costs 3,003 — the same verdict at a tenth of the price, in a rung whose entire
purpose is adding hours.

The section comes off the clip stem (`…_07_richards_0002` -> `…_07_richards`) by removing
the final token, which is the utterance index. One token, from the right: a title may
contain underscores and this does not care how many. A stem whose last token is not a
number is REFUSED rather than guessed at.

WHAT IT MEASURES. Word n-gram containment (default n=8) between the concatenated text of
each group:

    containment(A, B) = |A n B| / min(|A|, |B|)

Containment rather than Jaccard, because the two sides are deliberately different sizes and
Jaccard scores a true duplicate low for exactly the reason it IS one. Measured: Jaccard puts
the two real pairs at 0.114 and 0.028, under any threshold that excludes noise.

⚠ THE TEXT IS CONCATENATED BEFORE SHINGLING, in stem order, which is source order. Shingling
each clip separately means no n-gram ever spans a clip boundary and every clip shorter than n
words contributes NOTHING — which understated these containments by about 40% relative
(0.392 -> 0.667, 0.342 -> 0.657) in the version of this tool that did it.

⚠ n AND THE THRESHOLD ARE LOAD-BEARING JOINTLY, so both are flags. The separation is total
at n=8 (two nonzero pairs; every other pair exactly 0.0000, and so is the cross-reader
floor), but a larger n shrinks a true pair's score toward the threshold — at n=12 one of the
two drops below the 0.20 default. Re-run across a range before trusting a null.

WHICH SIDE GETS DROPPED IS NOT THIS SCRIPT'S CALL, but the answer is forced: standing rule 2
of the quality ladder is that every rung ADDS rows without re-rolling what came before, so a
v7 row cannot move. The duplicate material leaves on the Hi-Fi TTS side. This prints what
that costs AND how much LibriTTS-R material is actually implicated, because a cost with no
problem size beside it is how a ten-fold over-removal reads as reasonable.

THE CONTROLS RUN EVERY TIME, ON BOTH SIDES, because a containment table full of zeros looks
identical whether the corpora are disjoint or the text never loaded:

  * POSITIVE — every group on BOTH sides is scored against itself and must be 1.0, and any
    group that yielded no text at all is a REFUSAL naming it. ⚠ An earlier version scored
    only the LibriTTS-R side. Renaming the Hi-Fi TTS transcripts to a suffix `read_tree`
    skips then produced "none", both controls green, exit 0 — a vacuous pass on the very
    side being measured for deletion.
  * NEGATIVE — the best pair between two DIFFERENT readers, printed WITH the number of pairs
    it was computed over. A floor over an empty population is not a floor, and with one
    shared reader that population is empty.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HIFI = "/data/model-training/datasets/hifi_tts_24k"
LIBRITTS = "/data/model-training/datasets/LibriTTS_R"
SUBSETS = ("train-clean-100", "train-clean-360", "train-other-500", "dev-clean")
SHINGLE = 8
TEXT_SUFFIX = ".normalized.txt"

_WORD = re.compile(r"[a-z0-9']+")


def shingles(text, n=SHINGLE):
    """The set of word n-grams in `text`, lowercased and stripped of punctuation."""
    words = _WORD.findall(text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _clip_texts(directory):
    """`{stem: text}` for the transcripts directly under `directory`, in stem order."""
    return {name[:-len(TEXT_SUFFIX)]: open(os.path.join(directory, name), encoding="utf-8").read()
            for name in sorted(os.listdir(directory)) if name.endswith(TEXT_SUFFIX)}


def _section(stem):
    """The LibriVox section a clip stem belongs to: the stem minus its utterance index."""
    head, _, tail = stem.rpartition("_")
    if not head or not tail.isdigit():
        raise SystemExit(
            "clip stem %r does not end in an utterance index, so its section cannot be "
            "derived. The section is the comparison unit here — guessing it would silently "
            "change what is compared." % stem)
    return head


def hifi_sections(root, n):
    """`{speaker: {(book, section): shingles}}`, text concatenated in source order."""
    out = {}
    for speaker in sorted(os.listdir(root)):
        sdir = os.path.join(root, speaker)
        if not os.path.isdir(sdir):
            continue
        groups = {}
        for book in sorted(os.listdir(sdir)):
            bdir = os.path.join(sdir, book)
            if not os.path.isdir(bdir):
                continue
            texts = _clip_texts(bdir)
            if not texts:
                raise SystemExit(
                    "no %s files under %s. This tool measures that directory for deletion, "
                    "so reading none of it must not look like finding no overlap."
                    % (TEXT_SUFFIX, bdir))
            joined = {}
            for stem, text in texts.items():
                joined.setdefault(_section(stem), []).append(text)
            for section, parts in joined.items():
                groups[(book, section)] = shingles(" ".join(parts), n)
        out[speaker] = groups
    return out


def libritts_sections(root, subsets, n):
    """`{speaker: {(subset, chapter): shingles}}`. A LibriTTS-R chapter IS one section."""
    present = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    unknown = [s for s in subsets if s not in present]
    if unknown:
        # ⚠ The sibling half of this rung refuses the same way and for the same reason: a
        # typo'd name that silently measures a smaller population is this repo's
        # most-repeated failure. Dropping one subset here loses a whole flagged pair.
        raise SystemExit("no subset directory matches %s under %s — available: %s"
                         % (", ".join(repr(u) for u in unknown), root, ", ".join(present)))
    found = {}
    for subset in subsets:
        sdir = os.path.join(root, subset)
        for speaker in sorted(os.listdir(sdir)):
            spk_dir = os.path.join(sdir, speaker)
            if not os.path.isdir(spk_dir):
                continue
            for chapter in sorted(os.listdir(spk_dir)):
                cdir = os.path.join(spk_dir, chapter)
                if not os.path.isdir(cdir):
                    continue
                texts = _clip_texts(cdir)
                if not texts:
                    raise SystemExit("no %s files under %s" % (TEXT_SUFFIX, cdir))
                found.setdefault(speaker, {})[(subset, chapter)] = \
                    shingles(" ".join(texts[k] for k in sorted(texts)), n)
    return found


def containment(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def clip_cost(manifest_path):
    """`{(speaker, book, section): (clips, seconds)}` from the converter's own manifest."""
    cost = {}
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            key = (row["speaker"], row["book"], _section(row["id"]))
            clips, secs = cost.get(key, (0, 0.0))
            cost[key] = (clips + 1, secs + float(row["source_duration"]))
    return cost


def libritts_cost(root, subset, speaker, chapter):
    """`(clips, seconds)` for one LibriTTS-R chapter — the SIZE OF THE PROBLEM.

    Duration comes from the wav headers rather than a manifest, because LibriTTS-R ships no
    manifest. Read as a 44-byte header, not decoded.
    """
    import wave
    cdir = os.path.join(root, subset, speaker, chapter)
    clips = secs = 0
    for name in sorted(os.listdir(cdir)):
        if not name.endswith(".wav"):
            continue
        clips += 1
        with wave.open(os.path.join(cdir, name)) as w:
            secs += w.getnframes() / float(w.getframerate())
    return clips, secs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hifi-root", default=HIFI)
    ap.add_argument("--libritts-root", default=LIBRITTS)
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--shingle", type=int, default=SHINGLE,
                    help="word n-gram size; load-bearing jointly with --threshold")
    ap.add_argument("--threshold", type=float, default=0.20,
                    help="containment at or above which a pair is called the same recording")
    ap.add_argument("--out", default=None, help="write the full result as JSON here")
    args = ap.parse_args()

    manifest = os.path.join(args.hifi_root, "manifest.jsonl")
    for path in (args.hifi_root, args.libritts_root, manifest):
        if not os.path.exists(path):
            raise SystemExit("missing: %s" % path)
    if args.shingle < 1:
        raise SystemExit("--shingle must be >= 1")

    hifi = hifi_sections(args.hifi_root, args.shingle)
    lib = libritts_sections(args.libritts_root, args.subsets, args.shingle)
    shared = sorted(set(hifi) & set(lib),
                    key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))
    if not shared:
        raise SystemExit(
            "no reader appears in both corpora — that is either a real change in the "
            "speaker sets or a wrong --root, and both need a person. hifi=%d readers, "
            "libritts=%d readers" % (len(hifi), len(lib)))

    # POSITIVE CONTROL, BOTH SIDES. An empty group is a refusal, not a zero.
    controlled = 0
    for side, groups in (("hifi", hifi), ("libritts", lib)):
        for spk in shared:
            for key, sh in groups[spk].items():
                if not sh:
                    raise SystemExit(
                        "%s group %s/%s yielded no %d-grams — too little text to compare, and "
                        "scoring it as 0.0 would read as 'no overlap'"
                        % (side, spk, "/".join(key), args.shingle))
                if containment(sh, sh) != 1.0:
                    raise SystemExit("positive control FAILED on %s %s/%s" % (side, spk, key))
                controlled += 1

    # NEGATIVE CONTROL, with the population it was computed over.
    floor, floor_pair, compared = 0.0, None, 0
    for spk in shared:
        for other in shared:
            if other == spk:
                continue
            for hkey, hsh in hifi[spk].items():
                for lkey, lsh in lib[other].items():
                    compared += 1
                    c = containment(hsh, lsh)
                    if c > floor:
                        floor, floor_pair = c, ("%s/%s" % (spk, "/".join(hkey)),
                                                "%s/%s" % (other, "/".join(lkey)))

    cost = clip_cost(manifest)
    pairs, flagged = [], []
    for spk in shared:
        for (book, section), hsh in sorted(hifi[spk].items()):
            for (subset, chapter), lsh in sorted(lib[spk].items()):
                c = containment(hsh, lsh)
                pairs.append({"speaker": spk, "hifi_book": book, "hifi_section": section,
                              "libritts": "%s/%s" % (subset, chapter),
                              "containment": round(c, 6)})
                if c < args.threshold:
                    continue
                key = (spk, book, section)
                if key not in cost:
                    raise SystemExit(
                        "flagged section %s is absent from %s, so its clip count and duration "
                        "are unmeasured. Printing 0 clips would read as 'cheap to drop'."
                        % ("/".join(key), manifest))
                clips, secs = cost[key]
                lclips, lsecs = libritts_cost(args.libritts_root, subset, spk, chapter)
                flagged.append({"speaker": spk, "hifi_book": book, "hifi_section": section,
                                "libritts": "%s/%s" % (subset, chapter),
                                "containment": round(c, 6), "clips": clips,
                                "hours": secs / 3600.0, "libritts_clips": lclips,
                                "libritts_hours": lsecs / 3600.0})

    n_hifi = sum(len(hifi[s]) for s in shared)
    n_lib = sum(len(lib[s]) for s in shared)
    print("readers in both corpora : %s" % ", ".join(shared))
    print("sections compared       : %d same-reader pairs over %d hifi and %d libritts sections"
          % (len(pairs), n_hifi, n_lib))
    print("shingle / threshold     : n=%d, flag at containment >= %.2f" % (args.shingle, args.threshold))
    print("positive control        : %d groups self-score 1.0, both sides, none empty" % controlled)
    print("negative control        : best of %d cross-reader pairs = %.6f%s"
          % (compared, floor, "" if floor_pair is None else "  (%s vs %s)" % floor_pair))
    if compared == 0:
        print("  ⚠ NO CROSS-READER PAIRS EXIST in this population, so the floor above is the")
        print("    initial value and not a measurement. Widen --subsets to restore it.")
    print()
    print("SAME RECORDING")
    if not flagged:
        print("  none")
    for hit in sorted(flagged, key=lambda h: -h["clips"]):
        print("  reader %-5s %-46s %5d clips %5.2f h  containment %.3f" %
              (hit["speaker"], hit["hifi_section"], hit["clips"], hit["hours"], hit["containment"]))
        print("      matches LibriTTS-R %-22s %5d clips %5.2f h   <- the material actually duplicated"
              % (hit["libritts"], hit["libritts_clips"], hit["libritts_hours"]))
    print()
    drop_clips = sum(h["clips"] for h in flagged)
    drop_hours = sum(h["hours"] for h in flagged)
    books = {(h["speaker"], h["hifi_book"]) for h in flagged}
    book_clips = sum(c for (s, b, _sec), (c, _d) in cost.items() if (s, b) in books)
    book_hours = sum(d for (s, b, _sec), (_c, d) in cost.items() if (s, b) in books) / 3600.0
    print("DROP THE SECTIONS : %d clips, %.2f h of %d total hifi clips"
          % (drop_clips, drop_hours, sum(c for c, _ in cost.values())))
    print("DROP THE TITLES   : %d clips, %.2f h  — %.2f h of it not duplicated at all"
          % (book_clips, book_hours, book_hours - drop_hours))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"threshold": args.threshold, "shingle": args.shingle, "readers": shared,
                       "positive_control_groups": controlled,
                       "negative_control": {"pairs_compared": compared,
                                            "best_cross_reader": floor, "pair": floor_pair},
                       "flagged": flagged, "pairs": pairs}, fh, indent=2)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
