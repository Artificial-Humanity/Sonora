"""What homograph resolution would change in a corpus, without changing it (D-M4).

The G2P's homograph pass ships OFF. This is the report that says whether to turn it on:
it walks a derivation's transcripts, counts every homograph token, and reports which
ones the resolver would move and which it would leave alone — plus the words it knows
about but cannot reach at all (homographs.NOT_RESOLVABLE).

The decision counter runs whether or not the flag is set (see op_g2p._homograph_ipa), so
the default invocation is a true dry run: nothing is written and no phonemes change.
`--apply-sample` re-phonemizes a handful of affected lines both ways so the diff can be
read directly rather than inferred from counts.

Usage:
    .venv/bin/python scripts/measure_homographs.py --corpus data/libritts_r_vat_v3c
    .venv/bin/python scripts/measure_homographs.py --corpus data/libritts_r_vat_v3c \
        --apply-sample 20 --json notes/homograph-report.json
"""

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from matcha.text.homographs import (  # noqa: E402
    HOMOGRAPHS,
    NOT_RESOLVABLE,
    resolve,
)
from matcha.text.op_g2p import OpenPhonemizerG2P  # noqa: E402

PARTS = ("train_op.txt", "val_op.txt")
WORD_RE = re.compile(r"[a-z']+")


def transcript_for(wav_path):
    """LibriTTS-R keeps the normalized text beside the wav."""
    return wav_path[:-4] + ".normalized.txt" if wav_path.endswith(".wav") else None


def corpus_rows(corpus):
    """(part, wav, transcript) for every row that still has its transcript on disk."""
    for part in PARTS:
        path = os.path.join(corpus, part)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                wav = line.split("|", 1)[0]
                tpath = transcript_for(wav)
                if tpath and os.path.isfile(tpath):
                    with open(tpath, encoding="utf-8") as tf:
                        yield part, wav, tf.read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="derivation dir holding train_op.txt / val_op.txt")
    ap.add_argument("--assets", default=None)
    ap.add_argument("--apply-sample", type=int, default=0,
                    help="re-phonemize N affected lines both ways and print the diff")
    ap.add_argument("--json", default=None, help="write the report here as well")
    args = ap.parse_args()

    # Counted from the raw token stream, so "occurrences" is independent of whether the
    # resolver understands the word — that is the denominator the flip rate needs.
    occurrences = collections.Counter()
    flips = collections.Counter()
    senses = collections.Counter()
    unreachable = collections.Counter()
    rows_affected = 0
    total_rows = 0
    total_tokens = 0
    examples = collections.defaultdict(list)

    for _part, wav, text in corpus_rows(args.corpus):
        total_rows += 1
        toks = WORD_RE.findall(text.lower())
        total_tokens += len(toks)
        touched = False
        for i, tok in enumerate(toks):
            if tok in NOT_RESOLVABLE:
                unreachable[tok] += 1
            if tok not in HOMOGRAPHS:
                continue
            occurrences[tok] += 1
            decision = resolve(
                tok,
                prev=toks[i - 1] if i >= 1 else None,
                prev2=toks[i - 2] if i >= 2 else None,
                nxt=toks[i + 1] if i + 1 < len(toks) else None,
            )
            if decision is None:
                continue
            flips[tok] += 1
            senses[decision.sense] += 1
            touched = True
            if len(examples[tok]) < 6:
                left = " ".join(toks[max(0, i - 4):i])
                right = " ".join(toks[i + 1:i + 5])
                examples[tok].append({
                    "wav": wav,
                    "context": f"{left} [{tok}] {right}",
                    "from": HOMOGRAPHS[tok].default,
                    "to": decision.ipa,
                    "rule": decision.rule,
                })
        if touched:
            rows_affected += 1

    total_occ = sum(occurrences.values())
    total_flips = sum(flips.values())
    print(f"corpus              {os.path.abspath(args.corpus)}")
    print(f"rows                {total_rows}")
    print(f"word tokens         {total_tokens}")
    print(f"homograph tokens    {total_occ} "
          f"({100.0 * total_occ / max(total_tokens, 1):.3f}% of tokens)")
    print(f"would change        {total_flips} tokens in {rows_affected} rows "
          f"({100.0 * rows_affected / max(total_rows, 1):.2f}% of rows)")
    print(f"abstained           {total_occ - total_flips} "
          f"({100.0 * (total_occ - total_flips) / max(total_occ, 1):.1f}% of homograph tokens)")
    print(f"by sense            {dict(senses.most_common())}")

    print(f"\n{'word':<14}{'seen':>6}{'flips':>7}{'rate':>8}  {'default':<16} -> alternate")
    print("-" * 84)
    for word, seen in occurrences.most_common():
        n = flips[word]
        entry = HOMOGRAPHS[word]
        alt = " / ".join(sorted(set(entry.senses.values()))) if entry.senses else "-"
        print(f"{word:<14}{seen:>6}{n:>7}{100.0 * n / seen:>7.1f}%  "
              f"{entry.default:<16} -> {alt}")

    if unreachable:
        print(f"\nknown but out of reach of a POS rule ({sum(unreachable.values())} tokens):")
        for word, n in unreachable.most_common():
            print(f"  {word:<12}{n:>6}  {NOT_RESOLVABLE[word]}")

    print("\n=== sample contexts ===")
    for word, _ in flips.most_common(12):
        print(f"\n--- {word}: {HOMOGRAPHS[word].default} -> ")
        for ex in examples[word]:
            print(f"    {ex['to']:<14} {ex['rule']:<22} {ex['context']}")

    if args.apply_sample:
        print(f"\n=== {args.apply_sample} lines phonemized both ways ===")
        off = OpenPhonemizerG2P(assets_dir=args.assets, homographs=False)
        on = OpenPhonemizerG2P(assets_dir=args.assets, homographs=True)
        shown = 0
        for _part, _wav, text in corpus_rows(args.corpus):
            toks = set(WORD_RE.findall(text.lower()))
            if not toks & set(flips):
                continue
            before, after = off.phonemize(text), on.phonemize(text)
            if before == after:
                continue
            print(f"\n  text   {text}")
            print(f"  off    {before}")
            print(f"  on     {after}")
            shown += 1
            if shown >= args.apply_sample:
                break

    if args.json:
        report = {
            "corpus": os.path.abspath(args.corpus),
            "rows": total_rows,
            "word_tokens": total_tokens,
            "homograph_tokens": total_occ,
            "flips": total_flips,
            "rows_affected": rows_affected,
            "by_sense": dict(senses),
            "per_word": {w: {"seen": occurrences[w], "flips": flips[w],
                             "default": HOMOGRAPHS[w].default,
                             "alternates": sorted(set(HOMOGRAPHS[w].senses.values())),
                             "source": HOMOGRAPHS[w].source}
                         for w in occurrences},
            "unreachable": {w: {"seen": n, "reason": NOT_RESOLVABLE[w]}
                            for w, n in unreachable.items()},
            "examples": {w: examples[w] for w in examples},
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
