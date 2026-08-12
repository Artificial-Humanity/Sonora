"""Regenerate the phoneme column of an existing derivation, nothing else.

Written for the 2026-08-02 G2P contraction fix (review finding D-C1): the
espeak-free lane had no apostrophe keys in its dictionary and no "'" in the
neural charset, so every contraction was phonemized from its apostrophe-
stripped letters — `don't` -> dˈɔnt, `we'll` -> wˈɛl (identical to *well*).
The labels and the train/val split of the affected corpus are fine; only the
IPA is wrong, and the source transcripts are still on disk beside the wavs.

So this reads a derivation dir, re-phonemizes each row from its
`.normalized.txt`, and writes a new dir with the speaker index, the VAT
labels and the exact train/val membership preserved. That keeps the relabel
comparable to the original — a full re-derivation would re-measure audio and
re-roll the split for no reason.

The input dir is never modified; the previous version stays on disk as
lineage.

Usage:
    .venv/bin/python scripts/tools/rephonemize_corpus.py \
        --in data/libritts_r_vat_v3 --out data/libritts_r_vat_v3b
"""

import argparse
import json
import os
import shutil
import sys

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

from matcha.text.op_g2p import OpenPhonemizerG2P  # noqa: E402

PARTS = ("train_op.txt", "val_op.txt")
SIDECARS = ("speakers.json", "measures.jsonl")


def transcript_for(wav_path):
    """LibriTTS-R keeps the normalized text beside the wav."""
    return wav_path[:-4] + ".normalized.txt" if wav_path.endswith(".wav") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--assets", default=None)
    ap.add_argument("--no-neural-oov", action="store_true")
    ap.add_argument("--allow-unresolved", action="store_true",
                    help="write the output even if some words are unresolved "
                         "(they ship as bare letters, invisible to validate)")
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        sys.exit("ABORT: --out must differ from --in; the source derivation "
                 "is kept as lineage.")

    g2p = OpenPhonemizerG2P(assets_dir=args.assets,
                            use_neural_oov=not args.no_neural_oov)

    rewritten = {}
    changed = samples = 0
    missing_text = []
    violations = set()
    bad_rows = 0

    for part in PARTS:
        src_path = os.path.join(args.src, part)
        out_rows = []
        with open(src_path, encoding="utf-8") as f:
            rows = [ln.rstrip("\n") for ln in f if ln.strip()]
        for row in rows:
            wav, spk, old_ipa, vat = row.split("|")
            txt_path = transcript_for(wav)
            if not txt_path or not os.path.exists(txt_path):
                missing_text.append(wav)
                out_rows.append(row)
                continue
            with open(txt_path, encoding="utf-8") as f:
                text = f.read().strip()
            ipa = g2p.phonemize(text)
            bad = g2p.validate(ipa)
            if bad:
                bad_rows += 1
                violations.update(bad)
            if ipa != old_ipa:
                changed += 1
                if samples < 5:
                    print(f"  {os.path.basename(wav)}\n"
                          f"    was: {old_ipa[:90]}\n"
                          f"    now: {ipa[:90]}")
                    samples += 1
            out_rows.append("|".join([wav, spk, ipa, vat]))
        rewritten[part] = out_rows
        print(f"{part}: {len(rows)} rows")

    s = g2p.stats
    total = sum(s.values())
    print(f"\nG2P: {total} words | dict {s['dict_hits']} "
          f"({100 * s['dict_hits'] / max(total, 1):.3f}%) | "
          f"contractions {s['contraction_hits']} | "
          f"neural OOV {s['neural_hits']} | "
          f"apostrophe fallback {s['apostrophe_fallbacks']} | "
          f"unresolved {s['oov_misses']}")
    print(f"rows with a changed phoneme string: {changed}")

    fatal = []
    if missing_text:
        fatal.append(f"{len(missing_text)} rows have no .normalized.txt "
                     f"(e.g. {missing_text[:3]})")
    if bad_rows:
        fatal.append(f"{bad_rows} rows have out-of-vocab characters: {violations}")
    if s["oov_misses"] and not args.allow_unresolved:
        fatal.append(f"{s['oov_misses']} unresolved words would ship as bare "
                     f"letters, e.g. {sorted(g2p.oov_words)[:10]}")
    if fatal:
        sys.exit("ABORT, nothing written:\n  " + "\n  ".join(fatal))

    os.makedirs(args.dst, exist_ok=True)
    for part, out_rows in rewritten.items():
        with open(os.path.join(args.dst, part), "w", encoding="utf-8") as f:
            f.write("\n".join(out_rows) + "\n")
        print(f"wrote {len(out_rows)} rows -> {os.path.join(args.dst, part)}")
    for name in SIDECARS:
        src_side = os.path.join(args.src, name)
        if os.path.exists(src_side):
            shutil.copy2(src_side, os.path.join(args.dst, name))

    report_path = os.path.join(args.src, "derivation_report.json")
    report = {}
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    report["rephonemized_from"] = os.path.abspath(args.src)
    report["rephonemize_note"] = (
        "Phoneme column regenerated with the apostrophe-aware G2P "
        "(contraction table + clitic decomposition). Labels, speaker index "
        "and train/val membership carried over unchanged."
    )
    report["rephonemize_stats"] = {**s, "rows_changed": changed}
    with open(os.path.join(args.dst, "derivation_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("report ->", os.path.join(args.dst, "derivation_report.json"))


if __name__ == "__main__":
    main()
