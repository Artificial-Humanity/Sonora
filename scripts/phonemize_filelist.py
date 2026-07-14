"""Pre-phonemize training filelists with the espeak-free G2P lane.

Reads `path|text` (or `path|spk|text`) filelists, phonemizes the text column
through matcha.text.op_g2p (OpenPhonemizer dict primary, DeepPhonemizer
TFLite OOV fallback, U+0303 rule), validates every output character against
the locked 178-symbol vocab, and writes `<stem>_op.txt` alongside.

Train with `cleaners: [no_cleaners]` on the phonemized filelists (see
configs/data/ljspeech_op.yaml) — the training container then needs neither
espeak-ng nor the phonemizer package.

Usage:
    python scripts/phonemize_filelist.py data/LJSpeech-1.1/train.txt \
        data/LJSpeech-1.1/val.txt [--assets DIR] [--no-neural-oov]

Exit code 1 if any line failed vocab validation (report printed either way).
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from matcha.text.op_g2p import OpenPhonemizerG2P  # noqa: E402

_DIGIT_RE = re.compile(r"\d")


def out_path(in_path):
    stem, ext = os.path.splitext(in_path)
    return f"{stem}_op{ext or '.txt'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filelists", nargs="+")
    ap.add_argument("--assets", default=None,
                    help="litert-community/Matcha-TTS assets dir (default: "
                         "$SONORA_G2P_ASSETS or ../Reference/models/...)")
    ap.add_argument("--no-neural-oov", action="store_true",
                    help="dictionary only; OOV words become vocab violations")
    args = ap.parse_args()

    g2p = OpenPhonemizerG2P(assets_dir=args.assets,
                            use_neural_oov=not args.no_neural_oov)
    any_bad = False
    for filelist in args.filelists:
        bad_lines = 0
        digit_lines = 0
        violations = set()
        out_lines = []
        with open(filelist, encoding="utf-8") as f:
            rows = [line.rstrip("\n") for line in f if line.strip()]
        for row in rows:
            parts = row.split("|")
            text = parts[-1]
            if _DIGIT_RE.search(text):
                digit_lines += 1  # op_g2p does not expand digits
            ipa = g2p.phonemize(text)
            bad = g2p.validate(ipa)
            if bad:
                bad_lines += 1
                violations.update(bad)
            out_lines.append("|".join(parts[:-1] + [ipa]))
        dest = out_path(filelist)
        with open(dest, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"{filelist}: {len(rows)} lines -> {dest}")
        if digit_lines:
            print(f"  WARNING: {digit_lines} lines contain digits — feed "
                  "normalized text; digits are not expanded")
        if bad_lines:
            any_bad = True
            print(f"  FAIL: {bad_lines} lines have out-of-vocab characters: "
                  f"{violations}")
    s = g2p.stats
    total = s["dict_hits"] + s["neural_hits"] + s["oov_misses"]
    print(f"\nG2P: {total} words | dict {s['dict_hits']} "
          f"({100 * s['dict_hits'] / max(total, 1):.3f}%) | "
          f"neural OOV {s['neural_hits']} | unresolved {s['oov_misses']}")
    if g2p.oov_words:
        sample = sorted(g2p.oov_words)[:20]
        print(f"unresolved words (first 20): {sample}")
    sys.exit(1 if any_bad else 0)


if __name__ == "__main__":
    main()
