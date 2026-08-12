"""Pre-phonemize training filelists with the espeak-free G2P lane.

Reads `path|text` (or `path|spk|text`) filelists, phonemizes the text column
through matcha.text.op_g2p (OpenPhonemizer dict primary, DeepPhonemizer
TFLite OOV fallback, U+0303 rule), validates every output character against
the locked 178-symbol vocab, and writes `<stem>_op.txt` alongside.

Train with `cleaners: [no_cleaners]` on the phonemized filelists (see
configs/data/ljspeech_op.yaml) — the training container then needs neither
espeak-ng nor the phonemizer package.

Usage:
    python scripts/tools/phonemize_filelist.py data/LJSpeech-1.1/train.txt \
        data/LJSpeech-1.1/val.txt [--assets DIR] [--no-neural-oov]

Exit code 1 if any line failed vocab validation or contained an unresolved
word; the `_op` output is not written in that case, so a poisoned filelist
cannot outlive the FAIL message.
"""

import argparse
import os
import re
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
        unresolved_lines = 0
        violations = set()
        out_lines = []
        with open(filelist, encoding="utf-8") as f:
            rows = [line.rstrip("\n") for line in f if line.strip()]
        for row in rows:
            parts = row.split("|")
            text = parts[-1]
            if _DIGIT_RE.search(text):
                digit_lines += 1  # op_g2p does not expand digits
            before_oov = g2p.stats["oov_misses"]
            ipa = g2p.phonemize(text)
            bad = g2p.validate(ipa)
            if bad:
                bad_lines += 1
                violations.update(bad)
            if g2p.stats["oov_misses"] > before_oov:
                unresolved_lines += 1
            out_lines.append("|".join(parts[:-1] + [ipa]))
        dest = out_path(filelist)
        print(f"{filelist}: {len(rows)} lines")
        if digit_lines:
            print(f"  WARNING: {digit_lines} lines contain digits — feed "
                  "normalized text; digits are not expanded")
        if bad_lines:
            print(f"  FAIL: {bad_lines} lines have out-of-vocab characters: "
                  f"{violations}")
        if unresolved_lines:
            # Unresolved words are passed through as bare letters, which
            # validate() cannot see (a-z are in the vocab). Without this
            # check the poisoned filelist would look clean.
            print(f"  FAIL: {unresolved_lines} lines contain unresolved "
                  "words (passed through as letters — invisible to validate)")
        if bad_lines or unresolved_lines:
            # Do not write a poisoned filelist. It used to be written anyway,
            # leaving a file someone could train on long after the FAIL
            # scrolled off the terminal.
            any_bad = True
            print(f"  {dest} NOT written")
        else:
            with open(dest, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n")
            print(f"  -> {dest}")
    s = g2p.stats
    total = (s["dict_hits"] + s["neural_hits"] + s["oov_misses"]
             + s["contraction_hits"] + s["apostrophe_fallbacks"])
    print(f"\nG2P: {total} words | dict {s['dict_hits']} "
          f"({100 * s['dict_hits'] / max(total, 1):.3f}%) | "
          f"contractions {s['contraction_hits']} | "
          f"neural OOV {s['neural_hits']} | "
          f"apostrophe fallback {s['apostrophe_fallbacks']} | "
          f"unresolved {s['oov_misses']}")
    if g2p.apostrophe_fallback_words:
        sample = sorted(g2p.apostrophe_fallback_words)[:20]
        print(f"apostrophe words guessed from bare letters (first 20): {sample}")
    if g2p.oov_words:
        sample = sorted(g2p.oov_words)[:20]
        print(f"unresolved words (first 20): {sample}")
    sys.exit(1 if any_bad else 0)


if __name__ == "__main__":
    main()
