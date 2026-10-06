"""Pure helpers for the staged rebuild's corpora (`scripts/tools/prep_staged_corpora.py`).

Pre-registered in Notes: Sonora/staged-rebuild-preregistration.md. Two new corpora, both on
stock's 22.05 kHz audio path:

  vctk_22k        C0: stock's own data. VCTK 0.92 mic1 resampled to 22.05 kHz, the VITS
                  espeak phonemes stock trained on, and its speaker ids.
  libritts_r_22k  S1: the derisk corpus (LibriTTS-R train-clean-100, our G2P, the
                  247-speaker table) resampled to 22.05 kHz, with the VAT field dropped.

No torch here, so the host test suite covers it.
"""

import math
from pathlib import Path

try:                                    # imported as `lib.staged_data` by the tools
    from .baseline_bench import original_matcha_symbols, ours_symbols
except ImportError:                     # imported flat, with scripts/lib on the path
    from baseline_bench import original_matcha_symbols, ours_symbols

TARGET_SR = 22050
MEL_KEYS = ("n_fft", "n_feats", "sample_rate", "hop_length", "win_length", "f_min", "f_max")
STOCK_MEL = (1024, 80, 22050, 256, 1024, 0, 8000)      # upstream Matcha's VCTK mel

VCTK_22K_ROOT = "/data/model-training/datasets/VCTK/vctk_22k"
LIBRI_24K_ROOT = "/data/model-training/datasets/LibriTTS_R/train-clean-100"
LIBRI_22K_ROOT = "/data/model-training/datasets/LibriTTS_R/train-clean-100_22k"


def upstream_as_ours(text):
    """Upstream-Matcha phoneme text -> the text OUR symbol table encodes to the same ids.

    stock learned upstream's ids, and our table differs in its last two slots
    (`baseline_bench.original_matcha_symbols`). Fed through our `text_to_sequence` as they
    are, the VITS phonemes would give "'" and "ᵻ" the wrong embeddings, and C0 would differ
    from stock in its input as well as its training. Rewriting the text leaves the training
    code untouched: each character becomes the one that sits in OUR table at the id upstream
    gives it."""
    up = {s: i for i, s in enumerate(original_matcha_symbols())}
    bad = sorted({c for c in text if c not in up})
    if bad:
        raise ValueError("symbols not in upstream Matcha's table: %s" % bad)
    ours = ours_symbols()
    return "".join(ours[up[c]] for c in text)


def read_vits(path):
    """VITS filelist -> [(utterance, speaker, sid, text)] in file order."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p, sid, text = line.split("|", 2)
            out.append((Path(p).stem, p.split("/")[1], int(sid), text))
    return out


def vctk_22k_path(spk, utt):
    return "%s/%s/%s.wav" % (VCTK_22K_ROOT, spk, utt)


def c0_lines(rows, have):
    """VITS `.cleaned` rows -> C0 filelist lines `wav|sid|text`, and the utterances left out.

    `have` holds the utterances with a VCTK 0.92 mic1 recording. A row without one is
    dropped and counted, never given a neighbour's audio."""
    lines, missing = [], []
    for utt, spk, sid, text in rows:
        if utt not in have:
            missing.append(utt)
            continue
        lines.append("%s|%d|%s" % (vctk_22k_path(spk, utt), sid, upstream_as_ours(text)))
    return lines, missing


def to_22k(wav24):
    """A train-clean-100 recording's path -> its 22.05 kHz copy. Any other root is refused,
    so a row from another corpus cannot be written somewhere plausible and wrong."""
    if not wav24.startswith(LIBRI_24K_ROOT + "/"):
        raise ValueError("not under %s: %s" % (LIBRI_24K_ROOT, wav24))
    return LIBRI_22K_ROOT + wav24[len(LIBRI_24K_ROOT):]


def s1_line(line):
    """A derisk-corpus row `wav|spk|phonemes|v,a,t` -> `wav22|spk|phonemes`."""
    f = line.split("|")
    if len(f) != 4:
        raise ValueError("expected 4 fields wav|spk|phonemes|vat, got %d: %r"
                         % (len(f), line[:80]))
    return "%s|%s|%s" % (to_22k(f[0]), f[1], f[2])


def steps_per_epoch(n_rows, batch_size=32):
    """Optimizer steps per epoch under `BucketBatchSampler`: ceil(rows / batch). Checked
    against the derisk run: 29,441 rows, 92,100 steps over 100 epochs."""
    return math.ceil(n_rows / batch_size)
