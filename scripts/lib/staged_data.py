"""Pure helpers for the staged rebuild's corpora (`scripts/tools/prep_staged_corpora.py`).

Pre-registered in Notes: Sonora/staged-rebuild-preregistration.md. Three new corpora, all on
stock's 22.05 kHz audio path:

  vctk_22k        C0: stock's own data. VCTK 0.92 mic1 resampled to 22.05 kHz, the VITS
                  espeak phonemes stock trained on, and its speaker ids.
  libritts_r_22k  S1: the derisk corpus (LibriTTS-R train-clean-100, our G2P, the
                  247-speaker table) resampled to 22.05 kHz, with the VAT field dropped.
  libritts_22k    L (check 4): S1's rows with each recording replaced by the same
                  utterance's ORIGINAL LibriTTS recording (unrestored), resampled the same
                  way. The path relative to `train-clean-100/` is the same in both corpora.

The hum-fix batch (Notes: Sonora/hum-fix-batch-preregistration.md) adds three more:

  vctk_d1         D1: C0's train rows subsampled to 105 per speaker (`d1_subsample`).
  libritts_r_22k  G′: S1's filelists as `{train,val}_g.txt`, every `ᵻ` moved to `ᵊ`, the
    (_g files)    character our table puts at the id stock trained `ᵻ` on (`g_line`).
  f1_dense        F1: every LibriTTS-R speaker with at least 300 train rows in S1's
                  filelist and the derived train-clean-360 / train-other-500 filelists,
                  re-indexed into a table of its own (`f1_table`, `f1_line`).

No torch here, so the host test suite covers it.
"""

import math
import random
from collections import Counter
from pathlib import Path

try:                                    # imported as `lib.staged_data` by the tools
    from .baseline_bench import original_matcha_symbols, ours_symbols
except ImportError:                     # imported flat, with scripts/lib on the path
    from baseline_bench import original_matcha_symbols, ours_symbols

TARGET_SR = 22050
MEL_KEYS = ("n_fft", "n_feats", "sample_rate", "hop_length", "win_length", "f_min", "f_max")
STOCK_MEL = (1024, 80, 22050, 256, 1024, 0, 8000)      # upstream Matcha's VCTK mel

VCTK_22K_ROOT = "/data/model-training/datasets/VCTK/vctk_22k"
LIBRI_R_ROOT = "/data/model-training/datasets/LibriTTS_R"
LIBRI_24K_ROOT = LIBRI_R_ROOT + "/train-clean-100"
LIBRI_22K_ROOT = "/data/model-training/datasets/LibriTTS_R/train-clean-100_22k"
LIBRI_ORIG_ROOT = "/data/model-training/datasets/LibriTTS/train-clean-100"
LIBRI_ORIG_22K_ROOT = "/data/model-training/datasets/LibriTTS/train-clean-100_22k"
MAX_MISSING_SHARE = 0.01        # more of S1's rows than this without an original: L not trained


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


def _rebase(path, src, dst):
    """`path` under `src` -> the same relative path under `dst`. Any other root is refused,
    so a row from another corpus cannot be written somewhere plausible and wrong."""
    if not path.startswith(src + "/"):
        raise ValueError("not under %s: %s" % (src, path))
    return dst + path[len(src):]


def to_22k(wav24):
    """A train-clean-100 recording's path -> its 22.05 kHz copy."""
    return _rebase(wav24, LIBRI_24K_ROOT, LIBRI_22K_ROOT)


def to_original(wav24):
    """A LibriTTS-R train-clean-100 recording -> the same utterance's original LibriTTS one."""
    return _rebase(wav24, LIBRI_24K_ROOT, LIBRI_ORIG_ROOT)


def to_l22k(wav24):
    """A LibriTTS-R train-clean-100 recording -> L's 22.05 kHz copy of its original."""
    return _rebase(wav24, LIBRI_24K_ROOT, LIBRI_ORIG_22K_ROOT)


def original_of(wav22):
    """An S1 row's audio (LibriTTS-R at 22.05 kHz) -> the original LibriTTS recording."""
    return _rebase(wav22, LIBRI_22K_ROOT, LIBRI_ORIG_ROOT)


def s1_line(line):
    """A derisk-corpus row `wav|spk|phonemes|v,a,t` -> `wav22|spk|phonemes`."""
    f = line.split("|")
    if len(f) != 4:
        raise ValueError("expected 4 fields wav|spk|phonemes|vat, got %d: %r"
                         % (len(f), line[:80]))
    return "%s|%s|%s" % (to_22k(f[0]), f[1], f[2])


def l22k_of(wav22):
    """An S1 row's audio -> L's 22.05 kHz copy of the same utterance's original recording."""
    return _rebase(wav22, LIBRI_22K_ROOT, LIBRI_ORIG_22K_ROOT)


def l_line(line):
    """An S1 row `wav22|spk|phonemes` -> L's row: the same text, phonemes and speaker, with
    the audio moved to the 22.05 kHz copy of the utterance's original recording."""
    f = line.split("|")
    if len(f) != 3:
        raise ValueError("expected 3 fields wav|spk|phonemes, got %d: %r" % (len(f), line[:80]))
    return "|".join([l22k_of(f[0])] + f[1:])


def l_lines(rows, have):
    """S1 rows -> L's filelist lines, and the S1 paths left out.

    `have` holds the original recordings on disk (`original_of` paths). A row without one is
    dropped and counted, never given another recording."""
    lines, missing = [], []
    for line in rows:
        wav = line.split("|", 1)[0]
        if original_of(wav) not in have:
            missing.append(wav)
            continue
        lines.append(l_line(line))
    return lines, missing


def too_many_missing(n_missing, n_rows):
    """More than 1 % of S1's rows have no original recording (or there are no rows at all):
    L is not trained and the owner decides."""
    return n_rows == 0 or n_missing > MAX_MISSING_SHARE * n_rows


def steps_per_epoch(n_rows, batch_size=32):
    """Optimizer steps per epoch under `BucketBatchSampler`: ceil(rows / batch). Checked
    against the derisk run: 29,441 rows, 92,100 steps over 100 epochs."""
    return math.ceil(n_rows / batch_size)


# ---------------------------------------------------------------- the hum-fix batch
# Pre-registered in Notes: Sonora/hum-fix-batch-preregistration.md.

D1_SEED = 20261012
D1_PER_SPEAKER = 105            # S1's median train rows per speaker


def d1_subsample(lines, per=D1_PER_SPEAKER, seed=D1_SEED):
    """C0's train lines `wav|sid|text` -> D1's, as registered: ONE `random.Random(seed)`;
    speakers in order of first appearance; for each, `rng.sample(its rows in file order,
    per)`; the kept rows returned in their original file order.

    Positions are sampled, not lines: `random.sample` picks by position, so the draw is the
    same, and a line that occurs twice cannot be kept twice or ambiguously. A speaker with
    fewer than `per` rows, or a sid that names two voices (or a voice with two sids), is
    refused: the quota and the speaker table would no longer mean what the registration
    says."""
    rows, voice_of, sid_of = {}, {}, {}
    for i, line in enumerate(lines):
        wav, sid, _text = line.split("|", 2)
        sid, voice = int(sid), Path(wav).parent.name
        if voice_of.setdefault(sid, voice) != voice or sid_of.setdefault(voice, sid) != sid:
            raise ValueError("sid %d / voice %s is not one-to-one (line %d)" % (sid, voice, i + 1))
        rows.setdefault(sid, []).append(i)
    short = {voice_of[s]: len(v) for s, v in rows.items() if len(v) < per}
    if short:
        raise ValueError("speakers with fewer than %d rows: %s" % (per, short))
    rng = random.Random(seed)
    kept = set()
    for sid in rows:                    # dict order = order of first appearance
        kept.update(rng.sample(rows[sid], per))
    return [lines[i] for i in sorted(kept)]


G_FROM, G_TO = "ᵻ", "ᵊ"     # our id 176 (stock's untrained "'") -> our 177 (stock's ᵻ)


def g_line(line):
    """An S1 row `wav22|spk|phonemes` -> (G′'s row, how many `ᵻ` it moved). Only the
    phonemes change. Asserted per row: the result equals `upstream_as_ours` of S1's
    phonemes, i.e. G′ encodes S1's text exactly as C0's text already is."""
    f = line.split("|")
    if len(f) != 3:
        raise ValueError("expected 3 fields wav|spk|phonemes, got %d: %r" % (len(f), line[:80]))
    moved = f[2].replace(G_FROM, G_TO)
    try:
        same = moved == upstream_as_ours(f[2])
    except ValueError as e:
        same, why = False, str(e)
    else:
        why = "differs"
    if not same:
        raise ValueError("G′ row is not upstream_as_ours of S1's (%s): %r" % (why, line[:80]))
    return "|".join(f[:2] + [moved]), f[2].count(G_FROM)


F1_SUBSETS = ("train-clean-100", "train-clean-360", "train-other-500")
F1_MIN_ROWS = 300
F1_MIN_SECONDS, F1_MAX_SECONDS = 1.0, 16.0      # S1's filter, inclusive at both ends


def libri_r_22k(wav24):
    """A LibriTTS-R train-clean-100 / -360 / other-500 recording -> its 22.05 kHz copy in
    `<subset>_22k/`. For train-clean-100 that is `to_22k`: S1's existing copies. Any other
    root (dev sets, the original LibriTTS, a copy) is refused."""
    for s in F1_SUBSETS:
        src = "%s/%s" % (LIBRI_R_ROOT, s)
        if wav24.startswith(src + "/"):
            return _rebase(wav24, src, src + "_22k")
    raise ValueError("not under %s/{%s}: %s" % (LIBRI_R_ROOT, ",".join(F1_SUBSETS), wav24))


def libri_r_24k(wav22):
    """The inverse of `libri_r_22k`: a `<subset>_22k/` copy -> its 24 kHz LibriTTS-R recording
    (the file the bench measures a clip's length on). Any other root is refused."""
    for s in F1_SUBSETS:
        src = "%s/%s_22k" % (LIBRI_R_ROOT, s)
        if wav22.startswith(src + "/"):
            return _rebase(wav22, src, "%s/%s" % (LIBRI_R_ROOT, s))
    raise ValueError("not under %s/{%s}_22k: %s" % (LIBRI_R_ROOT, ",".join(F1_SUBSETS), wav22))


def libritts_id(wav):
    """`.../<subset>/<speaker>/<chapter>/<speaker>_<chapter>_....wav` -> the speaker id. The
    directory must agree with the file name."""
    p = Path(wav)
    spk, chapter = p.parts[-3], p.parts[-2]
    if not p.name.startswith("%s_%s_" % (spk, chapter)):
        raise ValueError("file name does not match its speaker/chapter directories: %s" % wav)
    return int(spk)


def f1_length_ok(seconds):
    return F1_MIN_SECONDS <= seconds <= F1_MAX_SECONDS


def f1_dense_speakers(train_wavs, min_rows=F1_MIN_ROWS):
    """LibriTTS ids with at least `min_rows` of `train_wavs`, ascending."""
    n = Counter(libritts_id(w) for w in train_wavs)
    return sorted(s for s, k in n.items() if k >= min_rows)


def f1_table(ids, v7):
    """F1's speakers.json: indices in ascending integer LibriTTS id order, and each
    speaker's index in production v7 (`v7` = v7's `libritts_id_to_index`), which is how the
    bench names LibriTTS-R voices. `v7_index` deliberately does not end in `_id_to_index`:
    `make_warmstart._index_map` would read it as a second namespace."""
    ids = sorted(int(i) for i in ids)
    missing = [i for i in ids if str(i) not in v7]
    if missing:
        raise ValueError("speakers without a v7 index: %s" % missing)
    return {"n_spks": len(ids),
            "libritts_id_to_index": {str(s): i for i, s in enumerate(ids)},
            "v7_index": {str(s): v7[str(s)] for s in ids}}


def f1_line(line, index):
    """A source row `wav24|spk|phonemes|vat` -> F1's `wav22|f1 index|phonemes`. `index` is
    `f1_table(...)["libritts_id_to_index"]`; a speaker outside it raises KeyError."""
    f = line.split("|")
    if len(f) != 4:
        raise ValueError("expected 4 fields wav|spk|phonemes|vat, got %d: %r"
                         % (len(f), line[:80]))
    return "%s|%d|%s" % (libri_r_22k(f[0]), index[str(libritts_id(f[0]))], f[2])


def check_speaker_column(line, index_to_id):
    """A source row's speaker column must name the speaker its path does (`index_to_id`
    inverts that source's own `libritts_id_to_index`)."""
    wav, spk = line.split("|", 2)[:2]
    want = str(libritts_id(wav))
    got = index_to_id.get(int(spk))
    if got != want:
        raise ValueError("speaker column %s names LibriTTS %s; the path says %s: %s"
                         % (spk, got, want, wav))
