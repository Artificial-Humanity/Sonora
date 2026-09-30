"""Pure helpers for the baseline hum bench (`scripts/tools/render_ear_baseline.py`).

THE QUESTION (2026-09-30). Every Sonora checkpoint descends by warm-starts from stock
`matcha_vctk`. Does that stock model hum as much as ours, at the same voice roughness? If
it does, the defect is Matcha's recipe and our training history is not the cause. If it
does not, our history introduced it.

Each item pairs a model render with the round trip of the same recording through the same
vocoder, so the within-item difference is "hum the model adds beyond its vocoder". Stock
items and ours are MATCHED ON SPEAKER HNR, one to one, because severity tracks HNR in every
bench so far: an unmatched comparison would measure the two corpora's roughness mix.

No torch here, so the host test suite covers it.
"""

import importlib.util
import itertools
from pathlib import Path

_SYMBOLS_PY = Path(__file__).resolve().parents[2] / "matcha" / "text" / "symbols.py"

# sha256 of "".join(symbols) in upstream Matcha-TTS `matcha/text/symbols.py`. The derivation
# below only checks the last two slots; this pins all 178, so an edit to any earlier symbol
# cannot silently misencode the stock model's input.
UPSTREAM_SYMBOLS_SHA256 = "3e81afeec2d0906de3d7acf2214d32fbc066be8218d2edafe355255391ea92f7"


def ours_symbols():
    """Sonora's symbol table, read from source without importing the `matcha` package."""
    spec = importlib.util.spec_from_file_location("_sonora_symbols", _SYMBOLS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.symbols)


def original_matcha_symbols():
    """Upstream Matcha-TTS's table, which `matcha_vctk` was trained against.

    ⚠ THEY DIFFER IN THE LAST TWO SLOTS. Upstream's IPA string carries a second "'" before
    "ᵻ"; Sonora replaced that duplicate with "ᵊ". Same 178 symbols by count, but through our
    table the stock model would read "'" and "ᵻ" as the wrong embeddings.
    """
    ours = ours_symbols()
    if ours[176:] != ["ᵻ", "ᵊ"] or len(ours) != 178:
        raise SystemExit("REFUSING: matcha/text/symbols.py no longer ends in ['ᵻ', 'ᵊ'] "
                         "(got %r of %d). The upstream table is derived from ours, and "
                         "that derivation has to be re-checked against upstream."
                         % (ours[176:], len(ours)))
    return ours[:176] + ["'", "ᵻ"]


def encode_original(text):
    """Phoneme text -> ids as upstream `text_to_sequence(text, [])` produced them.

    Upstream builds `{s: i for i, s in enumerate(symbols)}`, so the duplicated "'" maps to
    its LATER slot. That is reproduced, not corrected: it is what the checkpoint learned.
    """
    table = {s: i for i, s in enumerate(original_matcha_symbols())}
    bad = sorted({c for c in text if c not in table})
    if bad:
        raise ValueError("symbols not in upstream Matcha's table: %s" % bad)
    return [table[c] for c in text]


def prior_speakers(docs):
    """Every speaker id named under `items` in the given parsed bench files."""
    out = set()
    for d in docs:
        items = d.get("items") if isinstance(d, dict) else None
        vals = items.values() if isinstance(items, dict) else (items or [])
        for v in vals:
            # A VCTK name ("p326") is not a LibriTTS-R index and excludes nothing.
            if isinstance(v, dict) and str(v.get("spk", "")).isdigit():
                out.add(int(v["spk"]))
    return out


def match_speakers(vctk, libri, n, exclude, max_gap):
    """`n` VCTK speakers spread over HNR, each with the nearest unused LibriTTS-R speaker.

    `vctk` maps speaker -> HNR, `libri` maps speaker id -> HNR (only eligible speakers).
    Spread, not random: a random draw piles up at the median, and the question is whether
    the difference holds for ROUGH voices too. Refuses a match further than `max_gap` dB.

    The spread runs over the HNR range BOTH corpora cover (widened by `max_gap`): a VCTK
    voice rougher or cleaner than every candidate has no match to be refused over.
    """
    pool = [v for s, v in libri.items() if s not in exclude]
    if not pool:
        raise SystemExit("REFUSING: no LibriTTS-R speaker is left after the exclusion.")
    lo, hi = min(pool) - max_gap, max(pool) + max_gap
    order = sorted((s for s in vctk if lo <= vctk[s] <= hi), key=lambda s: (vctk[s], s))
    if len(order) < n or n < 2:
        raise SystemExit("REFUSING: %d VCTK speakers for %d picks." % (len(order), n))
    picks = [order[round(i * (len(order) - 1) / (n - 1))] for i in range(n)]
    used, out = set(exclude), []
    for v in picks:
        cand = [s for s in libri if s not in used]
        if not cand:
            raise SystemExit("REFUSING: no LibriTTS-R speaker left to match %s." % v)
        best = min(cand, key=lambda s: (abs(libri[s] - vctk[v]), s))
        gap = abs(libri[best] - vctk[v])
        if gap > max_gap:
            raise SystemExit("REFUSING: %s (HNR %.2f) is %.2f dB from its nearest unused "
                             "match, over --max-gap %.2f." % (v, vctk[v], gap, max_gap))
        used.add(best)
        out.append({"vctk": v, "vctk_hnr": vctk[v], "libri": best,
                    "libri_hnr": libri[best]})
    return out


def sign_flip_p(d):
    """Exact two-sided sign-flip p for the mean of paired differences `d`."""
    if len(d) > 20:
        raise ValueError("exact enumeration is for n <= 20; got %d" % len(d))
    obs = abs(sum(d))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        total += 1
        hits += abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-9
    return hits / total


ALPHA = 0.05


def outcome(ours, stock, matched, rt_means=None):
    """The pre-registered reading. The first three are `(mean difference, p)`.

    ours      our model over our round trip. Must be positive and significant, or the bench
              failed to reproduce the known effect and says nothing ("invalid").
    stock     stock model over its round trip: does stock Matcha add hum?
    matched   (ours gap - stock gap) over HNR-matched pairs: does ours add MORE?
    rt_means  (stock round-trip mean, ours round-trip mean). The two vocoders differ, and
              on a 0-5 scale a gap measured from a higher floor has less room. So the
              matched contrast is read only when the floors are within a point of each
              other and both are under 4; otherwise it counts as not shown.

    A non-significant contrast is "not shown", never "absent": n = 12 cannot show
    equivalence, and the readings in `unblind_ear_baseline.py` say so.
    """
    if not (ours[0] > 0 and ours[1] < ALPHA):
        return "invalid"
    floors_ok = rt_means is None or (abs(rt_means[0] - rt_means[1]) < 1.0
                                     and max(rt_means) < 4.0)
    stock_hums = stock[0] > 0 and stock[1] < ALPHA
    ours_more = floors_ok and matched[0] > 0 and matched[1] < ALPHA
    if stock_hums and ours_more:
        return "both"
    if stock_hums:
        return "matcha"
    if ours_more:
        return "lineage"
    return "inconclusive"


def balanced_flips(n, rng):
    """Per match, (stock side swapped, ours side swapped): half of each family swapped, and
    the two items of a match always on OPPOSITE sides, so a listener's lean toward A or B
    cancels within each family and within each matched pair."""
    base = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(base)
    return [(b, not b) for b in base]


def closest_duration(cands, target):
    """The name in `cands` [(name, seconds)] nearest `target` seconds; ties by name."""
    return min(cands, key=lambda c: (abs(c[1] - target), c[0]))[0]
