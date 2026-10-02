"""Pure helpers for the first-step hum bench (`scripts/tools/render_ear_first_step.py`).

THE QUESTION (2026-10-02). The baseline bench found vat7 ep005 adds +1.75 more hum than stock
`matcha_vctk` per HNR-matched pair. The lineage bench could not place that excess along our
chain (INVALID: vat7 p = 0.094 with 8 voices), but its gap looked flat from derisk-energy
to vat7, which points at the one step it did not test: stock -> derisk-energy. Does
derisk-energy already hum more than stock at the same voice roughness?

Each match is one VCTK voice and one LibriTTS-R voice of nearly the same HNR, with three
items: stock on the VCTK voice, and derisk-energy and vat7 on the LibriTTS-R voice (two
different recordings). Each item pairs a model render with the round trip of the same
recording through the same vocoder, as in the baseline bench. vat7 over stock repeats the
baseline bench's result and is the check that this bench can hear the excess at all.

No torch here, so the host test suite covers it.
"""

import statistics as st

try:                                    # imported as `lib.first_step_bench` by the tool
    from .baseline_bench import ALPHA, sign_flip_p
except ImportError:                     # imported flat, with scripts/lib on the path
    from baseline_bench import ALPHA, sign_flip_p

FAMILIES = ("stock", "derisk", "vat7")
# The checks the pre-registration names, in the order they are reported.
CONTRASTS = (("vat7", "stock"), ("derisk", "stock"), ("vat7", "derisk"))


def heard_vctk(docs):
    """Every VCTK voice named under `items` in the given parsed bench files.

    Keys store one under `vctk_spk` and speakers files under `vctk`. (`prior_speakers`
    reads only LibriTTS-R indices, so it cannot exclude these.)"""
    out = set()
    for d in docs:
        items = d.get("items") if isinstance(d, dict) else None
        vals = items.values() if isinstance(items, dict) else (items or [])
        for v in vals:
            if not isinstance(v, dict):
                continue
            for k in ("vctk_spk", "vctk"):
                if isinstance(v.get(k), str) and v[k]:
                    out.add(v[k])
    return out


def side_plan(n, rng):
    """{match: {family: model on A}}. Every item of a match is served the same way round, and
    half the matches each way: a listener's lean toward A or B then cancels exactly in every
    contrast, which is within a match, and balances within each family. (The baseline bench
    put a match's two items on OPPOSITE sides, which adds twice the lean to each matched
    difference as noise.)"""
    flips = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(flips)
    return {m: {f: flips[m] for f in FAMILIES} for m in range(n)}


def _test(d):
    return (st.mean(d), sign_flip_p(d), len(d)) if d else (0.0, 1.0, 0)


def reading(gaps, floors):
    """The pre-registered reading.

    gaps     {family: {match: model - round trip}}
    floors   {family: mean round-trip severity}. stock's vocoder differs from ours, and on a
             0-5 scale a gap measured from a higher floor has less room, so a contrast with
             stock is read only when each of our floors is within 1 point of stock's and all
             are under 4 (the baseline bench's rule). derisk and vat7 share a vocoder.

    level      each family's mean gap, p and n
    contrasts  {"a-b": (mean, p, n)} of gap_a - gap_b over the matches rated in both

    outcome  invalid      vat7 - stock is not > 0 at p < α, or the floors are not comparable:
                          the bench did not reproduce the known excess
             first_step   derisk - stock > 0 at p < α, and vat7 - derisk is not
             both_steps   derisk - stock and vat7 - derisk both > 0 at p < α
             later        vat7 - derisk > 0 at p < α, and derisk - stock is not
             inconclusive otherwise

    Also reported: `vat7_below_derisk` (vat7 - derisk < 0 at p < α). Three tests, two of
    them read for the outcome, UNCORRECTED. "Not shown" is never "absent".
    """
    level = {f: _test([gaps[f][m] for m in sorted(gaps[f])]) for f in FAMILIES}
    contrasts = {}
    for a, b in CONTRASTS:
        both = sorted(set(gaps[a]) & set(gaps[b]))
        contrasts["%s-%s" % (a, b)] = _test([gaps[a][m] - gaps[b][m] for m in both])
    floors_ok = all(abs(floors[f] - floors["stock"]) < 1.0 for f in ("derisk", "vat7")) \
        and max(floors.values()) < 4.0

    def up(k):
        m, p, _n = contrasts[k]
        return m > 0 and p < ALPHA

    first, later = up("derisk-stock"), up("vat7-derisk")
    if not (floors_ok and up("vat7-stock")):
        outcome = "invalid"
    elif first and later:
        outcome = "both_steps"
    elif first:
        outcome = "first_step"
    elif later:
        outcome = "later"
    else:
        outcome = "inconclusive"
    m, p, _n = contrasts["vat7-derisk"]
    return {"level": level, "contrasts": contrasts, "floors_ok": floors_ok,
            "vat7_below_derisk": m < 0 and p < ALPHA, "outcome": outcome}
