"""Pure helpers for the lineage hum bench (`scripts/tools/render_ear_lineage.py`).

THE QUESTION (2026-09-30). The baseline bench found stock `matcha_vctk` adds a little hum
over its vocoder (+0.75) and vat7 ep005 much more (+2.50; +1.75 per HNR-matched pair). Every
Sonora checkpoint descends from stock by warm-starts: derisk-energy -> vat3 -> vat5 -> vat6 ->
vat7. Where along that chain did the extra hum come in?

Each item pairs one checkpoint's render of a recording with that recording's round trip
through the 24 kHz vocoder, as in the baseline bench, so the within-item gap is the hum the
checkpoint adds. The same voices go through every checkpoint, each speaking a different
recording, and the reading compares each step voice by voice.

No torch here, so the host test suite covers it.
"""

import statistics as st

try:                                    # imported as `lib.lineage_bench` by the tool
    from .baseline_bench import ALPHA, sign_flip_p
except ImportError:                     # imported flat, with scripts/lib on the path
    from baseline_bench import ALPHA, sign_flip_p


def spread(pool, n):
    """`n` voices from `pool` {voice: HNR}, evenly spaced along HNR order.

    Spread, not random: the hum tracks HNR in every bench, and a random draw piles up at
    the median, where a step that matters for rough voices would barely show."""
    order = sorted(pool, key=lambda s: (pool[s], s))
    if len(order) < n or n < 2:
        raise SystemExit("REFUSING: %d voices for %d picks." % (len(order), n))
    return [order[round(i * (len(order) - 1) / (n - 1))] for i in range(n)]


def assign_clips(clips, ckpts, rng, target):
    """{voice: [(recording, seconds)]} -> {voice: {checkpoint: recording}}, distinct per voice.

    A different recording for each checkpoint, because a recording heard twice is judged
    against memory of the first time rather than against the clip beside it. They are the
    recordings nearest `target` seconds, so length cannot differ by checkpoint: more
    seconds is more exposure to the hum. Which checkpoint gets which is random."""
    out = {}
    for spk in sorted(clips):
        if len(clips[spk]) < len(ckpts):
            raise SystemExit("REFUSING: voice %s has %d usable recordings for %d "
                             "checkpoints." % (spk, len(clips[spk]), len(ckpts)))
        near = sorted(clips[spk], key=lambda c: (abs(c[1] - target), c[0]))[:len(ckpts)]
        out[spk] = dict(zip(ckpts, rng.sample(sorted(w for w, _s in near), len(ckpts))))
    return out


def side_swaps(spks, ckpts, rng):
    """{(voice, checkpoint): swapped}. Half the voices swapped, and each voice keeps its
    side along the whole chain: a listener's lean toward A or B then cancels exactly in
    every step and span difference, which are within-voice, and balances 4/4 within each
    checkpoint. (Alternating sides would add twice the lean as noise to every step.)"""
    order = list(spks)
    rng.shuffle(order)
    flip = rng.random() < 0.5
    return {(s, c): (i % 2 == 0) != flip for i, s in enumerate(order) for c in ckpts}


def _test(d):
    return (st.mean(d), sign_flip_p(d)) if d else (0.0, 1.0)


def reading(gaps, ckpts):
    """The pre-registered reading. `gaps` is {checkpoint: {voice: model - round trip}}.

    level    each checkpoint's mean gap and its sign-flip p against 0
    steps    (a, b, mean, p, n) for each adjacent pair: gap_b - gap_a over the voices rated
             at both, the change that step made
    span     the same from the first checkpoint to the last

    outcome  invalid  the last checkpoint does not reproduce its known hum (gap > 0, p < α)
             step     one or more steps rise significantly (`rises` names them)
             present_at_first  no step rises, first -> last is not shown to rise, and the
                      first checkpoint already hums. NOT "the first step caused it": the
                      step from stock is outside this bench, and stock itself hums (+0.75
                      in the baseline bench)
             gradual  no single step rises, but first -> last does: a creep
             inconclusive  otherwise

    Also reported, whatever the outcome: `falls` (steps that significantly lower the hum)
    and `present_at_first` (the first checkpoint's gap is > 0 at p < α).

    Steps are tested UNCORRECTED, four at α = 0.05 (about 19% chance of at least one false
    rise if nothing changed): a step found is where to look next, not a proof. With 8
    voices the smallest possible p is 2/256. "Not shown" is never "absent".
    """
    def paired(a, b):
        both = sorted(set(gaps[a]) & set(gaps[b]))
        d = [gaps[b][s] - gaps[a][s] for s in both]
        return (a, b) + _test(d) + (len(d),)

    level = {c: _test([gaps[c][s] for s in sorted(gaps[c])]) for c in ckpts}
    steps = [paired(a, b) for a, b in zip(ckpts, ckpts[1:])]
    span = paired(ckpts[0], ckpts[-1])
    rises = [(a, b) for a, b, m, p, _n in steps if m > 0 and p < ALPHA]
    falls = [(a, b) for a, b, m, p, _n in steps if m < 0 and p < ALPHA]
    last, first = level[ckpts[-1]], level[ckpts[0]]
    span_rises = span[2] > 0 and span[3] < ALPHA
    present_at_first = first[0] > 0 and first[1] < ALPHA
    if not (last[0] > 0 and last[1] < ALPHA):
        outcome = "invalid"
    elif rises:
        outcome = "step"
    elif present_at_first and not span_rises:
        outcome = "present_at_first"
    elif span_rises:
        outcome = "gradual"
    else:
        outcome = "inconclusive"
    return {"level": level, "steps": steps, "span": span, "rises": rises, "falls": falls,
            "present_at_first": present_at_first, "outcome": outcome}
