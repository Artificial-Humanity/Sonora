"""Pure helpers for the staged-rebuild hum bench (`scripts/tools/render_ear_staged.py`).

Pre-registered in Notes: Sonora/staged-rebuild-preregistration.md (owner, 2026-10-06). The
first fine-tune, stock -> derisk-energy, changed five things at once. Arms rebuilt from
stock separate them; three blind checks of 20 units x 3 arms read them, and a fourth
(2026-10-11, after checks 2-3 named the data) splits the data:

  check 1  stock | R, derisk         the gate: does 10 epochs of the derisk recipe hum?
  check 2  stock, C0 | S1            does fine-tuning, or the new data, add the hum?
  check 3  S1, S2, R                 does the 24 kHz path, or the conditioning, add it?
  check 4  S1, L                     does LibriTTS-R's restoration add it? L is S1 trained
                                     on the ORIGINAL LibriTTS recordings of the same rows

A UNIT is one HNR-matched (VCTK voice, LibriTTS-R voice) pair in checks 1-2 and one
LibriTTS-R voice in checks 3-4. An arm's GAP on a unit is its render's hum severity minus its
own vocoder round trip of the same real recording.

⚠ THE TEST IS ONE-SIDED, as registered. `baseline_bench.sign_flip_p` is two-sided and is
not used here.

No torch here, so the host test suite covers it.
"""

import itertools
import re
import statistics as st
from pathlib import Path

try:                                    # imported as `lib.staged_bench` by the tools
    from .baseline_bench import ALPHA
except ImportError:                     # imported flat, with scripts/lib on the path
    from baseline_bench import ALPHA

# floor_ref: the arm every other arm's round-trip floor must be within 1 point of (None: the
# check has no floor rule). min_units (check 4): fewer units rated in both arms is INVALID,
# and fewer eligible voices at selection is not run.
CHECKS = {
    1: {"arms": ("stock", "R", "derisk"), "vctk_arms": ("stock",),
        "libri_arms": ("R", "derisk"),
        "contrasts": (("derisk", "stock"), ("R", "stock"), ("R", "derisk")),
        "floor_ref": "stock"},
    2: {"arms": ("stock", "C0", "S1"), "vctk_arms": ("stock", "C0"),
        "libri_arms": ("S1",),
        "contrasts": (("C0", "stock"), ("S1", "stock")),
        "floor_ref": "stock"},
    3: {"arms": ("S1", "S2", "R"), "vctk_arms": (),
        "libri_arms": ("S1", "S2", "R"),
        "contrasts": (("R", "S1"), ("S2", "S1"), ("R", "S2")),
        "floor_ref": None},
    4: {"arms": ("S1", "L"), "vctk_arms": (),
        "libri_arms": ("S1", "L"),
        "contrasts": (("S1", "L"),),
        "floor_ref": "S1", "min_units": 16},
}
# Each arm's audio path: 22.05 kHz + universal vocoder (stock's), or 24 kHz + ours. L is on
# S1's path, but its round trip is of the ORIGINAL recording's 22.05 kHz copy, not
# LibriTTS-R's, so it is a family of its own.
FAMILY = {"stock": "vctk22", "C0": "vctk22", "S1": "libri22", "L": "orig22",
          "S2": "libri24", "R": "libri24", "derisk": "libri24"}
_TEN = re.compile(r"^checkpoint_epoch=009_step=\d{7}\.ckpt$")


def one_sided_p(d):
    """Exact one-sided sign-flip p that the mean of paired differences `d` is above 0:
    the share of the 2^n sign assignments whose sum is at least the observed one."""
    if len(d) > 20:
        raise ValueError("exact enumeration is for n <= 20; got %d" % len(d))
    obs = sum(d)
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        total += 1
        hits += sum(s * x for s, x in zip(signs, d)) >= obs - 1e-9
    return hits / total


def test(d):
    return (st.mean(d), one_sided_p(d), len(d)) if d else (0.0, 1.0, 0)


def heard_libri(docs):
    """Every LibriTTS-R speaker index named under `items` in parsed bench files: keys store
    one under `spk`, speakers files under `libri`. Check 3 is selected right after check 2,
    before check 2 has a key, so its speakers file must exclude its voices by itself."""
    out = set()
    for d in docs:
        items = d.get("items") if isinstance(d, dict) else None
        vals = items.values() if isinstance(items, dict) else (items or [])
        for v in vals:
            if not isinstance(v, dict):
                continue
            if str(v.get("spk", "")).isdigit():
                out.add(int(v["spk"]))
            if isinstance(v.get("libri"), int):
                out.add(v["libri"])
    return out


def side_plan(n, arms, rng):
    """{unit: {arm: model on A}}. Every item of a unit is served the same way round, half
    the units each way, so a listener's lean toward A or B cancels in every contrast."""
    flips = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(flips)
    return {u: {a: flips[u] for a in arms} for u in range(n)}


def floors_ok(floors, arms, ref="stock"):
    """Round-trip floors comparable: each arm's within 1 point of `ref`'s (strictly less than
    1), all under 4 (strictly). Pass `fractions.Fraction` floors: a float mean decides
    exact-1.0 gaps by rounding."""
    others = [a for a in arms if a != ref]
    return all(abs(floors[a] - floors[ref]) < 1.0 for a in others) \
        and max(floors[a] for a in arms) < 4.0


def split_contrast(model, rt, a, b):
    """a - b on the model ratings alone and on the round-trip ratings alone, each over the
    units rated in both arms: {"model": (mean, n), "round_trip": (mean, n)}. DESCRIPTIVE, not
    tested: it shows whether a difference in gaps comes from the renders or from a floor.
    `model` and `rt` are {arm: {unit: severity}}."""
    out = {}
    for name, sev in (("model", model), ("round_trip", rt)):
        both = sorted(set(sev[a]) & set(sev[b]))
        d = [sev[a][u] - sev[b][u] for u in both]
        out[name] = (st.mean(d) if d else 0.0, len(d))
    return out


def ten_epoch_ckpt(path):
    """Is this the step checkpoint written at the end of the tenth epoch (epoch index 9)?"""
    return bool(_TEN.match(Path(path).name))


def reading(check, gaps, floors):
    """The pre-registered reading of one check.

    gaps    {arm: {unit: model - round trip}}
    floors  {arm: mean round-trip severity}

    check 1  invalid  derisk - stock not > 0 at p < α, or floors not comparable
             go       R - stock > 0 at p < α           stop  otherwise
    check 2  invalid  floors not comparable
             shown ["fine_tuning"]  C0 - stock > 0
             shown ["data"]         S1 - stock > 0 and C0 - stock not shown
    check 3  invalid  neither R - S1 > 0 nor S1's own gap > 0, at p < α
             shown ⊆ ["audio_path" (S2 - S1 > 0), "conditioning" (R - S2 > 0)]
    check 4  invalid  S1's own gap not > 0 at p < α, or floors not comparable (L's within
                      1 point of S1's, both under 4), or fewer than 16 units rated in both
             shown ["restoration"]  S1 - L > 0      not_shown  otherwise
             Three p-values reported (S1's gap, S1 - L, L's gap), two of them deciding
             (S1's gap and S1 - L): `n_tests` 3, `n_deciding` 2.

    Every contrast is reported whatever the outcome, uncorrected; `n_tests` counts them.
    "Not shown" is never "absent".
    """
    spec = CHECKS[check]
    level = {a: test([gaps[a][u] for u in sorted(gaps[a])]) for a in spec["arms"]}
    contrasts = {}
    for a, b in spec["contrasts"]:
        both = sorted(set(gaps[a]) & set(gaps[b]))
        contrasts["%s-%s" % (a, b)] = test([gaps[a][u] - gaps[b][u] for u in both])

    def up(t):
        return t[0] > 0 and t[1] < ALPHA

    ref = spec["floor_ref"]
    fl = floors_ok(floors, spec["arms"], ref) if ref else None
    shown, extra = [], {}
    if check == 1:
        valid = fl and up(contrasts["derisk-stock"])
        outcome = "go" if up(contrasts["R-stock"]) else "stop"
    elif check == 2:
        valid = fl
        if up(contrasts["C0-stock"]):
            shown.append("fine_tuning")
        elif up(contrasts["S1-stock"]):
            shown.append("data")
        outcome = "shown" if shown else "not_shown"
    elif check == 3:
        valid = up(contrasts["R-S1"]) or up(level["S1"])
        if up(contrasts["S2-S1"]):
            shown.append("audio_path")
        if up(contrasts["R-S2"]):
            shown.append("conditioning")
        outcome = "shown" if shown else "not_shown"
    else:
        valid = fl and up(level["S1"]) and contrasts["S1-L"][2] >= spec["min_units"]
        if up(contrasts["S1-L"]):
            shown.append("restoration")
        outcome = "shown" if shown else "not_shown"
        extra["n_deciding"] = 2
    if not valid:
        outcome, shown = "invalid", []
    # n_tests: the contrasts, plus S1's own gap in check 3 (the validity alternative) and in
    # check 4 S1's and L's own gaps (both reported with their p).
    return dict({"level": level, "contrasts": contrasts, "floors_ok": fl,
                 "valid": bool(valid), "outcome": outcome, "shown": shown,
                 "n_tests": len(contrasts) + {3: 1, 4: 2}.get(check, 0)}, **extra)
