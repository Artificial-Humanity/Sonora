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

The hum-fix batch (Notes: Sonora/hum-fix-batch-preregistration.md, owner 2026-10-12) adds
checks A, B and C as checks 5-7, each a candidate cause and a candidate fix:

  check 5 (A)  C0, D1     clips per speaker: does VCTK hum with 105 clips per speaker?
  check 6 (B)  S1, G      the `ᵻ` symbol: does S1 hum less with `ᵻ` on stock's trained id?
                          (arm G is the registration's G′)
  check 7 (C)  S1 | F1    dense speakers: does S1's recipe hum less on LibriTTS-R speakers
                          with at least 300 clips each? (HNR- and F0-matched pairs)

A UNIT is one HNR-matched (VCTK voice, LibriTTS-R voice) pair in checks 1-2, one LibriTTS-R
voice in checks 3-4 and 6, one VCTK voice in check 5, and one (S1 voice, F1 voice) pair in
check 7. An arm's GAP on a unit is its render's hum severity minus its own vocoder round trip
of the same real recording.

⚠ THE TEST IS ONE-SIDED, as registered. `baseline_bench.sign_flip_p` is two-sided and is
not used here.

No torch here, so the host test suite covers it.
"""

import itertools
import math
import re
import statistics as st
from fractions import Fraction
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
    # the hum-fix batch: A, B, C, and follow-up D (check 8, arm E)
    5: {"arms": ("C0", "D1"), "vctk_arms": ("C0", "D1"),
        "libri_arms": (),
        "contrasts": (("D1", "C0"),),
        "floor_ref": "C0", "min_units": 16},
    6: {"arms": ("S1", "G"), "vctk_arms": (),
        "libri_arms": ("S1", "G"),
        "contrasts": (("S1", "G"),),
        "floor_ref": "S1", "min_units": 16},
    7: {"arms": ("S1", "F1"), "vctk_arms": (),
        "libri_arms": ("S1", "F1"),
        "contrasts": (("S1", "F1"),),
        "floor_ref": "S1", "min_units": 16},
    8: {"arms": ("S1", "E"), "vctk_arms": (),
        "libri_arms": ("S1", "E"),
        "contrasts": (("S1", "E"),),
        "floor_ref": "S1", "min_units": 16},
}
HUM_FIX_CHECKS = (5, 6, 7, 8)
# Check 4 is registered against ONE S1 checkpoint: the step that ended its tenth epoch.
# `select --check 4` (dry run included) refuses any other.
CHECK4_S1_SHA256 = "857f62d7ac47770d94f971cf4bae80d3078e42fd028d45465fe30bd0e2a0d0d5"
# The hum-fix batch's controls are not retrained: the staged rebuild's final C0 and S1,
# registered byte for byte. `select` (dry run included) and `render` refuse any other.
CONTROL_SHA256 = {"C0": "d0d6b49b2e92a59e1caf43b82f0359c4c0940d7960b428caadee1091ab24b879",
                  "S1": CHECK4_S1_SHA256}
# The hum-fix batch's registered selection inputs: `--hnr-json` (speaker_hnr_all.json) and
# `--vctk-hnr-from` (baseline_bench/speakers.json) and `--s1-hnr-json`
# (speaker_hnr_s1_rest.json). Checks 5-8 refuse any other.
INPUT_SHA256 = {"hnr_json": "b2f15110082e9abcc4f74454f3fb6ba1f493604f0937b956483865190abddab2",
                "vctk_hnr_from": "3bd8e2a61cdebd56dff96c7746b5418d6ee204f17b485b05fe7fb9083bd1be9f",
                "s1_hnr_json": "93a1d91eab6905a4652991c500841a509a978008a16a13c9fee7f311d620d27c"}
F1_MAX_HNR_GAP = Fraction(1, 2)     # check 7: an S1 voice and its F1 voice, dB, inclusive
# Each arm's audio path: 22.05 kHz + universal vocoder (stock's), or 24 kHz + ours. L is on
# S1's path, but its round trip is of the ORIGINAL recording's 22.05 kHz copy, not
# LibriTTS-R's, so it is a family of its own. D1 is on C0's path, G and E on S1's; F1
# round-trips its own 22.05 kHz copies, which for train-clean-360 / other-500 are not S1's.
FAMILY = {"stock": "vctk22", "C0": "vctk22", "S1": "libri22", "L": "orig22",
          "S2": "libri24", "R": "libri24", "derisk": "libri24",
          "D1": "vctk22", "G": "libri22", "F1": "f1_22", "E": "libri22"}
_TEN = re.compile(r"^checkpoint_epoch=009_step=\d{7}\.ckpt$")
# An arm whose final checkpoint is NOT the step that ended its tenth epoch, by exact file name
# (Notes: Sonora/hum-fix-batch-preregistration.md). Every other arm's final is that step
# (`ten_epoch_ckpt`). D1 is matched to S1 in optimizer steps: 26 epochs x 355 = 9,230. G and E
# (S1's 921 steps per epoch) and F1 (978) are named exactly too.
FINAL_CKPT = {"D1": "checkpoint_epoch=025_step=0009229.ckpt",
              "G": "checkpoint_epoch=009_step=0009209.ckpt",
              "F1": "checkpoint_epoch=009_step=0009779.ckpt",
              "E": "checkpoint_epoch=009_step=0009209.ckpt"}
_EPOCH = re.compile(r"^checkpoint_epoch=(\d{3})_step=\d{7}\.ckpt$")


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
    before check 2 has a key, so its speakers file must exclude its voices by itself. A check
    7 unit's F1 voice is its `f1` entry's `libri` (a v7 index, like every other)."""
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
            f1 = v.get("f1")
            if isinstance(f1, dict) and isinstance(f1.get("libri"), int):
                out.add(f1["libri"])
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


def final_ckpt(arm, path):
    """Is this `arm`'s registered final checkpoint? Its `FINAL_CKPT` name exactly, or, for an
    arm not registered there, the step checkpoint that ended the tenth epoch."""
    want = FINAL_CKPT.get(arm)
    return Path(path).name == want if want else ten_epoch_ckpt(path)


def final_epochs(arm):
    """How many epochs `arm`'s final checkpoint ends (for messages)."""
    want = FINAL_CKPT.get(arm)
    return int(_EPOCH.match(want).group(1)) + 1 if want else 10


def check_registered_ckpts(check, digests):
    """Raise ValueError unless a hum-fix check's controls are the registered checkpoints
    (`CONTROL_SHA256`) and its new arm's checkpoint (D1, G, F1 or E) is not a copy of a
    control's (C0's or S1's): a copy would load, pass every statistics check and compare a
    control with itself. `digests` is {arm: sha256 or None}; None is an arm not trained yet
    (a dry run). Checks 1-4 pass."""
    if check not in HUM_FIX_CHECKS:
        return
    for arm, want in sorted(CONTROL_SHA256.items()):
        if arm in digests and digests[arm] != want:
            raise ValueError("check %d's %s checkpoint is not the registered one (sha256 %s)."
                             % (check, arm, want))
    for arm in ("D1", "G", "F1", "E"):
        for ctrl, want in sorted(CONTROL_SHA256.items()):
            if digests.get(arm) is not None and digests[arm] == want:
                raise ValueError("check %d's %s checkpoint is %s's (sha256 %s)."
                                 % (check, arm, ctrl, want))


def _exact(x):
    """A JSON number as written (3 decimals), exactly: as floats 5.252 - 4.752 is
    0.5000000000000004 and would fall outside an inclusive 0.5 dB window."""
    return Fraction(repr(x))


def pair_f1(s1, f1, max_gap=F1_MAX_HNR_GAP):
    """Check 7's pairs, as registered. `s1` is {S1 voice: (HNR, F0)}, `f1` is {F1 voice:
    (HNR, F0, LibriTTS id)}; voices are v7 indices. In ascending (HNR, S1 index) order, each
    S1 voice takes the unused F1 voice whose HNR differs by at most `max_gap` dB (inclusive,
    exact) that minimises |ln F0_S1 - ln F0_F1|; ties go to the smaller HNR gap, then the lower
    LibriTTS id. An S1 voice with no candidate is skipped, not replaced. -> [(S1, F1)]."""
    used, pairs = set(), []
    for s in sorted(s1, key=lambda v: (s1[v][0], v)):
        h, f0 = s1[s]
        best = None
        for v, (fh, ff0, lid) in f1.items():
            gap = abs(_exact(h) - _exact(fh))
            if v in used or gap > max_gap:
                continue
            k = (abs(math.log(f0) - math.log(ff0)), gap, lid)
            if best is None or k < best[0]:
                best = (k, v)
        if best is not None:
            used.add(best[1])
            pairs.append((s, best[1]))
    return pairs


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
    check 5  invalid  floors not comparable (D1's within 1 point of C0's, both under 4), or
                      fewer than 16 units rated in both. No within-check positive control
             shown ["clips_per_speaker"]  D1 - C0 > 0         not_shown  otherwise
             Three p-values (D1 - C0, D1's gap, C0's gap), one deciding: `n_deciding` 1.
    check 6  invalid  as check 4, with G in L's place
             shown ["symbol"]  S1 - G > 0                     not_shown  otherwise
    check 7  invalid  as check 4, with F1 in L's place
             shown ["dense_speakers"]  S1 - F1 > 0            not_shown  otherwise
    check 8  invalid  as check 4, with E in L's place
             shown ["text_frontend"]  S1 - E > 0              not_shown  otherwise
             Checks 6-8: three p-values, two deciding (S1's gap and the contrast).

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
    elif check == 4:
        valid = fl and up(level["S1"]) and contrasts["S1-L"][2] >= spec["min_units"]
        if up(contrasts["S1-L"]):
            shown.append("restoration")
        outcome = "shown" if shown else "not_shown"
        extra["n_deciding"] = 2
    elif check == 5:
        valid = fl and contrasts["D1-C0"][2] >= spec["min_units"]
        if up(contrasts["D1-C0"]):
            shown.append("clips_per_speaker")
        outcome = "shown" if shown else "not_shown"
        extra["n_deciding"] = 1
    else:
        fix = spec["contrasts"][0][1]
        key = "S1-%s" % fix
        valid = fl and up(level["S1"]) and contrasts[key][2] >= spec["min_units"]
        if up(contrasts[key]):
            shown.append({6: "symbol", 7: "dense_speakers", 8: "text_frontend"}[check])
        outcome = "shown" if shown else "not_shown"
        extra["n_deciding"] = 2
    if not valid:
        outcome, shown = "invalid", []
    # n_tests: the contrasts, plus S1's own gap in check 3 (the validity alternative) and in
    # checks 4-8 both arms' own gaps (each reported with its p).
    return dict({"level": level, "contrasts": contrasts, "floors_ok": fl,
                 "valid": bool(valid), "outcome": outcome, "shown": shown,
                 "n_tests": len(contrasts) + {3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}.get(check, 0)},
                **extra)

