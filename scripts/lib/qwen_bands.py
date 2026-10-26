"""Qwen VoiceDesign's F0 admission bands: where the voices the owner heard in Phase 1S Step 1c-0
fell, per gender and zone, so a designed voice is admitted by its measured F0, not by its prompt.

Qwen does not take pitch from its instruct words (Notes:
Sonora/phase-1s-step1-preregistration.md, § 1c-0b result), so these bands, not a `pitch:` value,
say where a voice sits. Each band is the minimum and maximum F0 median and the maximum F0 excursion
(p90 - p10) of the clips the owner marked in that zone in `PROBE`'s `verdicts.csv`, read from its
`measures.jsonl` and split by its `manifest.jsonl` `gender` (the prompted one). The clips were
rendered at `RENDER_COMMIT` and measured by `render_qwen_1c0.py measure`: librosa's pYIN over
`F0_RANGE_HZ`. Compare only a voice measured the same way.

The values are the measured floats as `measures.jsonl` holds them, unrounded; the spec's table
rounds them to 0.1 Hz. Every bound is inclusive. `tests/test_qwen_bands.py` recomputes them from the
probe folder.

`VERDICT_ZONES_SHA256` fingerprints the verdicts the bands came from: the zone of every clip in
`verdicts.csv` (`verdict_zones_sha256` gives the canonical form). If the owner re-rates a clip,
the fingerprint no longer matches, and the bands must be recomputed before it is updated.

⚠ OUTSIDE EVERY BAND IS "unmeasured", NEVER "exclude". No clip was excluded for its pitch: the four
the owner excluded sit inside the core bands and were rejected for defects. The band edges mark
where Qwen rendered, not where usable voices end, so a voice beyond them is one this probe has no
verdict for: `admit` reports "unmeasured", and the owner's ear decides. A voice inside a band's F0
median range whose excursion exceeds the band's maximum is "unmeasured" too, as is one whose F0 is
not measured (None or NaN).

⚠ THE MALE NICHE BAND IS PROVISIONAL (`PROVISIONAL`). Its three clips are persona P3, the child boy,
and the owner heard all three as female or ambiguous. `admit` still reports "niche" for a male voice
in it; `provisional(gender, zone)` says so, and a caller must not cast a child boy from it without
the owner's review.

Standard library only.
"""
import csv
import hashlib
from collections import namedtuple

PROBE = "/data/model-training/datasets/probes/qwen-1c0"
RENDER_COMMIT = "c9e4de4"
F0_ESTIMATOR = "librosa pYIN (render_qwen_1c0.py measure)"
F0_RANGE_HZ = (50.0, 700.0)

GENDERS = ("Male", "Female")
ZONES = ("core", "niche")
UNMEASURED = "unmeasured"

# n: the clips in the zone. Hz throughout.
Band = namedtuple("Band", "n f0_median_min f0_median_max f0_excursion_max")

BANDS = {
    ("Male", "core"): Band(24, 77.55723809168673, 143.06459324352582, 111.13109552600812),
    ("Male", "niche"): Band(3, 265.4303484677136, 364.6889954232867, 207.34500937196702),
    ("Female", "core"): Band(25, 146.40856959456255, 257.8740616879158, 136.53405790294286),
    ("Female", "niche"): Band(4, 360.5001850443321, 388.61277646144237, 166.11356312958304),
}

# sha256 of the zones in PROBE's verdicts.csv that BANDS were computed from; see
# verdict_zones_sha256 for the canonical form.
VERDICT_ZONES_SHA256 = "b508c7b6483d38e278b557a04806bb328414a905cd36297e4b41f5110117edaa"

# (gender, zone) bands that must not cast unreviewed: Male niche is P3's child boy, heard as female
# or ambiguous.
PROVISIONAL = frozenset({("Male", "niche")})


def verdict_zones_sha256(path):
    """The sha256 (hex) of a verdicts.csv's zones. Canonical form: one line "<id>,<zone>" per CSV
    row, with the `id` and `zone` fields exactly as the CSV holds them; the lines sorted as
    Python strings, joined with "\\n" and no final newline, encoded as UTF-8. Every other column
    (scores, notes, timestamps) is left out, so only a changed zone, or an added, removed or
    renamed clip, changes the hash."""
    with open(path, encoding="utf-8", newline="") as fh:
        lines = sorted("%s,%s" % (r["id"], r["zone"]) for r in csv.DictReader(fh))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _gender(gender):
    if gender not in GENDERS:
        raise ValueError("gender must be one of %s, not %r" % (GENDERS, gender))
    return gender


def admit(gender, f0_median, f0_excursion):
    """"core" or "niche" when the F0 median (Hz) is inside that zone's band for `gender` and the F0
    excursion (Hz) is at most its maximum, bounds inclusive; otherwise "unmeasured": the owner's ear
    decides (module docstring). Never "exclude". Refuses a gender other than "Male" or "Female".

    The excursion has a maximum only, no minimum. A negative excursion (p90 below p10, which a
    measured clip cannot give) is therefore admitted like an excursion of 0; this function does
    not check how the excursion was computed."""
    _gender(gender)
    if f0_median is None or f0_excursion is None:
        return UNMEASURED
    for zone in ZONES:
        b = BANDS[(gender, zone)]
        if b.f0_median_min <= f0_median <= b.f0_median_max and f0_excursion <= b.f0_excursion_max:
            return zone
    return UNMEASURED


def provisional(gender, zone):
    """Whether `admit`'s `zone` for `gender` comes from a band that must not cast unreviewed."""
    return (_gender(gender), zone) in PROVISIONAL
