"""Qwen VoiceDesign's F0 admission bands are the ones the owner's 1c-0 verdicts and the probe's
measures give, and `admit` places a voice in a band or says it is unmeasured, never excluded."""
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib

import pytest
from scripts_layout import SCRIPTS  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS.on_path()
import qwen_bands as qb  # noqa: E402

PROBE = pathlib.Path(qb.PROBE)
needs_probe = pytest.mark.skipif(
    not all((PROBE / n).is_file() for n in ("verdicts.csv", "measures.jsonl", "manifest.jsonl",
                                             "render_meta.json", "measure_meta.json")),
    reason="the 1c-0 probe folder is not on this machine — the bands are not recomputed, and "
           "that is NOT a pass")


def _jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, fh))}


def _assert_verdict_zones(path):
    """The zones in `path` are the ones BANDS was computed from."""
    got = qb.verdict_zones_sha256(path)
    assert got == qb.VERDICT_ZONES_SHA256, (
        "%s was re-rated: its zones hash to %s, not the VERDICT_ZONES_SHA256 the bands were "
        "computed from (%s). Recompute BANDS from the probe folder, then update "
        "VERDICT_ZONES_SHA256." % (path, got, qb.VERDICT_ZONES_SHA256))


def _probe():
    """(verdicts, measures, manifest), each {id: row}. The zone fingerprint is asserted first, so
    a re-rated verdicts.csv fails with that message, not with a band mismatch."""
    _assert_verdict_zones(PROBE / "verdicts.csv")
    with open(PROBE / "verdicts.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    verdicts = {r["id"]: r for r in rows}
    assert len(verdicts) == len(rows), "verdicts.csv rates a clip twice"
    return verdicts, _jsonl(PROBE / "measures.jsonl"), _jsonl(PROBE / "manifest.jsonl")


# --- the constants are the probe's -----------------------------------------------------------

@needs_probe
def test_every_band_is_the_probe_verdicts_measured():
    """Per gender and zone: the clip count, the min and max F0 median and the max F0 excursion of
    the clips marked in that zone, exactly as measures.jsonl holds them."""
    verdicts, measures, manifest = _probe()
    assert set(verdicts) == set(manifest) == set(measures) and len(manifest) == 60
    zones = {}
    for cid, v in verdicts.items():
        zones.setdefault((manifest[cid]["gender"], v["zone"]), []).append(measures[cid])
    got = {}
    for key in itertools.product(qb.GENDERS, qb.ZONES):
        rows = zones[key]
        assert all(r["f0_median"] is not None and r["f0_excursion"] is not None for r in rows), key
        got[key] = qb.Band(len(rows), min(r["f0_median"] for r in rows),
                           max(r["f0_median"] for r in rows), max(r["f0_excursion"] for r in rows))
    assert got == qb.BANDS
    # The population that matters: every core and niche clip, 49 + 7 of the 60.
    assert sum(b.n for b in got.values()) == 56


@needs_probe
def test_a_rerated_verdict_fails_the_zone_fingerprint(tmp_path):
    """Mutation check on a copy of verdicts.csv (the real file is only read): one zone changed
    makes the fingerprint assertion fire, with the re-rating message."""
    _assert_verdict_zones(PROBE / "verdicts.csv")                   # positive control
    with open(PROBE / "verdicts.csv", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["zone"] = "niche" if rows[0]["zone"] != "niche" else "core"
    copy = tmp_path / "verdicts.csv"
    with open(copy, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(AssertionError, match="re-rated.*Recompute BANDS"):
        _assert_verdict_zones(copy)


def test_the_zone_fingerprint_reads_only_id_and_zone(tmp_path):
    """The canonical form: sorted "<id>,<zone>" lines, "\\n"-joined, no final newline. Row order
    and the other columns do not change it; a zone does."""
    def write(name, rows):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "zone", "note"])
            w.writerows(rows)
        return qb.verdict_zones_sha256(path)

    want = hashlib.sha256(b"A-s1,core\nB-s1,niche").hexdigest()
    assert write("a.csv", [("A-s1", "core", "x"), ("B-s1", "niche", "y")]) == want
    assert write("b.csv", [("B-s1", "niche", "other"), ("A-s1", "core", "")]) == want
    assert write("c.csv", [("A-s1", "exclude", "x"), ("B-s1", "niche", "y")]) != want


@needs_probe
def test_no_clip_was_excluded_for_its_pitch():
    """Why outside a band is "unmeasured": every excluded clip sits in its gender's core band."""
    verdicts, measures, manifest = _probe()
    excluded = [cid for cid, v in verdicts.items() if v["zone"] == "exclude"]
    assert sorted(excluded) == ["F5-s1", "M2-s1", "M5-s1", "M7-s3"]
    for cid in excluded:
        m = measures[cid]
        assert qb.admit(manifest[cid]["gender"], m["f0_median"], m["f0_excursion"]) == "core", cid


@needs_probe
def test_the_male_niche_band_is_the_child_boy_alone():
    verdicts, _measures, manifest = _probe()
    ids = sorted(cid for cid, v in verdicts.items()
                 if v["zone"] == "niche" and manifest[cid]["gender"] == "Male")
    assert ids == ["P3-s1", "P3-s2", "P3-s3"]
    assert {manifest[cid]["band"] for cid in ids} == {"Child boy"}


@needs_probe
def test_the_provenance_is_the_probe_folders():
    with open(PROBE / "render_meta.json", encoding="utf-8") as fh:
        render = json.load(fh)
    with open(PROBE / "measure_meta.json", encoding="utf-8") as fh:
        measure = json.load(fh)
    assert render["commit"].startswith(qb.RENDER_COMMIT) and len(qb.RENDER_COMMIT) >= 7
    assert measure["render_commit"] == render["commit"]
    assert measure["f0_range_hz"] == list(qb.F0_RANGE_HZ)


def test_the_f0_range_is_the_measuring_tools():
    spec = importlib.util.spec_from_file_location("render_qwen_1c0",
                                                  REPO / "scripts/tools/render_qwen_1c0.py")
    rq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rq)
    assert qb.F0_RANGE_HZ == rq.F0_RANGE


def test_the_bands_are_well_formed():
    assert set(qb.BANDS) == set(itertools.product(qb.GENDERS, qb.ZONES))
    lo, hi = qb.F0_RANGE_HZ
    for key, b in qb.BANDS.items():
        assert b.n > 0 and lo <= b.f0_median_min <= b.f0_median_max <= hi, key
        assert 0 < b.f0_excursion_max, key
    for g in qb.GENDERS:
        assert qb.BANDS[(g, "core")].f0_median_max < qb.BANDS[(g, "niche")].f0_median_min, g


# --- admit ------------------------------------------------------------------------------------

def _mid(b):
    return (b.f0_median_min + b.f0_median_max) / 2


@pytest.mark.parametrize("gender, zone", list(itertools.product(qb.GENDERS, qb.ZONES)))
def test_admit_places_a_voice_inside_each_band(gender, zone):
    b = qb.BANDS[(gender, zone)]
    assert qb.admit(gender, _mid(b), b.f0_excursion_max / 2) == zone


@pytest.mark.parametrize("gender, zone", list(itertools.product(qb.GENDERS, qb.ZONES)))
def test_admit_bounds_are_inclusive_and_end_there(gender, zone):
    b = qb.BANDS[(gender, zone)]
    exc = b.f0_excursion_max
    assert qb.admit(gender, b.f0_median_min, exc) == zone
    assert qb.admit(gender, b.f0_median_max, exc) == zone
    assert qb.admit(gender, math.nextafter(b.f0_median_min, 0), 0.0) == qb.UNMEASURED
    assert qb.admit(gender, math.nextafter(b.f0_median_max, math.inf), 0.0) == qb.UNMEASURED
    assert qb.admit(gender, _mid(b), math.nextafter(exc, math.inf)) == qb.UNMEASURED


@pytest.mark.parametrize("gender", qb.GENDERS)
def test_admit_reads_a_voice_outside_every_band_as_unmeasured(gender):
    core, niche = qb.BANDS[(gender, "core")], qb.BANDS[(gender, "niche")]
    for f0 in (40.0, core.f0_median_min - 1.0, (core.f0_median_max + niche.f0_median_min) / 2,
               niche.f0_median_max + 1.0, 1000.0):
        assert qb.admit(gender, f0, 1.0) == qb.UNMEASURED, f0


@pytest.mark.parametrize("f0, exc", [(None, 10.0), (120.0, None), (None, None),
                                     (math.nan, 10.0), (120.0, math.nan)])
def test_admit_reads_an_unmeasured_f0_as_unmeasured(f0, exc):
    assert qb.admit("Male", 120.0, 10.0) == "core"                  # positive control
    assert qb.admit("Male", f0, exc) == qb.UNMEASURED


@pytest.mark.parametrize("gender", qb.GENDERS)
def test_admit_admits_a_negative_excursion(gender):
    """The excursion has a maximum only (admit's docstring): a negative one is admitted."""
    b = qb.BANDS[(gender, "core")]
    assert qb.admit(gender, _mid(b), -1.0) == "core"


def test_admit_never_says_exclude():
    """Over a grid spanning both genders' bands and beyond, only the three answers come back, and
    each of them does (so the grid reached every branch)."""
    got = {qb.admit(g, f0, exc) for g in qb.GENDERS
           for f0 in range(30, 1001, 5) for exc in range(0, 301, 10)}
    assert got == {"core", "niche", qb.UNMEASURED}
    assert qb.UNMEASURED == "unmeasured"


@pytest.mark.parametrize("gender", ["male", "M", "Man", "Other", None, ""])
def test_an_unknown_gender_refuses(gender):
    with pytest.raises(ValueError, match="gender must be one of"):
        qb.admit(gender, 120.0, 10.0)
    with pytest.raises(ValueError, match="gender must be one of"):
        qb.provisional(gender, "core")


def test_the_male_niche_band_is_provisional_and_still_reported():
    assert qb.PROVISIONAL == {("Male", "niche")}
    b = qb.BANDS[("Male", "niche")]
    zone = qb.admit("Male", _mid(b), b.f0_excursion_max)
    assert zone == "niche" and qb.provisional("Male", zone) is True
    for key in set(qb.BANDS) - {("Male", "niche")}:
        assert qb.provisional(*key) is False, key
    assert qb.provisional("Male", qb.UNMEASURED) is False
