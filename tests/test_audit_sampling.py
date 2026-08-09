"""§7 audit/QC fixes: C-M1, C-M9, C-L5, C-L6 (2026-08-07).

What these share is that each one FAILS BY DOING NOTHING. A tail sample that rounds to
zero, a defer flag that only flips one way, a cohort glob that sweeps quarantined clips
into the statistics they are supposed to be excluded from — none of them error, and each
leaves a report that looks complete.
"""

import argparse
import csv
import json
import math
import os
import pathlib
import sys

import pytest

SYNTH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "synthesis"
sys.path.insert(0, str(SYNTH))


def _src(name):
    return (SYNTH / name).read_text(encoding="utf-8")


# --- C-L6: the trusted tier's tail sample rounded to zero -----------------------------


@pytest.mark.parametrize("remainder", [1, 5, 10, 16, 17, 33, 100])
def test_a_non_zero_tail_fraction_always_samples_at_least_one(remainder):
    """`round(len(rest) * 0.03)` is ZERO until the remainder reaches 17 — and banker's
    rounding makes it zero at exactly 16.67 too. The tail sample is the whole mechanism
    for noticing that a TRUSTED engine has started drifting, and those engines are audited
    1 per group precisely because it is watching them. On any batch under 17 it silently
    did not exist."""
    pa = pytest.importorskip("pick_audit_subset")
    frac = pa.TIERS["trusted"]["tail"]
    assert math.ceil(remainder * frac) >= 1
    assert "math.ceil(len(rest) * frac)" in _src("pick_audit_subset.py")


def test_the_old_rounding_is_what_produced_the_hole():
    """Kept as the record of the defect: for every batch a trusted engine realistically
    produces below 17 clips, the old expression sampled nothing at all."""
    assert [n for n in range(1, 17) if round(n * 0.03) == 0] == list(range(1, 17))
    assert round(17 * 0.03) == 1


def test_a_zero_fraction_still_means_no_sample():
    """Ceiling must not invent a sample where the policy says there is none."""
    assert math.ceil(50 * 0.0) == 0


# --- C-M1: deferral was one-way -------------------------------------------------------


def _select_fixture(tmp_path, monkeypatch, n=12, engine="qwen"):
    """A one-group campaign on disk: bank.json + ratings.csv, all rows unaudited."""
    pa = pytest.importorskip("pick_audit_subset")
    datasets = tmp_path / "datasets"
    (datasets / "camp").mkdir(parents=True)
    ids = [f"clip_{i:04d}" for i in range(n)]
    (datasets / "camp" / "bank.json").write_text(json.dumps({"lines": [
        {"id": i, "book": "a-book", "intended_delivery": "Neutral", "engine": engine}
        for i in ids]}), encoding="utf-8")

    ratings = tmp_path / "ratings.csv"
    with open(ratings, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["campaign", "id", "engine", "score",
                                          "note", "status"])
        w.writeheader()
        for i in ids:
            w.writerow({"campaign": "camp", "id": i, "engine": engine,
                        "score": "", "note": "", "status": "unaudited"})

    monkeypatch.setattr(pa, "RATINGS", str(ratings))
    monkeypatch.setattr(pa, "DATASETS", str(datasets))
    # The new-reader rule reads real manifests and profiles off /data; this campaign is
    # synthesis, so neither applies. Stubbed so the test isolates the defer/requeue flip.
    monkeypatch.setattr(pa, "_clip_readers", lambda: {})
    monkeypatch.setattr(pa, "_confirmed_reader_titles", lambda: set())
    return pa, ratings, ids


def _statuses(ratings):
    with open(ratings, newline="", encoding="utf-8") as f:
        return {r["id"]: r["status"] for r in csv.DictReader(f)}


def _select(pa, flags=None):
    pa.cmd_select(argparse.Namespace(campaign="camp", flags=flags))


def test_deferral_is_revisable(tmp_path, monkeypatch):
    """A clip deferred by an earlier `select` and QC-flagged by a LATER pass stayed
    deferred forever — invisible in todo, never heard. QC runs after every generation pass
    and `--flags` grows, so deferral has to be revisable or the flags cannot reach the ear
    at all. That quietly breaks this tool's one non-negotiable rule, which its own
    docstring states: every QC-flagged clip goes to the ear.

    QC-H2 (2026-08-09): this test used to assert the SOURCE STRINGS of the fix, and it
    passed for two days over a branch that could never execute — `by_group` was built from
    `status == "unaudited"` rows only, so every id in `keep_ids` was an unaudited row's id
    and the `elif ... == "deferred"` arm was unreachable. A grep test cannot see reachability.
    """
    pa, ratings, ids = _select_fixture(tmp_path, monkeypatch)

    _select(pa)                                   # first pass: sample, defer the rest
    after_first = _statuses(ratings)
    deferred = [i for i, s in after_first.items() if s == "deferred"]
    assert deferred, "the fixture must actually defer something for this to mean anything"

    # A later qc_gate pass flags one of the deferred clips.
    flags = tmp_path / "qc_flags.txt"
    flags.write_text(deferred[0] + "\n", encoding="utf-8")
    _select(pa, flags=str(flags))

    assert _statuses(ratings)[deferred[0]] == "unaudited", (
        "a QC-flagged clip that was previously deferred never reached the ear")


def test_a_verdict_the_ear_has_already_given_is_not_revisable(tmp_path, monkeypatch):
    """Deferral is a sampling decision and may be revisited; a `keep`/`drop` is the ear's
    answer and sampling must never touch it. Widening the candidate pool is only safe
    because the widening stops at `deferred`."""
    pa, ratings, ids = _select_fixture(tmp_path, monkeypatch)
    rows = list(csv.DictReader(open(ratings, newline="", encoding="utf-8")))
    for r in rows[:4]:
        r["status"], r["score"] = "keep", "4"
    with open(ratings, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    flags = tmp_path / "qc_flags.txt"
    flags.write_text("\n".join(r["id"] for r in rows[:4]), encoding="utf-8")
    _select(pa, flags=str(flags))

    after = _statuses(ratings)
    assert [after[r["id"]] for r in rows[:4]] == ["keep"] * 4


def test_selection_is_stable_across_reruns(tmp_path, monkeypatch):
    """A consequence of the fix worth pinning: because the candidate pool no longer
    shrinks each pass, the positional spread picks the SAME clips on a re-run. Under the
    old filter a second `select` re-sampled whatever was left and slowly dragged the whole
    campaign into the queue, one pass at a time."""
    pa, ratings, ids = _select_fixture(tmp_path, monkeypatch)
    _select(pa)
    first = _statuses(ratings)
    _select(pa)
    assert _statuses(ratings) == first


def test_the_rule_it_protects_is_still_written_down():
    doc = " ".join(_src("pick_audit_subset.py").split())
    assert "every QC-flagged clip goes to the ear" in doc


# --- C-M9: cohort statistics included clips the ear had already rejected ---------------


def test_quarantined_clips_are_out_of_the_cohort():
    """ROOM is a robust z against the cohort's OWN median/MAD, so sweeping `_dropped/`
    and `_superseded/` in meant clips the ear had already rejected were helping define
    what "normal" looks like for the clips still under judgement. On a cohort quarantined
    precisely because it was reverberant, that pulls the median toward the defect."""
    src = _src("qc_artifacts.py")
    assert '"_pre_loudnorm", "_dropped", "_superseded"' in src
    # matched as a path COMPONENT, not a substring: a campaign legitimately named
    # `..._dropped_takes` must not vanish from its own report.
    assert "p.split(os.sep)" in src


def test_the_manifest_glob_reaches_the_layout_the_pipeline_writes():
    """The wav glob was recursive and the manifest glob was not, so a campaign laid out
    as <campaign>/<engine-out>/ — which is what synth_bank.sh writes and the QC gate
    assumes — found no manifests at all. Every clip then fell back to engine "?" and the
    whole campaign became ONE cohort. The per-engine comparison is the entire point of
    this tool, and it was silently not happening; a "?" row reads as missing metadata
    rather than as a broken join."""
    src = _src("qc_artifacts.py")
    assert 'glob.glob(os.path.join(root, "**", "*_manifest.jsonl"), recursive=True)' in src


# --- C-L5: publish_tier had no policy for the real-audio lane -------------------------


def test_the_real_audio_lane_has_a_publication_policy():
    """`stage_pool` writes `engine: "librivox"` for the force-align lane and three books
    are already fetched, so the `unknown` check would hard-error the day the first staged
    real-audio clip reached a metadata.jsonl. It is a matter of when, not whether."""
    pt = pytest.importorskip("publish_tier")
    assert "librivox" in pt.ENGINE_POLICY
    licence, publishable = pt.ENGINE_POLICY["librivox"]
    assert publishable is True
    assert "PD" in licence


def test_the_librivox_entry_says_why_rather_than_inheriting_a_default():
    """These are human recordings, not synthesis: there is no model card to verify, so
    the surrounding "verify the weights licence" instruction is not merely unanswered for
    this row, it is the wrong question. The reasoning belongs on the record."""
    src = _src("publish_tier.py")
    assert "not an ENGINE" in src
    assert "C-L5" in src


# --- the instrumentation bargain (2026-08-08) -----------------------------------------


def test_an_engine_with_a_detector_has_the_detector_wired():
    """moss_vg left `scrutinized` because its defect became an instrument, not because
    the defect stopped happening.

    `qc_engine_defects.py` says in its own docstring that it is "what the tier system
    needs in order to ever hand those engines a ride-along back: coverage traded for
    instrumentation". Until 2026-08-08 it was a MANUAL step, so the trade was not real —
    a detector nobody runs trades nothing. If an engine is off scrutinized on the strength
    of a detector, the detector has to fire on every bank.
    """
    bank = _src("synth_bank.sh")
    assert "qc_engine_defects.py" in bank, \
        "the defect detectors are not wired into the bank pipeline"
    assert "--append-flags" in bank, \
        "detectors run but their flags never reach qc_flags.txt / the ear"

    defects = _src("qc_engine_defects.py")
    subset = _src("pick_audit_subset.py")
    # Every engine that HAS a detector must still be sampled, i.e. not trusted — the
    # detector buys a step down from scrutinized, never a step up to 1-per-group.
    for engine in ("moss_vg", "zonos"):
        assert f'"{engine}"' in defects, f"{engine} lost its detector"
        assert f'"{engine}": "trusted"' not in subset, \
            f"{engine} has a stochastic defect and a detector; trusted is not the trade"


def test_scrutinized_is_still_the_default_for_unknown_engines():
    """No engine is NAMED scrutinized any more, which must not be read as the tier being
    retired. It is the landing place for an engine with no track record — the onboarding
    rule — and the id-join failure path depends on it too."""
    src = _src("pick_audit_subset.py")
    assert 'DEFAULT_TIER = "scrutinized"' in src
    assert "SCRUTINIZED_ENGINES" in src


# --- the calibration tool's blind spot (2026-08-08) -----------------------------------


def test_the_drop_marker_is_a_verdict_not_a_missing_score():
    """`gate_calibration` exists to choose gate thresholds from the ear's own verdicts,
    and it was discarding almost all of them.

    `parse_score` searched for `[1-5]`, so the audition app's reject marker `x` returned
    None — and every caller then did `if score is None: continue`. Measured when found:
    **253 of the 286 dropped/reroll rows carry `x`, so 88% of every rejection was
    invisible**, and a sweep over a campaign full of drops printed "no rated clip matches
    the defect", which reads exactly like "this campaign is clean".

    0, not None: the downstream test is `score < keep_score`, and a drop has to compare
    as worse than any keep.
    """
    sys.path.insert(0, str(SYNTH))
    gc = pytest.importorskip("gate_calibration")
    assert gc.parse_score("x") == 0
    assert gc.parse_score("X") == 0
    assert gc.parse_score(" x ") == 0
    assert gc.parse_score("5") == 5
    assert gc.parse_score("") is None, "no verdict is still no verdict"
    assert gc.parse_score(None) is None
    # And it must sort below any keep bar.
    assert gc.parse_score("x") < 4


def test_the_retake_marker_is_a_verdict_too():
    """QC-M3: the 2026-08-08 fix covered `x` and stopped there. Retake writes `"0"`, and
    `re.search(r"[1-5]", "0")` finds nothing — so it returned None and `load_ear` dropped
    the row exactly as it had dropped the drops, for the remaining slice of rejections.
    34 rows on the live sheet, every one status `reroll`. Same failure, one marker over.
    """
    sys.path.insert(0, str(SYNTH))
    gc = pytest.importorskip("gate_calibration")
    assert gc.parse_score("0") == 0
    assert gc.parse_score(" 0 ") == 0
    assert gc.parse_score("0") < 4, "a retake must sort below any keep bar"


def test_a_recategorized_row_is_still_the_keep_it_was():
    """The counterweight to both marker fixes. `x1`..`x5` are legacy Recategorize rows
    (status `relabeled`, retired 2026-07-26): the ear KEPT the clip at that score and
    re-labelled it, and `stage_pool` counts `relabeled` as a keep. 57 on the sheet. Widen
    the `x` test to "starts with x" and all 57 silently become rejections."""
    sys.path.insert(0, str(SYNTH))
    gc = pytest.importorskip("gate_calibration")
    assert [gc.parse_score(f"x{n}") for n in range(1, 6)] == [1, 2, 3, 4, 5]
