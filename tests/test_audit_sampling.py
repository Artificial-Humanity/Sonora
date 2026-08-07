"""§7 audit/QC fixes: C-M1, C-M9, C-L5, C-L6 (2026-08-07).

What these share is that each one FAILS BY DOING NOTHING. A tail sample that rounds to
zero, a defer flag that only flips one way, a cohort glob that sweeps quarantined clips
into the statistics they are supposed to be excluded from — none of them error, and each
leaves a report that looks complete.
"""

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


def test_deferral_is_revisable():
    """A clip deferred by an earlier `select` and QC-flagged by a LATER pass stayed
    deferred forever — invisible in todo, never heard. QC runs after every generation pass
    and `--flags` grows, so deferral has to be revisable or the flags cannot reach the ear
    at all. That quietly breaks this tool's one non-negotiable rule, which its own
    docstring states: every QC-flagged clip goes to the ear."""
    src = _src("pick_audit_subset.py")
    assert 'r["id"] in keep_ids and r["status"] == "deferred"' in src
    assert 'r["status"] = "unaudited"' in src
    assert "re-queued" in src, "a silent re-queue is the same class of problem"


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
