"""§ 9's verdict must follow its data, in BOTH directions (issue #33).

§ 9 is the join § 8 says it cannot perform: rendered `A = 0` loudness against each lane's
own training-set mean, which separates an intercept displacement from the confound § 8
leaves open ("the named lanes may genuinely BE quieter deliveries").

The verdict is a printed sentence, and a printed sentence beside derived numbers is the
failure mode this whole script exists to end — three conclusions in it were hardcoded
strings that survived a run whose data contradicted them. So the guard here is not that § 9
prints the right thing on the real artifact; it is that **the branch follows the sign of the
rank correlation**, driven from both sides with synthetic corpora built to force each.

⚠ Deliberately NOT asserted against `/data`. The real artifact gives rho = -0.80, so a test
reading it would pass while checking only one branch — and would go red the day the corpus
changes, for a reason that is not a defect.
"""
import importlib.util
import io
import os
from contextlib import redirect_stdout

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "derive_a_channel_stats.py")

LANES = ("Dialogue", "Neutral", "Newscaster", "Speech")


def _load():
    spec = importlib.util.spec_from_file_location("derive_a_channel_stats", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _corpus(lane_lufs):
    """Corpus rows at A ~ 0 with a chosen native loudness per lane."""
    rows = []
    for lane, lufs in lane_lufs.items():
        for i in range(12):
            rows.append({"delivery": lane, "campaign": f"c-{lane}", "a": 0.01 * (i % 3 - 1),
                         "lufs_native": lufs + 0.05 * (i % 4), "lufs_adjusted": -18.0,
                         "lufs_offset": 5.0})
    return rows


def _probe(lane_lufs):
    """Probe rows at every level, with the A = 0 cell carrying the chosen loudness."""
    rows = []
    for lane, lufs in list(lane_lufs.items()) + [("unknown", -30.0)]:
        for a in (-1.0, 0.0, 1.0):
            for text in range(2):
                for rep in range(3):
                    rows.append({"lane": lane, "energy": a, "text": f"t{text}",
                                 "lufs": lufs + 4.0 * a + 0.02 * rep})
    return rows


def _run(corpus_lufs, probe_lufs):
    mod = _load()
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.intercept(_corpus(corpus_lufs), _probe(probe_lufs))
    return buf.getvalue()


def test_a_tracking_profile_is_reported_as_tracking():
    """The confound § 8 could not rule out, made true: the corpus's quiet lanes render
    quiet. § 9 must then say the displacement IS explained by lane character, because on
    this data it is."""
    order = {"Dialogue": -22.0, "Neutral": -23.0, "Newscaster": -24.0, "Speech": -26.0}
    rendered = {"Dialogue": -30.0, "Neutral": -31.0, "Newscaster": -32.0, "Speech": -34.0}
    out = _run(order, rendered)
    assert "TRACKS the training profile" in out
    assert "DOES NOT TRACK" not in out
    assert "genuinely quieter deliveries" in out


def test_an_inverted_profile_is_reported_as_not_tracking():
    """The real artifact's shape: the corpus's quietest lane renders loudest."""
    order = {"Dialogue": -22.0, "Neutral": -23.0, "Newscaster": -24.0, "Speech": -26.0}
    rendered = {"Dialogue": -34.0, "Neutral": -32.0, "Newscaster": -31.0, "Speech": -30.0}
    out = _run(order, rendered)
    assert "DOES NOT TRACK THE TRAINING PROFILE" in out
    assert "does\n    NOT explain" in out or "NOT explain" in out
    # The claim it must NOT make, however tempting the inversion looks.
    assert "IT DOES NOT: prove the three-reference-frames mechanism" in out, \
        "a negative correlation is consistent with the mechanism, not proof of it"


def test_the_global_offset_cannot_change_the_verdict():
    """Both profiles are centred on their own mean, so shifting every render by a constant
    — which is all a differently-calibrated checkpoint does — must leave § 9 unmoved."""
    order = {"Dialogue": -22.0, "Neutral": -23.0, "Newscaster": -24.0, "Speech": -26.0}
    rendered = {"Dialogue": -34.0, "Neutral": -32.0, "Newscaster": -31.0, "Speech": -30.0}
    import re
    base = _run(order, rendered)
    shifted = _run(order, {k: v - 12.0 for k, v in rendered.items()})

    def discrepancies(out):
        return [m.group(1) for m in
                re.finditer(r"^\s+\w+\s+\d+\s+\S+\s+\d+\s+\S+\s+\S+\s+\S+\s+(\S+)",
                            out, re.M)]

    d_base, d_shift = discrepancies(base), discrepancies(shifted)
    assert d_base and len(d_base) == len(LANES), \
        f"the § 9 table moved and this guard now reads nothing: {d_base}"
    assert d_base == d_shift, \
        f"a constant shift of every render changed the discrepancies: {d_base} vs {d_shift}"
    assert "DOES NOT TRACK THE TRAINING PROFILE" in base
    assert "DOES NOT TRACK THE TRAINING PROFILE" in shifted
    # The offset is the ONE thing that must move, otherwise the test above is vacuous.
    def offset(out):
        return re.search(r"global offset render - corpus = ([-+][\d.]+)", out).group(1)
    assert offset(base) != offset(shifted), \
        "the global offset did not move, so this test never exercised a shift"


def test_a_lane_missing_from_either_side_is_refused_by_name():
    """A read-only diagnostic owes a named refusal, not a traceback — and § 9 joins two
    artifacts, so one of them lacking a lane is an ordinary re-run, not a defect."""
    mod = _load()
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.intercept(_corpus({"Speech": -26.0}), _probe({"Dialogue": -30.0}))
    out = buf.getvalue()
    assert "there is no between-lane profile to" in out
    assert "Traceback" not in out
