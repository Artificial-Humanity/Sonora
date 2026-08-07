"""C-M4: the head of a clip, and the two thresholds that do not exist yet.

`tail_lost()` computed the head-loss word count — `blocks[0].a` is exactly that — and
returned only the tail, so nothing in the pipeline has ever seen a clip that starts late.
It is the same defect as a truncated tail and invisible for the same reason: dropping the
first six words of a 139-word passage is a 4% error rate and an unusable clip, and a global
WER cannot tell those apart.

MEASURED 2026-08-07 over every qc_measures.jsonl on disk (3,189 clips, 13 campaigns): 19
clips drop three or more opening words and EIGHT of those passed every gate, up to 25% of
the passage. Head loss runs about half as often as tail loss (0.6% vs 1.1% at >= 3 words).

The other half of this file is about what is deliberately NOT done. `head_ok` and
`speech_ok` are measured and gate nothing, because a gate on a guessed threshold either
passes truncated clips or rejects good ones and neither failure announces itself. These
tests exist to keep that honest — an uncalibrated threshold must not drift into `gates`,
and `None` must never be read as "passed".
"""

import json
import pathlib
import sys

import pytest

SYNTH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "synthesis"
sys.path.insert(0, str(SYNTH))

synth_common = pytest.importorskip("synth_common")
edge_loss = synth_common.edge_loss

PASSAGE = ("the quick brown fox jumps over the lazy dog and then runs "
           "away into the deep green wood")


def test_a_late_start_is_counted():
    """The head is what was thrown away."""
    e = edge_loss(PASSAGE, "jumps over the lazy dog and then runs away into the deep "
                           "green wood")
    assert e["head_words"] == 4
    assert e["head_frac"] == pytest.approx(4 / 18)
    assert e["tail_words"] == 0


def test_an_early_stop_still_measures_the_same_as_before():
    e = edge_loss(PASSAGE, "the quick brown fox jumps over the lazy dog and then runs")
    assert e["tail_words"] == 6
    assert e["head_words"] == 0


def test_both_ends_at_once_are_not_summed():
    """Reported separately because they are not the same failure and will not share a
    threshold: ASR has no left context at the first word."""
    e = edge_loss(PASSAGE, "jumps over the lazy dog and then runs")
    assert e["head_words"] == 4 and e["tail_words"] == 6


def test_a_clean_read_loses_nothing():
    e = edge_loss(PASSAGE, PASSAGE.upper() + ".")
    assert (e["head_words"], e["tail_words"]) == (0, 0)


def test_a_silent_render_loses_the_whole_passage_at_the_head():
    """No aligned block at all. The head count must be the full passage rather than 0,
    which is what a naive `blocks[0].a` on an empty list would produce."""
    e = edge_loss(PASSAGE, "")
    assert e["head_words"] == 18 and e["head_frac"] == 1.0


def test_empty_reference_does_not_divide_by_zero():
    assert edge_loss("", "anything") == {"head_frac": 0.0, "head_words": 0,
                                         "tail_frac": 0.0, "tail_words": 0}


def test_the_live_case_that_passed_every_gate():
    """`delivery-v1-narration/wuthering-heights_nar_0036_neu_CHA`, verbatim from its
    qc_measures row. The clip lost "There was scarcely time" — 4 of its 19 words, 21% of
    the passage, the entire subject and verb — and `hard_pass` was True, because a global
    WER of 0.211 sits comfortably under the 0.35 gate. This is the whole finding in one
    row: the instrument HAD the number and did not look at it."""
    ref = ("There was scarcely time to experience a thrill of horror before we saw "
           "that the little wretch was safe.")
    hyp = "to experience a thrill of horror before we saw that the little wretch was safe."
    e = edge_loss(ref, hyp)
    assert e["head_words"] == 4
    assert e["head_frac"] == pytest.approx(4 / 19)
    assert e["tail_words"] == 0

    qc = pytest.importorskip("qc_gate", reason="needs librosa/onnxruntime")
    assert qc.wer(ref, hyp) < qc.ASR_MAX_WER          # ...which is why it passed


# --- the thresholds that do not exist -------------------------------------------------


def test_the_uncalibrated_thresholds_are_still_none():
    """A guard, not a preference. The moment one of these is filled in, the sweep numbers
    that chose it belong beside it — and this test is where someone finds out that is
    expected of them."""
    src = (SYNTH / "qc_gate.py").read_text(encoding="utf-8")
    assert "SPEECH_MIN_SECONDS = None" in src
    assert "HEAD_LOST_MAX = None" in src
    assert "UNCALIBRATED" in src


def test_an_uncalibrated_measure_is_not_in_the_gates_dict():
    """`hard_pass` is `all(gates.values())` and `stage_pool.qc_flagged` reads `gates` too,
    so an uncalibrated entry there becomes a hard gate at whatever number was typed —
    and `None` is falsy, so it would fail every clip rather than pass them."""
    src = (SYNTH / "qc_gate.py").read_text(encoding="utf-8")
    assert 'gates["head_ok"]' not in src
    assert 'gates["speech_ok"]' not in src
    assert 'advisories = {' in src
    assert 'all(gates.values())' in src


def test_none_is_not_false():
    """The distinction the advisories dict rests on: 'no threshold set' and 'failed' are
    different answers, and a consumer that cannot tell them apart will report a clean
    campaign as broken or a broken one as clean."""
    advisories = {"head_ok": None, "speech_ok": None}
    assert not all(advisories.values())          # ...which is why they are not gates
    assert [k for k, v in advisories.items() if v is None] == ["head_ok", "speech_ok"]


def test_the_gate_says_out_loud_what_it_does_not_cover():
    """"The gate passed" must not be readable as "nothing is wrong with the head". Silence
    about an uncalibrated measure is how it becomes an assumed one."""
    src = (SYNTH / "qc_gate.py").read_text(encoding="utf-8")
    assert "NOT GATED (no calibrated threshold yet)" in src


def test_both_speech_measures_are_recorded():
    """The owner's 4 s floor was set against the energy measure and `librivox_align`
    enforces it with the same instrument on purpose. Recording both is what makes the
    switch a measurement rather than a guess — and the measurement, when it was run, said
    they agree (ratio 1.008, one clip in 150 changing side at the floor)."""
    src = (SYNTH / "qc_gate.py").read_text(encoding="utf-8")
    assert '"speech_dur": speech_dur' in src
    assert '"speech_dur_vad"' in src
    assert "1.008" in src


# --- the sweep ------------------------------------------------------------------------


@pytest.fixture()
def swept(tmp_path):
    """A campaign whose measures predate `head_lost_frac`, so the backfill has to fire."""
    gc = pytest.importorskip("gate_calibration")
    import csv
    cdir = tmp_path / "camp"
    cdir.mkdir()
    rows = []
    for i in range(6):                      # 3 truncated + 3 clean
        truncated = i < 3
        rows.append({"id": f"c{i}", "text": PASSAGE,
                     "asr_hyp": PASSAGE.split(" ", 5)[-1] if truncated else PASSAGE})
    (cdir / "qc_measures.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    ratings = tmp_path / "ratings.csv"
    with ratings.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "status", "score", "note"])
        w.writeheader()
        for i in range(6):
            w.writerow({"id": f"c{i}", "status": "dropped" if i < 3 else "keep",
                        "score": "2" if i < 3 else "5",
                        "note": "starts late, first words missing" if i < 3 else "good"})
    return gc, cdir, ratings


def test_the_sweep_backfills_edge_loss_for_rows_that_predate_it(swept, capsys):
    """Every existing measures row carries `text` and `asr_hyp`, so head loss is exactly
    recoverable. That is what lets a threshold be chosen against eleven campaigns of
    evidence instead of re-running the gate over every wav on disk."""
    gc, cdir, _ = swept
    measures = gc.load_measures(str(cdir))
    assert all(r["head_lost_frac"] is not None for r in measures.values())
    assert "recomputed for 6 row(s)" in capsys.readouterr().out


def test_the_sweep_prints_catch_against_false_flag(swept, monkeypatch, capsys):
    gc, cdir, ratings = swept
    monkeypatch.setattr(sys, "argv", [
        "gate_calibration", "--campaign-dir", str(cdir), "--sweep", "head_lost_frac",
        "--ratings", str(ratings), "--drop-note", "starts late"])
    gc.main()
    out = capsys.readouterr().out
    assert "defects  : 3" in out and "keeps    : 3" in out
    assert "catch" in out and "false flag" in out
    assert "3/3" in out                     # every labelled defect caught somewhere
    assert "thin" in out                    # 3 labels is not a calibration set


def test_the_sweep_refuses_to_pretend_when_nothing_is_labelled(swept, monkeypatch, capsys):
    """A zero here means the notes do not describe the defect, not that the defect is
    absent — which is exactly the live situation for head truncation."""
    gc, cdir, ratings = swept
    monkeypatch.setattr(sys, "argv", [
        "gate_calibration", "--campaign-dir", str(cdir), "--sweep", "head_lost_frac",
        "--ratings", str(ratings), "--drop-note", "no note says this"])
    gc.main()
    out = capsys.readouterr().out
    assert "Nothing to calibrate against" in out
    assert "free text" in out


def test_direction_is_named_per_measure_not_inferred():
    """A fraction lost is bad when HIGH; a duration is bad when LOW. Inferring it would
    invert the whole table for exactly the measure C-M4 adds."""
    gc = pytest.importorskip("gate_calibration")
    assert gc.SWEEP_DIRECTION["speech_dur_vad"] == "below"
    assert gc.SWEEP_DIRECTION.get("head_lost_frac", "above") == "above"
    assert gc.flags(3.0, 4.0, "below") is True
    assert gc.flags(3.0, 4.0, "above") is False
    assert gc.flags(None, 4.0, "below") is None      # not measured is not a pass
