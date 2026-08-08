"""C-M4: the head of a clip, and the two thresholds that do not exist yet.

`tail_lost()` computed the head-loss word count — `blocks[0].a` is exactly that — and
returned only the tail, so nothing in the pipeline has ever seen a clip that starts late.
It is the same defect as a truncated tail and invisible for the same reason: dropping the
first six words of a 139-word passage is a 4% error rate and an unusable clip, and a global
WER cannot tell those apart.

MEASURED 2026-08-07 over every qc_measures.jsonl on disk (3,189 clips, 13 campaigns): 19
clips drop three or more opening words and EIGHT of those passed every gate, up to 25% of
the passage. Head loss runs about half as often as tail loss (0.6% vs 1.1% at >= 3 words).

The other half of this file is about what is deliberately NOT done. `head_ok` is measured
and gates nothing, because a gate on a guessed threshold either passes truncated clips or
rejects good ones and neither failure announces itself. These tests exist to keep that
honest — an uncalibrated threshold must not drift into `gates`, and `None` must never be
read as "passed".

`speech_ok` left that company on **2026-08-07**: the owner made it a hard gate. It was
never the same kind of open question — the threshold has been theirs since 2026-07-25 (4 s
of speech), and what was missing was confidence that switching instruments would not move
it. The sweep answered that (mean VAD/energy ratio 1.008 over 150 clips, one clip in 150
changing side at the floor), so what remained was admission policy, and that is a call
rather than a measurement.
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
    qc_measures row: ASR could not match "There was scarcely time" — 4 of 19 words, 21% of
    the passage — while `hard_pass` stayed True, because a global WER of 0.211 sits under
    the 0.35 gate. The measurement is the point here: the instrument HAD the number and
    nothing looked at it.

    ⚠ What it does NOT show is a truncated clip. This one went to the ear on 2026-08-08
    and came back **keep, 5** — *"I hear all of the words in the printed text."* An earlier
    version of this docstring called it a clip that "lost the entire subject and verb",
    which was the measure's story rather than the audio's. Whisper misheard an opening
    that was spoken correctly, which is exactly the no-left-context asymmetry `edge_loss`
    warns about. The assertions below are about `edge_loss`, not about the clip's quality.
    """
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


def test_the_uncalibrated_threshold_is_still_none():
    """A guard, not a preference. The moment this is filled in, the sweep numbers that
    chose it belong beside it — and this test is where someone finds out that is expected
    of them."""
    src = (SYNTH / "qc_gate.py").read_text(encoding="utf-8")
    assert "HEAD_LOST_MAX = None" in src
    assert "UNCALIBRATED" in src


def test_an_uncalibrated_measure_is_not_in_the_gates_dict():
    """`hard_pass` is `all(gates.values())` and `stage_pool.qc_flagged` reads `gates` too,
    so an uncalibrated entry there becomes a hard gate at whatever number was typed —
    and `None` is falsy, so it would fail every clip rather than pass them."""
    src = (SYNTH / "qc_gate.py").read_text(encoding="utf-8")
    assert 'gates["head_ok"]' not in src
    assert 'advisories = {' in src
    assert 'all(gates.values())' in src


def test_the_speech_floor_is_the_owners_number_and_it_gates():
    """2026-08-07: hard gate, on the VAD measure, at the owner's 4 s.

    The threshold and the sweep that made switching instruments safe have to travel
    together — a bare `4.0` with no provenance is how a measured decision decays back into
    a typed one.
    """
    src = (SYNTH / "qc_gate.py").read_text(encoding="utf-8")
    assert "SPEECH_MIN_SECONDS = 4.0" in src
    assert 'gates["speech_ok"] = speech_dur_vad >= SPEECH_MIN_SECONDS' in src
    assert "1.008" in src, "the ratio that made the instrument switch safe"
    # Still guarded against being switched back OFF into a gate that fails everything,
    # since None is falsy and `all(gates.values())` cannot tell it from a failure.
    assert "if SPEECH_MIN_SECONDS is not None:" in src


def test_a_hyphen_is_a_separator_not_a_deletion():
    """Found 2026-08-08 while checking why 4 of the 8 head cases were one LibriVox title.

    The normalizer STRIPPED `-` rather than replacing it with a space, so a hyphenation
    disagreement between the canonical text and Whisper fabricated word-count differences
    out of nothing: `By-and-by` collapsed to one token against a three-token reference and
    reported a 3-word late start on a passage whose opening was spoken perfectly. Every
    other text path in the repo — op_g2p.phonemize, derive_vat_corpus — already splits on
    hyphens, because that is what espeak does.
    """
    spoken = edge_loss("By and by, if I were to marry you",
                       "By-and-by, if I were to marry you")
    assert spoken["head_words"] == 0, "hyphenation is not a late start"
    assert spoken["tail_words"] == 0
    # The same at the other end, where it has been GATING since 2026-07-31.
    assert edge_loss("he found an electric-light switch",
                     "he found an electric light switch")["tail_words"] == 0
    # ...and a real late start is still counted.
    assert edge_loss(PASSAGE, " ".join(PASSAGE.split()[4:]))["head_words"] == 4


def test_a_head_flagged_clip_is_owed_an_ear_even_though_nothing_gates_it():
    """C-M4's blind spot, found 2026-08-07 while queueing the eight clips.

    `stage_pool.qc_flagged` tested `all(gates.values())`, and `head_ok` is an ADVISORY —
    so the one finding that most needs an auditor was the one finding that could not
    reach one. A head-truncated clip kept the "known quantity" relaxation and folded as a
    silent `keep` with no ratings row at all. Four of the eight clips that dropped >= 3
    opening words while passing every gate are pooled LibriVox clips waiting to enter
    exactly that way.

    A threshold is only needed to REJECT a clip. Queueing one costs a listen, and the
    listen is where the threshold comes from.
    """
    clean = {"id": "a", "gates": {"asr_ok": True}, "head_words_lost": 0}
    late = {"id": "b", "gates": {"asr_ok": True}, "head_words_lost": 3}
    assert not synth_common.head_flagged(clean)
    assert synth_common.head_flagged(late)
    # One word is not a truncation, at either end.
    assert not synth_common.head_flagged({"id": "c", "head_words_lost": 1})


def test_a_row_written_before_the_head_measure_is_still_flagged():
    """Every qc_measures.jsonl on disk predates 2026-08-07, so a fallback that needs the
    field to exist would cover none of the corpus that actually has the problem."""
    row = {"id": "old", "gates": {"asr_ok": True},
           "text": PASSAGE,
           "asr_hyp": "jumps over the lazy dog and then runs away into the deep green wood"}
    assert row.get("head_words_lost") is None
    assert synth_common.head_flagged(row)


def test_one_threshold_not_two():
    """The number that QUEUES a clip and the number that tells the auditor what to listen
    for have to be the same, or the queue fills with clips whose note says nothing."""
    ra = pytest.importorskip("register_audition")
    assert ra.HEAD_WORDS_FLAG is synth_common.HEAD_WORDS_FLAG


def _queue_module(tmp_path, monkeypatch):
    """Import queue_head_audit against a synthetic data root (it reads env at import)."""
    import importlib
    monkeypatch.setenv("AUDITION_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDITION_RATINGS_DIR", str(tmp_path))
    sys.modules.pop("queue_head_audit", None)
    return importlib.import_module("queue_head_audit")


def test_the_queue_takes_the_clips_the_instrument_called_clean(tmp_path, monkeypatch):
    """The interesting population is the one that PASSED.

    A clip that already failed a gate is going to the ear anyway and its drop note will
    say why. Queueing it too would bury the eight cases the threshold has to come from in
    a list of ordinary QC failures.
    """
    camp = tmp_path / "campaign-x"
    camp.mkdir()
    late_clean = {"id": "late-clean", "hard_pass": True, "gates": {"asr_ok": True},
                  "head_words_lost": 4, "head_lost_frac": 0.21}
    late_failed = {"id": "late-failed", "hard_pass": False, "gates": {"asr_ok": False},
                   "head_words_lost": 4, "head_lost_frac": 0.21}
    on_time = {"id": "on-time", "hard_pass": True, "gates": {"asr_ok": True},
               "head_words_lost": 0}
    camp.joinpath("qc_measures.jsonl").write_text(
        "\n".join(json.dumps(r) for r in (late_clean, late_failed, on_time)),
        encoding="utf-8")

    mod = _queue_module(tmp_path, monkeypatch)
    assert [r["id"] for _, r in mod.candidates()] == ["late-clean"]


def test_queueing_never_discards_a_verdict(tmp_path, monkeypatch):
    """Four of the eight already carry an ear verdict, two of them a `keep` — one scored 5.

    The auditor is being asked a question nobody asked before ("does it start late?"),
    not told their previous answer was wrong. Status moves so the clip reaches the todo
    queue; the score and the auditor's own words stay exactly where they were.
    """
    mod = _queue_module(tmp_path, monkeypatch)
    src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    assert 'row["score"]' not in src, "the score column must not be written"
    assert 'row["status"] = "unaudited"' in src
    # The auditor's note is kept AHEAD of the generated one.
    assert 'f"{existing} | {note}"' in src
    # Report-only unless asked, through the shared transaction's dry_run.
    assert "dry_run=not args.apply" in src
    assert 'ap.add_argument("--apply"' in src


def test_none_is_not_false():
    """The distinction the advisories dict rests on: 'no threshold set' and 'failed' are
    different answers, and a consumer that cannot tell them apart will report a clean
    campaign as broken or a broken one as clean."""
    advisories = {"head_ok": None}
    assert not all(advisories.values())          # ...which is why it is not a gate
    assert [k for k, v in advisories.items() if v is None] == ["head_ok"]


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


def test_the_warmstart_widening_is_covered_by_the_mandatory_seam_gate():
    """The widening logic is torch-only, so it cannot run in the host suite — and a check
    that only ever skips protects nothing.

    It lives in `scripts/test_vat_dim_seams.py`, which is mandatory pre-flight before any
    training run and already covers every other vat_dim seam. This test's whole job is to
    notice if it is removed from there, since nothing else on the host would.
    """
    seams = (pathlib.Path(__file__).resolve().parent.parent
             / "scripts" / "test_vat_dim_seams.py").read_text(encoding="utf-8")
    assert "import make_warmstart" in seams
    for name in ("the VAT trunk widens, new channels at zero",
                 "the speaker table is REFUSED without a proven index map",
                 "new speakers keep the fresh init",
                 "a shrink is never a widen",
                 "an un-allowlisted tensor is never reshaped",
                 "a reordered speaker map fails the prefix proof",
                 "the prefix proof reads EVERY namespace, not just LibriTTS",
                 "two speakers on one embedding row is refused"):
        assert name in seams, f"the seam gate lost its widening check: {name}"
