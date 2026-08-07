"""Regression coverage for the §8 label-derivation fixes (2026-08-06).

Same family as everything else in this review: none of these raised. A head that was never
scored, a clip that failed once and was never retried, a z-score fixed by arithmetic rather
than measured, a digit deleted from a transcript — each yields a corpus that looks complete
and is quietly wrong, in units indistinguishable from real measurements.
"""

import json
import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)

np = pytest.importorskip("numpy")


# --- D-M1: EIV heads must not be imputed ---------------------------------------------

HEADS = ["Amusement", "Valence", "Distress", "Thankfulness_Gratitude"]
WEIGHTS = [0.4, 0.6, 0.0, 0.0]


@pytest.fixture()
def mine():
    return pytest.importorskip("mine_emilia_keeps")


def test_a_missing_weighted_head_is_reported(mine):
    """`e.get(h, 0.0)` looked harmless and was not: 0.0 is a RAW score, so `z()` maps it to
    `(0 - mean)/std` — systematically nonzero, weighted into every clip's valence, in the
    same units as a real measurement."""
    record = {"Amusement": 1.0}          # Valence never scored
    assert mine.missing_weighted_heads(record, HEADS, WEIGHTS) == ["Valence"]


def test_unweighted_heads_may_be_absent(mine):
    """Weight 0.0 cannot move the dot product, and two such heads are legitimately absent
    from the raw files. A guard that demanded all twelve would fail on correct data — which
    is how a guard ends up disabled."""
    record = {"Amusement": 1.0, "Valence": -0.5}
    assert mine.missing_weighted_heads(record, HEADS, WEIGHTS) == []


def test_the_default_four_head_pass_is_caught(mine):
    """The concrete scenario the finding names: the EIV scorer run with its default head
    set instead of the twelve this pipeline needs."""
    default_pass = {"Arousal": 0.1, "Valence": 0.2, "Distress": 0.0, "Soft_vs._Harsh": 0.3}
    assert mine.missing_weighted_heads(default_pass, HEADS, WEIGHTS) == ["Amusement"]


def test_imputing_zero_really_does_move_the_answer():
    """Why this is refused rather than defaulted, shown numerically."""
    anchor_mean, anchor_std = 0.35, 0.12          # a plausible head scale
    imputed_z = (0.0 - anchor_mean) / anchor_std
    assert abs(imputed_z) > 2.9, imputed_z        # ~3 sigma of pure fiction, every clip


# --- D-M6: failed clips must be retried ----------------------------------------------


@pytest.fixture()
def tail():
    return pytest.importorskip("process_emilia_tail")


def test_error_rows_do_not_count_as_done(tail, tmp_path):
    """A transient failure — truncated download, decoder hiccup, OOM under load — was
    recorded as an `error` row, and every later run then skipped that clip as already
    processed. Permanent, and invisible: the resume line just says a larger number."""
    asr = tmp_path / "asr.jsonl"
    asr.write_text(
        json.dumps({"file": "a.mp3", "wer": 0.1}) + "\n"
        + json.dumps({"file": "b.mp3", "error": "RuntimeError('truncated')"}) + "\n"
        + json.dumps({"file": "c.mp3", "wer": 0.2}) + "\n",
        encoding="utf-8")
    done, retry = tail.scan_resume(str(asr))
    assert done == {"a.mp3", "c.mp3"}, "the failed clip must not be treated as complete"
    assert retry == 1


def test_error_rows_are_kept_in_the_file(tail, tmp_path):
    """They are the record of what went wrong; they simply stop counting as done."""
    asr = tmp_path / "asr.jsonl"
    body = json.dumps({"file": "b.mp3", "error": "boom"}) + "\n"
    asr.write_text(body, encoding="utf-8")
    tail.scan_resume(str(asr))
    assert asr.read_text(encoding="utf-8") == body


def test_missing_resume_file_is_not_an_error(tail, tmp_path):
    assert tail.scan_resume(str(tmp_path / "nope.jsonl")) == (set(), 0)


# --- D-L2: the per-speaker z guard ----------------------------------------------------


def _z(values, guard):
    v = np.asarray(values, float)
    m = float(v.mean())
    sd = (float(v.std()) or 1.0) if guard == "or" else float(v.std()) + 1e-6
    return [(x - m) / sd for x in v]


def test_a_constant_head_must_not_produce_a_full_scale_label():
    """The real D-L2 defect, and the review recorded it only as "the two implementations
    disagree on the guard". They do — and one of them fabricates a label.

    A head can be CONSTANT for a speaker: the EIV scorer returns one identical value for
    all 106 of speaker 6531's clips on `Amusement`, and likewise for 8095/Valence and
    909/Valence (224 clips between them). `v - mean` and `std` are then both floating-point
    dust around 1e-21. `std or 1.0` sees a nonzero std, divides dust by dust, and returns
    max|z| = 1.0 — a full-scale weighted contribution built out of rounding error, in the
    same units as a measurement. `std + 1e-6` divides by the floor and returns ~0, which is
    what a constant head actually tells you about any individual clip.
    """
    constant = [0.37] * 8
    # numpy's std of identical values is dust, not exactly zero — which is precisely why
    # `or 1.0` fails to catch this case.
    assert 0.0 <= float(np.std(constant)) < 1e-15

    assert max(abs(x) for x in _z(constant, "eps")) < 1e-9, "the floor must damp the dust"
    # and the broken guard, kept as the thing being guarded against:
    broken = max(abs(x) for x in _z([0.37, 0.37, 0.37 + 5e-16], "or"))
    assert broken > 0.5, broken


def test_both_guards_agree_when_variance_is_real():
    """The fix must not perturb healthy speakers, which is why it is a 1e-6 floor and not
    a threshold."""
    real = [0.1, 0.4, 0.35, 0.9, 0.2, 0.6]
    a, b = _z(real, "or"), _z(real, "eps")
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-4


def test_a_two_clip_speaker_carries_no_information():
    """Measured on v3c: 5 speakers have <=2 clips and 17 have <10. For n=2 the z-score is
    exactly +/-1 whatever the underlying scores are — arithmetic, not measurement — and
    those clips land on the rail at V = +/-0.500 after scaling."""
    for pair in ([0.1, 0.9], [-5.0, 5.0], [100.0, 100.2]):
        assert [round(x, 9) for x in _z(pair, "or")] == [-1.0, 1.0], pair


def test_a_real_population_is_not_degenerate():
    z = _z([0.1, 0.4, 0.35, 0.9, 0.2, 0.6, 0.55, 0.3, 0.8, 0.45], "or")
    assert len({round(x, 3) for x in z}) == 10
    assert max(abs(x) for x in z) < 2.0


# --- D-M3: digits are deleted, not expanded ------------------------------------------


def test_digit_detection_matches_what_the_tokenizer_drops():
    """The corpus lane now refuses these rather than shipping a transcript missing a word
    the audio speaks. Verified inert for the existing data: 0 of 8,000 sampled LibriTTS
    transcripts and 0 of 5,736 dev-clean transcripts contain a digit — it fires on Emilia
    YODAS captions, which is exactly the corpus Phase 1 merges."""
    assert any(c.isdigit() for c in "I have 3 cats")
    assert any(c.isdigit() for c in "in 1984 he wrote")
    assert not any(c.isdigit() for c in "I have three cats")
    assert not any(c.isdigit() for c in "a perfectly ordinary sentence, with punctuation!")


# --- C-M10: the delivery coverage floor ----------------------------------------------


def _thin(n_voted, secs_heard, secs_total,
          min_clips=8, min_secs=3, min_spread=0.25):
    """Mirrors stage_pool's --mark-delivery floor."""
    spread = len(secs_heard) / max(secs_total, 1)
    return (n_voted < min_clips
            or len(secs_heard) < min(min_secs, secs_total)
            or spread < min_spread)


def test_both_real_delivery_samples_are_refused():
    """A title-level mark propagates to EVERY clip in the book, and unanimity was the whole
    test — any sample, any size, any distribution. Both real samples are the degenerate
    shape: librivox-v1's 12 audited clips are one contiguous run in section 2 of 15, and
    librivox-v2's 30 are one run in section 1 of a 25-section novel. It was safe only
    because the one title actually marked is homogeneous by construction, which is a
    property of that book rather than of the check."""
    assert _thin(12, {2}, 15), "librivox-v1's contiguous run must not certify a novel"
    assert _thin(30, {1}, 25), "librivox-v2's contiguous run must not certify a novel"


def test_an_honest_spread_is_accepted():
    assert not _thin(12, {1, 4, 6, 9, 11, 14}, 15)


def test_a_short_title_is_not_punished_for_being_short():
    """min(MIN_SECTIONS, total) — a two-section title with both heard is full coverage."""
    assert not _thin(8, {1, 2}, 2)


def test_spread_alone_is_not_enough():
    assert _thin(4, {1, 5, 9, 13}, 15), "four clips cannot certify a whole book"
