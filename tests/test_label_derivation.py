"""Regression coverage for the §8 label-derivation fixes (2026-08-06).

Same family as everything else in this review: none of these raised. A head that was never
scored, a clip that failed once and was never retried, a z-score fixed by arithmetic rather
than measured, a digit deleted from a transcript — each yields a corpus that looks complete
and is quietly wrong, in units indistinguishable from real measurements.
"""

import json
import os
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, str(REPO))

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


# --- Phase 1 #1: the Emilia merge (2026-08-08) -----------------------------------------
#
# The merge's whole safety argument is that v4's rows and speaker indices survive it
# untouched, because `spk_emb` row *i* is a person and renumbering silently reassigns every
# voice the model has learned. These check the corpus ON DISK rather than a stub: the merge
# is a one-off derivation, and the artifact is what a run will read.

MERGED = os.path.join(os.path.dirname(SCRIPTS), "data", "libritts_r_emilia_vat_v5")
BASE = os.path.join(os.path.dirname(SCRIPTS), "data", "libritts_r_vat_v4")

pytestmark_merged = pytest.mark.skipif(
    not os.path.isdir(MERGED), reason="the merged corpus is not on this machine")


def _rows(corpus, name):
    with open(os.path.join(corpus, name), encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


@pytestmark_merged
@pytest.mark.parametrize("name", ["train_op.txt", "val_op.txt"])
def test_the_base_corpus_survives_the_merge_verbatim(name):
    """Not "the same clips" — the same BYTES, in the same order.

    A row carries the speaker INDEX, so a re-derivation that produced identical clips with
    a renumbered speaker table would pass a set comparison and invalidate the warm start.
    Prefix equality is the property `make_warmstart --donor-speakers` proves, checked here
    against the artifact rather than against the script's intent.
    """
    base, merged = _rows(BASE, name), _rows(MERGED, name)
    assert merged[:len(base)] == base


@pytestmark_merged
def test_nothing_held_out_became_trainable():
    """The hash split's promise is that a clip's side is a property of the clip, so growth
    never moves one. v4's val clips must still be val — otherwise every comparison against
    a v4-lineage checkpoint quietly leaks."""
    val = {r.split("|", 1)[0] for r in _rows(BASE, "val_op.txt")}
    train = {r.split("|", 1)[0] for r in _rows(MERGED, "train_op.txt")}
    assert not val & train


@pytestmark_merged
def test_the_speaker_table_only_ever_grew():
    """LibriTTS ids keep their indices, the appended ones are contiguous above them, and no
    two speakers share an embedding row."""
    with open(os.path.join(BASE, "speakers.json"), encoding="utf-8") as fh:
        base = json.load(fh)
    with open(os.path.join(MERGED, "speakers.json"), encoding="utf-8") as fh:
        merged = json.load(fh)
    assert merged["libritts_id_to_index"] == base["libritts_id_to_index"]
    every = {**merged["libritts_id_to_index"], **merged["emilia_id_to_index"]}
    assert len(every) == merged["n_spks"], "an id appears in both namespaces"
    assert sorted(every.values()) == list(range(merged["n_spks"]))
    assert min(merged["emilia_id_to_index"].values()) == base["n_spks"]


@pytestmark_merged
def test_no_merged_row_carries_a_digit_or_the_wrong_width():
    """D-M3 at the artifact. The tokenizer DELETES digits, so a surviving digit means a
    clip whose transcript is missing a word its audio speaks — undetectable downstream,
    which is why it is checked where it would land rather than only where it is filtered.

    ⚠ TR-M1: the phoneme half of this assertion CANNOT FAIL, by construction — `_TOKEN_RE`
    deletes digits, so they are gone from the IPA whether or not the filter caught them.
    It is kept as a structural pin, and the real check lives in
    `test_the_digit_filter_asks_what_the_tokenizer_will_see` below, on the INPUT side.
    """
    for name in ("train_op.txt", "val_op.txt"):
        for row in _rows(MERGED, name):
            fields = row.split("|")
            assert len(fields) == 4
            assert len(fields[3].split(",")) == 8, "conditioning must be contract-v2 wide"
            assert not any(c.isdigit() for c in fields[2]), "a digit reached the phonemes"


def test_the_digit_filter_asks_what_the_tokenizer_will_see():
    """TR-M1. The rule existed in three spellings with three answers, and NONE of them ran
    on the string the tokenizer sees:

        merge     re.search(r"[0-9]", raw)          — misses full-width, superscript, ½
        derive    any(c.isdigit() for c in raw)     — misses ½ (that is `isnumeric`)
        validate  looks at the IPA                  — where digits can never appear

    All three ran BEFORE `convert_to_ascii`, and unidecode MANUFACTURES ASCII digits:
    `１００` -> `100`, `²` -> `2`, `½` -> ` 1/2`. So a caption passed the filter, the
    tokenizer deleted the numeral, and the clip trained with a transcript missing a word
    its audio speaks — the exact defect the drop exists to prevent, shipped through it.
    """
    op_g2p = pytest.importorskip("matcha.text.op_g2p")
    for text in ("chapter 100", "chapter １００", "a ² power", "half a ½ measure",
                 "2 o'clock"):
        assert op_g2p.carries_digits(text), text
        # ...and each of these really does lose a token to `_TOKEN_RE`.
        assert not any(t.isdigit() for t in
                       op_g2p._TOKEN_RE.findall(op_g2p.normalize_for_tokens(text)))
    for text in ("plain words only", "chapter Ⅻ", "a hyphen-joined word"):
        assert not op_g2p.carries_digits(text), text


def test_the_old_spellings_are_gone_from_both_lanes():
    """One rule, one implementation — beside the tokenizer that does the deleting."""
    for name in ("merge_emilia_corpus.py", "derive_vat_corpus.py"):
        src = (REPO / "scripts" / name).read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        assert "op_g2p.carries_digits(" in code, f"{name} does not use the shared rule"
        assert 'search(r"[0-9]"' not in code, f"{name} still carries its own spelling"
        assert "any(c.isdigit() for c in text)" not in code


def test_normalization_and_tokenization_have_not_drifted_apart():
    """`carries_digits` is only correct while it runs the SAME prefix `phonemize` does.
    Factoring it out is what makes that true; this is what keeps it true."""
    op_g2p = pytest.importorskip("matcha.text.op_g2p")
    src = pathlib.Path(op_g2p.__file__).read_text(encoding="utf-8")
    assert "text = normalize_for_tokens(text)" in src, "phonemize forked its own prefix"
    assert src.count("convert_to_ascii(text)") == 1


@pytestmark_merged
def test_every_emilia_row_carries_a_label_and_no_delivery_lane():
    """The failure the global anchor exists to prevent is an all-zero V/A/T — a clip
    selected FOR being extreme, trained as neutral (D-M1's principle). And Emilia rows must
    stay delivery-`unknown`: nobody has heard one, so a lane there would be a guess."""
    base_n = len(_rows(BASE, "train_op.txt"))
    emilia = _rows(MERGED, "train_op.txt")[base_n:]
    assert emilia, "the merge added no rows"
    for row in emilia:
        vat = [float(x) for x in row.split("|")[3].split(",")]
        assert any(abs(v) > 1e-9 for v in vat[:3]), "a keep labelled all-zero"
        assert sum(vat[3:]) == 0, "an Emilia row claims a delivery lane"


def test_one_wer_threshold_serves_both_lanes():
    """The corpus filter and the QC gate ask the same question of a transcript, and two
    copies of one threshold is the review's most repeated finding (B-L5, D-L2). `qc_gate`
    re-exports `synth_common`'s rather than declaring its own."""
    sys.path.insert(0, os.path.join(SCRIPTS, "synthesis"))
    sc = pytest.importorskip("synth_common")
    assert isinstance(sc.ASR_MAX_WER, float)
    gate = open(os.path.join(SCRIPTS, "synthesis", "qc_gate.py"), encoding="utf-8").read()
    assert "ASR_MAX_WER = synth_common.ASR_MAX_WER" in gate, "the threshold forked again"


# --- TR-M2: the anchor lane had no non-finite guard -----------------------------------


ANCHOR = {"lufs": (-20.0, 3.0), "alpha_db": (0.0, 1.0), "cpp": (0.0, 1.0),
          "h1h2": (0.0, 1.0), "head:Soft_vs._Harsh": (0.0, 1.0),
          "head:Amusement": (0.0, 1.0), "head:Valence": (0.0, 1.0),
          "t_composite": (0.0, 1.0), "v_composite": (0.0, 1.0),
          "weights": {"Amusement": 0.4, "Valence": 0.6}}
ROW = {"lufs": -20.0, "alpha_db": 0.1, "cpp": 0.2, "h1h2": 0.3,
       "head:Soft_vs._Harsh": 0.1, "head:Amusement": 0.2, "head:Valence": -0.1}


def _anchor_mod():
    return pytest.importorskip("anchor_emilia_labels")


def test_a_clean_row_still_labels():
    mod = _anchor_mod()
    v, a, t = mod.label(dict(ROW), ANCHOR)
    assert all(-1.0 <= x <= 1.0 for x in (v, a, t))


@pytest.mark.parametrize("field", ["lufs", "cpp", "head:Valence"])
def test_a_single_nan_measure_refuses_rather_than_shipping(field):
    """TR-M2. The derive lane aborts on a non-finite z for exactly this reason and this
    lane had no check, though it labels the same three channels of the same corpus.

    A NaN here is worse than a crash: `np.clip` propagates it, the filelist carries the
    string `"nan"`, `float("nan")` parses CLEANLY at load, and the poisoned conditioning
    vector reaches the FiLM trunk with nothing having failed anywhere. The artifact test
    only catches the all-three-NaN case, so a single-channel NaN passed everything.
    """
    mod = _anchor_mod()
    row = dict(ROW, **{field: float("nan")})
    with pytest.raises(mod.LabelError, match="non-finite"):
        mod.label(row, ANCHOR)


def test_the_clamp_is_why_a_nan_is_not_obvious():
    """Kept as the record of why this needed a guard rather than being self-evident:
    `np.clip(nan, -1, 1)` is `nan`, and `float(nan)` formats as a perfectly ordinary
    `nan` in a `%.4f` field — the row looks like every other row."""
    assert np.isnan(np.clip(float("nan"), -1.0, 1.0))
    assert f"{float('nan'):.4f}" == "nan"
    assert np.isnan(float("nan")), "and it parses straight back at load time"


def test_a_missing_weighted_head_refuses_instead_of_biasing_v():
    """The quieter half. Skipping a head the row does not carry does not fail — it biases
    V toward 0 in units indistinguishable from a measurement. `mine_emilia_keeps` checks
    one sample row per tar, so a degraded EIV run is invisible to it at row granularity."""
    mod = _anchor_mod()
    row = dict(ROW)
    del row["head:Valence"]                        # weight 0.6 — the dominant term
    with pytest.raises(mod.LabelError, match="weighted head"):
        mod.label(row, ANCHOR)


def test_the_merge_drops_an_unlabelable_clip_rather_than_dying():
    """A corpus build is 13k clips long; one bad row must cost that row, not the run."""
    src = (REPO / "scripts" / "merge_emilia_corpus.py").read_text(encoding="utf-8")
    assert "except anchor_mod.LabelError" in src
    assert 'drop("unlabelable"' in src


def test_an_asr_error_row_names_the_clip_it_dropped():
    """TR-L1. A clip whose transcription ERRORED carries `error` and no `wer`, so it is
    neither absent nor droppable by the `rec is None` test — `rec["wer"]` raised a bare
    `KeyError: 'wer'` late in a ~13k-clip run, naming no clip, from a script whose every
    other refusal says which clip and why."""
    src = (REPO / "scripts" / "merge_emilia_corpus.py").read_text(encoding="utf-8")
    assert 'if "wer" not in rec:' in src
    i, j = src.index('if "wer" not in rec:'), src.index('if rec["wer"] > ASR_MAX_WER')
    assert i < j, "the guard must precede the read it protects"
