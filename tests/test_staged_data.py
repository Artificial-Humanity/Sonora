"""The pure half of the staged rebuild's corpus preparation (`scripts/lib/staged_data.py`).

C0 must be stock fine-tuned on stock's own inputs, so its phonemes must reach the SAME
embedding ids stock learned; S1 must be the derisk corpus with only the audio path changed.
These pin both.
"""
import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import baseline_bench as bb  # noqa: E402
import staged_data as sd  # noqa: E402


def _ours_ids(text):
    # exactly how matcha.text builds its table: {s: i for i, s in enumerate(symbols)}
    table = {s: i for i, s in enumerate(bb.ours_symbols())}
    return [table[c] for c in text]


# ---------------------------------------------------------------- C0 symbols

def test_remapped_text_encodes_to_upstream_ids_for_every_symbol():
    every = "".join(sorted(set(bb.original_matcha_symbols())))
    assert _ours_ids(sd.upstream_as_ours(every)) == bb.encode_original(every)


def test_only_the_two_differing_slots_are_rewritten():
    moved = {c: sd.upstream_as_ours(c) for c in sorted(set(bb.original_matcha_symbols()))
             if sd.upstream_as_ours(c) != c}
    assert moved == {"'": "ᵻ", "ᵻ": "ᵊ"}


def test_a_symbol_upstream_lacks_is_refused():
    with pytest.raises(ValueError, match="ᵊ"):
        sd.upstream_as_ours("aᵊ")


# ---------------------------------------------------------------- C0 filelist

def test_vits_rows_keep_file_order_and_split_on_the_first_two_bars(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("DUMMY2/p282/p282_147.wav|83|hiː|x\nDUMMY2/p225/p225_001.wav|0|ðə\n",
                 encoding="utf-8")
    assert sd.read_vits(p) == [("p282_147", "p282", 83, "hiː|x"),
                               ("p225_001", "p225", 0, "ðə")]


def test_c0_lines_drop_and_count_rows_without_a_recording_never_substitute():
    rows = [("p225_001", "p225", 0, "ðə"), ("p225_002", "p225", 0, "ɐ'")]
    lines, missing = sd.c0_lines(rows, have={"p225_002"})
    assert missing == ["p225_001"]
    assert lines == ["%s/p225/p225_002.wav|0|ɐᵻ" % sd.VCTK_22K_ROOT]


# ---------------------------------------------------------------- S1 filelist

def test_s1_line_moves_the_audio_and_drops_only_the_vat_field():
    w = sd.LIBRI_24K_ROOT + "/1263/139804/1263_139804_000006_000000.wav"
    got = sd.s1_line("%s|63|ɪf juː|0.0,-0.5573,0.0" % w)
    assert got == "%s/1263/139804/1263_139804_000006_000000.wav|63|ɪf juː" % sd.LIBRI_22K_ROOT


def test_a_row_from_another_root_is_refused():
    with pytest.raises(ValueError, match="not under"):
        sd.to_22k("/data/model-training/datasets/LibriTTS_R/train-other-500/1/2/x.wav")


def test_a_row_without_its_vat_field_is_refused():
    with pytest.raises(ValueError, match="4"):
        sd.s1_line(sd.LIBRI_24K_ROOT + "/1/2/x.wav|5|ab")


# ---------------------------------------------------------------- run length

def test_steps_per_epoch_reproduces_the_derisk_run():
    # derisk-energy: 29,441 training rows, 92,100 steps over 100 epochs
    assert sd.steps_per_epoch(29441) == 921


def test_the_stock_mel_is_upstream_vctks():
    assert dict(zip(sd.MEL_KEYS, sd.STOCK_MEL)) == {
        "n_fft": 1024, "n_feats": 80, "sample_rate": 22050, "hop_length": 256,
        "win_length": 1024, "f_min": 0, "f_max": 8000}


# ---------------------------------------------------------------- L filelist (check 4)

def test_l_line_moves_only_the_audio_to_the_originals_22k_copy():
    rel = "/1263/139804/1263_139804_000006_000000.wav"
    got = sd.l_line("%s%s|63|ɪf juː" % (sd.LIBRI_22K_ROOT, rel))
    assert got == "%s%s|63|ɪf juː" % (sd.LIBRI_ORIG_22K_ROOT, rel)


def test_an_l_row_not_from_s1s_corpus_is_refused():
    with pytest.raises(ValueError, match="not under"):
        sd.l_line(sd.LIBRI_24K_ROOT + "/1/2/x.wav|5|ab")
    with pytest.raises(ValueError, match="3 fields"):
        sd.l_line(sd.LIBRI_22K_ROOT + "/1/2/x.wav|5|ab|0,0,0")


def test_every_corpus_path_of_one_utterance_shares_its_relative_path():
    w = sd.LIBRI_24K_ROOT + "/19/198/19_198_000000_000000.wav"
    assert sd.to_original(w) == sd.LIBRI_ORIG_ROOT + "/19/198/19_198_000000_000000.wav"
    assert sd.to_l22k(w) == sd.LIBRI_ORIG_22K_ROOT + "/19/198/19_198_000000_000000.wav"
    assert sd.original_of(sd.to_22k(w)) == sd.to_original(w)
    with pytest.raises(ValueError, match="not under"):
        sd.to_original(sd.LIBRI_22K_ROOT + "/19/198/x.wav")


def test_l_lines_drop_and_count_rows_without_an_original_never_substitute():
    a = sd.LIBRI_22K_ROOT + "/1/2/1_2_a.wav"
    b = sd.LIBRI_22K_ROOT + "/1/2/1_2_b.wav"
    lines, missing = sd.l_lines(["%s|0|ab" % a, "%s|0|cd" % b],
                                have={sd.original_of(b)})
    assert missing == [a]
    assert lines == ["%s/1/2/1_2_b.wav|0|cd" % sd.LIBRI_ORIG_22K_ROOT]


@pytest.mark.parametrize("missing,rows,refuse", [
    (0, 30351, False), (303, 30351, False), (304, 30351, True), (1, 99, True), (0, 0, True)])
def test_more_than_one_percent_without_an_original_refuses_l(missing, rows, refuse):
    assert sd.too_many_missing(missing, rows) is refuse


def test_an_s1_rows_l_copy_is_its_originals_22k_copy():
    w = sd.LIBRI_24K_ROOT + "/19/198/19_198_000000_000000.wav"
    assert sd.l22k_of(sd.to_22k(w)) == sd.to_l22k(w)


# ---------------------------------------------------------------- D1 (hum-fix batch)
# Pre-registered in Notes: Sonora/hum-fix-batch-preregistration.md.

def _vctk(spk, n, sid):
    return ["%s/%s/%s_%03d.wav|%d|t%d" % (sd.VCTK_22K_ROOT, spk, spk, i, sid, i)
            for i in range(n)]


def _spec_d1(lines, per, seed):
    """The registered rule, spelled as the pre-registration states it."""
    import random
    rng = random.Random(seed)
    order, rows = [], {}
    for l in lines:
        s = int(l.split("|")[1])
        if s not in rows:
            order.append(s)
            rows[s] = []
        rows[s].append(l)
    kept = set()
    for s in order:
        kept.update(rng.sample(rows[s], per))
    return [l for l in lines if l in kept]


def test_d1_keeps_exactly_per_speaker_rows_in_file_order_as_registered():
    a, b, c = _vctk("p225", 9, 0), _vctk("p226", 7, 1), _vctk("p227", 12, 2)
    lines = [b[0], a[0]] + a[1:5] + c + b[1:] + a[5:]     # interleaved; p226 appears first
    got = sd.d1_subsample(lines, per=5, seed=sd.D1_SEED)
    assert got == _spec_d1(lines, 5, sd.D1_SEED)
    assert len(got) == 15
    assert [lines.index(l) for l in got] == sorted(lines.index(l) for l in got)
    assert {int(l.split("|")[1]) for l in got} == {0, 1, 2}


def test_d1_depends_on_the_order_speakers_first_appear():
    a, b = _vctk("p225", 9, 0), _vctk("p226", 9, 1)
    ab, ba = sd.d1_subsample(a + b, per=4), sd.d1_subsample(b + a, per=4)
    assert set(ab) != set(ba)          # one rng: whoever comes first draws first


def test_d1_refuses_a_speaker_short_of_the_quota():
    with pytest.raises(ValueError, match="p226"):
        sd.d1_subsample(_vctk("p225", 9, 0) + _vctk("p226", 3, 1), per=5)


def test_d1_refuses_a_speaker_id_shared_by_two_voices():
    with pytest.raises(ValueError, match="sid"):
        sd.d1_subsample(_vctk("p225", 9, 0) + _vctk("p226", 9, 0), per=5)


def test_d1_constants_are_the_registered_ones():
    assert (sd.D1_SEED, sd.D1_PER_SPEAKER) == (20261012, 105)
    assert sd.steps_per_epoch(108 * 105) == 355
    assert 26 * 355 == 9230


# ---------------------------------------------------------------- G′ (hum-fix batch)

def test_g_moves_s1s_schwa_bar_to_stocks_trained_id():
    assert _ours_ids("ᵻ") == [176] and _ours_ids("ᵊ") == [177]
    assert bb.encode_original("ᵻ") == [177]           # stock trained ᵻ at 177
    w = sd.LIBRI_22K_ROOT + "/1/2/1_2_000000_000000.wav"
    line, n = sd.g_line("%s|63|pɚmˈɪɾᵻd ᵻ" % w)
    assert line == "%s|63|pɚmˈɪɾᵊd ᵊ" % w and n == 2


def test_g_equals_upstream_as_ours_or_refuses():
    w = sd.LIBRI_22K_ROOT + "/1/2/x.wav"
    # an apostrophe is the one symbol the two rules map differently
    with pytest.raises(ValueError, match="upstream_as_ours"):
        sd.g_line("%s|0|ɪt's" % w)
    with pytest.raises(ValueError, match="upstream_as_ours"):
        sd.g_line("%s|0|ᵊ" % w)          # ᵊ is not in upstream's table at all
    with pytest.raises(ValueError, match="3 fields"):
        sd.g_line("%s|0|ab|0,0,0" % w)


def test_g_leaves_a_row_without_the_symbol_alone():
    w = sd.LIBRI_22K_ROOT + "/1/2/x.wav"
    assert sd.g_line("%s|5|ðə kˈæt" % w) == ("%s|5|ðə kˈæt" % w, 0)


# ---------------------------------------------------------------- F1 (hum-fix batch)

def _r(subset, spk, k=0):
    return "%s/%s/%d/77/%d_77_%06d_000000.wav" % (sd.LIBRI_R_ROOT, subset, spk, spk, k)


def test_f1_22k_copies_sit_beside_each_subset_and_clean_100_is_s1s():
    for s in sd.F1_SUBSETS:
        w = _r(s, 19)
        assert sd.libri_r_22k(w) == w.replace("/%s/" % s, "/%s_22k/" % s)
    assert sd.libri_r_22k(_r("train-clean-100", 19)) == sd.to_22k(_r("train-clean-100", 19))


@pytest.mark.parametrize("bad", [
    "/data/model-training/datasets/LibriTTS_R/dev-clean/19/77/19_77_000000_000000.wav",
    "/data/model-training/datasets/LibriTTS/train-clean-360/19/77/19_77_000000_000000.wav",
    "/data/model-training/datasets/LibriTTS_R/train-clean-360_22k/19/77/19_77_0_0.wav",
    "/elsewhere/LibriTTS_R/train-other-500/19/77/19_77_000000_000000.wav",
])
def test_f1_22k_refuses_a_foreign_root(bad):
    with pytest.raises(ValueError, match="not under"):
        sd.libri_r_22k(bad)


def test_libritts_id_is_read_from_the_path_and_checked_against_the_file_name():
    assert sd.libritts_id(_r("train-other-500", 8088, 3)) == 8088
    with pytest.raises(ValueError, match="file name"):
        sd.libritts_id(sd.LIBRI_R_ROOT + "/train-other-500/8088/77/19_77_000000_000000.wav")


def test_the_length_filter_is_s1s_inclusive_one_to_sixteen_seconds():
    assert [sd.f1_length_ok(s) for s in (0.99, 1.0, 8.0, 16.0, 16.01)] == \
        [False, True, True, True, False]


def test_f1_speakers_are_those_with_the_minimum_train_rows():
    wavs = [_r("train-clean-360", 40, k) for k in range(3)] + \
           [_r("train-other-500", 7, k) for k in range(2)] + [_r("train-clean-100", 9)]
    assert sd.f1_dense_speakers(wavs, min_rows=2) == [7, 40]
    assert sd.F1_MIN_ROWS == 300


def test_f1_table_is_ascending_integer_id_order_with_each_v7_index():
    t = sd.f1_table([1088, 205, 98], {"98": 16, "205": 40, "1088": 3})
    assert t == {"n_spks": 3, "libritts_id_to_index": {"98": 0, "205": 1, "1088": 2},
                 "v7_index": {"98": 16, "205": 40, "1088": 3}}
    with pytest.raises(ValueError, match="v7"):
        sd.f1_table([98, 99], {"98": 16})


def test_f1_line_drops_the_vat_field_and_reindexes():
    w = _r("train-other-500", 205, 4)
    got = sd.f1_line("%s|17|ðə kˈæt|0.1,0.2,0.3,0,0,0,0,0" % w, {"205": 1})
    assert got == "%s|1|ðə kˈæt" % sd.libri_r_22k(w)
    with pytest.raises(ValueError, match="4 fields"):
        sd.f1_line("%s|17|ðə" % w, {"205": 1})
    with pytest.raises(KeyError):
        sd.f1_line("%s|17|ðə|0,0,0" % _r("train-other-500", 9), {"205": 1})


def test_a_source_rows_speaker_column_must_name_its_path_speaker():
    w = _r("train-clean-360", 205)
    sd.check_speaker_column("%s|17|x|0,0,0" % w, {17: "205"})
    with pytest.raises(ValueError, match="205"):
        sd.check_speaker_column("%s|17|x|0,0,0" % w, {17: "206"})


def test_f1_run_length_is_the_registered_one():
    assert sd.steps_per_epoch(31287) == 978 and 10 * 978 - 1 == 9779


# ---------------------------------------------------------------- E (hum-fix check D)

def test_s1_transcript_points_to_the_24k_normalized_txt_file():
    rel = "/1263/139804/1263_139804_000006_000000"
    assert sd.s1_transcript(sd.LIBRI_22K_ROOT + rel + ".wav") == \
        sd.LIBRI_24K_ROOT + rel + ".normalized.txt"
    with pytest.raises(ValueError, match="not under"):
        sd.s1_transcript(sd.LIBRI_24K_ROOT + rel + ".wav")


def test_e_line_remaps_espeak_phonemes_to_upstream_ids_and_checks_row():
    w = sd.LIBRI_22K_ROOT + "/1263/139804/1263_139804_000006_000000.wav"
    espeak = "pɚmˈɪɾᵻd ɪt."
    got, ds, dg = sd.e_line("%s|63|pɚmˈɪɾᵻd ɪt" % w, espeak)
    assert got == "%s|63|pɚmˈɪɾᵊd ɪt." % w and (ds, dg) == (1, 1)
    assert _ours_ids(got.split("|")[2]) == bb.encode_original(espeak)
    assert sd.G_FROM not in got.split("|")[2]
    assert sd.e_line("%s|63|pɚmˈɪɾᵻd" % w, "pɚmˈɪɾᵻd") == ("%s|63|pɚmˈɪɾᵊd" % w, 1, 0)
    assert sd.e_line("%s|63|ɪt" % w, "ɪt") == ("%s|63|ɪt" % w, 0, 0)


def test_e_line_refuses_empty_phonemes_or_bad_rows():
    w = sd.LIBRI_22K_ROOT + "/1/2/x.wav"
    with pytest.raises(ValueError, match="empty"):
        sd.e_line("%s|0|ab" % w, "")
    with pytest.raises(ValueError, match="not under"):
        sd.e_line(sd.LIBRI_24K_ROOT + "/1/2/x.wav|0|ab", "ab")
    with pytest.raises(ValueError, match="3 fields"):
        sd.e_line("%s|0|ab|0,0,0" % w, "ab")
    with pytest.raises(ValueError, match="ᵊ"):
        sd.e_line("%s|0|ab" % w, "aᵊ")
    with pytest.raises(ValueError, match="id 176"):
        sd.e_line("%s|0|ab" % w, "a'b")
