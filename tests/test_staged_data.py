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
