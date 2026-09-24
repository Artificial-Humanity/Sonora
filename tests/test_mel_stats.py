"""The mel-statistics correction on checkpoint load (matcha/mel_stats.py)."""

import os

from matcha.mel_stats import corrected_mel_buffers

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VAT7_HPARAMS = {"data_statistics": {"mel_mean": -5.543695, "mel_std": 2.430293}}
VAT7_BUFFERS = {"mel_mean": -6.630575180053711, "mel_std": 2.4829139709472656, "w": 0}


def test_the_measured_vat7_mismatch_is_corrected_to_the_training_config():
    assert corrected_mel_buffers(VAT7_HPARAMS, VAT7_BUFFERS) == {
        "mel_mean": -5.543695, "mel_std": 2.430293}


def test_agreeing_statistics_are_left_alone():
    """The positive control: a from-scratch checkpoint must not be touched, or the warning
    fires on every load and stops meaning anything."""
    sd = {"mel_mean": -5.543695, "mel_std": 2.430293}
    assert corrected_mel_buffers(VAT7_HPARAMS, sd) == {}


def test_one_disagreeing_value_is_enough():
    sd = {"mel_mean": -5.543695, "mel_std": 1.0}
    assert corrected_mel_buffers(VAT7_HPARAMS, sd)["mel_std"] == 2.430293


def test_nothing_to_correct_without_statistics_or_buffers():
    assert corrected_mel_buffers({}, VAT7_BUFFERS) == {}
    assert corrected_mel_buffers(None, VAT7_BUFFERS) == {}
    assert corrected_mel_buffers(VAT7_HPARAMS, None) == {}
    assert corrected_mel_buffers(VAT7_HPARAMS, {"w": 0}) == {}


def test_the_load_hook_applies_the_correction():
    """No torch on this host, so the wiring is pinned by source: the hook must call the
    helper with the LOADING module's hparams — the donor's would reproduce the bug on every
    warm start — and write into the checkpoint's state dict before Lightning loads it."""
    with open(os.path.join(REPO, "matcha/models/baselightningmodule.py"), encoding="utf-8") as f:
        src = f.read()
    hook = src[src.index("def on_load_checkpoint"):src.index("def on_save_checkpoint")]
    assert 'correct_state_dict(getattr(self, "hparams", None), checkpoint.get("state_dict"))' \
        in hook


def test_correct_state_dict_edits_in_place_and_reports_the_change():
    from matcha.mel_stats import correct_state_dict

    sd = dict(VAT7_BUFFERS)
    changed = correct_state_dict(VAT7_HPARAMS, sd)
    assert sd["mel_mean"] == -5.543695 and sd["mel_std"] == 2.430293
    assert changed["mel_mean"] == (VAT7_BUFFERS["mel_mean"], -5.543695)
    assert correct_state_dict(VAT7_HPARAMS, sd) == {}, "a second pass must be a no-op"


def test_a_state_dict_missing_one_buffer_is_left_alone():
    assert corrected_mel_buffers(VAT7_HPARAMS, {"mel_mean": -6.6}) == {}


def test_every_bare_load_that_vocodes_applies_the_correction():
    """The hook covers Lightning's two doors only. These three load with a bare
    `load_state_dict` AND turn the mel back into audio, so each must correct first."""
    for rel in ("scripts/stages/measure_harmonicity.py", "scripts/tools/render_vat_sweep.py",
                "scripts/tools/render_guidance_demo.py"):
        with open(os.path.join(REPO, rel), encoding="utf-8") as f:
            src = f.read()
        i = src.index("correct_state_dict(")
        assert i < src.index("load_state_dict(", i), rel + " corrects after it loads"
