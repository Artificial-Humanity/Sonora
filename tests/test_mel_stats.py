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
    assert 'corrected_mel_buffers(getattr(self, "hparams", None), sd)' in hook
    assert 'checkpoint.get("state_dict")' in hook
    assert "sd[k] = torch.tensor(v" in hook
