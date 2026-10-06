"""The staged rebuild's configs say what the pre-registration says (YAML level, host).

The composed-config proof (Hydra, model hparams, warm-start load) is the container gate
`scripts/gates/test_staged_experiments.py`; this pins the files themselves.
"""
import json
from pathlib import Path

import pytest
import yaml

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import staged_data as sd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA_CFG = REPO / "configs" / "data"
EXP = REPO / "configs" / "experiment"


def y(p):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


# arm: (data config, use_vat, vat_dim or None, data.load_vat or None)
ARMS = {
    "staged_r": ("libritts_r_vat", True, 3, None),
    "staged_c0": ("vctk_22k", False, None, None),
    "staged_s1": ("libritts_r_22k", False, None, None),
    "staged_s2": ("libritts_r_vat", False, None, False),
}


@pytest.mark.parametrize("name", sorted(ARMS))
def test_each_arm_is_a_ten_epoch_run_on_its_corpus(name):
    data, use_vat, vat_dim, load_vat = ARMS[name]
    c = y(EXP / ("%s.yaml" % name))
    assert c["defaults"] == [{"override /data": "%s.yaml" % data}]
    assert c["run_name"] == name
    assert c["trainer"]["max_epochs"] == 10
    assert c["model"]["use_vat"] is use_vat
    assert c["model"].get("vat_dim") == vat_dim
    assert (c.get("data") or {}).get("load_vat") == load_vat
    assert c["data"]["bucket_multiplier"] == 0      # owner ruling: July's shuffle, every arm


def test_c0_is_stocks_own_pipeline_with_pre_mapped_phonemes():
    c = y(DATA_CFG / "vctk_22k.yaml")
    assert c["defaults"] == ["vctk", "_self_"]
    assert c["cleaners"] == ["no_cleaners"]          # phonemes are already upstream ids
    assert c["train_filelist_path"] == "data/vctk_22k/train.txt"
    base = y(DATA_CFG / "vctk.yaml")
    assert base["n_spks"] == 109
    assert base["data_statistics"] == {"mel_mean": -6.630575, "mel_std": 2.482914}
    lj = y(DATA_CFG / "ljspeech.yaml")
    assert {k: lj[k] for k in sd.MEL_KEYS} == dict(zip(sd.MEL_KEYS, sd.STOCK_MEL))


def test_s1_is_the_derisk_corpus_on_stocks_mel():
    c = y(DATA_CFG / "libritts_r_22k.yaml")
    assert {k: c[k] for k in sd.MEL_KEYS} == dict(zip(sd.MEL_KEYS, sd.STOCK_MEL))
    assert c["n_spks"] == 247 and c["load_vat"] is False
    assert c["cleaners"] == ["no_cleaners"]
    assert c["train_filelist_path"] == "data/libritts_r_22k/train_op.txt"


@pytest.mark.skipif(not Path("/data/model-training/sonora/data/libritts_r_22k").is_dir(),
                    reason="/data not mounted")
def test_s1_statistics_are_the_measured_ones():
    got = json.loads((REPO / "data/libritts_r_22k/mel_statistics.json").read_text())
    want = y(DATA_CFG / "libritts_r_22k.yaml")["data_statistics"]
    assert want["mel_mean"] == pytest.approx(got["mel_mean"], abs=1e-6)
    assert want["mel_std"] == pytest.approx(got["mel_std"], abs=1e-6)
