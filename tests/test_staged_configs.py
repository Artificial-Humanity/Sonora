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
    "staged_l": ("libritts_22k", False, None, None),
    # the hum-fix batch (Notes: Sonora/hum-fix-batch-preregistration.md)
    "staged_d1": ("vctk_d1", False, None, None),
    "staged_g": ("libritts_r_22k_g", False, None, None),
    "staged_f1": ("f1_dense", False, None, None),
    "staged_e": ("libritts_r_22k_e", False, None, None),
    # Step 2c's Arm Q (Notes: Sonora/phase-1s-step1-preregistration.md § Step 2)
    "staged_q": ("q_synth", False, None, None),
}
# max_epochs where it is not 10: D1 is matched to S1 in optimizer steps (26 x 355 = 9,230)
EPOCHS = {"staged_d1": 26}


@pytest.mark.parametrize("name", sorted(ARMS))
def test_each_arm_is_a_registered_length_run_on_its_corpus(name):
    data, use_vat, vat_dim, load_vat = ARMS[name]
    c = y(EXP / ("%s.yaml" % name))
    assert c["defaults"] == [{"override /data": "%s.yaml" % data}]
    assert c["run_name"] == name
    assert c["trainer"]["max_epochs"] == EPOCHS.get(name, 10)
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
    got = json.loads(Path("/data/model-training/sonora/data/libritts_r_22k/mel_statistics.json")
                     .read_text())
    want = y(DATA_CFG / "libritts_r_22k.yaml")["data_statistics"]
    assert want["mel_mean"] == pytest.approx(got["mel_mean"], abs=1e-6)
    assert want["mel_std"] == pytest.approx(got["mel_std"], abs=1e-6)


# ---------------------------------------------------------------- arm L (check 4)

def test_l_is_s1_but_for_the_audio_and_its_statistics():
    s1, l_ = y(DATA_CFG / "libritts_r_22k.yaml"), y(DATA_CFG / "libritts_22k.yaml")
    differ = {k for k in set(s1) | set(l_) if s1.get(k) != l_.get(k)}
    assert differ == {"name", "train_filelist_path", "valid_filelist_path", "data_statistics"}
    assert l_["name"] == "libritts_22k"
    assert l_["train_filelist_path"] == "data/libritts_22k/train_op.txt"
    assert l_["valid_filelist_path"] == "data/libritts_22k/val_op.txt"


def test_l_is_s1s_experiment_on_ls_data_and_warm_starts_from_s1s_own_init():
    s1, l_ = y(EXP / "staged_s1.yaml"), y(EXP / "staged_l.yaml")
    assert {k: v for k, v in l_.items() if k not in ("defaults", "tags", "run_name")} == \
        {k: v for k, v in s1.items() if k not in ("defaults", "tags", "run_name")}
    text = (EXP / "staged_l.yaml").read_text(encoding="utf-8")
    assert "warmstart/staged_s1_init.ckpt" in text and "byte for byte" in text
    assert "67dc8c6a3d1c3d94eb8cf1698e8361c0161ee292514af7e628335917557ef434" in text


@pytest.mark.skipif(not Path("/data/model-training/sonora/data/libritts_22k").is_dir(),
                    reason="/data not mounted")
def test_l_statistics_are_the_measured_ones():
    got = json.loads(Path("/data/model-training/sonora/data/libritts_22k/mel_statistics.json")
                     .read_text())
    want = y(DATA_CFG / "libritts_22k.yaml")["data_statistics"]
    assert want["mel_mean"] == pytest.approx(got["mel_mean"], abs=1e-6)
    assert want["mel_std"] == pytest.approx(got["mel_std"], abs=1e-6)


def test_l_defaults_are_s1s_but_for_the_data_override():
    s1, l_ = y(EXP / "staged_s1.yaml")["defaults"], y(EXP / "staged_l.yaml")["defaults"]
    keep = lambda d: [x for x in d if not (isinstance(x, dict) and "override /data" in x)]
    assert keep(l_) == keep(s1)
    assert len(l_) == len(s1)
    assert sum(isinstance(x, dict) and "override /data" in x for x in l_) == 1
    assert l_ != s1



# ---------------------------------------------------------------- the hum-fix batch
# Pre-registered in Notes: Sonora/hum-fix-batch-preregistration.md.

def _differ(a, b):
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def _without(c, *keys):
    return {k: v for k, v in c.items() if k not in keys}


def test_d1_is_c0s_config_on_its_subsample():
    c0, d1 = y(DATA_CFG / "vctk_22k.yaml"), y(DATA_CFG / "vctk_d1.yaml")
    assert _differ(c0, d1) == {"name", "train_filelist_path", "valid_filelist_path"}
    assert d1["name"] == "vctk_d1"
    assert d1["train_filelist_path"] == "data/vctk_d1/train.txt"
    assert d1["valid_filelist_path"] == "data/vctk_d1/val.txt"
    # statistics and n_spks come from `vctk` through the same defaults as C0's: stock's
    assert d1["defaults"] == ["vctk", "_self_"]


def test_d1_is_c0s_experiment_for_26_epochs_from_c0s_own_init():
    c0, d1 = y(EXP / "staged_c0.yaml"), y(EXP / "staged_d1.yaml")
    rest = ("defaults", "tags", "run_name", "trainer")
    assert _without(d1, *rest) == _without(c0, *rest)
    assert d1["trainer"] == dict(c0["trainer"], max_epochs=26)
    assert d1["defaults"] == [{"override /data": "vctk_d1.yaml"}]
    text = (EXP / "staged_d1.yaml").read_text(encoding="utf-8")
    assert "warmstart/staged_c0_init.ckpt" in text and "byte for byte" in text
    assert "30fadd35e6cdff441b413e3a2f7657e033d53eadf63313f2fddfbcc4641c552d" in text
    assert "checkpoint_epoch=025_step=0009229" in text


def test_g_is_s1s_config_on_the_g_filelists_with_s1s_statistics():
    s1, g = y(DATA_CFG / "libritts_r_22k.yaml"), y(DATA_CFG / "libritts_r_22k_g.yaml")
    assert _differ(s1, g) == {"name", "train_filelist_path", "valid_filelist_path"}
    assert g["name"] == "libritts_r_22k_g"
    assert g["train_filelist_path"] == "data/libritts_r_22k/train_g.txt"
    assert g["valid_filelist_path"] == "data/libritts_r_22k/val_g.txt"


def test_g_is_s1s_experiment_from_s1s_own_init():
    s1, g = y(EXP / "staged_s1.yaml"), y(EXP / "staged_g.yaml")
    assert _without(g, "defaults", "tags", "run_name") == \
        _without(s1, "defaults", "tags", "run_name")
    assert g["defaults"] == [{"override /data": "libritts_r_22k_g.yaml"}]
    text = (EXP / "staged_g.yaml").read_text(encoding="utf-8")
    assert "warmstart/staged_s1_init.ckpt" in text and "byte for byte" in text
    assert "67dc8c6a3d1c3d94eb8cf1698e8361c0161ee292514af7e628335917557ef434" in text


def test_e_is_s1s_config_on_the_e_filelists_with_s1s_statistics():
    s1, e = y(DATA_CFG / "libritts_r_22k.yaml"), y(DATA_CFG / "libritts_r_22k_e.yaml")
    assert _differ(s1, e) == {"name", "train_filelist_path", "valid_filelist_path"}
    assert e["name"] == "libritts_r_22k_e"
    assert e["train_filelist_path"] == "data/libritts_r_22k/train_e.txt"
    assert e["valid_filelist_path"] == "data/libritts_r_22k/val_e.txt"
    assert e["cleaners"] == ["no_cleaners"]


def test_e_is_s1s_experiment_from_s1s_own_init():
    s1, e = y(EXP / "staged_s1.yaml"), y(EXP / "staged_e.yaml")
    assert _without(e, "defaults", "tags", "run_name") == \
        _without(s1, "defaults", "tags", "run_name")
    assert e["defaults"] == [{"override /data": "libritts_r_22k_e.yaml"}]
    text = (EXP / "staged_e.yaml").read_text(encoding="utf-8")
    assert "warmstart/staged_s1_init.ckpt" in text and "byte for byte" in text
    assert "67dc8c6a3d1c3d94eb8cf1698e8361c0161ee292514af7e628335917557ef434" in text


def test_f1_is_s1s_config_on_its_own_corpus_table_and_statistics():
    s1, f1 = y(DATA_CFG / "libritts_r_22k.yaml"), y(DATA_CFG / "f1_dense.yaml")
    assert _differ(s1, f1) == {"name", "train_filelist_path", "valid_filelist_path",
                                "n_spks", "data_statistics"}
    assert f1["name"] == "f1_dense" and f1["n_spks"] == 90
    assert f1["train_filelist_path"] == "data/f1_dense/train_op.txt"
    assert f1["valid_filelist_path"] == "data/f1_dense/val_op.txt"


def test_q_is_f1s_config_on_its_own_corpus_and_statistics():
    f1, q = y(DATA_CFG / "f1_dense.yaml"), y(DATA_CFG / "q_synth.yaml")
    # Q's statistics are set once `mel-stats` writes them, so they are compared by key, not by
    # value: the file's values are a placeholder until then.
    assert _differ(f1, q) - {"data_statistics"} == \
        {"name", "train_filelist_path", "valid_filelist_path"}
    assert q["name"] == "q_synth_22k" and q["n_spks"] == 90
    assert q["train_filelist_path"] == "data/q_synth_22k/train_op.txt"
    assert q["valid_filelist_path"] == "data/q_synth_22k/val_op.txt"
    assert set(q["data_statistics"]) == {"mel_mean", "mel_std"}


def test_f1_is_s1s_experiment_on_f1s_data_from_its_own_warm_start():
    s1, f1 = y(EXP / "staged_s1.yaml"), y(EXP / "staged_f1.yaml")
    assert _without(f1, "defaults", "tags", "run_name") == \
        _without(s1, "defaults", "tags", "run_name")
    assert f1["defaults"] == [{"override /data": "f1_dense.yaml"}]
    text = (EXP / "staged_f1.yaml").read_text(encoding="utf-8")
    assert "make_warmstart.py --experiment staged_f1 --donor matcha_vctk.ckpt" in text
    assert "staged_f1_init.ckpt" in text


def test_q_is_f1s_experiment_on_qs_data_from_its_own_warm_start():
    f1, q = y(EXP / "staged_f1.yaml"), y(EXP / "staged_q.yaml")
    assert _without(q, "defaults", "tags", "run_name") == \
        _without(f1, "defaults", "tags", "run_name")
    assert q["defaults"] == [{"override /data": "q_synth.yaml"}]
    text = (EXP / "staged_q.yaml").read_text(encoding="utf-8")
    assert "make_warmstart.py --experiment staged_q --donor matcha_vctk.ckpt" in text
    assert "staged_q_init.ckpt" in text


@pytest.mark.skipif(not Path("/data/model-training/sonora/data/f1_dense/mel_statistics.json")
                    .is_file(), reason="/data not mounted")
def test_f1_statistics_are_the_measured_ones():
    got = json.loads(Path("/data/model-training/sonora/data/f1_dense/mel_statistics.json")
                     .read_text())
    want = y(DATA_CFG / "f1_dense.yaml")["data_statistics"]
    assert want["mel_mean"] == pytest.approx(got["mel_mean"], abs=1e-6)
    assert want["mel_std"] == pytest.approx(got["mel_std"], abs=1e-6)


@pytest.mark.parametrize("arm,base", [("staged_d1", "staged_c0"), ("staged_g", "staged_s1"),
                                      ("staged_f1", "staged_s1"), ("staged_e", "staged_s1"),
                                      ("staged_q", "staged_s1")])
def test_each_batch_arm_overrides_only_its_base_arms_data(arm, base):
    a, b = y(EXP / ("%s.yaml" % arm))["defaults"], y(EXP / ("%s.yaml" % base))["defaults"]
    keep = lambda d: [x for x in d if not (isinstance(x, dict) and "override /data" in x)]
    assert keep(a) == keep(b) and len(a) == len(b)
    assert sum(isinstance(x, dict) and "override /data" in x for x in a) == 1
