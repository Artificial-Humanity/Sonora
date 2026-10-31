"""Regression tests for the core-model hygiene fixes (M-1 .. M-13 of the 2026-10-31 audit).

Each is a small, local defect with no coupling to a trained checkpoint: a padded tail left as
noise, a crop that never reaches the last frame, a cache keyed on half its inputs. They need
torch and the model's own stack, which the host venv does not carry, so they skip there:

    .venv/bin/python -m pytest tests/test_core_model_hygiene.py
"""

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
matcha_tts = pytest.importorskip("matcha.models.matcha_tts")
flow_matching = pytest.importorskip("matcha.models.components.flow_matching")
dit_decoder = pytest.importorskip("matcha.models.components.dit_decoder")
sf = pytest.importorskip("soundfile")

import omegaconf  # noqa: E402  (a dependency of the matcha import guarded above)

REPO = Path(__file__).resolve().parent.parent

N_FEATS = 16
N_VOCAB = 20


def _tiny_model(out_size=None, use_vat=False):
    """A DiT MatchaTTS small enough for CPU, with the real module wiring."""
    cfg = omegaconf.OmegaConf.create({
        "encoder": {
            "encoder_type": "RoPE Encoder",
            "encoder_params": {"n_feats": N_FEATS, "n_channels": 32, "filter_channels": 64,
                               "filter_channels_dp": 32, "n_heads": 2, "n_layers": 1,
                               "kernel_size": 3, "p_dropout": 0.0, "spk_emb_dim": 8,
                               "n_spks": 1, "prenet": True},
            "duration_predictor_params": {"filter_channels_dp": 32, "kernel_size": 3,
                                          "p_dropout": 0.0},
        },
        "decoder": {"type": "dit", "hidden_channels": 32, "filter_channels": 64, "n_layers": 2,
                    "n_heads": 2, "kernel_size": 3, "dropout": 0.0, "use_lsc": True},
        "cfm": {"name": "CFM", "solver": "euler", "sigma_min": 1e-4},
    })
    torch.manual_seed(0)
    model = matcha_tts.MatchaTTS(
        n_vocab=N_VOCAB, n_spks=1, spk_emb_dim=8, n_feats=N_FEATS,
        encoder=cfg.encoder, decoder=cfg.decoder, cfm=cfg.cfm,
        data_statistics={"mel_mean": -5.0, "mel_std": 2.0}, out_size=out_size,
        use_vat=use_vat, vat_dim=3, vat_cond_dim=16, vat_cond_dropout=0.0,
    )
    return model.eval()


def _ragged_text(lengths):
    x = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    for i, n in enumerate(lengths):
        x[i, :n] = torch.randint(1, N_VOCAB, (n,))
    return x, torch.tensor(lengths)


# ---- M-5 / I-3: the CFG branch the comment describes --------------------------------------


class _SpyEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        self.calls.append((mu.clone(), None if cond is None else cond.clone()))
        return torch.zeros_like(x)


def test_cfg_uncond_branch_keeps_conditional_mu_and_zeroes_decoder_cond():
    cfm_params = types.SimpleNamespace(solver="euler", sigma_min=1e-4)
    cfm = flow_matching.BASECFM(n_feats=4, cfm_params=cfm_params)
    cfm.estimator = _SpyEstimator()
    mu = torch.randn(1, 4, 8)
    cond = torch.ones(1, 3, 8)
    cfm(mu, torch.ones(1, 1, 8), n_timesteps=1, cond=cond, guidance=2.0)
    (mu_c, cond_c), (mu_u, cond_u) = cfm.estimator.calls
    assert torch.equal(mu_c, mu_u), "uncond branch must reuse the conditional encoder's mu"
    assert torch.equal(cond_c, cond)
    assert torch.count_nonzero(cond_u) == 0


# ---- M-1: batched synthesise zeroes the padded tail ----------------------------------------


def test_batched_synthesise_zeroes_frames_beyond_mel_lengths():
    model = _tiny_model()
    x, x_lengths = _ragged_text([12, 4])
    torch.manual_seed(1)
    out = model.synthesise(x, x_lengths, n_timesteps=2)
    lengths = out["mel_lengths"]
    assert lengths[0] != lengths[1], "the probe needs a ragged batch"
    for b in range(2):
        tail = slice(int(lengths[b]), None)
        assert torch.count_nonzero(out["decoder_outputs"][b, :, tail]) == 0
        assert torch.count_nonzero(out["mel"][b, :, tail]) == 0
    assert torch.count_nonzero(out["mel"][0, :, : int(lengths[0])]) > 0


def test_single_item_synthesise_is_unmasked_noop():
    model = _tiny_model()
    x, x_lengths = _ragged_text([10])
    torch.manual_seed(2)
    out = model.synthesise(x, x_lengths, n_timesteps=2)
    # B == 1 pads nothing, so the new mask must not zero a single real frame.
    assert out["decoder_outputs"].shape[-1] == int(out["mel_lengths"][0])
    assert (out["decoder_outputs"].abs().sum(dim=1) > 0).all()
    assert (out["mel"].abs().sum(dim=1) > 0).all()


# ---- M-1 (cli): a one-item final batch keeps its batch dim ---------------------------------


def test_batched_cli_one_item_final_batch_indexes_an_utterance(monkeypatch, tmp_path):
    cli = pytest.importorskip("matcha.cli")
    saved = []
    monkeypatch.setattr(cli, "process_text_for_lane",
                        lambda i, text, device, lane: {"x": torch.tensor([[1, 2, 3]]),
                                                       "x_lengths": torch.tensor([3])})
    monkeypatch.setattr(cli, "save_to_folder", lambda name, out, *a, **k: saved.append(out) or Path(name))

    class _Model:
        def synthesise(self, x, x_lengths, **kw):
            b = x.shape[0]
            return {"mel": torch.ones(b, N_FEATS, 4), "mel_lengths": torch.full((b,), 4), "rtf": 1.0}

    # The real `to_waveform` ends in `.squeeze()`; a [1, 1, 4*256] vocoder output becomes 1-D.
    monkeypatch.setattr(cli, "synthesis_conditioning", lambda args, device: {})
    args = types.SimpleNamespace(batch_size=2, steps=1, temperature=0.667, speaking_rate=1.0,
                                 lane="x", spk=None, output_folder=str(tmp_path), sample_rate=24000,
                                 denoiser_strength=0.0)
    vocoder = lambda mel: torch.ones(mel.shape[0], 1, mel.shape[-1] * 256)  # noqa: E731
    monkeypatch.setattr(torch.utils.data, "DataLoader",
                        lambda ds, batch_size, collate_fn, num_workers: iter(
                            [cli.batched_collate_fn([ds[0]])]))
    cli.batched_synthesis(args, "cpu", _Model(), vocoder, None, ["a"], None)
    assert len(saved) == 1
    assert saved[0]["waveform"].shape == (4 * 256,)


# ---- M-3: out_size crop -------------------------------------------------------------------


def _forward_cropped(model, y_lengths, out_size):
    x, x_lengths = _ragged_text([10, 10])
    y = torch.randn(2, N_FEATS, max(y_lengths))
    return model(x, x_lengths, y, torch.tensor(y_lengths), out_size=out_size)


def test_out_size_crop_can_reach_the_final_frame(monkeypatch):
    model = _tiny_model(out_size=8)
    seen = []
    real_choice = matcha_tts.random.choice
    monkeypatch.setattr(matcha_tts.random, "choice", lambda r: seen.append(r) or real_choice(r))
    _forward_cropped(model, [20, 20], 8)
    assert seen and all(max(r) == 20 - 8 for r in seen), "offset 12 (window ends at frame 20) is excluded"


def test_out_size_crop_when_every_utterance_is_shorter_than_out_size():
    model = _tiny_model(out_size=40)
    # Previously: RuntimeError, mask built from max(y_cut_lengths) while y_cut is out_size wide.
    dur_loss, prior_loss, diff_loss, attn = _forward_cropped(model, [32, 28], 40)
    assert attn.shape[-1] == 40
    assert torch.isfinite(diff_loss)


# ---- M-4: precomputed durations must sum to the mel length -------------------------------


def test_loaded_durations_must_sum_to_mel_frames(tmp_path):
    from matcha.data.text_mel_datamodule import TextMelDataset

    (tmp_path / "wavs").mkdir()
    (tmp_path / "durations").mkdir()
    wav = tmp_path / "wavs" / "a.wav"
    sf.write(wav, np.random.default_rng(0).standard_normal(22050).astype("float32") * 0.1, 22050)
    (tmp_path / "train.txt").write_text(f"{wav}|hi\n")
    ds = TextMelDataset(str(tmp_path / "train.txt"), 1, ["basic_cleaners"], load_durations=True)
    n_tokens = len(ds.get_text("hi", add_blank=True)[0])
    frames = ds.get_mel(str(wav)).shape[-1]

    durs = np.zeros(n_tokens, dtype=int)
    durs[0] = frames
    np.save(tmp_path / "durations" / "a.npy", durs)
    assert int(ds[0]["durations"].sum()) == frames

    for wrong in (frames - 3, frames + 3):
        durs[0] = wrong
        np.save(tmp_path / "durations" / "a.npy", durs)
        with pytest.raises(AssertionError, match="sum to"):
            ds[0]


# ---- M-8: measure_speaker_f0 exit codes and one bad file ----------------------------------


def _run_f0(corpus, *extra):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts/tools/measure_speaker_f0.py"), "--corpus", str(corpus),
         "--no-self-test", "--min-rows", "1", *extra],
        capture_output=True, text=True, cwd=REPO, check=False)


def test_f0_empty_filelist_exits_3_not_a_traceback(tmp_path):
    (tmp_path / "train_op.txt").write_text("")
    r = _run_f0(tmp_path)
    assert r.returncode == 3, r.stderr
    assert "Traceback" not in r.stderr


def _f0_row(path, spk):
    from matcha.delivery import VAT_DIM
    return f"{path}|{spk}|hello|{','.join(['0'] * VAT_DIM)}\n"


def test_f0_one_unreadable_wav_is_counted_unvoiced_not_fatal(tmp_path):
    sr = 24000
    t = np.arange(int(1.0 * sr)) / sr
    good = tmp_path / "good.wav"
    sf.write(good, (0.3 * sum(0.8 ** k * np.sin(2 * np.pi * 120 * (k + 1) * t) for k in range(5))
                    / 2).astype("float32"), sr)
    (tmp_path / "train_op.txt").write_text(
        _f0_row(tmp_path / "missing.wav", 0) + _f0_row(good, 1))
    r = _run_f0(tmp_path, "--clips-per-speaker", "1")
    assert r.returncode == 0, r.stderr
    assert "Traceback" not in r.stderr
    assert "EXCLUDED" in r.stdout and "[0]" in r.stdout


# ---- M-9: the mel filterbank cache keys on every parameter that shapes it -----------------


def test_mel_filterbank_cache_is_keyed_on_sampling_rate_and_fmin():
    pytest.importorskip("librosa")
    from librosa.filters import mel as librosa_mel_fn
    from matcha.utils import audio

    audio.mel_basis.clear()
    audio.hann_window.clear()
    y = torch.randn(1, 24000) * 0.1

    def run(sr, fmin, win=1024):
        return audio.mel_spectrogram(y, 1024, 80, sr, 256, win, fmin, 8000)

    a = run(22050, 0)
    b = run(24000, 0)        # same fmax, different sampling_rate
    c = run(24000, 100)      # same fmax and sampling_rate, different fmin
    assert not torch.allclose(a, b) and not torch.allclose(b, c)
    for (sr, fmin) in ((22050, 0), (24000, 0), (24000, 100)):
        want = torch.from_numpy(librosa_mel_fn(sr=sr, n_fft=1024, n_mels=80, fmin=fmin, fmax=8000)).float()
        got = [v for k, v in audio.mel_basis.items() if k[:5] == (sr, 1024, 80, fmin, 8000)]
        assert len(got) == 1 and torch.allclose(got[0], want)
    run(22050, 0, win=512)   # window cache is keyed on win_size too
    assert len({k for k in audio.hann_window}) == 2


# ---- M-11: Lightning module hygiene --------------------------------------------------------


def test_weights_only_checkpoint_loads_as_epoch_zero():
    model = _tiny_model()
    model.on_load_checkpoint({"state_dict": {}})
    assert model.ckpt_loaded_epoch == 0
    model.on_load_checkpoint({"state_dict": {}, "epoch": 7})
    assert model.ckpt_loaded_epoch == 7


def test_validation_hook_tolerates_a_one_item_first_batch(monkeypatch):
    model = _tiny_model()
    x, x_lengths = _ragged_text([6])
    one_batch = {"x": x, "x_lengths": x_lengths, "y": torch.zeros(1, N_FEATS, 8), "spks": None}
    logged = []

    class _Exp:
        def add_image(self, name, *a, **k):
            logged.append(name)

    fake_trainer = types.SimpleNamespace(is_global_zero=True, val_dataloaders=[one_batch])
    monkeypatch.setattr(type(model), "logger", property(lambda s: types.SimpleNamespace(experiment=_Exp())))
    monkeypatch.setattr(type(model), "trainer", property(lambda s: fake_trainer))
    monkeypatch.setattr(type(model), "current_epoch", property(lambda s: 0))
    monkeypatch.setattr("matcha.models.baselightningmodule.plot_tensor", lambda t: np.zeros((2, 2, 3)))
    model.on_validation_end()
    assert "original/0" in logged and "generated_dec/0" in logged
    assert not any(n.endswith("/1") for n in logged)


# ---- M-12: odd RoPE width ----------------------------------------------------------------


def test_attention_rejects_an_odd_rope_width_at_construction():
    dit_decoder.MultiHeadAttention(channels=64, n_heads=2)            # head_dim 32 -> rope 16
    with pytest.raises(AssertionError, match="RoPE width"):
        dit_decoder.MultiHeadAttention(channels=20, n_heads=2)        # head_dim 10 -> rope 5


# ---- M-13: GpuMemoryCap range and device ---------------------------------------------------


def test_gpu_memory_cap_refuses_non_positive_cap_at_construction():
    from matcha.utils.gpu_memory_cap import GpuMemoryCap

    for bad in (0, 0.0, -4):
        with pytest.raises(ValueError, match="cap_gib"):
            GpuMemoryCap(cap_gib=bad)
    assert GpuMemoryCap(cap_gib=None).cap_gib is None
    assert GpuMemoryCap(cap_gib=0.5).cap_gib == 0.5


def test_gpu_memory_cap_applies_to_the_trainers_device(monkeypatch):
    from matcha.utils.gpu_memory_cap import GpuMemoryCap

    queried, capped = [], []
    gib = 1024 ** 3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: queried.append(i) or types.SimpleNamespace(total_memory=100 * gib))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda f, i: capped.append((f, i)))

    trainer = types.SimpleNamespace(strategy=types.SimpleNamespace(root_device=torch.device("cuda", 1)))
    cb = GpuMemoryCap(cap_gib=25)
    cb.setup(trainer, None, "fit")
    assert queried == [1] and capped == [(0.25, 1)] and cb.fraction == 0.25

    # No strategy / an index-less "cuda" falls back to the current device rather than a literal 0.
    queried.clear(), capped.clear()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    GpuMemoryCap(cap_gib=25).setup(types.SimpleNamespace(), None, "fit")
    assert queried == [2] and capped == [(0.25, 2)]


# ---- M-2: record shape is covered by scripts/gates/test_memory_trace.py gate 1b ----------


def test_memory_trace_records_carry_batch_idx_on_both_phases(tmp_path):
    from matcha.utils.memory_trace import MemoryTrace

    (tmp_path / "gtt").write_text("1\n")
    (tmp_path / "mi").write_text("MemTotal: 1 kB\nMemAvailable: 1 kB\n")
    cb = MemoryTrace(gtt_path=str(tmp_path / "gtt"), meminfo_path=str(tmp_path / "mi"))
    trainer = types.SimpleNamespace(log_dir=str(tmp_path), default_root_dir=str(tmp_path), global_step=4)
    cb.on_train_start(trainer, None)
    batch = {"y": torch.zeros(2, 8, 10)}
    cb.on_train_batch_start(trainer, None, batch, 3)
    trainer.global_step = 5
    cb.on_train_batch_end(trainer, None, None, batch, 3)
    recs = [json.loads(line) for line in (tmp_path / "memory_trace.jsonl").read_text().splitlines()]
    assert [(r["phase"], r["step"], r["batch_idx"]) for r in recs] == [("start", 4, 3), ("end", 5, 3)]

