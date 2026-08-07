"""Regression coverage for the E-H2 / E-M2 lane and control fixes (2026-08-06).

Each test below fails against the pre-fix source. The defects they cover share a shape
worth naming: none of them raised. A Sonora checkpoint driven through the espeak lane, a
24 kHz waveform written with a 22.05 kHz header, a digit deleted from the input, an
out-of-range valence pushed into the FiLM trunk — every one produces confident, fluent,
wrong audio. That is why these are asserted rather than left to the ear.

Runs on the host venv: nothing here needs a GPU or a checkpoint.
"""

import argparse

import pytest

torch = pytest.importorskip("torch")
cli = pytest.importorskip("matcha.cli")


def _args(**kw):
    """A parsed-args stand-in carrying only what the functions under test read."""
    base = dict(lane="vat", vat_dim=3, vat=None, guidance=1.0, checkpoint_path=None)
    base.update(kw)
    return argparse.Namespace(**base)


# --- parse_vat: the control contract -------------------------------------------------


def test_vat_within_range_is_accepted():
    assert cli.parse_vat(_args(vat="0.4,-0.2,0.0")) == [0.4, -0.2, 0.0]


@pytest.mark.parametrize("bad", ["5,0,0", "0,0,-1.5", "1.01,0,0"])
def test_vat_outside_two_sigma_is_refused(bad):
    """±1 is the edge of the trained range: derivation clamps per-speaker z at 2 sigma.

    Values beyond it do not make the channel stronger, they move the FiLM activation off
    the manifold — and the model still renders speech, which is the trap.
    """
    with pytest.raises(SystemExit, match=r"within \[-1, 1\]"):
        cli.parse_vat(_args(vat=bad))


def test_vat_width_must_match_the_checkpoint():
    """The guard that will matter when delivery makes vat_dim 4 (todo § 1)."""
    with pytest.raises(SystemExit, match="takes 3 VAT channels, got 2"):
        cli.parse_vat(_args(vat="0,0"))
    with pytest.raises(SystemExit, match="takes 3 VAT channels, got 4"):
        cli.parse_vat(_args(vat="0,0,0,0"))


def test_vat_on_a_legacy_checkpoint_is_refused():
    """No conditioning trunk to drive — silently ignoring the flag would be worse."""
    with pytest.raises(SystemExit, match="no conditioning trunk"):
        cli.parse_vat(_args(lane="legacy", vat="0,0,0"))


def test_vat_rejects_non_numeric():
    with pytest.raises(SystemExit, match="comma-separated numbers"):
        cli.parse_vat(_args(vat="warm,0,0"))


def test_no_vat_flag_means_neutral_not_an_error():
    assert cli.parse_vat(_args(vat=None)) is None


# --- synthesis_conditioning: what actually reaches the model -------------------------


def test_conditioning_is_empty_on_the_legacy_lane():
    """A legacy checkpoint must not be handed `vat=` or `guidance=` at all."""
    assert cli.synthesis_conditioning(_args(lane="legacy"), torch.device("cpu")) == {}


def test_conditioning_carries_guidance_and_vat():
    args = _args(guidance=2.5)
    args.vat_vector = [0.4, -0.2, 0.0]
    out = cli.synthesis_conditioning(args, torch.device("cpu"))
    assert out["guidance"] == 2.5
    assert out["vat"].shape == (1, 3)
    # approx: the tensor is float32, so 0.4 comes back as 0.4000000059604645.
    assert out["vat"].tolist() == [pytest.approx([0.4, -0.2, 0.0])]


# --- process_text_for_lane: the phoneme lane ------------------------------------------


def test_digits_are_refused_rather_than_deleted():
    """D-M3 at this boundary: the tokenizer DROPS digits, it does not expand them.

    Verified live 2026-08-06: "I have 3 cats" phonemized to `ˈaɪ hˈæv kˈæts` and
    synthesised cleanly. `g2p.validate()` cannot catch it — nothing illegal is present,
    a word is simply gone.
    """
    with pytest.raises(ValueError, match="silently DROPS"):
        cli.process_text_for_lane(1, "I have 3 cats", torch.device("cpu"), "vat")


def test_g2p_failure_is_a_valueerror_not_a_systemexit():
    """SystemExit is a BaseException, so `except Exception` in the Vocalizer's HTTP
    handler would not catch it — one bad request would have killed the uvicorn worker
    instead of returning an error. This function is shared, so the exception type is
    part of its contract."""
    with pytest.raises(ValueError):
        cli.process_text_for_lane(1, "99 bottles", torch.device("cpu"), "vat")


# --- assert_required_models_available: the forced download ----------------------------


def test_local_checkpoint_is_not_downloaded(tmp_path, monkeypatch):
    """E-H2's headline: the guard read `not hasattr(args, "checkpoint_path") and
    args.checkpoint_path is None`, which argparse makes permanently False — so
    `--checkpoint_path` still fetched the upstream LJSpeech checkpoint over the network
    before ignoring it."""
    ckpt = tmp_path / "local.ckpt"
    ckpt.write_bytes(b"")
    voc = tmp_path / "vocoder"
    voc.write_bytes(b"")

    called = []
    monkeypatch.setattr(cli, "assert_model_downloaded", lambda *a, **k: called.append(a))

    paths = cli.assert_required_models_available(
        _args(checkpoint_path=str(ckpt), vocoder_path=str(voc), vocoder=None)
    )
    assert called == [], f"downloaded despite local paths: {called}"
    assert paths["matcha"] == ckpt
    assert paths["vocoder"] == voc


def test_missing_local_vocoder_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "assert_model_downloaded", lambda *a, **k: None)
    ckpt = tmp_path / "local.ckpt"
    ckpt.write_bytes(b"")
    with pytest.raises(SystemExit, match="no vocoder at"):
        cli.assert_required_models_available(
            _args(checkpoint_path=str(ckpt), vocoder_path=str(tmp_path / "nope"), vocoder=None)
        )


# --- save_to_folder: the sample rate --------------------------------------------------


def test_waveform_is_written_at_the_lane_sample_rate(tmp_path):
    """Hardcoded 22050 meant every Sonora render was tagged 22.05 kHz and played ~9%
    slow — audible, but as "the model sounds sluggish" rather than as a header bug."""
    sf = pytest.importorskip("soundfile")
    output = {
        "mel": torch.zeros(1, 80, 16),
        "waveform": torch.zeros(4096),
    }
    cli.save_to_folder("clip", output, str(tmp_path), 24000)
    assert sf.info(str(tmp_path / "clip.wav")).samplerate == 24000


# --- the ONNX "Plan B" lane -----------------------------------------------------------
#
# Auditing the claim that `matcha/app.py` was the last espeak leak turned up two more, both
# here. The source-level half of this guard lives in `test_lane_containment.py` so it runs
# on the host venv, where torch is absent and this whole module skips.


def test_onnx_infer_requires_an_explicit_lane():
    """No default is defensible: an ONNX graph does not record its phoneme vocabulary, so
    either default is silently wrong for half the graphs it will be pointed at."""
    infer = pytest.importorskip("matcha.onnx.infer")
    parser = _onnx_infer_parser(infer)
    with pytest.raises(SystemExit):
        parser.parse_args(["model.onnx", "--text", "hello"])
    assert parser.parse_args(["model.onnx", "--text", "hello", "--lane", "vat"]).lane == "vat"


def test_onnx_infer_writes_the_lane_sample_rate():
    infer = pytest.importorskip("matcha.onnx.infer")
    assert infer.LANE_SAMPLE_RATE == {"vat": 24000, "legacy": 22050}


def _onnx_infer_parser(infer):
    """Build `infer.main`'s parser without running inference."""
    captured = {}
    real = argparse.ArgumentParser

    class _Capture(real):
        def parse_args(self, *a, **kw):  # noqa: D102
            captured["parser"] = self
            raise _Stop

    infer.argparse.ArgumentParser = _Capture
    try:
        with pytest.raises(_Stop):
            infer.main()
    finally:
        infer.argparse.ArgumentParser = real
    return captured["parser"]


class _Stop(Exception):
    pass


def test_onnx_export_refuses_a_conditioned_checkpoint():
    """The exporter builds no VAT input node, so a Sonora graph would render neutral
    forever — real speech, so nothing downstream could tell it had happened."""
    export = pytest.importorskip("matcha.onnx.export")
    conditioned = argparse.Namespace(use_vat=True, vat_dim=3)
    with pytest.raises(SystemExit, match="refusing to export"):
        export.assert_exportable_here(conditioned)


def test_onnx_export_still_accepts_the_legacy_checkpoints_it_is_for():
    """`vat_dim` defaults to 3 whether or not a trunk was built, so guarding on the width
    instead of on `use_vat` would refuse every checkpoint this exporter is still for."""
    export = pytest.importorskip("matcha.onnx.export")
    export.assert_exportable_here(argparse.Namespace(use_vat=False, vat_dim=3))
