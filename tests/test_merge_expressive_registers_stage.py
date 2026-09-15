"""`merge_expressive_registers._stage_24k` — the resample that had never been exercised.

⚠ THIS FILE EXISTS BECAUSE #476 WAS LATENT FOR A MONTH AND NOTHING COULD HAVE CAUGHT IT.
`_stage_24k` writes to `<dst>.tmp` so the rename is atomic, and soundfile infers the
container from the EXTENSION — which is `.tmp`. Without an explicit `format="WAV"` every
resample raises `TypeError: No format specified...`.

It went unnoticed because all 832 staged wavs date from the day the file was written: every
run since has matched the reuse branch on samplerate and frame count, and never reached the
write at all. So the defect is reachable only by a FIRST stage or a `--restage`, and the one
corpus that would trigger it is already built.

The test therefore stages into an empty directory, which is the only state that reaches the
line. A test that reused would pass against the broken version — which is precisely how the
defect survived.
"""
from __future__ import annotations

import os

import pytest

sf = pytest.importorskip("soundfile")
np = pytest.importorskip("numpy")
pytest.importorskip("librosa")

from scripts_layout import SCRIPTS  # noqa: E402

SCRIPTS.on_path()

import merge_expressive_registers as M  # noqa: E402


@pytest.fixture
def source(tmp_path):
    """One 44.1 kHz wav, so the resample branch is the one taken."""
    src = tmp_path / "bank"
    src.mkdir()
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    y = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    path = src / "clip_one.wav"
    sf.write(str(path), y, sr, subtype="PCM_16")
    return [{"id": "clip_one", "wav": str(path)}]


def test_a_first_stage_actually_writes_a_24k_wav(source, tmp_path):
    """The write path, reached because the destination does not exist yet.

    Asserts the POSTCONDITION — a readable 24 kHz file — rather than that the call returned.
    `_stage_24k` returns its bookkeeping either way, so a run that raised per clip and a run
    that wrote every clip are distinguishable only on disk.
    """
    stage = tmp_path / "stage"
    out, seconds = M._stage_24k(source, str(stage))
    dst = out["clip_one"]
    assert os.path.exists(dst), "no file at %s — the write did not happen" % dst
    info = sf.info(dst)
    assert info.samplerate == M.TARGET_SR
    assert info.channels == 1 and info.subtype == "PCM_16"
    assert abs(seconds["clip_one"] - 1.0) < 0.01
    assert not [f for f in os.listdir(os.path.dirname(dst)) if f.endswith(".tmp")], "a .tmp survived"


def test_a_second_stage_actually_reuses_rather_than_redecoding(source, tmp_path, monkeypatch):
    """The reuse branch — the one every run since the corpus was built has taken (#476).

    ⚠⚠ THIS DOCSTRING CLAIMED TWO THINGS THAT WERE BOTH FALSE (#480). It said the test
    "passes with or without the fix", and it did not: its FIRST call is itself a first stage,
    so it reaches the broken write and fails under the mutant — my own "2 failed" mutation
    count said so and I wrote the opposite line above it. And it called itself a positive
    control while asserting only `os.path.exists`, which a full re-decode satisfies exactly
    as well as a reuse. **It could not fail for the reason it was there.**

    So it now counts decodes. `librosa.load` is the only way audio is read on this path, and
    the reuse branch must not call it at all on the second pass. That is the postcondition
    the name claims, and unlike a file's existence it distinguishes reuse from re-doing the
    work and arriving at the same bytes.
    """
    stage = tmp_path / "stage"
    M._stage_24k(source, str(stage))          # first stage — this one legitimately decodes

    import librosa
    calls = []
    real = librosa.load
    monkeypatch.setattr(librosa, "load", lambda *a, **k: (calls.append(a[:1]), real(*a, **k))[1])

    out, _ = M._stage_24k(source, str(stage))
    assert os.path.exists(out["clip_one"])
    assert calls == [], "the second stage decoded %d clip(s); it should have reused" % len(calls)


def test_a_dry_run_writes_nothing_but_still_plans(source, tmp_path):
    """`--dry-run` is what a reader reaches for before a restage, so it must not be the only
    thing that works."""
    stage = tmp_path / "stage"
    out, seconds = M._stage_24k(source, str(stage), dry_run=True)
    assert out and seconds
    assert not os.path.exists(out["clip_one"])
    assert abs(seconds["clip_one"] - 1.0) < 0.01
