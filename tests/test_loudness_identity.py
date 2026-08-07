"""C-L4: the loudnorm sidecar keyed on PATH, and a path is not a clip.

A reroll re-renders in place, keeping the filename. The stale record matched, the new take
was skipped, and it shipped at whatever level its engine produced — into a bank whose
entire purpose is that every clip sits at one level, and into the reference pool four
cloning engines condition on. Nothing said so: the clip was counted under `already-done`.

MEASURED 2026-08-07 over every loudnorm.jsonl on disk (1,029 clips, 11 banks): 15 clips
are 0.90 to 7.09 dB away from the level their record says they were left at, while the
other 1,014 are within 0.185 dB. The 15 are rerolls — `the-return_nar_0059..0063_doc_QWE`
were re-rendered twenty minutes after their gain was applied. This is a defect that had
already fired, not one waiting to.

The same path key produced the OPPOSITE failure once before (the quarantine move, where a
clip normalized under its old path looked unprocessed and was gained twice). One key, two
failures in opposite directions, because the key does not identify the audio.
"""

import json
import pathlib
import sys

import pytest

SYNTH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "synthesis"
sys.path.insert(0, str(SYNTH))

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("pyloudnorm")
nl = pytest.importorskip("normalize_loudness")

RATE = 24000


def _tone(path, seconds=3.0, amp=0.10, freq=220.0, seed=0):
    """A wav with a real integrated loudness. Noise-modulated so two different `seed`s
    are genuinely different audio rather than the same waveform at another gain."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    sig = np.sin(2 * np.pi * freq * t) * (0.7 + 0.3 * rng.random(len(t)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), (sig * amp).astype(np.float32), RATE, subtype="PCM_16")
    return path


def _lufs(path):
    d, sr = sf.read(str(path), always_2d=False)
    if d.ndim > 1:
        d = d.mean(axis=1)
    return nl.measure(d, sr)


def _run(bank, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["normalize_loudness", "--dir", str(bank), *extra])
    nl.main()


def _sidecar(bank):
    path = bank / nl.SIDECAR
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["file"]] = rec          # last record per path wins
    return out


@pytest.fixture()
def bank(tmp_path):
    _tone(tmp_path / "a.wav", amp=0.10, seed=1)
    _tone(tmp_path / "b.wav", amp=0.03, seed=2)
    return tmp_path


# --- the fix --------------------------------------------------------------------------


def test_a_reroll_in_place_is_gained_and_reported(bank, monkeypatch, capsys):
    """THE REGRESSION. Under the path key the rerolled take was skipped as `already-done`
    and shipped at its engine's native level."""
    _run(bank, monkeypatch)
    capsys.readouterr()
    assert abs(_lufs(bank / "a.wav") - (-23.0)) < 0.5

    _tone(bank / "a.wav", amp=0.30, seed=99)          # a reroll: same name, new audio
    assert _lufs(bank / "a.wav") > -20.0

    _run(bank, monkeypatch)
    out = capsys.readouterr().out
    assert "normalized 1" in out
    assert "reroll" in out and "a.wav" in out
    assert abs(_lufs(bank / "a.wav") - (-23.0)) < 0.5


def test_an_untouched_clip_is_still_skipped(bank, monkeypatch, capsys):
    """Idempotence is the property that lets a bank be re-run after adding one engine.
    Content keying must not cost it."""
    _run(bank, monkeypatch)
    capsys.readouterr()
    before = {p.name: p.read_bytes() for p in bank.glob("*.wav")}

    _run(bank, monkeypatch)
    out = capsys.readouterr().out
    assert "normalized 0" in out and "already-done 2" in out
    assert {p.name: p.read_bytes() for p in bank.glob("*.wav")} == before


def test_the_rerolled_original_is_kept_beside_the_first_takes(bank, monkeypatch):
    """The backup was keyed on path too, so a reroll found the FIRST take's original
    already there and kept it — preserving the pristine copy of a wav that no longer
    exists while the new take's original was overwritten on the spot."""
    _run(bank, monkeypatch)
    first = (bank / nl.BACKUP_DIR / "a.wav").read_bytes()

    _tone(bank / "a.wav", amp=0.30, seed=99)
    rerolled = (bank / "a.wav").read_bytes()
    _run(bank, monkeypatch)

    assert (bank / nl.BACKUP_DIR / "a.wav").read_bytes() == first
    others = [p for p in (bank / nl.BACKUP_DIR).glob("a.*.wav")]
    assert len(others) == 1
    assert others[0].read_bytes() == rerolled


# --- the migration --------------------------------------------------------------------


def test_a_legacy_record_is_adopted_rather_than_re_gained(bank, monkeypatch, capsys):
    """No record written before 2026-08-07 carries a digest. If those were treated as
    unknown, the first run after this change would gain every clip in every existing bank
    a second time — silently, and in the same direction for all of them. The fallback is
    the instrument: a clip still sitting at the level its record says it was left at IS
    that take."""
    _run(bank, monkeypatch)
    capsys.readouterr()
    level = {p.name: _lufs(p) for p in bank.glob("*.wav")}

    strip = [{k: v for k, v in rec.items() if not k.startswith("sha")}
             for rec in _sidecar(bank).values()]
    (bank / nl.SIDECAR).write_text(
        "".join(json.dumps(r) + "\n" for r in strip), encoding="utf-8")

    _run(bank, monkeypatch)
    out = capsys.readouterr().out
    assert "normalized 0" in out
    assert "no content digest" in out
    assert {p.name: _lufs(p) for p in bank.glob("*.wav")} == level
    # ...and the digest is now recorded, so the next run is cheap and exact.
    assert all(rec.get("sha_out") for rec in _sidecar(bank).values())


def test_a_legacy_record_does_not_hide_a_reroll(bank, monkeypatch, capsys):
    """The fallback must not become a blanket amnesty: a legacy record whose clip is NOT
    at the recorded level is exactly the case C-L4 is about."""
    _run(bank, monkeypatch)
    capsys.readouterr()
    strip = [{k: v for k, v in rec.items() if not k.startswith("sha")}
             for rec in _sidecar(bank).values()]
    (bank / nl.SIDECAR).write_text(
        "".join(json.dumps(r) + "\n" for r in strip), encoding="utf-8")
    _tone(bank / "a.wav", amp=0.30, seed=99)

    _run(bank, monkeypatch)
    assert "reroll" in capsys.readouterr().out
    assert abs(_lufs(bank / "a.wav") - (-23.0)) < 0.5


def test_expected_level_follows_the_gain_applied_not_the_target():
    """A peak-capped clip lands SHORT of target on purpose. Comparing against the target
    would call every one of them a reroll — and the ceiling hits are exactly the clips
    that must not be gained again."""
    capped = {"lufs_in": -10.0, "gain_db": -4.0, "lufs_target": -23.0, "peak_capped": True}
    assert nl.expected_lufs(capped) == -14.0
    assert nl.expected_lufs({"file": "x", "skipped": "too-short-or-silent"}) is None
    assert nl.expected_lufs({"file": "x"}) is None


def test_the_tolerance_separates_the_two_measured_populations():
    """Sized from data, not chosen: 1,014 honest clips land within 0.185 dB and the 15
    rerolls start at 0.904 dB. The threshold has to sit in that gap, and the docstring
    has to keep saying which numbers put it there."""
    assert 0.185 < nl.LEGACY_LUFS_TOLERANCE < 0.904
    src = (SYNTH / "normalize_loudness.py").read_text(encoding="utf-8")
    assert "0.185" in src and "7.09" in src and "1,029" in src


# --- the sidecar itself ---------------------------------------------------------------


def test_a_torn_sidecar_line_cannot_lose_a_record(bank, monkeypatch):
    """Every reader of these sidecars skips a line it cannot parse, so a torn append does
    not raise — it forgets that the clip was processed, and the clip gets gained twice.
    The sidecar is rewritten whole through tmp+rename, so it only ever holds complete
    records."""
    _run(bank, monkeypatch)
    text = (bank / nl.SIDECAR).read_text(encoding="utf-8")
    assert text.endswith("\n")
    for line in text.splitlines():
        json.loads(line)                     # every line parses, or this raises

    lock = bank / f"{nl.SIDECAR}.lock"
    assert lock.exists(), "the sidecar rewrite must take the shared lock"


def test_dry_run_writes_nothing_at_all(bank, monkeypatch, capsys):
    before = {p.name: p.read_bytes() for p in bank.glob("*.wav")}
    _run(bank, monkeypatch, "--dry-run")
    assert "would normalize 2" in capsys.readouterr().out
    assert not (bank / nl.SIDECAR).exists()
    assert not (bank / nl.BACKUP_DIR).exists()
    assert {p.name: p.read_bytes() for p in bank.glob("*.wav")} == before
