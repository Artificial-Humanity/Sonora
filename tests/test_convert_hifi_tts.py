"""`scripts/tools/convert_hifi_tts.py` — rung 4's parquet -> 24 kHz wav conversion.

Everything here runs against a SYNTHETIC parquet built in `tmp_path`. Nothing touches
`/data`: the real corpus is 40 GB and a test that needs it is a test that skips on every
machine but one, which is the shape this repo keeps finding behind a green suite.

⚠ THE POINT OF THIS FILE IS THE CONSUMER, NOT THE PRODUCER. The converter's output is only
correct if `derive_vat_corpus.find_clips` can read it, so the layout assertions call THAT
FUNCTION rather than re-stating its rule. A copy of the rule would keep passing after the
rule changed — and `find_clips` skips a wav whose sibling text is missing WITHOUT SAYING SO,
so the failure it would hide is a silent shrink of the corpus.
"""
from __future__ import annotations

import io
import os

import pytest

pa = pytest.importorskip("pyarrow", reason="parquet reader; see the dataprep extra")
pq = pytest.importorskip("pyarrow.parquet")
sf = pytest.importorskip("soundfile")
np = pytest.importorskip("numpy")

from scripts_layout import SCRIPTS  # noqa: E402

SCRIPTS.on_path()

import convert_hifi_tts as C  # noqa: E402


# --------------------------------------------------------------------------- #
# fixture — a parquet shaped exactly like Hi-Fi TTS's, three rows
# --------------------------------------------------------------------------- #
SRC_SR = 44100


def _flac(seconds, sr=SRC_SR, freq=220.0):
    """A FLAC blob, the way the real shards carry audio: bytes inside an `audio` struct."""
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    y = (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def shard(tmp_path):
    """One `train.other-...parquet` holding three rows.

    ⚠ TWO OF THE THREE ROWS ARE SPEAKER `6097` UNDER DIFFERENT SUBSET DIRECTORIES —
    `6097_clean` and `6097_other`. That is not a contrived case: measured 2026-09-14, 6097 is
    in both subsets of Hi-Fi TTS's own `train` split. It is here because keying the output on
    the directory name instead of the `speaker` column splits one reader into two identities,
    each getting its own embedding row, and nothing downstream would report it.
    """
    data = tmp_path / "src" / "data"
    data.mkdir(parents=True)
    rows = [
        ("6097", "audio/6097_clean/14411/nada_a_0001.flac", 1.5, "first line here"),
        ("6097", "audio/6097_other/15277/nada_b_0002.flac", 2.0, "second line here"),
        ("8051", "audio/8051_other/7670/slave_c_0003.flac", 1.25, "third line here"),
    ]
    tbl = pa.table({
        "speaker": pa.array([r[0] for r in rows]),
        "file": pa.array([r[1] for r in rows]),
        "duration": pa.array([r[2] for r in rows], type=pa.float32()),
        "text_normalized": pa.array([r[3] for r in rows]),
        "audio": pa.array([{"bytes": _flac(r[2]), "path": r[1].split("/")[-1]} for r in rows]),
    })
    path = data / "train.other-00000-of-00001-deadbeef.parquet"
    pq.write_table(tbl, path)
    return path


# --------------------------------------------------------------------------- #
# floors — before any claim about content
# --------------------------------------------------------------------------- #
def test_an_unknown_split_is_refused_and_names_what_exists(shard):
    """A typo'd split converting zero clips, quietly, is the failure this repo keeps paying
    for. The refusal must also SAY what the directory holds, or the reader has to go look."""
    src = str(shard.parent.parent)
    with pytest.raises(SystemExit) as e:
        C.shards(src, ["train.clen"])
    msg = str(e.value)
    assert "train.clen" in msg and "available" in msg, msg
    assert "train.other" in msg, "the refusal did not name the split that DOES exist: %s" % msg


def test_a_real_split_finds_its_shards(shard):
    """⚠ POSITIVE CONTROL for the test above. A `shards()` that refused everything would
    satisfy the refusal test for reasons that have nothing to do with the split name."""
    src = str(shard.parent.parent)
    assert C.shards(src, ["train.other"]) == [str(shard)]


# --------------------------------------------------------------------------- #
# the conversion
# --------------------------------------------------------------------------- #
def _run(shard, out, force=False):
    return C._convert_shard((str(shard), str(out), False, force))


def test_the_output_is_readable_by_derive_vat_corpus(shard, tmp_path):
    """The layout assertion, made against the REAL consumer rather than a copy of its rule."""
    out = tmp_path / "out"
    _, written, _, failed, _, _ = _run(shard, out)
    assert (written, failed) == (3, 0)

    from derive_vat_corpus import find_clips
    clips = sorted(find_clips(str(out)))
    assert len(clips) == 3, "find_clips saw %d of 3 — it skips a wav whose text is missing" % len(clips)
    for wav, text, spk in clips:
        assert text, "empty transcript for %s" % wav
        assert spk in ("6097", "8051")


def test_the_speaker_is_the_column_and_not_the_directory(shard, tmp_path):
    """⚠ The `6097_clean` / `6097_other` trap, asserted as one identity rather than two."""
    out = tmp_path / "out"
    _run(shard, out)
    from derive_vat_corpus import find_clips
    by_spk = {}
    for wav, _, spk in find_clips(str(out)):
        by_spk.setdefault(spk, []).append(wav)
    assert sorted(by_spk) == ["6097", "8051"], (
        "expected two readers; got %s. A `6097_clean`/`6097_other` split here means the "
        "converter keyed on the source directory instead of the `speaker` column." % sorted(by_spk))
    assert len(by_spk["6097"]) == 2, "6097's two subsets did not land under one speaker"


def test_the_audio_is_24k_mono_pcm16_and_faithful(shard, tmp_path):
    """Rate, channels and subtype are what `derive_vat_corpus` gates on — it REJECTS a
    non-24 kHz clip rather than resampling it, so a native-rate tree derives an empty corpus.

    Fidelity is checked against an independent decode of the same source, not against a
    stored number: the bound is PCM_16's quantisation step, which is a property of the
    format rather than of this run.
    """
    out = tmp_path / "out"
    _run(shard, out)
    import librosa
    from derive_vat_corpus import SAMPLE_RATE

    tbl = pq.read_table(shard, columns=["file", "audio"])
    srcs = dict(zip(tbl["file"].to_pylist(), [a["bytes"] for a in tbl["audio"].to_pylist()]))
    from derive_vat_corpus import find_clips
    for wav, _, _ in find_clips(str(out)):
        info = sf.info(wav)
        assert info.samplerate == SAMPLE_RATE, "%s is %d Hz" % (wav, info.samplerate)
        assert info.channels == 1 and info.subtype == "PCM_16"
        key = next(k for k in srcs if os.path.basename(k)[:-5] == os.path.basename(wav)[:-4])
        ref, _ = librosa.load(io.BytesIO(srcs[key]), sr=SAMPLE_RATE, mono=True)
        got, _ = sf.read(wav, dtype="float32")
        n = min(len(ref), len(got))
        assert abs(len(ref) - len(got)) <= 2, "length drift on %s" % wav
        assert float(np.abs(ref[:n] - got[:n]).max()) <= 1.0 / 32768 + 1e-9


def test_a_tmp_suffix_does_not_defeat_the_format_inference(shard, tmp_path):
    """⚠ REGRESSION. The destination is `<stem>.wav.tmp` so the rename can be atomic, and
    soundfile infers the container from the EXTENSION — which is `.tmp`. Without an explicit
    `format="WAV"` every clip raises `TypeError: No format specified...`, and because the
    transcript is written first the tree fills with texts and no audio: measured, 3,600
    transcripts and 0 wavs, and `find_clips` then reports an EMPTY corpus rather than a
    broken one.

    Asserted through the real conversion rather than by grepping for the argument, so it
    keeps holding if the write moves.
    """
    out = tmp_path / "out"
    _, written, _, failed, _, rows = _run(shard, out)
    assert failed == 0, "conversion failed: %s" % [r.get("error") for r in rows if r.get("error")]
    assert written == 3
    wavs = [os.path.join(dp, f) for dp, _, fs in os.walk(out) for f in fs if f.endswith(".wav")]
    texts = [os.path.join(dp, f) for dp, _, fs in os.walk(out) for f in fs if f.endswith(".normalized.txt")]
    assert len(wavs) == len(texts) == 3, (
        "%d wav against %d transcript — a text-only tree is the signature of the format bug"
        % (len(wavs), len(texts)))
    assert not [f for dp, _, fs in os.walk(out) for f in fs if f.endswith(".tmp")], "a .tmp survived"


# --------------------------------------------------------------------------- #
# resumability — the property an interrupted 322,478-clip run depends on
# --------------------------------------------------------------------------- #
def test_a_second_run_reuses_and_does_not_redecode(shard, tmp_path):
    out = tmp_path / "out"
    _run(shard, out)
    _, written, reused, failed, _, _ = _run(shard, out)
    assert (written, reused, failed) == (0, 3, 0)


def test_a_truncated_wav_is_redone_and_only_that_one(shard, tmp_path):
    """⚠ The reuse decision is a FRAME COUNT, not an existence check, and this is what makes
    that matter: a run killed mid-write leaves a short file, and an existence check would
    trust it forever. Mutating one file must move exactly one clip back to `written`.
    """
    out = tmp_path / "out"
    _run(shard, out)
    victim = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(out)
                    for f in fs if f.endswith(".wav"))[0]
    with open(victim, "r+b") as f:
        f.truncate(200)
    _, written, reused, failed, _, _ = _run(shard, out)
    assert (written, reused, failed) == (1, 2, 0), (
        "expected exactly the truncated clip to be redone; got written=%d reused=%d" % (written, reused))
    assert sf.info(victim).samplerate == 24000


def test_force_redoes_everything(shard, tmp_path):
    """⚠ POSITIVE CONTROL for the reuse path. Without this, a `_convert_shard` that had
    stopped writing at all would satisfy the reuse test by never producing anything new."""
    out = tmp_path / "out"
    _run(shard, out)
    _, written, reused, failed, _, _ = _run(shard, out, force=True)
    assert (written, reused, failed) == (3, 0, 0)


# --------------------------------------------------------------------------- #
# #477 — a clip with no transcript must not be converted at all
# --------------------------------------------------------------------------- #
def test_a_clip_with_no_transcript_is_skipped_not_silently_orphaned(tmp_path):
    """⚠ `find_clips` drops a wav whose sibling text is missing WITHOUT SAYING SO.

    So writing the audio for an empty `text_normalized` produces a clip that counts as
    converted, occupies disk, and then vanishes from the corpus with nothing in any log —
    four lines under a comment warning about exactly that shape. The clip must be refused
    here, counted, and marked in the manifest.

    Latent on the real corpus: 0 of 323,978 rows have an empty text. That is why it is
    tested rather than trusted — a defect nothing can currently trigger is one nothing will
    report when something finally does.
    """
    data = tmp_path / "src" / "data"
    data.mkdir(parents=True)
    rows = [("6097", "audio/6097_clean/1/a_0001.flac", 1.0, "a real line"),
            ("6097", "audio/6097_clean/1/b_0002.flac", 1.0, "   "),
            ("8051", "audio/8051_other/2/c_0003.flac", 1.0, "")]
    tbl = pa.table({
        "speaker": pa.array([r[0] for r in rows]),
        "file": pa.array([r[1] for r in rows]),
        "duration": pa.array([r[2] for r in rows], type=pa.float32()),
        "text_normalized": pa.array([r[3] for r in rows]),
        "audio": pa.array([{"bytes": _flac(r[2]), "path": r[1].split("/")[-1]} for r in rows]),
    })
    shard = data / "train.clean-00000-of-00001-feedface.parquet"
    pq.write_table(tbl, shard)

    out = tmp_path / "out"
    _, written, reused, failed, notext, manifest = C._convert_shard((str(shard), str(out), False, False))
    assert (written, reused, failed, notext) == (1, 0, 0, 2), (
        "written=%d reused=%d failed=%d no-text=%d — a blank transcript must not produce a clip"
        % (written, reused, failed, notext))

    wavs = [f for _, _, fs in os.walk(out) for f in fs if f.endswith(".wav")]
    assert len(wavs) == 1, "an orphan wav was written for a row with no transcript: %s" % wavs

    from derive_vat_corpus import find_clips
    assert len(list(find_clips(str(out)))) == 1

    skipped = [r for r in manifest if "skipped" in r]
    assert len(skipped) == 2, "the manifest must say which rows were skipped and why"
    assert all("text_normalized" in r["skipped"] for r in skipped)


def test_a_populated_transcript_still_converts(tmp_path, shard):
    """⚠ POSITIVE CONTROL for the test above. A `_convert_shard` that refused every row
    would satisfy the skip assertion for reasons that have nothing to do with the text."""
    out = tmp_path / "out"
    _, written, _, failed, notext, _ = C._convert_shard((str(shard), str(out), False, False))
    assert (written, failed, notext) == (3, 0, 0)
