"""The words sent to the model must be the words everything else records.

orpheus and dia render from `direction.render_text`; the WER gate, the manifest,
the audition card and the training corpus all read `line["text"]`. Nothing kept
those two in sync until check_bank.py, and the failure is silent by construction:
QC scores the audio against `text`, so when the two disagree it is measuring
against a reference the model never saw.
"""
import json
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                       / "scripts" / "synthesis"))

import check_bank  # noqa: E402


def _bank(lines):
    fd = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8")
    json.dump({"version": "t", "campaign": "t", "lines": lines}, fd)
    fd.close()
    return fd.name


def line(cid="c1", engine="orpheus", text="hello there", render_text=None,
         **direction):
    d = dict(direction)
    if render_text is not None:
        d["render_text"] = render_text
    return {"id": cid, "engine": engine, "text": text, "direction": d}


def test_identical_text_passes():
    assert check_bank.check(_bank([line(text="a b c", render_text="a b c")])) == []


def test_the_madison_regression():
    """The exact defect: text expanded, render_text left stale."""
    p = check_bank.check(_bank([line(
        text="a society in Madison, Wisconsin. An hour",
        render_text="a society in Madison, Wis. An hour")]))
    assert len(p) == 1
    assert "Wisconsin." in p[0][2] and "Wis." in p[0][2]


def test_orpheus_tags_are_allowed():
    """render_text exists TO carry engine markup — that is not a divergence."""
    assert check_bank.check(_bank([line(
        text="We did it! Every one of them said we couldn't.",
        render_text="We did it! <laugh> Every one of them said we couldn't.")])) == []


def test_dia_speaker_tags_and_nonverbals_are_allowed():
    assert check_bank.check(_bank([line(
        cid="c2", engine="dia",
        text="I know. I know she isn't coming.",
        render_text="[S1] I know. (sighs) I know she isn't coming. [S1]")])) == []


def test_markup_stripped_from_both_sides():
    """Stripping one side only is its own bug — it flags every tagged line."""
    assert check_bank.check(_bank([line(
        text="We did it! <laugh> Every one.",
        render_text="We did it! <laugh> Every one.")])) == []


def test_engine_without_render_text_is_skipped():
    """qwen/chatterbox/zonos have no render_text channel; absence is not a defect."""
    assert check_bank.check(_bank([
        line(cid="c3", engine="chatterbox", text="a b c", design="a man")])) == []


def test_dropped_word_is_caught():
    p = check_bank.check(_bank([line(text="one two three four",
                                     render_text="one two four")]))
    assert len(p) == 1


def test_whitespace_only_difference_passes():
    assert check_bank.check(_bank([line(text="a  b\nc", render_text="a b c")])) == []


def test_reroll2_updates_both_copies():
    """The reroll script must expand the abbreviation in render_text too."""
    import book_ingest as bi
    src = "a society in Madison, Wis. An hour before the time set."
    assert bi.normalize_speakable(src) == \
        "a society in Madison, Wisconsin. An hour before the time set."
    # Same function applied to both copies keeps them canonically equal.
    assert check_bank.canon(bi.normalize_speakable(src)) == \
        check_bank.canon(bi.normalize_speakable(src))


@pytest.mark.parametrize("bank", sorted(
    pathlib.Path("/data/model-training/datasets").glob("*/bank*.json"))
    if pathlib.Path("/data/model-training/datasets").is_dir() else [])
def test_every_bank_on_disk_is_consistent(bank):
    """Standing guard over the real banks, not just synthetic fixtures."""
    assert check_bank.check(str(bank)) == []
