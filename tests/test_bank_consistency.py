"""The words sent to the model must be the words everything else records.

orpheus and dia render from `direction.render_text`; the WER gate, the manifest,
the audition card and the training corpus all read `line["text"]`. Nothing kept
those two in sync until check_bank.py, and the failure is silent by construction:
QC scores the audio against `text`, so when the two disagree it is measuring
against a reference the model never saw.
"""
import json
import os
import pathlib
import sys
import tempfile

import pytest

from scripts_layout import SCRIPTS  # noqa: E402

SCRIPTS.on_path()

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


# --- a reference pinned by file (`ref_file`, Phase 1S Step 1c-i) ------------------------
#
# check_bank pre-flights `ref_file` before synth_bank.sh books a GPU: its fields, an absolute
# path under ref_select.REF_FILE_ROOT, a file that exists and a sha256 equal to the file's.
# REF_FILE_ROOT is pointed at the test's tmp dir, since /data is not writable here.


@pytest.fixture
def ref_root(tmp_path, monkeypatch):
    import ref_select
    monkeypatch.setattr(ref_select, "REF_FILE_ROOT", str(tmp_path) + "/")
    clip = tmp_path / "anchor.wav"
    clip.write_bytes(b"the anchor take")
    import hashlib
    return clip, hashlib.sha256(clip.read_bytes()).hexdigest()


def _pinned(spec, **extra):
    return dict(line(cid="p1", engine="zonos", text="a b c"), ref_file=spec, **extra)


def test_a_well_formed_ref_file_passes(ref_root):
    clip, sha = ref_root
    bank = _bank([_pinned({"path": str(clip), "sha256": sha, "text": "the anchor take",
                           "id": "M4-s3", "gender": "Male"})])
    assert check_bank.check_refs(bank) == []
    assert check_bank.main(["check_bank.py", bank]) == 0


@pytest.mark.parametrize("mutate,needle", [
    (lambda s, clip: dict(s, sha256="0" * 64), "does not have sha256"),
    (lambda s, clip: dict(s, sha256="ABC"), "not 64 lowercase hex"),
    (lambda s, clip: dict(s, path=os.path.relpath(str(clip))), "absolute path"),
    (lambda s, clip: dict(s, path="anchor.wav"), "absolute path"),
    (lambda s, clip: dict(s, path=str(clip.parent / ".." / clip.parent.name / clip.name)),
     "normalised"),
    (lambda s, clip: dict(s, path="/etc/passwd"), "absolute path"),
    (lambda s, clip: dict(s, path=str(clip.parent / "missing.wav")), "does not exist"),
    (lambda s, clip: {k: v for k, v in s.items() if k != "text"}, "has no `text`"),
    (lambda s, clip: dict(s, text="  "), "text is empty"),
    (lambda s, clip: dict(s, speaker="x"), "unknown key `speaker`"),
    (lambda s, clip: "a/path.wav", "must be an object"),
    # a falsy pin is an error, as in the renderers, never "no pin" (the line would be cast)
    (lambda s, clip: {}, "has no `path`"),
    (lambda s, clip: None, "must be an object"),
])
def test_a_bad_ref_file_is_refused(ref_root, mutate, needle):
    clip, sha = ref_root
    spec = mutate({"path": str(clip), "sha256": sha, "text": "the anchor take"}, clip)
    bank = _bank([_pinned(spec)])
    problems = check_bank.check_refs(bank)
    assert len(problems) >= 1 and any(needle in p[2] for p in problems), problems
    assert check_bank.main(["check_bank.py", bank]) == 1


def test_a_line_pinning_both_ref_id_and_ref_file_is_refused(ref_root):
    clip, sha = ref_root
    bank = _bank([_pinned({"path": str(clip), "sha256": sha, "text": "t"}, ref_id="pool_1")])
    assert any("both ref_id" in p[2] for p in check_bank.check_refs(bank))


def test_a_bank_without_ref_file_is_untouched_by_the_ref_check():
    """Existing banks: pool pins and cast lines raise no reference problem."""
    bank = _bank([line(cid="c4", engine="chatterbox", text="a b c"),
                  dict(line(cid="c5", engine="zonos", text="a b c"), ref_id="pool_1")])
    assert check_bank.check_refs(bank) == []
    assert check_bank.main(["check_bank.py", bank]) == 0


def test_a_ref_file_symlinked_out_of_the_root_is_refused(ref_root, tmp_path_factory):
    """The REALPATH must be under REF_FILE_ROOT: the containers mount only that tree."""
    clip, sha = ref_root
    outside = tmp_path_factory.mktemp("outside") / "anchor.wav"
    outside.write_bytes(clip.read_bytes())
    link = clip.parent / "link-out.wav"
    link.symlink_to(outside)
    bank = _bank([_pinned({"path": str(link), "sha256": sha, "text": "t"})])
    problems = check_bank.check_refs(bank)
    assert any("outside" in p[2] and str(outside) in p[2] for p in problems), problems
    assert check_bank.main(["check_bank.py", bank]) == 1
    inside = clip.parent / "link-in.wav"              # control: a link that stays inside is fine
    inside.symlink_to(clip)
    assert check_bank.check_refs(_bank([_pinned({"path": str(inside), "sha256": sha,
                                                 "text": "t"})])) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode-000 file")
def test_an_unreadable_ref_file_is_a_problem_not_a_crash(ref_root):
    clip, sha = ref_root
    bank = _bank([_pinned({"path": str(clip), "sha256": sha, "text": "t"})])
    clip.chmod(0)
    try:
        problems = check_bank.check_refs(bank)
        code = check_bank.main(["check_bank.py", bank])
    finally:
        clip.chmod(0o644)
    assert any("cannot be read" in p[2] for p in problems), problems
    assert code == 1
