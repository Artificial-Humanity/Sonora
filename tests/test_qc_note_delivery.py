"""QC-H1 (2026-08-09): the triage note has to reach the auditor's row, end to end.

`register_audition.main()` computed `qc_map`, printed "N clips carry a QC finding — all
queued for the ear", and then called `register_audio_dir(...)` **without** `qc=qc_map`.
The parameter defaults to `None`, so the `row["note"] = qc[cid]` branch was dead code and
`qc_noted` printed 0 for nine days. Coverage survived (`qc_flags.txt` still carried the
ids, so `pick_audit_subset --flags` still queued the clips); the GUIDANCE did not — every
"stops early — LISTEN TO THE END" note was computed and discarded.

That is why this file tests `main()` rather than `register_audio_dir()`. The helper was
correct the whole time; the wiring was the defect, and only an end-to-end assertion on
what lands in ratings.csv can see it. The failure it exists to prevent is concrete:
`the-return_nar_0051_doc_MOS`, a tail-truncated clip that sounds clean until its missing
ending and was scored 5.
"""

import csv
import importlib
import json
import pathlib
import sys

import pytest

SYNTH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "synthesis"
sys.path.insert(0, str(SYNTH))

PASSAGE = "the whole of that day he wandered by the river and did not come home at all"


def _register_module(tmp_path, monkeypatch):
    """Import register_audition against a synthetic data root (it reads env at import)."""
    monkeypatch.setenv("AUDITION_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDITION_RATINGS_DIR", str(tmp_path / "ratings"))
    monkeypatch.setenv("BOOK_PROSE_ROOT", str(tmp_path / "book-prose"))
    sys.modules.pop("register_audition", None)
    return importlib.import_module("register_audition")


def _campaign(tmp_path):
    """A one-clip campaign: a truncated render, its manifest, and its qc measures."""
    audio = tmp_path / "campaign-x" / "audio"
    audio.mkdir(parents=True)
    (audio / "clip.wav").write_bytes(b"RIFF")            # only is_file() is checked
    audio.joinpath("bank_manifest.jsonl").write_text(
        json.dumps({"id": "truncated-0001", "wav": "clip.wav", "engine": "qwen",
                    "campaign": "campaign-x", "register": "narration"}) + "\n",
        encoding="utf-8")
    qc = tmp_path / "campaign-x" / "qc_measures.jsonl"
    qc.write_text(json.dumps({
        "id": "truncated-0001",
        "gates": {"asr_ok": False},
        "text": PASSAGE,
        "asr_wer": 0.41,
        "asr_hyp": " ".join(PASSAGE.split()[:6]),        # stops early
    }) + "\n", encoding="utf-8")
    return audio, qc


def _rows(tmp_path):
    with open(tmp_path / "ratings" / "ratings.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_qc_note_lands_on_the_row_the_auditor_opens(tmp_path, monkeypatch):
    """The end-to-end claim STATE.md makes: "findings attach as direction-aware notes"."""
    mod = _register_module(tmp_path, monkeypatch)
    audio, qc = _campaign(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["register_audition", "--audio-dir", str(audio), "--qc", str(qc)])
    mod.main()

    rows = _rows(tmp_path)
    assert [r["id"] for r in rows] == ["truncated-0001"]
    note = rows[0]["note"]
    assert note, "the finding was computed and then dropped on the floor — QC-H1"
    assert note.startswith("QC: ")
    # The note must point the ear at the right END of the clip. A truncation and an
    # over-run are opposite defects and the wrong instruction is worse than none.
    assert "CHECK THE END" in note
    # Note only. The owner's rule is that every QC failure is auditioned, so the
    # instrument never gets to pre-empt the verdict.
    assert rows[0]["status"] == "unaudited"
    assert rows[0]["score"] == ""


def test_a_clean_clip_is_queued_without_a_note(tmp_path, monkeypatch):
    """The note means something only if its absence also means something."""
    mod = _register_module(tmp_path, monkeypatch)
    audio, qc = _campaign(tmp_path)
    qc.write_text(json.dumps({"id": "truncated-0001", "gates": {"asr_ok": True},
                              "text": PASSAGE, "asr_hyp": PASSAGE}) + "\n",
                  encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["register_audition", "--audio-dir", str(audio), "--qc", str(qc)])
    mod.main()

    rows = _rows(tmp_path)
    assert rows[0]["note"] == ""
    assert rows[0]["status"] == "unaudited"


def test_without_qc_the_registration_still_works(tmp_path, monkeypatch):
    """`--qc` is optional, and `qc=None` must stay a legal call."""
    mod = _register_module(tmp_path, monkeypatch)
    audio, _ = _campaign(tmp_path)
    monkeypatch.setattr(sys, "argv", ["register_audition", "--audio-dir", str(audio)])
    mod.main()
    assert _rows(tmp_path)[0]["note"] == ""


def test_the_flags_file_and_the_note_carry_the_same_ids(tmp_path, monkeypatch):
    """Coverage and guidance are two halves of one rule.

    `qc_flags.txt` is what `pick_audit_subset --flags` reads to force the clip into the
    queue; the note is what tells the auditor what to listen for once it gets there. When
    the wiring broke, the first half kept working — which is exactly why nobody noticed.
    """
    mod = _register_module(tmp_path, monkeypatch)
    audio, qc = _campaign(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["register_audition", "--audio-dir", str(audio), "--qc", str(qc)])
    mod.main()

    flags = {ln.strip() for ln in (qc.parent / "qc_flags.txt").read_text().splitlines()
             if ln.strip()}
    noted = {r["id"] for r in _rows(tmp_path) if r["note"]}
    assert flags == noted == {"truncated-0001"}


def test_a_verdict_without_its_number_still_produces_a_note(tmp_path, monkeypatch):
    """Found while writing this file: `f"{r.get('asr_wer'):.2f}"` raised TypeError on a
    row carrying `asr_ok: False` and no `asr_wer` — and the raise happened inside the
    triage, so the ENTIRE registration died and none of the campaign's clips reached the
    queue. An incomplete measure must degrade the note, not the run."""
    mod = _register_module(tmp_path, monkeypatch)
    audio, qc = _campaign(tmp_path)
    qc.write_text(json.dumps({"id": "truncated-0001", "gates": {"asr_ok": False},
                              "text": PASSAGE,
                              "asr_hyp": " ".join(PASSAGE.split()[:6])}) + "\n",
                  encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["register_audition", "--audio-dir", str(audio), "--qc", str(qc)])
    mod.main()
    assert "CHECK THE END" in _rows(tmp_path)[0]["note"]


def test_the_helper_is_still_reachable_with_a_note_map(tmp_path, monkeypatch):
    """A unit-level pin under the branch that was dead, so a future refactor that drops
    the keyword again fails here as well as end to end."""
    mod = _register_module(tmp_path, monkeypatch)
    audio, _ = _campaign(tmp_path)
    rows = mod.register_audio_dir(audio, "campaign-x", set(), mod.DEFAULT_FIELDS, False,
                                  qc={"truncated-0001": "QC: something to hear"})
    assert rows[0]["note"] == "QC: something to hear"


def test_main_passes_the_map_it_computes(tmp_path, monkeypatch):
    """The wiring itself, asserted directly: whatever `_qc_triage` returns is what
    `register_audio_dir` receives. `git log -S "qc=qc_map"` was empty for nine days."""
    mod = _register_module(tmp_path, monkeypatch)
    audio, qc = _campaign(tmp_path)
    seen = {}
    real = mod.register_audio_dir

    def spy(*a, **kw):
        seen["qc"] = kw.get("qc", "NOT PASSED")
        return real(*a, **kw)

    monkeypatch.setattr(mod, "register_audio_dir", spy)
    monkeypatch.setattr(sys, "argv",
                        ["register_audition", "--audio-dir", str(audio), "--qc", str(qc)])
    mod.main()
    assert seen["qc"] != "NOT PASSED", "main() computed the map and did not pass it"
    assert set(seen["qc"]) == {"truncated-0001"}
