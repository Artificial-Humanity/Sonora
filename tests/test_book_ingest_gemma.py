"""book_ingest's two Gemma passes, routed through gemma_client.

`chat` is replaced by a recorder, so these pin what the passes SEND and how they react to
what comes back — the network itself is gemma_client's tests' job.
"""

import json

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import book_ingest as bi  # noqa: E402
import gemma_client as gc  # noqa: E402

LABELS = {"V": -0.2, "A": 0.1, "T": 0.3, "register": "neutral_narration"}
GOOD_ZONOS = {"voice_design": "a calm middle-aged narrator", "emotion": None,
              "pitch_std": 30, "speaking_rate": 14}


class Recorder:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, user, **kw):
        self.calls.append({"system": system, "user": user, **kw})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_the_director_is_the_client_director():
    assert bi.MODEL == gc.DIRECTOR


def test_casting_pass_sends_exactly_what_casting_messages_builds(monkeypatch):
    rec = Recorder([json.dumps(GOOD_ZONOS)])
    monkeypatch.setattr(bi, "chat", rec)
    assert bi.casting_pass("The rain fell.", "zonos", labels=LABELS) == GOOD_ZONOS
    system, user = bi.casting_messages("The rain fell.", "zonos", LABELS)
    call = rec.calls[0]
    assert (call["system"], call["user"]) == (system, user)
    assert call["model"] == gc.DIRECTOR
    assert call["max_tokens"] == 900
    assert call["temperature"] == 0.2
    assert call["schema"] == bi._json_schema("zonos")


def test_casting_messages_puts_the_labels_and_the_skill_file_in_the_prompt():
    system, user = bi.casting_messages("The rain fell.", "zonos", LABELS)
    assert "===== SKILL FILE: zonos =====" in system
    assert "register: neutral_narration" in user
    assert "V=-0.20" in user
    assert system.rstrip().endswith(bi._schema_str("zonos"))


def test_casting_pass_retries_a_failed_call_then_accepts(monkeypatch):
    rec = Recorder([gc.GemmaError("m: transport: refused"), json.dumps(GOOD_ZONOS)])
    monkeypatch.setattr(bi, "chat", rec)
    assert bi.casting_pass("The rain fell.", "zonos", labels=LABELS) == GOOD_ZONOS
    assert len(rec.calls) == 2


def test_truncated_json_is_retried_and_never_accepted(monkeypatch):
    # Review Focus 2: max_tokens hit mid-object.
    cut = '{"voice_design": "a calm middle-aged narr'
    rec = Recorder([cut, cut])
    monkeypatch.setattr(bi, "chat", rec)
    assert bi.casting_pass("The rain fell.", "zonos", labels=LABELS, retries=2) is None
    assert len(rec.calls) == 2


def test_director_tag_sends_the_director_prompt_and_gives_up_after_its_retries(monkeypatch):
    rec = Recorder([gc.GemmaError("m: empty content")] * 2)
    monkeypatch.setattr(bi, "chat", rec)
    chunk = {"chunk_type": "narration", "text": "The rain fell on the harbour.",
             "source_ref": {}}
    assert bi.director_tag(chunk, retries=2) is None
    assert len(rec.calls) == 2
    assert rec.calls[0]["system"] == bi.DIRECTOR_SYSTEM
    assert rec.calls[0]["user"] == "Narration passage: The rain fell on the harbour."
    assert rec.calls[0]["model"] == gc.DIRECTOR
    assert rec.calls[0]["max_tokens"] == 400
    assert rec.calls[0]["temperature"] == 0.2
    assert "schema" not in rec.calls[0]
