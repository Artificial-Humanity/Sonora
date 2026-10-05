"""The three hand-run tools that call Gemma, after the move to gemma_client."""

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import gemma_client as gc  # noqa: E402
import judge_passages as jp  # noqa: E402
import make_quote_pilot_bank as qp  # noqa: E402
import tag_spike as ts  # noqa: E402

UNIT = '{"dialect": "none", "speakable": true, "unit": true, "speaker": "narrator"}'


class Recorder:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, system, user, **kw):
        self.calls.append({"system": system, "user": user, **kw})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


# ------------------------------------------------------------------ judge_passages

def test_judge_asks_the_given_model_for_a_json_object(monkeypatch):
    rec = Recorder(UNIT)
    monkeypatch.setattr(jp, "chat", rec)
    d, err = jp.ask(gc.VOLUME, "The rain fell.")
    assert err is None and d["unit"] is True
    call = rec.calls[0]
    assert call["system"] is None
    assert call["user"] == jp.SYSTEM + "\n\nPASSAGE:\nThe rain fell."
    assert call["model"] == gc.VOLUME
    assert call["as_json"] is True
    assert (call["max_tokens"], call["temperature"], call["timeout"]) == (160, 0.0, 120)


def test_judge_parses_a_fenced_reply(monkeypatch):
    # Review Focus 3.
    monkeypatch.setattr(jp, "chat", Recorder("```json\n" + UNIT + "\n```"))
    d, err = jp.ask(gc.VOLUME, "x")
    assert err is None and d["unit"] is True


def test_judge_reports_a_dead_server_as_a_transport_error(monkeypatch):
    # Review Focus 5.
    monkeypatch.setattr(jp, "chat", Recorder(gc.GemmaError("m: transport: refused")))
    d, err = jp.ask(gc.VOLUME, "x")
    assert d is None and err.startswith("transport: ")


def test_judge_still_flags_missing_keys(monkeypatch):
    monkeypatch.setattr(jp, "chat", Recorder('{"dialect": "none"}'))
    d, err = jp.ask(gc.VOLUME, "x")
    assert d is None and err.startswith("missing keys")


def test_judge_defaults_to_the_volume_model():
    assert jp.build_parser().get_default("model") == gc.VOLUME


# ------------------------------------------------------------------ tag_spike

def test_tag_spike_sends_a_user_only_json_request(monkeypatch):
    rec = Recorder('{"tags": []}')
    monkeypatch.setattr(ts, "chat", rec)
    assert ts.ask(gc.VOLUME, "prompt") == '{"tags": []}'
    call = rec.calls[0]
    assert call["system"] is None and call["user"] == "prompt"
    assert call["as_json"] is True
    assert (call["max_tokens"], call["temperature"], call["timeout"]) == (700, 0.2, 300)


# ------------------------------------------------------------------ quote pilot

def test_quote_pilot_retries_a_dead_server_and_gives_up(monkeypatch):
    rec = Recorder(gc.GemmaError("m: transport: refused"))
    monkeypatch.setattr(qp, "chat", rec)
    assert qp.call_director("a line", gc.DIRECTOR, retries=3) is None
    assert len(rec.calls) == 3
    assert rec.calls[0]["model"] == gc.DIRECTOR
    assert (rec.calls[0]["max_tokens"], rec.calls[0]["timeout"]) == (400, 300)


@pytest.mark.parametrize("mod", [jp, ts, qp])
def test_no_tool_names_the_old_server(mod):
    src = (SCRIPTS / (mod.__name__ + ".py")).read_text(encoding="utf-8").lower()
    assert "ollama" not in src and "11434" not in src
