"""make_teacher_ab_bank's retry arithmetic survives the move to gemma_client.

Issue #115: transport failures are retried inside `_ask` only, never re-asked by `direct`.
Issue #121: a semantic miss (a well-formed answer without a required key) is re-asked,
DIRECT_RETRIES times. Review Focus 5: a server that is down costs exactly
TRANSPORT_RETRIES calls, then None — not a crash.
"""

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import gemma_client as gc  # noqa: E402
import make_teacher_ab_bank as tab  # noqa: E402


class Recorder:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, system, user, **kw):
        self.calls.append(kw)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_a_dead_server_costs_exactly_the_transport_retries(monkeypatch):
    rec = Recorder(gc.GemmaError("m: transport: refused"))
    monkeypatch.setattr(tab, "chat", rec)
    assert tab._ask("s", "u", "m") is None
    assert len(rec.calls) == tab.TRANSPORT_RETRIES == 3


def test_direct_does_not_re_ask_after_a_transport_failure(monkeypatch):
    rec = Recorder(gc.GemmaError("m: transport: refused"))
    monkeypatch.setattr(tab, "chat", rec)
    assert tab.direct("brief", "A line long enough to perform.", "qwen", "m",
                      tab.DIRECT_RETRIES) is None
    assert len(rec.calls) == tab.TRANSPORT_RETRIES


def test_direct_re_asks_a_well_formed_answer_missing_its_key(monkeypatch):
    rec = Recorder('{"voice": "x"}')
    monkeypatch.setattr(tab, "chat", rec)
    assert tab.direct("brief", "A line long enough to perform.", "qwen", "m",
                      tab.DIRECT_RETRIES) is None
    assert len(rec.calls) == tab.DIRECT_RETRIES


def test_ask_passes_its_budget_and_a_long_timeout(monkeypatch):
    rec = Recorder('{"instruct": "calm"}')
    monkeypatch.setattr(tab, "chat", rec)
    assert tab._ask("s", "u", "m", max_tokens=900) == {"instruct": "calm"}
    assert rec.calls[0] == {"model": "m", "max_tokens": 900, "temperature": 0.2,
                            "timeout": 300}
