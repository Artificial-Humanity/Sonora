"""The Gemma client: one request shape, one attempt, every failure a GemmaError.

Pinned against a real local HTTP server, not a mocked urlopen, so the request that leaves
the process is the one Lemonade would receive.
"""

import http.server
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import gemma_client as gc  # noqa: E402


def _ok(content):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers["Content-Length"])
        self.server.bodies.append(json.loads(self.rfile.read(n)))
        status, payload, delay = self.server.reply
        if delay:
            time.sleep(delay)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload if isinstance(payload, bytes) else json.dumps(payload).encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.bodies = []
    srv.reply = (200, _ok('{"a": 1}'), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(gc, "URL", f"http://127.0.0.1:{srv.server_port}/v1/chat/completions")
    yield srv
    srv.shutdown()
    srv.server_close()


def _call(**kw):
    args = dict(model="m", max_tokens=50, temperature=0.2)
    args.update(kw)
    return gc.chat("sys", "usr", **args)


# ------------------------------------------------------------------ request shape

def test_plain_request_carries_thinking_off_and_the_sampling_pins(server):
    assert _call() == '{"a": 1}'
    body = server.bodies[0]
    assert body["model"] == "m"
    assert body["messages"] == [{"role": "system", "content": "sys"},
                                {"role": "user", "content": "usr"}]
    assert body["max_tokens"] == 50
    assert body["temperature"] == 0.2
    assert body["top_k"] == gc.TOP_K == 64
    assert body["top_p"] == gc.TOP_P == 0.95
    assert body["stream"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in body


def test_a_none_system_prompt_sends_no_system_message(server):
    gc.chat(None, "usr", model="m", max_tokens=5, temperature=0.0)
    assert server.bodies[0]["messages"] == [{"role": "user", "content": "usr"}]


def test_schema_mode_sends_a_strict_json_schema(server):
    schema = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
    _call(schema=schema)
    rf = server.bodies[0]["response_format"]
    assert rf == {"type": "json_schema",
                  "json_schema": {"name": "out", "strict": True, "schema": schema}}


def test_as_json_sends_a_generic_object_schema_not_json_object(server):
    _call(as_json=True)
    rf = server.bodies[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == {"type": "object"}


def test_schema_and_as_json_together_are_refused_before_any_request(server):
    with pytest.raises(ValueError):
        _call(schema={"type": "object"}, as_json=True)
    assert server.bodies == []


# ------------------------------------------------------------------ failures

def test_http_error_is_a_gemma_error_naming_model_and_status(server):
    server.reply = (500, {"error": "boom"}, 0)
    with pytest.raises(gc.GemmaError, match=r"m: HTTP 500"):
        _call()


def test_refused_connection_is_a_gemma_error(monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(gc, "URL", f"http://127.0.0.1:{port}/v1/chat/completions")
    with pytest.raises(gc.GemmaError, match="transport"):
        _call()


def test_timeout_reaches_urlopen(server):
    # Review Focus 1: a cold reload must fit the caller's budget, so the budget must arrive.
    server.reply = (200, _ok("late"), 0.6)
    with pytest.raises(gc.GemmaError):
        _call(timeout=0.2)
    assert _call(timeout=5) == "late"


def test_non_json_body_is_a_gemma_error(server):
    server.reply = (200, b"<html>gateway</html>", 0)
    with pytest.raises(gc.GemmaError, match="malformed"):
        _call()


def test_missing_choices_is_a_gemma_error(server):
    server.reply = (200, {"object": "chat.completion"}, 0)
    with pytest.raises(gc.GemmaError, match="malformed"):
        _call()


@pytest.mark.parametrize("content", ["", None])
def test_empty_or_null_content_is_a_gemma_error_not_a_value(server, content):
    # Review Focus 4: thinking that ate the budget arrives as 200 with nothing in content.
    server.reply = (200, _ok(content), 0)
    with pytest.raises(gc.GemmaError, match="empty content"):
        _call()


def test_a_truncated_body_is_a_gemma_error(monkeypatch):
    # A handler that sends Content-Length larger than actual payload
    class _TruncatedHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"choices"')  # Incomplete JSON, closes connection

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _TruncatedHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    monkeypatch.setattr(gc, "URL", f"http://127.0.0.1:{srv.server_port}/v1/chat/completions")
    try:
        with pytest.raises(gc.GemmaError, match="transport"):
            _call()
    finally:
        srv.shutdown()
        srv.server_close()


def test_an_unreadable_error_body_is_still_a_gemma_error(monkeypatch):
    # Mock urlopen to raise HTTPError whose .read() also raises
    def mock_urlopen(req, timeout=None):
        err = urllib.error.HTTPError(
            "http://localhost:13305/v1/chat/completions",
            503,
            "Service Unavailable",
            {},
            None
        )
        # Make .read() raise ConnectionResetError
        def read_raises(*args, **kwargs):
            raise ConnectionResetError("Connection reset by peer")
        err.read = read_raises
        raise err

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    with pytest.raises(gc.GemmaError, match="HTTP 503"):
        _call()


def test_the_defaults_name_lemonade_on_localhost_and_the_registered_models():
    assert gc.URL == "http://localhost:13305/v1/chat/completions"
    assert gc.DIRECTOR == "gemma-4-31b-mtp"
    assert gc.VOLUME == "gemma-4-e4b-mtp"
