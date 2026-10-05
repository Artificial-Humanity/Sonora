"""The one way Sonora reaches Gemma: Lemonade's OpenAI-style chat endpoint.

The endpoint, both model ids and the request/response shape live here and nowhere else
(Notes/Sonora/lemonade-migration-design.md). Every other file imports them.

⚠ ONE ATTEMPT PER CALL. Callers own their retry loops, and their counts were bounded on
purpose (issues #115, #121): a retry here would multiply every one of them again.

⚠ THINKING IS ALWAYS OFF. Gemma 4 thinks by default on this server; with a short token
budget the reasoning eats all of it and `content` comes back empty (measured 2026-10-05).

⚠ JSON IS ALWAYS A SCHEMA. The server enforces `json_schema` by grammar but not
`json_object`, which came back inside a markdown fence. `as_json=True` therefore sends the
schema `{"type": "object"}`.
"""

import json
import urllib.error
import urllib.request

URL = "http://localhost:13305/v1/chat/completions"
DIRECTOR = "gemma-4-31b-mtp"   # judgement: the director and casting passes
VOLUME = "gemma-4-e4b-mtp"     # volume: passage judging and markup labelling
# Sent on every call. The server starts llama.cpp with its own sampling defaults, so
# leaving these out would let a server setting change what Sonora samples.
TOP_K, TOP_P = 64, 0.95


class GemmaError(RuntimeError):
    """A Gemma call produced no usable content. The message names the model and the cause."""


def request_body(system, user, *, model, max_tokens, temperature, schema=None, as_json=False):
    """The JSON body `chat` sends. Separate so a caller can see the exact request."""
    if schema is not None and as_json:
        raise ValueError("pass schema or as_json, not both")
    messages = [] if system is None else [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": user})
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_k": TOP_K, "top_p": TOP_P, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if as_json:
        schema = {"type": "object"}
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema":
                                   {"name": "out", "strict": True, "schema": schema}}
    return body


def chat(system, user, *, model, max_tokens, temperature, schema=None, as_json=False,
         timeout=180):
    """One chat completion. Returns the reply text; raises GemmaError on any failure."""
    body = request_body(system, user, model=model, max_tokens=max_tokens,
                        temperature=temperature, schema=schema, as_json=as_json)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:   # before URLError: it is a subclass
        detail = e.read()[:200].decode("utf-8", "replace")
        raise GemmaError(f"{model}: HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise GemmaError(f"{model}: transport: {e}") from e
    try:
        content = json.loads(raw)["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise GemmaError(f"{model}: malformed response: {raw[:200]!r}") from e
    if not content:
        raise GemmaError(f"{model}: empty content")
    return content
