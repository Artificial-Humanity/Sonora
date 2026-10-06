"""The pure half of the director obedience bench: which passages, how a reply scores, and
how two arms' scores become the pre-registered verdict."""

import json
import sys

import pytest

from scripts_layout import ASSETS, SCRIPTS

SCRIPTS.on_path()
import director_obedience_bench as bench  # noqa: E402
import gemma_server  # noqa: E402

GOOD = {"voice_design": "a calm narrator", "emotion": None, "pitch_std": 30, "speaking_rate": 14}


def _bank(tmp_path, book, lines):
    d = tmp_path / book
    d.mkdir()
    p = d / f"{book}_bank.json"
    p.write_text(json.dumps({"campaign": "c", "version": 1, "license_note": "", "lines": lines}))
    return str(p)


def _line(i, book, kind="narration", engine="zonos"):
    return {"id": f"{book}_{kind[:3]}_{i:04d}", "engine": engine, "chunk_type": kind,
            "register": "neutral_narration", "intended": {"V": 0.0, "A": 0.1, "T": 0.2},
            "text": f"Line {i} of {book}."}


# ------------------------------------------------------------------ selection

def test_selection_takes_only_zonos_narration_spread_over_books(tmp_path):
    banks = [_bank(tmp_path, b, [_line(i, b) for i in range(6)]
                   + [_line(9, b, kind="dialogue"), _line(8, b, engine="qwen")])
             for b in ("alpha", "beta", "gamma")]
    picked = bench.select_passages(banks, n=5, seed=1)
    assert len(picked) == 5
    assert all("_nar_" in p["id"] and p["id"] != f"{p['book']}_nar_0008" for p in picked)
    counts = sorted(sum(p["book"] == b for p in picked) for b in ("alpha", "beta", "gamma"))
    assert counts == [1, 2, 2]
    assert set(picked[0]) == {"id", "book", "text", "register", "V", "A", "T"}


def test_selection_is_deterministic_per_seed(tmp_path):
    banks = [_bank(tmp_path, b, [_line(i, b) for i in range(10)]) for b in ("alpha", "beta")]
    ids = lambda s: [p["id"] for p in bench.select_passages(banks, n=6, seed=s)]
    assert ids(7) == ids(7)
    assert ids(7) != ids(8)


def test_selection_refuses_a_population_smaller_than_asked(tmp_path):
    banks = [_bank(tmp_path, "alpha", [_line(i, "alpha") for i in range(3)])]
    with pytest.raises(ValueError, match="only 3 eligible"):
        bench.select_passages(banks, n=4)


def test_selection_refuses_an_empty_glob():
    with pytest.raises(ValueError, match="only 0 eligible"):
        bench.select_passages([], n=1)


# ------------------------------------------------------------------ scoring

def test_score_columns():
    rows = [GOOD,
            dict(GOOD, emotion=[0, 0, 0, 0, 0, 0, 0, 1.0]),   # parsed, emotion not omitted
            dict(GOOD, speaking_rate=17),                     # parsed, rate out
            dict(GOOD, pitch_std=60),                         # parsed, pitch out
            dict(GOOD),                                       # duplicate of the first
            {"voice_design": "x"},                            # missing keys
            None]                                             # unparseable reply
    s = bench.score(rows)
    assert s == {"n": 7, "parsed": 5, "emotion_omitted": 4, "rate": 4, "pitch": 4,
                 "distinct": 4}


def test_score_rejects_what_the_casting_validator_rejects():
    assert not bench.parsed(dict(GOOD, speaking_rate=99))   # outside zonos 5-30
    assert bench.parsed(GOOD)


# ------------------------------------------------------------------ verdict

def _s(parsed=24, emotion=24, rate=24, pitch=24, n=24):
    return {"n": n, "parsed": parsed, "emotion_omitted": emotion, "rate": rate,
            "pitch": pitch, "distinct": 10}


def test_invalid_when_the_reference_misses_a_parse():
    v, _ = bench.verdict(_s(parsed=23), _s(), 24)
    assert v == "INVALID"


def test_invalid_when_the_reference_is_two_below_on_a_column():
    v, _ = bench.verdict(_s(rate=22), _s(), 24)
    assert v == "INVALID"


def test_pass_at_exactly_one_below_the_reference():
    assert bench.verdict(_s(), _s(emotion=23, rate=23, pitch=23), 24) == ("PASS", [])


def test_fail_at_two_below_names_the_column():
    v, reasons = bench.verdict(_s(), _s(pitch=22), 24)
    assert v == "FAIL" and any("pitch" in r for r in reasons)


def test_fail_when_the_candidate_misses_a_parse():
    v, reasons = bench.verdict(_s(), _s(parsed=23), 24)
    assert v == "FAIL" and any("parsed" in r for r in reasons)


def test_fail_when_the_volume_smoke_misses():
    v, reasons = bench.verdict(_s(), _s(), 23)
    assert v == "FAIL" and any("volume" in r for r in reasons)


def test_verdict_refuses_an_empty_or_mismatched_run():
    with pytest.raises(ValueError):
        bench.verdict(_s(n=0), _s(n=0), 0)
    with pytest.raises(ValueError):
        bench.verdict(_s(n=24), _s(n=23), 24)


# ------------------------------------------------------------------ the arm runner

def test_run_arm_sends_casting_messages_and_records_a_failed_call(monkeypatch):
    sent = []

    def call(system, user, schema):
        sent.append((system, user, schema))
        if len(sent) == 2:
            raise RuntimeError("boom")
        return json.dumps(GOOD)

    passages = [dict(id=f"p{i}", book="b", text=f"Line {i}.", register="neutral_narration",
                     V=0.0, A=0.1, T=0.2) for i in range(3)]
    rows = bench.run_arm(call, passages)
    assert [r["casting"] for r in rows] == [GOOD, None, GOOD]
    assert rows[1]["error"] == "RuntimeError('boom')"
    labels = {"V": 0.0, "A": 0.1, "T": 0.2, "register": "neutral_narration"}
    assert sent[0][:2] == bench.casting_messages("Line 0.", "zonos", labels)
    assert sent[0][2] == bench._json_schema("zonos")


# ------------------------------------------------------------------ main and the timeouts

def test_a_run_without_a_recorded_reference_is_refused_before_anything_is_written(
        tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"

    def no_network(*a, **k):
        raise AssertionError("nothing may be called without a reference")

    monkeypatch.setattr(bench, "ASSET", tmp_path / "unread.json")
    monkeypatch.setattr(bench, "chat", no_network)
    monkeypatch.setattr(bench, "GemmaServer", no_network)
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out)])
    with pytest.raises(SystemExit) as e:
        bench.main()
    assert e.value.code == 2
    assert "--reference RESULTS.json is required" in capsys.readouterr().err
    assert not out.exists()


def test_candidate_call_uses_the_same_timeout_as_the_reference_arm(monkeypatch):
    seen = {}
    monkeypatch.setattr(bench, "chat", lambda *a, **k: seen.update(k) or "{}")
    bench.candidate_call("s", "u", {"type": "object"})
    assert seen["timeout"] == 300


# ------------------------------------------------------------------ the servers

PASSING_REFERENCE = {"n": 2, "parsed": 2, "emotion_omitted": 2, "rate": 2, "pitch": 2,
                     "distinct": 1}


def _two_passage_run(tmp_path, monkeypatch):
    asset = tmp_path / "passages.json"
    asset.write_text(json.dumps([dict(id=f"p{i}", book="b", text=f"Line {i}.",
                                      register="neutral_narration", V=0.0, A=0.1, T=0.2)
                                 for i in range(2)]))
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"arms": {"reference": {"source": "recorded",
                                                      "scores": PASSING_REFERENCE}}}))
    out = tmp_path / "out"
    monkeypatch.setattr(bench, "ASSET", asset)
    monkeypatch.setattr(sys, "argv", ["x", "--reference", str(ref), "--out", str(out)])
    return out


def test_the_candidate_arm_and_the_volume_smoke_each_run_inside_their_own_server(
        tmp_path, monkeypatch):
    events = []

    class FakeServer:
        def __init__(self, role):
            self.role = role

        def __enter__(self):
            events.append(("enter", self.role))
            return self

        def __exit__(self, *exc):
            events.append(("exit", self.role))
            return False

    monkeypatch.setattr(bench, "GemmaServer", FakeServer)
    monkeypatch.setattr(bench, "chat", lambda *a, **k: events.append(("call", "candidate"))
                        or json.dumps(GOOD))
    monkeypatch.setattr(bench.judge_passages, "ask",
                        lambda role, text: events.append(("call", "volume", role))
                        or ({"unit": True}, None))
    out = _two_passage_run(tmp_path, monkeypatch)
    bench.main()
    # warm-up + 2 casting calls, then the smoke's 2 calls; nothing outside a server
    assert events == [("enter", "director"), *[("call", "candidate")] * 3, ("exit", "director"),
                      ("enter", "volume"), *[("call", "volume", "volume")] * 2,
                      ("exit", "volume")]
    saved = json.loads(next(out.glob("bench_*.json")).read_text(encoding="utf-8"))
    assert saved["verdict"] == "PASS" and "error" not in saved


def test_a_server_that_fails_to_start_still_leaves_the_results_file_with_the_error(
        tmp_path, monkeypatch):
    class Refusing:
        def __init__(self, role):
            pass

        def __enter__(self):
            raise gemma_server.ServerError("no GPU")

        def __exit__(self, *exc):
            return False

    def no_network(*a, **k):
        raise AssertionError("nothing may be called without a server")

    monkeypatch.setattr(bench, "GemmaServer", Refusing)
    monkeypatch.setattr(bench, "chat", no_network)
    monkeypatch.setattr(bench.judge_passages, "ask", no_network)
    out = _two_passage_run(tmp_path, monkeypatch)
    with pytest.raises(gemma_server.ServerError):
        bench.main()
    saved = json.loads(next(out.glob("bench_*.json")).read_text(encoding="utf-8"))
    assert "no GPU" in saved["error"]
    assert saved["arms"]["reference"]["scores"] == PASSING_REFERENCE
    assert "candidate" not in saved["arms"]


# ------------------------------------------------------------------ the committed asset

def test_the_committed_passages_are_the_pre_registered_set():
    passages = json.loads((ASSETS / "director_bench_passages.json").read_text(encoding="utf-8"))
    assert len(passages) == 24
    assert len({p["id"] for p in passages}) == 24
    assert len({p["book"] for p in passages}) >= 4
    for p in passages:
        assert set(p) == {"id", "book", "text", "register", "V", "A", "T"}
        assert "_nar_" in p["id"]
