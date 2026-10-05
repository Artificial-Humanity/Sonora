"""The pure half of the director obedience bench: which passages, how a reply scores, and
how two arms' scores become the pre-registered verdict."""

import json

import pytest

from scripts_layout import ASSETS, SCRIPTS

SCRIPTS.on_path()
import director_obedience_bench as bench  # noqa: E402

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


# ------------------------------------------------------------------ the committed asset

def test_the_committed_passages_are_the_pre_registered_set():
    passages = json.loads((ASSETS / "director_bench_passages.json").read_text(encoding="utf-8"))
    assert len(passages) == 24
    assert len({p["id"] for p in passages}) == 24
    assert len({p["book"] for p in passages}) >= 4
    for p in passages:
        assert set(p) == {"id", "book", "text", "register", "V", "A", "T"}
        assert "_nar_" in p["id"]
