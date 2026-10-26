"""Tests for the Ear Test and Dataset Listening slider surfaces (notes/listening-sliders.md).

Covers:
1. Ear Test (`eartest/app/main.py`):
   - Legacy forced-choice and single-slider severity sets stay intact.
   - Per-defect severity sets (`scale: {"defects": True, ...}`) expose five Unset (-1) -> 0..5
     sliders per side (`buzz`, `hum`, `layering`, `noise`, `robot`), derive `sev_a`/`sev_b`
     and `choice`, require all ten sliders to be set (0 is a valid answer), and migrate
     `verdicts_history.csv` headers without losing old rows.
2. Dataset Listening (`audition/app/main.py` & `static/index.html`):
   - Five-band per-gender Age taxonomy (`Child girl..Elderly woman`, `Child boy..Elderly man`).
   - Delivery v2 Discourse Contexts (5) and Vocal Manners (8) imported from `matcha.delivery`.
   - Two-panel card layout (`Voice & Delivery` vs `Quality & Defects`).
   - Pre-filled attribute sliders (`age` 1-5, `pitch` 1-5, `depth` 1-5, `strain` 0-5)
     recording `intended_*`, `audited_*`, `delta_*` (`audited - intended`), and `touched_*`.
   - Five 0-5 defect sliders (`buzz`, `hum`, `layering`, `noise`, `robot`, default 0).
   - Calibration probe endpoints (`/api/probes`, `/api/probe/{name}`, `POST /api/probe/{name}/rate`)
     saving `zone` (`core` / `niche` / `exclude`) and sliders to `probes/<name>/verdicts.csv`
     without touching `ratings.csv`.
"""

import csv
import importlib
import json
import pathlib
import sys

import pytest
from fastapi import HTTPException

REPO = pathlib.Path(__file__).resolve().parent.parent


def _load_eartest(tmp_path, monkeypatch, items_payload):
    monkeypatch.setenv("EARTEST_ROOT", str(tmp_path))
    (tmp_path / "clips").mkdir(parents=True, exist_ok=True)
    (tmp_path / "items.json").write_text(json.dumps(items_payload), encoding="utf-8")
    eartest_dir = str(REPO / "eartest")
    if eartest_dir not in sys.path:
        sys.path.insert(0, eartest_dir)
    sys.modules.pop("eartest_main", None)
    spec = importlib.util.spec_from_file_location(
        "eartest_main", REPO / "eartest/app/main.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_audition(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDITION_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDITION_RATINGS_DIR", str(tmp_path))
    ratings = tmp_path / "ratings.csv"
    if not ratings.exists():
        with ratings.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "campaign", "id", "engine", "register", "gender", "age",
                    "accent", "delivery", "score", "note", "status", "link",
                ],
            )
            w.writeheader()
            w.writerow({
                "campaign": "camp1", "id": "clip_01_F_s1", "engine": "qwen",
                "register": "neutral_narration", "gender": "Female", "age": "",
                "accent": "US - General", "delivery": "Neutral", "score": "",
                "note": "", "status": "unaudited", "link": "camp1/clip_01_F_s1.wav",
            })
    sys.modules.pop("audition_main", None)
    spec = importlib.util.spec_from_file_location(
        "audition_main", REPO / "audition/app/main.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- 1. Ear Test per-defect sliders -------------------------------------------


def test_eartest_per_defect_sliders_save_and_derive_choice(tmp_path, monkeypatch):
    payload = {
        "test": "defect_bench",
        "sets": {
            "hum_defects": {
                "title": "Per-defect bench",
                "ask": "Rate each defect 0-5.",
                "scale": {"defects": True, "primary": "hum"},
            }
        },
        "items": [
            {"id": "item_01", "set": "hum_defects", "text": "Hello.", "A": "a1", "B": "b1"}
        ],
    }
    mod = _load_eartest(tmp_path, monkeypatch, payload)
    assert mod.DEFECT_KEYS == ("buzz", "hum", "layering", "noise", "robot")

    # Partial defect sliders -> item is saved but not yet counted as judged
    mod.verdict(mod.Verdict(
        item="item_01",
        buzz_a="0", hum_a="3", layering_a="0", noise_a="0", robot_a="1",
        buzz_b="0", hum_b="-1", layering_b="0", noise_b="0", robot_b="0",
    ))
    prog = mod.progress()
    assert prog["hum_defects"] == {"total": 1, "done": 0}

    # Complete all 10 sliders (0 is a real answer) -> judged, primary 'hum' drives sev_a/sev_b
    mod.verdict(mod.Verdict(
        item="item_01",
        buzz_a="2", hum_a="3", layering_a="1", noise_a="0", robot_a="2",
        buzz_b="0", hum_b="1", layering_b="0", noise_b="0", robot_b="0",
    ))
    prog = mod.progress()
    assert prog["hum_defects"] == {"total": 1, "done": 1}

    rows = mod._read()
    r = rows["item_01"]
    assert r["sev_a"] == "3"
    assert r["sev_b"] == "1"
    assert r["choice"] == "A"
    assert r["buzz_a"] == "2"
    assert r["layering_a"] == "1"
    assert r["noise_a"] == "0"
    assert r["robot_a"] == "2"


def test_eartest_legacy_single_slider_set_still_works(tmp_path, monkeypatch):
    payload = {
        "test": "legacy_hum",
        "sets": {
            "hum": {
                "title": "Legacy hum",
                "ask": "Rate hum 0-5.",
                "scale": {"anchors": {"0": "none", "5": "bad"}},
            }
        },
        "items": [
            {"id": "item_01", "set": "hum", "text": "Hello.", "A": "a1", "B": "b1"}
        ],
    }
    mod = _load_eartest(tmp_path, monkeypatch, payload)
    mod.verdict(mod.Verdict(item="item_01", sev_a="0", sev_b="2"))
    assert mod.progress()["hum"] == {"total": 1, "done": 1}
    r = mod._read()["item_01"]
    assert r["choice"] == "B"
    assert r["sev_a"] == "0"
    assert r["sev_b"] == "2"


def test_eartest_html_includes_five_defect_sliders():
    src = (REPO / "eartest/app/main.py").read_text(encoding="utf-8")
    for label in ("Buzz", "Hum", "Layering", "Background noise", "Human–Robot"):
        assert label in src, f"missing {label!r} in eartest UI"


# --- 2. Dataset Listening two-panel sliders & delta tracking ------------------


def test_audition_age_bands_and_delivery_v2_vocabularies(tmp_path, monkeypatch):
    mod = _load_audition(tmp_path, monkeypatch)
    assert mod.AGE_BANDS_BY_GENDER["Female"] == (
        "Child girl", "Young girl", "Young woman", "Woman", "Elderly woman"
    )
    assert mod.AGE_BANDS_BY_GENDER["Male"] == (
        "Child boy", "Young boy", "Young man", "Man", "Elderly man"
    )
    assert mod.DISCOURSE_CONTEXTS == (
        "", "Narration", "Dialogue", "Internal", "Address", "Verse"
    )
    assert mod.VOCAL_MANNERS == (
        "", "Measured", "Conversational", "Intimate", "Crisp",
        "Oratory", "Hushed", "Projected", "Cadenced",
    )


def test_audition_rate_persists_sliders_deltas_and_defects(tmp_path, monkeypatch):
    mod = _load_audition(tmp_path, monkeypatch)
    res = mod.rate(mod.Rating(
        id="clip_01_F_s1",
        action="keep",
        score=4,
        note="slightly young",
        context="Narration",
        manner="Measured",
        intended_age=4,
        audited_age=3,
        touched_age=True,
        intended_pitch=3,
        audited_pitch=4,
        touched_pitch=True,
        intended_depth=3,
        audited_depth=3,
        touched_depth=False,
        intended_strain=0,
        audited_strain=1,
        touched_strain=True,
        buzz=1,
        hum=0,
        layering=2,
        noise=0,
        robot=1,
    ))
    row = res["row"]
    assert row["score"] == "4"
    assert row["status"] == "keep"
    assert row["context"] == "Narration"
    assert row["manner"] == "Measured"
    # Female age 3 -> Young woman, delta_age = 3 - 4 = -1
    assert row["age"] == "Young woman"
    assert row["intended_age"] == "4"
    assert row["audited_age"] == "3"
    assert row["delta_age"] == "-1"
    assert row["touched_age"] == "1"
    # Pitch: intended 3, audited 4 -> delta +1
    assert row["delta_pitch"] == "+1"
    assert row["touched_pitch"] == "1"
    # Untouched depth: touched_depth is "0", audited_depth & delta_depth stay blank
    assert row["intended_depth"] == "3"
    assert row["touched_depth"] == "0"
    assert row["audited_depth"] == ""
    assert row["delta_depth"] == ""
    # Strain: intended 0, audited 1 -> delta +1
    assert row["delta_strain"] == "+1"
    # Defects (0-5)
    assert row["buzz"] == "1"
    assert row["hum"] == "0"
    assert row["layering"] == "2"
    assert row["noise"] == "0"
    assert row["robot"] == "1"


def test_audition_static_html_has_two_panels_and_five_defect_sliders():
    html = (REPO / "audition/app/static/index.html").read_text(encoding="utf-8")
    assert "Voice &amp; Delivery" in html or "Voice & Delivery" in html
    assert "Quality &amp; Defects" in html or "Quality & Defects" in html
    for label in ("Buzz", "Hum", "Layering", "Background noise", "Human–Robot"):
        assert label in html, f"missing {label!r} in audition static/index.html"
    assert "function fmtAttrVal(" in html
    assert "(Δ ${dStr})" in html
    assert "(+ac.f0_median_hz).toFixed(1)" in html


# --- 3. Calibration Probe Mode in Dataset Listening (1c-0) --------------------


def test_audition_probe_mode_saves_without_touching_ratings_csv(tmp_path, monkeypatch):
    mod = _load_audition(tmp_path, monkeypatch)
    ratings_before = (tmp_path / "ratings.csv").read_text(encoding="utf-8")

    probe_dir = tmp_path / "probes" / "qwen_1c0"
    probe_dir.mkdir(parents=True)
    manifest = {
        "probe": "qwen_1c0",
        "title": "Phase 1S Step 1c-0 — Qwen Anchor Range Calibration",
        "clips": [
            {
                "id": "M4_s1",
                "group": "M4",
                "gender": "Male",
                "text": "By the time the rain stopped...",
                "wav": "probes/qwen_1c0/M4_s1.wav",
                "acoustics": {"f0_median_hz": 115.2, "f0_excursion_hz": 34.1, "hnr_db": 18.4},
                "intended_age": 4,
                "intended_pitch": 3,
                "intended_depth": 3,
                "intended_strain": 0,
            }
        ],
    }
    (probe_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    probes = mod.list_probes()
    assert probes["probes"] == ["qwen_1c0"]

    res = mod.rate_probe(
        "qwen_1c0",
        mod.ProbeRating(
            id="M4_s1",
            zone="core",
            score=5,
            intended_age=4,
            audited_age=4,
            touched_age=True,
            intended_pitch=3,
            audited_pitch=3,
            touched_pitch=True,
            buzz=0,
            hum=0,
            layering=0,
            noise=0,
            robot=0,
            note="clean core narrator",
        ),
    )
    assert res["ok"] is True
    assert res["row"]["zone"] == "core"
    assert res["row"]["delta_age"] == "0"

    loaded = mod.get_probe("qwen_1c0")
    assert loaded["clips"][0]["zone"] == "core"
    assert loaded["clips"][0]["score"] == "5"
    assert (tmp_path / "ratings.csv").read_text(encoding="utf-8") == ratings_before


def test_audition_probe_mode_reads_manifest_jsonl_and_measures_jsonl(tmp_path, monkeypatch):
    mod = _load_audition(tmp_path, monkeypatch)
    probe_dir = tmp_path / "probes" / "qwen-1c0"
    (probe_dir / "clips").mkdir(parents=True)
    (probe_dir / "clips" / "M1_s1.wav").write_bytes(b"RIFF")
    (probe_dir / "manifest.jsonl").write_text(
        json.dumps({
            "id": "M1_s1", "gender": "Male", "rung": "M1", "age_band": "Man",
            "instruct": "gender: Male. pitch: Extremely low male pitch, stable.",
            "text": "By the time the rain stopped...", "seed": 9111, "seconds": 6.1,
        }) + "\n",
        encoding="utf-8",
    )
    (probe_dir / "measures.jsonl").write_text(
        json.dumps({
            "id": "M1_s1", "f0_median_hz": 84.5, "f0_excursion_hz": 22.0,
            "hnr_db": 14.1, "cpp_db": 10.8,
        }) + "\n",
        encoding="utf-8",
    )
    loaded = mod.get_probe("qwen-1c0")
    assert len(loaded["clips"]) == 1
    clip = loaded["clips"][0]
    assert clip["id"] == "M1_s1"
    assert clip["acoustics"]["f0_median_hz"] == 84.5
    assert clip["wav"].endswith("clips/M1_s1.wav")

    # Rating a manifest.jsonl probe clip with no intended_* fields must not fabricate
    # intended_age="3" or delta_age when touched_age=False, and must record audited_age
    # with blank delta_age when touched_age=True and intended_age=None.
    res_untouched = mod.rate_probe(
        "qwen-1c0",
        mod.ProbeRating(
            id="M1_s1",
            zone="core",
            audited_age=3,
            touched_age=False,
        ),
    )
    assert res_untouched["row"].get("intended_age", "") == ""
    assert res_untouched["row"].get("audited_age", "") == ""
    assert res_untouched["row"].get("delta_age", "") == ""
    assert res_untouched["row"].get("touched_age") == "0"

    res_touched = mod.rate_probe(
        "qwen-1c0",
        mod.ProbeRating(
            id="M1_s1",
            zone="niche",
            intended_age=None,
            audited_age=4,
            touched_age=True,
        ),
    )
    assert res_touched["row"].get("intended_age", "") == ""
    assert res_touched["row"].get("audited_age") == "4"
    assert res_touched["row"].get("delta_age", "") == ""
    assert res_touched["row"].get("touched_age") == "1"

    vcsv = (probe_dir / "verdicts.csv").read_text(encoding="utf-8")
    vrows = list(csv.DictReader(vcsv.splitlines()))
    assert len(vrows) == 1
    assert vrows[0]["intended_age"] == ""
    assert vrows[0]["audited_age"] == "4"
    assert vrows[0]["delta_age"] == ""
    assert vrows[0]["touched_age"] == "1"


