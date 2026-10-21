"""Phase 1S Step 1c-0's ladder is the pre-registered one, and its F0 measure sees the top of it."""
import importlib.util
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("render_qwen_1c0",
                                               REPO / "scripts/tools/render_qwen_1c0.py")
rq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq)

# The pre-registration's text, copied here so a drift in either is a failure.
PREREG_M4 = ("gender: Male. pitch: Medium male pitch, stable. speed: Steady, unhurried. volume: "
             "Moderate conversational. age: Middle-aged adult. clarity: High clarity with distinct "
             "pronunciation. fluency: Highly fluent. accent: American English. texture: Clear and "
             "natural. emotion: Calm, even. tone: Even, narratorial. personality: Composed and "
             "unobtrusive.")


def _keys(instruct):
    return dict(p.split(": ", 1) for p in instruct.rstrip(".").split(". "))


def test_sixty_items_in_listening_order_with_the_registered_seeds():
    items = rq.ladder()
    assert len(items) == 60
    assert [i["id"] for i in items[:3]] == ["M1-s1", "M1-s2", "M1-s3"]
    assert [i["id"] for i in items[21:24]] == ["F1-s1", "F1-s2", "F1-s3"]
    assert [i["id"] for i in items[-3:]] == ["P6-s1", "P6-s2", "P6-s3"]
    assert len({i["id"] for i in items}) == 60 and len({i["seed"] for i in items}) == 60
    seeds = {i["id"]: i["seed"] for i in items}
    assert (seeds["M1-s1"], seeds["M4-s2"], seeds["F7-s3"], seeds["P3-s1"]) == (9111, 9142, 9273, 9331)


def test_the_middle_rung_is_the_registered_instruct():
    assert next(i for i in rq.ladder() if i["id"] == "M4-s1")["instruct"] == PREREG_M4


def test_the_ladder_moves_pitch_and_nothing_else():
    for g in ("M", "F"):
        rungs = [_keys(i["instruct"]) for i in rq.ladder() if i["id"].startswith(g) and i["sentence"] == 1]
        assert len(rungs) == 7
        assert len({r["pitch"] for r in rungs}) == 7
        for r in rungs:
            assert {k: v for k, v in r.items() if k != "pitch"} == \
                {k: v for k, v in rungs[0].items() if k != "pitch"}
    pitches = [_keys(i["instruct"])["pitch"] for i in rq.ladder() if i["id"].startswith("M") and i["sentence"] == 1]
    assert pitches[0] == "Extremely low male pitch, stable" and pitches[-1] == "Extremely high male pitch, stable"


def test_personas_change_only_gender_pitch_age_texture():
    fixed = {k: v for k, v in rq.FIXED.items() if k not in ("age", "texture")}
    for i in rq.ladder():
        if i["group"] != "niche personas":
            continue
        k = _keys(i["instruct"])
        assert {key: k[key] for key in fixed} == fixed
    bands = [i["band"] for i in rq.ladder() if i["group"] == "niche personas" and i["sentence"] == 1]
    assert bands == ["Child girl", "Young girl", "Child boy", "Man", "Elderly man", "Elderly woman"]


def test_every_item_reads_the_same_three_sentences():
    for i in rq.ladder():
        assert i["text"] == rq.SENTENCES[i["sentence"] - 1]
    assert len(set(rq.SENTENCES)) == 3


def test_build_instruct_refuses_a_missing_or_unknown_key():
    keys = dict(rq.FIXED, gender="Male", pitch="Low male pitch, stable")
    assert rq.build_instruct(keys).startswith("gender: Male. pitch: Low male pitch, stable. ")
    with pytest.raises(ValueError):
        rq.build_instruct({k: v for k, v in keys.items() if k != "tone"})
    with pytest.raises(ValueError):
        rq.build_instruct(dict(keys, colour="blue"))


def test_render_refuses_any_qwen_tts_but_the_studied_release():
    assert rq.require_qwen_tts(lambda _pkg: "0.1.1") == "0.1.1"
    with pytest.raises(SystemExit, match="REFUSING"):
        rq.require_qwen_tts(lambda _pkg: "0.1.2")


def _voice(f0, sr=24000, seconds=1.5):
    t = np.arange(int(sr * seconds)) / sr
    return 0.3 * sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3, 4)), sr


@pytest.mark.parametrize("f0", [90.0, 220.0, 520.0])
def test_f0_profile_finds_a_voice_across_the_whole_band(f0):
    x, sr = _voice(f0)
    got = rq.f0_profile(x, sr)
    assert got is not None and abs(got["f0_median"] - f0) / f0 < 0.03, got


def test_the_repo_default_band_would_miss_a_child_voice():
    """Positive control for the wide band: at measure_speaker_hnr's 350 Hz ceiling, a 520 Hz voice
    is not measured as 520 Hz, which is why this probe measures to 700 Hz."""
    from periodicity import frames
    x, sr = _voice(520.0)
    f0, _h = frames(np.asarray(x, dtype=np.float64), sr, 55.0, 350.0, 0.01, 0.3)
    assert not len(f0) or abs(float(np.median(f0)) - 520.0) / 520.0 > 0.1


def test_f0_profile_is_none_on_silence():
    assert rq.f0_profile(np.zeros(24000), 24000) is None
