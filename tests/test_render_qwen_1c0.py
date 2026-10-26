"""Phase 1S Step 1c-0's ladder and 1c-0b's methods are the pre-registered ones, the F0 measure reads
the whole range or says it cannot, the renderer imports in the qwen container and decodes in float32,
a resume cannot mix two renders or two probes, 1c-0b's steer reading is the spec's rule, and the
audition app's probe mode reads the folder a render leaves."""
import argparse
import ast
import importlib.util
import json
import pathlib
import subprocess
import sys

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
# § 1c-0's rung table, verbatim: rung | `pitch:` value | seeds (sentences 1-3).
PREREG_RUNGS = """\
| M1 | `pitch: Extremely low male pitch, stable.` | 9111–9113 |
| M2 | `pitch: Very low male pitch, stable.` | 9121–9123 |
| M3 | `pitch: Low male pitch, stable.` | 9131–9133 |
| M4 | `pitch: Medium male pitch, stable.` | 9141–9143 |
| M5 | `pitch: High male pitch, stable.` | 9151–9153 |
| M6 | `pitch: Very high male pitch, stable.` | 9161–9163 |
| M7 | `pitch: Extremely high male pitch, stable.` | 9171–9173 |
| F1 | `pitch: Extremely low female pitch, stable.` | 9211–9213 |
| F2 | `pitch: Very low female pitch, stable.` | 9221–9223 |
| F3 | `pitch: Low female pitch, stable.` | 9231–9233 |
| F4 | `pitch: Medium female pitch, stable.` | 9241–9243 |
| F5 | `pitch: High female pitch, stable.` | 9251–9253 |
| F6 | `pitch: Very high female pitch, stable.` | 9261–9263 |
| F7 | `pitch: Extremely high female pitch, stable.` | 9271–9273 |
"""
# § Niche personas' table, verbatim: persona | age band | gender | pitch | age | texture | seeds.
PREREG_PERSONAS = """\
| P1 | Child girl | Female | High child pitch, stable | Child, about eight years old | Light and clear | 9311–9313 |
| P2 | Young girl | Female | High, stable | Young teenager, about thirteen | Light and natural | 9321–9323 |
| P3 | Child boy | Male | High child pitch, stable | Child, about nine years old | Light and clear | 9331–9333 |
| P4 | Man (large) | Male | Very low male pitch, stable | Adult | Deep, full and chest-resonant | 9341–9343 |
| P5 | Elderly man | Male | Low male pitch, stable | Elderly, about seventy-five | Slightly hoarse, with light age strain | 9351–9353 |
| P6 | Elderly woman | Female | Low female pitch, stable | Elderly, about seventy-five | Slightly hoarse, with light age strain | 9361–9363 |
"""


def _keys(instruct):
    return dict(p.split(": ", 1) for p in instruct.rstrip(".").split(". "))


def _table(text):
    return [[c.strip() for c in line.strip("|").split("|")] for line in text.splitlines()]


def _seeds(cell):
    lo, hi = (int(s) for s in cell.split("–"))
    return list(range(lo, hi + 1))


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


def test_every_rung_is_the_spec_tables_pitch_and_seeds():
    rows = _table(PREREG_RUNGS)
    assert len(rows) == 14 and sum(r[0].startswith("F") for r in rows) == 7
    by_id = {i["id"]: i for i in rq.ladder()}
    for rung, pitch, seeds in rows:
        got = [by_id["%s-s%d" % (rung, s)] for s in (1, 2, 3)]
        assert ["pitch: %s." % _keys(i["instruct"])["pitch"] for i in got] == [pitch.strip("`")] * 3
        assert [i["seed"] for i in got] == _seeds(seeds)
        assert {i["gender"] for i in got} == {"Male" if rung[0] == "M" else "Female"}
        assert {_keys(i["instruct"])["gender"] for i in got} == {got[0]["gender"]}


def test_every_persona_is_the_spec_tables_row():
    rows = _table(PREREG_PERSONAS)
    assert len(rows) == 6
    by_id = {i["id"]: i for i in rq.ladder()}
    for pid, band, gender, pitch, age, texture, seeds in rows:
        got = [by_id["%s-s%d" % (pid, s)] for s in (1, 2, 3)]
        for i in got:
            k = _keys(i["instruct"])
            assert (k["gender"], k["pitch"], k["age"], k["texture"]) == (gender, pitch, age, texture)
            assert i["gender"] == gender
            # The taxonomy has no size band: "Man (large)" is band Man, size goes in the label.
            assert i["band"] == band.replace(" (large)", "")
            assert i["label"].startswith(pid + " ")
        assert [i["seed"] for i in got] == _seeds(seeds)
    assert by_id["P4-s1"]["label"] == "P4 large man"


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


# --- 1c-0b's plan ---------------------------------------------------------------------------

# § 1c-0b's methods, as the spec words them: (id part, key, male low, male high, female low,
# female high). None for the key is the free-text instruct, which replaces the 12 keys.
PREREG_1C0B = (
    ("num", "pitch", "Very low, around 80 Hz, stable", "Very high, around 160 Hz, stable",
     "Very low, around 150 Hz, stable", "Very high, around 300 Hz, stable"),
    ("desc", "pitch", "Deep bass, far below an average man's voice, stable",
     "Light and high, far above an average man's voice, stable",
     "Deep bass, far below an average woman's voice, stable",
     "Light and high, far above an average woman's voice, stable"),
    ("age", "age", "Elderly man", "Young man", "Elderly woman", "Young woman"),
    ("weight", "texture", "Deep, heavy and chest-resonant", "Light, thin and head-voiced",
     "Deep, heavy and chest-resonant", "Light, thin and head-voiced"),
    ("free", None,
     "A man with a very deep, low-pitched voice, calm and composed, reading narration in American English.",
     "A man with an unusually high-pitched voice, calm and composed, reading narration in American English.",
     "A woman with a very deep, low-pitched voice, calm and composed, reading narration in American English.",
     "A woman with an unusually high-pitched voice, calm and composed, reading narration in American English."),
)


def test_1c0b_is_66_items_in_listening_order():
    items = rq.plan_1c0b()
    assert len(items) == 66 and len({i["id"] for i in items}) == 66
    arms = ["base"] + ["%s-%s" % (m[0], pole) for m in PREREG_1C0B for pole in ("lo", "hi")]
    want = ["%s-%s-k%d" % (g, arm, k) for g in "MF" for arm in arms for k in (1, 2, 3)]
    assert [i["id"] for i in items] == want
    assert want[:4] == ["M-base-k1", "M-base-k2", "M-base-k3", "M-num-lo-k1"]
    assert [i["gender"] for i in items] == ["Male"] * 33 + ["Female"] * 33
    assert {i["group"] for i in items[:33]} == {"Male 1c-0b"}
    assert {i["group"] for i in items[33:]} == {"Female 1c-0b"}
    assert [i["band"] for i in items] == ["Man"] * 33 + ["Woman"] * 33
    for i in items:
        assert set(i) == {"id", "group", "method", "pole", "gender", "seed", "sentence", "text",
                          "instruct", "label", "band"}, i
        assert i["sentence"] == 2 and i["text"] == rq.SENTENCES[1]
        assert (i["method"] == "baseline") is (i["pole"] is None)


def test_1c0b_seeds_are_unique_deterministic_and_disjoint_from_1c0s():
    items = rq.plan_1c0b()
    seeds = [i["seed"] for i in items]
    assert len(set(seeds)) == 66 and seeds == [i["seed"] for i in rq.plan_1c0b()]
    assert not set(seeds) & {i["seed"] for i in rq.ladder()}
    by_id = {i["id"]: i["seed"] for i in items}
    assert (by_id["M-base-k1"], by_id["M-num-lo-k2"], by_id["M-num-hi-k3"], by_id["F-free-hi-k3"],
            by_id["F-age-lo-k1"]) == (951001, 951112, 951123, 952523, 952311)
    methods = ["baseline"] + [m[1] for m in rq.STEER_METHODS]
    for i in items:
        g = 1 if i["gender"] == "Male" else 2
        pole = {None: 0, "low": 1, "high": 2}[i["pole"]]
        k = int(i["id"].rsplit("-k", 1)[1])
        assert i["seed"] == int("95%d%d%d%d" % (g, methods.index(i["method"]), pole, k)), i


def test_the_1c0b_baseline_is_1c0s_rung_4():
    ladder = {i["id"]: i for i in rq.ladder()}
    items = {i["id"]: i for i in rq.plan_1c0b()}
    assert items["M-base-k1"]["instruct"] == ladder["M4-s2"]["instruct"] == PREREG_M4
    assert items["F-base-k2"]["instruct"] == ladder["F4-s2"]["instruct"]
    assert items["M-base-k3"]["text"] == ladder["M4-s2"]["text"]


def test_each_1c0b_method_sets_its_spec_words_and_changes_only_its_key():
    by_id = {i["id"]: i for i in rq.plan_1c0b()}
    assert [m[0] for m in rq.STEER_METHODS] == [m[0] for m in PREREG_1C0B]
    for part, key, m_lo, m_hi, f_lo, f_hi in PREREG_1C0B:
        for g, lo, hi in (("M", m_lo, m_hi), ("F", f_lo, f_hi)):
            base = _keys(by_id["%s-base-k1" % g]["instruct"])
            for pole, value in (("lo", lo), ("hi", hi)):
                got = [by_id["%s-%s-%s-k%d" % (g, part, pole, k)] for k in (1, 2, 3)]
                assert len({i["instruct"] for i in got}) == 1
                assert {i["pole"] for i in got} == {"low" if pole == "lo" else "high"}
                if key is None:
                    assert got[0]["instruct"] == value
                    assert ": " not in got[0]["instruct"]          # not the 12-key format
                    continue
                k = _keys(got[0]["instruct"])
                assert k[key] == value, (part, g, pole, k)
                assert {n: v for n, v in k.items() if n != key} == \
                    {n: v for n, v in base.items() if n != key}, (part, g, pole)
                assert k[key] != base[key]


def test_the_1c0b_age_method_uses_the_ratified_band_words():
    """The band labels as written, the taxonomy's own words (audition's AGE_BANDS_BY_GENDER)."""
    ages = {(i["gender"], i["pole"]): _keys(i["instruct"])["age"]
            for i in rq.plan_1c0b() if i["method"] == "age"}
    assert ages == {("Male", "low"): "Elderly man", ("Male", "high"): "Young man",
                    ("Female", "low"): "Elderly woman", ("Female", "high"): "Young woman"}
    src = (REPO / "audition/app/main.py").read_text(encoding="utf-8")
    for band in ages.values():
        assert '"%s"' % band in src, band


def test_the_1c0b_numeric_and_descriptive_pitch_strings():
    pitch = {(i["gender"], i["method"], i["pole"]): _keys(i["instruct"])["pitch"]
             for i in rq.plan_1c0b() if i["method"] in ("numeric pitch", "descriptive pitch")}
    assert pitch[("Male", "numeric pitch", "low")] == "Very low, around 80 Hz, stable"
    assert pitch[("Male", "numeric pitch", "high")] == "Very high, around 160 Hz, stable"
    assert pitch[("Female", "numeric pitch", "low")] == "Very low, around 150 Hz, stable"
    assert pitch[("Female", "numeric pitch", "high")] == "Very high, around 300 Hz, stable"
    assert pitch[("Male", "descriptive pitch", "low")] == \
        "Deep bass, far below an average man's voice, stable"
    assert pitch[("Female", "descriptive pitch", "high")] == \
        "Light and high, far above an average woman's voice, stable"
    instruct = next(i for i in rq.plan_1c0b() if i["id"] == "M-num-lo-k1")["instruct"]
    assert "pitch: Very low, around 80 Hz, stable. " in instruct


def test_plan_prints_the_chosen_probe(capsys):
    rq.main(["plan", "--probe", "1c0b"])
    out = capsys.readouterr().out
    assert out.rstrip().endswith("66 items") and "M-free-hi-k1" in out
    assert "A woman with an unusually high-pitched voice" in out
    rq.main(["plan"])
    out = capsys.readouterr().out
    assert out.rstrip().endswith("60 items") and "M-free" not in out


# --- F0 -------------------------------------------------------------------------------------

SR = 24000
# A real Qwen VoiceDesign clip the per-clip two-pass autocorrelation read at 235 Hz, with no flag,
# where librosa's pYIN reads about 371 Hz (16 kHz, 50-800 Hz, frame 1024: the review's reference).
QWEN_HIGH_CLIP = pathlib.Path("/data/model-training/datasets/sonora-expressive-registers/"
                              "quote-pilot-v3/qp3_04_content_G26.wav")
# Real expressive-register clips: in range at ~246 Hz with noise near 1 kHz, and above range in part.
_REGISTERS = pathlib.Path("/data/model-training/datasets/sonora-expressive-registers/v1/audio")
REAL_RANGE_CLIPS = {"grief": _REGISTERS / "grief_herenow_00_brokenMF_s5150.wav",
                    "victory": _REGISTERS / "victory_whimsical_00_brightF_s5150.wav"}


def _voice(f0, sr=SR, seconds=1.5):
    t = np.arange(int(sr * seconds)) / sr
    return 0.3 * sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3, 4)), sr


def _stack(f0, seconds=1.5, amps=None):
    """A periodic voice: harmonics at 1/k² up to 0.45 sr, or the given amplitudes."""
    t = np.arange(int(SR * seconds)) / SR
    amps = amps or [1.0 / k ** 2 for k in range(1, 400) if k * f0 < 0.45 * SR]
    return 0.3 * sum(a * np.sin(2 * np.pi * f0 * k * t) for k, a in enumerate(amps, 1)) / sum(amps)


def _glide(lo, hi, seconds=1.5):
    """A 1/k² stack whose F0 moves linearly from `lo` to `hi` Hz."""
    n = int(SR * seconds)
    phase = 2 * np.pi * np.cumsum(np.linspace(lo, hi, n)) / SR
    return 0.3 * sum(np.sin(k * phase) / k ** 2 for k in range(1, int(0.45 * SR / hi) + 1))


def _noisy(x, snr_db, seed):
    rng = np.random.default_rng(seed)
    return x + rng.standard_normal(len(x)) * np.sqrt(np.mean(x ** 2) / 10 ** (snr_db / 10))


def _pause(voice, kind, seconds=0.3, db=-20.0, seed=0):
    """A pause after `voice` holding, `db` below its level and in white noise 10 dB below that, a
    980 Hz tone or a 785-1185 Hz noise band."""
    from scipy.signal import butter, sosfiltfilt
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    if kind == "band":
        s = sosfiltfilt(butter(2, [785.0, 1185.0], btype="band", fs=SR, output="sos"),
                        rng.standard_normal(n))
    else:
        s = np.sin(2 * np.pi * 980.0 * np.arange(n) / SR)
    s = s / np.sqrt(np.mean(s ** 2)) * np.sqrt(np.mean(voice ** 2)) * 10 ** (db / 20)
    return np.concatenate([voice, _noisy(s, 10.0, seed)])


def _st(a, b):
    return 12.0 * np.log2(a / b)


@pytest.mark.parametrize("f0", [55.0, 60.0, 70.0, 80.0, 90.0, 220.0, 520.0, 650.0])
def test_f0_profile_reads_a_harmonic_stack_across_the_range(f0):
    """Within 3%, off both search limits. The audit autocorrelation agrees below its 350 Hz ceiling
    and, reading a subharmonic, disagrees above it."""
    got = rq.f0_profile(_stack(f0), SR)
    assert abs(got["f0_median"] - f0) / f0 < 0.03, got
    assert got["f0_edge_frac"] == 0 and got["f0_above_range_frac"] == 0, got
    assert not got["f0_out_of_range"], got
    assert got["f0_methods_disagree"] is (f0 > rq.ACF_BAND[1]), got
    assert got["voiced_frames"] > 100 and got["f0_range_hz"] == list(rq.F0_RANGE)


@pytest.mark.parametrize("f0", [90.0, 220.0, 520.0])
def test_f0_profile_finds_a_four_harmonic_voice(f0):
    x, sr = _voice(f0)
    got = rq.f0_profile(x, sr)
    assert got is not None and abs(got["f0_median"] - f0) / f0 < 0.03, got
    assert not got["f0_out_of_range"], got


@pytest.mark.parametrize("f0", [500.0, 600.0])
def test_a_noisy_high_voice_is_not_read_at_half(f0):
    """7 dB SNR white noise. The two-pass autocorrelation read these at 250 / 300 Hz with no flag
    (600 Hz on every seed here, 500 Hz on seed 2)."""
    for seed in range(4):
        got = rq.f0_profile(_noisy(_stack(f0), 7.0, seed), SR)
        assert abs(got["f0_median"] - f0) / f0 < 0.03, (seed, got)
        assert not got["f0_out_of_range"], (seed, got)


def test_a_noisy_glide_across_350_hz_reads_its_range():
    """250 -> 450 Hz at 10 dB SNR: the two-pass autocorrelation stayed in its low band and read the
    median at about 273 Hz."""
    got = rq.f0_profile(_noisy(_glide(250.0, 450.0), 10.0, 0), SR)
    assert abs(got["f0_median"] - 350.0) / 350.0 < 0.03, got
    assert 262.0 < got["f0_p10"] < 278.0 and 422.0 < got["f0_p90"] < 440.0, got
    assert not got["f0_out_of_range"], got


def test_a_low_glide_reads_its_range_in_semitones():
    got = rq.f0_profile(_glide(70.0, 110.0), SR)
    assert abs(got["f0_median"] - 90.0) / 90.0 < 0.03, got
    assert got["f0_edge_frac"] == 0 and not got["f0_out_of_range"], got
    assert 72.0 < got["f0_p10"] < 78.0 and 102.0 < got["f0_p90"] < 108.0, got
    assert 5.0 < got["f0_excursion_st"] < 7.0, got
    assert got["f0_excursion"] == pytest.approx(got["f0_p90"] - got["f0_p10"])
    assert got["f0_excursion_st"] == pytest.approx(_st(got["f0_p90"], got["f0_p10"]))


def test_a_brief_high_excursion_leaves_the_median_at_the_held_pitch():
    """100 Hz for 80% of the clip, 400 Hz for 20%: the median stays at 100 Hz; a mean of the voiced
    frames would read about 160 Hz. p90 shows the high stretch is voiced and counted."""
    got = rq.f0_profile(np.concatenate([_stack(100.0, 1.2), _stack(400.0, 0.3)]), SR)
    assert abs(got["f0_median"] - 100.0) / 100.0 < 0.03, got
    assert got["f0_p90"] > 300.0, got


@pytest.mark.skipif(not QWEN_HIGH_CLIP.exists(), reason="the real Qwen clip is not on this disk")
def test_a_real_high_qwen_clip_reads_where_pyin_reads_it():
    import soundfile as sf
    x, sr = sf.read(str(QWEN_HIGH_CLIP), dtype="float32")
    got = rq.f0_profile(x.mean(axis=1) if x.ndim > 1 else x, sr)
    assert abs(_st(got["f0_median"], 371.0)) < 1.0, got
    assert not got["f0_out_of_range"], got


def test_the_repo_default_band_would_miss_a_child_voice():
    """Positive control for the 700 Hz range: at measure_speaker_hnr's 350 Hz ceiling, a 520 Hz
    voice is not measured as 520 Hz."""
    from periodicity import frames
    x, sr = _voice(520.0)
    f0, _h = frames(np.asarray(x, dtype=np.float64), sr, 55.0, 350.0, 0.01, 0.3)
    assert not len(f0) or abs(float(np.median(f0)) - 520.0) / 520.0 > 0.1


def test_a_flat_wide_autocorrelation_pins_a_low_voice_to_its_top_edge():
    """Why F0 is not one 50-700 Hz `periodicity.frames` search: it reads 70 Hz as ~706 Hz."""
    from periodicity import frames
    f0, _h = frames(_stack(70.0), SR, 50.0, 700.0, 0.01, 0.3)
    assert abs(float(np.median(f0)) - SR / int(SR / 700.0)) < 1.0


@pytest.mark.parametrize("f0", [720.0, 800.0, 1000.0])
def test_a_voice_above_the_range_is_flagged(f0):
    """The 50-700 Hz pass reads these at half their F0 with no frame at an edge (positive control);
    the extended pass reads them above 700 Hz."""
    got = rq.f0_profile(_stack(f0), SR)
    assert abs(got["f0_median"] - f0 / 2) / f0 < 0.03 and got["f0_edge_frac"] == 0, got
    assert got["f0_above_range_frac"] > 0.9 and got["f0_out_of_range"], got


@pytest.mark.parametrize("seed", range(4))
def test_noise_near_1khz_that_only_the_extended_pass_voices_is_not_above_range(seed):
    """A 300 Hz voice, then 0.3 s of a noise band around 1 kHz at -20 dB: the extended pass voices
    the band above 700 Hz, the 50-700 Hz pass leaves it unvoiced. Counted over the extended pass's
    own voiced frames it read 0.12-0.13 and flagged an in-range voice."""
    got = rq.f0_profile(_pause(_stack(300.0, 1.2), "band", seed=seed), SR)
    assert abs(got["f0_median"] - 300.0) / 300.0 < 0.03, got
    assert got["f0_above_range_frac"] == 0 and not got["f0_out_of_range"], got


def test_a_steady_tone_above_the_range_counts_at_any_level():
    """The limit (module docstring): a 980 Hz tone the 50-700 Hz pass voices at 490 Hz is, frame by
    frame, a voice above the range, so a 0.3 s one at -20 dB flags the clip."""
    got = rq.f0_profile(_pause(_stack(300.0, 1.2), "tone"), SR)
    assert 0.1 < got["f0_above_range_frac"] < 0.2 and got["f0_out_of_range"], got


def test_above_range_counts_frames_both_passes_voice_over_the_first_pass(monkeypatch):
    """Frames 0-9 voiced by the 50-700 Hz pass, all 20 by the extended pass, 900 Hz in 8-19: two of
    the ten count. Over the extended pass's own frames it would be 12 in 20."""
    def pyin(_x, _sr, fmax):
        if fmax == rq.F0_RANGE[1]:
            return np.array([200.0] * 10 + [np.nan] * 10), np.arange(20) < 10
        return np.array([200.0] * 8 + [900.0] * 12), np.ones(20, bool)
    monkeypatch.setattr(rq, "_pyin", pyin)
    got = rq.f0_profile(_stack(200.0), SR)
    assert got["voiced_frames"] == 10 and got["f0_above_range_frac"] == pytest.approx(0.2), got
    assert got["f0_out_of_range"], got


def test_the_two_passes_must_share_a_frame_grid(monkeypatch):
    def pyin(_x, _sr, fmax):
        n = 20 if fmax == rq.F0_RANGE[1] else 21
        return np.full(n, 200.0), np.ones(n, bool)
    monkeypatch.setattr(rq, "_pyin", pyin)
    with pytest.raises(ValueError, match="frame grids"):
        rq.f0_profile(_stack(200.0), SR)


@pytest.mark.skipif(not all(p.exists() for p in REAL_RANGE_CLIPS.values()),
                    reason="the real expressive clips are not on this disk")
def test_real_clips_above_range_only_where_the_voice_is():
    """A 246 Hz clip that opens on ~985 Hz noise only the extended pass voices read 0.22 above range
    and was flagged; a 516 Hz clip whose high stretches both passes voice is still flagged."""
    import soundfile as sf
    got = {}
    for name, path in REAL_RANGE_CLIPS.items():
        x, sr = sf.read(str(path), dtype="float32")
        got[name] = rq.f0_profile(x.mean(axis=1) if x.ndim > 1 else x, sr)
    grief, victory = got["grief"], got["victory"]
    assert abs(_st(grief["f0_median"], 246.0)) < 1.0, grief
    assert grief["f0_above_range_frac"] < 0.05 and not grief["f0_out_of_range"], grief
    assert abs(_st(victory["f0_median"], 516.0)) < 1.0, victory
    assert victory["f0_above_range_frac"] > 0.2 and victory["f0_out_of_range"], victory


@pytest.mark.parametrize("f0", [40.0, 45.0])
def test_a_voice_below_the_floor_reads_at_the_floor_and_is_flagged(f0):
    got = rq.f0_profile(_stack(f0), SR)
    assert got["f0_edge_frac"] > 0.9 and got["f0_above_range_frac"] == 0, got
    assert got["f0_out_of_range"], got


def test_a_dominant_second_harmonic_misread_is_visible():
    """pYIN's octave error: H2 16 dB over H1 reads 220 Hz as ~441 Hz (positive control). The
    autocorrelation reads 220 Hz, so the row says the methods disagree."""
    got = rq.f0_profile(_stack(220.0, amps=[0.15, 1.0, 0.1, 0.5, 0.05, 0.2]), SR)
    assert abs(got["f0_median"] - 440.0) / 440.0 < 0.03, got
    assert abs(got["f0_median_acf"] - 220.0) / 220.0 < 0.03, got
    assert got["f0_methods_disagree"] is True, got


def test_an_octave_error_both_methods_make_is_not_flagged():
    """The audit's limit (module docstring): at 110 Hz with H2 14 dB over H1 pYIN and the
    autocorrelation both read about 220 Hz, so the row does not say they disagree."""
    amps = [1.0, 10 ** (14 / 20)] + [1.0 / k ** 2 for k in range(3, 40)]
    got = rq.f0_profile(_stack(110.0, amps=amps), SR)
    assert abs(got["f0_median"] - 220.0) / 220.0 < 0.03, got
    assert abs(got["f0_median_acf"] - 220.0) / 220.0 < 0.03, got
    assert got["f0_methods_disagree"] is False and not got["f0_out_of_range"], got


def test_the_methods_disagree_beyond_one_semitone_only(monkeypatch):
    """`f0_methods_disagree` is |pYIN - ACF| > METHODS_DISAGREE_ST semitones, either sign; None
    when the autocorrelation finds no voiced frame."""
    import periodicity
    x = _stack(200.0)
    for acf_hz, want in ((200.0 * 2 ** (0.9 / 12), False), (200.0 * 2 ** (-0.9 / 12), False),
                         (200.0 * 2 ** (1.1 / 12), True), (200.0 * 2 ** (-1.1 / 12), True)):
        monkeypatch.setattr(periodicity, "frames",
                            lambda *a, v=acf_hz: (np.full(10, v), np.zeros(10)))
        got = rq.f0_profile(x, SR)
        assert abs(got["f0_median"] - 200.0) < 2.0 and got["f0_methods_disagree"] is want, (acf_hz, got)
    monkeypatch.setattr(periodicity, "frames", lambda *a: (np.zeros(0), np.zeros(0)))
    got = rq.f0_profile(x, SR)
    assert got["f0_median_acf"] is None and got["f0_methods_disagree"] is None and got["hnr"] is None


def test_out_of_range_needs_more_than_a_tenth_of_the_frames(monkeypatch):
    """OUT_OF_RANGE_FRAC holds for either fraction: 2 edge (or above-range) frames in 20 is not out
    of range, 3 in 20 is. Both limits count as an edge, within EDGE_TOL."""
    def fake(frac_edge, frac_above, edge_hz):
        def pyin(_x, _sr, fmax):
            n = 20
            if fmax == rq.F0_RANGE[1]:
                k = round(frac_edge * n)
                return np.array([edge_hz] * k + [200.0] * (n - k)), np.ones(n, bool)
            k = round(frac_above * n)
            return np.array([900.0] * k + [200.0] * (n - k)), np.ones(n, bool)
        return pyin
    x = _stack(200.0)
    lo, hi = rq.F0_RANGE
    for edge_hz in (lo * 1.025, hi / 1.025):
        for fe, fa, want in ((0.10, 0.0, False), (0.15, 0.0, True), (0.0, 0.10, False), (0.0, 0.15, True)):
            monkeypatch.setattr(rq, "_pyin", fake(fe, fa, edge_hz))
            got = rq.f0_profile(x, SR)
            assert got["f0_edge_frac"] == pytest.approx(fe) and got["f0_above_range_frac"] == pytest.approx(fa)
            assert got["f0_out_of_range"] is want, (edge_hz, fe, fa, got)
    for clear_hz in (lo * 1.035, hi / 1.035):
        monkeypatch.setattr(rq, "_pyin", fake(0.5, 0.0, clear_hz))
        assert rq.f0_profile(x, SR)["f0_edge_frac"] == 0, clear_hz


def test_pyin_frame_holds_two_periods_of_the_floor():
    assert rq.pyin_frame(24000) == 1024
    for sr in (16000, 22050, 24000, 44100, 48000):
        n = rq.pyin_frame(sr)
        assert n & (n - 1) == 0 and sr / rq.F0_RANGE[0] < n // 2 and not sr / rq.F0_RANGE[0] < n // 4


def test_hnr_rises_with_f0_for_a_perfectly_periodic_voice():
    """Why each row carries `hnr_note`: the same perfect periodicity reads >5 dB apart."""
    low, high = rq.f0_profile(_stack(70.0), SR), rq.f0_profile(_stack(330.0), SR)
    assert high["hnr"] - low["hnr"] > 5.0, (low["hnr"], high["hnr"])


def test_f0_profile_is_none_on_silence():
    assert rq.f0_profile(np.zeros(24000), 24000) is None


def test_f0_keys_are_the_profiles_keys():
    """A clip with no voiced frame gets a row of nulls under exactly the keys a voiced clip gets."""
    assert tuple(rq.f0_profile(_stack(200.0), SR)) == rq.F0_KEYS


# --- phonation ------------------------------------------------------------------------------

def _f0_row(p10, p90, out_of_range=False):
    return {"f0_p10": p10, "f0_p90": p90, "f0_out_of_range": out_of_range}


@pytest.mark.parametrize("p10, p90, null", [
    (100.0, 300.0, False), (59.0, 200.0, True), (100.0, 401.0, True), (60.0, 400.0, False)])
def test_phonation_is_null_outside_its_cepstral_band_at_either_end(p10, p90, null):
    import derive_vat_corpus as dvc
    assert (dvc.F0_MIN, dvc.F0_MAX) == (60.0, 400.0)
    row = rq.phonation_row(_stack(110.0), SR, _f0_row(p10, p90))
    if null:
        assert row["cpp"] is None and row["h1h2"] is None and row["alpha_db"] is None, row
        assert "leaves phonation_measures' 60-400 Hz cepstral band" in row["phonation_null_reason"], row
    else:
        assert row["cpp"] is not None and row["phonation_null_reason"] is None, row


def test_phonation_is_null_when_f0_is_out_of_range():
    """A 720 Hz voice reads p10-p90 inside the cepstral band (at half its F0); its phonation would
    measure the wrong peak."""
    f0 = rq.f0_profile(_stack(720.0), SR)
    assert 60.0 < f0["f0_p10"] and f0["f0_p90"] < 400.0 and f0["f0_out_of_range"], f0
    row = rq.phonation_row(_stack(720.0), SR, f0)
    assert row["cpp"] is None and "F0 not measured" in row["phonation_null_reason"], row


# --- render's imports -----------------------------------------------------------------------

LIB = REPO / "scripts" / "lib"
HEAVY = ("matcha", "torch", "hydra", "lightning", "monotonic_align")


def _heavy_after(code):
    """Modules from HEAVY in sys.modules after `code` runs in a fresh interpreter."""
    probe = ("import sys; sys.path[:0] = [%r, %r]\n%s\n"
             "print(sorted(m for m in %r if m in sys.modules))" % (str(REPO), str(LIB), code, HEAVY))
    got = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout.strip().replace("'", '"'))


def test_render_imports_only_dependency_light_lib_modules():
    tree = ast.parse((REPO / "scripts/tools/render_qwen_1c0.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cmd_render")
    names = {a.name.split(".")[0] for n in ast.walk(fn) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(fn) if isinstance(n, ast.ImportFrom)}
    assert {m for m in names if (LIB / (m + ".py")).exists()} == \
        {"loudness", "qwen_load", "synth_common"}
    assert _heavy_after("import loudness, qwen_load, synth_common") == []
    assert _heavy_after("import matcha.delivery") == ["matcha"]     # the probe can see one


def test_ear_bench_re_exports_the_one_match_loudness():
    """Both import spellings of ear_bench hand back loudness.match_loudness itself. Its torch and
    Matcha imports are stubbed: this checks the re-export, not the bench."""
    stubs = ("import types\n"
             "for name, attrs in {'torch': {'device': lambda *a: None},"
             " 'matcha.cli': dict.fromkeys(['detect_lane', 'load_matcha', 'load_vocoder_24k',"
             " 'process_text_for_lane', 'to_waveform']), 'matcha.text': {'text_to_sequence': None},"
             " 'matcha.utils': {}, 'matcha.utils.utils': {'intersperse': None}}.items():\n"
             "    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m\n"
             "sys.path.insert(0, %r)\n"
             "import ear_bench, lib.ear_bench, loudness, lib.loudness\n"
             "assert ear_bench.match_loudness is loudness.match_loudness\n"
             "assert lib.ear_bench.match_loudness is lib.loudness.match_loudness\n"
             "assert 'def match_loudness' not in open(ear_bench.__file__).read()\n"
             % str(REPO / "scripts"))
    _heavy_after(stubs)


def test_an_import_error_inside_loudness_surfaces_from_ear_bench():
    """The package spelling imports `.loudness` and nothing else: a broken `lib.loudness` raises
    its own ImportError rather than falling back to the flat module and hiding it."""
    stubs = ("import types\n"
             "for name, attrs in {'torch': {'device': lambda *a: None},"
             " 'matcha.cli': dict.fromkeys(['detect_lane', 'load_matcha', 'load_vocoder_24k',"
             " 'process_text_for_lane', 'to_waveform']), 'matcha.text': {'text_to_sequence': None},"
             " 'matcha.utils': {}, 'matcha.utils.utils': {'intersperse': None}}.items():\n"
             "    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m\n"
             "sys.path.insert(0, %r)\n"
             "sys.modules['lib.loudness'] = None\n"
             "try:\n"
             "    import lib.ear_bench\n"
             "except ImportError as e:\n"
             "    assert 'lib.loudness' in str(e), e\n"
             "else:\n"
             "    raise AssertionError('lib.ear_bench imported around a broken lib.loudness')\n"
             % str(REPO / "scripts"))
    _heavy_after(stubs)


# --- render resume --------------------------------------------------------------------------

def _model(tmp_path):
    d = tmp_path / "model"
    (d / "speech_tokenizer" / "sub").mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"weights")
    (d / "speech_tokenizer" / "config.json").write_text("{}")
    (d / "speech_tokenizer" / "sub" / "model.bin").write_bytes(b"tok")
    return d


def _meta(commit, weights, probe="1c0", ids=None):
    """The RESUME_KEYS of a render_meta.json."""
    return {"commit": commit, "weights_sha256": weights, "qwen_tts": "0.1.1",
            "speech_tokenizer_dtype": "float32", "probe": probe, "ids": ids}


def _rendered(out, commit, weights, items=None, wav=lambda it: b"RIFF", probe="1c0", ids=None):
    """An --out a render left: meta, a manifest row per item and its (placeholder) wav. `ids` is the
    meta's id subset; without `items` the rows are its items (the whole plan when None)."""
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "render_meta.json").write_text(json.dumps(_meta(commit, weights, probe, ids)))
    rows = []
    for it in items if items is not None else rq.select_ids(rq.plan_items(probe), ids):
        (out / "clips" / (it["id"] + ".wav")).write_bytes(wav(it))
        rows.append(dict(it, file="clips/%s.wav" % it["id"], sr=SR, commit=commit,
                         speech_tokenizer_dtype="float32"))
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


@pytest.fixture
def render_env(tmp_path, monkeypatch):
    """cmd_render against a fake model dir. torch and qwen_tts are made unimportable, so a render
    that got as far as loading the model fails loudly instead of loading one."""
    model = _model(tmp_path)
    monkeypatch.setattr(rq, "MODEL_DIR", str(model))
    monkeypatch.setattr(rq, "require_qwen_tts", lambda _v: "0.1.1")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "qwen_tts", None)
    return tmp_path / "out", rq.weights_sha256(str(model)), model


def _render(out, commit="abc123", probe="1c0", ids=None):
    rq.cmd_render(argparse.Namespace(out=str(out), commit=commit, probe=probe, ids=ids))


def test_weights_hash_covers_every_speech_tokenizer_file(tmp_path):
    w = rq.weights_sha256(str(_model(tmp_path)))
    assert sorted(w) == ["model.safetensors", "speech_tokenizer/config.json",
                         "speech_tokenizer/sub/model.bin"]
    empty = tmp_path / "empty"
    (empty / "speech_tokenizer").mkdir(parents=True)
    (empty / "model.safetensors").write_bytes(b"w")
    with pytest.raises(SystemExit, match="REFUSING"):
        rq.weights_sha256(str(empty))


def test_a_fresh_render_writes_its_meta_before_loading_the_model(render_env):
    out, weights, _model_dir = render_env
    with pytest.raises(ImportError):
        _render(out)
    meta = json.loads((out / "render_meta.json").read_text())
    assert (meta["commit"], meta["weights_sha256"], meta["qwen_tts"]) == ("abc123", weights, "0.1.1")
    assert (meta["batch_size"], meta["items"], meta["lufs"]) == (1, 60, rq.LUFS)
    assert (meta["speech_tokenizer_dtype"], meta["probe"], meta["ids"]) == ("float32", "1c0", None)
    assert set(rq.RESUME_KEYS) <= set(meta)


def test_a_fresh_1c0b_render_records_its_probe(render_env):
    out, _weights, _model_dir = render_env
    with pytest.raises(ImportError):
        _render(out, probe="1c0b")
    meta = json.loads((out / "render_meta.json").read_text())
    assert (meta["probe"], meta["items"], meta["speech_tokenizer_dtype"]) == ("1c0b", 66, "float32")
    assert {"speech_tokenizer_dtype", "probe"} <= set(rq.RESUME_KEYS)


def test_render_loads_through_load_qwen_and_records_the_decoder_dtype_per_row(render_env,
                                                                            monkeypatch):
    """A fake load_qwen and torch: the model comes from qwen_load, and every manifest row records
    the decoder's dtype."""
    import types
    import qwen_load
    out, _weights, model_dir = render_env
    loaded = []

    class Model:
        def generate_voice_design(self, text, language, instruct):
            return [np.zeros(SR // 10, dtype=np.float32)], SR
    monkeypatch.setattr(qwen_load, "load_qwen", lambda d: loaded.append(d) or Model())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(manual_seed=lambda s: None))
    monkeypatch.setitem(sys.modules, "loudness", types.SimpleNamespace(
        match_loudness=lambda x, sr, lufs: (x, False)))
    _render(out, probe="1c0b")
    assert loaded == [str(model_dir)]
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [it["id"] for it in rq.plan_1c0b()]
    assert {r["speech_tokenizer_dtype"] for r in rows} == {"float32"}


def test_a_finished_render_exits_before_loading_the_model(render_env, capsys):
    out, weights, _model_dir = render_env
    _rendered(out, "abc123", weights)
    _render(out)
    assert "the model was not loaded" in capsys.readouterr().out


@pytest.mark.parametrize("change, match", [
    (lambda r: r.update(instruct=r["instruct"].replace("Calm", "Bright")), "instruct"),
    (lambda r: r.update(seed=r["seed"] + 1), "seed"),
    (lambda r: r.update(text="Another sentence."), "text"),
    (lambda r: r.update(commit="0ld"), "commit"),
    (lambda r: r.update(file="clips/M1-s1.wav"), "file"),     # another row's wav, which exists
    (lambda r: r.update(gender="Female"), "gender"),
    (lambda r: r.pop("gender"), "gender"),
])
def test_resume_refuses_a_done_row_that_is_not_its_ladder_item(render_env, change, match):
    out, weights, _model_dir = render_env
    rows = _rendered(out, "abc123", weights, items=rq.ladder()[:10])
    change(rows[4])
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="REFUSING: manifest row %s .*%s" % (rows[4]["id"], match)):
        _render(out)


def test_resume_refuses_a_missing_wav_or_a_repeated_id(render_env):
    out, weights, _model_dir = render_env
    rows = _rendered(out, "abc123", weights, items=rq.ladder()[:10])
    (out / "clips" / "M1-s2.wav").unlink()
    with pytest.raises(SystemExit, match="REFUSING: manifest row M1-s2 has no wav"):
        _render(out)
    _rendered(out, "abc123", weights, items=rq.ladder()[:10])
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + rows[:1]))
    with pytest.raises(SystemExit, match="REFUSING: manifest id 'M1-s1' is repeated"):
        _render(out)


@pytest.mark.parametrize("what", ["commit", "weights_sha256", "qwen_tts",
                                  "speech_tokenizer_dtype", "probe", "ids"])
def test_resume_refuses_another_renders_meta(render_env, what):
    out, weights, model = render_env
    _rendered(out, "abc123", weights, items=rq.ladder()[:10])
    commit, probe = "abc123", "1c0"
    meta = json.loads((out / "render_meta.json").read_text())
    if what == "commit":
        commit = "def456"
    elif what == "weights_sha256":
        (model / "speech_tokenizer" / "sub" / "model.bin").write_bytes(b"retrained")
    elif what == "qwen_tts":
        (out / "render_meta.json").write_text(json.dumps(dict(meta, qwen_tts="0.1.2")))
    elif what == "speech_tokenizer_dtype":
        # A folder rendered with the bf16 decoder records no decoder dtype (the 1c-0 probe).
        del meta["speech_tokenizer_dtype"]
        (out / "render_meta.json").write_text(json.dumps(meta))
    elif what == "ids":
        (out / "render_meta.json").write_text(json.dumps(
            dict(meta, ids=[it["id"] for it in rq.ladder()[:10]])))
    else:
        probe = "1c0b"
    with pytest.raises(SystemExit, match="REFUSING: resuming .*%s" % what):
        _render(out, commit=commit, probe=probe)


def test_resume_refuses_a_meta_rendered_before_the_decoder_dtype_was_recorded(render_env):
    """The 1c-0 probe's own render_meta.json keys: commit, weights, qwen-tts, nothing else."""
    out, weights, _model = render_env
    _rendered(out, "abc123", weights, items=rq.ladder()[:10])
    (out / "render_meta.json").write_text(json.dumps(
        {"commit": "abc123", "weights_sha256": weights, "qwen_tts": "0.1.1"}))
    with pytest.raises(SystemExit, match="REFUSING: resuming .*speech_tokenizer_dtype, probe"):
        _render(out)


def test_resume_drops_only_a_truncated_final_line(tmp_path, capsys):
    out = tmp_path / "out"
    rows = _rendered(out, "abc123", {"w": "x"}, items=rq.ladder()[:5])
    meta = _meta("abc123", {"w": "x"})
    whole = (out / "manifest.jsonl").read_text()
    (out / "manifest.jsonl").write_text(whole + json.dumps(rows[0])[:40])
    assert rq.resume_state(str(out), meta) == {r["id"] for r in rows}
    assert "truncated final manifest line" in capsys.readouterr().out
    assert (out / "manifest.jsonl").read_text() == whole
    (out / "manifest.jsonl").write_text(whole[:30] + "\n" + whole)
    with pytest.raises(SystemExit, match="REFUSING: .* line 1 is not JSON"):
        rq.resume_state(str(out), meta)


# --- measure --------------------------------------------------------------------------------

def _measure(out, probe="1c0", ids=None):
    rq.cmd_measure(argparse.Namespace(out=str(out), probe=probe, ids=ids))


def _clip_f0(it):
    return 520.0 if it["id"].startswith("P1") else 210.0 if it["id"].startswith("F") else 110.0


def _measure_dir(tmp_path, out=None, f0_of=_clip_f0, probe="1c0", seconds=1.0, ids=None):
    import soundfile as sf
    out = out or tmp_path / "probe"
    _rendered(out, "abc123", {"w": "x"}, probe=probe, ids=ids)
    for it in rq.select_ids(rq.plan_items(probe), ids):
        sf.write(str(out / "clips" / (it["id"] + ".wav")),
                 _stack(f0_of(it), seconds).astype("float32"), SR)
    return out


def test_measure_writes_a_row_per_ladder_clip(tmp_path, capsys):
    out = _measure_dir(tmp_path)
    _measure(out)
    rows = [json.loads(line) for line in (out / "measures.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [it["id"] for it in rq.ladder()]
    by_id = {r["id"]: r for r in rows}
    for it in rq.ladder():
        r, f0 = by_id[it["id"]], _clip_f0(it)
        assert abs(r["f0_median"] - f0) / f0 < 0.03, r
        assert r["hnr_note"] == rq.HNR_NOTE
        assert not r["f0_out_of_range"] and r["f0_methods_disagree"] is (f0 > 350), r
        if f0 > 400:
            assert r["cpp"] is None and r["h1h2"] is None, r
            assert "cepstral band" in r["phonation_null_reason"], r
        else:
            assert r["cpp"] is not None and r["phonation_null_reason"] is None, r
        assert r["phonation_f0_band_hz"] == [60.0, 400.0]
        for alias, key in rq.READER_ALIASES:
            assert r[alias] == r[key], (alias, r)
    assert ("60 clips measured; 0 without a voiced frame; 3 without a phonation measure; 0 with F0 "
            "out of pYIN's 50-700 Hz range; 3 where pYIN and the autocorrelation disagree") \
        in capsys.readouterr().out


def test_measure_refuses_a_missing_manifest_a_wrong_id_set_or_a_missing_wav(tmp_path):
    with pytest.raises(SystemExit, match="REFUSING: no manifest"):
        _measure(tmp_path / "nothing")
    out = _measure_dir(tmp_path)
    manifest = out / "manifest.jsonl"
    lines = manifest.read_text().splitlines(keepends=True)
    manifest.write_text("".join(lines[:-1]))
    with pytest.raises(SystemExit, match=r"REFUSING: 59 of 60 .*P6-s3"):
        _measure(out)
    manifest.write_text("".join(lines[:-1] + lines[:1]))
    with pytest.raises(SystemExit, match="REFUSING: manifest id 'M1-s1' is repeated"):
        _measure(out)
    stranger = dict(json.loads(lines[-1]), id="Q9-s1", file="clips/Q9-s1.wav")
    manifest.write_text("".join(lines[:-1]) + json.dumps(stranger) + "\n")
    with pytest.raises(SystemExit, match="REFUSING: manifest id 'Q9-s1' is not a plan id"):
        _measure(out)
    manifest.write_text("".join(lines))
    (out / "clips" / "F3-s2.wav").unlink()
    with pytest.raises(SystemExit, match="REFUSING: manifest row F3-s2 has no wav"):
        _measure(out)


def test_measure_refuses_a_truncated_manifest(tmp_path):
    out = _measure_dir(tmp_path)
    manifest = out / "manifest.jsonl"
    text = manifest.read_text()
    manifest.write_text(text[:-40])                     # the last row loses its tail and newline
    with pytest.raises(SystemExit, match="REFUSING: .* ends in a truncated line"):
        _measure(out)


def test_measure_refuses_a_missing_meta_or_rows_from_another_commit(tmp_path):
    out = _measure_dir(tmp_path)
    meta = out / "render_meta.json"
    text = meta.read_text()
    meta.unlink()
    with pytest.raises(SystemExit, match="REFUSING: .* no readable render_meta.json commit"):
        _measure(out)
    meta.write_text(json.dumps({"qwen_tts": "0.1.1"}))
    with pytest.raises(SystemExit, match="REFUSING: .* no readable render_meta.json commit"):
        _measure(out)
    meta.write_text(json.dumps(dict(json.loads(text), commit="def456")))
    with pytest.raises(SystemExit, match="REFUSING: manifest row M1-s1 .*commit"):
        _measure(out)


def test_an_out_of_range_row_nulls_its_f0_statistics():
    """The profile keeps what the 50-700 Hz pass read (360 Hz for a 720 Hz voice); the row does not,
    so a band derived from rows cannot take it in. The flag, the fractions and HNR stay."""
    x = _stack(720.0)
    f0 = rq.f0_profile(x, SR)
    assert abs(f0["f0_median"] - 360.0) / 360.0 < 0.03 and f0["f0_out_of_range"], f0
    row = rq.measure_row("P1-s1", x, SR)
    assert all(row[k] is None for k in rq.F0_STATS), row
    assert row["f0_median_hz"] is None and row["f0_excursion_hz"] is None and row["cpp_db"] is None
    assert row["f0_out_of_range"] is True and row["f0_above_range_frac"] > 0.9, row
    assert row["voiced_frames"] == f0["voiced_frames"] and row["hnr_db"] == row["hnr"] == f0["hnr"]
    assert "F0 not measured" in row["phonation_null_reason"], row
    inside = rq.measure_row("M4-s1", _stack(110.0), SR)
    assert inside["f0_median_hz"] == inside["f0_median"] and abs(inside["f0_median"] - 110.0) < 3.0
    assert inside["cpp"] is not None and inside["cpp_db"] == inside["cpp"], inside
    silent = rq.measure_row("M4-s2", np.zeros(SR), SR)
    assert silent["voiced_frames"] is None and silent["f0_median_hz"] is None, silent


def test_measure_writes_its_meta_beside_the_measures(tmp_path):
    import librosa
    out = _measure_dir(tmp_path)
    _measure(out)
    meta = json.loads((out / "measure_meta.json").read_text())
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                            text=True, check=True).stdout
    assert (meta["commit"], meta["tree_clean"]) == (head, not status.strip()), meta
    assert (meta["render_commit"], meta["tool"]) == ("abc123", rq.TOOL)
    assert (meta["librosa"], meta["numpy"]) == (librosa.__version__, np.__version__)
    assert meta["f0_range_hz"] == [50.0, 700.0] and meta["f0_extended_fmax_hz"] == 1400.0
    assert meta["pyin_frame_samples"] == {"24000": 1024} and meta["pyin_hop_s"] == 0.010
    assert (meta["edge_tol"], meta["out_of_range_frac"], meta["methods_disagree_st"]) == \
        (rq.EDGE_TOL, rq.OUT_OF_RANGE_FRAC, rq.METHODS_DISAGREE_ST)
    assert (meta["acf_band_hz"], meta["acf_rms_floor"], meta["acf_periodicity"]) == \
        ([50.0, 350.0], rq.RMS_FLOOR, rq.PERIODICITY)
    assert meta["phonation_f0_band_hz"] == [60.0, 400.0]
    assert meta["measured_utc"].endswith("+00:00"), meta


def test_measure_meta_says_when_the_commit_is_unknown(monkeypatch):
    import subprocess as sp

    def no_git(*_a, **_k):
        raise OSError("no git")
    monkeypatch.setattr(sp, "run", no_git)
    meta = rq.measure_meta("abc123", {24000})
    assert meta["commit"] is None and meta["tree_clean"] is None, meta


# --- the audition app's probe mode ----------------------------------------------------------

def _audition(tmp_path, monkeypatch):
    """audition/app/main.py with its data root at `tmp_path` (as tests/test_listening_sliders.py)."""
    monkeypatch.setenv("AUDITION_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDITION_RATINGS_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("audition_main_1c0", REPO / "audition/app/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_audition_probe_mode_reads_a_measured_probe_folder(tmp_path, monkeypatch):
    """Render's manifest and measure's rows, loaded by the probe mode itself: every clip with its
    gender, wav and the measures under the names its card shows; an out-of-range clip shows none."""
    out = _measure_dir(tmp_path, tmp_path / "probes" / "qwen-1c0",
                       f0_of=lambda it: 720.0 if it["id"] == "P2-s1" else _clip_f0(it))
    _measure(out)
    measures = {r["id"]: r for r in map(json.loads, (out / "measures.jsonl").read_text().splitlines())}
    app = _audition(tmp_path, monkeypatch)
    assert app.list_probes()["probes"] == ["qwen-1c0"]
    clips = app.get_probe("qwen-1c0")["clips"]
    assert [c["id"] for c in clips] == [it["id"] for it in rq.ladder()]
    personas = {p["id"]: p["gender"] for p in rq.PERSONAS}
    for c in clips:
        want = {"M": "Male", "F": "Female"}.get(c["id"][0]) or personas[c["id"].split("-")[0]]
        assert c["gender"] == want, c
        assert c["wav"] == "probes/qwen-1c0/clips/%s.wav" % c["id"], c
        assert c["acoustics"] == measures[c["id"]], c
        for key in ("f0_median_hz", "f0_excursion_hz", "hnr_db", "cpp_db"):
            assert key in c["acoustics"], (key, c)
    by_id = {c["id"]: c for c in clips}
    assert abs(by_id["M4-s1"]["acoustics"]["f0_median_hz"] - 110.0) < 3.0
    assert by_id["P2-s1"]["acoustics"]["f0_out_of_range"] is True
    assert by_id["P2-s1"]["acoustics"]["f0_median_hz"] is None
    assert by_id["P2-s1"]["acoustics"]["f0_excursion_hz"] is None
    # The probe mode rates the clip into the folder's verdicts.csv, nowhere else.
    res = app.rate_probe("qwen-1c0", app.ProbeRating(id="P2-s1", zone="exclude"))
    assert res["row"]["zone"] == "exclude" and (out / "verdicts.csv").is_file()


# --- --probe threads through check_rows and measure -----------------------------------------

def test_check_rows_refuses_the_other_probes_rows(tmp_path):
    out = tmp_path / "out"
    rows_b = _rendered(out / "b", "abc123", {"w": "x"}, probe="1c0b")
    rows_0 = _rendered(out / "0", "abc123", {"w": "x"})
    rq.check_rows(rows_b, str(out / "b"), rq.plan_1c0b(), commit="abc123")
    rq.check_rows(rows_0, str(out / "0"), rq.ladder(), commit="abc123")
    with pytest.raises(SystemExit, match="REFUSING: manifest id 'M-base-k1' is not a plan id"):
        rq.check_rows(rows_b, str(out / "b"), rq.ladder())
    with pytest.raises(SystemExit, match="REFUSING: manifest id 'M1-s1' is not a plan id"):
        rq.check_rows(rows_0, str(out / "0"), rq.plan_1c0b())


def test_measure_refuses_a_folder_of_the_other_probe(tmp_path):
    b = tmp_path / "b"
    _rendered(b, "abc123", {"w": "x"}, probe="1c0b")
    with pytest.raises(SystemExit, match="REFUSING: .* records probe 1c0b, not 1c0"):
        _measure(b)
    meta = json.loads((b / "render_meta.json").read_text())
    del meta["probe"]                                # no probe recorded: the rows still refuse
    (b / "render_meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="REFUSING: manifest id 'M-base-k1' is not a plan id"):
        _measure(b)
    zero = tmp_path / "0"
    _rendered(zero, "abc123", {"w": "x"})
    with pytest.raises(SystemExit, match="REFUSING: .* records probe 1c0, not 1c0b"):
        _measure(zero, probe="1c0b")


def test_measure_reads_a_1c0b_folder_in_its_plan_order(tmp_path, capsys):
    out = _measure_dir(tmp_path, probe="1c0b", seconds=0.5)
    _measure(out, probe="1c0b")
    rows = [json.loads(line) for line in (out / "measures.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [it["id"] for it in rq.plan_1c0b()]
    assert all(abs(r["f0_median"] - _clip_f0({"id": r["id"]})) < 5.0 for r in rows)
    assert json.loads((out / "measure_meta.json").read_text())["probe"] == "1c0b"
    assert "66 clips measured" in capsys.readouterr().out


# --- --ids: part of a plan ------------------------------------------------------------------

ANCHORS = ["M4-s3", "F4-s3", "F6-s2"]


def test_select_ids_keeps_plan_order_and_refuses_an_unknown_or_repeated_id():
    plan = rq.ladder()
    assert rq.select_ids(plan, None) is plan
    assert [it["id"] for it in rq.select_ids(plan, ["F6-s2", "M4-s3", "F4-s3"])] == ANCHORS
    assert rq.select_ids(plan, ["M4-s3"]) == [next(it for it in plan if it["id"] == "M4-s3")]
    assert rq.subset_ids(plan, ["F6-s2", "M4-s3", "F4-s3"]) == ANCHORS
    assert rq.subset_ids(plan, None) is None
    with pytest.raises(SystemExit, match="REFUSING: --ids Q9-s9 is not in the plan"):
        rq.select_ids(plan, ["M4-s3", "Q9-s9"])
    with pytest.raises(SystemExit, match="REFUSING: --ids M-base-k1, Q9 are not in the plan"):
        rq.select_ids(plan, ["M-base-k1", "Q9"])
    with pytest.raises(SystemExit, match="REFUSING: --ids names M4-s3 more than once"):
        rq.select_ids(plan, ["M4-s3", "F4-s3", "M4-s3"])


def test_render_refuses_an_unknown_id_before_writing_anything(render_env):
    out, _weights, _model_dir = render_env
    with pytest.raises(SystemExit, match="REFUSING: --ids Q9-s9 is not in the plan"):
        rq.main(["render", "--out", str(out), "--commit", "abc123", "--ids", "M4-s3", "Q9-s9"])
    with pytest.raises(SystemExit, match="REFUSING: --ids M4-s3 is not in the plan"):
        rq.main(["render", "--out", str(out), "--commit", "abc123", "--probe", "1c0b",
                 "--ids", "M4-s3"])
    assert not out.exists()


def test_a_subset_render_renders_only_its_ids_in_plan_order(render_env, monkeypatch, capsys):
    """A fake load_qwen and torch, as above: the anchors from 1c-0's plan, with their registered
    seeds and instructs; render_meta.json records the subset."""
    import types
    import qwen_load
    out, _weights, _model_dir = render_env
    seeds = []

    class Model:
        def generate_voice_design(self, text, language, instruct):
            return [np.zeros(SR // 10, dtype=np.float32)], SR
    monkeypatch.setattr(qwen_load, "load_qwen", lambda d: Model())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(manual_seed=seeds.append))
    monkeypatch.setitem(sys.modules, "loudness", types.SimpleNamespace(
        match_loudness=lambda x, sr, lufs: (x, False)))
    rq.main(["render", "--out", str(out), "--commit", "abc123", "--ids", "F6-s2", "M4-s3", "F4-s3"])
    meta = json.loads((out / "render_meta.json").read_text())
    assert (meta["ids"], meta["items"], meta["probe"]) == (ANCHORS, 3, "1c0")
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    plan = {it["id"]: it for it in rq.ladder()}
    assert [r["id"] for r in rows] == ANCHORS
    assert seeds == [plan[i]["seed"] for i in ANCHORS] == [9143, 9243, 9262]
    for r in rows:
        assert {k: r[k] for k in ("seed", "text", "instruct", "gender")} == \
            {k: plan[r["id"]][k] for k in ("seed", "text", "instruct", "gender")}
    assert sorted(p.name for p in (out / "clips").iterdir()) == sorted(i + ".wav" for i in ANCHORS)
    # The same subset, named in another order, is the same render: nothing is left to do.
    rq.main(["render", "--out", str(out), "--commit", "abc123", "--ids", "M4-s3", "F6-s2", "F4-s3"])
    assert "all 3 clips already rendered; the model was not loaded." in capsys.readouterr().out


@pytest.mark.parametrize("folder, run", [
    (None, ANCHORS),                    # a whole-plan folder, resumed as a subset
    (ANCHORS, None),                    # a subset folder, resumed as the whole plan
    (ANCHORS, ["M4-s3", "F4-s3"]),      # a subset folder, resumed as another subset
])
def test_a_resume_refuses_another_id_subset(render_env, folder, run):
    out, weights, _model_dir = render_env
    plan = rq.ladder()
    _rendered(out, "abc123", weights, items=rq.select_ids(plan, folder)[:2], ids=folder)
    with pytest.raises(SystemExit, match="REFUSING: resuming .*this run's ids differ"):
        _render(out, ids=run)


def test_a_subset_resume_refuses_a_row_outside_the_subset(render_env):
    out, weights, _model_dir = render_env
    plan = {it["id"]: it for it in rq.ladder()}
    _rendered(out, "abc123", weights, items=[plan["M4-s3"], plan["M1-s1"]], ids=ANCHORS)
    with pytest.raises(SystemExit, match="REFUSING: manifest id 'M1-s1' is not a plan id"):
        _render(out, ids=ANCHORS)


def test_measure_reads_a_subset_folder(tmp_path, capsys):
    out = _measure_dir(tmp_path, ids=ANCHORS, seconds=0.5)
    _measure(out, ids=["F6-s2", "F4-s3", "M4-s3"])
    rows = [json.loads(line) for line in (out / "measures.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ANCHORS
    assert all(abs(r["f0_median"] - _clip_f0({"id": r["id"]})) < 5.0 for r in rows)
    meta = json.loads((out / "measure_meta.json").read_text())
    assert (meta["ids"], meta["probe"], meta["render_commit"]) == (ANCHORS, "1c0", "abc123")
    assert "3 clips measured" in capsys.readouterr().out


def test_measure_refuses_another_subset_than_the_render_and_misses_only_within_it(tmp_path):
    out = _measure_dir(tmp_path, ids=ANCHORS, seconds=0.5)
    with pytest.raises(SystemExit, match="REFUSING: .*records --ids M4-s3 F4-s3 F6-s2, not the "
                                         "whole plan; measure the ids it rendered"):
        _measure(out)
    with pytest.raises(SystemExit, match="REFUSING: .*records --ids M4-s3 F4-s3 F6-s2, not --ids "
                                         "M4-s3 F4-s3;"):
        _measure(out, ids=["M4-s3", "F4-s3"])
    with pytest.raises(SystemExit, match="REFUSING: --ids Q9-s9 is not in the plan"):
        _measure(out, ids=ANCHORS + ["Q9-s9"])
    lines = (out / "manifest.jsonl").read_text().splitlines(keepends=True)
    (out / "manifest.jsonl").write_text("".join(lines[:2]))
    with pytest.raises(SystemExit, match=r"REFUSING: 2 of 3 1c0 clips rendered \(missing \['F6-s2'\]\)"):
        _measure(out, ids=ANCHORS)
    assert not (out / "measures.jsonl").exists()
    whole = _measure_dir(tmp_path / "whole", seconds=0.5)
    with pytest.raises(SystemExit, match="REFUSING: .*records the whole plan, not --ids M4-s3 "
                                         "F4-s3 F6-s2"):
        _measure(whole, ids=ANCHORS)


def test_steer_refuses_a_subset_of_1c0b(tmp_path):
    out = tmp_path / "b"
    _rendered(out, "abc123", {"w": "x"}, probe="1c0b", ids=["M-base-k1", "F-base-k1"])
    with pytest.raises(SystemExit, match="REFUSING: .*records --ids M-base-k1 F-base-k1, not the "
                                         "whole plan"):
        rq.main(["steer", "--out", str(out)])


# --- 1c-0b's steer reading ------------------------------------------------------------------

def _semis(hz, st):
    return hz * 2 ** (st / 12.0)


def _gmean(hz):
    return float(np.exp(np.mean(np.log(hz))))


# Per gender and method: (low-pole F0 medians, high-pole F0 medians) by seed.
STEER_F0 = {
    "Male": {"baseline": [100.0, 102.0, 98.0],
             "numeric pitch": ([80.0, 82.0, 85.0], [160.0, 150.0, 170.0]),       # steers
             "descriptive pitch": ([100.0, 110.0, 105.0], [104.0, 120.0, 100.0]),  # overlaps
             "age": ([100.0, 101.0, 102.0], [110.0, 112.0, 111.0]),             # separated, small
             "vocal weight": ([80.0, 80.0, 80.0], [200.0, None, 210.0]),        # a null F0
             "free-text instruct": ([100.0] * 3, [_semis(100.0, 4.01)] * 3)},   # just over 4 st
    "Female": {"baseline": [None, 180.0, 170.0],
               "numeric pitch": ([150.0] * 3, [_semis(150.0, 12.0)] * 3),
               "descriptive pitch": ([200.0] * 3, [_semis(200.0, 3.99)] * 3),   # just under 4 st
               "age": ([220.0, 230.0, 240.0], [200.0, 190.0, 180.0]),           # reversed
               "vocal weight": ([180.0] * 3, [181.0] * 3),
               "free-text instruct": ([None] * 3, [None] * 3)},
}


def _steer_measures():
    out = {}
    for it in rq.plan_1c0b():
        f0 = STEER_F0[it["gender"]][it["method"]]
        k = int(it["id"].rsplit("-k", 1)[1]) - 1
        value = f0[k] if it["pole"] is None else f0[0 if it["pole"] == "low" else 1][k]
        out[it["id"]] = {"id": it["id"], "f0_median": value, "f0_out_of_range": value is None}
    return out


def test_the_steer_reading_is_the_spec_rule():
    res = rq.steer_readout(rq.plan_1c0b(), _steer_measures())
    got = {(m["gender"], m["method"]): m for m in res["methods"]}
    assert len(got) == 10
    steers = {key: m["steers"] for key, m in got.items()}
    assert steers == {("Male", "numeric pitch"): True, ("Male", "descriptive pitch"): False,
                      ("Male", "age"): False, ("Male", "vocal weight"): False,
                      ("Male", "free-text instruct"): True, ("Female", "numeric pitch"): True,
                      ("Female", "descriptive pitch"): False, ("Female", "age"): False,
                      ("Female", "vocal weight"): False, ("Female", "free-text instruct"): False}
    num = got[("Male", "numeric pitch")]
    assert num["low_f0_median_hz"] == [80.0, 82.0, 85.0] and num["high_f0_median_hz"] == [160.0, 150.0, 170.0]
    assert num["low_ids"] == ["M-num-lo-k1", "M-num-lo-k2", "M-num-lo-k3"]
    assert num["mean_shift_st"] == pytest.approx(_st(_gmean([160, 150, 170]), _gmean([80, 82, 85])))
    assert num["complete_separation"] is True and num["unmeasured"] == []
    assert got[("Male", "descriptive pitch")]["complete_separation"] is False
    small = got[("Male", "age")]
    assert small["complete_separation"] is True and 1.0 < small["mean_shift_st"] < 2.0
    weight = got[("Male", "vocal weight")]
    assert weight["complete_separation"] is False and weight["unmeasured"] == ["M-weight-hi-k2"]
    assert weight["mean_shift_st"] == pytest.approx(_st(_gmean([200, 210]), 80))
    assert got[("Male", "free-text instruct")]["mean_shift_st"] == pytest.approx(4.01)
    assert got[("Female", "descriptive pitch")]["complete_separation"] is True
    assert got[("Female", "descriptive pitch")]["mean_shift_st"] == pytest.approx(3.99)
    rev = got[("Female", "age")]
    assert rev["complete_separation"] is False and rev["mean_shift_st"] < -1.0
    none = got[("Female", "free-text instruct")]
    assert none["mean_shift_st"] is None and none["steers"] is False and len(none["unmeasured"]) == 6
    assert res["baseline"]["Male"]["median_hz"] == 100.0
    assert res["baseline"]["Female"] == {"ids": ["F-base-k1", "F-base-k2", "F-base-k3"],
                                         "f0_median_hz": [None, 180.0, 170.0], "median_hz": 175.0,
                                         "unmeasured": ["F-base-k1"]}
    assert all(m["pole_cosine"] is None for m in res["methods"])


def _one_method(lo, hi):
    """steer_readout's Male numeric-pitch row when its low and high poles read `lo` and `hi` (Hz by
    seed, None for a clip whose F0 is not measured)."""
    measures = _steer_measures()
    for it in rq.plan_1c0b():
        if (it["gender"], it["method"]) == ("Male", "numeric pitch"):
            value = (lo if it["pole"] == "low" else hi)[int(it["id"].rsplit("-k", 1)[1]) - 1]
            measures[it["id"]] = {"id": it["id"], "f0_median": value, "f0_out_of_range": value is None}
    res = rq.steer_readout(rq.plan_1c0b(), measures)
    return next(m for m in res["methods"] if (m["gender"], m["method"]) == ("Male", "numeric pitch"))


def test_the_mean_shift_takes_every_measured_clip_of_each_pole():
    """The poles' seeds are independent draws: seed k's low clip says nothing about seed k's high
    one, so a null high clip does not drop its seed's low clip. Pairing by seed would read
    mean(st(200/80), st(180/90)) = 13.93 st, discarding the 100 Hz low clip."""
    m = _one_method([80.0, 100.0, 90.0], [200.0, None, 180.0])
    assert m["unmeasured"] == ["M-num-hi-k2"] and m["complete_separation"] is False
    assert m["mean_shift_st"] == pytest.approx(_st(_gmean([200, 180]), _gmean([80, 100, 90])))
    assert m["mean_shift_st"] == pytest.approx(12.99, abs=0.01)
    paired = np.mean([_st(200, 80), _st(180, 90)])
    assert abs(m["mean_shift_st"] - paired) > 0.9
    # The semitones' reference cancels: every clip an octave up reads the same shift.
    up = _one_method([160.0, 200.0, 180.0], [400.0, None, 360.0])
    assert up["mean_shift_st"] == pytest.approx(m["mean_shift_st"])


def test_a_tie_between_the_poles_is_not_separated():
    tie = _one_method([100.0, 110.0, 120.0], [120.0, 300.0, 400.0])
    assert tie["mean_shift_st"] > 2 * rq.STEER_SHIFT_ST
    assert tie["complete_separation"] is False and tie["steers"] is False
    above = _one_method([100.0, 110.0, 120.0], [120.001, 300.0, 400.0])
    assert above["complete_separation"] is True and above["steers"] is True


def test_a_mean_shift_of_exactly_the_threshold_steers():
    assert rq.STEER_SHIFT_ST == 4.0
    exact = _one_method([100.0] * 3, [_semis(100.0, 4.0)] * 3)
    assert exact["mean_shift_st"] == 4.0 and exact["complete_separation"] is True
    assert exact["steers"] is True
    under = _one_method([100.0] * 3, [_semis(100.0, 3.99)] * 3)
    assert under["mean_shift_st"] == pytest.approx(3.99) and under["complete_separation"] is True
    assert under["steers"] is False


def _measure_meta(out, **change):
    """The keys of a measure_meta.json that steer reads, as `measure` writes them for _steer_dir."""
    (out / "measure_meta.json").write_text(json.dumps(
        dict({"tool": rq.TOOL, "commit": "def456", "tree_clean": True, "probe": "1c0b",
              "render_commit": "abc123"}, **change)))


def _steer_dir(tmp_path, embeddings=None):
    out = tmp_path / "qwen-1c0b"
    _rendered(out, "abc123", {"w": "x"}, probe="1c0b")
    (out / "measures.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in _steer_measures().values()))
    _measure_meta(out)
    if embeddings is not None:
        (out / "speaker_embeddings.jsonl").write_text("".join(
            json.dumps({"id": i, "embedding": "zonos_SpeakerEmbeddingLDA_128", "emb": e}) + "\n"
            for i, e in embeddings.items()))
    return out


def test_steer_prints_and_writes_the_reading(tmp_path, capsys):
    out = _steer_dir(tmp_path)
    rq.main(["steer", "--out", str(out)])
    printed = capsys.readouterr().out
    res = json.loads((out / "steer.json").read_text())
    assert res["render_commit"] == "abc123" and res["measure_commit"] == "def456"
    assert res["steer_shift_st"] == 4.0
    assert res["speaker_embeddings"] is False and "commit" in res and "tree_clean" in res
    assert res["methods"] == rq.steer_readout(rq.plan_1c0b(), _steer_measures())["methods"]
    assert "3 of 10 methods steer" in printed
    assert "F0 not measured (out of range or unvoiced), so not separating: M-weight-hi-k2" in printed
    assert "the pole cosine is null" in printed


def test_steer_reports_the_pole_cosine_from_the_speaker_embeddings(tmp_path):
    emb = {it["id"]: [1.0, 0.0] for it in rq.plan_1c0b()}
    emb.update({it["id"]: [0.0, 1.0] for it in rq.plan_1c0b()
                if it["method"] == "age" and it["pole"] == "high"})
    out = _steer_dir(tmp_path, embeddings=emb)
    rq.main(["steer", "--out", str(out)])
    res = json.loads((out / "steer.json").read_text())
    cos = {(m["gender"], m["method"]): m["pole_cosine"] for m in res["methods"]}
    assert cos[("Male", "age")] == pytest.approx(0.0) and cos[("Female", "age")] == pytest.approx(0.0)
    assert cos[("Male", "numeric pitch")] == pytest.approx(1.0)
    assert res["speaker_embeddings"] is True
    del emb["F-num-hi-k2"]
    out = _steer_dir(tmp_path / "again", embeddings=emb)
    with pytest.raises(SystemExit, match="REFUSING: .*speaker_embeddings.jsonl lacks 1 of 66"):
        rq.main(["steer", "--out", str(out)])


def test_steer_refuses_a_folder_that_is_not_a_measured_1c0b_render(tmp_path):
    zero = tmp_path / "zero"
    _rendered(zero, "abc123", {"w": "x"})
    with pytest.raises(SystemExit, match="REFUSING: .* records probe 1c0; steer reads a 1c0b"):
        rq.main(["steer", "--out", str(zero)])
    out = _steer_dir(tmp_path)
    meta = json.loads((out / "render_meta.json").read_text())
    (out / "render_meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "probe"}))
    with pytest.raises(SystemExit, match="REFUSING: .* records probe None; steer reads a 1c0b"):
        rq.main(["steer", "--out", str(out)])
    (out / "render_meta.json").write_text(json.dumps(meta))
    lines = (out / "measures.jsonl").read_text().splitlines(keepends=True)
    (out / "measures.jsonl").write_text("".join(lines[:-1]))
    with pytest.raises(SystemExit, match="REFUSING: .*measures.jsonl lacks 1 of 66 .*measure first"):
        rq.main(["steer", "--out", str(out)])
    (out / "measures.jsonl").unlink()
    with pytest.raises(SystemExit, match="REFUSING: .*lacks 66 of 66"):
        rq.main(["steer", "--out", str(out)])
    assert not (out / "steer.json").exists()


def test_steer_refuses_measures_that_are_not_this_renders_1c0b_measures(tmp_path):
    out = _steer_dir(tmp_path)
    (out / "measure_meta.json").unlink()
    with pytest.raises(SystemExit, match="REFUSING: .* no readable measure_meta.json .*measure first"):
        rq.main(["steer", "--out", str(out)])
    (out / "measure_meta.json").write_text("[]")
    with pytest.raises(SystemExit, match="REFUSING: .* no readable measure_meta.json"):
        rq.main(["steer", "--out", str(out)])
    _measure_meta(out, probe="1c0")
    with pytest.raises(SystemExit, match="REFUSING: .*measure_meta.json records probe 1c0; steer reads"):
        rq.main(["steer", "--out", str(out)])
    _measure_meta(out, probe=None)
    with pytest.raises(SystemExit, match="REFUSING: .*measure_meta.json records probe None; steer reads"):
        rq.main(["steer", "--out", str(out)])
    _measure_meta(out, render_commit="abc999")
    with pytest.raises(SystemExit, match="REFUSING: .*measured render abc999, not render_meta.json's "
                                         "abc123; measure again"):
        rq.main(["steer", "--out", str(out)])
    assert not (out / "steer.json").exists()
    _measure_meta(out)
    rq.main(["steer", "--out", str(out)])
    assert json.loads((out / "steer.json").read_text())["measure_commit"] == "def456"


# --- qwen_load: the float32 waveform decoder ------------------------------------------------

def _fake_qwen(monkeypatch, attach=True, tokenizer_honours_dtype=True, decoder_dtypes=None):
    """`torch` and `qwen_tts` stand-ins in sys.modules. As qwen-tts 0.1.1 does, the model's own
    speech tokenizer comes up in the main model's dtype. A tokenizer's model holds an encoder and a
    decoder and, like the nn.Module that registers the encoder first, lists the encoder's
    parameters first. Every parameter is in the loaded dtype, except that `decoder_dtypes` (dtype
    names) are the decoder's parameters in the tokenizer `load_qwen` loads."""
    import itertools
    import types

    class DType:
        def __init__(self, name):
            self.name = name

        def __str__(self):
            return "torch." + self.name
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float32 = DType("bfloat16"), DType("float32")
    calls = {}

    def module(dtypes):
        return types.SimpleNamespace(
            parameters=lambda: iter([types.SimpleNamespace(dtype=d) for d in dtypes]))

    class Tokenizer:
        def __init__(self, dtype, decoder=None):
            encoder, decoder = module([dtype] * 2), module(decoder or [dtype] * 2)
            self.model = types.SimpleNamespace(
                encoder=encoder, decoder=decoder,
                parameters=lambda: itertools.chain(encoder.parameters(), decoder.parameters()))

        @classmethod
        def from_pretrained(cls, path, device_map, dtype):
            calls["tokenizer"] = (path, device_map, dtype)
            return cls(dtype if tokenizer_honours_dtype else torch.bfloat16,
                       decoder=decoder_dtypes and [getattr(torch, n) for n in decoder_dtypes])

    class Inner:
        def __init__(self, dtype):
            self.speech_tokenizer = Tokenizer(dtype)

        def load_speech_tokenizer(self, tokenizer):
            calls["attached"] = tokenizer
            if attach:
                self.speech_tokenizer = tokenizer

    class Model:
        @classmethod
        def from_pretrained(cls, path, device_map, dtype):
            calls["model"] = (path, device_map, dtype)
            m = cls()
            m.model = Inner(dtype)
            return m
    qwen_tts = types.ModuleType("qwen_tts")
    qwen_tts.Qwen3TTSModel, qwen_tts.Qwen3TTSTokenizer = Model, Tokenizer
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_tts)
    return torch, calls


def test_load_qwen_attaches_a_float32_speech_tokenizer(monkeypatch):
    import qwen_load
    torch, calls = _fake_qwen(monkeypatch)
    model = qwen_load.load_qwen("/models/qwen")
    assert calls["model"] == ("/models/qwen", "cuda:0", torch.bfloat16)
    assert calls["tokenizer"] == ("/models/qwen/speech_tokenizer", "cuda:0", torch.float32)
    assert calls["attached"] is model.model.speech_tokenizer
    assert qwen_load.decoder_dtype(model) == qwen_load.TOKENIZER_DTYPE == "float32"
    assert qwen_load.decoder_dtypes(model) == {"float32"}


@pytest.mark.parametrize("fault", ["not attached", "tokenizer ignores dtype"])
def test_load_qwen_refuses_a_decoder_that_is_not_float32(monkeypatch, fault):
    import qwen_load
    _fake_qwen(monkeypatch, attach=fault != "not attached",
               tokenizer_honours_dtype=fault != "tokenizer ignores dtype")
    with pytest.raises(SystemExit, match="REFUSING: Qwen's waveform decoder loaded as bfloat16"):
        qwen_load.load_qwen("/models/qwen")


@pytest.mark.parametrize("decoder, got", [(("bfloat16", "bfloat16"), "bfloat16"),
                                          (("float32", "bfloat16"), r"bfloat16\+float32")])
def test_load_qwen_reads_the_decoder_not_the_encoder(monkeypatch, decoder, got):
    """The reloaded tokenizer's encoder is float32 and its decoder is not, or not wholly: its first
    parameter is the encoder's, so only every decoder parameter tells."""
    import qwen_load
    _fake_qwen(monkeypatch, decoder_dtypes=decoder)
    with pytest.raises(SystemExit, match="REFUSING: Qwen's waveform decoder loaded as %s, not "
                                         "float32" % got):
        qwen_load.load_qwen("/models/qwen")


def _loads_qwen_directly(src):
    """Whether `src` calls a `from_pretrained` (the model's own load, which decodes in bf16)."""
    return any(isinstance(n, ast.Attribute) and n.attr == "from_pretrained"
               for n in ast.walk(ast.parse(src)))


@pytest.mark.parametrize("rel", ["scripts/stages/synth_qwen.py", "scripts/tools/render_qwen_1c0.py"])
def test_qwen_renderers_load_through_load_qwen(rel):
    src = (REPO / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert not _loads_qwen_directly(src), rel
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "load_qwen"
               for n in ast.walk(tree)), rel
    assert any(isinstance(n, ast.ImportFrom) and n.module == "qwen_load"
               and "load_qwen" in {a.name for a in n.names} for n in ast.walk(tree)), rel
    # Positive control: the load both used before qwen_load is seen.
    assert _loads_qwen_directly(
        "model = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map='cuda:0', dtype=torch.bfloat16)")

