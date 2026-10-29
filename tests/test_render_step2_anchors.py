"""Phase 1S Step 2a's anchor pool is the pre-registered one: 114 grounded adult core-band candidates
with seeds disjoint from every earlier probe, the three automated gates filter in the registered
order and bound, a resume cannot mix two renders, and the admission report names what it filtered."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("render_step2_anchors",
                                               REPO / "scripts/tools/render_step2_anchors.py")
s2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2)
rq = s2.rq

FORBIDDEN_TONE = ("whimsical", "magical", "playful", "sparkling", "bubbly")
KEYS = ("gender", "pitch", "speed", "volume", "age", "clarity", "fluency", "accent", "texture",
        "emotion", "tone", "personality")


def _emb(x, y):
    return {"zonos_lda": [x, y], "ecapa": [x, y]}


def _meas(cid, gender, zone="core", out=False):
    return {"id": cid, "gender": gender, "zone": zone, "f0_out_of_range": out}


def test_the_plan_is_114_candidates_57_per_gender():
    items = s2.plan_items()
    assert len(items) == 114
    assert sum(it["gender"] == "Female" for it in items) == 57
    assert sum(it["gender"] == "Male" for it in items) == 57
    assert len({it["id"] for it in items}) == 114


def test_the_plan_is_deterministic():
    assert s2.plan_items() == s2.plan_items()


def test_seeds_are_96_prefixed_by_gender_and_disjoint_from_earlier_probes():
    items = s2.plan_items()
    seeds = [it["seed"] for it in items]
    assert len(set(seeds)) == 114
    for it in items:
        gd = 1 if it["gender"] == "Male" else 2
        assert it["seed"] in range(960000 + 1000 * gd, 960000 + 1000 * gd + s2.PER_GENDER)
    earlier = {it["seed"] for it in rq.plan_items("1c0")} | {it["seed"] for it in rq.plan_items("1c0b")}
    assert earlier.isdisjoint(seeds)


def test_every_instruct_has_all_twelve_keys_and_grounded_middle_pitch():
    for it in s2.plan_items():
        parts = dict(p.split(": ", 1) for p in it["instruct"].split(". ") if ": " in p)
        assert [k for k in KEYS if k in parts] == list(KEYS)
        pitch = parts["pitch"].split(" ")[0]
        assert pitch in ("Low", "Medium", "High"), it["id"]
        assert not any(w in it["instruct"].lower() for w in FORBIDDEN_TONE), it["id"]


def test_age_and_texture_vary_across_each_gender():
    for gender in ("Male", "Female"):
        items = [it for it in s2.plan_items() if it["gender"] == gender]
        assert len({it["label"].split(",")[0] for it in items}) == 3
        assert len({it["instruct"].split("texture: ")[1].split(".")[0] for it in items}) == len(s2.TEXTURES)


def test_each_candidate_reads_one_of_the_three_authored_sentences():
    for it in s2.plan_items():
        assert it["text"] in rq.SENTENCES
        assert it["text"] == rq.SENTENCES[it["sentence"] - 1]


def test_the_plan_selects_ids_and_refuses_unknown_ones():
    items = s2.plan_items()
    assert s2.select_ids(items, ["s2a-M01"])[0]["id"] == "s2a-M01"
    assert s2.subset_ids(items, None) is None
    with pytest.raises(SystemExit):
        s2.select_ids(items, ["s2a-M99"])


def test_admission_filters_qc_then_f0_then_distinctness_in_order():
    candidates = [_meas("a", "Female"), _meas("b", "Female"), _meas("c", "Female", zone="niche"),
                  _meas("d", "Female", out=True), _meas("e", "Female")]
    qc = {"a": {"id": "a", "hard_pass": True}, "b": {"id": "b", "hard_pass": True},
          "c": {"id": "c", "hard_pass": True}, "d": {"id": "d", "hard_pass": True},
          "e": {"id": "e", "hard_pass": False}}
    emb = {"a": _emb(1, 0), "b": _emb(0, 1), "c": _emb(1, 0), "d": _emb(1, 0), "F6-s2": _emb(-1, 0)}
    report = s2.admit(candidates, qc, emb, {"F6-s2": "Female"})
    stages = {f["id"]: f["stage"] for f in report["filtered"]}
    assert stages == {"e": "qc", "c": "f0", "d": "f0"}
    assert report["admitted"]["Female"] == ["a", "b"]


def _pair(cos):
    """A (zonos, ecapa) pair whose zonos cosine to (1, 0) is `cos`; ecapa is orthogonal to (0, 1)."""
    return {"zonos_lda": [cos, (1 - cos * cos) ** 0.5], "ecapa": [1.0, 0.0]}


def test_distinctness_is_strictly_below_the_bound():
    locked = {"zonos_lda": [1.0, 0.0], "ecapa": [0.0, 1.0]}
    emb = {"M4-s3": locked, "just-under": _pair(s2.DISTINCT_COS - 1e-6),
           "just-over": _pair(s2.DISTINCT_COS + 1e-6)}
    qc = {k: {"id": k, "hard_pass": True} for k in ("just-under", "just-over")}
    candidates = [_meas("just-over", "Male"), _meas("just-under", "Male")]
    report = s2.admit(candidates, qc, emb, {"M4-s3": "Male"})
    assert [(f["id"], f["stage"]) for f in report["filtered"]] == [("just-over", "distinct")]
    assert report["admitted"]["Male"] == ["just-under"]


def test_a_candidate_near_on_only_the_ecapa_model_is_still_refused():
    emb = {"M4-s3": {"zonos_lda": [1.0, 0.0], "ecapa": [0.0, 1.0]},
           "x": {"zonos_lda": [0.0, 1.0], "ecapa": [0.0, 1.0]}}
    report = s2.admit([_meas("x", "Male")], {"x": {"id": "x", "hard_pass": True}}, emb,
                      {"M4-s3": "Male"})
    assert report["admitted"]["Male"] == []
    assert "ecapa" in report["filtered"][0]["reason"]


def test_the_recommendation_is_capped_and_a_surplus_is_not_a_shortfall():
    n = 45
    dim = n + 1
    candidates = [_meas("F%02d" % i, "Female") for i in range(n)]
    qc = {c["id"]: {"id": c["id"], "hard_pass": True} for c in candidates}
    emb = {}
    for i, c in enumerate(candidates):
        one_hot = [1.0 if j == i else 0.0 for j in range(dim)]
        emb[c["id"]] = {"zonos_lda": one_hot, "ecapa": one_hot}
    emb["F6-s2"] = {"zonos_lda": [1.0 if j == n else 0.0 for j in range(dim)],
                    "ecapa": [1.0 if j == n else 0.0 for j in range(dim)]}
    report = s2.admit(candidates, qc, emb, {"F6-s2": "Female"})
    assert len(report["admitted"]["Female"]) == n
    assert report["recommended"]["Female"] == ["F%02d" % i for i in range(s2.NEED["Female"])]
    assert report["shortfall"] == {"Male": s2.NEED["Male"], "Female": 0}


def test_a_candidate_is_compared_only_to_its_own_gender():
    candidates = [_meas("f1", "Female")]
    qc = {"f1": {"id": "f1", "hard_pass": True}}
    emb = {"f1": _emb(1, 0), "M4-s3": _emb(1, 0), "F6-s2": _emb(0, 1)}
    report = s2.admit(candidates, qc, emb, {"M4-s3": "Male", "F6-s2": "Female"})
    assert report["admitted"]["Female"] == ["f1"]


def test_a_missing_embedding_refuses_rather_than_passing():
    qc = {"a": {"id": "a", "hard_pass": True}}
    with pytest.raises(SystemExit):
        s2.admit([_meas("a", "Male")], qc, {"M4-s3": _emb(1, 0)}, {"M4-s3": "Male"})
    with pytest.raises(SystemExit):
        s2.admit([_meas("a", "Male")], qc, {"a": _emb(1, 0)}, {"M4-s3": "Male"})


def test_the_recommendation_is_capped_and_shortfall_is_named():
    n = 45
    candidates = [_meas("F%02d" % i, "Female") for i in range(n)]
    qc = {c["id"]: {"id": c["id"], "hard_pass": True} for c in candidates}
    emb = {c["id"]: _emb(1, 0.001 * i) for i, c in enumerate(candidates)}
    emb["F6-s2"] = _emb(0, 1)
    report = s2.admit(candidates, qc, emb, {"F6-s2": "Female"})
    assert len(report["recommended"]["Female"]) <= s2.NEED["Female"]
    assert report["shortfall"]["Male"] == s2.NEED["Male"]


def test_a_resume_with_a_different_commit_is_refused(tmp_path):
    plan = s2.plan_items()
    meta = {"commit": "aaaa", "weights_sha256": {}, "qwen_tts": "0.1.1",
            "speech_tokenizer_dtype": "float32", "probe": s2.PROBE, "ids": None}
    (tmp_path / "manifest.jsonl").write_text("")
    (tmp_path / "render_meta.json").write_text(json.dumps(dict(meta, commit="bbbb")))
    with pytest.raises(SystemExit, match="commit"):
        s2.resume_state(str(tmp_path), dict(meta, items=114), plan)


def test_a_measure_refuses_a_render_of_another_probe(tmp_path):
    (tmp_path / "manifest.jsonl").write_text("")
    (tmp_path / "render_meta.json").write_text(json.dumps({"commit": "x", "probe": "1cii", "ids": None}))
    with pytest.raises(SystemExit, match="probe"):
        s2.rendered(str(tmp_path))
