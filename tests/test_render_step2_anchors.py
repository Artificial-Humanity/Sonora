"""Phase 1S Step 2a's anchor pool is the pre-registered one: 114 grounded adult core-band candidates
with seeds disjoint from every earlier probe, the three automated gates filter in the registered
order and bound, a resume cannot mix two renders, and the admission report names what it filtered."""
import argparse
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
    assert rq.select_ids(items, ["s2a-M01"])[0]["id"] == "s2a-M01"
    assert rq.subset_ids(items, None) is None
    with pytest.raises(SystemExit):
        rq.select_ids(items, ["s2a-M99"])


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
    # Every later clip sits within the bound of F00, so only F00 is admitted and the rest is short.
    assert report["admitted"]["Female"] == ["F00"]
    assert report["shortfall"]["Female"] == s2.NEED["Female"] - 1
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


def _audition(tmp_path, monkeypatch):
    """audition/app/main.py imported with `tmp_path` as its data root and ratings dir."""
    root = tmp_path.resolve()
    monkeypatch.setenv("AUDITION_DATA_ROOT", str(root))
    monkeypatch.setenv("AUDITION_RATINGS_DIR", str(root))
    spec = importlib.util.spec_from_file_location("audition_main_s2a",
                                                  REPO / "audition/app/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_manifest_row_names_its_clip_by_file_and_audition_resolves_it(tmp_path, monkeypatch):
    it = s2.plan_items()[0]
    row = s2.manifest_row(it, n_samples=48000, sr=24000, wall=1.2345, peak_limited=False,
                          commit="c0", dtype="float32")
    assert "wav" not in row
    assert row["file"] == "clips/%s.wav" % it["id"]
    assert row["seconds"] == 2.0
    probe = tmp_path / "probes" / "qwen-step2-anchors"
    clip_path = probe / "clips" / ("%s.wav" % it["id"])
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"RIFF")
    (probe / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    audition = _audition(tmp_path, monkeypatch)
    clip = audition._load_probe_manifest("qwen-step2-anchors", probe)["clips"][0]
    assert clip["wav"] == "probes/qwen-step2-anchors/clips/%s.wav" % it["id"]
    assert audition._resolve_audio(clip["wav"]) == clip_path.resolve()


@pytest.mark.parametrize("field", ("zonos_lda", "ecapa"))
def test_a_nan_candidate_embedding_refuses_rather_than_admitting(field):
    cand = {"zonos_lda": [1.0, 0.0], "ecapa": [1.0, 0.0]}
    cand[field] = [float("nan"), 0.0]
    emb = {"M4-s3": _emb(0, 1), "cand-nan": cand}
    qc = {"cand-nan": {"id": "cand-nan", "hard_pass": True}}
    with pytest.raises(SystemExit, match="cand-nan"):
        s2.admit([_meas("cand-nan", "Male")], qc, emb, {"M4-s3": "Male"})


def test_a_nan_reference_embedding_refuses_rather_than_admitting():
    emb = {"M4-s3": {"zonos_lda": [float("nan"), 0.0], "ecapa": [1.0, 0.0]}, "cand": _emb(1, 0)}
    qc = {"cand": {"id": "cand", "hard_pass": True}}
    with pytest.raises(SystemExit, match="M4-s3"):
        s2.admit([_meas("cand", "Male")], qc, emb, {"M4-s3": "Male"})


def _measure_row(cid):
    return {"id": cid, "gender": "Male", "zone": "core", "f0_out_of_range": False}


def test_admit_filter_takes_only_a_whole_plan_measure_in_plan_order():
    plan = s2.plan_items()
    rows = [_measure_row(it["id"]) for it in reversed(plan)]
    got = s2.plan_measures(rows, {"ids": None}, plan)
    assert [r["id"] for r in got] == [it["id"] for it in plan]


def test_admit_filter_refuses_a_measure_of_a_subset():
    plan = s2.plan_items()
    rows = [_measure_row(it["id"]) for it in plan]
    with pytest.raises(SystemExit, match="whole-plan"):
        s2.plan_measures(rows, {"ids": ["s2a-M01"]}, plan)


def test_admit_filter_refuses_a_plan_id_with_no_measures_row():
    plan = s2.plan_items()
    rows = [_measure_row(it["id"]) for it in plan[1:]]
    with pytest.raises(SystemExit, match="no measures row"):
        s2.plan_measures(rows, {"ids": None}, plan)


def test_cmd_admit_filter_refuses_a_partial_measure_on_disk(tmp_path):
    ids = ["s2a-M01", "s2a-M02"]
    (tmp_path / "measures.jsonl").write_text(
        "".join(json.dumps(_measure_row(i)) + "\n" for i in ids), encoding="utf-8")
    (tmp_path / "measure_meta.json").write_text(json.dumps({"ids": ids}), encoding="utf-8")
    (tmp_path / "qc_measures.jsonl").write_text(
        "".join(json.dumps({"id": i, "hard_pass": True}) + "\n" for i in ids), encoding="utf-8")
    emb = tmp_path / "embeddings.jsonl"
    emb.write_text("", encoding="utf-8")
    args = argparse.Namespace(out=str(tmp_path), qc_jsonl=None, embeddings_jsonl=str(emb))
    with pytest.raises(SystemExit, match="whole-plan"):
        s2.cmd_admit_filter(args)
    assert not (tmp_path / "admission_report.json").exists()


def test_the_grid_is_mixed_radix_and_its_first_57_are_distinct_per_gender():
    axes = [s2.grid_axes(i) for i in range(72)]
    assert len(set(axes)) == 72
    assert len(set(axes[:57])) == 57
    assert all(a < 3 and r < 3 and t < 4 and o < 2 for a, r, t, o in axes)


def test_every_age_band_reads_all_three_sentences():
    items = [it for it in s2.plan_items() if it["gender"] == "Female"]
    for band in {it["band"] for it in items}:
        sents = {it["text"] for it in items if it["band"] == band}
        assert sents == set(rq.SENTENCES), band


def test_both_tones_and_all_textures_appear_within_each_age_band():
    items = [it for it in s2.plan_items() if it["gender"] == "Male"]
    for band in {it["band"] for it in items}:
        sub = [it for it in items if it["band"] == band]
        assert {it["instruct"].split("tone: ")[1].split(".")[0] for it in sub} == \
            {t for t, _p in s2.TONES}
        assert {it["instruct"].split("texture: ")[1].split(".")[0] for it in sub} == \
            set(s2.TEXTURES)


def test_the_qc_manifest_adds_wav_beside_file_and_nothing_else():
    rows = [{"id": "s2a-M01", "file": "clips/s2a-M01.wav", "text": "t", "seed": 1}]
    out = s2.qc_manifest_rows(rows)
    assert out == [{"id": "s2a-M01", "file": "clips/s2a-M01.wav", "text": "t", "seed": 1,
                    "wav": "clips/s2a-M01.wav"}]
    assert "wav" not in rows[0]


def test_embed_clips_refuses_a_missing_wav_and_lists_the_anchors(tmp_path):
    rows = [{"id": "s2a-M01", "file": "clips/s2a-M01.wav"}]
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "s2a-M01.wav").write_bytes(b"")
    anchors = tmp_path / "anchors"
    (anchors / "clips").mkdir(parents=True)
    for cid in s2.LOCKED_ANCHORS:
        (anchors / "clips" / (cid + ".wav")).write_bytes(b"")
    clips = s2.embed_clips(rows, str(tmp_path), str(anchors))
    assert [cid for cid, _p in clips] == ["s2a-M01", "M4-s3", "F6-s2"]
    (anchors / "clips" / "F6-s2.wav").unlink()
    with pytest.raises(SystemExit, match="missing"):
        s2.embed_clips(rows, str(tmp_path), str(anchors))


def test_embed_rows_takes_one_row_per_clip_with_both_models():
    clips = [("a", "/x/a.wav"), ("M4-s3", "/y/M4-s3.wav")]
    out = s2.embed_rows(clips, lambda p: [1.0, 2.0], lambda p: [3.0])
    assert out == [{"id": "a", "zonos_lda": [1.0, 2.0], "ecapa": [3.0]},
                   {"id": "M4-s3", "zonos_lda": [1.0, 2.0], "ecapa": [3.0]}]


def test_ecapa_refuses_a_checkpoint_that_differs_from_its_pin(tmp_path):
    (tmp_path / "embedding_model.ckpt").write_bytes(b"not the pinned bytes")
    with pytest.raises(SystemExit, match="sha256"):
        s2.verify_ecapa_hashes(str(tmp_path), {"embedding_model.ckpt": "0" * 64})


def test_the_admitted_manifest_holds_only_admitted_clips_in_plan_order(tmp_path):
    plan = s2.plan_items()
    ids = [it["id"] for it in plan]
    rows = {i: {"id": i, "file": "clips/%s.wav" % i, "group": "step2 anchor pool", "gender": "Male"}
            for i in ids}
    measures = {i: {"id": i, "zone": "core"} for i in ids}
    report = {"admitted": {"Male": ["s2a-M03", "s2a-M01"], "Female": []}}
    m = s2.probe_manifest(report, plan, rows, measures, "qwen-step2-anchors", "probes/qwen-step2-anchors")
    assert [c["id"] for c in m["clips"]] == ["s2a-M01", "s2a-M03"]
    c = m["clips"][0]
    assert c["wav"] == "probes/qwen-step2-anchors/clips/s2a-M01.wav"
    assert c["acoustics"] == {"id": "s2a-M01", "zone": "core"}
    assert c["group"] == "step2 anchor pool"
    assert m["probe"] == "qwen-step2-anchors"


def test_the_audition_relpath_is_under_the_data_root_or_refused(tmp_path):
    assert s2.audition_relpath("/data/model-training/datasets/probes/qwen-step2-anchors") == \
        "probes/qwen-step2-anchors"
    with pytest.raises(SystemExit, match="outside"):
        s2.audition_relpath(str(tmp_path))
