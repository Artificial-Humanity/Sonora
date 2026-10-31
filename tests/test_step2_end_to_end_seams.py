"""Phase 1S Step 2c to Check 9, across script boundaries.

Every module in this chain has its own unit tests, and each one builds the others' inputs by hand.
That is the gap these tests close: here the output of one real command is the input of the next,
with nothing adapted in between.

    render_step2_corpus  lock-speakers -> plan -> (render, simulated) -> build-corpus
        -> q_synth_22k/{train_op.txt, val_op.txt, speakers.json, provenance.json, wavs/}
    make_warmstart        reads that speakers.json
    measure_speaker_hnr   reads that train_op.txt + provenance.json['raw_root'] -> HNR report
    render_ear_staged     reads that report and filelist (check 9's Q half)
    staged_bench          pairs the (S1, F1, Q) triplets and reads check 9

Only the GPU render, the engine weights' pin, the G2P dictionary and torch (mel-stats) are stood in for. The audio is real:
24 kHz harmonic stacks with a per-speaker F0 and noise level, resampled by the real
`prep_staged_corpora.resample_to`, then measured by the real periodicity estimator.

The corpus is the registered 90 speakers, with `MIN_QC` lowered to 14 clips each (4 train, 10
validation) so the fixture is minutes of audio, not hours.

Names in the task description that this checkout does not have: `eval_staged_bench.py`,
`load_hnr_for_arm`, `check9_decision`, `check9_verdict`, `run_check9`, `_q_raw_wav`, `_read_rows`.
Their roles are played by `render_ear_staged.hnr_table`/`q_clips`, `staged_bench.pair_q_triplets`/
`rated_in_all`/`reading(9, ...)`, and `measure_speaker_hnr.source_24k` plus the row parse in `main`.
"""
import ast
import collections
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from scripts_layout import REPO, SCRIPTS

SCRIPTS.on_path()

import measure_speaker_hnr as mh  # noqa: E402
import prep_staged_corpora as prep  # noqa: E402
import render_ear_staged as res  # noqa: E402
import render_step2_corpus as s2c  # noqa: E402
import staged_bench as sb  # noqa: E402

N_SPK = s2c.N_SPEAKERS
N_CLIPS = 14                    # QC-passing clips a speaker is built from: 4 train + 10 val
N_BAD = 2                       # extra clips per speaker that must be filtered out
SR_RAW = 24000
CLIP_SECONDS = 1.2
HEARD = "hɛloʊ"                 # what the stand-in G2P returns; every symbol is in the stock table


class _Warmstart:
    """`make_warmstart._index_map` and `_speaker_prefix_ok`, from the module's own source.

    The module imports torch, hydra and lightning at the top and the host test group installs
    none of them, so importing it would skip these seams on exactly the machines that run CI.
    These two functions need only `collections`; the real module is imported as well, and the
    two are compared, whenever torch is present (the container)."""

    def __init__(self):
        path = SCRIPTS / "make_warmstart.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        want = {"_index_map", "_speaker_prefix_ok"}
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
        assert {f.name for f in fns} == want
        ns = {"collections": collections}
        exec(compile(ast.Module(fns, []), str(path), "exec"), ns)
        self._index_map, self._speaker_prefix_ok = ns["_index_map"], ns["_speaker_prefix_ok"]


mw = _Warmstart()


def _snr_of(s):
    """40 dB down to 5 dB, in an order (7 is coprime to 90) that is not the F0 order: the estimator
    is sensitive to F0 as well as noise, and a ramp that moved with F0 would confound the two."""
    return 40.0 - 35.0 * ((7 * s) % N_SPK) / (N_SPK - 1)


def _f0_of(s):
    return 100.0 + 1.5 * s


def _clip(s, path):
    """A 24 kHz harmonic stack at speaker `s`'s F0, with a speaker-dependent noise level (`_snr_of`) (so the
    speakers spread over HNR, as a pairing needs)."""
    t = np.arange(int(SR_RAW * CLIP_SECONDS)) / SR_RAW
    x = sum(np.sin(2 * np.pi * _f0_of(s) * k * t) / k for k in range(1, 16))
    x = x / np.abs(x).max()
    noise = np.random.default_rng(s).normal(0, 1, len(x))
    snr_db = _snr_of(s)
    x = x + noise * (np.sqrt(np.mean(x ** 2)) / 10 ** (snr_db / 20.0))
    x = 0.5 * x / np.abs(x).max()
    sf.write(str(path), x.astype("float32"), SR_RAW, "PCM_16")


def _ref_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(2400, dtype="float32"), SR_RAW, "PCM_16")


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class _FakeG2P:
    def phonemize(self, text):
        return HEARD


def _run(argv):
    return s2c.main(argv)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, request):
    """The whole of step 2c, run once: -> a namespace of the paths and what each step wrote."""
    base = tmp_path_factory.mktemp("step2")
    mp = pytest.MonkeyPatch()
    request.addfinalizer(mp.undo)            # a failed step must not leave the patches in place
    anchors, locked, text = base / "anchors", base / "locked", base / "text"
    raw = base / "datasets" / "q_synth_raw"                  # `datasets/`: partition_of keys on it
    out = base / "data" / "q_synth_22k"

    locked.mkdir(parents=True)
    # ---- lock-speakers: 45 + 50 admitted anchors, a few of them failing the owner's ear check
    males = ["M%02d" % i for i in range(47)]
    females = ["F%02d" % i for i in range(48)]
    rows, verdicts = [], []
    for gender, ids in (("Male", males), ("Female", females)):
        for i, cid in enumerate(ids):
            rows.append({"id": cid, "file": "clips/%s.wav" % cid, "text": "anchor %s." % cid,
                         "gender": gender})
            _ref_wav(anchors / "clips" / (cid + ".wav"))
            # index 3 is out of zone and index 11 hums: the lock must skip both, in plan order
            verdicts.append({"id": cid, "zone": "niche" if i == 3 else "core",
                             "hum": "2" if i == 11 else "0"})
    _jsonl(anchors / "manifest.jsonl", rows)
    (anchors / "admission_report.json").write_text(
        json.dumps({"admitted": {"Male": males, "Female": females}}), encoding="utf-8")
    (anchors / "verdicts.csv").write_text(
        "id,zone,hum\n" + "".join("%(id)s,%(zone)s,%(hum)s\n" % v for v in verdicts),
        encoding="utf-8")
    _jsonl(locked / "manifest.jsonl",
           [{"id": "M4-s3", "file": "M4-s3.wav", "text": "m.", "gender": "Male"},
            {"id": "F6-s2", "file": "F6-s2.wav", "text": "f.", "gender": "Female"}])
    _ref_wav(locked / "M4-s3.wav")
    _ref_wav(locked / "F6-s2.wav")
    raw.mkdir(parents=True)
    assert _run(["lock-speakers", "--anchors-dir", str(anchors), "--locked-dir", str(locked),
                 "--out-root", str(raw)]) == 0

    # ---- plan: the frozen-bank check, then bank.json + plan.json per speaker
    (text / "speakers").mkdir(parents=True)
    (text / "meta.json").write_text(json.dumps({"sha256": s2c.TEXT_SHA}), encoding="utf-8")
    for s in range(N_SPK):
        _jsonl(text / "speakers" / ("spk_%02d.jsonl" % s),
               [{"id": "2b-%05d" % (s * 100 + k), "text": "Speaker %d says line %d." % (s, k),
                 "role": "primary", "speaker_line_index": k}
                for k in range(1, N_CLIPS + N_BAD + 1)])
    mp.setattr(s2c.r2, "engine_pin", lambda engine: {"engine": engine, "wheel": "stand-in"})
    # the fixture writes N_CLIPS + N_BAD primary rows a speaker for speed; the plan's frozen-bank
    # check (`_verified_speaker_rows`) takes that as the whole bank, with no reserve rows
    mp.setattr(s2c, "PRIMARY", N_CLIPS + N_BAD)
    mp.setattr(s2c, "SPEAKER_LINES", N_CLIPS + N_BAD)
    assert _run(["plan", "--text-dir", str(text), "--out-root", str(raw), "--allow-dirty"]) == 0

    # ---- render, simulated: what synth_bank.sh + qc_gate leave behind, from the planned banks
    table = json.loads((raw / "speakers.json").read_text())["speakers"]
    for sp in table:
        d = raw / sp["name"]
        bank = json.loads((d / "bank.json").read_text())
        (d / "audio").mkdir(exist_ok=True)
        _clip(sp["index"], d / "audio" / "_first.wav")
        man, qc = [], []
        for n, line in enumerate(bank["lines"]):
            shutil.copyfile(d / "audio" / "_first.wav", d / "audio" / (line["id"] + ".wav"))
            man.append({"id": line["id"], "wav": line["id"] + ".wav", "text": line["text"]})
            good = n < N_CLIPS
            qc.append({"id": line["id"], "hard_pass": good, "duration": 6.0 if good else 0.5})
        (d / "audio" / "_first.wav").unlink()
        _jsonl(d / "audio" / ("%s_manifest.jsonl" % sp["engine"]), man)
        _jsonl(d / "qc_measures.jsonl", qc)

    # ---- build-corpus: MIN_QC and the G2P are stood in for, nothing else
    mp.setattr(s2c, "MIN_QC", N_CLIPS)
    import matcha.text.op_g2p as op_g2p
    mp.setattr(op_g2p, "OpenPhonemizerG2P", _FakeG2P)
    assert _run(["build-corpus", "--raw-root", str(raw), "--out-dir", str(out)]) == 0
    mp.undo()

    class P:
        pass
    p = P()
    p.base, p.raw, p.out, p.table = base, raw, out, table
    p.train = (out / "train_op.txt").read_text(encoding="utf-8").splitlines()
    p.val = (out / "val_op.txt").read_text(encoding="utf-8").splitlines()
    p.speakers = json.loads((out / "speakers.json").read_text())
    p.prov = json.loads((out / "provenance.json").read_text())
    yield p


# ------------------------------------------------------------------------------ the written corpus

def test_lock_plan_and_build_write_the_registered_corpus(pipeline):
    p = pipeline
    assert len(p.train) == N_SPK * (N_CLIPS - 10) and len(p.val) == N_SPK * 10
    assert p.speakers["n_spks"] == N_SPK == len(p.speakers["speakers"])
    assert [s["index"] for s in p.speakers["speakers"]] == list(range(N_SPK))
    assert p.prov["rows"] == {"clipped": 0, "train": len(p.train), "val": len(p.val)}
    assert p.prov["sample_rate"] == 22050 and p.prov["speakers"] == N_SPK
    # the lock skipped the niche (index 3) and the hummy (index 11) Male anchors
    anchor_ids = [s["anchor"]["id"] for s in p.speakers["speakers"][2:44]]
    assert "M03" not in anchor_ids and "M11" not in anchor_ids and anchor_ids[:3] == ["M00", "M01", "M02"]
    # every row is 3-field, absolute, under wavs/<spk_NN>/, and its index is its directory's
    for line in p.train + p.val:
        wav, idx, ph = line.split("|")
        assert Path(wav).is_absolute() and Path(wav).parent.parent == p.out / "wavs"
        assert Path(wav).parent.name == "spk_%02d" % int(idx) and ph == HEARD
        assert sf.info(wav).samplerate == 22050 and sf.info(wav).subtype == "PCM_16"


def test_the_clips_that_fail_qc_or_the_duration_bounds_reach_no_filelist(pipeline):
    p = pipeline
    ids = {Path(l.split("|")[0]).stem for l in p.train + p.val}
    assert len(ids) == N_SPK * N_CLIPS
    for s in (0, 44, 89):
        extra = [("spk_%02d-%03d" % (s, k)) for k in (N_CLIPS + 1, N_CLIPS + 2)]
        assert not set(extra) & ids
        assert not any((p.out / "wavs" / ("spk_%02d" % s) / (e + ".wav")).exists() for e in extra)


def test_the_corpus_agrees_with_every_registered_description_of_it(pipeline):
    cfg = yaml.safe_load((REPO / "configs" / "data" / "q_synth.yaml").read_text())
    assert (cfg["n_spks"] == prep.STATS_CORPORA["q_synth_22k"] == pipeline.speakers["n_spks"]
            == s2c.N_SPEAKERS)
    assert prep.STATS_ALIASES["q_synth"] == "q_synth_22k" == cfg["train_filelist_path"].split("/")[1]
    assert cfg["sample_rate"] == pipeline.prov["sample_rate"]
    assert Path(cfg["train_filelist_path"]).name == "train_op.txt"
    assert Path(cfg["valid_filelist_path"]).name == "val_op.txt"


def test_mel_stats_hands_prep_the_corpus_the_build_wrote(pipeline, monkeypatch):
    seen = []
    monkeypatch.setattr(prep, "DATA", str(pipeline.out.parent))
    monkeypatch.setattr(prep, "stats", lambda a: seen.append((a.corpus, a.workers)))
    assert s2c.cmd_mel_stats(type("A", (), {"out_dir": str(pipeline.out), "workers": 3})()) == 0
    assert seen == [("q_synth_22k", 3)]
    assert prep.STATS_ALIASES.get("q_synth_22k", "q_synth_22k") in prep.STATS_CORPORA


# ----------------------------------------------------------------------------- make_warmstart seam

def test_make_warmstart_reads_the_written_speakers_json_and_finds_no_id_map(pipeline):
    """Q's table is a list of speaker records, not a `*_id_to_index` namespace. `_index_map`
    takes it without error and returns nothing, so a donor-prefix proof against it REFUSES. That
    is the registered path: staged_q.yaml warm-starts WITHOUT --donor-speakers (a fresh 90-row
    table), and a refusal is the safe answer to a prefix claim Q cannot support."""
    assert mw._index_map(pipeline.speakers) == {}
    ok, detail = mw._speaker_prefix_ok({"libritts_id_to_index": {"a": 0}}, pipeline.speakers)
    assert ok is False and "no `*_id_to_index` block" in detail
    ok, _ = mw._speaker_prefix_ok(pipeline.speakers, pipeline.speakers)
    assert ok is False


def test_the_extracted_functions_are_the_modules_own_when_torch_is_present(pipeline):
    pytest.importorskip("torch")
    import make_warmstart as real
    assert real._index_map(pipeline.speakers) == mw._index_map(pipeline.speakers) == {}
    donor = {"libritts_id_to_index": {"a": 0, "b": 1}}
    assert real._speaker_prefix_ok(donor, donor) == mw._speaker_prefix_ok(donor, donor)


def test_the_staged_q_experiment_warm_starts_with_a_fresh_table_not_a_prefix():
    exp = (REPO / "configs" / "experiment" / "staged_q.yaml").read_text()
    assert "--experiment staged_q --donor matcha_vctk.ckpt" in exp
    assert "donor-speakers" not in exp and "fresh 90-row speaker table" in exp


# ----------------------------------------------------------------- measure_speaker_hnr seam (main)

@pytest.fixture(scope="module")
def hnr(pipeline):
    """measure_speaker_hnr.main over the written corpus, with every audio read recorded."""
    mp = pytest.MonkeyPatch()
    read = []
    real = sf.read

    def spy(path, *a, **k):
        read.append(str(path))
        return real(path, *a, **k)
    mp.setattr(mh.sf, "read", spy)
    report = pipeline.base / "q_hnr.json"
    mp.setattr(sys, "argv", ["measure_speaker_hnr.py", "--corpus", str(pipeline.out),
                             "--min-rows", "4", "--clips-per-speaker", "2",
                             "--json", str(report)])
    mh.main()
    mp.undo()
    return type("H", (), {"path": report, "doc": json.loads(report.read_text()), "read": read})


def test_every_22k_train_row_resolves_through_provenance_to_its_24k_source(pipeline):
    raw_root = pipeline.prov["raw_root"]
    assert raw_root == str(pipeline.raw)
    for line in pipeline.train:
        wav22 = line.split("|")[0]
        src = Path(mh.source_24k(wav22, raw_root))
        assert src == pipeline.raw / Path(wav22).parent.name / "audio" / Path(wav22).name
        assert sf.info(str(src)).samplerate == 24000


def test_main_measures_the_24k_sources_not_the_22k_copies(pipeline, hnr):
    doc = hnr.doc
    assert doc["raw_root"] == str(pipeline.raw) and doc["split"] == "train"
    assert doc["filelist"] == str(pipeline.out / "train_op.txt") and doc["dataset"] is None
    assert doc["skipped"] == [] and sorted(map(int, doc["speakers"])) == list(range(N_SPK))
    assert hnr.read, "the spy saw no read"
    assert all(Path(w).parent.name == "audio" and str(w).startswith(str(pipeline.raw))
               for w in hnr.read)
    assert not any(str(pipeline.out / "wavs") in w for w in hnr.read)


def test_the_report_recovers_each_speakers_planted_f0_and_orders_hnr(pipeline, hnr):
    sp = hnr.doc["speakers"]
    for s in range(N_SPK):
        assert sp[str(s)]["f0"] == pytest.approx(_f0_of(s), rel=0.03)
        assert sp[str(s)]["rows"] == N_CLIPS - 10 and sp[str(s)]["clips"] == 2
        assert sp[str(s)]["partition"] == "q_synth_raw"
    by_snr = sorted(range(N_SPK), key=_snr_of)
    quiet = np.mean([sp[str(s)]["hnr"] for s in by_snr[-10:]])
    noisy = np.mean([sp[str(s)]["hnr"] for s in by_snr[:10]])
    assert quiet > noisy + 1.5, "the noise ramp left no HNR spread to pair on"


# ------------------------------------------------------------- check 9 seam (render_ear_staged etc.)

def _args(report, **kw):
    return type("A", (), dict({"q_hnr_json": str(report), "min_seconds": 1, "max_seconds": 16}, **kw))()


def test_check9_reads_the_hnr_report_and_the_filelist_it_was_measured_from(pipeline, hnr, monkeypatch):
    monkeypatch.setattr(res, "DATA", str(pipeline.out.parent))
    qh = res.hnr_table(hnr.path)
    assert sorted(qh) == list(range(N_SPK)) and all({"hnr", "f0"} <= set(v) for v in qh.values())
    recorded = []
    clips, row_of = res.q_clips(_args(hnr.path), qh, lambda k, v: recorded.append(k))
    assert sorted(clips) == list(range(N_SPK)) and all(len(c) == N_CLIPS - 10 for c in clips.values())
    assert {w for cl in clips.values() for w, _ in cl} == set(row_of) == {
        l.split("|")[0] for l in pipeline.train}
    assert all(isinstance(sec, Fraction) and 1 < sec < 2 for cl in clips.values() for _w, sec in cl)
    assert recorded == ["q_speakers", "q_synth_22k/train_op.txt"]


def test_check9_pairs_triplets_from_the_measured_report_and_reads_them(pipeline, hnr, monkeypatch):
    monkeypatch.setattr(res, "DATA", str(pipeline.out.parent))
    qh = res.hnr_table(hnr.path)
    clips, _ = res.q_clips(_args(hnr.path), qh, lambda k, v: None)
    q_pool = {v: (qh[v]["hnr"], qh[v]["f0"], v) for v, cl in clips.items() if len(cl) >= res.MIN_CLIPS}
    assert len(q_pool) == N_SPK
    # S1 / F1 stand-ins read off the same instrument (offset, so no voice pairs with itself by
    # construction): 20 S1 voices and a 60-voice F1 pool.
    s1 = {1000 + i: (qh[i]["hnr"] + 0.01, qh[i]["f0"]) for i in range(20)}
    f1 = {2000 + i: (qh[i + 20]["hnr"] - 0.01, qh[i + 20]["f0"], 3000 + i) for i in range(60)}
    triplets = sb.pair_q_triplets(s1, f1, q_pool, min_triplets=16)
    assert len(triplets) >= 16
    assert len({t["q"] for t in triplets}) == len(triplets) and {t["q"] for t in triplets} <= set(range(N_SPK))
    assert all(set(t["widened"]) == {"F1", "Q"} for t in triplets)
    assert sum(not any(t["widened"].values()) for t in triplets) >= 1
    # and the reading takes those units: every triplet is a unit rated in all three arms
    arms = ("S1", "F1", "Q")
    units = range(len(triplets))
    gaps = {"S1": {u: 1.0 for u in units}, "F1": {u: 0.5 for u in units}, "Q": {u: 0.0 for u in units}}
    assert sb.rated_in_all(gaps, arms) == set(units)
    floors = {"S1": Fraction(1, 2), "F1": Fraction(3, 5), "Q": Fraction(1, 2)}
    r = sb.reading(9, gaps, floors)
    assert r["valid"] is True and r["outcome"] == "shown"


def test_a_pool_the_report_cannot_fill_is_a_refusal_not_a_short_check(pipeline, hnr):
    qh = res.hnr_table(hnr.path)
    q_pool = {v: (qh[v]["hnr"], qh[v]["f0"], v) for v in range(10)}      # only ten Q voices
    s1 = {1000 + i: (qh[i]["hnr"], qh[i]["f0"]) for i in range(20)}
    f1 = {2000 + i: (qh[i]["hnr"], qh[i]["f0"], i) for i in range(60)}
    with pytest.raises(ValueError, match="registered minimum"):
        sb.pair_q_triplets(s1, f1, q_pool, min_triplets=16)


# -------------------------------------------------------------------- refusal seams across scripts

def _mini(tmp_path, n_spk=3, rows=4, provenance=True, raw_root=True):
    """A hand-built 3-speaker q_synth-shaped corpus (real 24 kHz sources, real 22.05 kHz copies)."""
    corpus, raw = tmp_path / "data" / "q_synth_22k", tmp_path / "datasets" / "q_synth_raw"
    lines = []
    for s in range(n_spk):
        for k in range(rows):
            cid = "spk_%02d-%03d" % (s, k)
            src = raw / ("spk_%02d" % s) / "audio" / (cid + ".wav")
            src.parent.mkdir(parents=True, exist_ok=True)
            _clip(s, src)
            dst = corpus / "wavs" / ("spk_%02d" % s) / (cid + ".wav")
            s2c.resample_to(str(src), str(dst))
            lines.append("%s|%d|%s" % (dst, s, HEARD))
    (corpus / "train_op.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if provenance:
        (corpus / "provenance.json").write_text(
            json.dumps({"raw_root": str(raw)} if raw_root else {"speakers": n_spk}), encoding="utf-8")
    return corpus, raw


def _measure(monkeypatch, corpus, report, *extra):
    monkeypatch.setattr(sys, "argv", ["measure_speaker_hnr.py", "--corpus", str(corpus),
                                      "--min-rows", "1", "--clips-per-speaker", "1",
                                      "--json", str(report), *extra])
    mh.main()


def test_the_mini_corpus_is_measurable_so_the_refusals_below_are_the_cause(tmp_path, monkeypatch):
    corpus, _ = _mini(tmp_path)
    _measure(monkeypatch, corpus, tmp_path / "r.json")
    assert sorted(json.loads((tmp_path / "r.json").read_text())["speakers"]) == ["0", "1", "2"]


def test_a_provenance_without_raw_root_is_a_refusal_naming_the_flag(tmp_path, monkeypatch):
    corpus, _ = _mini(tmp_path, raw_root=False)
    with pytest.raises(SystemExit, match="records no raw_root; pass --raw-24k-root"):
        _measure(monkeypatch, corpus, tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_a_missing_provenance_is_a_refusal(tmp_path, monkeypatch):
    corpus, _ = _mini(tmp_path, provenance=False)
    with pytest.raises(SystemExit, match="no provenance.json naming its raw_root"):
        _measure(monkeypatch, corpus, tmp_path / "r.json")


def test_the_flag_stands_in_for_a_missing_provenance(tmp_path, monkeypatch):
    corpus, raw = _mini(tmp_path, provenance=False)
    _measure(monkeypatch, corpus, tmp_path / "r.json", "--raw-24k-root", str(raw))
    assert json.loads((tmp_path / "r.json").read_text())["raw_root"] == str(raw)


def test_a_missing_24k_source_refuses_though_the_22k_copy_exists(tmp_path, monkeypatch):
    corpus, raw = _mini(tmp_path)
    gone = raw / "spk_02" / "audio" / "spk_02-003.wav"
    gone.unlink()
    assert (corpus / "wavs" / "spk_02" / "spk_02-003.wav").is_file()
    with pytest.raises(SystemExit, match=r"spk_02-003\.wav is missing|spk_02-003\.wav.*missing"):
        _measure(monkeypatch, corpus, tmp_path / "r.json")      # one clip drawn per speaker: the
    assert not (tmp_path / "r.json").exists()                   # check still reaches every row


def test_a_22k_file_standing_in_for_the_24k_source_is_refused(tmp_path, monkeypatch):
    corpus, raw = _mini(tmp_path)
    shutil.copyfile(corpus / "wavs" / "spk_01" / "spk_01-000.wav",
                    raw / "spk_01" / "audio" / "spk_01-000.wav")
    with pytest.raises(SystemExit, match="22050 Hz, not 24000"):
        _measure(monkeypatch, corpus, tmp_path / "r.json")


def test_raw_24k_root_is_refused_on_a_4_field_filelist(tmp_path, monkeypatch):
    corpus, raw = _mini(tmp_path)
    lines = (corpus / "train_op.txt").read_text().splitlines()
    (corpus / "train_op.txt").write_text("\n".join(l + "|0.1" for l in lines) + "\n")
    with pytest.raises(SystemExit):
        _measure(monkeypatch, corpus, tmp_path / "r.json", "--raw-24k-root", str(raw))


def test_a_filelist_voice_missing_from_the_hnr_report_is_refused_by_check9(pipeline, hnr, monkeypatch):
    """The speaker-count seam: a report measured from a corpus with fewer speakers than the
    filelist now being read."""
    monkeypatch.setattr(res, "DATA", str(pipeline.out.parent))
    qh = res.hnr_table(hnr.path)
    del qh[57]
    with pytest.raises(SystemExit, match="Q voice 57 has no entry"):
        res.q_clips(_args(hnr.path), qh, lambda k, v: None)


def test_a_filelist_index_beyond_the_speaker_table_is_refused_by_check9(pipeline, hnr, tmp_path,
                                                                        monkeypatch):
    """The index seam: a train_op.txt row naming speaker 90 in a 90-speaker corpus."""
    data = tmp_path / "data"
    shutil.copytree(pipeline.out, data / "q_synth_22k", ignore=shutil.ignore_patterns("wavs"),
                    symlinks=True)
    lines = (pipeline.out / "train_op.txt").read_text().splitlines()
    wav, _idx, ph = lines[0].split("|")
    lines[0] = "%s|%d|%s" % (wav, N_SPK, ph)
    (data / "q_synth_22k" / "train_op.txt").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(res, "DATA", str(data))
    n_table = pipeline.speakers["n_spks"]
    assert max(int(l.split("|")[1]) for l in lines) >= n_table
    with pytest.raises(SystemExit, match="Q voice %d has no entry" % N_SPK):
        res.q_clips(_args(hnr.path), res.hnr_table(hnr.path), lambda k, v: None)


def test_a_missing_22k_clip_is_refused_by_check9_before_it_is_paired(pipeline, hnr, tmp_path,
                                                                      monkeypatch):
    data = tmp_path / "data"
    shutil.copytree(pipeline.out, data / "q_synth_22k", ignore=shutil.ignore_patterns("wavs"),
                    symlinks=True)
    lines = (pipeline.out / "train_op.txt").read_text().splitlines()
    lines[0] = lines[0].replace("/wavs/", "/nowhere/")
    (data / "q_synth_22k" / "train_op.txt").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(res, "DATA", str(data))
    with pytest.raises(SystemExit, match="Q clip .* is missing"):
        res.q_clips(_args(hnr.path), res.hnr_table(hnr.path), lambda k, v: None)


def test_build_corpus_will_not_run_on_a_speaker_table_of_the_wrong_size(pipeline, tmp_path, capsys):
    short = tmp_path / "raw"
    short.mkdir()
    table = json.loads((pipeline.raw / "speakers.json").read_text())
    table["speakers"] = table["speakers"][:-1]
    (short / "speakers.json").write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(SystemExit):
        s2c.cmd_build_corpus(type("A", (), {"raw_root": str(short), "out_dir": str(tmp_path / "o")})())
    assert "89 speakers" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()
