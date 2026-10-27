"""Phase 1S Step 1c-ii's planner and the Orpheus renderer's timing fields.

`scripts/tools/render_1cii.py` writes one campaign dir per 1c-ii cell for
`synth_bank.sh`, reusing render_1ci's input checks against the same pinned text
bank and anchors. These tests run plan/batched/render against fake inputs (the
registered hashes and weights dirs swapped for fakes), so they need neither
/data nor a GPU. The synth_orpheus timing test stubs the engine modules the
same way test_renderer_pins_and_timing does.
"""
import ast
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts_layout import REPO, SCRIPTS

SCRIPTS.on_path()

import probe_text_bank  # noqa: E402
import ref_select  # noqa: E402
import render_1ci as r1  # noqa: E402
import render_1cii as r2  # noqa: E402

N = probe_text_bank.BANK_SIZE
CELLS = ["M4-s3_qwen_base", "F6-s2_qwen_base", "ryan_qwen_custom",
         "aiden_qwen_custom", "jess_orpheus"]


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _text_rows():
    return [{"id": "1ci-%04d" % i, "text": "This is narration line number %d of the bank." % i}
            for i in range(1, N + 1)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fake text bank, anchors and weights wired into render_1ci/render_1cii."""
    rows = _text_rows()
    tdir = tmp_path / "text"
    tdir.mkdir()
    (tdir / "lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sha = probe_text_bank.canonical_sha256(rows)
    (tdir / "meta.json").write_text(json.dumps({"sha256": sha}))
    monkeypatch.setattr(r1, "TEXT_BANK", str(tdir / "lines.jsonl"))
    monkeypatch.setattr(r1, "TEXT_META", str(tdir / "meta.json"))
    monkeypatch.setattr(r1, "TEXT_SHA256", sha)

    adir = tmp_path / "anchors"
    (adir / "clips").mkdir(parents=True)
    pins, manifest = [], []
    for aid, gender, blob in (("M4-s3", "Male", b"male clip"),
                              ("F6-s2", "Female", b"female clip")):
        (adir / "clips" / (aid + ".wav")).write_bytes(blob)
        pins.append((aid, _sha(blob)))
        manifest.append({"id": aid, "file": "clips/%s.wav" % aid, "gender": gender,
                         "text": "The anchor %s says this." % aid})
    (adir / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    monkeypatch.setattr(r1, "ANCHOR_DIR", str(adir))
    monkeypatch.setattr(r1, "ANCHORS", tuple(pins))
    monkeypatch.setattr(ref_select, "REF_FILE_ROOT", str(tmp_path) + "/")

    weights = {}
    for engine in ("qwen_base", "qwen_custom"):
        d = tmp_path / "weights" / engine
        (d / "speech_tokenizer").mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"model-" + engine.encode())
        (d / "speech_tokenizer" / "model.safetensors").write_bytes(b"tokenizer")
        weights[engine] = {
            "dir": str(d),
            "model_safetensors_sha256": _sha(b"model-" + engine.encode()),
            "speech_tokenizer_sha256": _sha(b"tokenizer")}
    monkeypatch.setattr(r2, "QWEN_WEIGHTS", weights)
    orpheus = tmp_path / "weights" / "orpheus"
    orpheus.mkdir()
    monkeypatch.setattr(r2, "ORPHEUS_WEIGHTS", str(orpheus))

    git = {"commit": "c0ffee", "tree_clean": True}
    monkeypatch.setattr(r1, "tool_commit", lambda: dict(git))
    return {"root": tmp_path, "out": tmp_path / "out", "rows": rows, "git": git,
            "weights": weights}


def _plan(env, *extra):
    return r2.main(["plan", "--out-root", str(env["out"]), *extra])


def _bank(env, cell):
    return json.loads((env["out"] / cell / "bank.json").read_text())


def _meta(env, cell):
    return json.loads((env["out"] / cell / "plan_meta.json").read_text())


# --- plan: the five cells and what their banks carry ---------------------------

def test_plan_writes_the_five_cells_with_their_registered_shapes(env):
    _plan(env)
    assert sorted(p.name for p in env["out"].iterdir()) == sorted(CELLS)
    for cell in CELLS:
        bank = _bank(env, cell)
        assert bank["version"] == "1cii-v1" and bank["campaign"] == "1cii-" + cell
        assert len(bank["lines"]) == N and "batch_size" not in bank
        assert [ln["seed"] for ln in bank["lines"]] == [1_000_000 + i for i in range(1, N + 1)]
        assert [ln["text"] for ln in bank["lines"]] == [r["text"] for r in env["rows"]]

    base = _bank(env, "M4-s3_qwen_base")["lines"][0]
    assert base["direction"] == {"design": ""}
    assert base["ref_file"]["id"] == "M4-s3"
    assert base["ref_file"]["text"] == "The anchor M4-s3 says this."
    assert _bank(env, "F6-s2_qwen_base")["lines"][0]["ref_file"]["id"] == "F6-s2"

    for cell, speaker in (("ryan_qwen_custom", "ryan"), ("aiden_qwen_custom", "aiden")):
        line = _bank(env, cell)["lines"][0]
        assert line["direction"] == {"speaker": speaker, "instruct": "Calm and measured."}
        assert "ref_file" not in line

    orpheus = _bank(env, "jess_orpheus")["lines"]
    assert all(ln["direction"] == {"voice": "jess", "render_text": ln["text"]}
               for ln in orpheus)
    assert "ref_file" not in orpheus[0]


def test_plan_meta_records_the_verified_engine_pins(env):
    _plan(env)
    pin = _meta(env, "M4-s3_qwen_base")["engine_pin"]
    assert pin["wheel"] == "qwen-tts==0.1.1"
    assert pin["model_safetensors_sha256"] == _sha(b"model-qwen_base")
    assert pin["speech_tokenizer_sha256"] == _sha(b"tokenizer")
    assert pin["weights_dir"] == env["weights"]["qwen_base"]["dir"]
    meta = _meta(env, "jess_orpheus")
    assert meta["engine_pin"]["weights_dir"] == r2.ORPHEUS_WEIGHTS
    assert meta["anchor"] is None and meta["cell"] == "jess_orpheus"


def test_plan_refuses_weights_that_do_not_match_the_pin(env):
    Path(env["weights"]["qwen_custom"]["dir"], "model.safetensors").write_bytes(b"swapped")
    with pytest.raises(SystemExit, match="not the pinned"):
        _plan(env)
    assert not env["out"].exists() or not list(env["out"].glob("*/bank.json"))


def test_plan_refuses_a_missing_orpheus_weights_dir(env, monkeypatch, tmp_path):
    monkeypatch.setattr(r2, "ORPHEUS_WEIGHTS", str(tmp_path / "no-orpheus"))
    with pytest.raises(SystemExit, match="Orpheus weights dir"):
        _plan(env)


def test_the_banks_pass_check_bank(env):
    import check_bank
    _plan(env)
    for cell in CELLS:
        path = str(env["out"] / cell / "bank.json")
        assert check_bank.check(path) == [], cell
        assert check_bank.check_refs(path) == [], cell
        assert check_bank.main(["check_bank.py", path]) == 0, cell


# --- batched: the throughput cell ---------------------------------------------

def test_batched_plans_a_100_line_cell_with_its_batch_size_recorded(env):
    r2.main(["batched", "--out-root", str(env["out"]), "--batch-size", "8"])
    bank = _bank(env, r2.BATCHED_CELL)
    assert bank["batch_size"] == 8 and len(bank["lines"]) == 100
    assert [ln["text"] for ln in bank["lines"]] == [r["text"] for r in env["rows"][:100]]
    assert bank["lines"][0]["ref_file"]["id"] == "M4-s3"
    meta = _meta(env, r2.BATCHED_CELL)
    assert meta["batch_size"] == 8 and "batched_of" in meta
    # the measured cells are not planned as a side effect
    assert sorted(p.name for p in env["out"].iterdir()) == [r2.BATCHED_CELL]


def test_batched_refuses_a_batch_size_of_one(env):
    with pytest.raises(SystemExit, match="at least 2"):
        r2.main(["batched", "--out-root", str(env["out"]), "--batch-size", "1"])


def test_render_refuses_a_tampered_batched_batch_size(env):
    r2.main(["batched", "--out-root", str(env["out"]), "--batch-size", "4"])
    meta_path = env["out"] / r2.BATCHED_CELL / "plan_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["batch_size"] = 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="batch_size"):
        r2.main(["render", r2.BATCHED_CELL, "--out-root", str(env["out"])])


def test_render_prints_the_batched_cells_command_after_its_checks(env, capsys):
    r2.main(["batched", "--out-root", str(env["out"]), "--batch-size", "4"])
    capsys.readouterr()
    assert r2.main(["render", r2.BATCHED_CELL, "--out-root", str(env["out"])]) == 0
    out = capsys.readouterr().out
    assert "synth_bank.sh" in out and "printed only" in out


# --- render: the launch gate ---------------------------------------------------

def test_render_refuses_a_tampered_bank(env):
    _plan(env)
    path = env["out"] / "ryan_qwen_custom" / "bank.json"
    bank = json.loads(path.read_text())
    bank["lines"][0]["text"] = "A different line entirely."
    path.write_text(json.dumps(bank))
    with pytest.raises(SystemExit, match="not the bank these inputs"):
        r2.main(["render", "ryan_qwen_custom", "--out-root", str(env["out"])])


def test_render_refuses_engine_pin_drift(env):
    _plan(env)
    Path(env["weights"]["qwen_base"]["dir"], "model.safetensors").write_bytes(b"swapped")
    with pytest.raises(SystemExit, match="not the pinned"):
        r2.main(["render", "M4-s3_qwen_base", "--out-root", str(env["out"])])


def test_render_of_an_unknown_cell_is_refused(env):
    with pytest.raises(SystemExit, match="no cell"):
        r2.main(["render", "nobody_qwen_base", "--out-root", str(env["out"])])


def test_status_reports_an_unrendered_cell(env, capsys):
    _plan(env)
    capsys.readouterr()
    assert r2.main(["status", "jess_orpheus", "--out-root", str(env["out"])]) == \
        r1.PARTIAL_EXIT
    assert "0 of %d" % N in capsys.readouterr().out


# --- the render-affecting classification (render_1ci's test shape, both lists) -

def test_every_render_affecting_path_is_tracked_here():
    tracked = set(subprocess.run(["git", "-C", str(REPO), "ls-files"],
                                 capture_output=True, text=True, check=True).stdout.split())
    # this worktree's own new files are tracked once committed; accept on-disk too
    missing = [p for p in r2.RENDER_AFFECTING_PATHS
               if p not in tracked and not (REPO / p).is_file()]
    assert not missing, missing
    assert len(r2.RENDER_AFFECTING_PATHS) == len(set(r2.RENDER_AFFECTING_PATHS))


def _module_level_imports(path):
    names, todo = [], list(ast.parse(path.read_text(encoding="utf-8")).body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
        todo += [c for c in ast.iter_child_nodes(node) if isinstance(c, ast.stmt)]
    return names


def _repo_file(module):
    parts = module.split(".")
    for base in (REPO / "scripts" / "lib", REPO / "scripts" / "stages",
                 REPO / "scripts" / "tools", REPO):
        f = base.joinpath(*parts).with_suffix(".py")
        if f.is_file():
            return f
    return None


def test_the_render_affecting_paths_cover_every_repo_module_they_import():
    listed = set(r2.RENDER_AFFECTING_PATHS)
    seen, todo, unlisted = set(), [REPO / p for p in listed if p.endswith(".py")], []
    assert len(todo) >= 10, "the enumeration found too few files to mean anything"
    while todo:
        f = todo.pop()
        rel = f.relative_to(REPO).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        if rel not in listed:
            if rel not in r2.NOT_RENDER_AFFECTING:
                unlisted.append(rel)
            continue
        for module in _module_level_imports(f):
            hit = _repo_file(module)
            if hit is not None:
                todo.append(hit)
            if "." in module:
                init = _repo_file(module.split(".")[0] + ".__init__")
                if init is not None:
                    todo.append(init)
    assert not unlisted, "imported by a render-affecting file but in neither list: %s" % unlisted
    assert {"scripts/lib/qwen_load.py", "scripts/tools/render_1ci.py"} <= seen


def test_every_script_synth_bank_names_is_classified():
    src = "".join((REPO / rel).read_text(encoding="utf-8")
                  for rel in ("scripts/stages/synth_bank.sh", "scripts/container_env.sh"))
    named = set(re.findall(r"scripts/(?:stages|lib|tools)?/?[A-Za-z0-9_]+\.(?:py|sh)", src))
    named = {n.replace("scripts//", "scripts/") for n in named}
    assert len(named) >= 10, named
    unclassified = sorted(n for n in named
                          if n not in r2.RENDER_AFFECTING_PATHS
                          and n not in r2.NOT_RENDER_AFFECTING)
    assert not unclassified, unclassified


def test_the_registered_constants_are_the_spec_values():
    assert [c["name"] for c in r2.CELLS] == CELLS
    assert r2.BATCHED_CELL == "M4-s3_qwen_base_batched" and r2.BATCHED_LINES == 100
    assert r2.QWEN_WEIGHTS["qwen_base"]["model_safetensors_sha256"] == \
        "38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c"
    assert r2.QWEN_WEIGHTS["qwen_custom"]["model_safetensors_sha256"] == \
        "38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137"
    assert r2.LANE == "Neutral" and r2.SEED_BASE == r1.SEED_BASE


@pytest.mark.skipif(not os.path.isfile(r1.TEXT_BANK), reason="needs the real text bank")
def test_the_real_inputs_verify():
    """Standing check on the real /data inputs: text bank, anchors, weights on disk.

    The safetensors hashes are verified by engine_pin at plan/render time (and
    against fakes above); re-hashing ~8 GB on every suite run would tax every
    run for a fact the launch gate already owns, so here only presence is
    asserted."""
    assert len(r1.load_text()) == N
    assert [a["id"] for a in r1.load_anchors()] == ["M4-s3", "F6-s2"]
    for w in r2.QWEN_WEIGHTS.values():
        assert os.path.isfile(os.path.join(w["dir"], "model.safetensors"))
        assert os.path.isfile(os.path.join(w["dir"], "speech_tokenizer",
                                           "model.safetensors"))
    assert r2.engine_pin("orpheus")["weights_dir"] == r2.ORPHEUS_WEIGHTS


# --- synth_orpheus: throughput fields (stubbed engine, no GPU) -----------------

class _FakeTensor:
    def __init__(self, arr=None):
        self.arr = np.asarray(arr if arr is not None else np.zeros(4), dtype=np.float32)

    def squeeze(self, *_a):
        return _FakeTensor(self.arr.squeeze())

    def unsqueeze(self, *_a):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def to(self, *_a, **_k):
        return self

    def numpy(self):
        return self.arr


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


@pytest.fixture
def orpheus(monkeypatch):
    """synth_orpheus imported against stub torch/transformers/snac modules."""
    calls = []

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    torch = _mod("torch",
                 manual_seed=lambda s: calls.append(("seed", s)),
                 tensor=lambda *a, **k: _FakeTensor(),
                 cat=lambda parts, dim=0: _FakeTensor(),
                 no_grad=lambda: _NoGrad(),
                 bfloat16="bfloat16",
                 cuda=types.SimpleNamespace(synchronize=lambda: calls.append(("sync",))))
    monkeypatch.setitem(sys.modules, "torch", torch)

    class _Seq:
        def tolist(self):
            frame = [0, 4096, 8192, 12288, 16384, 20480, 24576]
            return [128257] + [128266 + c for _ in range(4) for c in frame]

    class _Model:
        def to(self, *_a):
            return self

        def eval(self):
            return self

        def generate(self, ids, **kw):
            calls.append(("generate", kw.get("max_new_tokens")))
            return [_Seq()]

    class _Tok:
        def __call__(self, prompt, return_tensors=None):
            calls.append(("tokenize", prompt))
            return types.SimpleNamespace(input_ids=_FakeTensor())

    transformers = _mod(
        "transformers",
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=lambda *a, **k: _Model()),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tok()))
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class _Snac:
        def to(self, *_a):
            return self

        def eval(self):
            return self

        def decode(self, layers):
            calls.append(("decode", len(layers)))
            return _FakeTensor(np.zeros(2 * 24000))

    monkeypatch.setitem(sys.modules, "snac", _mod(
        "snac", SNAC=types.SimpleNamespace(from_pretrained=lambda *a, **k: _Snac())))
    sys.modules.pop("synth_orpheus", None)
    try:
        yield importlib.import_module("synth_orpheus"), calls
    finally:
        sys.modules.pop("synth_orpheus", None)


def test_orpheus_rows_carry_throughput_fields(orpheus, monkeypatch, tmp_path):
    mod, calls = orpheus
    ticks = iter([10.0, 14.0])
    monkeypatch.setattr(mod, "time",
                        types.SimpleNamespace(monotonic=lambda: next(ticks)))
    bank = tmp_path / "bank.json"
    text = "This is a narration line long enough to render."
    bank.write_text(json.dumps({
        "version": "t1", "campaign": "t",
        "lines": [
            {"id": "o0", "engine": "orpheus", "seed": 7, "text": text,
             "direction": {"voice": "tara", "render_text": text}},
            {"id": "o1", "engine": "orpheus", "seed": 1000001, "text": text,
             "direction": {"voice": "jess", "render_text": text}}]}))
    out = tmp_path / "audio"
    monkeypatch.setattr(sys, "argv", [mod.__file__, "--bank", str(bank), "--out", str(out)])
    mod.main()
    rows = [json.loads(x) for x in (out / "orpheus_manifest.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["o1"], "the banned voice must be skipped untimed"
    row = rows[0]
    assert row["ref_condition_seconds"] == 0.0
    assert row["render_seconds"] == 4.0 and row["audio_seconds"] == 2.0
    assert row["rtf"] == 2.0 and row["intended_gender"] == "Female"
    assert [c[0] for c in calls] == ["tokenize", "seed", "generate", "decode", "sync"], calls
