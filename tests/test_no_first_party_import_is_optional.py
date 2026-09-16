"""A test module that skips itself takes its whole file down, and reports one line.

⚠⚠ THE MEASUREMENT THIS EXISTS FOR. Same file, same commit, same machine, 2026-09-11:

    .venv/bin/python -m pytest tests/test_license_wall.py   ->  50 passed
    .venv/bin/pytest            tests/test_license_wall.py  ->   1 skipped

Fifty assertions became the word "skipped", because `python -m pytest` puts the working
directory on `sys.path` and the console script does not. In a full run the imports worked for a
second accidental reason — nine modules anchored `sys.path` as a side effect of being imported
first — which `pytest-randomly` is free to reorder.

`conftest.py` removes both accidents. **This is the guard that says so when it stops working**,
because the failure mode is silent by construction: a module-level `importorskip` that misses
reports ONE skipped item, not the fifty it swallowed, and "1 skipped" in a green run looks
exactly like a deliberate environmental skip.

⚠ THE DISCRIMINATOR IS WHAT MAKES THIS SAFE TO ASSERT. Six in-repo targets legitimately cannot
import here — `matcha.cli` needs matplotlib, five `matcha.models`/`matcha.onnx` modules need
torch, and the repo venv deliberately has neither (AGENTS.md §3). So the rule is not "every
target must import". It is: **a target may only fail on something that is NOT ours.** A missing
third-party package is what `importorskip` is for; a missing FIRST-PARTY module means the path
is wrong or a file moved, and that is the defect this file catches.
"""
import ast
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _importorskip_targets():
    """Every `pytest.importorskip("...")` argument in the suite, read from the AST.

    ⚠ Enumerated from disk, never listed here (#247): a hand-kept list beside a directory goes
    stale the moment a module is added, and this one would go stale toward passing.
    """
    found = {}
    for path in sorted(REPO.glob("tests/*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "importorskip" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                found.setdefault(node.args[0].value, set()).add(path.name)
    return found


def _is_ours(name):
    """True when `name` resolves to a file in this repo rather than to a dependency.

    ⚠ THIS ANSWERS "IS THERE A FILE", NOT "IS THIS OURS" (#446). A first-party module that was
    RENAMED resolves to no file, so it classifies as external and is never probed — the guard
    narrows its own population silently, which is the vacuous pass it exists to prevent, one
    level up. `_EXTERNAL_OPTIONALS` below is what closes that: a target that is neither a file
    here nor a named dependency is a target naming nothing, and that is the rename symptom.
    """
    parts = name.split(".")
    if (REPO.joinpath(*parts).with_suffix(".py").exists()
            or REPO.joinpath(*parts, "__init__.py").exists()):
        return True
    return bool(list(REPO.glob(f"scripts/*/{parts[0]}.py")))


# ⚠ A CLOSED LIST, DELIBERATELY, and the one place in this file that is not derived from disk.
# It cannot be derived: `pyproject.toml` declares torch, matplotlib, numpy, soundfile and pysbd,
# but NOT `ai_edge_litert` (the LiteRT harness, installed with the data) or `pyloudnorm` —
# measured, so "declared as a dependency" would flag two legitimate externals as missing.
# Adding a name here is therefore a deliberate act with a reason beside it, which is the
# property that makes a hand list acceptable where a globbed one is not.
_EXTERNAL_OPTIONALS = {
    "torch": "the repo venv deliberately excludes it (AGENTS.md §3)",
    "matplotlib": "same, and only `matcha.cli` needs it",
    "numpy": "declared, but optional for the tests that guard on it",
    "soundfile": "declared; audio-only paths",
    "librosa": "declared, and heavy — it drags numba/llvmlite, so a lean container that "
               "never resamples legitimately has no librosa",
    "pysbd": "declared; sentence splitting in the book lane",
    "pyloudnorm": "undeclared; loudness measurement",
    "ai_edge_litert": "undeclared; the LiteRT harness venv lives with the data",
    # ⚠ TOP-LEVEL NAMES ONLY. Both lookups key on `name.split(".")[0]`, so a dotted entry
    # here is DEAD — it matches nothing and silently covers nothing. `pyarrow.parquet` was
    # listed beside this one until #478, with a reason saying the submodule "is the actual
    # import site", which reads as though the entry were doing work. It was not: the
    # `pyarrow` key already covers every `pyarrow.*` target. An unreachable entry with a
    # confident reason is worse than no entry, because it answers the question a reader
    # came here to ask.
    "pyarrow": "declared in the `dataprep` EXTRA, not in `dependencies` — nothing on the "
               "training or inference path reads parquet, so a container legitimately has "
               "no pyarrow and `convert_hifi_tts` is a workstation tool",
}


def test_every_target_names_something_that_exists():
    """⚠ A RENAMED IN-REPO MODULE WOULD OTHERWISE LEAVE THE POPULATION SILENTLY (#446).

    `_is_ours` asks whether a FILE exists. Rename `matcha/data/license_wall.py` and the target
    stops being "ours", stops being probed, and the test that `importorskip`s it skips forever —
    with this guard still green, because the thing it would have checked is no longer in its
    population. Measured: `_is_ours('matcha.data.license_wall_MOVED')` is False.

    So every target must be one of two things: a file in this repo, or a name on
    `_EXTERNAL_OPTIONALS` with a stated reason. Anything else names nothing.
    """
    targets = _importorskip_targets()
    assert targets, "no targets found — this test would pass over nothing"
    orphans = sorted(n for n in targets
                     if not _is_ours(n) and n.split(".")[0] not in _EXTERNAL_OPTIONALS)
    assert not orphans, (
        "these `importorskip` targets are neither a file in this repo nor a named external "
        f"optional, so they name nothing and their tests skip forever: {orphans}. If one was "
        "renamed, fix the target; if it is a new dependency, add it to _EXTERNAL_OPTIONALS "
        "with the reason it may legitimately be absent")


def test_the_enumeration_and_the_classifier_both_work():
    """⚠ THE CONTROL, and it is not optional. The test below asserts a NEGATIVE — "nothing
    failed" — which an empty enumeration or an all-external classifier satisfies silently. That
    is the vacuous pass `test_doc_claims_registry` was built to end, in the file written to end
    a different instance of it.
    """
    targets = _importorskip_targets()
    assert len(targets) > 20, f"the AST scan found only {len(targets)} targets; it is broken"
    ours = {n for n in targets if _is_ours(n)}
    theirs = set(targets) - ours
    assert ours, "the classifier called every target external, so the test below checks nothing"
    assert theirs, "the classifier called every target ours, so it is not discriminating"
    assert "torch" in theirs, "torch must classify as external or the discriminator is inverted"
    assert "matcha.data.license_wall" in ours, "an in-repo module classified as external"


def _probe_import_in_a_clean_interpreter(name):
    """Import ONE name in a fresh, isolated interpreter. Returns None, or [type, missing].

    ⚠⚠ ONE NAME PER PROCESS, AND THE VERSION THAT BATCHED THEM WAS WRONG IN THE EXACT WAY THIS
    FILE EXISTS TO CATCH. It imported all 31 targets in a single interpreter, in sorted order.
    The script modules under `scripts/` anchor the repo root themselves when imported — that is
    AGENTS.md §5c's "one repo-root anchor per entry point" — so `book_ingest`, arriving early in
    the alphabet, put the root on `sys.path` and every `matcha.*` target after it then imported
    no matter what `conftest.py` did. Measured: with conftest's root anchor commented out, the
    batched probe reported `matcha.data.license_wall` as importing FINE.

    So the probe reproduced, inside itself, the accumulation bug it was built to detect: state
    from an earlier import deciding a later one. Twice now in this file — first in-process, then
    in-subprocess. **The only arrangement with no accumulation is one import per interpreter.**

    ⚠ `-I` isolates: no cwd on `sys.path`, no user site, no environment. `conftest.py` is loaded
    BY LOCATION rather than imported, so finding it costs no `sys.path` entry — otherwise the
    bootstrap would supply the very anchor being measured.
    """
    probe = (
        "import json, sys, importlib, importlib.util\n"
        f"_spec = importlib.util.spec_from_file_location('_probe_conftest', {str(REPO / 'conftest.py')!r})\n"
        "assert _spec and _spec.loader, 'conftest.py could not be loaded — nothing anchors the suite'\n"
        "_m = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_m)  # its sys.path side effect IS the subject\n"
        "try:\n"
        f"    importlib.import_module({name!r}); print(json.dumps(None))\n"
        "except ModuleNotFoundError as e:\n"
        "    print(json.dumps(['ModuleNotFoundError', e.name]))\n"
        "except Exception as e:\n"
        "    print(json.dumps([type(e).__name__, str(e)]))\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", probe], cwd=str(REPO),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (
        f"the probe interpreter failed before reporting on {name!r}, so nothing is measured: "
        f"{r.stderr[-600:]}")
    return json.loads(r.stdout)


def test_no_first_party_import_is_optional():
    """An in-repo `importorskip` target must import, or fail on something that is not ours."""
    ours = sorted(n for n in _importorskip_targets() if _is_ours(n))
    imported, excused, broken = [], [], []
    for name in ours:
        err = _probe_import_in_a_clean_interpreter(name)
        if err is None:
            imported.append(name)
        elif (err[0] == "ModuleNotFoundError" and err[1]
              and err[1].split(".")[0] in _EXTERNAL_OPTIONALS):
            # ⚠⚠ EXCUSED ONLY BY THE NAMED LIST, NEVER BY "not ours" (#446, second reopen).
            # Testing `not _is_ours(e.name)` fails for a BARE name: rename
            # `scripts/lib/schemas.py` and `_is_ours('schemas')` goes False, so the three files
            # that import it were excused exactly as though torch were missing — and they would
            # have skipped in silence. `_is_ours` cannot answer this, because the question is
            # "may this legitimately be absent", and a file that has been deleted looks
            # identical to a dependency that was never installed.
            #
            # So `_EXTERNAL_OPTIONALS` is now the single definition of "may be absent", read
            # here and by `test_every_target_names_something_that_exists`. That also settles the
            # derivation question I raised: the list is not derivable, but it need only exist
            # ONCE.
            excused.append(f"{name} (needs {err[1]})")
        else:
            broken.append(f"{name} -> {err[0]} {err[1]!r}")

    assert imported, (
        "no in-repo target imported at all in a clean interpreter, so `excused` cannot be "
        f"trusted either — the path is broken for everything. excused={excused}")
    assert not broken, (
        "these in-repo modules do not import in a clean interpreter, so every test that "
        "`importorskip`s them SKIPS and its file reports one line instead of its assertions: "
        f"{broken}")


def test_this_module_does_not_importorskip_itself():
    """The guard must not be able to vanish the way the thing it guards can.

    ⚠ CHECKED ON THE AST, NOT THE TEXT. The first version searched this file for the string
    "pytest.importorskip" — which appears in three docstrings and in the matcher above, because
    the subject of this module IS that call. It failed on its own prose. A text search answers
    "does this word occur"; the claim is "is this call made", and only the tree knows.
    """
    tree = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "importorskip"]
    assert not calls, (
        f"this file calls importorskip at line(s) {[n.lineno for n in calls]}, so it can skip "
        "itself and take the guard with it")
