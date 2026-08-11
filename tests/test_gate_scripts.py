"""Collect the repo's gate scripts into one pytest entry point.

`pyproject.toml` pointed pytest at `tests/`, which did not exist, while every
real test lived as `scripts/test_*.py` and was never collected — so `make test`
tested nothing. The scripts are standalone programs with their own exit codes
(that is how they are used in pipelines), so they are run here as subprocesses
and judged on those codes rather than being imported. Importing them would drag
torch, a GPU and the litert harness into a plain `make test`.

Anything needing hardware or an external checkpoint is marked `slow` and skipped
when its prerequisite is absent, so the default `pytest -k "not slow"` stays
runnable on any machine.
"""

import os
import re
import subprocess
import sys
import warnings

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
PY = os.path.join(REPO, ".venv", "bin", "python")

# Scripts that need nothing but the repo venv.
FAST = ["test_skill_files.py", "test_text_selection.py"]

# Needs a DATA artifact but not torch — a third category, added 2026-08-09.
#
# `test_doc_claims.py` verifies the notes' corpus numbers against the corpus itself, so it
# needs the filelists under `data/`. Those are NOT tracked: root `.gitignore` carries
# `/data`, and `data/` here is a working directory of derived artifacts. They exist on
# ai-lab-0 and on nothing else, so the harness SKIPS when they are absent.
#
# ⚠ Skipping here and failing loudly inside the script are both correct, and the split is
# the point: "the corpus is not on this machine" is not a finding, while "the corpus is
# here and a document disagrees with it" is. Putting this in FAST — which the first version
# did — turned every laptop into a red build for a fact it had no way to check.
#
# ⚠ THE PREREQUISITES ARE A UNION, NOT AN INTERSECTION — and they were the wrong one.
# `test_doc_claims.py` now carries the prerequisite PER FACT and skips-with-notice the facts
# whose artifact is absent, so it is safe to run wherever ANY corpus is present. ANDing the
# list here meant a host holding v5 and not v6 — a partial mount, a rollback, ai-lab-0
# itself between the two merges — checked nothing at all and reported it as one skip naming
# only the first missing path, discarding the ten v5 facts it could have enforced (#52).
# Coverage was the intersection of every corpus the registry had ever read, and that shrinks
# with every generation: at v7 the gate would run on one machine on a good day.
#
# The list is still exhaustive, because it is now what decides "is anything checkable here"
# and what the skip message names.
DATA_GATED = [
    ("test_doc_claims.py", [
        ("SONORA_CORPUS_V5",
         os.path.join(REPO, "data", "libritts_r_emilia_vat_v5", "derivation_report.json")),
        ("SONORA_CORPUS_V6",
         os.path.join(REPO, "data", "libritts_r_emilia_expressive_vat_v6",
                      "derivation_report.json")),
    ]),
]

# (script, env var naming its prerequisite, default path to probe)
SLOW = [
    ("test_vat_identity.py", "SONORA_PHASE0_CKPT",
     "/data/model-training/sonora/logs/train/ljspeech/runs/"
     "2026-07-11_04-02-23/checkpoints/checkpoint_epoch=199.ckpt"),
    ("test_film_export_gate.py", "SONORA_LITERT_HARNESS",
     "/data/toolchain/litert-conversion"),
]


def _run(script):
    return subprocess.run([PY, os.path.join(SCRIPTS, script)],
                          capture_output=True, text=True, timeout=900)


@pytest.mark.parametrize("script", FAST)
def test_fast_gate(script):
    r = _run(script)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}"


class PartialCoverage(Warning):
    """Raised as a warning when a gate ran against only some of its artifacts.

    Deliberately NOT a `UserWarning`: `pyproject.toml` sets `filterwarnings =
    ["ignore::UserWarning", ...]`, so anything under that root would be swallowed — and a
    coverage reduction nobody can see is the whole defect (#52). Subclassing `Warning`
    directly keeps it in pytest's warnings summary without `-s`.
    """


@pytest.mark.parametrize("script,prereqs", DATA_GATED)
def test_data_gated_gate(script, prereqs):
    present, missing = [], []
    for env_var, default in prereqs:
        target = os.environ.get(env_var, default)
        (present if os.path.exists(target) else missing).append(f"{env_var} ({target})")

    if not present:
        # Nothing checkable. Name EVERY prerequisite, not just the first one found missing:
        # the old message stopped at the first and read as "v6 is absent" on a host that was
        # missing both, which is a different situation with a different remedy.
        pytest.skip(f"{script}: none of its prerequisites is present — " +
                    "; ".join(missing) + ". The corpus artifacts under data/ are untracked "
                    "working files, present only where the corpus was built")
    if missing:
        # Ran, but not on everything. The script prints which FACTS it skipped; this says
        # the run was partial at the level pytest reports, so a green suite cannot be read
        # as full coverage.
        warnings.warn(
            f"{script} ran against a PARTIAL set of artifacts — absent: " +
            "; ".join(missing) + ". The facts reading them were skipped and named in the "
            "script's own output; every other fact was enforced.", PartialCoverage)

    r = _run(script)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}"


def _has_torch():
    r = subprocess.run([PY, "-c", "import torch"], capture_output=True)
    return r.returncode == 0


@pytest.mark.slow
@pytest.mark.parametrize("script,env_var,default", SLOW)
def test_slow_gate(script, env_var, default):
    # These two live in the training container / litert harness venv, which is
    # where torch is. The host venv deliberately does not carry it, so the
    # prerequisite to check is the interpreter's capability, not just the
    # data path.
    if not _has_torch():
        pytest.skip(f"{script}: no torch in {PY} — run inside the "
                    f"training container or the litert harness venv")
    target = os.environ.get(env_var, default)
    if not os.path.exists(target):
        pytest.skip(f"{script}: {env_var} target not present ({target})")
    r = _run(script)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}"


def test_python_is_the_repo_venv():
    """The run-mode rule (owner 2026-08-01): host scripts use .venv/bin/python."""
    assert os.path.exists(PY), (
        f"{PY} is missing. Host scripts run the repo venv directly, not `uv run` "
        "(which binds to pyproject.toml and ignores inline PEP 723 blocks). "
        "Create it with: uv venv && uv pip install --python .venv/bin/python ...")


def test_no_uv_run_in_host_shell_scripts():
    """`uv run` on the host is the retired invocation style; keep it retired."""
    offenders = []
    for root, _dirs, files in os.walk(SCRIPTS):
        for name in files:
            if not name.endswith(".sh"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    if "uv run" in stripped:
                        offenders.append(f"{os.path.relpath(path, REPO)}:{i}")
    assert not offenders, ("`uv run` used on the host in: " + ", ".join(offenders))


def test_no_uv_run_in_python_docstrings():
    """The .sh check above missed where the drift actually lived.

    Review finding D-L1: nineteen host scripts still opened with `Run: uv run …`.
    Nothing executes a docstring, so the shell-script guard never saw them — but a
    docstring is the copy-paste source, which is exactly how a retired invocation
    style comes back. Swept 2026-08-02; this keeps it swept.

    `uv` itself is untouched by this: it still creates and populates the venv
    (`uv pip install --python .venv/bin/python`). Only `uv run` is out, because it
    resolves against pyproject.toml and ignores the inline PEP 723 blocks these
    scripts carry.
    """
    # Matches a PRESCRIPTION only, and the anchor is what does that work.
    # `derive_vat_corpus.py` explains at length why `uv run` is wrong here, and
    # `qc_gate.py` recounts the day it failed — prose about the retired style has
    # to stay legal, or the guard deletes its own rationale. A command sits at the
    # head of its line (after a `Run:`/`Usage:`/`$` lead-in); an explanation
    # mentions it mid-sentence, usually in backticks.
    invocation = re.compile(
        r"^\s*(?:[Rr]un:|Usage:|\$|#)?\s*`?uv run"
        r"(?:\s+--\S+(?:\s+\S+)?)*\s+(?:python\b|[\w./-]+\.py)")
    offenders = []
    for root, _dirs, files in os.walk(SCRIPTS):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if invocation.search(line):
                        offenders.append(f"{os.path.relpath(path, REPO)}:{i}")
    assert not offenders, (
        "`uv run` prescribed in host Python docs: " + ", ".join(offenders)
        + " — use `.venv/bin/python <path>` instead.")
