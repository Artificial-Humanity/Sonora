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
from scripts_layout import SCRIPTS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES = os.path.join(REPO, "scripts", "gates")
PY = os.path.join(REPO, ".venv", "bin", "python")

# Scripts that need nothing but the repo venv.
#
# ⚠ `test_doc_links.py` belongs here and NOT in DATA_GATED, even though half its job is
# cross-repo. Its in-repo half needs only this checkout, and its sibling half skips-and-prints
# when a sibling is absent — the state on every CI runner — so a bare clone is green by design
# rather than by luck. It also never FAILS on an inbound link: those live in the sibling's
# files, and a check no commit here can turn green is one everybody learns to ignore.
FAST = ["test_skill_files.py", "test_text_selection.py", "test_doc_links.py"]

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
# ⚠ THE LIST HAS TO BE EXHAUSTIVE UNDER OR SEMANTICS, AND HAND-WRITTEN IT WAS NOT — it
# named 2 of the 7 artifacts the registry reads (v5's report and v6's report; not v5's
# train/val filelists, not v4's, not the holdout). With AND that was merely redundant, since
# any missing entry made the whole thing skip. With OR it decides "is anything checkable
# here", so a host holding only v4, or only the holdout, was told nothing is checkable and
# skipped facts it could have enforced — #52 reappearing on the corpora the list forgot.
#
# READ FROM THE REGISTRY, NOT RESTATED. Every fact already carries its own `artifacts`, and
# `unreadable()` is the function the script itself uses; a second copy here is the fork this
# repo keeps paying for (`_app_csv_fields` reads the app's CSV_FIELDS for the same reason).
# Adding a fact for a new corpus now extends this automatically.
#
# ⚠ NO ENV-VAR OVERRIDE, because there is nothing to override. The first version of this
# derived a `SONORA_ARTIFACT_*` name per path and a comment said it was "kept so a host can
# point the gate at a corpus mounted elsewhere". `scripts/gates/test_doc_claims.py` reads **no
# environment variable at all** — `V5`, `V6`, `V4` and `HOLDOUT` are module constants — so
# setting one moved this list and left the gate reading the original path, i.e. the skip
# logic and the script would have disagreed about which corpus was present. The comment
# described a capability that was never built. Prerequisites are now bare paths; if a real
# override is ever wanted it belongs in the gate script first, and this list will follow it
# for free.
#
# ⚠ LOADED UNDER A NAME OF ITS OWN, via the loader `tests/test_doc_claims_registry.py`
# already owns. `import test_doc_claims` is the spelling that file's docstring explicitly
# forbids — pytest would try to collect the gate script as a test module and run its
# module-level code under a second identity — and doing it here was worse than the case it
# warns about, because this runs at COLLECTION time rather than inside a test.
def _registry_artifacts():
    from test_doc_claims_registry import _load_gate           # noqa: E402
    seen, out = set(), []
    for fact in _load_gate().FACTS:
        for p in fact["artifacts"]:
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


DATA_GATED = [("test_doc_claims.py", _registry_artifacts())]

# (script, env var naming its prerequisite, default path to probe)
SLOW = [
    ("test_vat_identity.py", "SONORA_PHASE0_CKPT",
     "/data/model-training/sonora/logs/train/ljspeech/runs/"
     "2026-07-11_04-02-23/checkpoints/checkpoint_epoch=199.ckpt"),
    ("test_film_export_gate.py", "SONORA_LITERT_HARNESS",
     "/data/toolchain/litert-conversion"),
]

# Needs torch and NOTHING ELSE — a fourth category, added 2026-08-21 (#197).
#
# ⚠ `test_vat_dim_seams.py` WAS IN NO LIST AT ALL, so it got neither a run nor a skip: the
# suite was green, the gate did not run, and nothing said so. That is AGENTS.md §5b's exact
# subject, with the twist that this one never entered the enumeration to begin with — and it
# is described as MANDATORY in two places (`tests/test_edge_truncation.py:504` calls it
# "mandatory pre-flight", and `scripts/tools/merge_expressive_registers.py:487` prints it as
# step 2 of the post-merge checklist). ⚠ The suite's only other references to it `read_text()`
# its source and assert things about its CONTENTS. A text-scan over a gate is not the gate.
#
# ⚠ IT CANNOT GO IN FAST: it imports torch, which the host venv deliberately does not carry
#     .venv/bin/python scripts/gates/test_vat_dim_seams.py
#     ModuleNotFoundError: No module named 'torch'
# It is a container-side gate. Nor does it fit SLOW, whose tuple names an external checkpoint
# or harness path to probe — this script builds its own fixtures in a tempdir and needs no
# artifact at all. Its only prerequisite is an interpreter that has torch, so forcing it into
# SLOW would mean inventing an env var and a path that do not exist, and a probe that passes
# for a reason unrelated to the prerequisite is the defect this file keeps meeting.
TORCH_ONLY = ["test_vat_dim_seams.py"]


def _run(script):
    # SKIP, NOT ERROR, WHEN THE REPO VENV IS ABSENT. These shell out to `.venv/bin/python`
    # per the run-mode rule, and a checkout that has never had `uv venv` run in it — every
    # CI runner, including the review lane's — produced a bare `FileNotFoundError` here
    # rather than a skip. "The interpreter these scripts are specified to run under is not
    # on this machine" is the same class of fact as "the corpus is not on this machine",
    # which the FAST/DATA split above already treats as a skip and not a finding.
    #
    # ⚠ It is NOT the same as `test_python_is_the_repo_venv`, which asserts the venv exists
    # and must keep failing where that is a real defect. See the guard on that test.
    if not os.path.exists(PY):
        pytest.skip(f"{PY} is absent — this checkout has no repo venv, so the gate "
                    f"scripts cannot be run as specified (run-mode rule, AGENTS.md).")
    return subprocess.run([PY, os.path.join(GATES, script)],
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
    for target in prereqs:
        # Named relative to the repo: the absolute paths are long enough that a skip message
        # listing seven of them buries the one fact it is trying to convey.
        (present if os.path.exists(target) else missing).append(os.path.relpath(target, REPO))

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
    """-> True / False / None, where None means THERE IS NO INTERPRETER TO ASK.

    Returning `False` for a missing venv collapsed two causes into one, and `test_slow_gate`
    then printed the skip reason for the wrong one: "no torch in .../.venv/bin/python — run
    inside the training container or the litert harness venv" names a remedy that cannot fix
    an absent interpreter. The careful message `_run` carries never fired for these tests,
    because `_has_torch()` is reached first and short-circuits past it.

    Three-valued for the same reason `qc_verdict` is: an absent measurement and a negative
    measurement are different facts, and the repair differs.
    """
    if not os.path.exists(PY):
        return None
    r = subprocess.run([PY, "-c", "import torch"], capture_output=True)
    return r.returncode == 0


@pytest.mark.slow
@pytest.mark.parametrize("script,env_var,default", SLOW)
def test_slow_gate(script, env_var, default):
    # These two live in the training container / litert harness venv, which is
    # where torch is. The host venv deliberately does not carry it, so the
    # prerequisite to check is the interpreter's capability, not just the
    # data path.
    torch = _has_torch()
    if torch is None:
        # `is None`, not falsy: naming torch here would send someone to install a 2 GB wheel
        # into an interpreter that does not exist.
        pytest.skip(f"{script}: {PY} is absent — this checkout has no repo venv, so there "
                    f"is no interpreter to ask about torch (run-mode rule, AGENTS.md).")
    if torch is False:
        pytest.skip(f"{script}: no torch in {PY} — run inside the "
                    f"training container or the litert harness venv")
    target = os.environ.get(env_var, default)
    if not os.path.exists(target):
        pytest.skip(f"{script}: {env_var} target not present ({target})")
    r = _run(script)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}"


@pytest.mark.slow
@pytest.mark.parametrize("script", TORCH_ONLY)
def test_torch_only_gate(script):
    """Runs where torch is; SKIPS WITH NOTICE where it is not. Never silently absent.

    The skip is the whole point of adding it here: "this interpreter has no torch" is the
    same class of fact as "the corpus is not on this machine", which FAST/DATA_GATED already
    treat as a printed skip rather than a finding. Before this entry existed the script
    produced neither — it was simply not run, and the suite reported green.
    """
    torch = _has_torch()
    if torch is None:
        # `is None`, not falsy — see `_has_torch`. Naming torch for an absent interpreter
        # sends someone to install a 2 GB wheel into a python that is not there.
        pytest.skip(f"{script}: {PY} is absent — this checkout has no repo venv, so there "
                    f"is no interpreter to ask about torch (run-mode rule, AGENTS.md).")
    if torch is False:
        pytest.skip(f"{script}: no torch in {PY} — this is a CONTAINER-SIDE gate; run it "
                    f"inside the training container. It needs no data artifact, only torch.")
    r = _run(script)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}"


def test_every_gate_script_is_enumerated_exactly_once():
    """⚠ THE GUARD THAT MAKES #197 UNREPEATABLE, rather than fixed once.

    `scripts/gates/` held SEVEN scripts and this file named SIX. The missing one was in
    neither the run lists nor the skip lists, so it contributed no test id, no skip line and
    no failure — it was invisible in exactly the way a deleted test is invisible.

    ⚠ `tests/test_gate_scripts.py`'s own docstring records that this repo has ALREADY been
    bitten by an uncollected `scripts/test_*.py`; this file exists because of that incident,
    and a seventh script still slipped past it. A list maintained by hand beside a directory
    that grows is the same bug with a new filename, so the directory is now the authority and
    the lists are checked against it.

    Adding a gate script therefore forces a decision about how it runs. That is the intent:
    the failure mode is not "someone chose the wrong list", it is "nobody chose at all".
    """
    on_disk = {f for f in os.listdir(GATES)
               if f.startswith("test_") and f.endswith(".py")}
    # ⚠ AN EMPTY DIRECTORY MUST BE RED, NOT SILENT. With `on_disk` empty every assertion
    # below is vacuously true and this test reports green having checked nothing — which is
    # precisely the defect it was written for, one level up. 7 measured 2026-08-21.
    assert len(on_disk) >= 6, (
        f"only {len(on_disk)} gate script(s) found in {GATES}, from the 7 measured on "
        f"2026-08-21. Every check below is vacuous at this count — suspect the directory "
        f"or the listing before lowering this floor.")
    enumerated = {}
    for name, bucket in (("FAST", FAST), ("TORCH_ONLY", TORCH_ONLY)):
        for s in bucket:
            enumerated.setdefault(s, []).append(name)
    for s, _prereqs in DATA_GATED:
        enumerated.setdefault(s, []).append("DATA_GATED")
    for s, _env, _default in SLOW:
        enumerated.setdefault(s, []).append("SLOW")

    missing = sorted(on_disk - set(enumerated))
    assert not missing, (
        "gate script(s) in scripts/gates/ that NO list names, so they are neither run nor "
        "skipped — a green suite says nothing about them:\n  " + "\n  ".join(missing) +
        "\nPut each in FAST, DATA_GATED, SLOW or TORCH_ONLY."
    )
    phantom = sorted(set(enumerated) - on_disk)
    assert not phantom, (
        "list(s) name a gate script that is not in scripts/gates/ — the entry runs nothing:\n"
        "  " + "\n  ".join(phantom))
    twice = sorted(f"{s} in {'+'.join(b)}" for s, b in enumerated.items() if len(b) > 1)
    assert not twice, ("gate script(s) named by more than one list, so one run is "
                       "unaccounted for:\n  " + "\n  ".join(twice))


def test_python_is_the_repo_venv():
    """The run-mode rule (owner 2026-08-01): host scripts use .venv/bin/python.

    ⚠ THIS ONE IS NOT GUARDED, DELIBERATELY, AND IT IS THE REASON THE OTHERS CAN BE.
    `_run` skips when the venv is absent because "the interpreter is not on this machine"
    is not a finding about the code. That is only defensible while SOMETHING still says
    the venv is required — otherwise a host that quietly lost its venv would go green
    across the whole file, which is the failure mode the skip is one step away from.

    It is a HOST assertion. It fails on a bare CI checkout by design, and the fix there is
    to create the venv (this repo's own `ci.yml` does, before running anything) rather than
    to soften the rule. A lane that cannot create one is a lane that cannot run the gate
    scripts as specified, and should be told so once, here, instead of five times in
    tracebacks that look like defects in the scripts.
    """
    assert os.path.exists(PY), (
        f"{PY} is missing. Host scripts run the repo venv directly, not `uv run` "
        "(which binds to pyproject.toml and ignores inline PEP 723 blocks). "
        "Create it with: uv venv && uv pip install --python .venv/bin/python ...")


def test_no_uv_run_in_host_shell_scripts():
    """`uv run` on the host is the retired invocation style; keep it retired."""
    offenders = []
    for root, _dirs, files in os.walk(os.path.join(REPO, "scripts")):
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
    for root, _dirs, files in os.walk(os.path.join(REPO, "scripts")):
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


def test_a_missing_interpreter_is_not_reported_as_a_missing_torch(monkeypatch):
    """`_has_torch()` returned `False` for an absent venv, collapsing two causes into one.

    `test_slow_gate` reaches it before `_run`, so the careful "this checkout has no repo
    venv" message never fired for these tests — they printed "no torch in .../.venv/bin/python
    — run inside the training container", naming a remedy that cannot create an interpreter.
    Three-valued for the same reason `qc_verdict` is: an absent measurement and a negative
    one are different facts with different repairs.
    """
    monkeypatch.setattr(sys.modules[__name__], "PY", os.path.join(REPO, "no", "such", "python"))
    assert _has_torch() is None, "an absent interpreter must not answer a question about torch"


def test_the_data_gated_prerequisites_are_the_registry_s_own_artifacts():
    """The list decides "is anything checkable here" under OR semantics, so a short one
    silently narrows coverage (#52, reopened on the corpora a hand-written list forgot).
    Derived, so it cannot drift; asserted here so the derivation cannot quietly return
    nothing."""
    from test_doc_claims_registry import _load_gate
    expected = {p for fact in _load_gate().FACTS for p in fact["artifacts"]}
    _script, prereqs = DATA_GATED[0]
    assert set(prereqs) == expected and len(prereqs) == len(expected)
    assert len(prereqs) >= 7, "the registry reads at least 7 artifacts; a shorter list is a hole"
