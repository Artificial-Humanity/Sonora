"""A declared dependency must appear in the environment record — enforced, not asked for.

WHY THIS FILE EXISTS (2026-08-19, issues #153/#154)
---------------------------------------------------
`environments/README.md` says, in as many words:

    Refresh both after any dependency change, and say so in the commit message —
    a stale snapshot asserting the wrong versions is the failure mode these files
    exist to prevent.

Nothing enforced it. `grep -rn environments/ tests/` returned no matches, so the rule was
prose, and this branch broke it the first time it was tested: `pydantic>=2` went into
`[project.dependencies]`, both snapshots were left alone, and `pydantic==2.13.4` sat
installed in the live host venv while `environments/host-venv.txt` listed neither it nor
`pydantic-core`. Fifteen reviews of the diff did not catch it, because **the defect is not
in the diff** — it is in a file the diff does not touch.

⚠ **THE OBVIOUS RULE IS THE WRONG RULE.** "Every declared dependency appears in
`host-venv.txt`" fails for 13 of 21 today, and correctly so: that file's own header says
*"Deliberately NO torch: the host lane is espeak-free CPU tooling … every model-touching job
runs in the ROCm container."* A guard that condemns a deliberate choice is the over-reach
`test_no_module_keeps_its_own_idea_of_what_counts_as_a_number` had to be narrowed twice to
avoid. So the two rules below are the two that are actually true.
"""

import os
import pathlib
import re
import subprocess
import tomllib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ENV = REPO / "environments"


def _norm(name):
    """A requirement line or dependency spec reduced to its distribution name."""
    return re.split(r"[<>=!~\[ ;]", name, 1)[0].strip().lower().replace("_", "-")


def _declared():
    pj = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return [_norm(d) for d in pj["project"]["dependencies"]]


def _snapshot(name):
    return {_norm(l) for l in (ENV / f"{name}.txt").read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")}


def test_the_snapshots_exist_and_are_not_empty():
    """⚠ A guard whose input has vanished passes silently. This is the falsifier for the two
    below — both are "every X is in Y", which an empty X satisfies for free."""
    assert len(_declared()) >= 10, "pyproject declares almost nothing — is the parse working?"
    for name in ("host-venv", "training-container"):
        assert len(_snapshot(name)) >= 20, f"{name}.txt looks empty or unreadable"


@pytest.mark.parametrize("dep", _declared())
def test_every_declared_dependency_is_recorded_in_the_container_snapshot(dep):
    """The container lane takes every dependency — `pyproject.toml` calls pydantic "CORE …
    it sits on the teacher-synthesis path every container takes". So a name declared and
    absent from that record is either undeclared drift or an unrefreshed snapshot."""
    assert dep in _snapshot("training-container"), (
        f"{dep!r} is in [project.dependencies] and not in environments/training-container.txt. "
        f"Refresh it (README.md § Refreshing) or explain why the container does not take it.")


def test_the_host_snapshot_records_every_declared_dependency_it_actually_has():
    """⚠ THE ONE THAT WOULD HAVE CAUGHT #153, and it is deliberately narrow.

    Asserting the host snapshot equals `uv pip freeze` would fail on any machine carrying an
    unrelated local package — a noisy gate is a gate someone turns off, which this repo has
    already measured. So the claim is only: **a declared dependency that IS installed here
    must be recorded here.** `pydantic` was installed and unrecorded; `torch` is neither, and
    stays legitimately absent.
    """
    venv = REPO / ".venv" / "bin" / "python"
    if not venv.exists():
        pytest.skip("no .venv in this checkout — the host snapshot cannot be compared, and "
                    "that is NOT a pass")
    r = subprocess.run(["uv", "pip", "freeze", "--python", str(venv)],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"`uv pip freeze` unavailable ({r.stderr.strip()[:80]}) — NOT a pass")
    live = {_norm(l) for l in r.stdout.splitlines() if l.strip()}
    recorded = _snapshot("host-venv")
    unrecorded = sorted(d for d in _declared() if d in live and d not in recorded)
    assert not unrecorded, (
        f"these are declared, installed in .venv, and missing from "
        f"environments/host-venv.txt: {unrecorded}. Refresh it with "
        f"`uv pip freeze --python .venv/bin/python`, keeping the header (README.md § Refreshing).")
