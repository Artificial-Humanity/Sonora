"""The pipeline-stage ratchet: `scripts/pipeline_manifest.py` must match the shells (#26 step 4).

Three times now, a written instrument in this repo turned out never to run — most recently
`qc_verdict.py`, named in a `synth_bank.sh` comment for a month while 695 directed clips
went to the ear with no direction check (issue #24). Each time it was found by hand, and
each time the reason it survived was the same: in `scripts/`, being uninvoked is the normal
state for most of the 100 non-test files there, so an unwired *stage* is indistinguishable
from an operator tool that was never meant to be called.

`tests/test_audit_sampling.py` already guards individual stages, and guards them more
deeply than this file does — it checks the flags without which a stage runs and achieves
nothing, and the one ordering constraint that decides whether a `gate_calibration` round is
worth anything. What it cannot do is notice a stage nobody thought to write a test for.
That is the gap here: **coverage of the set**, iterated from a declaration, in both
directions —

* a stage declared and not invoked fails (the #24 shape);
* a script invoked and not declared fails (so the manifest cannot rot while the shells move);
* a `.sh` under `scripts/` in none of the three categories fails (so the *next* orchestrator
  cannot arrive with its stages unexamined).

The parser is imported from `test_audit_sampling`, not copied: "is this line a call?" has
one definition in this repo and needs to keep having one. It drops comments (mechanism 1)
and `echo`es (mechanism 2), both of which name real scripts in these files.

Source-level and torch-free, so it runs wherever `make test` runs.
"""

import ast
import importlib.util
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "scripts" / "pipeline_manifest.py"


def _load_manifest():
    """Loaded by path: `pyproject.toml` excludes `scripts*` from the package, deliberately
    (see the comment there), so there is no import name to reach the manifest by."""
    spec = importlib.util.spec_from_file_location("sonora_pipeline_manifest", MANIFEST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load_manifest()

# `qc_gate.py` -> `qc_gate.py`. Basenames, because the same script is invoked under three
# different prefixes: `$SONORA/scripts/...` on the host, `/sonora/scripts/...` inside the
# throwaway container, and `/tmp/sonora/scripts/...` in the holdout lane's build copy.
SCRIPT_RE = re.compile(r"([A-Za-z0-9_]+\.(?:py|sh))")


def _commands(rel, invocations_only=True):
    """Shell commands in `rel`, using the repo's ONE definition of "an invocation"."""
    # pylint: disable=import-outside-toplevel
    from test_audit_sampling import _invocations, _shell_commands

    source = _invocations if invocations_only else _shell_commands
    return [cmd for _, cmd in source(rel)]


def _scripts_named(rel, invocations_only=True):
    named = set()
    for cmd in _commands(rel, invocations_only):
        named.update(SCRIPT_RE.findall(cmd))
    return named


def _tracked_shells():
    """Every tracked `.sh` in the repo — not just under `scripts/`.

    The glob was `scripts/*.sh` until 2026-08-12, so "no orchestrator can arrive unexamined"
    was true only of shells in one directory; one written anywhere else was exempt by
    construction. Every `.sh` in this repo happens to live under `scripts/` today, so this
    widening changes no result — it removes the exemption. `check=True` so a missing git
    fails loudly rather than returning an empty list, which would pass by vacuum.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.sh"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return sorted({p for p in out.stdout.split("\0") if p})


ALL_STAGES = [(orch, stage) for orch, spec in M.ORCHESTRATORS.items() for stage in spec["stages"]]
UNWIRED = [
    (orch, script, why)
    for orch, spec in M.ORCHESTRATORS.items()
    for script, why in spec["deliberately_not_invoked"].items()
]


# --- the manifest is not empty ---------------------------------------------------------
#
# Every test below iterates the manifest, so an emptied manifest would collect zero cases
# and report green. That is silent-disarm mode 1 (AGENTS.md § 5b) and it gets its own test.


def test_the_manifest_declares_the_lanes_it_is_supposed_to_cover():
    assert len(M.ORCHESTRATORS) >= 4, \
        "fewer orchestrators than the four container/eval lanes — was a lane dropped rather than retired?"
    assert len(ALL_STAGES) >= 15, f"only {len(ALL_STAGES)} stages declared; the bank pass alone has 14"
    for orch, spec in M.ORCHESTRATORS.items():
        assert spec["stages"], f"{orch} is declared an orchestrator with no stages"
        assert spec["purpose"].strip(), f"{orch} has no purpose recorded"


# --- direction 1: declared, therefore wired -------------------------------------------


@pytest.mark.parametrize("orch,stage", ALL_STAGES, ids=[f"{o}::{s.script}" for o, s in ALL_STAGES])
def test_every_declared_stage_is_actually_invoked(orch, stage):
    """The #24 shape. A stage is WIRED or it is merely WRITTEN ABOUT, and these shells
    explain every stage at length in prose and print each one's re-run command on failure —
    so both a comment and an `echo` name the script without running it."""
    assert (REPO / stage.script).is_file(), f"declared stage does not exist: {stage.script}"
    assert stage.script.startswith("scripts/"), f"declared stage is outside scripts/: {stage.script}"
    assert stage.role.strip(), f"{stage.script} is declared without a role"

    basename = pathlib.PurePosixPath(stage.script).name
    assert basename in _scripts_named(orch), (
        f"{basename} is DECLARED a stage of {orch} but is not invoked by it "
        f"(a comment is not a call, and neither is an echo). Either wire it, or move it to "
        f"that orchestrator's `deliberately_not_invoked` with the reason."
    )
    # The DIRECTORY too, not just the basename. Every comparison in this file collapses to a
    # basename — the shells spell the same script three ways (`$SONORA/…`, `/sonora/…` inside
    # the container, `/tmp/sonora/…` in the holdout build copy) so an absolute prefix cannot
    # be matched. That made the manifest's repo-relative paths decoration: a declaration
    # could name the wrong bucket and still pass. The bucket segment IS present in every
    # spelling, so it is checkable.
    bucket = pathlib.PurePosixPath(stage.script).parent.name
    tail = f"{bucket}/{basename}"
    assert any(tail in cmd for cmd in _commands(orch)), (
        f"{orch} invokes {basename}, but not at the declared path {stage.script} — no "
        f"invocation contains {tail!r}. The manifest names a bucket the shell does not use."
    )


@pytest.mark.parametrize("orch", sorted(M.ORCHESTRATORS), ids=sorted(M.ORCHESTRATORS))
def test_every_declared_sub_orchestrator_is_invoked_and_itself_covered(orch):
    """`synth_bank.sh` runs `eiv_score.sh`, whose output `qc_verdict --eiv` consumes. An
    audit that greps for `.py` does not see it, and a nested lane that stops running takes
    the stage downstream of it with it."""
    for sub in M.ORCHESTRATORS[orch]["invokes_orchestrators"]:
        basename = pathlib.PurePosixPath(sub).name
        assert basename in _scripts_named(orch), f"{orch} no longer invokes {sub}"
        assert sub in M.ORCHESTRATORS, f"{orch} runs {sub}, which is not declared, so its own stages are unchecked"


# --- direction 2: wired, therefore declared -------------------------------------------


@pytest.mark.parametrize("orch", sorted(M.ORCHESTRATORS), ids=sorted(M.ORCHESTRATORS))
def test_no_orchestrator_invokes_a_script_the_manifest_does_not_know_about(orch):
    """Without this the manifest rots one-directionally: it would keep describing the
    pipeline as it was, and a stage added to a shell would be as undeclared — and so as
    unguarded — as if the manifest did not exist."""
    spec = M.ORCHESTRATORS[orch]
    allowed = {pathlib.PurePosixPath(s.script).name for s in spec["stages"]}
    allowed |= {pathlib.PurePosixPath(s).name for s in spec["invokes_orchestrators"]}
    for extra in spec["non_stage_scripts"]:
        # `stages` entries are existence- and location-checked; these were not, so
        # `"setup.py"` sat as a bare basename permanently whitelisting that name in this
        # orchestrator regardless of which setup.py runs. It IS repo-relative (repo root),
        # so hold it to the same standard.
        assert (REPO / extra).is_file(), f"{orch} declares non-stage script {extra}, which does not exist"
    allowed |= {pathlib.PurePosixPath(s).name for s in spec["non_stage_scripts"]}
    allowed |= set(M.STRUCTURAL_HELPERS)
    allowed.add(pathlib.PurePosixPath(orch).name)  # its own usage string / self-reference

    undeclared = sorted(_scripts_named(orch) - allowed)
    assert not undeclared, (
        f"{orch} invokes {undeclared}, which scripts/pipeline_manifest.py does not declare. "
        f"Add each one to `stages` (it is a pipeline stage), `invokes_orchestrators` (it is "
        f"a nested lane), or `non_stage_scripts` (it is a build step) — with a role."
    )


# --- deliberate non-invocation is a decision, and decisions have to survive ------------


def test_the_echo_filter_still_has_a_live_fixture():
    """`deliberately_not_invoked` is the only thing in this suite that would fail if the
    parser stopped dropping `echo`es — every other assertion here wants scripts to be
    found, so a parser that over-matches makes them all pass. If this list ever empties,
    the echo filter is untested and mechanism 2 is back in play."""
    assert UNWIRED, (
        "no deliberately-unwired script left in the manifest, so nothing here would notice "
        "if `_invocations` regressed to counting recovery-hint echoes as calls"
    )


@pytest.mark.parametrize("orch,script,why", UNWIRED, ids=[f"{o}::{s}" for o, s, _ in UNWIRED])
def test_a_deliberately_unwired_script_is_still_referenced_and_still_not_invoked(orch, script, why):
    """Both halves matter. Referenced-and-not-invoked is a recorded decision; unreferenced
    means the decision is stale prose about a file the shell forgot, and invoked means
    someone reversed the decision without reading it — which for `stage_pool` is 652
    unaudited rows into the audition queue."""
    assert len(why) > 40, f"{script} is declared unwired without a usable reason"
    # Declarations are checked against the TREE as well as the shell. Without this, deleting
    # `stage_pool.py` while leaving the echo hint keeps 400 characters of reasoning about the
    # 652-row flood pointing at nothing — and keeps
    # `test_the_echo_filter_still_has_a_live_fixture` reporting the filter as covered.
    assert (REPO / script).is_file(), (
        f"{script} is declared deliberately-unwired but no longer exists; the reason recorded "
        f"for it is now prose about a file the tree forgot"
    )
    basename = pathlib.PurePosixPath(script).name
    assert basename in _scripts_named(orch, invocations_only=False), \
        f"{orch} no longer mentions {basename} at all — this entry now records nothing"
    assert basename not in _scripts_named(orch), (
        f"{orch} now INVOKES {basename}, which the manifest says it deliberately must not:\n"
        f"  {why}\n"
        f"If that decision has been reversed on purpose, move it to `stages` in the same commit."
    )


# --- completeness: no orchestrator arrives unexamined ---------------------------------


def test_every_shell_under_scripts_is_classified_exactly_once():
    """The generalisation of #24. That issue was one stage nobody noticed had stopped
    running; the class is one *orchestrator* nobody noticed had started."""
    shells = _tracked_shells()
    assert len(shells) >= 10, f"only {len(shells)} shells found under scripts/ — is the enumeration working?"

    categories = {
        "ORCHESTRATORS": set(M.ORCHESTRATORS),
        "DYNAMIC_DISPATCH": set(M.DYNAMIC_DISPATCH),
        "NOT_ORCHESTRATORS": set(M.NOT_ORCHESTRATORS),
    }
    declared = set().union(*categories.values())

    missing = sorted(set(shells) - declared)
    assert not missing, (
        f"shells under scripts/ that scripts/pipeline_manifest.py does not classify: {missing}. "
        f"Add each to ORCHESTRATORS (it runs pipeline stages), DYNAMIC_DISPATCH (it picks its "
        f"target at runtime) or NOT_ORCHESTRATORS (it is a helper, a tool, or dead) — with the reason."
    )
    stale = sorted(declared - set(shells))
    assert not stale, f"manifest classifies shells that no longer exist: {stale}"

    for name, members in categories.items():
        for other, others in categories.items():
            if other <= name:
                continue
            overlap = sorted(members & others)
            assert not overlap, f"{overlap} classified as both {name} and {other}"


@pytest.mark.parametrize("path", sorted(M.NOT_ORCHESTRATORS), ids=sorted(M.NOT_ORCHESTRATORS))
def test_a_shell_declared_not_an_orchestrator_says_why(path):
    assert len(M.NOT_ORCHESTRATORS[path]) > 40, f"{path} is dismissed without a reason worth reading"


def test_no_test_module_reaches_into_scripts_by_hand():
    """`SCRIPTS.on_path()` mutates the PROCESS-GLOBAL `sys.path`, so once any module calls
    it every later module inherits every bucket — and a module still inserting a path that
    no longer exists becomes indistinguishable from a migrated one.

    Not hypothetical: `tests/test_bank_consistency.py` was MISSED by #26 step 3 and kept
    `sys.path.insert(0, REPO/"scripts"/"synthesis")` plus a hard `import check_bank`. The
    full suite stayed green purely on **alphabetical collection order** —
    `test_acquisition_lane.py` sorts earlier and left `scripts/stages` on the path for it.
    Running that one file, which is how anyone would debug it, was a hard
    `ModuleNotFoundError`, and a collection error aborts the WHOLE session rather than
    failing one module. So the leakage is exactly what hid the miss.
    """
    # AST, not a text scan: a text scan matched this very docstring, which is the
    # trailing-comment defect in miniature — prose about code is not code.
    offenders = []
    files = [r for r in subprocess.run(
        ["git", "ls-files", "-z", "tests/*.py"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\0") if r]
    assert len(files) >= 20, f"only {len(files)} test modules enumerated — is the listing working?"
    for rel in sorted(files):
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"), rel)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in ("insert", "append"):
                continue
            if "sys.path" not in ast.unparse(node.func):
                continue
            arg = ast.unparse(node).lower()
            if "scripts" in arg:
                offenders.append(f"{rel}:{node.lineno}: {ast.unparse(node)}")
    assert not offenders, (
        "these test modules reach into scripts/ by hand instead of `SCRIPTS.on_path()`, so a "
        "layout change breaks them silently — the suite stays green on collection order alone:\n"
        + "\n".join(offenders)
    )


def test_nothing_outside_a_declared_orchestrator_runs_a_stage():
    """A stage does not have to be launched from a shell, and the manifest only classifies
    shells — so a Python file, a Makefile target or a workflow step could run one and be as
    undeclared, and as unguarded, as before #24. Measured 2026-08-12: nothing does. This is
    what keeps that true; the alternative was a paragraph saying it might not be.
    """
    stage_names = {pathlib.PurePosixPath(s.script).name
                   for spec in M.ORCHESTRATORS.values() for s in spec["stages"]}
    assert stage_names, "no stages declared — nothing to check"
    declared = set(M.ORCHESTRATORS) | set(M.DYNAMIC_DISPATCH)

    candidates = [r for r in subprocess.run(
        ["git", "ls-files", "-z", "*.py", "Makefile", ".github/workflows/*.yml"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\0") if r]
    assert len(candidates) >= 50, f"only {len(candidates)} candidates — is the listing working?"

    # AST for python, comment-stripped text for Makefile/workflows. A first attempt matched
    # `python <stage>` anywhere in the file and reported SEVEN false positives — six were a
    # stage's own usage docstring, and `stage_pool.py:265` is an error message telling the
    # operator to run `qc_gate.py`. That is the printed-hint class again: prose about a
    # command is not the command.
    LAUNCHERS = ("run", "Popen", "call", "check_call", "check_output", "system", "execv", "execvp")
    offenders = []
    for rel in sorted(candidates):
        if rel in declared or rel.startswith("tests/"):
            continue
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        if rel.endswith(".py"):
            try:
                tree = ast.parse(text, rel)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr not in LAUNCHERS:
                    continue
                rendered = ast.unparse(node)
                for stage in sorted(stage_names):
                    if stage in rendered and not rel.endswith(stage):
                        offenders.append(f"{rel}:{node.lineno}: {node.func.attr}(...) launches {stage}")
        else:
            for n, line in enumerate(text.splitlines(), 1):
                bare = line.split("#", 1)[0]
                for stage in sorted(stage_names):
                    if stage in bare and re.search(r"(python|\$PY)\S*\s", bare):
                        offenders.append(f"{rel}:{n}: {bare.strip()}")
    assert not offenders, (
        "these files launch a declared pipeline stage but are not declared orchestrators, so "
        "nothing checks their wiring:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("path", sorted(M.DYNAMIC_DISPATCH), ids=sorted(M.DYNAMIC_DISPATCH))
def test_the_dynamic_dispatch_lanes_still_dispatch_dynamically(path):
    """These are the declared blind spot: static reachability cannot see the export lane
    because `run.sh` takes the script name as `$1`. That exemption is only honest while it
    is true — the moment a target is hardcoded, it is a stage like any other and belongs in
    ORCHESTRATORS where it gets checked."""
    assert len(M.DYNAMIC_DISPATCH[path]) > 40, f"{path} is exempted without a reason"
    hardcoded = sorted(_scripts_named(path) - {pathlib.PurePosixPath(path).name} - set(M.STRUCTURAL_HELPERS))
    assert not hardcoded, (
        f"{path} is declared a dynamic-dispatch wrapper but now names {hardcoded} directly. "
        f"Move it to ORCHESTRATORS and declare its stages."
    )
