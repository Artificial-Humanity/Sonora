"""The pipeline-stage ratchet: `scripts/pipeline_manifest.py` must match the shells (#26 step 4).

Three times now, a written instrument in this repo turned out never to run — most recently
`qc_verdict.py`, named in a `synth_bank.sh` comment for a month while 695 directed clips
went to the ear with no direction check (issue #24). Each time it was found by hand, and
each time the reason it survived was the same: in `scripts/`, being uninvoked is the normal
state for most of the 100 non-test `.py` files there (106 tracked, less the 6 gate
scripts), so an unwired *stage* is indistinguishable
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


def sys_path_offenders(rel, text):
    """Sites in `text` that mutate `sys.path` to reach into `scripts/` by hand.

    A FUNCTION, not an inline loop, so the fixtures below can call it on synthetic source
    and assert on the MESSAGE. ⚠ That is the whole point: the previous round verified its
    sibling guard by mutating a real file and reading pytest's `1 failed` as "caught" — but
    the guard was raising `AttributeError` while catching, and a crash and a catch are the
    same colour. Checking a guard's colour does not check the guard.

    Covers four mutators (`insert`/`append`/`extend`, `+=`, slice-assignment) and resolves
    IMPORT aliases both ways: `import sys as s` → `s.path.insert(…)` and `from sys import
    path` → `path.insert(…)` each walked past when only the literal `sys.path` was matched.

    ⚠ RESIDUAL, stated for the same reason its sibling states one: import aliases are the
    only aliases resolved. A local rebinding (`p = sys.path; p.insert(…)`), an
    `importlib`-mediated reference, or a path assembled at runtime is invisible — as is any
    mutation performed by a helper this function does not follow into.
    """
    # ⚠ NOT `except SyntaxError: return []`. The inline version parsed unguarded, so an
    # unparseable module under `tests/` was a loud error; the extraction quietly turned that
    # into a pass, which is a guard losing coverage to a refactor. An unparseable test module
    # is itself a finding.
    try:
        tree = ast.parse(text, rel)
    except SyntaxError as exc:
        return [f"{rel}:{exc.lineno}: does not parse ({exc.msg}) — cannot be checked"]
    sys_aliases, path_aliases = {"sys"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            sys_aliases |= {(a.asname or a.name) for a in node.names if a.name == "sys"}
        elif isinstance(node, ast.ImportFrom) and node.module == "sys":
            path_aliases |= {(a.asname or a.name) for a in node.names if a.name == "path"}
    targets = {f"{a}.path" for a in sys_aliases} | path_aliases

    def touches(expr):
        return any(expr == t or expr.startswith(t + "[") or expr.startswith(t + ".")
                   for t in targets)

    found = []
    for node in ast.walk(tree):
        rendered = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("insert", "append", "extend") and touches(ast.unparse(node.func.value)):
                rendered = ast.unparse(node)
        elif isinstance(node, ast.AugAssign):                       # path += [...]
            if touches(ast.unparse(node.target)):
                rendered = ast.unparse(node)
        elif isinstance(node, ast.Assign):                          # path[:0] = [...]
            if any(touches(ast.unparse(t)) for t in node.targets):
                rendered = ast.unparse(node)
        if rendered and "scripts" in rendered.lower():
            found.append(f"{rel}:{node.lineno}: {rendered}")
    return found


@pytest.mark.parametrize("src,expect", [
    ('import sys\nsys.path.insert(0, "scripts/synthesis")', "sys.path.insert(0,"),
    ('import sys\nsys.path.append("scripts/synthesis")', "sys.path.append("),
    ('import sys\nsys.path.extend(["scripts/synthesis"])', "sys.path.extend("),
    ('import sys\nsys.path += ["scripts/synthesis"]', "sys.path +="),
    ('import sys\nsys.path[:0] = ["scripts/synthesis"]', "sys.path[:0] ="),
    ('import sys as s\ns.path.insert(0, "scripts/synthesis")', "s.path.insert(0,"),
    ('from sys import path\npath.insert(0, "scripts/synthesis")', "path.insert(0,"),
    ('from sys import path as p\np.extend(["scripts/synthesis"])', "p.extend("),
])
def test_every_sys_path_spelling_is_reported_by_name(src, expect):
    """Asserts the OFFENDER STRING names the actual mutation.

    ⚠ An earlier version asserted `"scripts" in out[0] and "t.py:" in out[0]` — both of
    which are the function's own append condition and format string, so it could not fail
    for any input and did not deliver the property its own docstring claimed. The expected
    fragment is now per-spelling, so a guard that reported the wrong site would be caught.
    """
    out = sys_path_offenders("t.py", src)
    assert out, f"not detected: {src!r}"
    assert expect in out[0], f"reported {out[0]!r}, expected it to name {expect!r}"
    assert out[0].startswith("t.py:2:"), f"wrong location: {out[0]!r}"


@pytest.mark.parametrize("src", [
    'import sys\nsys.path.insert(0, "/opt/other")',      # not scripts/
    'import sys\nprint(sys.path)',                       # read, not mutate
    'from scripts_layout import SCRIPTS\nSCRIPTS.on_path()',
    '"""docstring mentioning sys.path.insert(0, "scripts/synthesis")"""',
])
def test_the_sys_path_guard_does_not_fire_on_innocent_code(src):
    """A guard that goes red on correct code gets switched off. The last row is the one that
    matters: a text scan matched this guard's own docstring."""
    assert sys_path_offenders("t.py", src) == []


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
        offenders += sys_path_offenders(rel, (REPO / rel).read_text(encoding="utf-8"))
    assert not offenders, (
        "these test modules reach into scripts/ by hand instead of `SCRIPTS.on_path()`, so a "
        "layout change breaks them silently — the suite stays green on collection order alone:\n"
        + "\n".join(offenders)
    )


# ⚠ The alias set is only as complete as this list, and it omitted the two easiest launchers
# in the stdlib. `getoutput`/`getstatusoutput` and `popen`/`spawn*` all run a command line.
LAUNCHERS = (
    # subprocess
    "run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput",
    # os — the exec/spawn families are CLOSED SETS, so they are completed mechanically
    # rather than sampled. ⚠ The env-passing variants were the ones missing, and passing an
    # environment is exactly what launching a stage looks like here
    # (`SONORA_REPO=… python scripts/stages/…`).
    "system", "popen",
    "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "posix_spawn", "posix_spawnp",
    # runpy — a different shape: in-process, no `python` on a command line at all, which
    # makes it the most likely ACCIDENTAL spelling of "run the stage".
    "run_path", "run_module",
)
# Receivers a launcher may be called on. ⚠ Without this, ANY attribute call whose final name
# is in LAUNCHERS counted — `self.run(...)`, `client.call(...)` — and the first such method to
# appear beside a stage name in a string would turn this guard red for nothing.
LAUNCH_MODULES = ("subprocess", "os", "runpy")


def _argv_runs_a_printer(node):
    """True when a launcher's own argv is `echo`/`printf` — a printed hint, not a launch.

    `subprocess.run(["echo", "docs at scripts/stages/qc_gate.py"])` is a real call to a real
    launcher whose effect is to print prose about a stage. The AST rewrite retired the
    docstring sub-case of the printed-hint class; this is the rest of it.
    """
    if not node.args:
        return False
    first = node.args[0]
    if isinstance(first, (ast.List, ast.Tuple)):
        if not first.elts or not isinstance(first.elts[0], ast.Constant):
            return False
        head = str(first.elts[0].value)
    elif isinstance(first, ast.Constant):
        head = str(first.value)
    else:
        return False
    if not head.strip():
        return False
    # ⚠ NOT `head.split()[0]`. That treats a COMPOUND shell command as if it were argv[0],
    # so a leading `echo` exempted everything after it — verified misses:
    #     os.system("echo starting; SONORA_REPO=. python scripts/stages/qc_gate.py")
    #     subprocess.run("echo hi && python .../qc_gate.py", shell=True)
    # A new silent-disarm hole, opened by the commit that closed the previous one — and the
    # fix is the function that same commit imported into the sibling text branch. Reusing it
    # keeps ONE definition of "printed text" instead of a second, weaker one here.
    # pylint: disable=import-outside-toplevel
    from test_audit_sampling import _without_printed_text

    remaining = _without_printed_text(head).strip()
    # Wholly printed (`echo x`) -> exempt. Partly printed (`echo x; python stage`) -> not.
    return not remaining


def stage_launch_offenders(rel, text, stage_names):
    """Sites in `text` that LAUNCH one of `stage_names` — as opposed to mentioning it.

    Dotted (`subprocess.run`), bare (`from subprocess import run`) and ALIASED
    (`import run as _r`) — the alias is what a real evasion uses, and it walked past the
    first attempt. ⚠ RESIDUAL, stated rather than implied: a target assembled at runtime
    (`_cmd = "scripts/stages/" + "qc_gate.py"`) is invisible to any static check.

    Extracted so the fixtures can assert the MESSAGE. The inline version formatted its
    report with `node.func.attr` while the alias branch admitted `ast.Name`, so it raised
    `AttributeError` at the exact moment it caught an evasion — and the mutation check that
    was supposed to prove it worked only ever read pytest's exit colour.
    """
    found = []
    if rel.endswith(".py"):
        try:
            tree = ast.parse(text, rel)
        except SyntaxError as exc:
            return [f"{rel}:{exc.lineno}: does not parse ({exc.msg}) — cannot be checked"]
        local, module_aliases = set(), set(LAUNCH_MODULES)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in LAUNCH_MODULES:
                local |= {(a.asname or a.name) for a in node.names if a.name in LAUNCHERS}
            elif isinstance(node, ast.Import):
                module_aliases |= {(a.asname or a.name) for a in node.names
                                   if a.name in LAUNCH_MODULES}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                # QUALIFIED by receiver: `subprocess.run`, not `self.run`.
                recv = ast.unparse(node.func.value)
                fn = node.func.attr if (node.func.attr in LAUNCHERS
                                        and recv in module_aliases) else None
            else:
                # bare name only if it was actually IMPORTED from one of those modules
                name = getattr(node.func, "id", None)
                fn = name if name in local else None
            if fn is None:
                continue
            rendered = ast.unparse(node)
            # ⚠ A real launcher whose ARGV is prose is still a printed hint:
            # `subprocess.run(["echo", "docs at scripts/stages/qc_gate.py"])`. The AST
            # rewrite retired the docstring sub-case, not this one.
            if _argv_runs_a_printer(node):
                continue
            for stage in sorted(stage_names):
                if stage in rendered and not rel.endswith(stage):
                    found.append(f"{rel}:{node.lineno}: {fn}(...) launches {stage}")
    else:
        # ⚠ Makefile / workflow lines. This branch had ZERO fixture coverage in either
        # direction, and it still fired on prose: `run: echo "then run python
        # scripts/stages/qc_gate.py"` was reported as a launch. The AST branch retired the
        # printed-hint class for Python only. Reuse the repo's one definition of printed
        # text rather than growing a second — `_without_printed_text` already handles
        # `echo`, `printf`, redirections, braces and `VAR=` prefixes.
        # pylint: disable=import-outside-toplevel
        from test_audit_sampling import _strip_trailing_comment, _without_printed_text

        for n, line in enumerate(text.splitlines(), 1):
            bare = _without_printed_text(_strip_trailing_comment(line))
            for stage in sorted(stage_names):
                if stage in bare and re.search(r"(python|\$PY)\S*\s", bare):
                    found.append(f"{rel}:{n}: {bare.strip()}")
    return found


@pytest.mark.parametrize("src", [
    'import subprocess\nsubprocess.run(["python", "scripts/stages/qc_gate.py"])',
    'import subprocess as _s\n_s.Popen(["python", "scripts/stages/qc_gate.py"])',
    'from subprocess import run\nrun(["python", "scripts/stages/qc_gate.py"])',
    'from subprocess import run as _r\n_r(["python", "scripts/stages/qc_gate.py"])',
    'from os import system as _sy\n_sy("python scripts/stages/qc_gate.py")',
    'import subprocess\nsubprocess.getoutput("python scripts/stages/qc_gate.py")',
    'import os\nos.popen("python scripts/stages/qc_gate.py")',
])
def test_every_launcher_spelling_is_reported_by_name(src):
    """Asserts the MESSAGE. The aliased and bare spellings each walked past a version of this
    guard, and once it did catch them it raised `AttributeError` while formatting the
    report — which pytest renders as the same `1 failed` as a real catch."""
    out = stage_launch_offenders("scripts/tools/x.py", src, {"qc_gate.py"})
    assert out, f"not detected: {src!r}"
    assert "launches qc_gate.py" in out[0], out


@pytest.mark.parametrize("src", [
    '"""Usage: python scripts/stages/qc_gate.py --campaign-dir X"""',
    'print("  Run:  .venv/bin/python scripts/stages/qc_gate.py")',
    'import subprocess\nsubprocess.run(["ls"])',
    # a REAL launcher whose argv is prose — the rest of the printed-hint class
    'import subprocess\nsubprocess.run(["echo", "docs at scripts/stages/qc_gate.py"])',
    # a launcher NAME on an unrelated receiver
    'class C:\n    def go(self):\n        self.run("scripts/stages/qc_gate.py")',
    'def run(x):\n    pass\nrun("see scripts/stages/qc_gate.py for docs")',
])
def test_the_launch_guard_does_not_fire_on_prose(src):
    """The seven false positives an earlier text version produced were six usage docstrings
    and one error message telling the operator to run the stage. The last three rows are the
    ones the AST rewrite did NOT retire: a real launcher printing prose, and any object with
    a `.run`/`.call` method."""
    assert stage_launch_offenders("scripts/tools/x.py", src, {"qc_gate.py"}) == []


# --- the Makefile / workflow branch, which had no fixtures at all ----------------------


@pytest.mark.parametrize("rel,line", [
    ("Makefile", "\tSONORA_REPO=. python scripts/stages/qc_gate.py --campaign-dir $(C)"),
    (".github/workflows/ci.yml", "        run: python scripts/stages/qc_gate.py"),
])
def test_a_real_launch_in_a_makefile_or_workflow_is_reported(rel, line):
    """⚠ The message assertion must not restate the detector's own append condition. The
    text branch appends only when `stage in bare` and appends `bare.strip()`, so
    `"qc_gate.py" in out[0]` held by construction once `assert out` passed — the
    cannot-fail defect again, in the fixture written one commit after it was fixed in the
    sibling. Assert the LOCATION and the launcher token instead, neither of which the
    membership condition determines."""
    out = stage_launch_offenders(rel, line, {"qc_gate.py"})
    assert out, f"not detected in {rel}: {line!r}"
    assert out[0].startswith(f"{rel}:1:"), f"wrong location: {out[0]!r}"
    assert "python" in out[0], f"report does not show the launch: {out[0]!r}"


@pytest.mark.parametrize("rel,line", [
    (".github/workflows/ci.yml", '        run: echo "then run python scripts/stages/qc_gate.py"'),
    ("Makefile", '\t@echo "use python scripts/stages/qc_gate.py"'),
    ("Makefile", "\t# python scripts/stages/qc_gate.py  (retired)"),
    (".github/workflows/ci.yml", "        # see python scripts/stages/qc_gate.py"),
])
def test_the_text_branch_does_not_fire_on_printed_prose(rel, line):
    """⚠ This branch had ZERO coverage in either direction and still fired on prose: the
    AST rewrite retired the printed-hint class for PYTHON only. It now runs the line through
    the repo's one definition of printed text (`_strip_trailing_comment` +
    `_without_printed_text`) rather than growing a second one. `Makefile`,
    `.github/workflows/ci.yml` and `claude-review.yml` all contain exactly these shapes."""
    assert stage_launch_offenders(rel, line, {"qc_gate.py"}) == []


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
    offenders = []
    for rel in sorted(candidates):
        if rel in declared or rel.startswith("tests/"):
            continue
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        offenders += stage_launch_offenders(rel, text, stage_names)
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
