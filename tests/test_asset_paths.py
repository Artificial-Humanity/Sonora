"""Every in-repo path a script builds from `__file__` must actually point at the thing.

This guard exists for one failure mode, and it is the expensive one. A script that
resolves its data relative to its own location — `os.path.join(os.path.dirname(
os.path.abspath(__file__)), "director_skills")` — keeps importing perfectly after the file
moves one directory deeper or shallower. It fails when something *reads* the path, which
for the synthesis lane is minutes to hours into a GPU render, and for `register_lexicon.json`
is a bank built without the register vocabulary rather than a bank that failed to build.

Measured 2026-08-12: **3 files build a path to `director_skills/`** and **4 to
`register_lexicon.json`**, spread across `lib/`, `tools/` and `gates/` — so one layout change
touches several different depths at once. (An earlier version of this docstring said
"eleven files including `audition/app/main.py`". That was the count of files MENTIONING the
name, most of them in prose, and `main.py` is not one of the readers at all — it derives the
vocabulary and says so.)

**What is asserted:** if a path expression's final literal segment names something that
exists somewhere in this repo, the path the code builds must resolve to it. Both named
constants and paths used inline are checked.
No allow-list, and nothing to keep in sync: a constant naming a runtime *output* has a tail
that exists nowhere in the tree and is skipped, and a constant naming a repo asset cannot be
skipped by accident because the tail is what selects it.

Source-level and torch-free. Deliberately does NOT import the modules under test — several
pull in torch, and importing to learn a path would make this guard un-runnable exactly where
it matters.
"""

import ast
import os
import pathlib
import re
import subprocess

import pytest
# NOTE: this guard enumerates `scripts/**/*.py` ITSELF (see SCRIPTS below), not via
# `scripts_layout.SCRIPTS` — different populations: that resolver knows seven bucket
# directories, this needs every tracked .py. Importing it here only shadowed the name.

REPO = pathlib.Path(__file__).resolve().parent.parent

# `os.path.join(os.path.dirname(os.path.abspath(__file__)), "x")` and the pathlib spellings.
_PATH_CALLS = {"join", "dirname", "abspath", "realpath", "expanduser", "normpath", "Path"}


class Unresolvable(Exception):
    """The expression depends on something this evaluator deliberately does not model."""


def _tracked_basenames():
    # `-z`: `git ls-files` C-quotes non-ASCII and leaves a space unquoted, so splitting
    # on newlines/whitespace shreds both into fragments that match nothing.
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True)
    names = {}
    for rel in out.stdout.split("\0"):
        if not rel:
            continue
        p = pathlib.PurePosixPath(rel)
        names.setdefault(p.name, []).append(rel)
        for parent in p.parents:
            if parent.name:
                names.setdefault(parent.name, []).append(str(parent))
    return names


def _tracked_paths():
    """Every repo-relative path `git ls-files` knows, plus each of their parent directories.

    Parents are included because an asset target is often a DIRECTORY —
    `scripts/assets/director_skills` — which `ls-files` never lists on its own.
    """
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True)
    paths = set()
    for rel in out.stdout.split("\0"):
        if not rel:
            continue
        p = pathlib.PurePosixPath(rel)
        paths.add(rel)
        for parent in p.parents:
            if parent.name:
                paths.add(str(parent))
    return paths


def _forgotten_paths(tracked_paths):
    """Paths that exist, are NOT tracked, and are NOT ignored — i.e. someone forgot `git add`.

    ⚠ THE DISCRIMINATOR IS `--exclude-standard`, AND IT IS THE WHOLE DESIGN. A runtime output
    is untracked *because it is gitignored*; a forgotten file is untracked because nobody
    added it. Only the second is a defect, and conflating them would make this guard red on
    every generated bank — a guard that goes red on correct code gets switched off, which is
    the outcome this file's docstring exists to avoid.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z", "--others", "--exclude-standard"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    paths = set()
    for rel in out.stdout.split("\0"):
        if not rel:
            continue
        paths.add(rel)
        # A directory counts as forgotten only if git carries NOTHING in it — otherwise it is
        # a tracked directory that merely happens to contain an untracked file.
        for parent in pathlib.PurePosixPath(rel).parents:
            if parent.name and str(parent) not in tracked_paths:
                paths.add(str(parent))
    return paths


TRACKED = _tracked_basenames()
TRACKED_PATHS = _tracked_paths()
FORGOTTEN = _forgotten_paths(TRACKED_PATHS)


def _eval(node, here, env):
    """Evaluate a path expression with `__file__` bound to `here`. Raises Unresolvable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return str(here)
        if node.id in env:
            return env[node.id]
        raise Unresolvable(node.id)
    if isinstance(node, ast.Attribute):
        # `Path(...).parent`, `.parents[n]` handled below via Subscript
        base = _eval(node.value, here, env)
        if node.attr == "parent":
            return str(pathlib.PurePath(base).parent)
        raise Unresolvable(f".{node.attr}")
    if isinstance(node, ast.Subscript):
        # `pathlib.Path(__file__).resolve().parents[2]`
        value, sl = node.value, node.slice
        if isinstance(value, ast.Attribute) and value.attr == "parents" and isinstance(sl, ast.Constant):
            base = _eval(value.value, here, env)
            return str(pathlib.PurePath(base).parents[sl.value])
        raise Unresolvable("subscript")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return os.path.join(_eval(node.left, here, env), _eval(node.right, here, env))
    if isinstance(node, ast.Call):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name in ("resolve", "absolute"):
            if not isinstance(node.func, ast.Attribute):
                raise Unresolvable(f"bare {name}()")
            return _eval(node.func.value, here, env)
        if name not in _PATH_CALLS:
            raise Unresolvable(f"{name}()")
        args = [_eval(a, here, env) for a in node.args]
        if name == "join":
            if not args:
                raise Unresolvable("join()")
            return os.path.join(*args)
        if name == "dirname":
            if not args:
                raise Unresolvable("dirname()")
            return os.path.dirname(args[0])
        # ⚠ `Path("a", "b")` is `a/b`, NOT `a`. Returning `args[0]` discarded the tail that
        # SELECTS the asset, so `Path(parents[2], "director_skills")` resolved to
        # `…/scripts/synthesis` — a tail that exists — and PASSED at a deliberately wrong
        # depth. A false green on exactly the mutation this file exists to catch.
        if name == "Path":
            if not args:
                raise Unresolvable("Path()")
            return os.path.join(*args)
        # abspath / realpath / normpath / expanduser: identity is right for ONE argument.
        # As a zero-arg METHOD (`Path(x).expanduser()`) there are no `node.args` at all, and
        # `args[0] if args else ""` returned "" — a RELATIVE result, so the guard reported
        # "built at the wrong depth" about code that was correct. A guard that goes red on
        # correct code gets switched off, which is the outcome this file's docstring is
        # written to avoid. Raise instead: a skipped constant costs coverage, a wrong one
        # costs the guard.
        if len(args) != 1:
            raise Unresolvable(f"{name}() with {len(args)} arg(s)")
        return args[0]
    raise Unresolvable(type(node).__name__)


def _path_constants(rel):
    """(name, lineno, resolved_path) for every assignment in `rel` that resolves to a path.

    Function bodies are walked too, and that is not thoroughness for its own sake.
    `book_ingest.py`'s register-lexicon load is function-local AND wrapped in
    `except Exception: return []`, so a path built at the wrong depth does not raise — it
    returns an empty controlled vocabulary and every bank after it is built without one.
    A module-level-only walk misses exactly the site where failing loudly was already
    given up on.
    """
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"), rel)
    here = REPO / rel
    env, out = {}, []

    def visit(nodes):
        for node in nodes:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            # ONLY `__file__`-rooted expressions, directly or through a name that is
            # itself `__file__`-rooted. Seeding `env` from bare string literals instead
            # would let `X = "foo"; Y = join(X, "main.py")` resolve to a path this repo
            # never builds, and `main.py` is a real basename here — a false positive that
            # would get this whole guard switched off.
            rooted = "__file__" in ast.dump(node.value) or any(
                isinstance(n, ast.Name) and n.id in env and n.id != target.id
                for n in ast.walk(node.value)
            )
            if not rooted:
                continue
            try:
                value = _eval(node.value, here, env)
            except (Unresolvable, IndexError, TypeError, AttributeError, KeyError):
                continue
            env[target.id] = value
            out.append((target.id, node.lineno, value))

    visit(tree.body)  # module level first, so later scopes can use its constants
    visit([n for n in ast.walk(tree) if not any(n is b for b in tree.body)])

    # ⚠ A path that is USED rather than NAMED was invisible until 2026-08-12, and one of the
    # two assets this file exists for was unguarded at one of its three sites because of it:
    #
    #   LEXICON = json.load(open(os.path.join(
    #       os.path.dirname(os.path.abspath(__file__)), "register_lexicon.json")))["lexicon"]
    #
    # The `Assign` target is `LEXICON` and its value is a `json.load(...)` call →
    # `Unresolvable("load()")` → skipped. `make_teacher_ab_bank.py`'s `SKILL_DIR` on the same
    # file WAS caught, so the file looked covered. Verified: mutating that `dirname` to a
    # wrong depth left the suite green. So every join/Path expression is evaluated wherever
    # it appears, not only where it is bound to a name.
    seen = {(lineno, value) for _, lineno, value in out}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if fname not in ("join", "Path"):
            continue
        # Rooted directly OR through a name already resolved from `__file__` — the whole
        # point is `os.path.join(_SONORA_REPO, "scripts", "assets", "x.json")`, whose own
        # node never mentions `__file__`. Requiring it here is what left this site
        # unguarded on the first attempt at this fix.
        if not ("__file__" in ast.dump(node) or any(
            isinstance(n, ast.Name) and n.id in env for n in ast.walk(node)
        )):
            continue
        try:
            value = _eval(node, here, env)
        except (Unresolvable, IndexError, TypeError, AttributeError, KeyError):
            continue
        if (node.lineno, value) not in seen:
            seen.add((node.lineno, value))
            out.append((f"<{fname}() at line {node.lineno}>", node.lineno, value))
    return out


SCRIPTS = sorted(
    p for p in subprocess.run(
        ["git", "ls-files", "-z", "scripts/*.py", "scripts/**/*.py"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\0") if p
)


def test_the_guard_actually_asserts_something():
    """The floor has to count FALSIFIABLE assertions, not files enumerated.

    `len(SCRIPTS)` was the wrong population. Measured on 2026-08-12, all four numbers from
    one run of this test's own body:

        scripts enumerated                 106
        constants resolving to a repo path  32
        of which SELF-ANCESTOR              8   <- cannot fail, by construction
        FALSIFIABLE                        24   <- what the floor below counts

    Most scripts build no in-repo path at all, so a floor on the file count says nothing.
    And a self-ancestor (`HERE = dirname(abspath(__file__))`) resolves to a directory that
    contains the file being parsed — counting those as coverage is silent-disarm mode 1
    wearing a number.

    ⚠ These four were stale within a round: an earlier version said "107 green cases of
    which 93 assert nothing" and "six … self-ancestors", written before the walk was
    extended to unnamed path expressions. The floor was re-derived and the prose beside it
    was not, so the two disagreed about the same measurement. If you change what is
    counted, re-derive every number that describes the count — not just the assertion.
    """
    assert TRACKED, "git ls-files returned nothing"
    assert len(SCRIPTS) >= 80, f"only {len(SCRIPTS)} scripts enumerated — is the listing working?"

    falsifiable = []
    for rel in SCRIPTS:
        here = (REPO / rel).resolve()
        for name, lineno, value in _path_constants(rel):
            if pathlib.PurePosixPath(value).name not in TRACKED:
                continue                      # a runtime output, not an asset
            target = pathlib.Path(value)
            if not target.is_absolute():
                target = REPO / target
            if target.resolve() in here.parents:
                continue                      # self-ancestor: true by construction
            falsifiable.append(f"{rel}:{lineno} {name}")
    # ⚠ 24, RE-DERIVED. An earlier version of this said "the 13 measured on 2026-08-12"
    # with a floor of 12 — a number carried over from a review comment written before the
    # walk was extended to unnamed path expressions, and a floor that permitted losing HALF
    # the coverage silently. The floor is the measurement: this is a ratchet, so any drop in
    # falsifiable coverage is meant to be noticed, including a legitimate one.
    assert len(falsifiable) >= 24, (
        f"falsifiable asset paths dropped to {len(falsifiable)} from the 24 measured on\n"
        f"2026-08-12. Either the evaluator stopped resolving a spelling, or coverage really\n"
        f"shrank — if the latter, re-derive and lower this number deliberately:\n"
        + "\n".join(falsifiable)
    )


@pytest.mark.parametrize("rel", SCRIPTS, ids=SCRIPTS)
def test_in_repo_asset_paths_resolve(rel):
    """If the tail names something in this repo, the built path must point at it — IN THE INDEX.

    ⚠ `Path.exists()` IS A QUESTION ABOUT THE WORKING TREE, AND TRACKED-NESS IS A QUESTION
    ABOUT THE INDEX. They disagree in exactly the case that matters: an untracked file sitting
    on disk answers `True`, so the guard confirmed "this path resolves for me, right now" when
    the property worth guarding is "this path resolves for anyone who clones this."

    Measured 2026-08-20 (#207), on the mistake that produced #201:

        git rm --cached -q workflow/scripts/merge_floor.py   # untracked, still on disk
        .venv/bin/python -m pytest -q
        -> 1391 passed, 10 skipped

    The whole suite passed while `merge_branch.sh` imported a module the repo did not
    contain. On a fresh checkout that gate dies `FLOOR_MODULE_MISSING` and no branch can
    merge. The resolution is now `git ls-files`, and "exists but untracked" gets its OWN
    message: it is a different mistake from "does not exist" and sends the reader somewhere
    different — `git add`, not "fix the path".
    """
    broken = []
    for name, lineno, value in _path_constants(rel):
        tail = pathlib.PurePosixPath(value).name
        target = pathlib.Path(value)
        if not target.is_absolute():
            target = REPO / target
        try:
            rel_target = pathlib.PurePosixPath(target.resolve().relative_to(REPO)).as_posix()
        except ValueError:
            rel_target = None   # outside this checkout; `ls-files` cannot speak for it

        # (A) FORGOTTEN — runs FIRST and is deliberately NOT gated on the tail heuristic.
        # ⚠ The tail heuristic reads `git ls-files`, so untracking a file removes its basename
        # from TRACKED and the check below skips it as "a runtime output". Measured while
        # fixing this: `git rm --cached scripts/assets/register_lexicon.json` left the suite
        # GREEN through an index-aware version of (B), because (B) never got to run. A guard
        # whose selector goes blind to precisely the file that went missing is the disarm
        # shape in AGENTS.md §5b, so the question is asked here on its own terms.
        if rel_target is not None and rel_target in FORGOTTEN:
            broken.append(
                f"  {rel}:{lineno}  {name} = {value}\n"
                f"      EXISTS BUT IS NOT TRACKED ({rel_target}), and is not gitignored.\n"
                f"      It resolves for you and for nobody who clones this repo — on a fresh\n"
                f"      checkout this path is absent. `Path.exists()` cannot see this: it asks\n"
                f"      about the WORKING TREE, and this is a question about the INDEX.\n"
                f"      Fix with `git add {rel_target}`, NOT by changing the path."
            )
            continue

        # (B) WRONG DEPTH — the original check, unchanged in meaning.
        if tail not in TRACKED:
            continue  # names nothing in the tree — a runtime output, not an asset
        if not target.exists():
            broken.append(
                f"  {rel}:{lineno}  {name} = {value}\n"
                f"      does not exist. `{tail}` IS in the repo, at: {sorted(set(TRACKED[tail]))[:4]}\n"
                f"      so this path is built at the wrong depth for where {rel} now lives."
            )
    assert not broken, f"{rel} builds path(s) that do not resolve:\n" + "\n".join(broken)


# ─────────────────────────────────────────────────────────────────────────────────────────
# Shell scripts — the half of #201 that fixing `Path.exists()` alone would still have missed
# ─────────────────────────────────────────────────────────────────────────────────────────
# ⚠ THE ENUMERATION ABOVE IS `scripts/*.py` AND `scripts/**/*.py`, SO `workflow/scripts/*.sh`
# WAS ENTIRELY OUTSIDE IT — and `merge_branch.sh`'s `from merge_floor import rides` is
# precisely the dependency that broke. Both gaps in #207 are real and independent: an
# index-aware check that never looks at shell scripts still reports green on #201.
#
# These scripts reach Python through an embedded heredoc that puts the script's own directory
# on `sys.path`:
#
#     UNSETTLED="$(python3 - "$REPO_SLUG" "$BRANCH" "$_SCRIPT_DIR" <<'PY'
#     sys.path.insert(0, sys.argv[3])
#     from merge_floor import rides as _rides
#
# so a sibling `.py` that is on disk but not in the index imports fine here and is absent on
# a fresh clone. That is the same working-tree-vs-index confusion as above, in the language
# the AST evaluator cannot read.

SHELL_SCRIPTS = sorted(
    p for p in subprocess.run(
        ["git", "ls-files", "-z", "*.sh"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\0") if p
)

_HEREDOC_START = re.compile(r"<<-?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?")


def test_the_shell_enumeration_is_not_empty():
    """⚠ AN EMPTY ENUMERATION MUST BE RED, NOT ABSENT — measured, not assumed.

    `@pytest.mark.parametrize` over an EMPTY list does not fail and does not error: pytest
    reports `1 skipped` and exits 0. So if `git ls-files '*.sh'` ever returned nothing — the
    directory thins out, the scripts move, the listing breaks — the guard below would
    contribute a skip and the suite would stay green, having checked nothing. That is the
    same silent-disarm shape the rest of this file exists to catch, one level up: not a wrong
    assertion, but no assertion at all.

    Floor, not equality: adding a shell script must not fail this, and losing most of them
    must. 14 measured 2026-08-21, of which 4 carry embedded python heredocs.
    """
    assert len(SHELL_SCRIPTS) >= 10, (
        f"only {len(SHELL_SCRIPTS)} shell script(s) enumerated, from the 14 measured on "
        f"2026-08-21. Either `git ls-files '*.sh'` stopped working or they really moved — "
        f"if the latter, re-derive this floor deliberately rather than lowering it to fit.")
    with_heredocs = [rel for rel in SHELL_SCRIPTS
                     if _embedded_python((REPO / rel).read_text(encoding="utf-8"))]
    # ⚠ The population that MATTERS is the scripts with python in them. All 14 could be
    # enumerated while the heredoc parser silently matched none of them — a floor on the
    # outer list would not notice, and the import guard would pass on every file by having
    # nothing to look at.
    assert len(with_heredocs) >= 3, (
        f"only {len(with_heredocs)} shell script(s) yielded a parseable python heredoc, from "
        f"the 4 measured on 2026-08-21. The import guard has almost nothing to check — "
        f"suspect `_embedded_python` before believing the scripts changed.")


def _embedded_python(text):
    """Every `python … <<'MARKER' … MARKER` heredoc body in a shell script, with its offset."""
    lines = text.split("\n")
    blocks, i = [], 0
    while i < len(lines):
        line = lines[i]
        m = _HEREDOC_START.search(line)
        if m and "python" in line:
            marker, start = m.group(1), i + 1
            j = start
            while j < len(lines) and lines[j].strip() != marker:
                j += 1
            blocks.append((start + 1, "\n".join(lines[start:j])))
            i = j
        i += 1
    return blocks


@pytest.mark.parametrize("rel", SHELL_SCRIPTS, ids=SHELL_SCRIPTS)
def test_shell_scripts_import_only_tracked_modules(rel):
    """A module a shell script imports from its own directory must be IN THE INDEX.

    Reproduce the failure this exists for:

        git rm --cached -q workflow/scripts/merge_floor.py   # untracked, still on disk
        .venv/bin/python -m pytest tests/test_asset_paths.py -q

    ⚠ Checked against `git ls-files`, never `Path.exists()` — the module sits on disk in
    exactly the case worth catching, so existence answers `True` and proves nothing.
    """
    text = (REPO / rel).read_text(encoding="utf-8")
    script_dir = pathlib.PurePosixPath(rel).parent
    broken = []
    for offset, body in _embedded_python(text):
        try:
            tree = ast.parse(body)
        except SyntaxError:
            # A heredoc that is not valid Python on its own (templated, or not Python at all)
            # is skipped rather than failed — this guard is about tracked-ness, and guessing
            # at unparseable text is how a guard starts reporting things that are not defects.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.level == 0 and node.module else []
            else:
                continue
            for dotted in names:
                mod = dotted.split(".")[0]
                sibling = str(script_dir / f"{mod}.py")
                # Only a module this script could resolve from its OWN directory. A stdlib or
                # site-packages import has no sibling file and is not this guard's business.
                if not (REPO / sibling).exists():
                    continue
                if sibling not in TRACKED_PATHS:
                    broken.append(
                        f"  {rel}: the embedded python at line ~{offset + node.lineno - 1} "
                        f"imports `{mod}`,\n"
                        f"      which resolves to {sibling} — ON DISK BUT NOT TRACKED.\n"
                        f"      It imports for you and fails on a fresh clone. `git add {sibling}`."
                    )
    assert not broken, f"{rel} imports untracked module(s):\n" + "\n".join(broken)
