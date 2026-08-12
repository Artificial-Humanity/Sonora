"""Every in-repo path a script builds from `__file__` must actually point at the thing.

This guard exists for one failure mode, and it is the expensive one. A script that
resolves its data relative to its own location — `os.path.join(os.path.dirname(
os.path.abspath(__file__)), "director_skills")` — keeps importing perfectly after the file
moves one directory deeper or shallower. It fails when something *reads* the path, which
for the synthesis lane is minutes to hours into a GPU render, and for `register_lexicon.json`
is a bank built without the register vocabulary rather than a bank that failed to build.

`director_skills/` is read from four future buckets' worth of modules and
`register_lexicon.json` from eleven files including `audition/app/main.py`, so a layout
change (#26 step 3) touches every one of these depths at once. Reading them is not
verification: none of the lanes that consume them can run on the host — no torch — so
"it looked right" is the whole of the evidence otherwise available.

**What is asserted:** if a module-level path constant's final literal segment names
something that exists somewhere in this repo, the path the code builds must resolve to it.
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
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

# `os.path.join(os.path.dirname(os.path.abspath(__file__)), "x")` and the pathlib spellings.
_PATH_CALLS = {"join", "dirname", "abspath", "realpath", "expanduser", "normpath", "Path"}


class Unresolvable(Exception):
    """The expression depends on something this evaluator deliberately does not model."""


def _tracked_basenames():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    names = {}
    for rel in out.stdout.splitlines():
        if not rel:
            continue
        p = pathlib.PurePosixPath(rel)
        names.setdefault(p.name, []).append(rel)
        for parent in p.parents:
            if parent.name:
                names.setdefault(parent.name, []).append(str(parent))
    return names


TRACKED = _tracked_basenames()


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
            return os.path.join(*args)
        if name in ("dirname",):
            return os.path.dirname(args[0])
        return args[0] if args else ""
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
    return out


SCRIPTS = sorted(
    p for p in subprocess.run(
        ["git", "ls-files", "scripts/*.py", "scripts/**/*.py"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split()
)


def test_the_enumeration_is_not_empty():
    """Every case below is parametrized off this list; an empty one reports green."""
    assert len(SCRIPTS) >= 80, f"only {len(SCRIPTS)} scripts found — is the enumeration working?"
    assert TRACKED, "git ls-files returned nothing"


@pytest.mark.parametrize("rel", SCRIPTS, ids=SCRIPTS)
def test_in_repo_asset_paths_resolve(rel):
    """If the tail names something in this repo, the built path must point at it."""
    broken = []
    for name, lineno, value in _path_constants(rel):
        tail = pathlib.PurePosixPath(value).name
        if tail not in TRACKED:
            continue  # names nothing in the tree — a runtime output, not an asset
        target = pathlib.Path(value)
        if not target.is_absolute():
            target = REPO / target
        if not target.exists():
            broken.append(
                f"  {rel}:{lineno}  {name} = {value}\n"
                f"      does not exist. `{tail}` IS in the repo, at: {sorted(set(TRACKED[tail]))[:4]}\n"
                f"      so this path is built at the wrong depth for where {rel} now lives."
            )
    assert not broken, f"{rel} builds path(s) that do not resolve:\n" + "\n".join(broken)
