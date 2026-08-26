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
import subprocess

import pytest
# NOTE: this guard enumerates `scripts/**/*.py` ITSELF (see SCRIPTS below), not via
# `scripts_layout.SCRIPTS` — different populations: that resolver knows seven bucket
# directories, this needs every tracked .py. Importing it here only shadowed the name.

REPO = pathlib.Path(__file__).resolve().parent.parent

# `os.path.join(os.path.dirname(os.path.abspath(__file__)), "x")` and the pathlib spellings.
# ⚠ `str` is here for one reason and it is measured, not speculative: the repo's standard
# "env var wins, else repo-relative" idiom writes the fallback as
# `str(pathlib.Path(__file__).resolve().parents[N])`, and refusing `str()` was what actually
# kept `SONORA` out of `env` in `convert_vat.py` — not the `or`, which is handled below.
# Identity is correct here: `str()` of a path IS that path, and the surrounding `rooted` test
# means we only ever reach it on a `__file__`-derived expression.
_PATH_CALLS = {"join", "dirname", "abspath", "realpath", "expanduser", "normpath", "Path", "str"}


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
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        # ⚠ #315, SECOND HALF. `SONORA = os.environ.get("SONORA_REPO") or str(Path(__file__)
        # .resolve().parents[2])` is the repo's standard "env var wins, else repo-relative"
        # idiom (2 instances, measured). Refusing the whole expression left `SONORA` out of
        # `env`, so the assertion one line below it — `Path(SONORA)/"matcha"/"models"/
        # "matcha_tts.py"` — was enumerated by NEITHER half of the walk. That is the site
        # #315's own body named as unguarded, and the first fix did not reach it.
        #
        # The LAST operand is the right one to take, and not merely the convenient one: the
        # env override is runtime configuration and makes no claim about depth, while the
        # fallback is precisely the "where does this file sit relative to the repo" assertion
        # this guard exists to check. If the fallback does not resolve, refuse — do not fall
        # back to an earlier operand, which would resolve a path this code never builds.
        return _eval(node.values[-1], here, env)
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

    # ⚠ #315. A `/`-composed path is an `ast.BinOp`, NOT a `Call`, so an enumeration that
    # collected only `join`/`Path` calls never saw the whole expression. `ast.walk` then
    # descended past it and scored the innermost `Path(__file__)` on its own. Two costs, and
    # the second is the worse one:
    #   1. the real asset went unchecked — measured on `scripts/gates/test_vat_dim_seams.py`,
    #      where a `parents[1]` at the WRONG DEPTH was the whole defect and was never
    #      evaluated;
    #   2. the innermost `Path(__file__)` resolves to the PARSED FILE ITSELF, which is
    #      tracked by definition and so cannot fail — and it was counted toward the
    #      falsifiable floor below. A guard padding its own ratchet with assertions that
    #      cannot go red is silent-disarm mode 1 again, wearing a bigger number.
    # `_eval` was never the limitation: it has handled `BinOp`/`Div` since line 138. The
    # enumeration simply never handed it the node.
    def _cand_name(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return "/"
        if isinstance(node, ast.Call):
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if fname in ("join", "Path"):
                return fname
        return None

    def scan(node):
        fname = _cand_name(node)
        # Rooted directly OR through a name already resolved from `__file__` — the whole
        # point is `os.path.join(_SONORA_REPO, "scripts", "assets", "x.json")`, whose own
        # node never mentions `__file__`. Requiring it here is what left this site
        # unguarded on the first attempt at this fix.
        if fname is not None and ("__file__" in ast.dump(node) or any(
            isinstance(n, ast.Name) and n.id in env for n in ast.walk(node)
        )):
            try:
                value = _eval(node, here, env)
            except (Unresolvable, IndexError, TypeError, AttributeError, KeyError):
                pass          # fall through and try the parts: an outer expression this
                              # evaluator refuses must not hide a resolvable inner one
            else:
                if (node.lineno, value) not in seen:
                    seen.add((node.lineno, value))
                    out.append((f"<{fname} at line {node.lineno}>", node.lineno, value))
                # ⚠ STOP. The enclosing expression IS the path; descending would re-score
                # its own `Path(__file__)` as a second, self-referential "asset".
                return
        for child in ast.iter_child_nodes(node):
            scan(child)

    scan(tree)
    return out


SCRIPTS = sorted(
    p for p in subprocess.run(
        ["git", "ls-files", "-z", "scripts/*.py", "scripts/**/*.py"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\0") if p
)


def test_the_guard_actually_asserts_something():
    """The floor has to count FALSIFIABLE assertions, not files enumerated.

    `len(SCRIPTS)` was the wrong population. RE-DERIVED 2026-08-26 (#315), all five numbers
    from one run of this test's own body:

        scripts enumerated                 110
        constants resolving to a repo path  40
        of which SELF-ANCESTOR              10   <- cannot fail, by construction
        of which the PARSED FILE ITSELF      2   <- ditto; see below
        FALSIFIABLE                         28   <- what the floor below counts

    Most scripts build no in-repo path at all, so a floor on the file count says nothing.
    And a self-ancestor (`HERE = dirname(abspath(__file__))`) resolves to a directory that
    contains the file being parsed — counting those as coverage is silent-disarm mode 1
    wearing a number.

    ⚠ The **parsed file itself** is the same class and evaded the self-ancestor test for a
    year, because a file is not in its own `.parents`. Those entries were manufactured by
    the enumeration bug in `_path_constants` (#315): it scored the inner `Path(__file__)` of
    an expression it could not see whole, and that inner call resolves to the file being
    parsed, which is tracked by definition. **The guard was padding its own ratchet.**
    Genuine coverage at the time of that finding was 23 against a floor of 24 — the floor was
    only satisfied by assertions that could not go red.

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
            resolved = target.resolve()
            if resolved in here.parents:
                continue                      # self-ancestor: true by construction
            # ⚠ #315. And the file ITSELF, which the `.parents` test above cannot catch —
            # a file is not in its own `.parents`. `pathlib.Path(__file__).name` inside an
            # f-string is not an asset path in any sense, yet it resolved to a tracked
            # name and was counted. Same class as the self-ancestor: tracked by
            # construction, so it can never go red.
            if resolved == here:
                continue
            falsifiable.append(f"{rel}:{lineno} {name}")
    # ⚠ 25, RE-DERIVED 2026-08-26 (#315). ⚠⚠ AND IT CANNOT BE RECONCILED AGAINST THE OLD 24
    # BY ARITHMETIC AT ALL — the old floor was measured on 2026-08-12 against a DIFFERENT
    # TREE, so no sum of today's terms produces it. (#320: an earlier version of this comment
    # offered exactly such a sum, "23 genuine + 4 self-referential minus 3", in the same
    # breath as telling the reader not to reconcile by arithmetic. The subtrahend was 2, and
    # the subject — 23+4=27 — is today's PRE-FIX measurement, never the old floor. Reasoned,
    # not measured, in the one file whose docstring warns against that.)
    #
    # THE ACTUAL DERIVATION, every enumeration run over the SAME tree on 2026-08-26:
    #
    #   pre-fix   : 23 genuine + 4 self-file -> 27 counted as falsifiable (floor was 24)
    #   after #315: 25 genuine + 2 self-file -> 25 counted (self-file now excluded)
    #   after #320: 28 genuine + 2 self-file -> 28 counted   <- this floor
    #
    #   28 = 23 genuine, NONE LOST
    #      +  2 `/`-composed sites the Call-only walk could not see (test_vat_dim_seams:58,:81)
    #      +  3 reached once `str()` and `X or Y` stopped being refused — convert_vat.py:138
    #         (the site #315's OWN BODY named as unguarded, and which the first fix did not
    #         reach), probe_delivery_intercept.py:111, schemas.py:161
    #
    # Of the 4 pre-fix self-file entries, 2 vanish because the new scan scores their real
    # OUTER expression instead (`register_audition.py:70`, `sweep_dropped.py:50` — both were
    # already counted as named constants, so the self-file entry was a duplicate), and 2 are
    # removed by the `resolved == here` filter (`convert_vat.py:137`, `publish_tier.py:117`).
    #
    # Re-run the body rather than trusting the block above; that instruction outranks it.
    # The floor is the measurement: this is a ratchet, so any drop in falsifiable coverage
    # is meant to be noticed, including a legitimate one.
    assert len(falsifiable) >= 28, (
        f"falsifiable asset paths dropped to {len(falsifiable)} from the 28 measured on\n"
        f"2026-08-26. Either the evaluator stopped resolving a spelling, or coverage really\n"
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

    Reproduce, against an asset THIS guard covers (measured 2026-08-21):

        git rm --cached -q scripts/assets/register_lexicon.json   # untracked, still on disk
        .venv/bin/python -m pytest tests/test_asset_paths.py -q
        -> EXISTS BUT IS NOT TRACKED (scripts/assets/register_lexicon.json)

    ⚠ AND THE FIRST VERSION OF THIS FIX WAS VACUOUS UNDER THAT EXACT COMMAND. Making the
    check below index-aware left it GREEN, because it is gated on `tail not in TRACKED` and
    TRACKED is built from `git ls-files` — untracking a file removes its basename, so the
    guard skipped it as "a runtime output". A selector that goes blind to precisely the file
    that went missing is the disarm mode this file's docstring is about, which is why the
    tracked-ness question is asked FIRST, ungated by the tail heuristic.

    The resolution is `git ls-files`, and "exists but untracked" gets its OWN message: a
    different mistake from "does not exist", sending the reader to `git add` rather than to
    "fix the path".

    ⚠ SCOPE: `scripts/**/*.py` only (see SCRIPTS). #207 also asked for shell scripts, whose
    live population was entirely under `workflow/` — retired by owner ruling 2026-08-21, so
    that half is deliberately NOT here and #207 stays open for it.
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



def test_the_scripts_readme_counts_match_the_tree():
    """⚠ THE RED FOR THE THREE BARE-INTEGER CELLS — `gates/`, `lib/`, `tools/` — NOT ALL SEVEN.

    `scripts/README.md`'s table said 6 gate scripts and 10 `lib/` files while the tree held 7
    and 11 (#274). Both went stale on this branch, which ADDED the seventh gate script — the
    author of the drift and the reader of the table were the same session. This derives every
    cell whose count is a bare integer (#280). The other four carry units or compounds
    (`17 py + 4 sh`, `12 py`, `4 + 9`) that `stated == str(actual)` cannot match, so a stale
    count in those rows will NOT go red here — deriving them means parsing per-suffix or
    reshaping the cells, and that is a decision, not a drive-by.

    The repo's rule is "derive, never duplicate", and a README table cannot derive. So the
    number stays where a human wants to read it and the derivation lives here — for the cells
    above, the only arrangement in which prose and tree can disagree loudly.
    """
    table = (REPO / "scripts" / "README.md").read_text(encoding="utf-8")
    for directory, pattern in (("gates", "scripts/gates/*.py"), ("lib", "scripts/lib/*.py"),
                               ("tools", "scripts/tools/*.py")):
        actual = len(subprocess.run(
            ["git", "ls-files", pattern], cwd=REPO,
            capture_output=True, text=True, check=True).stdout.split())
        row = next((ln for ln in table.split("\n") if f"**`{directory}/`**" in ln), None)
        assert row, f"no `{directory}/` row in scripts/README.md — the table moved"
        stated = row.rsplit("|", 2)[1].strip()
        assert stated == str(actual), (
            f"scripts/README.md says {directory}/ holds {stated}; git ls-files says {actual}. "
            f"Update the table — or if the count is no longer worth stating, delete it and "
            f"this assertion together.")
