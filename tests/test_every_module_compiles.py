"""Every `.py` this repo owns must COMPILE — not merely parse, and tracked or not.

⚠ This exists because `ast.parse` is not enough, and the gap cost a shipped `SyntaxError`.

`scripts/litert_export/convert_vat.py` — the LiteRT export converter — went out in `7557aea`
with `SyntaxError: 'return' outside function`. #26 step 3's prologue insertion replaced an
*indented* `sys.path.insert` inside `g2p_parity_gate()` with an unindented block, dedenting
the function body so its `return` landed at module level. The identical breakage in
`book_ingest.py` was caught the same afternoon and fixed — by a sweep that used
`ast.parse`, which reported ONE broken file when there were TWO.

**`ast.parse` builds a syntax tree; it does not run the symbol-table pass.** A `return`
outside a function, a `yield` outside a function, `await` outside `async def`, duplicate
parameter names, `nonlocal` with no binding — all of these parse cleanly and only fail at
`compile()`. So a tree-level check is structurally incapable of finding this class, and
every AST-based guard in this suite shares that blind spot.

It went unnoticed for another reason worth recording: nothing in the suite imports the
export lane (it needs torch and the LiteRT harness), and nothing else compiles it. A file
that only a GPU host executes gets its first real syntax check on that host, hours in.

One `compile()` per file, no imports, no side effects — so it covers exactly the files
nothing else can.
"""

import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def _tracked_python():
    """Every .py this repo owns, TRACKED OR NOT.

    ⚠ `--cached --others --exclude-standard` — THE SAME FIX AS #110 (issue #159). That fix
    corrected `test_stage_coverage`'s enumeration and left this one, which is the same
    `git ls-files` blind spot in the same class of guard: a new module is invisible until it
    is tracked, so it goes uncompiled exactly while it is most likely to be broken.

    ⚠ IT WAS FOUND BY A ONE-TEST COUNT DISCREPANCY, not by anything failing.
    `tests/test_declared_dependencies.py` was untracked when its own commit's suite ran, so
    the 1360 reported as that commit's evidence became 1361 the moment it was committed. **A
    guard that silently covers less reports a SMALLER number, and nobody reads a smaller
    green number as a failure.** `--exclude-standard` keeps .gitignore honoured, so this does
    not start compiling build output.
    """
    return _enumerate(REPO)


def _enumerate(root):
    """The enumeration itself, parameterised by root SO THAT IT CAN BE TESTED (issue #161).

    ⚠ The first regression guard for this hardcoded its own copy of the flags and passed
    while the real command was reverted — the defect #161 filed, reproduced inside its own
    fix. A guard that does not call the code under test is asserting about a string.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    return sorted(p for p in out.stdout.split("\0") if p)


FILES = _tracked_python()


def test_the_enumeration_found_the_tree():
    """Every case below is parametrized off this list, so an empty one reports green."""
    assert len(FILES) >= 150, f"only {len(FILES)} .py files — is the listing working?"


def test_the_enumeration_would_see_an_untracked_module(tmp_path):
    """⚠ THE REGRESSION GUARD #159 SHIPPED WITHOUT (issue #161).

    Reverting the `--cached --others --exclude-standard` flags changes nothing observable:
    every file is tracked right now, so the count is identical either way and the revert is
    invisible — which is the exact "smaller green number nobody reads as a failure" that
    #159's own docstring names. A fix for an unfalsifiable guard that is itself
    unfalsifiable is the shape §5b exists for.

    So this runs the real command against a scratch repo holding one tracked and one
    untracked module, and asserts both come back.
    """
    import subprocess as sp

    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    sp.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=env)
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True, env=env)
    (tmp_path / "untracked.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("z = 3\n", encoding="utf-8")

    seen = set(_enumerate(tmp_path))          # ⚠ THE REAL FUNCTION, not a copy of its flags
    assert "tracked.py" in seen
    assert "untracked.py" in seen, (
        "the enumeration no longer sees untracked modules — a new file goes uncompiled "
        "exactly while it is most likely to be broken (issues #110, #159)")
    assert "ignored.py" not in seen, (
        "--exclude-standard is gone; the enumeration has started compiling ignored files")


@pytest.mark.parametrize("rel", FILES, ids=FILES)
def test_the_module_compiles(rel):
    """`compile()`, not `ast.parse`. See the module docstring for why that distinction is
    the whole point of this file."""
    path = REPO / rel
    try:
        # ⚠ `utf-8-sig`, not `utf-8`: a UTF-8 BOM survives the latter as a leading
        # U+FEFF and `compile()` rejects it, while Python's import machinery strips
        # it and loads the file fine. No tracked file has one today, so plain
        # `utf-8` would have been a trap, not a failure — the first file touched by
        # an editor that writes one would fail a syntax check for not being one.
        compile(path.read_text(encoding="utf-8-sig"), rel, "exec")
    except SyntaxError as exc:
        pytest.fail(f"{rel}:{exc.lineno}: {exc.msg}\n    {(exc.text or '').rstrip()}")


def test_ast_parse_would_not_have_caught_it():
    """The claim above, asserted rather than trusted.

    If a future Python makes `ast.parse` reject this, the module docstring's reasoning stops
    being true and this file's justification needs rewriting — better to hear it from a test
    than to keep repeating a stale rationale.
    """
    import ast  # pylint: disable=import-outside-toplevel

    broken = "def f():\n    pass\nreturn 1\n"
    ast.parse(broken)                      # parses fine — this is the blind spot
    with pytest.raises(SyntaxError):
        compile(broken, "<probe>", "exec")  # and only compile() sees it


def test_a_utf8_bom_does_not_read_as_a_syntax_error(tmp_path):
    """The `utf-8-sig` choice above, asserted rather than trusted.

    Python's import machinery strips a UTF-8 BOM and loads the file; `read_text("utf-8")`
    keeps it as a leading U+FEFF and `compile()` then rejects it. So a guard reading plain
    `utf-8` would report a SyntaxError about a file Python runs fine — a false red, which is
    how a check gets switched off. No tracked file has a BOM today, which is exactly why this
    is a fixture and not a finding.
    """
    f = tmp_path / "bom.py"
    f.write_bytes(b"\xef\xbb\xbfx = 1\n")
    with pytest.raises(SyntaxError):
        compile(f.read_text(encoding="utf-8"), "bom.py", "exec")     # the trap
    compile(f.read_text(encoding="utf-8-sig"), "bom.py", "exec")      # what this file does
