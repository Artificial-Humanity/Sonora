"""Every tracked `.py` must COMPILE — not merely parse.

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

import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def _tracked_python():
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return sorted(p for p in out.stdout.split("\0") if p)


FILES = _tracked_python()


def test_the_enumeration_found_the_tree():
    """Every case below is parametrized off this list, so an empty one reports green."""
    assert len(FILES) >= 150, f"only {len(FILES)} tracked .py files — is the listing working?"


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
