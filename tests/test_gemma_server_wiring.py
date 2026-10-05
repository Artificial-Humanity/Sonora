"""Every Sonora entry point that calls Gemma starts its server for the run.

A main that forgot the `with GemmaServer(...)` would send every call to a closed port and
spend its retries on transport errors; this walks each main's AST for the with-statement."""

import ast

import pytest

from scripts_layout import REPO, SCRIPTS

MAINS = {"book_ingest.py": "DIRECTOR", "make_teacher_ab_bank.py": "args.model",
         "make_quote_pilot_bank.py": "args.model", "tag_spike.py": "args.model",
         "judge_passages.py": "m", "make_narration_bank.py": "DIRECTOR"}


def _with_roles(src):
    tree = ast.parse(src)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return {ast.unparse(item.context_expr.args[0]) for w in ast.walk(main) if isinstance(w, ast.With)
            for item in w.items if isinstance(item.context_expr, ast.Call)
            and ast.unparse(item.context_expr.func) == "GemmaServer"}


@pytest.mark.parametrize("name,role", sorted(MAINS.items()))
def test_each_main_starts_its_server(name, role):
    assert _with_roles((SCRIPTS / name).read_text(encoding="utf-8")) == {role}


def test_the_guard_catches_a_main_without_one():
    assert _with_roles("def main():\n    for x in y:\n        ask(x)\n") == set()


GEMMA_ENTRY_POINTS = {"casting_pass", "director_tag"}   # book_ingest's two Gemma passes
NOT_SCANNED = ("teacher_audition", "gates")


def _is_gemma_caller(src):
    """True if the file reaches Gemma: it imports gemma_client in any form, or CALLS one of
    book_ingest's Gemma passes (a transitive caller never imports the client itself)."""
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import) and any(a.name == "gemma_client" for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and n.module == "gemma_client":
            return True
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Name) and f.id in GEMMA_ENTRY_POINTS) or (
                    isinstance(f, ast.Attribute) and f.attr in GEMMA_ENTRY_POINTS):
                return True
    return False


def test_the_discovery_sees_a_transitive_caller():
    assert _is_gemma_caller('import book_ingest as bi\ndef main():\n    bi.casting_pass(x, "zonos")\n')
    assert _is_gemma_caller("from book_ingest import director_tag\ndef main():\n    director_tag(c)\n")


def test_the_discovery_sees_every_import_form():
    assert _is_gemma_caller("import gemma_client\n")
    assert _is_gemma_caller("import gemma_client as gc\n")
    assert _is_gemma_caller("from gemma_client import chat\n")


def test_the_discovery_ignores_a_mere_mention():
    assert not _is_gemma_caller(
        '# casting_pass and director_tag are called elsewhere\n'
        'NOTE = "gemma_client casting_pass(x)"\n'
        'def main():\n    """calls director_tag()"""\n')


def test_the_guard_covers_every_gemma_caller():
    # The bench starts its servers per arm (Task 11), so it is checked by its own test.
    callers = {p.name for p in (REPO / "scripts").rglob("*.py")
               if not any(part in NOT_SCANNED for part in p.relative_to(REPO / "scripts").parts)
               and _is_gemma_caller(p.read_text(encoding="utf-8"))
               and p.name != "director_obedience_bench.py"}
    assert callers == set(MAINS)
