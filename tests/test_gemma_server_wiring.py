"""Every Sonora entry point that calls Gemma starts its server for the run.

A main that forgot the `with GemmaServer(...)` would send every call to a closed port and
spend its retries on transport errors; this walks each main's AST for the with-statement."""

import ast

import pytest

from scripts_layout import REPO, SCRIPTS

MAINS = {"book_ingest.py": "DIRECTOR", "make_teacher_ab_bank.py": "args.model",
         "make_quote_pilot_bank.py": "args.model", "tag_spike.py": "args.model",
         "judge_passages.py": "m"}


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


def test_the_guard_covers_every_gemma_caller():
    # The bench starts its servers per arm (Task 11), so it is checked by its own test.
    callers = {p.name for p in (REPO / "scripts").rglob("*.py")
               if "from gemma_client import" in p.read_text(encoding="utf-8")
               and p.name != "director_obedience_bench.py"}
    assert callers == set(MAINS)
