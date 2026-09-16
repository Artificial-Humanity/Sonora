"""How a session acquires its role, and the structural guards that went down with the lane.

⚠⚠ THIS FILE IS A SALVAGE, AND THE REASON MATTERS MORE THAN THE CONTENT. `tests/test_personas.py`
was deleted on 2026-09-15 with the review cycle, correctly — most of it tested a reviewer
process that no longer exists. Two of its thirteen tests were not about the lane at all, and
they went with it:

  * `test_every_import_target_exists` — the `@import` in `CLAUDE.md` resolving
  * `test_every_markdown_table_run_carries_its_own_header_and_delimiter` — over `AGENTS.md`
    and `CLAUDE.md` as much as over the persona

⚠ AND THE FIRST ONE'S EXACT FAILURE WAS COMMITTED THE DAY AFTER IT WAS DELETED. That test's
docstring said: *"Renaming or moving a persona file would disarm the default for every session
in the repo, and the only symptom would be an agent that quietly is not the developer."* The
very next commit moved the persona file and hand-edited the import. It happened to be correct.
Nothing would have said so if it were not.

⚠⚠ **THE IDENTITY HALF OF THIS FILE WENT ON 2026-09-16** (owner), with the roster file,
its resolver script and `docs/personas/`. `test_the_import_and_the_roster_name_the_same_persona`
and `test_every_roster_entry_resolves` guarded a second copy of the persona path that no longer
exists, and `test_claude_md_does_not_import_the_reviewer_persona` guarded against importing a
file that is gone. **Identity is now stated once, in `PERSONA.md`.** The import test below is
what remains, and it is the load-bearing one: it is the only thing standing between a moved
persona and a session that silently has no role.

⚠ `PERSONA.md` is excluded from the doc-link gate by the owner's ruling of 2026-08-21
(`_SCAN_EXCLUDE` in `scripts/gates/test_doc_links.py`), so nothing else in the repo reads a
path inside it. That exclusion is why this file has to exist rather than being covered
incidentally.
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
CLAUDE_MD = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
IMPORT_RE = re.compile(r"^@(\S+)\s*$", re.M)


# --------------------------------------------------------------------------- #
# the import — the repo's canonical silent-disarm shape
# --------------------------------------------------------------------------- #
def test_every_import_target_exists():
    """⚠ A BROKEN `@import` FAILS SILENTLY — no error, no warning, just a session with no role.

    A green result indistinguishable from a correct one. The only symptom of a moved persona
    is an agent that quietly is not Sonya, in a repo where that agent then commits.
    """
    targets = IMPORT_RE.findall(CLAUDE_MD)
    assert targets, "CLAUDE.md imports nothing — the default role is unwired"
    for t in targets:
        assert (REPO / t).is_file(), "CLAUDE.md imports a file that does not exist: %s" % t


# --------------------------------------------------------------------------- #
# markdown tables — #353, #360, #363, and the population is now Sonora's own files
# --------------------------------------------------------------------------- #
# ⚠ THIS LIST HAS LOST THREE ENTRIES, EACH BECAUSE THE FILE WENT — the review lane's own
# WORKFLOW.md, then `docs/personas/{DEVELOPER,REVIEWER}.md` on 2026-09-16. What is left is the
# repo's rules of record plus the persona, which is the reason this guard had to be salvaged
# rather than dropped with the lane it sat beside. The anti-vacuity check below is what makes a
# fourth disappearance fail loudly instead of silently shrinking the population.
TABLE_MD = [
    "PERSONA.md",
    "AGENTS.md",
    "CLAUDE.md",
]


def _table_runs(rel):
    """Every maximal run of consecutive `|`-prefixed lines, with its start line.

    ⚠ FENCED BLOCKS ARE SKIPPED (#360): a WELL-FORMED table inside a fence passes and would
    increment the anti-vacuity floor by itself, so a fenced example could satisfy the floor
    while every real table had gone.

    ⚠ The unclosed opener is RETURNED, not discarded (#363). Carrying fence state gives the
    scanner a way to be silently wrong: every line after an unbalanced marker is excluded, so
    one stray fence drops a whole file's tables and the floor cannot see it. The caller
    asserts on it.
    """
    lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
    runs, cur, start, fence = [], [], 0, ""
    for n, l in enumerate(lines, 1):
        stripped = l.lstrip()
        if fence:
            if stripped.startswith(fence):
                fence = ""
            continue
        opener = re.match(r"(`{3,}|~{3,})", stripped)
        if opener:
            fence = opener.group(1)
            if cur:
                runs.append((start, cur)); cur = []
            continue
        if stripped.startswith("|"):
            if not cur:
                start = n
            cur.append(l)
        elif cur:
            runs.append((start, cur)); cur = []
    if cur:
        runs.append((start, cur))
    return runs, fence


def test_every_markdown_table_run_carries_its_own_header_and_delimiter():
    """⚠⚠ #353 — A SECTION WAS INSERTED BETWEEN ROW 1 AND ROW 2 OF A TABLE, and the remaining
    rows became literal text: no header, no delimiter, no blank line.

    The damage is silent in both directions — the file still parses, every row is still present
    in the source, and a reader skimming the diff sees a new section rather than a broken
    table. Only rendering shows it, and nothing here renders anything.

    ⚠ STRUCTURAL, NOT A RENDERER. It asserts every run of `|` lines opens with a header and a
    `|---|` delimiter, which is what GFM requires. It cannot see column-count mismatches or
    escaping errors, and it does not cover #353's second fault (no blank line above the table),
    which is UNSETTLED — no markdown library is installed here to decide it.

    ⚠ The anti-vacuity floor is derived, never a count: a hardcoded `>= 4` goes stale the first
    time a file is restructured.
    """
    missing = [rel for rel in TABLE_MD if not (REPO / rel).exists()]
    assert not missing, (
        "the table guard's population no longer resolves: %s — a rename or a path typo would "
        "otherwise leave this green while checking fewer files than it names" % missing)

    broken, total_runs, unbalanced = [], 0, []
    for rel in TABLE_MD:
        runs, unclosed = _table_runs(rel)
        if unclosed:
            unbalanced.append("%s — a `%s` fence is never closed" % (rel, unclosed))
        for start, run in runs:
            total_runs += 1
            if len(run) < 2:
                broken.append("%s:%d — a single `|` line, not a table" % (rel, start))
                continue
            if not re.match(r"^\s*\|[\s:|-]+\|\s*$", run[1]):
                broken.append("%s:%d — run of %d rows whose 2nd line is not a delimiter:\n"
                              "      %s\n      %s" % (rel, start, len(run), run[0][:70], run[1][:70]))

    # ⚠ BEFORE the floor, because an unbalanced fence is what makes the floor lie (#363).
    assert not unbalanced, (
        "unbalanced code fence(s) — every line after the opener is excluded from this guard, "
        "so the file's tables silently stop being checked:\n  " + "\n  ".join(unbalanced))
    assert total_runs, (
        "the table guard found NO tables in any of %s — a pass over an empty population, "
        "indistinguishable from a clean one." % TABLE_MD)
    assert not broken, "malformed markdown table run(s):\n  " + "\n  ".join(broken)
