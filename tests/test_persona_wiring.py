"""How a session acquires its role, and the structural guards that went down with the lane.

⚠⚠ THIS FILE IS A SALVAGE, AND THE REASON MATTERS MORE THAN THE CONTENT. `tests/test_personas.py`
was deleted on 2026-09-15 with the review cycle, correctly — most of it tested a reviewer
process that no longer exists. Three of its thirteen tests were not about the lane at all, and
they went with it:

  * `test_every_import_target_exists` — the `@import` in `CLAUDE.md` resolving
  * `test_claude_md_does_not_import_the_reviewer_persona`
  * `test_every_markdown_table_run_carries_its_own_header_and_delimiter` — over `AGENTS.md`
    and `CLAUDE.md` as much as over the personas

⚠ AND THE FIRST ONE'S EXACT FAILURE WAS COMMITTED THE DAY AFTER IT WAS DELETED. That test's
docstring said: *"Renaming or moving a persona file would disarm the default for every session
in the repo, and the only symptom would be an agent that quietly is not Ozzy."* The very next
commit moved `FerroStep/personas/DEVELOPER.md` to `docs/personas/DEVELOPER.md` and hand-edited
the import. It happened to be correct. Nothing would have said so if it were not.

⚠ These files are excluded from the doc-link gate by the owner's ruling of 2026-08-21
(`_SCAN_EXCLUDE` in `scripts/gates/test_doc_links.py`), so nothing else in the repo reads a
path inside them. That exclusion is why this file has to exist rather than being covered
incidentally.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

# ⚠ A PLAIN IMPORT, NOT `importorskip`. `pyproject.toml` declares PyYAML as "CORE, NOT
# TRANSITIVE, AND NOT OPTIONAL", and `tests/test_commit_hygiene.py` hard-fails without it.
# Skipping here would have given two files two answers to the same missing dependency, and
# the file that vanishes silently is this one — the eval-trap guard.
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
CLAUDE_MD = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
IMPORT_RE = re.compile(r"^@(\S+)\s*$", re.M)
ROSTER = REPO / "roster.yaml"


# --------------------------------------------------------------------------- #
# the import — the repo's canonical silent-disarm shape
# --------------------------------------------------------------------------- #
def test_every_import_target_exists():
    """⚠ A BROKEN `@import` FAILS SILENTLY — no error, no warning, just a session with no role.

    A green result indistinguishable from a correct one. The only symptom of a moved persona
    is an agent that quietly is not Ozzy, in a repo where that agent then commits.
    """
    targets = IMPORT_RE.findall(CLAUDE_MD)
    assert targets, "CLAUDE.md imports nothing — the default role is unwired"
    for t in targets:
        assert (REPO / t).is_file(), "CLAUDE.md imports a file that does not exist: %s" % t


def test_the_import_and_the_roster_name_the_same_persona():
    """⚠ CLAUDE.md calls its import path "the one deliberate second copy" of the roster's
    `persona` value and says "change one, change both". That pair had no mechanism.

    An `@import` cannot read YAML, so the duplication is unavoidable — which makes it exactly
    the kind of pair that needs a test rather than an instruction. AGENTS.md's own rule is
    that a rule without a mechanism is not enforcement.
    """
    doc = yaml.safe_load(ROSTER.read_text(encoding="utf-8"))
    persona = doc["agents"][doc["default_agent"]]["persona"]
    targets = IMPORT_RE.findall(CLAUDE_MD)
    assert persona in targets, (
        "CLAUDE.md imports %s but roster.yaml's default agent (%r) names %r — the two copies "
        "have drifted, and the symptom is a session holding the wrong persona or none"
        % (targets, doc["default_agent"], persona))


def test_claude_md_does_not_import_the_reviewer_persona():
    """Importing REVIEWER.md would hand every ordinary session a second, competing role.

    ⚠ Kept although nothing launches a reviewer any more: the file still exists as a
    depiction, the import is still a glob away from picking it up, and the cost of it
    happening is a session that believes it must not write.
    """
    for t in IMPORT_RE.findall(CLAUDE_MD):
        assert "REVIEWER" not in t.upper(), "CLAUDE.md imports the reviewer persona: %s" % t


# --------------------------------------------------------------------------- #
# the roster resolves for EVERY entry, not just the default
# --------------------------------------------------------------------------- #
def test_every_roster_entry_resolves():
    """⚠ `tests/test_agent_env.py` only ever resolved the DEFAULT agent, so a broken
    `reviewer` entry — a moved persona, a blank email — was invisible.

    That is the same silent-disarm shape as the import above, one file over. Derived from the
    roster rather than parametrised on a hand-written list, so a new entry is covered the day
    it is added.
    """
    doc = yaml.safe_load(ROSTER.read_text(encoding="utf-8"))
    titles = sorted(doc["agents"])
    assert titles, "roster.yaml declares no agents"
    for title in titles:
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "agent_env.py"), "--agent", title],
            capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 0, "roster entry %r does not resolve:\n%s" % (title, r.stderr)
        env = dict(l.split("=", 1) for l in r.stdout.strip().splitlines())
        assert (REPO / env["AGENT_PERSONA"].strip("'\"")).is_file()


# --------------------------------------------------------------------------- #
# markdown tables — #353, #360, #363, and the population is now Sonora's own files
# --------------------------------------------------------------------------- #
# ⚠ `FerroStep/workflow/WORKFLOW.md` left this list when it was deleted; the personas moved.
# What is left is mostly AGENTS.md and CLAUDE.md — the repo's rules of record — which is the
# reason this guard had to be salvaged rather than dropped with the lane it sat beside.
TABLE_MD = [
    "docs/personas/REVIEWER.md",
    "docs/personas/DEVELOPER.md",
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
