"""The persona wiring: how a session acquires a role, and how the reviewer refuses one.

Since 2026-08-17 the developer role needs no flag — `CLAUDE.md` `@import`s
`workflow/DEVELOPER.md`, and Claude Code auto-discovers `CLAUDE.md`. That removed the owner's
standing irritation (*"I'm still not very keen on having to start claude code with a
pre-prompt"*) and replaced a remembered flag with a mechanism.

It also bought a cost, and both halves are pinned here:

    --system-prompt-file replaces the DEFAULT ASSISTANT PROMPT. It does not suppress
    CLAUDE.md, and imports ride along with it.  (measured 2026-08-17)

So Janis receives Ozzy's full persona on every run — a competing role telling it to commit, to
increment `agent_passes`, and to fix rather than report. Nothing structural separates the two;
the precedence rule does, stated at three points. These tests exist because that rule is prose,
and prose is what drifts.

Verified live the same day, both directions:
  * bare `claude -p` with NO flags        -> "Ozzy", quotes the right `git -c` pair
  * `-p --system-prompt-file REVIEWER.md` -> "Janis", refuses to commit, refuses the counter,
                                             and names Ozzy's persona as present-but-outranked
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLAUDE_MD = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
REVIEWER = (REPO / "workflow" / "REVIEWER.md").read_text(encoding="utf-8")
LAUNCHER = (REPO / "workflow" / "scripts" / "request_review.sh").read_text(encoding="utf-8")

IMPORT_RE = re.compile(r"^@(\S+)$", re.M)


def test_claude_md_imports_the_developer_persona():
    """The import IS the mechanism. A link would be a request the session may decline."""
    assert "@workflow/DEVELOPER.md" in CLAUDE_MD


def test_every_import_target_exists():
    """⚠ A BROKEN `@import` FAILS SILENTLY — no error, no warning, just a session with no role.

    This is the repo's canonical silent-disarm shape: a green result indistinguishable from a
    correct one. Renaming or moving a persona file would disarm the default for every session
    in the repo, and the only symptom would be an agent that quietly is not Ozzy.
    """
    targets = IMPORT_RE.findall(CLAUDE_MD)
    assert targets, "CLAUDE.md imports nothing — the default role is unwired"
    for t in targets:
        assert (REPO / t).is_file(), "CLAUDE.md imports a file that does not exist: %s" % t


def test_claude_md_does_not_import_the_reviewer_persona():
    """Importing REVIEWER.md would hand every ordinary session the role that must not write."""
    assert "@workflow/REVIEWER.md" not in CLAUDE_MD


def test_the_reviewer_persona_revokes_the_imported_developer_persona():
    """REVIEWER.md must say the import is not addressed to it — being the system prompt is not
    self-evidently higher precedence to a model reading both."""
    assert "IGNORE IT" in REVIEWER
    assert "CLAUDE.md" in REVIEWER


def test_the_reviewer_persona_names_the_conflicts_that_matter():
    """A bare "ignore that" does not survive contact with 290 lines of specific instruction.

    The four conflicts are the ones with consequences: identity, the counter, closing, and
    writing. Each is something Ozzy's persona actively tells the reader to do.
    """
    section = REVIEWER[REVIEWER.index("IGNORE IT"):][:2500]
    assert "agent_passes" in section
    assert "commit" in section
    for phrase in ("do not fix", "do not commit", "never touch"):
        assert phrase.lower() in section.lower(), phrase


def test_the_launcher_revokes_it_again_in_the_brief():
    """Third statement, and the last thing in the context — recency favours it there.

    ⚠ The role is decided AT THE CALL SITE. This script is the only thing that launches Janis,
    so it knows the persona it is starting; the reviewer never infers it from its invocation.
    """
    assert "You are Janis, the reviewer. You are not Ozzy." in LAUNCHER


def test_the_launcher_still_replaces_rather_than_appends_the_persona():
    """`--system-prompt-file`, not `--append-…`: REVIEWER.md is written to stand alone."""
    assert "--system-prompt-file" in LAUNCHER


def test_no_document_treats_print_mode_as_the_role_test():
    """⚠ `-p` DOES NOT MEAN "reviewer" — Ozzy runs under `-p` too, via review_cycle.sh.

    `CLAUDE_CODE_ENTRYPOINT` (`cli` vs `sdk-cli`) is real and measured, but it distinguishes
    the INVOCATION, not the ROLE. Any document that promotes it from falsifier to mechanism has
    introduced a wrong branch, so each mention must keep it labelled.
    """
    for name, text in (("CLAUDE.md", CLAUDE_MD), ("REVIEWER.md", REVIEWER)):
        if "CLAUDE_CODE_ENTRYPOINT" in text:
            window = text[max(0, text.index("CLAUDE_CODE_ENTRYPOINT") - 700):]
            assert "falsifier" in window[:1400].lower(), name


# --- #347: the reviewer must be able to read what the routing rule tells it to verify -----

def test_every_existing_sibling_is_granted_not_just_the_first():
    """⚠⚠ `SIBLING_REPO_CANDIDATES` WAS A FALLBACK CHAIN WEARING THE SPELLING OF A LIST.

    The resolver `break`s on the first directory that exists, so only ever one repo was
    granted — and that looked correct for as long as the two configured entries were the same
    repo by two paths (a relative and an absolute AI-Lab-AMD).

    It became wrong when REVIEWER.md started routing findings to FerroStep and telling the
    reviewer to **verify** the boundary rather than remember it. The reviewer reported the
    consequence itself: it declined to file a FerroStep engine finding because it could not
    check a claim about FerroStep's source, and filed that gap instead. An instruction to
    consult a repo the launcher cannot grant is the documented-but-unreachable defect, and
    this lane has now paid for it five times.
    """
    code = "\n".join(l.split("#", 1)[0] for l in LAUNCHER.splitlines())
    assert "SIBLINGS=(" in code, "the launcher still resolves a single sibling"

    # ⚠ THE CANDIDATE LOOP SPECIFICALLY, NOT "any break in the resolver" — which is what the
    # first version of this assertion said, and it went red on correct code within a minute of
    # being written. The de-duplication uses an inner `for … break` legitimately, and a scan
    # over the whole block cannot tell the two loops apart. Wrong scope, right idea: the same
    # substring-over-too-much defect this file exists to catch elsewhere.
    body = code[code.index("for _cand in"):]
    body = body[:body.index("\ndone")]
    assert "break" not in body, (
        "the CANDIDATE loop still breaks — it grants the FIRST existing candidate and "
        "silently drops the rest:\n" + body)


def test_the_sibling_paragraph_is_generated_from_what_was_actually_granted():
    """⚠ THE OTHER HALF, AND IT POINTS THE OPPOSITE WAY. The brief named "the **AI-Lab-AMD**
    infrastructure repo" in hardcoded prose while the grant came from config — so a second
    sibling would have been readable and undescribed. A reader is told what they may read;
    an affordance nobody mentions is as good as absent (`review_cycle.sh` was named in three
    files and run by nobody for exactly that reason).
    """
    assert "AI-Lab-AMD** infrastructure repo" not in LAUNCHER, (
        "the sibling paragraph hardcodes a repo name again; it must be built from SIBLINGS")
    assert "_SIB_LIST" in LAUNCHER, "the paragraph is not generated from the granted list"


def test_the_reviewer_can_reach_the_repo_the_routing_rule_names():
    """The end-to-end property, config included — the two halves above are satisfiable and
    still leave the rule unfollowable if nothing lists FerroStep."""
    cfg = (REPO / "workflow" / "config.env").read_text(encoding="utf-8")
    line = next((l for l in cfg.splitlines() if l.startswith("SIBLING_REPO_CANDIDATES=")), "")
    assert line, "SIBLING_REPO_CANDIDATES is absent from config.env"
    reviewer = (REPO / "workflow" / "REVIEWER.md").read_text(encoding="utf-8")
    if "FerroStep" in reviewer:
        assert "FerroStep" in line, (
            "REVIEWER.md routes findings to FerroStep and tells the reviewer to VERIFY the "
            "boundary, but no candidate path names FerroStep — so the rule cannot be "
            "followed:\n  " + line)


# --- #353: a section inserted mid-table is silent damage ----------------------------------

# ⚠ THE POPULATION, AND WHY IT IS NOT JUST THE PERSONAS (#356). This was `PERSONA_MD`, three
# files of which TWO HELD NO TABLES AT ALL — an effective population of one, under a test named
# "every". `AGENTS.md` was outside it while holding as many table runs as REVIEWER.md, and it
# is the repo's rules of record, always in scope per REVIEWER.md §3, and exactly the kind of
# page a section gets appended into. Measured at the time: REVIEWER 4, DEVELOPER 0, WORKFLOW 0,
# AGENTS 4, CLAUDE 0. Renamed off "PERSONA" because the guard is about tables, not personas.
TABLE_MD = [
    "workflow/REVIEWER.md",
    "workflow/DEVELOPER.md",
    "workflow/WORKFLOW.md",
    "AGENTS.md",
    "CLAUDE.md",
]

# ⚠ THE SUBSET THAT IS KNOWN TO HOLD TABLES, so a per-file floor can be asserted (#363).
# Membership, deliberately NOT counts: a count here would go stale on the next edit that adds
# a row, and this repo's rule is to derive numbers rather than state them. The others in
# TABLE_MD legitimately hold none, so flooring them would fail on a true fact.
# ⚠ THIS LIST IS THE THING THAT MAKES A PER-FILE COLLAPSE VISIBLE. `assert total_runs` sees
# only a TOTAL collapse; REVIEWER.md is the file with a live history of the defect (#353) and
# was exactly the one that could drop out unnoticed while AGENTS.md kept the total non-zero.
TABLES_EXPECTED_IN = ("workflow/REVIEWER.md", "AGENTS.md")


def _table_runs(rel):
    """Every maximal run of consecutive `|`-prefixed lines, with its start line.

    ⚠ FENCED BLOCKS ARE SKIPPED, and #360 is why. This did not skip them, on a stated
    premise that the resulting false positive "fails loudly, which is the cheap direction".
    That holds for ONE of the two sub-cases and not the other, measured: a lone pipe-prefixed
    line inside a fence goes RED (loud, as claimed), but a WELL-FORMED table inside a fence
    goes GREEN *and* increments `total_runs` — so a fenced EXAMPLE table could satisfy the
    anti-vacuity floor by itself, and the floor would report a pass over a population whose
    real tables had all gone. That is the precise failure the floor was added to end, so the
    premise for leaving it was wrong and it is fixed rather than named.

    Pages that DOCUMENT table structure are exactly the ones that grow fenced example
    tables, and two files in `TABLE_MD` are such pages.
    """
    lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
    runs, cur, start, fence = [], [], 0, ""
    for n, l in enumerate(lines, 1):
        stripped = l.lstrip()
        if fence:                                  # inside a fence: nothing is a table
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
    # ⚠ THE UNCLOSED OPENER IS RETURNED, NOT DISCARDED (#363). Carrying fence STATE gave the
    # scanner a way to be silently WRONG that it did not have before: every line after an
    # unbalanced marker is excluded, so one stray fence drops a whole file's tables and the
    # global floor below cannot see it — measured, a single inserted ``` line at REVIEWER.md:6
    # took that file's 4 runs to 0 and the suite stayed GREEN. The caller asserts on this.
    return runs, fence
    return runs


def test_every_markdown_table_run_carries_its_own_header_and_delimiter():
    """⚠⚠ #353 — A SECTION WAS INSERTED BETWEEN ROW 1 AND ROW 2 OF THE FIELDS TABLE, and the
    remaining rows became literal text: no header, no delimiter, no blank line. That table is
    what tells the reviewer what to put in every field of a new issue, INCLUDING the `severity`
    row annotated "the merge gate reads this".

    ⚠ The damage is silent in both directions — the file still parses, every row is still
    present in the source, and a reader skimming the diff sees a new section rather than a
    broken table. Only rendering shows it, and nothing here rendered anything.

    ⚠ THIS IS A STRUCTURAL CHECK, NOT A RENDERER. It asserts that every run of `|` lines starts
    with a header and a `|---|` delimiter — which is what GFM requires to treat a run as a
    table. It cannot see column-count mismatches or escaping errors, and no markdown library is
    installed in this venv to check the rendering itself.

    ⚠ THE OTHER LIMITS, NAMED SO THE LIST ABOVE DOES NOT READ AS COMPLETE (#356):
      * Fenced blocks ARE skipped, as of #360 — see `_table_runs`. The claim that used to
        stand here, that not skipping them "fails loudly, which is the cheap direction", was
        true of a lone pipe line in a fence and FALSE of a well-formed table in one, which
        passed silently and could satisfy the floor below on its own.
      * ⚠ Skipping them opened the OPPOSITE direction, and it is the silent one (#363): an
        unbalanced fence marker excludes every line after it, so a stray ``` drops that
        file's tables from the population entirely. Both new assertions below exist for
        that — fence balance per file, and a per-file floor for the files that hold tables.
      * #353 named two structural faults; this covers fault 1 (no header/delimiter) and NOT
        fault 2 (no blank line between the table and the paragraph above it). Whether fault
        2 alone breaks rendering is UNSETTLED — no markdown library is installed to decide it.

    ⚠ AND IT HAS AN ANTI-VACUITY FLOOR, because it did not (#356). Nothing asserted that any
    table was ever found: the population emptied, or reduced to only the table-less files,
    both went GREEN over nothing. An empty result and a broken instrument are
    indistinguishable — DEVELOPER.md § self-check item 2, and the shape this branch already
    paid for at #315, #330 and #333. The floor is derived, not a magic count: every path must
    resolve, and the sweep must have found at least one table somewhere. A hardcoded
    `>= 4` would go stale the first time REVIEWER.md's tables were restructured.
    """
    missing = [rel for rel in TABLE_MD if not (REPO / rel).exists()]
    assert not missing, (
        f"the table guard's population no longer resolves: {missing} — a rename or a path "
        "typo would otherwise leave this green while checking fewer files than it names")

    broken, total_runs, unbalanced, per_file = [], 0, [], {}
    for rel in TABLE_MD:
        runs, unclosed = _table_runs(rel)
        if unclosed:
            unbalanced.append(f"{rel} — a `{unclosed}` fence is never closed")
        per_file[rel] = len(runs)
        for start, run in runs:
            total_runs += 1
            if len(run) < 2:
                broken.append(f"{rel}:{start} — a single `|` line, not a table")
                continue
            if not re.match(r"^\s*\|[\s:|-]+\|\s*$", run[1]):
                broken.append(
                    f"{rel}:{start} — run of {len(run)} rows whose 2nd line is not a "
                    f"delimiter:\n      {run[0][:70]}\n      {run[1][:70]}")

    # ⚠ BEFORE the floor, because an unbalanced fence is what makes the floor lie (#363).
    assert not unbalanced, (
        "unbalanced code fence(s) — every line after the opener is excluded from this "
        "guard, so the file's tables silently stop being checked:\n  "
        + "\n  ".join(unbalanced))

    assert total_runs, (
        "the table guard found NO tables in any of "
        f"{TABLE_MD} — it reported a pass over an empty population, which is "
        "indistinguishable from a clean one. Check the paths and `_table_runs`.")

    # ⚠ AND THE FLOOR IS PER FILE, NOT ONLY GLOBAL (#363). `assert total_runs` sees a TOTAL
    # collapse and is blind to a per-file one: with AGENTS.md still supplying runs, REVIEWER.md
    # — the single file with a live history of the defect this guard was built for (#353) —
    # could drop to zero and the total stayed non-empty. Only files KNOWN to hold tables are
    # floored; the rest legitimately have none, and hardcoding a count here would go stale on
    # the next edit, so the population is derived.
    for rel, n in per_file.items():
        if rel in TABLES_EXPECTED_IN:
            assert n, (
                f"{rel} contributed NO table runs to this guard. It is listed as a file that "
                "holds tables, so zero means the scan lost it — a fence, a rename, or a "
                "rewrite — not that the tables are fine.")

    assert not broken, (
        "markdown table run(s) with no header/delimiter — these render as literal text:\n  "
        + "\n  ".join(broken)
        + "\n\n⚠ Usually caused by inserting a section INTO a table. Move the section after "
          "the table, or give the orphaned rows their own header.")
