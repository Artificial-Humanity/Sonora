"""The persona wiring: how a session acquires a role, and how the reviewer refuses one.

Since 2026-08-17 the developer role needs no flag — `CLAUDE.md` `@import`s
`Personas/DEVELOPER.md`, and Claude Code auto-discovers `CLAUDE.md`. That removed the owner's
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
REVIEWER = (REPO / "Personas" / "REVIEWER.md").read_text(encoding="utf-8")
LAUNCHER = (REPO / "scripts" / "request_review.sh").read_text(encoding="utf-8")

IMPORT_RE = re.compile(r"^@(\S+)$", re.M)


def test_claude_md_imports_the_developer_persona():
    """The import IS the mechanism. A link would be a request the session may decline."""
    assert "@Personas/DEVELOPER.md" in CLAUDE_MD


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
    assert "@Personas/REVIEWER.md" not in CLAUDE_MD


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
