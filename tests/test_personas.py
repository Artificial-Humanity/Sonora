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
