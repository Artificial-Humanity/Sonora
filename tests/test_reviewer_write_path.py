"""⚠ #321. Every tracker command REVIEWER.md documents must be one the launcher GRANTS.

The defect this exists to prevent, measured: the fix for #318 pointed the reviewer's `state`
moves at `workflow/scripts/issue.py` — correct, and the referee accepts those writes — while
`REVIEWER_ALLOW` in `request_review.sh` contained no `issue.py` entry at all. So the blocker
MOVED rather than went away: the store started accepting the write and the harness started
refusing the command. Janis hit it on every tracker write of that pass.

⚠ The cost is not a refused command. REVIEWER.md §4 tells the reviewer — correctly — that a
tracker it cannot write to means the findings are lost and must go into the summary instead.
A reviewer that runs the documented command, is refused, and does not independently think to
prefix an interpreter lands in that paragraph and abandons filing. **A stale instruction here
costs a complete review**, which is the exact outcome #318's own text was written to prevent,
arriving from the other side.

THE ASSERTION SHAPE IS THE POINT. Both sides are PARSED FROM THEIR REAL ARTIFACTS — the
persona a `claude -p` process is actually handed, and the array actually passed to
`--allowedTools`. Restating either list here would create the third copy and reintroduce the
drift one layer up. (Owed to FerroStep's resident, who used the same shape to tie a printed
hunting list to generated hook text rather than to a second copy of the field names.)
"""

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
REVIEWER_MD = REPO / "workflow" / "REVIEWER.md"
LAUNCHER = REPO / "workflow" / "scripts" / "request_review.sh"

# Verbs `issue.py` exposes that belong to the DEVELOPER, never to the reviewer. ⚠ All four
# are the developer's — none of them is the OWNER's, and an earlier version of this file said
# `escalate` was. The owner's part is DECIDING on an escalated issue (`user_decision`, which
# no agent writes); putting it there is the developer's move. Ruled 2026-08-17, and stated
# from both sides: REVIEWER.md §1 "YOU DO NOT ESCALATE. ESCALATION IS OZZY'S, AND ONLY
# OZZY'S", DEVELOPER.md § "ESCALATION IS YOURS. JANIS DOES NOT ESCALATE". `take`/`grade`/
# `review` are the developer's half of the loop, and the reviewer incrementing `agent_passes`
# would corrupt the spend ceiling.
FORBIDDEN_TO_REVIEWER = ("escalate", "take", "grade", "review")


def _documented_subcommands():
    """Every `issue.py <subcommand>` REVIEWER.md tells the reviewer to run."""
    text = REVIEWER_MD.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"issue\.py\s+([a-z]+)\b", text)))


def _allowlist():
    """The REVIEWER_ALLOW array, read from the launcher rather than restated."""
    text = LAUNCHER.read_text(encoding="utf-8")
    m = re.search(r"^REVIEWER_ALLOW=\((.*?)^\)", text, re.S | re.M)
    assert m, "could not find the REVIEWER_ALLOW array — did the launcher get restructured?"
    body = m.group(1)
    # Quoted entries only; comment lines inside the array must not be mistaken for grants.
    body = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    return re.findall(r'"([^"]+)"', body)


def _granted_issue_subcommands():
    return sorted({m.group(1) for e in _allowlist()
                   for m in [re.match(r"Bash\(\.?/?workflow/scripts/issue\.py ([a-z]+):", e)]
                   if m})


# --------------------------------------------------------------- floors, before any claim

def test_both_sides_actually_parsed_something():
    """⚠ The empty-enumeration trap, and it would be invisible here.

    If either regex stopped matching — a rename, a reformat, a quoting change — every
    assertion below would pass over an empty set and this file would report green while
    checking nothing. Floors first, with the measured numbers in the message.
    """
    documented = _documented_subcommands()
    granted = _granted_issue_subcommands()
    entries = _allowlist()
    assert len(entries) >= 30, (
        f"only {len(entries)} REVIEWER_ALLOW entries parsed — **47** measured 2026-08-26. "
        f"The array parse is probably broken, not the array.")
    assert len(documented) >= 4, (
        f"only {len(documented)} issue.py subcommands found in REVIEWER.md — **4** measured "
        f"2026-08-26 (close, comment, file, reopen). Did §4's command block move or change "
        f"spelling?")
    assert len(granted) >= 4, (
        f"only {len(granted)} issue.py grants parsed from REVIEWER_ALLOW — **6** measured "
        f"2026-08-26 (the 4 documented, plus list and show).")


# --------------------------------------------------------------- the actual invariant

def test_reviewer_md_does_not_document_a_forbidden_verb():
    """⚠ THE HOLE THE SUBTRACTION BELOW OPENED, AND IT SWALLOWED THE CASE THIS FILE EXISTS FOR.

    `test_every_documented_command_is_granted` subtracts FORBIDDEN_TO_REVIEWER from the
    documented set, so that documenting a verb the allowlist deliberately withholds is not
    reported as a missing grant. Correct as far as it goes — but with nothing else asserting
    it, REVIEWER.md could drift back to documenting `issue.py escalate` and the whole suite
    stayed green. MEASURED 2026-08-26 by adding that exact line to §4: **9 passed.**

    A verb that is both documented and deliberately ungranted is the WORST of the two states
    this file guards, not an exempt one: the reviewer is told to run a command, the harness
    refuses it, and §4 tells it that a tracker it cannot write to means the review is lost.
    So it gets its own assertion, ahead of the subtraction.
    """
    documented = set(_documented_subcommands())
    bad = sorted(documented & set(FORBIDDEN_TO_REVIEWER))
    assert not bad, (
        f"REVIEWER.md documents {bad}, which the reviewer is forbidden to run and which "
        f"REVIEWER_ALLOW deliberately withholds. `escalate` is the DEVELOPER's (owner ruling, "
        f"2026-08-17: DEVELOPER.md \u00a7 'ESCALATION IS YOURS. JANIS DOES NOT ESCALATE'); "
        f"take/grade/review are the worker's. Either the page is wrong, or the ruling "
        f"changed and the allowlist and §1 have to change with it.")


def test_every_documented_command_is_granted():
    """#321 itself. Documented-but-refused is the state that costs a review.

    ⚠ The FORBIDDEN subtraction here is not an exemption — see the test above, which fails on
    that case directly. Without it this assertion would report a forbidden verb as "missing a
    grant", pointing the fixer at the allowlist when the defect is in the page.
    """
    documented = set(_documented_subcommands())
    granted = set(_granted_issue_subcommands())
    missing = sorted(documented - granted - set(FORBIDDEN_TO_REVIEWER))
    assert not missing, (
        "REVIEWER.md documents issue.py subcommand(s) that REVIEWER_ALLOW does not grant:\n"
        + "\n".join(f"  issue.py {s}" for s in missing)
        + "\n\nA reviewer running these verbatim is refused by the harness, and §4 tells it "
          "that an unwritable tracker means the whole review goes in the summary. Add "
          '"Bash(workflow/scripts/issue.py <sub>:*)" AND the "./" spelling to REVIEWER_ALLOW.')


def test_the_reviewer_is_not_granted_the_developers_verbs():
    """⚠ The 2026-08-17 escalation ruling as a MECHANISM, not as prose.

    REVIEWER.md §1 has said "YOU DO NOT ESCALATE" since 2026-08-17 and nothing enforced it.
    Granting `Bash(workflow/scripts/issue.py:*)` — the obvious one-line fix for #321 — would
    have pre-approved the single move the role is forbidden to make. This is why #321's fix
    enumerates subcommands instead.
    """
    granted = set(_granted_issue_subcommands())
    leaked = sorted(granted & set(FORBIDDEN_TO_REVIEWER))
    assert not leaked, (
        f"REVIEWER_ALLOW grants the reviewer {leaked}, which REVIEWER.md forbids it. "
        f"`escalate` is the DEVELOPER's, not yours and not the owner's — the owner DECIDES on "
        f"an escalated issue, the developer is what puts it there (2026-08-17). "
        f"take/grade/review are the developer's too.")


def test_pb_auth_superuser_is_never_granted():
    """⚠ The one `pb_*` tool that must stay out, and the reason is not access — it is disclosure.

    `pb_auth_superuser` takes the password as a TOOL ARGUMENT, so calling it writes a live
    superuser credential into the session transcript in plain text. REVIEWER.md §4 tells the
    reviewer never to reach for it when the MCP goes stale; that was prose only, and prose is
    what this file exists to convert into a mechanism. The reviewer needs no auth tool at all:
    the MCP is pre-authenticated, and `issue.py` mints its own token in-process.

    ⚠ THE GENERAL VERSION OF THIS CHECK WAS CONSIDERED AND DELIBERATELY NOT BUILT. "Every
    `pb_*` tool REVIEWER.md names must be granted" sounds like the #321 guard one layer out,
    and it is a trap: measured 2026-08-26, a naive scan returns `pb_auth_superuser` (named in
    a PROHIBITION — the desired state) and `pb_hooks` (a DIRECTORY in a file path, not a tool
    at all). Both would go red on correct prose, and a guard that goes red on correct code
    gets switched off. Distinguishing "named as an instruction" from "named as a warning"
    needs a judgement this file cannot make reliably, so it asserts the invariant it CAN state
    exactly and leaves the fuzzy one alone.
    """
    granted = {e.split("__")[-1] for e in _allowlist() if e.startswith("mcp__pocketbase__")}
    assert granted, "no mcp__pocketbase__ grants parsed — the array parse is broken"
    assert "pb_auth_superuser" not in granted, (
        "REVIEWER_ALLOW grants pb_auth_superuser. It takes the password as a tool argument, "
        "so any call writes a live superuser credential into the transcript in plain text. "
        "The reviewer never needs it: the MCP is pre-authenticated and issue.py mints its "
        "own token in-process.")
    for writer in ("pb_collection_create", "pb_collection_delete", "pb_collection_patch",
                   "pb_settings", "pb_backup"):
        assert writer not in granted, (
            f"REVIEWER_ALLOW grants {writer}, which mutates the store's SHAPE rather than a "
            f"record. The reviewer reads and files; it does not administer the tracker.")


def test_no_unscoped_grant_of_the_whole_script():
    """The wildcard that would silently re-open everything the test above closes."""
    bad = [e for e in _allowlist()
           if re.match(r"Bash\(\.?/?workflow/scripts/issue\.py:?\*?\)?$", e)
           or re.match(r"Bash\(\.?/?workflow/scripts/issue\.py:\*\)$", e)]
    assert not bad, (
        f"REVIEWER_ALLOW grants issue.py unscoped: {bad}. That pre-approves `escalate`, "
        f"which the owner reserved to Ozzy. Enumerate the subcommands instead.")


@pytest.mark.parametrize("sub", ["file", "close", "reopen", "comment"])
def test_both_spellings_are_granted(sub):
    """⚠ Prefix matches are literal. `./workflow/...` and `workflow/...` are different
    strings, and the launcher's own request_review entry already carries both — a reviewer
    that writes the `./` form would otherwise be refused for a path-prefix nobody thought
    about."""
    entries = set(_allowlist())
    for prefix in ("", "./"):
        want = f"Bash({prefix}workflow/scripts/issue.py {sub}:*)"
        assert want in entries, f"missing grant: {want}"


def test_reviewer_md_documents_the_write_path_at_all():
    """A positive control for the parse above: if §4 stopped naming issue.py, every
    documented-vs-granted assertion would pass vacuously and #318 would be back."""
    text = REVIEWER_MD.read_text(encoding="utf-8")
    assert "workflow/scripts/issue.py" in text
    assert "refereed" in text.lower(), "§4 no longer explains WHY the MCP write is refused"
    assert "pb_record_mutate" in text, "§4 no longer warns against the tool that is refused"
