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
    # ⚠⚠ #330, FOURTH TIME. THIS MESSAGE NO LONGER CARRIES A REMEMBERED ENTRY COUNT, AND THAT
    # IS THE FIX — the previous three were all "N measured on DATE" literals that rotted.
    #
    # The sequence, because it is funnier and more useful than the rule: 47 at `56e70e5`; 48
    # at `50d55b6` when `pb_logs` was granted; back to **47** at `1dd8080` — the commit that
    # removed `pb_record_mutate`, WHICH IS THE SAME COMMIT THAT RAISED THE LITERAL 47 -> 48.
    # I measured, then edited, then shipped the measurement. Three of the four instances of
    # that error in this repo have been exactly this: a snapshot written down as if it were a
    # baseline.
    #
    # A THRESHOLD is a deliberate ratchet and belongs here. A COUNT is a fact about right now
    # and belongs in the failure output, where it is computed at the moment it is read and
    # cannot be stale. So the floor stays and the number goes — nothing left to re-derive,
    # which is the only version of this fix that has not needed fixing.
    assert len(entries) >= 30, (
        f"only {len(entries)} REVIEWER_ALLOW entries parsed. The floor is 30 — well under any "
        f"real array — so this is almost certainly the PARSE breaking, not entries being "
        f"removed. Check the `REVIEWER_ALLOW=(` regex against the launcher's current shape.")
    # ⚠ #330. THE FLOOR WAS 4 AND THE POPULATION IS 6, AND THE SLACK WAS EXACTLY THE THING
    # THIS PAGE MOST NEEDS. `list` and `show` are the MCP-401 reroute — the commands a
    # reviewer falls back to when the tools start returning 401 — so at a floor of 4 they
    # could BOTH vanish from §4 and the suite would stay green, silently removing the escape
    # hatch from the one section that turns an unreachable tracker into a lost review. A
    # floor has to sit under the population that matters, not under the one that was there
    # when it was written.
    # ⚠ These two floors ARE deliberate ratchets, not snapshots: 6 is the write path
    # (close, comment, file, reopen) plus the 401 reroute (list, show). Losing `list`/`show`
    # costs §4's fallback, not just a line — which is why the floor is on the population that
    # matters rather than on whatever was there when it was written. The live sets go in the
    # message, for the same reason as above.
    assert len(documented) >= 6, (
        f"REVIEWER.md §4 documents only {len(documented)} issue.py subcommands: "
        f"{sorted(documented)}. Floor is 6 — the four write verbs plus `list`/`show`, which "
        f"are the MCP-401 reroute. Losing those costs the fallback, not just a line.")
    assert len(granted) >= 6, (
        f"REVIEWER_ALLOW grants only {len(granted)} issue.py subcommands: {sorted(granted)}. "
        f"Floor is 6, matching the documented set.")


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
    # ⚠ #331. Revoked 2026-08-26. An allowlist cannot forbid ONE OPERATION of a tool, so while
    # this was granted, `operation: "delete"` on the sole record of a finding was held back by
    # REVIEWER.md §1 and nothing else. It can forbid the tool, so it does. Every mention of it
    # on that page is a prohibition; `issue.py` covers every write the role may make.
    assert "pb_record_mutate" not in granted, (
        "REVIEWER_ALLOW grants pb_record_mutate. It accepts operation=delete/bulkDelete, and "
        "the tracker is the sole record that a finding ever existed — so deletion must be "
        "held by the harness, not by a sentence in REVIEWER.md §1. If a reviewer genuinely "
        "needs it, restore it deliberately and say what for; do not re-add it in passing.")


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

    # ⚠ #332 then #333, and the second one is a lesson about guards, not about this page.
    #
    # #332: when `pb_record_mutate` was revoked, §1 was updated and §4 — two sections down,
    # same file, SAME COMMIT — kept saying a direct create "would succeed". The assertion
    # above did not notice, because it only asked whether the string APPEARED.
    #
    # ⚠⚠ #333: the guard added for that was WORSE THAN NOTHING, and measured to fail in BOTH
    # directions at once. It matched `revoked` across the WHOLE page, and §1 — the half that
    # was never wrong — satisfied it. So it PASSED on the exact #332 defect state, and FAILED
    # on correct prose that said "withdrawn". I also described it in the commit as "tied to
    # the ALLOWLIST STATE rather than to a keyword": it is a keyword; the allowlist state only
    # decides whether the keyword is REQUIRED. ⚠ And this file argues against exactly this
    # shape ~60 lines above, where the general `pb_*` version was measured and declined. **The
    # argument was already here and I built the thing anyway, below it.**
    #
    # Below is the remedy the reviewer MEASURED rather than suggested: scope the match to §4,
    # and widen the vocabulary. That inverts both bad rows.
    #
    # ⚠ IT IS STILL A KEYWORD CHECK AND ITS LIMITS ARE STATED, not glossed. What makes it
    # acceptable here and not for `pb_*` generally is the population: ONE tool, with an exact
    # condition (ungranted -> §4 must say so). It can still go red on a rewording outside the
    # vocabulary — so the message says to widen the list or delete the assertion, and NOT to
    # contort the prose to satisfy it. A guard that makes a page worse to read has lost.
    granted = {e.split("__")[-1] for e in _allowlist() if e.startswith("mcp__pocketbase__")}
    if "pb_record_mutate" not in granted:
        s4 = text.split("\n## 4.", 1)
        assert len(s4) == 2, "could not find §4 — has REVIEWER.md been renumbered?"
        section4 = s4[1].split("\n## 5.", 1)[0].lower()
        assert "pb_record_mutate" in section4, (
            "§4 no longer mentions pb_record_mutate at all, so the check below is vacuous.")
        vocab = ("revoked", "withdrawn", "not granted", "no longer granted", "is refused",
                 "cannot call", "removed from the allowlist")
        assert any(v in section4 for v in vocab), (
            "REVIEWER_ALLOW does not grant pb_record_mutate, but §4 never says so — it still "
            "describes a tool the reader cannot call (the #332 shape, which #333 proved this "
            "assertion could not see when it searched the whole page).\n"
            f"Looked in §4 for any of: {vocab}\n"
            "⚠ If §4 is CORRECT and simply words it differently, widen that list or delete "
            "this assertion. Do NOT reword the page to satisfy a keyword match.")
