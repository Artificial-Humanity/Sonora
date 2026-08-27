"""A command a persona shows must survive the prose a persona tells you to put in it.

⚠⚠ #334 — REPRODUCED, THEN REINTRODUCED BY THE AGENT FIXING IT, WITHIN THE HOUR. `REVIEWER.md`
§4 showed the write path as `--comment "…"` and, twenty-five lines below, handed the reader a
model comment containing backticks: *"Verified fixed in `abc1234`; re-ran the gate"*. Inside
DOUBLE quotes bash runs the backticks, so the tool receives `Verified fixed in ; re-ran the
gate` and the comment posts **with its only evidence deleted**, exit status 0, nothing red.

The reintroduction is why this is a test and not a wording fix. On 2026-08-27, while #334 sat
open and had just been read aloud, a new section was added to that same file containing
`--comment "… inside the `if` at 209 …"` — the identical defect, in the identical file.

⚠⚠ **AND THE INVARIANT IS NOT "THIS EXAMPLE HAS NO BACKTICKS", WHICH IS THE VERSION THAT WAS
WRITTEN FIRST AND WAS HALF A FIX.** #334's actual shape is a **JOIN ACROSS TWO PLACES**: the
`"`-quoted template lives in a fenced block and the backticked model comment lives in a
blockquote twenty-five lines away. A scanner looking for both on one line reports the file
CLEAN, because on the page they never meet — they meet in the reader, who substitutes one into
the other. So the property tested is the **template**: a prose-valued option in a command an
agent is shown is SINGLE-quoted, and is therefore safe for any value the reader brings to it.

⚠ WHAT THIS DOES NOT DO, stated so it is not mistaken for a shell linter. It checks the
quoting of five named options inside fenced blocks in the persona set. It cannot see `$(…)`,
an unquoted `$VAR`, a heredoc, or a value carrying a literal single quote — which it skips
rather than mangles, and which would need escaping a reader is unlikely to get right. It is
the narrow guard for the measured defect, not a general claim of shell safety.
"""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
# The documents that hand an agent a command to run. `WORKFLOW.md` is included because
# DEVELOPER.md §3 says it outranks the persona wherever the two differ — so it is read too.
PERSONA_DOCS = ["workflow/REVIEWER.md", "workflow/DEVELOPER.md", "workflow/WORKFLOW.md"]

# The options whose values are free prose written by an agent, and therefore the ones that
# attract backticks. `--branch`, `--severity`, `--kind` and friends take fixed vocabularies
# and are deliberately absent: widening this to every flag would flag correct lines and get
# the guard switched off, which is how a guard stops being there for the real case.
PROSE_OPTS = ("--comment", "--text", "--note", "--body", "--title")

_FENCE = re.compile(r"```[a-z]*\n(.*?)```", re.S)
_DOUBLE_QUOTED = re.compile(r"(%s)\s+\"([^\"]*)\"" % "|".join(map(re.escape, PROSE_OPTS)))


def _fenced(rel):
    return _FENCE.findall((REPO / rel).read_text(encoding="utf-8"))


def test_the_scan_reaches_actual_command_blocks():
    """⚠ FLOOR FIRST. Every assertion below passes over an empty list of blocks, so if the
    fence convention or the paths change, this file reports green while reading nothing —
    the vacuous pass this repo has shipped more than once."""
    blocks = [b for rel in PERSONA_DOCS for b in _fenced(rel)]
    assert len(blocks) >= 10, (
        "only %d fenced blocks found across %s — the fence convention or the paths have "
        "changed and this guard is scanning nothing" % (len(blocks), PERSONA_DOCS))
    assert any("issue.py" in b for b in blocks), (
        "no block invokes issue.py; the population this guard is aimed at is not being read")
    # ⚠ And the options must still OCCUR. If `--comment` were renamed, every assertion below
    # would pass over nothing while the defect walked in under the new name.
    joined = "\n".join(blocks)
    assert sum(o in joined for o in PROSE_OPTS) >= 2, (
        "fewer than two of %s appear in any command block — the flags have been renamed and "
        "this guard is watching a vocabulary nobody uses" % (PROSE_OPTS,))


def test_the_detector_fires_on_the_measured_cases():
    """⚠⚠ A KNOWN-ANSWER FIXTURE, because a scanner that silently stops matching is
    indistinguishable from a clean tree. Both real cases must be caught — #334's own text and
    the line the fix pass reintroduced — and the remedy must NOT be flagged, or the guard
    would argue against its own advice."""
    from_334 = 'issue.py comment 114 --text "Verified fixed in `abc1234`; re-ran the gate"'
    reintroduced = 'issue.py reopen 335 --comment "line 214 is inside the `if` at 209"'
    # ⚠ THE JOIN CASE, and the reason this detector keys on the QUOTE rather than the content.
    # No backtick on this line; the backticked prose that lands here arrives from a blockquote
    # elsewhere in the document. The first version of this test called this line clean.
    the_join = 'issue.py close 114 --comment "…"'
    remedy = "issue.py comment 114 --text 'Verified fixed in `abc1234`; re-ran the gate'"

    assert _DOUBLE_QUOTED.search(from_334), "misses the finding it was written for"
    assert _DOUBLE_QUOTED.search(reintroduced), "misses the line the fix pass reintroduced"
    assert _DOUBLE_QUOTED.search(the_join), (
        "misses the JOIN case — this is the one a same-line backtick scan calls clean, and it "
        "is the shape #334 actually reported")
    assert not _DOUBLE_QUOTED.search(remedy), (
        "single quotes are the remedy; flagging them would make the remedy look wrong")


def test_no_persona_hands_an_agent_a_template_bash_would_rewrite():
    """The invariant. See the module docstring for why it is the quote and not the content."""
    bad = []
    for rel in PERSONA_DOCS:
        for block in _fenced(rel):
            for opt, value in _DOUBLE_QUOTED.findall(block):
                if "'" in value:
                    continue          # needs escaping; out of scope, see the docstring
                bad.append('%s: %s "%s"' % (rel, opt, value[:70]))

    assert not bad, (
        "a persona shows a prose-valued option in DOUBLE quotes:\n"
        + "\n".join("  " + b for b in bad)
        + "\n\n⚠ Inside double quotes, `…` is command substitution — and the prose these "
          "documents tell an agent to write is full of backticks, because that is how prose "
          "marks code. The tool then receives the value with that span DELETED and exits 0, "
          "so the comment posts missing exactly the SHA it was citing (#334, reproduced).\n"
          "  ⚠ The example on the page need not contain a backtick to be the defect. The "
          "value is a placeholder; the backticks arrive from whatever the reader substitutes.\n"
          "  Fix: single-quote the value.")
