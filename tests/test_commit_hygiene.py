"""Commit authorship and trailers on the current branch — checked, not remembered.

WHY THIS FILE EXISTS (2026-08-18, issue #101)
---------------------------------------------
Two rules in `PERSONA.md` §1 had no enforcement:

  * commits carry a **`Co-Authored-By:` trailer naming the agent**, because the author line is
    the repo's configured identity — the owner's, deliberately — so a forgotten trailer does
    not error, it silently produces a commit that credits nobody;
  * **no commit is AUTHORED as the agent** — the author line stays the owner's, and re-authoring
    to an agent misattributes a human's work.

⚠⚠ **THESE TWO RULES INVERTED ON 2026-09-16** (owner), when the roster file and
its resolver script were removed and `PERSONA.md` became the single identity source. Sonora
used to do the opposite: the agent was the AUTHOR via a `-c` pair and a co-author trailer was
forbidden as misattribution. **Both spellings are enforceable and only one is live** — the
detection machinery below was built for the old rule and is reused unchanged, because what it
measures (how git parses a trailer) does not depend on which direction the rule points.

⚠ **BOTH WERE BROKEN FOR EIGHT COMMITS AND NOTHING NOTICED.** The trailer came from a
`CLAUDE.md` at the workspace root that had already been DELETED; an agent went on obeying it
from a summary written before the deletion. A reviewer reading `PERSONA.md` caught it by
hand. That is the definition of a rule without a mechanism, and `AGENTS.md` §1 is explicit
that a rule is not one.

⚠ **HISTORY IS NOT REWRITTEN.** The owner's decision, 2026-08-18: leave the existing commits
alone, fix the hygiene going forward. So this checks commits **after** `GRANDFATHERED_THROUGH`
and says plainly which ones it is not looking at — an exemption that cannot quietly widen.

⚠⚠ **THE SCOPE IS THIS BRANCH'S OWN UNMERGED WORK, AND THE FIRST VERSION GOT THAT WRONG**
(issue #102). It checked every commit reachable from `HEAD` after the boundary, which is not
"the agent's commits" — it is *everyone's*. Two consequences, both measured on 2026-08-18:

  * A `git merge --no-ff` runs with no `-c` pair, so the merge commit
    carries the configured identity. Performing that merge in a throwaway clone produced
    `fcff394 lmcfarlin <2363604+lmcfarlin@users.noreply.github.com>` and turned this file red
    on `main`. **The script that lands a branch was the thing that broke the suite.**
  * The owner is 388 of 396 commits on `main` and hand-commits there regularly. PERSONA.md
    §1 says the configured identity was left as theirs **deliberately**, so the guard
    forbade exactly what the repo permits on purpose.

And the failure message told whoever saw it to re-author the commit as Sonya. The likely
reader was the owner, looking at their own work; a guard whose remedy misattributes a human's
commits to an agent is worse than no guard, since #101 was about attribution being silently
wrong and this made it loudly wrong in the other direction.

So the range is `BASE..HEAD`, non-merge, after the boundary — the commits this lane produced
and has not yet landed. On the base branch itself there is nothing to check, and this file
SKIPS with a message rather than passing quietly.
"""

import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _agent():
    """The agent identity, FROM `PERSONA.md`, never typed out here.

    ⚠⚠ THIS HAS NOW READ FOUR DIFFERENT SOURCES, AND THE FIRST ONE FAILED SILENTLY. It read
    the review lane's `config.env` and fell back to hardcoded values when the file was missing;
    that fallback fired for real when the config was deleted on 2026-09-15, these tests went on
    passing, and the "second definition that drifts" the old docstring warned against became the
    ONLY definition — still under a comment insisting it was not typed out here. It then read
    a roster file, which was removed on 2026-09-16 with its resolver.

    ⚠⚠ THE FOURTH IS THE TRAILER ITSELF, BECAUSE PARSING PROSE ABOUT IT BROKE `main`. Until
    2026-09-17 this read two sentences — `You are Sonya.` and `Your co-authoring of git commits
    will be done as "…"`. `c4840e4` rewrote `PERSONA.md` and neither sentence survived, so the
    floor test below went red on `main` and every check under it SKIPPED. Nothing in that commit
    was wrong; the guard had made a claim about WORDING rather than about identity.

    So it now reads the `Co-authored-by:` line the persona spells out verbatim — the artifact a
    commit has to carry. That is far more stable than a sentence about it, because rewording it
    changes what the rule REQUIRES rather than merely how it is described. ⚠ It is not immune:
    the anchor is sensitive to Markdown formatting too, and an indented, bolded or backticked
    copy of the line REFUSES rather than matches. Refusing loudly is the intended failure — there
    is no fallback here by design — but it is the same class of event that broke the old parse,
    so `_parse_agent` below is tested against those shapes rather than trusted against them.

    ⚠ The old regex had a second defect that a wording repair would have left live. Over a
    persona whose opening line continues past the name, `You are (.+?)` up to a line-final period
    returns the whole clause as the NAME — parsing cleanly, matching no trailer ever written.
    (Measured 2026-09-17 against the persona of that date. The name is deliberately not quoted
    here: these personas get renamed, and a quoted example rots into a false statement.)

    **A default that silently replaces its own source is worse than no source**, because the test
    keeps reporting green while measuring something else. So: no default, at any point. A
    `PERSONA.md` this cannot parse is a refusal, reported where it is used.

    ⚠ THE OWNER RENAMES THESE AGENTS DELIBERATELY AND OFTEN — the name is a mnemonic for the
    project, not a stable key. That is exactly why it is parsed rather than typed: a rename is an
    edit to one file, not a sweep.
    """
    path = os.path.join(REPO, "PERSONA.md")
    with open(path, encoding="utf-8") as f:
        return _parse_agent(f.read())


def _parse_agent(text):
    """`(name, address)` from the persona's `Co-authored-by:` line. PURE, so it is testable.

    ⚠ IT IS SPLIT OUT FOR EXACTLY ONE REASON: `_agent()` reads a fixed path, so the only input
    the suite could ever hand it is a file that currently parses — and a guard whose refusal path
    has no control is the shape AGENTS.md §5c is about. Everything else in this file is already
    parameterised for that reason (`_commits(repo=…)`, `carries_agent_trailer(…, agent=…)`).
    `test_the_persona_parse_refuses_what_it_should` is the control.
    """
    # The trailer alone on its line, which is what git parses and what a commit must carry.
    # Line-anchored so a sentence mentioning the trailer mid-line is not mistaken for one.
    # ⚠ `\r?$` rather than `$`: a CRLF line ending would otherwise leave `\r` inside the
    # address, and the identity would differ from the commit's by one invisible byte.
    hits = re.findall(r"^Co-authored-by:[ \t]*(.+?)[ \t]*<([^>]+)>[ \t]*\r?$",
                      text, re.M | re.I)
    if not hits:
        raise ValueError(
            "PERSONA.md states no `Co-authored-by: Name <address>` line — identity cannot be "
            "resolved. That line is the agent's identity here; a persona without one leaves "
            "every commit check below with nothing to compare against")
    if len(hits) > 1:
        # ⚠ NOT "take the first". A persona that documents a counterexample — an address NOT to
        # use — puts a second line here, and first-match would adopt whichever came first in the
        # file. Two candidates is an ambiguous document, and the refusal names both.
        raise ValueError(
            "PERSONA.md states %d `Co-authored-by:` lines and identity must be unambiguous: %s"
            % (len(hits), ", ".join("%s <%s>" % h for h in hits)))
    name, email = hits[0]
    return name.strip(), email.strip()


try:
    AGENT, PERSONA_ERROR = _agent(), None
except Exception as e:                      # noqa: BLE001 — surfaced in the tests, not at import
    # ⚠ NOT raised here. A module-scope raise is a COLLECTION error, and a collection error
    # aborts the whole session rather than failing this file — which is how one unreadable
    # file takes the suite down with it.
    AGENT, PERSONA_ERROR = None, "%s: %s" % (type(e).__name__, e)

# ⚠ `main` is this repo's base branch and is not an identity fact. It is written here because
# there is nowhere better since `config.env` went, and because it has never varied. If it ever
# does, it needs a declared home rather than this line.
BASE_BRANCH = "main"

# ⚠ THE KEY, NOT THE WHOLE LINE. The name and address come from `PERSONA.md` above, so a rename
# does not touch this file.
TRAILER_KEY = "Co-Authored-By"

# ⚠ THE BOUNDARY, AND IT IS A COMMIT, NOT A DATE. Everything up to and including this SHA
# predates the owner's decision and is deliberately exempt. A date would drift with the
# clock; a SHA is a fact. `80be925` is the last commit made under the old habit.
GRANDFATHERED_THROUGH = "80be925"


def _base_ref(repo=REPO):
    """The remote-tracking base if there is one, else the local branch, else None."""
    for ref in ("refs/remotes/origin/" + BASE_BRANCH, "refs/heads/" + BASE_BRANCH):
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
                          cwd=repo, capture_output=True).returncode == 0:
            return ref
    return None


def _on_base_branch(repo=REPO):
    r = subprocess.run(["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
                       cwd=repo, capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == BASE_BRANCH


def _commits(repo=REPO, boundary=GRANDFATHERED_THROUGH):
    """(sha, author, email, trailers, body) for THIS BRANCH'S own unmerged commits after the
    boundary.

    ⚠ FIVE FIELDS, NOT FOUR (#251). `0afe042` added `%(trailers)` and updated every caller,
    but left this line reading `(sha, author, email, body)` — so the next caller written from
    the docstring unpacks four values and raises `ValueError`. Nothing was broken at the time,
    which is exactly why it survived: the cost lands on whoever writes the next caller.

    ⚠ `--no-merges`, and scoped to `BASE..HEAD` (issue #102). A merge commit is made by
    a hand merge without a `-c` pair and carries the configured identity, which is the
    owner's on purpose; and everything already on the base branch is history this guard was
    never given a mandate over. Empty when the boundary is not an ancestor — a branch cut
    from elsewhere is not evidence of anything."""
    base = _base_ref(repo)
    if base is None or _on_base_branch(repo):
        return []
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", boundary, "HEAD"],
                         cwd=repo, capture_output=True)
    if anc.returncode != 0:
        return []
    r = subprocess.run(
        # ⚠ `%(trailers)` AS WELL AS `%B` — git's own parse, not a substring of the body.
        # See test_every_commit_carries_the_agent_trailer for why.
        ["git", "log", "--no-merges", f"{base}..HEAD", f"^{boundary}",
         "--format=%H%x1f%an%x1f%ae%x1f%(trailers)%x1f%B%x1e"],
        cwd=repo, capture_output=True, text=True)
    out = []
    for rec in r.stdout.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, name, email, trailers, body = rec.split("\x1f", 4)
        # ⚠ FULL SHA, TRUNCATED ONLY WHERE IT IS PRINTED (issue #108). This used to store
        # `sha[:9]`, and the regression test below then compared that set against `%h`
        # output — seven characters here, because `core.abbrev=auto` scales with repo size.
        # A nine-character string never equals a seven-character one, so BOTH of its
        # assertions were incapable of failing, whatever the range contained. Comparing
        # full object names removes the question rather than picking a matching width.
        out.append((sha, name, email, trailers, body))
    return out


def _short(sha):
    return sha[:9]


def carries_agent_trailer(trailers, body, agent=None):
    """True if this commit carries the agent's co-author line, by EITHER mechanism.

    ⚠ TWO CHECKS, BECAUSE EACH ALONE HAS A MEASURED HOLE (#242), AND THE INVERSION MADE THE
    SECOND ONE MATTER MORE. `%(trailers)` is git's own parse and is the right primary: it
    excludes a mention of the name in ordinary prose.

    But git only parses a trailer block in the LAST paragraph. A commit whose message is
    `…trailer…` + blank line + one closing sentence carries the line verbatim and parses as
    having no trailers at all. Measured over four shapes in a throwaway repo.

    ⚠⚠ **UNDER THE OLD RULE THAT GAP LET A FORBIDDEN TRAILER THROUGH; UNDER THE CURRENT ONE IT
    FAILS A COMMIT THAT IS CORRECT.** Same hole, opposite symptom — a developer who writes a
    closing sentence after their trailer would be told they omitted it. The body check is what
    prevents that, so it is not redundant in either direction.

    So: the parsed field, OR a LINE-ANCHORED match on the body. Line-anchored is what keeps
    prose out — a sentence mentioning the trailer mid-line does not start a line with
    `Co-Authored-By:`.
    ⚠ THE SECOND CHECK MATCHES THE TRAILER'S SHAPE, NOT ITS WORDS. A real trailer is
    `Key: Name <address>` alone on its line. Prose is not, however it wraps.

    ⚠⚠ BOTH COMPARISONS ARE CASE-FOLDED ON THE KEY, AND THE FIRST ONE WAS NOT UNTIL 2026-09-17.
    `TRAILER_KEY` is spelled `Co-Authored-By` and `PERSONA.md` spells the line it tells an agent
    to copy `Co-authored-by`. Git treats trailer keys case-insensitively; a Python `in` does not.
    So an agent copying the persona verbatim — which is now exactly what it is told to do —
    produced commits where the branch this docstring calls "the right primary" never fired, and
    every such commit passed on the body fallback alone. Measured on this branch's own HEAD.

    ⚠ AN EARLIER VERSION OF THIS DOCSTRING GAVE A FALSE REASON FOR KEEPING THE FIRST CHECK: that
    a trailer with no `<address>` is caught by it and missed by the regex. Both interpolate
    `<%s>`, so `Co-Authored-By: Sonya` with no address is caught by NEITHER (measured). The real
    and sufficient reason is the last-paragraph one above.
    """
    name, email = agent if agent else AGENT
    # ⚠ `.lower()` on both sides, not on the key alone: the name and address are compared as
    # written, and git itself does not case-fold those. This folds the whole comparison, which
    # is deliberately the looser of the two — the body check below is already `re.I`.
    if ("%s: %s <%s>" % (TRAILER_KEY, name, email)).lower() in trailers.lower():
        return True
    pattern = r"%s:\s*%s\s*<%s>\s*" % (re.escape(TRAILER_KEY), re.escape(name), re.escape(email))
    return any(re.fullmatch(pattern, line, re.I) for line in body.splitlines())


# ⚠ THE CONTROL FOR THE PARSE, and the reason `_parse_agent` is a separate function at all.
# `_agent()` reads one fixed path, so the only document the suite can hand it is one that
# currently parses — which is a guard with no broken case, the shape AGENTS.md §5c refuses.
# Every row below was RUN, not reasoned about; the ones that refuse are the point of the table.
_GOOD = "Co-authored-by: Sonya <Sonya@artificialhumanity.io>"
_PARSE_CASES = [
    ("the canonical line", _GOOD, ("Sonya", "Sonya@artificialhumanity.io")),
    ("the key in the other casing", "CO-AUTHORED-BY: A <a@b>", ("A", "a@b")),
    ("a CRLF line ending leaves no \\r in the address", "Co-authored-by: A <a@b>\r\n", ("A", "a@b")),
    ("prose around a real line is irrelevant", "text\n\n%s\n\nmore text" % _GOOD,
     ("Sonya", "Sonya@artificialhumanity.io")),
    ("no trailer line at all", "You are Sonya, and you co-author your commits.", None),
    ("the phrase mid-line in prose", "Every commit has a Co-authored-by: A <a@b> trailer.", None),
    ("line-initial but continued by prose", "Co-authored-by: A <a@b> is the required trailer.", None),
    ("an indented copy", "    Co-authored-by: A <a@b>", None),
    ("a bolded copy", "**Co-authored-by: A <a@b>**", None),
    ("no address", "Co-authored-by: Sonya", None),
    ("two candidates", "%s\nCo-authored-by: Other <other@example.com>" % _GOOD, None),
]


@pytest.mark.parametrize("label,text,expected", _PARSE_CASES,
                         ids=[c[0] for c in _PARSE_CASES])
def test_the_persona_parse_refuses_what_it_should(label, text, expected):
    """⚠ A REFUSAL IS THE RESULT HERE, not an error to be tolerated.

    Four of these refuse on FORMATTING rather than on wording — indented, bolded, address-less,
    ambiguous. That is deliberate and it is the residual risk written down: a meaning-preserving
    Markdown edit to `PERSONA.md` can still break identity resolution. It will do so LOUDLY, by
    failing the floor test with the persona named, which is what the predecessor of this parse
    did not do.
    """
    if expected is None:
        with pytest.raises(ValueError):
            _parse_agent(text)
    else:
        assert _parse_agent(text) == expected


def test_the_two_candidate_refusal_names_both():
    """A refusal that does not say WHICH two lines sends the reader to re-read the whole file."""
    with pytest.raises(ValueError) as e:
        _parse_agent("%s\nCo-authored-by: Other <other@example.com>" % _GOOD)
    assert "Sonya@artificialhumanity.io" in str(e.value) and "other@example.com" in str(e.value)


def test_the_persona_is_readable_and_this_file_is_not_guessing():
    """⚠ FLOOR. Every identity assertion below rests on `PERSONA.md` being parseable.

    Two predecessors of this file read a source that had been deleted. The first caught the
    error and fell back to hardcoded values, so it kept passing while measuring its own
    defaults. This is what stops that shape returning: if the persona cannot be parsed, the
    failure names the persona instead of surfacing as a mystified identity mismatch twenty
    lines down.
    """
    assert PERSONA_ERROR is None, "PERSONA.md could not be read: %s" % PERSONA_ERROR
    assert AGENT and all(AGENT), "PERSONA.md gave an empty agent identity: %r" % (AGENT,)
    assert "@" in AGENT[1], "the agent email looks wrong: %r" % (AGENT[1],)


# ⚠ EVERYTHING BELOW THAT USES `AGENT` IS SKIPPED WHEN THE PERSONA IS UNPARSEABLE, and
# `test_the_persona_is_readable_and_this_file_is_not_guessing` above is deliberately NOT — so
# the suite reports ONE failure naming the persona instead of several `TypeError: 'NoneType' is
# not subscriptable` from tests that were never the cause. A skip beside an unconditional
# failure is not a hidden result; it is the same result, said once.
_needs_persona = pytest.mark.skipif(PERSONA_ERROR is not None,
                                    reason="PERSONA.md unparseable — see the floor test above")


@_needs_persona
def test_the_commit_template_states_the_same_identity_as_the_persona():
    """⚠ THE LAST UNGUARDED COPY OF THE IDENTITY IN THIS TREE, and `.gitmessage` says so itself:
    *"a template cannot read a file — so a rename has to touch both."* True, and until now the
    second copy was checked by nobody. It is guardable the moment identity is parsed from a
    trailer line, because the template carries one too — so this compares the two trailers
    rather than restating either.

    ⚠ THE KEY IS COMPARED CASE-INSENSITIVELY and the name and address are not. Git folds trailer
    keys and does not fold the rest, so this is the same tolerance `carries_agent_trailer` uses;
    the template spells the key `Co-Authored-By` and the persona spells it `Co-authored-by`, and
    both are correct.
    """
    path = os.path.join(REPO, ".gitmessage")
    if not os.path.exists(path):        # the template is a local convenience, not a guarantee
        pytest.skip(".gitmessage is not present in this checkout")
    with open(path, encoding="utf-8") as f:
        template = f.read()
    assert _parse_agent(template) == AGENT, (
        ".gitmessage states a different agent identity from PERSONA.md:\n"
        "  .gitmessage: %r\n  PERSONA.md : %r\n"
        "PERSONA.md is the source — a rename has to touch both, and this is the check that "
        "says so at commit time rather than after the commit." % (_parse_agent(template), AGENT))



@_needs_persona
def test_every_commit_carries_the_agent_trailer():
    """⚠ A FORGOTTEN TRAILER DOES NOT ERROR — it produces a normal-looking commit crediting
    nobody, and nothing downstream says so. That silence is the whole reason this is a test."""
    bad = [f"  {_short(sha)}  {name}" for sha, name, _e, trailers, body in _commits()
           if not carries_agent_trailer(trailers, body)]
    assert not bad, (
        "unmerged commits on this branch carry no `%s: %s <%s>` trailer:\n"
        % (TRAILER_KEY, AGENT[0], AGENT[1]) + "\n".join(bad)
        + "\n\n⚠ READ WHICH OF THESE TWO IT IS BEFORE ACTING.\n"
        "  * An AGENT wrote them and forgot the trailer. Add it while the commit is still "
        "unpushed:\n"
        f"      git commit --amend --no-edit --trailer \"{TRAILER_KEY}: {AGENT[0]} "
        f"<{AGENT[1]}>\"\n"
        "  * The OWNER hand-committed on an agent's branch. Then the absence is CORRECT and "
        "must not be 'fixed' — their own commits are theirs alone. Move the commit to its own "
        "branch, or add its SHA to an exemption here as a deliberate decision.\n"
        "⚠ This guard cannot tell those two apart, and it is not trying to. Do not add an "
        "agent trailer to a human's commit to make it pass.")


@_needs_persona
def test_no_commit_is_authored_as_the_agent():
    """⚠ THE OLD CONVENTION'S HABIT, CAUGHT FROM THE OTHER SIDE.

    Until 2026-09-16 the agent WAS the author here, via a `-c` pair. A session carrying that
    habit — or an agent arriving from a sibling repo where it is still correct — re-authors a
    commit to itself and the work stops being attributable to the owner. Nothing warns.
    """
    bad = [f"  {_short(sha)}  {name} <{email}>" for sha, name, email, _t, _b in _commits()
           if (name, email) == AGENT]
    assert not bad, (
        "commits are AUTHORED as the agent, but the author line here stays the owner's and the "
        "agent goes in a trailer:\n" + "\n".join(bad)
        + "\n\nThis is the pre-2026-09-16 convention, or a habit carried from a sibling repo. "
        "Re-author to the configured identity and add the trailer instead.")


def test_the_grandfather_boundary_still_exists_and_is_named():
    """⚠ A guard whose exemption points at nothing is not exempting — it is disarmed.

    If the boundary SHA ever leaves this history (a rebase, a fresh clone with a shallow
    fetch), `_commits()` returns empty and BOTH tests above pass while checking nothing. This
    is the test that notices.
    """
    r = subprocess.run(["git", "cat-file", "-e", GRANDFATHERED_THROUGH + "^{commit}"],
                       cwd=REPO, capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"{GRANDFATHERED_THROUGH} is not in this checkout's history — the "
                    f"hygiene checks above are inert here and are NOT reporting a pass")
    anc = subprocess.run(["git", "merge-base", "--is-ancestor",
                          GRANDFATHERED_THROUGH, "HEAD"], cwd=REPO, capture_output=True)
    if anc.returncode != 0:
        pytest.skip(f"{GRANDFATHERED_THROUGH} is not an ancestor of HEAD — nothing to check "
                    f"on this branch, and that is not a pass either")
    if _base_ref() is None:
        pytest.skip(f"no {BASE_BRANCH} ref in this checkout — the range BASE..HEAD cannot be "
                    f"formed, so the checks above are inert and are NOT reporting a pass")
    if _on_base_branch():
        pytest.skip(f"HEAD is {BASE_BRANCH} — this guard covers a branch's own unmerged "
                    f"commits, so there is nothing here to check (issue #102)")
    assert True


OWNER = ("lmcfarlin", "owner@example.invalid")


def _git(repo, *args, **kw):
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, env=env, **kw)
    assert r.returncode == 0, f"git {' '.join(args)} failed:\n{r.stderr}"
    return r.stdout.strip()


def _commit(repo, msg, who, n=[0]):
    # ⚠ THE CHOKE POINT. Every fixture that builds a history passes an identity through here,
    # so guarding the two call sites missed three tests that reached it by another route and
    # died on `'NoneType' is not subscriptable` — a failure naming this line rather than the
    # persona. Guard where the value is USED, not where you remember passing it.
    if who is None or not all(who):
        pytest.skip("PERSONA.md unparseable — see the floor test; %s" % PERSONA_ERROR)
    n[0] += 1
    (repo / f"f{n[0]}.txt").write_text(msg, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", f"user.name={who[0]}", "-c", f"user.email={who[1]}",
         "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def landed_branch(tmp_path):
    """A branch that has merged the base INTO itself, which is where `--no-merges` bites.

    ⚠ BUILT RATHER THAN OBSERVED, and that is the whole point of issue #108. The two
    properties this guard holds — no merge commits, nothing already on base — are
    INDISTINGUISHABLE from the real repo's current shape: `origin/main..HEAD` contains no
    merge, and it happens to equal `GRANDFATHERED_THROUGH..HEAD` on this branch. Measured:
    with `--no-merges` deleted, or the range reverted to `{GRANDFATHERED_THROUGH}..HEAD`,
    the whole file stayed green. A guard for a condition the repo does not currently
    exhibit has to create the condition.

    ⚠ Note the shape that does NOT work here, because it is the obvious one: landing the
    branch onto base and checking from base. After a branch lands, `base..HEAD`
    on the branch is EMPTY — base contains everything the branch had — and on `main` the
    guard skips by design. The first version of this fixture did that and its own control
    test caught it.
    """
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", BASE_BRANCH)
    boundary = _commit(repo, "boundary", OWNER)          # stands in for GRANDFATHERED_THROUGH
    _git(repo, "branch", "work")
    owner_on_base = _commit(repo, "owner hand-commit on base", OWNER)
    _git(repo, "checkout", "-q", "work")
    dev = _commit(repo, "agent work on the branch", AGENT)
    # Catching the branch up on base. `git merge` takes the CONFIGURED identity — no `-c`
    # pair, exactly as a hand merge does — so this commit is owner-authored and sits
    # squarely inside `base..HEAD`.
    _git(repo, "-c", f"user.name={OWNER[0]}", "-c", f"user.email={OWNER[1]}",
         "merge", "--no-ff", BASE_BRANCH, "-q", "-m", f"merge {BASE_BRANCH} into work")
    merge = _git(repo, "rev-parse", "HEAD")
    return {"repo": repo, "boundary": boundary, "owner_on_base": owner_on_base,
            "dev": dev, "merge": merge}


def test_a_merge_commit_never_enters_the_range(landed_branch):
    """The failure #102 was filed about, reproduced in miniature."""
    shas = {sha for sha, _n, _e, _t, _b in _commits(landed_branch["repo"], landed_branch["boundary"])}
    assert landed_branch["merge"] not in shas, (
        "the merge commit is in the range, so the author check will fail on it — and it is "
        "made by a hand merge with the owner's configured identity, on purpose")


def test_a_commit_already_on_the_base_branch_never_enters_the_range(landed_branch):
    shas = {sha for sha, _n, _e, _t, _b in _commits(landed_branch["repo"], landed_branch["boundary"])}
    assert landed_branch["owner_on_base"] not in shas, (
        "a commit already on the base branch is in the range. That is history this guard "
        "has no mandate over, and the owner is 388 of 396 commits there")


def test_the_branch_own_work_DOES_enter_the_range(landed_branch):
    """⚠ THE CONTROL. Both assertions above are 'X is not in a set', which an empty set
    satisfies for free — the exact way the first version of this test passed while checking
    nothing. This one fails if the range has gone inert."""
    shas = {sha for sha, _n, _e, _t, _b in _commits(landed_branch["repo"], landed_branch["boundary"])}
    assert landed_branch["dev"] in shas, (
        "the branch's own commit is NOT in the range, so the two exclusion tests above are "
        "passing on an empty set and prove nothing")


def test_the_range_excludes_what_it_is_not_a_guard_over():
    """⚠ THE #102 REGRESSION TEST, as a property of the range rather than of one SHA.

    Two things must stay out of `_commits()`: anything already on the base branch, and merge
    commits. The first is history this guard has no mandate over; the second is made by
    a hand merge with the configured identity, and would fail the moment a branch lands.
    """
    base = _base_ref()
    if base is None or _on_base_branch():
        pytest.skip("no branch range to check here")
    shas = {sha for sha, _n, _e, _t, _b in _commits()}
    if not shas:
        pytest.skip("no unmerged commits after the boundary on this branch")
    assert all(len(s) == 40 for s in shas), "expected full object names from --format=%H"

    def _log(*args):
        return set(subprocess.run(["git", "log", *args, "--format=%H"],
                                  cwd=REPO, capture_output=True, text=True).stdout.split())

    # ⚠ THE CONTROL. Both assertions below are "an intersection is empty", which is also
    # what a comparison of two incompatible formats produces — that is exactly how the
    # first version of this test passed while checking nothing (issue #108). So first
    # prove the two sides CAN meet: HEAD is on HEAD's own log.
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    assert head in _log("-1", "HEAD"), (
        "the comparison itself is broken — a commit is not matching its own log entry, so "
        "the two assertions below would pass no matter what the range held")

    overlap = shas & _log(base)
    assert not overlap, f"the range reaches commits already on {base}: {sorted(overlap)}"
    assert not (shas & _log("--merges", f"{base}..HEAD")), \
        "the range includes a merge commit"


_PROBE = ("Probe", "probe@artificialhumanity.io")
_T = "%s: %s <%s>" % (TRAILER_KEY, _PROBE[0], _PROBE[1])


@pytest.mark.parametrize("shape,message,should_fire,trailers_carry", [
    ("A — trailer as the last paragraph",
     "fix: something\n\n%s\n" % _T, True, True),
    # ⚠ B IS THE SHAPE `%(trailers)` ALONE MISSES, and since 2026-09-16 it is the one that
    # would FAIL A CORRECT COMMIT rather than pass a forbidden one: a developer who writes a
    # closing sentence after their trailer gets told they omitted it. The body branch is what
    # keeps that from happening.
    ("B — trailer, then a closing sentence",
     "fix: something\n\n%s\n\nAnd some closing prose here.\n" % _T, True, False),
    # ⚠ C CARRIES THE LITERAL TRAILER TEXT IN MID-SENTENCE PROSE, deliberately (#241). A
    # substring check would call this a trailer; the line-anchored match does not. With the full
    # phrase present but not alone on its line, a revert to substring matching turns red here,
    # which is what C is for.
    ("C — the phrase in prose only",
     "fix: something\n\nThis explains why a %s trailer is used.\n\nSuite: 1429 passed.\n" % _T,
     False, False),
    ("D — alongside another trailer",
     "fix: something\n\nSigned-off-by: Someone <s@example.com>\n%s\n" % _T, True, True),
])
def test_the_trailer_detection_actually_fires(tmp_path, shape, message, should_fire,
                                              trailers_carry):
    """⚠ THE CONTROL #241 SAID WAS MISSING, AND IT WAS RIGHT.

    `0afe042` moved this guard onto `%(trailers)` — an unvalidated git format field — and
    added no proof the field is ever non-empty. If it returned "" for every commit (an older
    git, a typo in the format, a field reorder, a change to the trailer's spelling) the guard
    would pass green forever while checking nothing.

    That hazard is what this entire FILE is built around: three of its tests exist solely as
    controls, and its module docstring narrates issue #108, where both assertions of a guard
    were incapable of failing whatever the range contained. The one assertion with no control
    was the one that grew a new dependency.

    ⚠ The `landed_branch` fixture cannot serve: every commit it builds has a one-line message,
    so `%(trailers)` is empty for all of them. It exercises the RANGE, never the FIELD.

    Shapes A–D are the four measured in #242. B is the one `%(trailers)` alone misses.
    """
    repo = tmp_path / "t"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", BASE_BRANCH)
    (repo / "f.txt").write_text(shape, encoding="utf-8")
    _git(repo, "add", "-A")
    # ⚠ A SYNTHETIC PROBE IDENTITY, NOT THE LIVE ONE. This control measures how git parses a
    # trailer, which does not depend on whose name is in it — and pinning it to `PERSONA.md`
    # would make the owner's next rename edit a test that has nothing to do with identity.
    _git(repo, "-c", "user.name=Probe", "-c", "user.email=probe@example.invalid",
         "commit", "-q", "-m", message)
    trailers = _git(repo, "log", "-1", "--format=%(trailers)")
    body = _git(repo, "log", "-1", "--format=%B")

    # ⚠⚠ THE TWO BRANCHES ARE PINNED SEPARATELY, AND THE COMBINED CALL IS NOT ENOUGH (#241).
    # `carries_agent_trailer` is an OR, and the body regex alone returns the right answer for all
    # four shapes — so forcing `trailers=""` changed NO result and the first version of this
    # control could not fail if `%(trailers)` broke, which is the whole of #241. The trailers
    # branch is not redundant: a trailer with no `<address>` is caught by it and missed by the
    # `<...>` regex, and that coverage could vanish with nothing going red.
    assert (_T in trailers) is trailers_carry, (
        f"shape {shape}: `%(trailers)` did not parse as expected — got {trailers!r}. This is "
        "the field the guard's primary branch reads; if it is empty for every shape the guard "
        "is checking nothing.")
    assert carries_agent_trailer(trailers, body, _PROBE) is should_fire, (
        f"shape {shape}: detection returned the wrong answer")
