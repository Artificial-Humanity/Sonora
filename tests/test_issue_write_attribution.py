"""Unattributed writes: every `issue.py` subcommand that CAN write must refuse without an author (#461).

`FerroStep/workflow/scripts/issue.py` gates the whole tracker on one line in `main()` — no
author, no write — and the set that line consults, `READ_ONLY_COMMANDS`, is the only thing
between a tracker record and an anonymous one. Nothing in `tests/` has ever exercised it. The
guard has failed twice in the same direction, and a reviewer reading the ledger caught it both
times; the suite was green through both:

* **#211** — `grade` was not enrolled, so a severity change could land with no author. Severity
  is the field the MERGE GATE reads, which makes "who decided this blocks the branch?" a
  question somebody actually asks, and it was unanswerable.
* **#459** — `rescope` was not enrolled, the same way, in the very commit whose own comment
  explained which subcommands still had to be written out by hand.

#459's fix INVERTED the check: the module now names the READS and treats everything else as a
write, so a write added tomorrow is attributed by default. That closes the direction both
failures arrived from, and this file does not re-test it. What is still open is the other
direction — a subcommand wrongly ADDED to `READ_ONLY_COMMANDS`, or a read that grows a side
effect — which lands unattributed with nothing to disagree with it.

WHAT IS LOCKED HERE, and why each one rather than "the guard is there":

1. **BEHAVIOUR, NOT MEMBERSHIP.** `READ_ONLY_COMMANDS == {...}` would pin a choice rather than
   test one: it passes for whatever set someone edits it to, which is exactly how a non-read
   gets in, and it fails for a legitimate addition. What is asserted is what an operator sees —
   run the subcommand with no author, read the refusal.
2. **THE WRITE SIDE IS DERIVED FROM THE CODE, NOT FROM THAT SET.** Classifying by
   `READ_ONLY_COMMANDS` would ask the guard to confirm itself and would agree with it by
   construction on the day it is wrong. A subcommand counts as a write here if its handler can
   REACH a mutating tracker call — an HTTP POST/PATCH/PUT/DELETE through `PB.call`, or an
   invocation of the `ferrostep` referee pointed at `--store` — propagated across the call
   graph out of the AST. Two consequences worth stating, because both are load-bearing:
   `take` writes only through the engine and never POSTs, so the HTTP rule alone would have
   read it as a read; and `reviewer_name()` shells out to `ferrostep agent-env` with NO
   `--store`, so it cannot reach a record and the referee rule alone would have read it as a
   write.
3. **THE SUBCOMMAND LIST COMES FROM THE TOOL**, parsed out of `issue.py --help`, so a
   subcommand added tomorrow is probed tomorrow without anyone remembering to add it here. A
   hand-typed list is the shape that produced #211 in the module itself, and it would rot the
   same way one directory over.
4. **FLOORS BEFORE ANY CLAIM.** Three derivations feed the assertions — the subcommand names,
   the subcommand-to-handler bindings, and the write classification — and every one of them is
   a parse that can silently stop matching. An empty or one-sided parse would make this file
   green while checking nothing, so it is refused up front.

HERMETIC, AND MEASURED RATHER THAN HOPED. Every probe runs with `ISSUE_AUTHOR` unset and `HOME`
pointed at an empty directory, and neither branch opens a socket:

* a WRITE refuses at the author line in `main()`, which is evaluated BEFORE `PB()` is
  constructed — the refusal precedes the first network call in the process;
* a READ passes that line and dies inside `PB.__init__`, which reads credentials from
  `~/.claude.json` before it authenticates — an empty HOME stops it one step short of the wire.

`test_no_probe_reached_the_tracker` asserts that per subcommand instead of leaving it a claim
in this docstring. Any refusal that is neither of those two — "tracker unreachable", "refused
by the tracker" — means a probe got further than intended, and this file must go red rather
than keep talking to the live board.
"""

import ast
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ISSUE = REPO / "FerroStep" / "workflow" / "scripts" / "issue.py"
SOURCE = ISSUE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

# The refusal this file is about, matched on the phrase the guard prints rather than on the
# exit status: `die()` is status 1 for every refusal in the module, so the code alone cannot
# tell the author gate apart from the credential wall — and telling them apart IS the test.
AUTHOR_REFUSAL = "--author is required for writes"
# The wall a read is expected to hit instead, one step short of the network.
CREDENTIAL_REFUSAL = "cannot read PocketBase credentials"
# Phrases that can only be printed by a process that already talked to the board. If one of
# these ever appears, the safety argument in the docstring is wrong and the file says so.
CONTACT_PHRASES = ("tracker unreachable", "cannot authenticate to the tracker",
                   "refused by the tracker", "lookup refused", "query refused",
                   "no issue #")

# One value for every argument a usage line says is required. A digit string is deliberate:
# it parses as `type=int` for the `number`/`numbers` positionals AND as a plain string
# everywhere else, so the builder below needs no per-argument knowledge — which is what keeps
# it from becoming the hand-kept table this file exists to avoid. The value is far outside the
# live range on purpose; nothing in these probes ever gets far enough to look it up, and that
# is asserted, but a placeholder that could name a real finding is a bad habit either way.
PLACEHOLDER = "99999999"


def _env(home=None):
    """The environment every probe runs in: no author, and a HOME with no credentials in it."""
    env = dict(os.environ)
    # ⚠ THE WHOLE SAFETY ARGUMENT RESTS ON THIS POP. `--author` defaults to `$ISSUE_AUTHOR`,
    # so a machine that exports it — the developer's own shell does — would hand every probe
    # an author, sail past the gate, and turn this file into a live tracker client.
    env.pop("ISSUE_AUTHOR", None)
    if home is not None:
        env["HOME"] = str(home)
    return env


def _run(argv, home=None):
    p = subprocess.run([sys.executable, str(ISSUE), *argv], cwd=str(REPO), env=_env(home),
                       capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


# --------------------------------------------------------------------------- #
# derivation 1 — what subcommands exist, asked of the tool
# --------------------------------------------------------------------------- #
def _usage(*argv):
    """The usage block for `issue.py [subcommand] --help`, whitespace-normalised.

    Normalising matters: argparse WRAPS the usage line, and it wraps between an option and its
    metavar (`--severity\\n{low,medium,high,critical}`). A line-oriented parse would read the
    two halves as unrelated tokens and build an invocation argparse then rejects — the same
    wrapped-line trap AGENTS.md records for the doc-claims gate, arriving in a parser.
    """
    rc, out = _run([*argv, "--help"])
    assert rc == 0, "`issue.py %s --help` exited %d:\n%s" % (" ".join(argv), rc, out)
    m = re.search(r"^usage:(.*?)(?:\n\s*\n)", out, re.S | re.M)
    assert m, "no usage block in `issue.py %s --help`:\n%s" % (" ".join(argv), out)
    return " ".join(m.group(1).split())


def _subcommands():
    """The names argparse itself advertises, read out of the choices group in the usage line."""
    usage = _usage()
    m = re.search(r"\{([a-z][a-z,_-]*)\}", usage)
    assert m, "could not find the subcommand choices in `issue.py --help`: %s" % usage
    return tuple(m.group(1).split(","))


def _required_tokens(usage, cmd):
    """The tokens a usage line presents as REQUIRED — i.e. everything outside `[...]`.

    Bracket depth rather than a regex: the optional groups nest (`[numbers ...]` sits inside a
    positional, `[--label {bug,documentation,enhancement}]` carries a brace group), and a
    non-greedy `\\[.*?\\]` closes on the wrong bracket for both.
    """
    head = "issue.py %s" % cmd if cmd else "issue.py"
    assert usage.startswith(head), "unexpected usage shape for %r: %s" % (cmd, usage)
    rest, depth, kept = usage[len(head):], 0, []
    for ch in rest:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif depth == 0:
            kept.append(ch)
    return "".join(kept).split()


def _value_for(token):
    """A value argparse will accept for a metavar: the first choice, or the placeholder."""
    if token.startswith("{") and token.endswith("}"):
        return token[1:-1].split(",")[0]
    return PLACEHOLDER


def _invocation(cmd):
    """A parseable, author-less invocation of one subcommand, built from its own usage line.

    ⚠ IT ONLY HAS TO REACH THE AUTHOR CHECK, NOT TO BE SENSIBLE. `rescope 99999999` with
    neither --branch nor --to-repo would die "nothing to set"; `file --title 99999999` with no
    body would die about the body. Both of those refusals live INSIDE the handler, which runs
    after `PB()`, which runs after the author gate — so an invocation that is nonsense to the
    handler still measures exactly the line under test, and a subcommand that stops somewhere
    else is visible as a refusal this file does not recognise rather than as a pass.
    """
    tokens, argv, i = _required_tokens(_usage(cmd), cmd), [], 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok.startswith("--"):
            argv.append(tok)
            # A required option's metavar is the next token unless another flag follows it.
            # (No required store_true flag exists here; one would need this to look ahead for
            # a metavar rather than assume the next non-flag token is its value.)
            if i < len(tokens) and not tokens[i].startswith("-"):
                argv.append(_value_for(tokens[i]))
                i += 1
        else:
            argv.append(_value_for(tok))
    return argv


SUBCOMMANDS = _subcommands()
INVOCATIONS = {c: _invocation(c) for c in SUBCOMMANDS}


# --------------------------------------------------------------------------- #
# derivation 2 — which handler each subcommand runs, read out of `main()`
# --------------------------------------------------------------------------- #
def _definitions():
    """Every function a call site inside this module could name, keyed by that name.

    ⚠ `PB.__init__` IS DELIBERATELY EXCLUDED, and the file would be vacuous without that
    exclusion. It POSTs an auth request, and `main()` constructs a `PB()` for EVERY
    subcommand including the reads — so counting it would classify all thirteen as writes,
    every assertion below would be satisfied by the reads too, and the one-sided-parse floor
    is what would catch it. Authenticating a session is not writing a record.
    """
    defs = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
    for node in TREE.body:
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name != "__init__":
                    # A method sharing a module function's name would make every call site
                    # naming it ambiguous, and the resolution below would silently pick one.
                    assert sub.name not in defs, (
                        "%s.%s collides with a module-level function of the same name; the "
                        "call-graph resolution here cannot tell them apart" % (node.name, sub.name))
                    defs[sub.name] = sub
    return defs


DEFS = _definitions()


def _parser_name(call):
    """`add("grade")` / `sub.add_parser("grade")` -> "grade"."""
    fn = call.func
    named = (isinstance(fn, ast.Name) and fn.id == "add") or \
            (isinstance(fn, ast.Attribute) and fn.attr == "add_parser")
    if named and call.args and isinstance(call.args[0], ast.Constant) \
            and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def _set_defaults_handler(call):
    """`s.set_defaults(fn=cmd_grade)` -> "cmd_grade"."""
    if isinstance(call.func, ast.Attribute) and call.func.attr == "set_defaults":
        for kw in call.keywords:
            if kw.arg == "fn" and isinstance(kw.value, ast.Name):
                return kw.value.id
    return None


def _handler_bindings():
    """subcommand -> handler function name, read from the parser `main()` actually builds.

    Not the `cmd_<name>` naming convention: it holds today for all thirteen, and a convention
    is a claim about names while the binding is a fact about behaviour. A subcommand wired to
    the wrong handler would satisfy the convention and be invisible.

    Two shapes are read, because `main()` writes the bindings two ways: the sequential
    `s = add("grade") ... s.set_defaults(fn=cmd_grade)`, and the loop over
    `(("review", cmd_review), ...)` that builds the four transition subcommands from a tuple.
    The floor below asserts the pair of readers between them covers EVERY advertised
    subcommand, so a third shape arriving later is a failure here rather than a gap.
    """
    out, current = {}, None
    for stmt in DEFS["main"].body:
        if isinstance(stmt, ast.For) and isinstance(stmt.iter, (ast.Tuple, ast.List)):
            pairs = [(e.elts[0].value, e.elts[1].id) for e in stmt.iter.elts
                     if isinstance(e, ast.Tuple) and len(e.elts) == 2
                     and isinstance(e.elts[0], ast.Constant) and isinstance(e.elts[0].value, str)
                     and isinstance(e.elts[1], ast.Name)]
            if pairs:
                out.update(pairs)
                continue
        calls = [n for n in ast.walk(stmt) if isinstance(n, ast.Call)]
        for call in calls:
            name = _parser_name(call)
            if name:
                current = name
        for call in calls:
            handler = _set_defaults_handler(call)
            if handler:
                assert current, "a set_defaults(fn=%s) with no subcommand in scope" % handler
                out[current] = handler
    return out


BINDINGS = _handler_bindings()


# --------------------------------------------------------------------------- #
# derivation 3 — which handlers can reach a tracker write
# --------------------------------------------------------------------------- #
MUTATING_HTTP = {"POST", "PATCH", "PUT", "DELETE"}


def _direct_write(fn):
    """Why this one function writes, or None. Two primitives, both of them the real surface.

    ⚠ THE REFEREE RULE KEYS ON `--store`, NOT ON THE WORD `ferrostep`. Since phase 2 the state
    moves are the engine's: `ferrostep_move` and `ferrostep_rescope` shell out and never touch
    HTTP, so an HTTP-only rule reads `take` — whose entire effect is an `open -> open`
    developer move that spends a fix pass — as a read. But `reviewer_name()` also shells out
    to `ferrostep`, for `agent-env`, which resolves a name from a roster file on disk. What
    separates them is that the writing calls hand the engine the tracker store to act on.
    """
    strings = {n.value for n in ast.walk(fn)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    if "ferrostep" in strings and "--store" in strings:
        return "invokes the ferrostep referee against --store"
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "call":
            continue
        method = None
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            method = node.args[1].value
        for kw in node.keywords:
            if kw.arg == "method" and isinstance(kw.value, ast.Constant):
                method = kw.value.value
        if method in MUTATING_HTTP:
            return "issues an HTTP %s to the tracker" % method
    return None


def _callees(fn):
    """The in-module functions one function calls, by the name the call site uses.

    Attribute calls resolve on the attribute alone (`pb.patch`, `self.call`), which is loose —
    but the only names it can collide with are other definitions in this same module, and
    `_definitions` refuses a collision outright.
    """
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in DEFS and name != fn.name:
                out.add(name)
    return out


def _write_reachable():
    """Every function that writes, directly or through anything it calls. Fixpoint, so a cycle
    terminates and a write three hops down (`cmd_review` -> `transition` -> `ferrostep_move`)
    is not missed the way a one-level scan would miss it."""
    writers = {}
    for name, fn in DEFS.items():
        why = _direct_write(fn)
        if why:
            writers[name] = why
    graph = {name: _callees(fn) for name, fn in DEFS.items()}
    changed = True
    while changed:
        changed = False
        for name, callees in graph.items():
            if name in writers:
                continue
            for callee in sorted(callees):
                if callee in writers:
                    writers[name] = "via %s, which %s" % (callee, writers[callee])
                    changed = True
                    break
    return writers


WRITE_REACHABLE = _write_reachable()
WRITE_SUBCOMMANDS = tuple(sorted(c for c, h in BINDINGS.items() if h in WRITE_REACHABLE))
READ_SUBCOMMANDS = tuple(sorted(set(SUBCOMMANDS) - set(WRITE_SUBCOMMANDS)))


# --------------------------------------------------------------------------- #
# the probe
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    """Run one subcommand with no author and an empty HOME; cached, so each runs once."""
    home = tmp_path_factory.mktemp("issue_attribution_home")
    cache = {}

    def _probe(cmd):
        if cmd not in cache:
            cache[cmd] = _run([cmd, *INVOCATIONS[cmd]], home=home)
        return cache[cmd]

    return _probe


def _skip_if_unbuildable(cmd, rc, out):
    """A probe argparse refused measures nothing. Skip it BY NAME and say why.

    Deliberately a skip rather than a failure: this file's job is the author gate, and an
    invocation this builder cannot express is a gap in the builder, not evidence about the
    guard. It is safe to skip precisely because `test_the_refusal_detector_discriminates`
    refuses to let the whole population go quiet — a skip that spread to everything would take
    the floors down with it.
    """
    if rc == 2 or ("usage: issue.py" in out and "error:" in out):
        pytest.skip("could not build an invocation `%s` accepts from its own usage line, so "
                    "the author check was never reached; argparse said:\n%s" % (cmd, out.strip()))


# --------------------------------------------------------------- floors, before any claim
def test_the_derivations_all_parsed_something_two_sided():
    """⚠ The empty-enumeration trap, and here it would be invisible three separate ways.

    If `--help` stopped advertising choices, if the binding readers stopped matching, or if
    the write classifier stopped recognising a primitive, the parametrised tests below would
    run over an empty list and this file would report green having probed nothing. Worse than
    empty is ONE-SIDED: a classifier that calls everything a write makes the refusal assertion
    trivially true, and one that calls everything a read makes it vacuous.

    No counts are stated. Every number here is derived from the module in the same run, and a
    remembered count is the thing that rots (#330's four attempts, one directory over).
    """
    assert SUBCOMMANDS, "`issue.py --help` advertised no subcommands"
    assert set(BINDINGS) == set(SUBCOMMANDS), (
        "the subcommand -> handler readers do not cover what argparse advertises; "
        "missing %s, extra %s" % (sorted(set(SUBCOMMANDS) - set(BINDINGS)),
                                  sorted(set(BINDINGS) - set(SUBCOMMANDS))))
    assert all(h in DEFS for h in BINDINGS.values()), (
        "a subcommand is bound to a handler this module does not define: %s"
        % sorted(h for h in BINDINGS.values() if h not in DEFS))
    assert WRITE_SUBCOMMANDS, (
        "no subcommand classified as a write — the write classifier has stopped recognising "
        "the tracker's mutating primitives, and every assertion in this file is now vacuous")
    assert READ_SUBCOMMANDS, (
        "every subcommand classified as a write — the classifier is over-broad (did PB.__init__ "
        "get counted?), and 'writes refuse' is now satisfied by a module that refuses everything")


# --------------------------------------------------------------- the guard, run
@pytest.mark.parametrize("cmd", WRITE_SUBCOMMANDS)
def test_a_writing_subcommand_refuses_without_an_author(cmd, probe):
    """#211, #459, #461: a subcommand that can reach a tracker write must not run unattributed.

    The classification is this file's, derived from what the handler can reach; the module's
    own answer is `READ_ONLY_COMMANDS`, and this test is the place the two are made to agree.
    A subcommand added to that set while its handler still writes fails HERE, which is the
    direction #459's inversion left open.
    """
    rc, out = probe(cmd)
    _skip_if_unbuildable(cmd, rc, out)
    assert rc != 0, "`issue.py %s` ran to completion with no author:\n%s" % (cmd, out)
    assert AUTHOR_REFUSAL in out, (
        "`issue.py %s` can write (%s) but did not refuse on the author line — it stopped "
        "somewhere else, which means the write was attributed to nobody up to that point. "
        "Is it in READ_ONLY_COMMANDS? Output:\n%s"
        % (cmd, WRITE_REACHABLE[BINDINGS[cmd]], out.strip()))


@pytest.mark.parametrize("cmd", sorted(SUBCOMMANDS))
def test_no_probe_reached_the_tracker(cmd, probe):
    """The safety property these probes rest on, asserted rather than assumed.

    A write stops at the author line, which precedes the `PB()` construction; a read stops
    inside `PB.__init__`, which reads `~/.claude.json` before it authenticates. Both are
    strictly before the first socket, so an empty HOME makes the whole file network-free — and
    the day that ordering changes, this test is what says so, instead of a suite quietly
    filing records against the live board on every run.
    """
    rc, out = probe(cmd)
    _skip_if_unbuildable(cmd, rc, out)
    contacted = [p for p in CONTACT_PHRASES if p in out]
    assert not contacted, (
        "`issue.py %s` got as far as the tracker (%s). The refusal ordering this file depends "
        "on has changed; fix that before running this suite again:\n%s"
        % (cmd, contacted, out.strip()))
    assert AUTHOR_REFUSAL in out or CREDENTIAL_REFUSAL in out, (
        "`issue.py %s` stopped at neither the author gate nor the credential wall, so what it "
        "did instead is unknown:\n%s" % (cmd, out.strip()))


def test_the_refusal_detector_discriminates(probe):
    """⚠ POSITIVE CONTROL. Without this, a module that refused EVERY subcommand on the author
    line — or one whose refusals all happened to contain the phrase — would satisfy the test
    above for reasons that have nothing to do with the guard, and a `READ_ONLY_COMMANDS` that
    had been emptied would read as a pass.

    So: something classified as a read must get PAST the author gate and die at the credential
    wall instead. That is one observation proving three things at once — the probes really run,
    the two refusals are distinguishable, and the author line is a decision rather than an
    unconditional refusal.
    """
    got_past = {}
    for cmd in READ_SUBCOMMANDS:
        rc, out = probe(cmd)
        if rc == 2 or ("usage: issue.py" in out and "error:" in out):
            continue
        got_past[cmd] = out
    assert got_past, ("no read subcommand could be probed at all, so every assertion in this "
                      "file is standing on skipped work")
    passed_the_gate = {c: o for c, o in got_past.items()
                       if AUTHOR_REFUSAL not in o and CREDENTIAL_REFUSAL in o}
    assert passed_the_gate, (
        "every probed read refused on the author line too, so 'writes refuse' is being "
        "satisfied by a module that refuses everything and this file measures nothing. "
        "Probed: %s" % {c: o.strip() for c, o in got_past.items()})
