"""§7 audit/QC fixes: C-M1, C-M9, C-L5, C-L6 (2026-08-07).

What these share is that each one FAILS BY DOING NOTHING. A tail sample that rounds to
zero, a defer flag that only flips one way, a cohort glob that sweeps quarantined clips
into the statistics they are supposed to be excluded from — none of them error, and each
leaves a report that looks complete.
"""

import argparse
import csv
import json
import math
import os
import re
import pathlib
import sys
import time

import pytest
from scripts_layout import SCRIPTS  # noqa: E402

SYNTH = SCRIPTS
SCRIPTS.on_path()
def _src(name):
    return (SYNTH / name).read_text(encoding="utf-8")


BANK = "scripts/stages/synth_bank.sh"


# What ENDS a word, and therefore what a comment-opening `#` may follow. ⚠ The first
# version of this was `line[i-1].isspace()`, which is wrong: `;`, `&`, `|`, `(` and `)`
# terminate a word too, so a `#` right after any of them opens a comment. Verified against
# the shell that actually reads these files (`bash -c 'echo hi;# qc_verdict.py'` prints
# `hi`), and the #24 shape walked through one punctuation mark to the left of where it had
# just been closed.
_WORD_END = " \t;&|()"


def _strip_trailing_comment(line):
    """Drop an unquoted `#`-to-end-of-line. POSIX rule: `#` opens a comment only at the
    start of a word, so `${VAR#pat}`, `$#` and `sha#1` survive, and so does a `#` inside
    quotes.

    ⚠ Added 2026-08-12 because a WHOLE-LINE comment filter is not a comment filter.
    `_code_lines` drops a line whose *first* character is `#`; a trailing comment stayed in
    the joined command and satisfied every wiring assertion. Reproduced: every real
    `qc_verdict.py` invocation removed from `synth_bank.sh` and the name left in a trailing
    comment → the whole suite green. That is issue #24 verbatim, walking through the ratchet
    built to stop it.
    """
    out, quote, i = [], None, 0
    while i < len(line):
        c = line[i]
        if quote:
            out.append(c)
            if c == "\\" and quote == '"' and i + 1 < len(line):
                i += 1
                out.append(line[i])
            elif c == quote:
                quote = None
        elif c in "'\"":
            quote = c
            out.append(c)
        elif c == "#" and (i == 0 or line[i - 1] in _WORD_END):
            break
        else:
            out.append(c)
        i += 1
    return "".join(out).rstrip()


# `echo`/`printf` possibly behind redirections, a brace/paren/`!`, or VAR= prefixes.
#
# ⚠ MAKEFILE RECIPE PREFIXES (`@` silent, `-` ignore-errors, `+` always-run) ARE ANCHORED TO
# THE START OF THE LINE, in a pattern of their own. They were briefly in the general prefix
# class, which made them valid after ANY `;&|{(` or whitespace — i.e. everywhere in a shell
# command — and that WIDENED the repo's core #24 filter rather than narrowing it. Measured:
#
#     find . -name x -printf "%p qc_verdict.py" -print
#
# had `qc_verdict.py` removed as printed text, because `-printf` matched `-` + `printf`. A
# real script name vanishing from an invocation is the #24 failure with the sign flipped, so
# the guard against describing-instead-of-running was made able to hide a run.
# ⚠ OPT-IN, and only for a caller holding the RAW line. `[@+-]` cannot distinguish a Make
# recipe prefix from an option: `-printf '%p x' -print` matched too, so anchoring the class
# to column 0 relocated the erasure rather than closing it. A recipe prefix is only a recipe
# prefix after the leading TAB of a recipe line, which is exactly what the Makefile/workflow
# branch can check and what a shell command never has. Shell scripts — the #24 filter — never
# enable this.
# `^\s*` because the caller passes the line with its leading TAB intact — that tab is
# what licensed `recipe_prefixes` in the first place, so stripping it here would make
# the flag unreachable. (It was briefly `^[@+-]+`, which never matched a real recipe.)
_RECIPE_ECHO = re.compile(r"^\s*[@+-]+\s*(?:echo|printf)\b")
# What ends a printed clause. Named because `is_wholly_printed` has to strip exactly this set
# back off the remainder, and two hand-written copies of it would drift.
_CLAUSE_END = ";&|\n"
# Redirections, a brace/paren/`!`, or VAR= assignments, in any order, before the verb.
# ⚠ THE ASSIGNMENT ALTERNATIVE ENDS IN `\s+`, NOT `\s*`. With `\s*` the greedy `\S*` simply
# backtracks and hands the verb to itself: `FOO=echo python <stage>` parsed as assignment
# `FOO=` + verb `echo`, the whole clause was skipped as printed text, and a real launch went
# MISSED. Requiring whitespace after the assignment token means `FOO=echo` can only ever be
# the assignment, never assignment-plus-verb.
# ⚠ ONE `\s*` OWNS THE WHITESPACE BEFORE THE VERB — at the END, not on both sides of the
# `*`-quantified alternation. Spelled `(?:\s*(?:…)\s*|…)*`, the engine has two ways to assign
# the space between consecutive prefix tokens (iteration k's trailing `\s*` vs k+1's leading
# one), so a failing match tries every partition: ~4x per two characters, measured at 1.1 s for
# `'a ' + '> ' * 22 + 'x'` and ~10 s at 24. Relocating it makes that 0.19 ms. Not reachable
# from anything in the tree today (it needs ~20 alternating `> ` / `( ` / `! ` tokens on one
# line with no verb after them) — but a scanned line that looks like a HANG rather than a
# failure is worth a one-token fix.
_ECHO_PREFIX = r"""(?:\s*(?:&[<>]{1,2}|[0-9]*[<>]&?[0-9-]*|[{(!])|\s*[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*\s*"""
_ECHO_HEAD = re.compile(
    r"""(?:
          # ── a TRUE clause start: string start, or after a shell operator. ONLY here may the
          # verb be PATH-QUALIFIED, because only here is the next token argv[0].
          (?:^|(?<=[;&|{(`]))""" + _ECHO_PREFIX + r"""
          (?:[^\s;&|<>'"`(){}]*/)?               # /bin/echo, /usr/bin/printf, ./echo
          (?:echo|printf)\b
        |
          # ── mid-clause, after any whitespace: BARE verb only. Pre-PR behaviour, unchanged.
          (?<=\s)""" + _ECHO_PREFIX + r"""(?:echo|printf)\b
        )""",
    re.X,
)
# ⚠ THE TWO BRANCHES EXIST BECAUSE `\s` IS NOT A CLAUSE START. Spelled as one branch with
# `(?:^|(?<=[;&|{(`]|\s))` in front of the path prefix, ANY whitespace-separated token ending
# in `/echo` matched — argument position included — and the clause skip then ate the rest of
# the line. Measured: `python scripts/echo/qc_gate.py` reduced to `'python  '`, so the
# Makefile/workflow branch MISSED the launch and `_invocations` read a wired stage as unwired.
# That is the `-printf` incident documented above, rebuilt at a new address: a real script name
# vanishing from an invocation is #24 with the sign flipped. No path segment in this repo is
# named `echo` today, so it was latent — which is exactly why it needs a fixture and not a note.
# ⚠ THE PATH PREFIX IS WHY THIS PATTERN IS NOW THE ONLY DEFINITION. It landed 2026-08-13 to
# retire a SECOND, weaker one. `_argv_runs_a_printer` used to basename argv[0] itself
# (`PurePosixPath(words[0]).name`) before calling in here, because this pattern requires a
# clause start before the verb and `/` is not one. That caller-side normalisation was
# `head.split()` + `" ".join(...)` — which splits on ALL whitespace, including `\n`, and
# rejoins with spaces. So it flattened the very newline that this function's clause-terminator
# list had just been extended to see, and the HIGH the terminator was added for shipped INERT:
# two fixes to two findings, in one commit, that cancelled each other out. `\n` looked like
# live code and was unreachable for its only caller.
# Handling the prefix HERE fixes both, and covers a case neither spelling could: a
# path-qualified printer at a LATER clause (`/bin/echo a; /bin/echo <stage>`) is out of reach
# of anything that only normalises argv[0].
# Non-matches, asserted in the fixtures rather than assumed: `/usr/bin/notecho` (the prefix
# has to end at `/`) and `/path/echoserver` (the trailing `\b`).


def _substitution_body(cmd, start, closer):
    """The body of a substitution opened just before `start`, plus the index after its close.

    `closer` is `")"` for `$(…)` (nesting-aware, so `$(a $(b))` and `$((1+1))` both work) or
    "`" for a backtick (no nesting — bash has none there either). An unterminated substitution
    yields the rest of the string, which keeps the caller's scan monotonic rather than looping.
    """
    if closer == "`":
        j = start
        while j < len(cmd):
            if cmd[j] == "\\":
                j += 2
                continue
            if cmd[j] == "`":
                return cmd[start:j], j + 1
            j += 1
        return cmd[start:], len(cmd)
    depth, j, quote = 1, start, None
    while j < len(cmd):
        c = cmd[j]
        if quote:
            if c == "\\" and quote == '"':
                j += 1
            elif c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return cmd[start:j], j + 1
        j += 1
    return cmd[start:], len(cmd)


def _without_printed_text(cmd, recipe_prefixes=False):
    """`cmd` with every `echo`/`printf` and its arguments removed.

    ⚠ Added 2026-08-12. The old filter tested `cmd.split()[0]`, so it only ever caught a
    LEADING BARE `echo` — and these files are full of the other spellings:
    `>&2 echo "…"`, `[ -f x ] || { echo "…"; }`, `case … *) echo "…" >&2;; esac`. Any of
    them applied to a recovery hint left a stage fully unwired and the suite green
    (reproduced on both forms). `librivox_align.sh`'s `stage_pool` line is a fixture for
    the leading-bare form ONLY, which is why the redirected and braced forms get their own
    assertions below.

    Residual, and stated rather than hidden: a WRAPPER (`die "…run qc_verdict.py…"`) is not
    an `echo` and is not removed. No such wrapper exists in any orchestrator today; if one
    lands, it belongs in this pattern.
    """
    out, i = [], 0
    while i < len(cmd):
        m = _ECHO_HEAD.search(cmd, i)
        if i == 0 and recipe_prefixes:
            recipe = _RECIPE_ECHO.match(cmd)
            if recipe and (m is None or recipe.start() <= m.start()):
                m = recipe
        if not m:
            out.append(cmd[i:])
            break
        out.append(cmd[i:m.start()])
        # skip to the end of this clause, respecting quotes
        j, quote = m.end(), None
        while j < len(cmd):
            c = cmd[j]
            # ⚠ A COMMAND SUBSTITUTION IS A NESTED COMMAND, NOT THE END OF THE PRINTED CLAUSE.
            # It does run — and it runs inside DOUBLE quotes too (`echo "Result: $(python
            # <stage>)"`), so only a single quote makes it literal. But the first spelling of
            # this simply `break`ed at `$(`, which surrendered everything AFTER the substitution
            # to the remainder and broke both directions at once:
            #   * `_invocations` kept `'$(date) with … stage_pool.py --apply"'` from a pure
            #     recovery hint, so the hint COUNTED AS AN INVOCATION and the wiring guard would
            #     have stayed green with the real call deleted — the #24 failure this filter
            #     exists to prevent, caused by the filter;
            #   * `echo "at $(date) rerun python <stage>"` became a reported launch.
            # It also changed a real line: `librivox_align.sh:67` stopped being wholly printed,
            # harmless only because `stage_pool.py` happens to sit BEFORE its `$(`.
            # So: scan the substitution to its matching delimiter, recurse (a launch inside
            # still survives), and RESUME the printed clause after it.
            if quote != "'" and cmd.startswith("$(", j):
                # ⚠ NO CARVE-OUT FOR `$((` ARITHMETIC, DELIBERATELY — second spelling of this.
                # Arithmetic expansion runs no command, so skipping the whole span looks correct;
                # measured, it is strictly worse. `$(( $(python <stage>) + 1 ))` nests a real
                # substitution inside arithmetic, and skipping MISSED it, while recursing costs
                # nothing — an arithmetic body holds no script name, so it cannot false-positive.
                # THE MUTATION BATTERY IS WHAT FOUND THIS: inverting the carve-out left the suite
                # green, i.e. no fixture could tell the branch from its absence. An untested
                # branch whose only measurable effect was a silent miss.
                # A Makefile's `$(VAR)`/`$(shell …)` lands here too — a Make variable the shell
                # never sees — and recursing on it is harmless for the same reason, whereas
                # `break` made the first `@echo` help line naming a stage a false red in CI.
                body, j = _substitution_body(cmd, j + 2, ")")
                out.append(_without_printed_text(body))
                continue
            if quote != "'" and c == "`":
                body, j = _substitution_body(cmd, j + 1, "`")
                out.append(_without_printed_text(body))
                continue
            if quote:
                if c == "\\" and quote == '"':
                    j += 1
                elif c == quote:
                    quote = None
            elif c in "'\"":
                quote = c
            # ⚠ `&` IS NOT A CLAUSE END WHEN IT BELONGS TO A REDIRECTION. It is the same
            # terminator-debris defect `is_wholly_printed` was written for, one spelling
            # further on: `echo "…" >&2` broke the scan mid-redirection, left `&2` behind, and
            # `.strip()` removed the `&` but not the `2` — so a stderr recovery hint, the way
            # such a hint is normally written, was reported as a stage LAUNCH. The LEADING
            # redirection was always handled (it is in `_ECHO_PREFIX`); the trailing one was not.
            # ⚠ BOTH ORDERS. `>&2` puts the operator before the `&`; bash's `&>file` / `&>>file`
            # — the ordinary way to discard both streams — puts it after, so a check on the
            # preceding character alone left that half open. `&&` is still a clause end: neither
            # of its first `&`'s neighbours is a redirection operator.
            elif c == "&" and ((j > 0 and cmd[j - 1] in "<>") or cmd[j + 1:j + 2] in ("<", ">")):
                pass
            elif c in _CLAUSE_END:
                # ⚠ `\n` IS a clause terminator. Without it an `echo` on the first line of a
                # multi-line command string swallowed EVERYTHING after it, and the caller
                # read the empty remainder as "wholly printed, therefore exempt" — so a
                # real launch on line 2 was exempted by a greeting on line 1. The
                # `_shell_commands` path never hits this (it joins continuations with
                # spaces); `stage_launch_offenders` passes raw source strings, which do.
                break
            j += 1
        out.append(" ")
        i = j
    return "".join(out)


def is_wholly_printed(cmd, recipe_prefixes=False):
    """True when `cmd` prints and does nothing else — the "is this only a hint?" question.

    ⚠ THIS EXISTS BECAUSE `not _without_printed_text(cmd).strip()` IS THE WRONG TEST, and it
    was the shipped one. The terminator that ended each printed clause is left in the
    remainder, and `.strip()` removes whitespace but not punctuation — so two printers in a
    row reduced to `';'`, which is non-empty, which read as "a command survives":

        echo "step 1 failed"; echo "run scripts/stages/qc_verdict.py"   -> remaining ';'

    i.e. a pure recovery hint reported as a stage LAUNCH. A false red on innocent code, and
    the direction that gets a guard switched off rather than the direction that hides a bug.
    Measured on `main` at 07edf39, so it predates the path-prefix work above and is not a
    consequence of it; it stayed invisible because no real orchestrator writes that shape yet.

    Stripping only the ENDS is what makes this safe: a real command word is not in
    `_CLAUSE_END`, and an interior one cannot be reached by `strip()` at all.
    """
    return not _without_printed_text(cmd, recipe_prefixes).strip(_CLAUSE_END + " \t")


def _shell_commands(rel):
    """(lineno, command) for each shell command in `rel`: comments dropped, `\\`-joined.

    A pipeline stage is WIRED or it is merely WRITTEN ABOUT, and `synth_bank.sh` documents
    every stage at length — so a substring searched over the whole file cannot tell the
    two apart, and a guard that a comment can satisfy is how this file came to name a
    stage it never ran (issue #24). `_code_lines` is imported rather than copied: it is
    `test_container_env.py`'s definition of "a code line", and having one is the point.
    TRAILING comments are stripped here rather than there, because `_code_lines`' other
    callers want the whole line.

    Continuations are joined because the invocations under test span several lines, so a
    per-line search would see `--append-flags` and the script it belongs to as unrelated
    facts.
    """
    # pylint: disable=import-outside-toplevel
    from test_container_env import _code_lines

    start, parts = None, []
    for n, line in _code_lines(rel):
        line = _strip_trailing_comment(line)
        if not line:
            continue
        if start is None:
            start = n
        if line.endswith("\\"):
            parts.append(line[:-1].strip())
            continue
        parts.append(line)
        yield start, " ".join(parts)
        start, parts = None, []
    if parts:
        yield start, " ".join(parts)


def _invocations(rel):
    """`_shell_commands` minus everything that is merely PRINTED.

    Every stage in `synth_bank.sh` prints the command to run by hand when it fails, so the
    recovery hints name the very scripts these tests are checking are RUN. An `echo` that
    mentions `qc_verdict.py` is one more way to describe a stage without invoking it, and
    is exactly what would keep a wiring test green after the invocation was removed.

    A command is no longer dropped wholesale — the printed CLAUSES are removed and the rest
    of the command is kept, so `cmd && echo "hint"` still reports the `cmd`.
    """
    for n, cmd in _shell_commands(rel):
        kept = _without_printed_text(cmd).strip()
        if kept:
            yield n, kept


def _first_line_with(rel, needle, extra=None):
    for n, cmd in _invocations(rel):
        if needle in cmd and (extra is None or extra in cmd):
            return n
    return None


# --- the parser is the guard, so the parser gets its own tests -------------------------
#
# Every wiring assertion in this repo reduces to "does `_invocations` report this script".
# Until 2026-08-12 the answer was yes for two spellings that run nothing, and the reason it
# went unnoticed is that the mutation battery only ever mutated CODE — never PROSE. These
# are literals rather than file mutations so they cannot drift with the orchestrators.


@pytest.mark.parametrize("line,expected", [
    # the defect: a trailing comment naming a stage
    ('"$PY" "$S/qc_gate.py" --dir "$D"  # then qc_verdict.py merges the axes',
     '"$PY" "$S/qc_gate.py" --dir "$D"'),
    # code with no comment is untouched
    ('"$PY" "$S/qc_verdict.py" --eiv x', '"$PY" "$S/qc_verdict.py" --eiv x'),
    # a `#` that does not START a word is not a comment — these must survive intact
    ('grep "#" file', 'grep "#" file'),
    ("echo '# literal'", "echo '# literal'"),
    ('N="${VAR#prefix}"', 'N="${VAR#prefix}"'),
    ('test $# -gt 0', 'test $# -gt 0'),
    # a whole-line comment reduces to nothing
    ('# comment naming qc_verdict.py', ''),
    # ⚠ POSIX word delimiters, not just whitespace — `bash -c 'echo hi;# x'` prints `hi`
    ('cd d;# qc_verdict.py', 'cd d;'),
    ('( x )# qc_verdict.py', '( x )'),
    ('run &# qc_verdict.py', 'run &'),
    ('a|# qc_verdict.py', 'a|'),
])
def test_a_trailing_comment_is_not_code(line, expected):
    """`_code_lines` only ever dropped lines STARTING with `#`. A trailing comment survived
    into the joined command and satisfied the wiring assertions — issue #24's exact shape,
    passing through the ratchet built for it. The `${VAR#pat}` and `$#` rows are the reason
    this is a quote-aware scan and not `line.split("#")[0]`."""
    assert _strip_trailing_comment(line) == expected


@pytest.mark.parametrize("cmd", [
    'echo "  !! Recover with: $PY $S/qc_verdict.py --eiv x"',
    '>&2 echo "  !! Recover with: $PY $S/qc_verdict.py --eiv x"',
    '2>&1 printf "%s\\n" "$PY $S/qc_verdict.py"',
    '[ -f x ] || { echo "run $PY $S/qc_verdict.py"; }',
    'if [ -z "$D" ]; then echo "see $S/qc_verdict.py"; fi',
    'LC_ALL=C echo "see $S/qc_verdict.py"',
    '`echo run $S/qc_verdict.py`',          # backtick substitution is a clause start too
    'X=$(echo "run $S/qc_verdict.py")',
    # ⚠ A SUBSTITUTION BEFORE THE SCRIPT NAME. `break`ing at `$(` surrendered the rest of the
    # clause, so this hint's remainder held `qc_verdict.py` and `_invocations` COUNTED IT AS AN
    # INVOCATION — the wiring guard would have stayed green with the real call deleted, which is
    # the #24 failure this filter exists to prevent, produced by the filter. Every fixture at
    # the time put the substitution LAST, so none of them could see it.
    'echo "hint: rerun at $(date) with $PY $S/qc_verdict.py --eiv x"',
    'echo "at $(basename "$OUT") rerun $PY $S/qc_verdict.py"',      # nested quotes inside it
    'echo "n=$(( 1 + 1 )) rerun $PY $S/qc_verdict.py"',             # arithmetic runs nothing
    'echo "wall=$(echo "$a $b" | awk "{print}") $S/qc_verdict.py"',  # a printer INSIDE one
])
def test_printed_text_is_not_an_invocation(cmd):
    """The old filter tested the FIRST TOKEN only, so every spelling here counted as a
    call. `librivox_align.sh`'s `stage_pool` hint is a fixture for the leading bare form
    alone — these are the ones it never covered."""
    assert "qc_verdict.py" not in _without_printed_text(cmd), cmd


@pytest.mark.parametrize("cmd", [
    '"$PY" "$S/qc_verdict.py" --eiv x && echo "  ok, wrote verdicts"',
    '"$PY" "$S/qc_verdict.py" --eiv x || { echo "  !! failed"; exit 1; }',
    # ⚠ NEWLINE as the clause terminator. This row is the one the `\n` in `_CLAUSE_END`
    # exists for, and until 2026-08-13 there was no such row: deleting `\n` from that set
    # left the whole suite green (923 passed), so the fix and its absence were
    # indistinguishable — which is precisely how it shipped inert for its only real caller.
    'echo starting\n"$PY" "$S/qc_verdict.py" --eiv x',
    # the same shape with the printer LAST, so the row cannot pass by position alone
    'echo starting\n"$PY" "$S/qc_verdict.py" --eiv x\necho done',
    # …and the other direction of the substitution fix: a stage RUN inside `$( )` is a run.
    # The recursion is what keeps this true while the clause resumes after the `)`.
    'echo "count: $("$PY" "$S/qc_verdict.py" --count)"',
    'echo "at $(dirname $("$PY" "$S/qc_verdict.py"))"',     # nested two deep
    'echo `"$PY" "$S/qc_verdict.py"`',
])
def test_a_real_invocation_survives_its_own_success_message(cmd):
    """Dropping the whole command when it contains an `echo` would be the opposite defect:
    every stage in `synth_bank.sh` reports what it did, and a filter that removed those
    commands wholesale would report the pipeline as empty."""
    assert "qc_verdict.py" in _without_printed_text(cmd), cmd


@pytest.mark.parametrize("cmd,printed_only", [
    # PATH-QUALIFIED printers — `_ECHO_HEAD`'s path prefix. Named in prose as the victims of a
    # regression and asserted nowhere until now; `grep -rn "bin/echo" tests/` found only the
    # comment describing the fix.
    ('/bin/echo "run $S/qc_verdict.py"', True),
    ('/usr/bin/printf "%s\\n" "$S/qc_verdict.py"', True),
    ('./echo "run $S/qc_verdict.py"', True),
    ('/bin/echo a; /bin/echo "run $S/qc_verdict.py"', True),
    # …and the boundaries of that prefix, so it cannot quietly widen into a launcher exemption
    ('/usr/bin/notecho "$S/qc_verdict.py"', False),   # prefix must end AT the `/`
    ('/path/echoserver "$S/qc_verdict.py"', False),   # the trailing `\b`
    ('/bin/echo a; "$PY" "$S/qc_verdict.py"', False),  # a real launch after a path printer
    # ⚠ ARGUMENT POSITION. The path prefix is anchored to a TRUE clause start precisely so
    # these do not match: spelled with `\s` in the lookbehind, any token ending in `/echo`
    # matched wherever it appeared and the clause skip ate the rest of the line, so a real
    # invocation lost its script name. Every row above puts the verb AT a clause start, so
    # none of them can see this — which is how it shipped in the first place.
    ('python scripts/echo/qc_gate.py', False),
    ('python scripts/echo.py --run', False),
    ('bash tools/printf.sh', False),
    ('"$PY" "$S/echo/qc_verdict.py" --eiv x', False),
    # a VAR= assignment may not hand the verb to itself (`\S*` used to backtrack)
    ('FOO=echo "$PY" "$S/qc_verdict.py"', False),
    ('LC_ALL=C echo "see $S/qc_verdict.py"', True),   # the legitimate assignment prefix
    # TRAILING redirection — `&` belongs to `>&`, and is not a clause end. The leading form
    # was always covered by `_ECHO_PREFIX`; this is how a stderr hint is actually written.
    ('echo "see $S/qc_verdict.py" >&2', True),
    ('echo "see $S/qc_verdict.py" 1>&2', True),
    ('echo "see $S/qc_verdict.py" > /dev/null', True),
    ('echo "x" >&2; "$PY" "$S/qc_verdict.py"', False),  # …but a launch after one still counts
    # bash's `&>` / `&>>` — the operator comes AFTER the `&`, so a check on the preceding
    # character alone left the commoner spelling of "discard both streams" reading as a launch
    ('echo "see $S/qc_verdict.py" &>/dev/null', True),
    ('echo "see $S/qc_verdict.py" &>>run.log', True),
    ('echo "x" &>/dev/null; "$PY" "$S/qc_verdict.py"', False),   # …and it still ends at the `;`
    ('echo hi && "$PY" "$S/qc_verdict.py"', False),              # `&&` is still a clause end
])
def test_a_path_qualified_printer_is_still_only_printing(cmd, printed_only):
    """Both directions of the path prefix, in the file that owns the pattern.

    `is_wholly_printed` rather than `not _without_printed_text(...).strip()`: the leftover
    clause terminator makes the naive spelling call row 4 a launch. That row is the reason the
    predicate exists — two printers in sequence is what a recovery hint looks like.
    """
    assert is_wholly_printed(cmd) is printed_only, (cmd, _without_printed_text(cmd))


# --- C-L6: the trusted tier's tail sample rounded to zero -----------------------------


@pytest.mark.parametrize("remainder", [1, 5, 10, 16, 17, 33, 100])
def test_a_non_zero_tail_fraction_always_samples_at_least_one(remainder):
    """`round(len(rest) * 0.03)` is ZERO until the remainder reaches 17 — and banker's
    rounding makes it zero at exactly 16.67 too. The tail sample is the whole mechanism
    for noticing that a TRUSTED engine has started drifting, and those engines are audited
    1 per group precisely because it is watching them. On any batch under 17 it silently
    did not exist."""
    pa = pytest.importorskip("pick_audit_subset")
    frac = pa.TIERS["trusted"]["tail"]
    assert math.ceil(remainder * frac) >= 1
    assert "math.ceil(len(rest) * frac)" in _src("pick_audit_subset.py")


def test_the_old_rounding_is_what_produced_the_hole():
    """Kept as the record of the defect: for every batch a trusted engine realistically
    produces below 17 clips, the old expression sampled nothing at all."""
    assert [n for n in range(1, 17) if round(n * 0.03) == 0] == list(range(1, 17))
    assert round(17 * 0.03) == 1


def test_a_zero_fraction_still_means_no_sample():
    """Ceiling must not invent a sample where the policy says there is none."""
    assert math.ceil(50 * 0.0) == 0


# --- C-M1: deferral was one-way -------------------------------------------------------


def _select_fixture(tmp_path, monkeypatch, n=12, engine="qwen"):
    """A one-group campaign on disk: bank.json + ratings.csv, all rows unaudited."""
    pa = pytest.importorskip("pick_audit_subset")
    datasets = tmp_path / "datasets"
    (datasets / "camp").mkdir(parents=True)
    ids = [f"clip_{i:04d}" for i in range(n)]
    (datasets / "camp" / "bank.json").write_text(json.dumps({"lines": [
        {"id": i, "book": "a-book", "intended_delivery": "Neutral", "engine": engine}
        for i in ids]}), encoding="utf-8")

    ratings = tmp_path / "ratings.csv"
    with open(ratings, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["campaign", "id", "engine", "score",
                                          "note", "status"])
        w.writeheader()
        for i in ids:
            w.writerow({"campaign": "camp", "id": i, "engine": engine,
                        "score": "", "note": "", "status": "unaudited"})

    monkeypatch.setattr(pa, "RATINGS", str(ratings))
    monkeypatch.setattr(pa, "DATASETS", str(datasets))
    # The new-reader rule reads real manifests and profiles off /data; this campaign is
    # synthesis, so neither applies. Stubbed so the test isolates the defer/requeue flip.
    monkeypatch.setattr(pa, "_clip_readers", lambda: {})
    monkeypatch.setattr(pa, "_confirmed_reader_titles", lambda: set())
    return pa, ratings, ids


def _statuses(ratings):
    with open(ratings, newline="", encoding="utf-8") as f:
        return {r["id"]: r["status"] for r in csv.DictReader(f)}


def _select(pa, flags=None):
    pa.cmd_select(argparse.Namespace(campaign="camp", flags=flags))


def test_deferral_is_revisable(tmp_path, monkeypatch):
    """A clip deferred by an earlier `select` and QC-flagged by a LATER pass stayed
    deferred forever — invisible in todo, never heard. QC runs after every generation pass
    and `--flags` grows, so deferral has to be revisable or the flags cannot reach the ear
    at all. That quietly breaks this tool's one non-negotiable rule, which its own
    docstring states: every QC-flagged clip goes to the ear.

    QC-H2 (2026-08-09): this test used to assert the SOURCE STRINGS of the fix, and it
    passed for two days over a branch that could never execute — `by_group` was built from
    `status == "unaudited"` rows only, so every id in `keep_ids` was an unaudited row's id
    and the `elif ... == "deferred"` arm was unreachable. A grep test cannot see reachability.
    """
    pa, ratings, ids = _select_fixture(tmp_path, monkeypatch)

    _select(pa)                                   # first pass: sample, defer the rest
    after_first = _statuses(ratings)
    deferred = [i for i, s in after_first.items() if s == "deferred"]
    assert deferred, "the fixture must actually defer something for this to mean anything"

    # A later qc_gate pass flags one of the deferred clips.
    flags = tmp_path / "qc_flags.txt"
    flags.write_text(deferred[0] + "\n", encoding="utf-8")
    _select(pa, flags=str(flags))

    assert _statuses(ratings)[deferred[0]] == "unaudited", (
        "a QC-flagged clip that was previously deferred never reached the ear")


def test_a_verdict_the_ear_has_already_given_is_not_revisable(tmp_path, monkeypatch):
    """Deferral is a sampling decision and may be revisited; a `keep`/`drop` is the ear's
    answer and sampling must never touch it. Widening the candidate pool is only safe
    because the widening stops at `deferred`."""
    pa, ratings, ids = _select_fixture(tmp_path, monkeypatch)
    rows = list(csv.DictReader(open(ratings, newline="", encoding="utf-8")))
    for r in rows[:4]:
        r["status"], r["score"] = "keep", "4"
    with open(ratings, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    flags = tmp_path / "qc_flags.txt"
    flags.write_text("\n".join(r["id"] for r in rows[:4]), encoding="utf-8")
    _select(pa, flags=str(flags))

    after = _statuses(ratings)
    assert [after[r["id"]] for r in rows[:4]] == ["keep"] * 4


def test_selection_is_stable_across_reruns(tmp_path, monkeypatch):
    """A consequence of the fix worth pinning: because the candidate pool no longer
    shrinks each pass, the positional spread picks the SAME clips on a re-run. Under the
    old filter a second `select` re-sampled whatever was left and slowly dragged the whole
    campaign into the queue, one pass at a time."""
    pa, ratings, ids = _select_fixture(tmp_path, monkeypatch)
    _select(pa)
    first = _statuses(ratings)
    _select(pa)
    assert _statuses(ratings) == first


def test_the_rule_it_protects_is_still_written_down():
    doc = " ".join(_src("pick_audit_subset.py").split())
    assert "every QC-flagged clip goes to the ear" in doc


# --- C-M9: cohort statistics included clips the ear had already rejected ---------------


def test_quarantined_clips_are_out_of_the_cohort():
    """ROOM is a robust z against the cohort's OWN median/MAD, so sweeping `_dropped/`
    and `_superseded/` in meant clips the ear had already rejected were helping define
    what "normal" looks like for the clips still under judgement. On a cohort quarantined
    precisely because it was reverberant, that pulls the median toward the defect."""
    src = _src("qc_artifacts.py")
    assert '"_pre_loudnorm", "_dropped", "_superseded"' in src
    # matched as a path COMPONENT, not a substring: a campaign legitimately named
    # `..._dropped_takes` must not vanish from its own report.
    assert "p.split(os.sep)" in src


def test_the_manifest_glob_reaches_the_layout_the_pipeline_writes():
    """The wav glob was recursive and the manifest glob was not, so a campaign laid out
    as <campaign>/<engine-out>/ — which is what synth_bank.sh writes and the QC gate
    assumes — found no manifests at all. Every clip then fell back to engine "?" and the
    whole campaign became ONE cohort. The per-engine comparison is the entire point of
    this tool, and it was silently not happening; a "?" row reads as missing metadata
    rather than as a broken join."""
    src = _src("qc_artifacts.py")
    assert 'glob.glob(os.path.join(root, "**", "*_manifest.jsonl"), recursive=True)' in src


# --- C-L5: publish_tier had no policy for the real-audio lane -------------------------


def test_the_real_audio_lane_has_a_publication_policy():
    """`stage_pool` writes `engine: "librivox"` for the force-align lane and three books
    are already fetched, so the `unknown` check would hard-error the day the first staged
    real-audio clip reached a metadata.jsonl. It is a matter of when, not whether."""
    pt = pytest.importorskip("publish_tier")
    assert "librivox" in pt.ENGINE_POLICY
    licence, publishable = pt.ENGINE_POLICY["librivox"]
    assert publishable is True
    assert "PD" in licence


def test_the_librivox_entry_says_why_rather_than_inheriting_a_default():
    """These are human recordings, not synthesis: there is no model card to verify, so
    the surrounding "verify the weights licence" instruction is not merely unanswered for
    this row, it is the wrong question. The reasoning belongs on the record."""
    src = _src("publish_tier.py")
    assert "not an ENGINE" in src
    assert "C-L5" in src


# --- the instrumentation bargain (2026-08-08) -----------------------------------------


def test_an_engine_with_a_detector_has_the_detector_wired():
    """moss_vg left `scrutinized` because its defect became an instrument, not because
    the defect stopped happening.

    `qc_engine_defects.py` says in its own docstring that it is "what the tier system
    needs in order to ever hand those engines a ride-along back: coverage traded for
    instrumentation". Until 2026-08-08 it was a MANUAL step, so the trade was not real —
    a detector nobody runs trades nothing. If an engine is off scrutinized on the strength
    of a detector, the detector has to fire on every bank.

    Comment-blind since 2026-08-11: this asserted two substrings over the whole FILE, and
    `synth_bank.sh` describes the detector step in a nine-line comment — so commenting the
    real invocation out left it green. That is the same blindness that let `qc_verdict` be
    named in a comment and never run (issue #24).
    """
    cmds = [c for _, c in _invocations(BANK) if "qc_engine_defects.py" in c]
    assert cmds, "the defect detectors are not INVOKED by the bank pipeline"
    assert any("--append-flags" in c for c in cmds), \
        "detectors run but their flags never reach qc_flags.txt / the ear"

    defects = _src("qc_engine_defects.py")
    subset = _src("pick_audit_subset.py")
    # Every engine that HAS a detector must still be sampled, i.e. not trusted — the
    # detector buys a step down from scrutinized, never a step up to 1-per-group.
    for engine in ("moss_vg", "zonos"):
        assert f'"{engine}"' in defects, f"{engine} lost its detector"
        assert f'"{engine}": "trusted"' not in subset, \
            f"{engine} has a stochastic defect and a detector; trusted is not the trade"


def test_scrutinized_is_still_the_default_for_unknown_engines():
    """No engine is NAMED scrutinized any more, which must not be read as the tier being
    retired. It is the landing place for an engine with no track record — the onboarding
    rule — and the id-join failure path depends on it too."""
    src = _src("pick_audit_subset.py")
    assert 'DEFAULT_TIER = "scrutinized"' in src
    assert "SCRUTINIZED_ENGINES" in src


# --- the calibration tool's blind spot (2026-08-08) -----------------------------------


def test_the_drop_marker_is_a_verdict_not_a_missing_score():
    """`gate_calibration` exists to choose gate thresholds from the ear's own verdicts,
    and it was discarding almost all of them.

    `parse_score` searched for `[1-5]`, so the audition app's reject marker `x` returned
    None — and every caller then did `if score is None: continue`. Measured when found:
    **253 of the 286 dropped/reroll rows carry `x`, so 88% of every rejection was
    invisible**, and a sweep over a campaign full of drops printed "no rated clip matches
    the defect", which reads exactly like "this campaign is clean".

    0, not None: the downstream test is `score < keep_score`, and a drop has to compare
    as worse than any keep.
    """
    SCRIPTS.on_path()
    gc = pytest.importorskip("gate_calibration")
    assert gc.parse_score("x") == 0
    assert gc.parse_score("X") == 0
    assert gc.parse_score(" x ") == 0
    assert gc.parse_score("5") == 5
    assert gc.parse_score("") is None, "no verdict is still no verdict"
    assert gc.parse_score(None) is None
    # And it must sort below any keep bar.
    assert gc.parse_score("x") < 4


def test_the_retake_marker_is_a_verdict_too():
    """QC-M3: the 2026-08-08 fix covered `x` and stopped there. Retake writes `"0"`, and
    `re.search(r"[1-5]", "0")` finds nothing — so it returned None and `load_ear` dropped
    the row exactly as it had dropped the drops, for the remaining slice of rejections.
    34 rows on the live sheet, every one status `reroll`. Same failure, one marker over.
    """
    SCRIPTS.on_path()
    gc = pytest.importorskip("gate_calibration")
    assert gc.parse_score("0") == 0
    assert gc.parse_score(" 0 ") == 0
    assert gc.parse_score("0") < 4, "a retake must sort below any keep bar"


def test_a_recategorized_row_is_still_the_keep_it_was():
    """The counterweight to both marker fixes. `x1`..`x5` are legacy Recategorize rows
    (status `relabeled`, retired 2026-07-26): the ear KEPT the clip at that score and
    re-labelled it, and `stage_pool` counts `relabeled` as a keep. 57 on the sheet. Widen
    the `x` test to "starts with x" and all 57 silently become rejections."""
    SCRIPTS.on_path()
    gc = pytest.importorskip("gate_calibration")
    assert [gc.parse_score(f"x{n}") for n in range(1, 6)] == [1, 2, 3, 4, 5]


# --- QC stage 2 was written and never invoked (issue #24, 2026-08-11) ------------------
#
# `qc_verdict.py` is the intended-vs-measured direction check. `synth_bank.sh` NAMED it in
# a comment for three weeks and never ran it, so 695 directed clips across five campaigns
# reached the ear with nine intrinsic-quality gates and no direction check at all — the
# third time a written instrument in this file turned out to be un-invoked.

def test_the_direction_check_actually_runs_rather_than_being_described():
    """The failure this closes is not "the check is wrong", it is "the check never ran".

    Deliberately comment-blind, and that is not a detail. The detector guard above used to
    assert a substring over the whole file, and `synth_bank.sh` explains every stage at
    length in prose — so commenting the real invocation out left it GREEN. A wiring test a
    comment can satisfy cannot tell `wired` from `written about`, which is the exact
    confusion that produced this issue: the file named `qc_verdict.py` and ran nothing.
    """
    cmds = [c for _, c in _invocations(BANK) if "qc_verdict.py" in c]
    assert cmds, "qc_verdict.py is not INVOKED by synth_bank.sh (a comment is not a call)"
    assert any("--eiv" in c for c in cmds), \
        "no invocation writes verdicts — --eiv is what merges EIV scores into qc_verdicts"
    assert any("--append-flags" in c for c in cmds), \
        "verdicts are written but axis failures never reach qc_flags.txt / the ear"
    assert any("--count-directed" in c for c in cmds), \
        ("no skip path: real-audio banks carry no `intended` labels, and running the "
         "stage on them spends a GPU labelling pass to confirm nothing")
    assert any("eiv_score.sh" in c for _, c in _invocations(BANK)), \
        "qc_verdict has no scores to merge — eiv_score.sh is not run over qc_filelist.txt"


def test_the_verdicts_are_frozen_before_the_ear_is_shown_the_clips():
    """Ordering is the load-bearing part, and it is invisible at runtime.

    `gate_calibration.py`: "Every full-coverage round is a free opportunity to measure it,
    but only if the instrument verdicts are frozen BEFORE the owner rates. Afterwards
    there is nothing to compare against and the round is spent." Dataset pipeline 1.0 —
    the owner spot-checking instead of auditing every clip — is gated on that measurement,
    so a stage-2 run that lands after `register_audition` produces a file that looks
    identical and is worth nothing to the calibration.
    """
    gate = _first_line_with(BANK, "qc_gate.py")
    eiv = _first_line_with(BANK, "eiv_score.sh")
    verdict = _first_line_with(BANK, "qc_verdict.py", extra="--eiv")
    register = _first_line_with(BANK, "register_audition.py")
    assert None not in (gate, eiv, verdict, register)
    assert gate < eiv < verdict, "the verdict must merge the gate's filelist and its scores"
    assert verdict < register, \
        ("the verdicts are written AFTER the clips are queued for audition — the round "
         "is spent and gate_calibration has nothing frozen to compare against")


def test_the_direction_check_is_advisory_and_the_quality_gate_is_not():
    """The two stages must be wired differently, and the difference is the whole fix.

    `keep` in qc_verdict is a LABEL-CONFIRMATION verdict — `gate_calibration.py`: "A
    beautiful clip whose director-assigned labels were wrong is a legitimate not-keep." So
    copying the qc_gate wiring, which exits without registering, would discard good audio
    because Gemma mislabelled it. Stage 2 belongs on the qc_engine_defects pattern: it
    announces failure and carries on.

    The qc_gate half of the assertion is the control: without it this test would also pass
    against a file that had stopped stopping for anything at all.

    THE WINDOW IS ANCHORED AT THE FIRST COMMAND OF STAGE 2, NOT ITS LAST (issue #57). It
    used to start at the `--eiv` invocation — the stage's LAST command — so the guard saw
    ten lines of a ~75-line block: the `--count-directed` probe, the whole `case`, and all
    three skip/failure branches sat outside it. MEASURED 2026-08-11: an `exit 2` in the
    real-audio branch, which stops every librivox bank before `register_audition`, left
    this suite at 27 passed. The coverage assertions below are part of the fix — they
    fail if the window is ever narrowed back onto a subset of the block.
    """
    cmds = list(_shell_commands(BANK))
    probe = _first_line_with(BANK, "qc_verdict.py", extra="--count-directed")
    verdict = _first_line_with(BANK, "qc_verdict.py", extra="--eiv")
    # Stage 3 (the engine defect detectors) is where stage 2 ends. Bounding at
    # register_audition instead puts another stage's commands inside a window this test
    # names stage 2 — and stage 3 is where a legitimate future `exit` would sit.
    defects = _first_line_with(BANK, "qc_engine_defects.py")
    register = _first_line_with(BANK, "register_audition.py")
    assert None not in (probe, verdict, register), \
        "stage 2 is not wired — see the wiring and ordering tests above"
    # From the banner the stage announces itself with, so the lines BETWEEN the banner and
    # the probe are inside too; the probe is the fallback if the banner is ever reworded.
    banner = next((n for n, c in cmds if c.startswith("echo") and "qc verdicts" in c), None)
    start = min(n for n in (banner, probe, verdict) if n is not None)
    end = min(n for n in (defects, register) if n is not None and n > start)
    stage2 = [c for n, c in cmds if start <= n < end]
    assert stage2, "no stage-2 commands between the probe and the next stage — see the " \
                   "wiring and ordering tests above for which of the two is missing"

    # The window covers the WHOLE block, proven against its landmarks rather than trusted:
    # every one of these lines was outside the old window.
    for landmark in ("--count-directed", "real-audio bank", "eiv_score.sh", "--eiv",
                     "esac"):
        assert any(landmark in c for c in stage2), \
            (f"the stage-2 window stops short of `{landmark}` — an `exit` there would "
             f"stop the bank unnoticed, which is exactly how #57 went green")

    offenders = [c for c in stage2 if c.split("#")[0].strip().startswith("exit")
                 or " exit " in c]
    assert not offenders, \
        ("the direction check can stop the bank: " + " | ".join(offenders) + " — it is "
         "advisory, and a mislabelled clip must not be pulled out of the queue")
    assert any("!!" in c and "qc_verdict" in c for c in stage2), \
        "a failed direction check is silent; it must be announced like the defect detectors"

    gate = _first_line_with(BANK, "qc_gate.py")
    hard = [c for n, c in cmds if gate <= n < start and c.startswith("exit")]
    assert hard, "qc_gate no longer stops the pipeline — it is not advisory and must"


def test_a_bank_with_no_intended_labels_is_a_clean_no_op():
    """`want = r["intended"][axis]` was unguarded, and real audio has no `intended` at all
    (librivox-v1: 664 rows, librivox-v2: 1,366). The first row would have taken the whole
    campaign down with a KeyError — so the stage could not simply be run everywhere.

    An unlabelled axis is ABSENT from the checks rather than False: `keep` is
    `all(checks.values())`, and there is no claim to fail. `all({})` is True, so an
    undirected clip's verdict is exactly its hard_pass.
    """
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")
    librivox_row = {"id": "uneasy-money_lv002_0199", "hard_pass": True, "engine": "librivox"}
    assert qv.intended_labels(librivox_row) == {}
    checks, notes = qv.axis_verdicts(qv.intended_labels(librivox_row),
                                     {"V": 2.0, "A": -1.0, "T": 0.4}, 0.25, 0.3)
    assert checks == {} and notes == []
    assert bool(librivox_row["hard_pass"] and all(checks.values())) is True

    # ...and a row that carries only SOME of the axes constrains only those.
    partial = qv.intended_labels({"intended": {"V": 0.9, "A": None}})
    assert partial == {"V": 0.9}
    checks, _ = qv.axis_verdicts(partial, {"V": -1.0}, 0.25, 0.3)
    assert checks == {"V": False}, "a directed axis pointing the wrong way still fails"


def test_the_verdict_says_which_definition_of_A_it_used():
    """The synth lane's A is the EIV Arousal head; the LibriTTS corpus lane's A is the
    per-speaker z-score of integrated loudness (`normalize_loudness.py` documents why the
    two disagree on purpose). A clip can satisfy one and fail the other, so a verdict that
    does not name its own definition is a number two lanes will read differently."""
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")
    a = qv.MEASURED_FROM["A"]
    assert "Arousal" in a and "LUFS" in a, a
    assert "NOT" in a, "the divergence has to be stated, not implied"
    assert "delivery" in qv.MEASURED_FROM, \
        "contract v2 shipped a delivery channel; a row must say it went unchecked"


# --- stage 2: what the instrument may and may not assert (#54-#58) --------------------
#
# These run qc_verdict end to end against a STUB anchor. The real one is 30,351 LibriTTS
# rows on /data beside a corpus this checkout does not carry, and a verdict test that
# skips wherever the corpus is absent is a verdict test that never runs in CI.

ANCHOR_HEADS = ["Amusement", "Valence", "Sadness"]
ANCHOR_WEIGHTS = [0.5, 0.3, -0.2]


def _stub_anchor(tmp_path, qv, monkeypatch, weights=None, drop=()):
    """Point qc_verdict's LibriTTS anchor at 40 stub rows. `drop` = heads to omit."""
    root = tmp_path / "anchor"
    root.mkdir(exist_ok=True)
    meas, eiv = [], []
    for i in range(40):
        wav = f"/anchor/{i}.wav"
        meas.append({"wav": wav, "alpha_db": 5.0 + i * 0.1, "cpp": 1.0 + i * 0.02,
                     "h1h2": -13.0 + i * 0.05})
        row = {"wav": wav, "Arousal": 0.2 * i, "Soft_vs._Harsh": -0.1 * i}
        for k, h in enumerate(ANCHOR_HEADS):
            if h not in drop:
                row[h] = 0.05 * i + 0.1 * k
        eiv.append(row)
    for name, data in (("measures.jsonl", meas), ("eiv.jsonl", eiv)):
        (root / name).write_text("".join(json.dumps(d) + "\n" for d in data),
                                 encoding="utf-8")
    (root / "fam.jsonl").write_text("", encoding="utf-8")
    (root / "combo.json").write_text(json.dumps(
        {"heads": ANCHOR_HEADS, "weights": list(weights or ANCHOR_WEIGHTS)}),
        encoding="utf-8")
    monkeypatch.setattr(qv, "LIB_MEASURES", str(root / "measures.jsonl"))
    monkeypatch.setattr(qv, "LIB_EIV", str(root / "eiv.jsonl"))
    monkeypatch.setattr(qv, "LIB_FAM", str(root / "fam.jsonl"))
    monkeypatch.setattr(qv, "COMBO", str(root / "combo.json"))


def _stub_campaign(tmp_path, n=4, scored=None, intended=None, stamp=True):
    """A campaign dir: n directed clips, the first `scored` of them carrying an EIV row."""
    camp = tmp_path / "campaign"
    camp.mkdir(exist_ok=True)
    scored = n if scored is None else scored
    rows, eiv, files = [], [], []
    for i in range(n):
        wav = camp / f"clip{i}.wav"
        wav.write_bytes(b"RIFF" + b"\0" * 40)
        rows.append({"id": f"clip{i}", "engine": "qwen", "wav": wav.name,
                     "wav_abs": str(wav), "hard_pass": True,
                     "intended": dict(intended or {"V": -0.9, "A": 0.8, "T": 0.7}),
                     "phonation": {"alpha_db": 6.0, "cpp": 1.1, "h1h2": -13.5}})
        files.append(str(wav))
        if i < scored:
            row = {"wav": str(wav)}
            if stamp:
                row["wav_mtime"] = os.path.getmtime(wav)
            for h in ANCHOR_HEADS + list(qv_extra_heads()):
                row[h] = 0.2 + 0.01 * i
            eiv.append(row)
    (camp / "qc_measures.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (camp / "eiv_scores.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in eiv), encoding="utf-8")
    (camp / "qc_filelist.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    return camp


def qv_extra_heads():
    SCRIPTS.on_path()
    return pytest.importorskip("qc_verdict").EIV_EXTRA_HEADS


def _run_verdict(camp, monkeypatch, qv, *extra):
    monkeypatch.setattr(sys, "argv", ["qc_verdict.py", "--campaign-dir", str(camp),
                                      *extra])
    qv.main()


def test_a_scoring_pass_that_produced_nothing_is_refused_not_reported_as_failure(
        tmp_path, monkeypatch, capsys):
    """The head guard iterates ROWS, so with zero rows it found nothing absent and passed
    the file — and every axis of every clip then read as a direction failure.

    MEASURED on 20713f1: an empty eiv_scores.jsonl gave `V:FAIL A:FAIL T:FAIL` on all four
    clips, `0/4 keeps`, four ids in qc_flags.txt and exit 0. eiv_score.sh exits 0 whenever
    the container came up, so an empty filelist, unreadable wavs or an OOM-skip reaches
    this with nothing else looking wrong (issue #55).
    """
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")
    _stub_anchor(tmp_path, qv, monkeypatch)
    camp = _stub_campaign(tmp_path, scored=0)
    with pytest.raises(SystemExit) as e:
        _run_verdict(camp, monkeypatch, qv, "--eiv", str(camp / "eiv_scores.jsonl"),
                     "--append-flags")
    assert "0 scored rows" in str(e.value)
    assert not (camp / "keeps.jsonl").exists(), \
        "a keeps file was written from scores that do not exist"
    assert not (camp / "qc_flags.txt").exists(), \
        "clips were sent to the ear on the strength of a measurement nobody made"


def test_an_unscored_clip_is_unmeasured_not_a_direction_failure(
        tmp_path, monkeypatch, capsys):
    """Unmeasured and pointed-the-wrong-way used to produce identical output — the same
    console line, the same by-axis tally, the same qc_flags.txt (issue #55). Unmeasured
    still cannot keep, because nothing was confirmed; it simply is not evidence."""
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")
    _stub_anchor(tmp_path, qv, monkeypatch)
    camp = _stub_campaign(tmp_path, n=4, scored=2)
    _run_verdict(camp, monkeypatch, qv, "--eiv", str(camp / "eiv_scores.jsonl"),
                 "--append-flags")
    out = capsys.readouterr().out
    verdicts = {json.loads(l)["id"]: json.loads(l)
                for l in (camp / "qc_verdicts.jsonl").read_text().splitlines() if l.strip()}
    assert verdicts["clip2"]["axis_checks"] == {"V": None, "A": None, "T": None}
    assert verdicts["clip2"]["axes_unmeasured"] == ["A", "T", "V"]
    assert verdicts["clip2"]["axes_checked"] == []
    assert verdicts["clip2"]["keep"] is False, "unmeasured cannot confirm a keep either"
    assert "unmeasured" in out and "NO EIV row" in out
    assert "direction failures among hard-pass clips: 2 clip(s)" in out, \
        "the unscored clips are being counted as engines ignoring direction"
    flags = (camp / "qc_flags.txt").read_text().split()
    assert flags == ["clip0", "clip1"], flags


def test_a_reroll_is_still_caught_after_the_scores_file_is_appended_to(
        tmp_path, monkeypatch, capsys):
    """The guard compared each wav against the mtime of the whole scores FILE, and
    eiv_score.sh appends to that file immediately before this runs — so one newly-scored
    clip refreshed the clock for every other clip and the guard went silent (issue #56).
    Here clip0 is re-rendered after scoring and an unrelated row is appended afterwards,
    which is exactly the order synth_bank.sh produces."""
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")
    _stub_anchor(tmp_path, qv, monkeypatch)
    camp = _stub_campaign(tmp_path, n=2, scored=2)
    scores = camp / "eiv_scores.jsonl"
    now = time.time()
    os.utime(camp / "clip0.wav", (now + 10, now + 10))  # re-rendered in place
    row = {"wav": str(camp / "unrelated.wav"), "wav_mtime": now + 20}
    row.update({h: 0.3 for h in ANCHOR_HEADS + list(qv.EIV_EXTRA_HEADS)})
    with scores.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    os.utime(scores, (now + 30, now + 30))  # the file is now newer than every wav

    _run_verdict(camp, monkeypatch, qv, "--eiv", str(scores))
    out = capsys.readouterr().out
    assert "RE-RENDERED" in out and "clip0" in out, \
        "the reroll is judged on the previous take's scores and nothing says so"
    assert os.path.getmtime(camp / "clip0.wav") < os.path.getmtime(scores), \
        "the file-mtime clock this replaces would have seen nothing here"


def test_a_numeric_string_is_a_label_and_a_word_is_a_complaint(tmp_path, monkeypatch,
                                                               capsys):
    """`--count-directed` printing 0 makes synth_bank.sh announce "(real-audio bank)" and
    skip the stage, so 0 is a positive claim about the campaign. It must never be reached
    by a label the reader could not parse (issue #58): book_ingest.py writes V/A/T straight
    out of the LLM and validates only `register` and `engine`."""
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")
    assert qv.intended_labels({"intended": {"V": "0.7", "A": 0.2}}) == {"V": 0.7, "A": 0.2}
    bad = []
    assert qv.intended_labels({"id": "c0", "intended": {"V": "very sad", "A": True}},
                              bad) == {}
    assert [(i, ax) for i, ax, _ in bad] == [("c0", "V"), ("c0", "A")]
    assert qv.intended_labels({"intended": {"V": None}}, bad := []) == {} and bad == [], \
        "an explicit null is a writer saying `no label`, not a malformed one"

    _stub_anchor(tmp_path, qv, monkeypatch)
    camp = _stub_campaign(tmp_path, n=2, intended={"V": "-0.9", "A": "0.8"})
    _run_verdict(camp, monkeypatch, qv, "--count-directed")
    assert capsys.readouterr().out.strip() == "2", "a numerically-valued label IS a label"

    camp = _stub_campaign(tmp_path, n=2, intended={"V": "very sad"})
    with pytest.raises(SystemExit) as e:
        _run_verdict(camp, monkeypatch, qv, "--count-directed")
    assert "real-audio bank" in str(e.value), \
        "a bank whose labels are unreadable must not be announced as one with no labels"


def test_an_unreadable_intended_label_cannot_keep(tmp_path, monkeypatch, capsys):
    """`intended_labels` DROPS an axis it cannot parse, so a row whose labels were all
    gibberish reached `keep` with `checks == {}` — and `all(...)` over an empty dict is
    True, so the clip was written to keeps.jsonl, which the module docstring defines as
    "clips + CONFIRMED labels". Nothing was confirmed. It was also counted as `undirected`,
    beneath a line saying those rows have nothing to confirm, which is exactly what a row
    stating an unreadable direction does not get to claim.

    `--count-directed` does not cover this: it refuses only when EVERY label in the bank is
    unreadable, so the ordinary partial case — 99 good rows, 1 malformed — sails past the
    pre-flight and lands here."""
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")
    _stub_anchor(tmp_path, qv, monkeypatch)
    camp = _stub_campaign(tmp_path, n=2, intended={"V": "very sad", "A": "angry", "T": "x"})
    _run_verdict(camp, monkeypatch, qv, "--eiv", str(camp / "eiv_scores.jsonl"))
    out = capsys.readouterr().out
    verdicts = [json.loads(l) for l in
                (camp / "qc_verdicts.jsonl").read_text().splitlines() if l.strip()]
    assert [v["keep"] for v in verdicts] == [False, False], \
        "a clip whose stated direction nobody could read was confirmed as a keep"
    assert (camp / "keeps.jsonl").read_text().strip() == ""
    assert verdicts[0]["axis_checks"] == {"V": "unreadable", "A": "unreadable",
                                          "T": "unreadable"}
    assert verdicts[0]["axes_unreadable"] == ["A", "T", "V"]
    assert verdicts[0]["axes_unmeasured"] == [], \
        "an unparsable label is not a measurement we failed to take — different repair"
    assert "row(s) carry NO intended V/A/T" not in out, \
        "rows with unreadable labels are being reported as rows with no labels"


def test_one_bad_axis_is_enough_to_hold_a_clip_out_of_keeps(tmp_path, monkeypatch, capsys):
    """The partial case, decided by the owner 2026-08-11: an axis whose label is present
    and unreadable confirmed nothing, so it blocks the keep on its own — the same doctrine
    as NONE IS NOT FALSE, applied to the label side. The readable axis is still checked and
    still reported; the bank's repair is to fix the label, not to re-score the clip.

    ⚠ The intended V must CONFIRM here, and the first version of this test had it pointing
    the wrong way. That made the clip a direction failure as well, so "held out on the label
    alone" was false of it — and the assertion still passed, because the summary counted
    every row with an unreadable axis rather than the rows the label actually held out.
    Two defects agreeing. See the sibling test for the case this one must not cover.
    """
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")
    _stub_anchor(tmp_path, qv, monkeypatch)
    camp = _stub_campaign(tmp_path, n=1, intended={"V": -0.9, "A": "angry"})
    _run_verdict(camp, monkeypatch, qv, "--eiv", str(camp / "eiv_scores.jsonl"))
    out = capsys.readouterr().out
    v = json.loads((camp / "qc_verdicts.jsonl").read_text().strip())
    assert v["axis_checks"]["A"] == "unreadable" and v["axes_unreadable"] == ["A"]
    assert v["axis_checks"]["V"] is True, \
        "the readable axis must still be CHECKED and confirmed — the hold-out is the label"
    assert v["keep"] is False
    assert "1 clip(s) are held out of keeps.jsonl" in out, \
        "the summary must say the hold-out happened; a silent one reads as a scoring gap"
    assert "T" not in v["axis_checks"], \
        "an axis with no label at all is still ABSENT, not unreadable"


def test_only_the_clips_the_label_actually_held_out_are_reported_as_a_label_repair(
        tmp_path, monkeypatch, capsys):
    """"On that alone" is a claim, and it was counted as "has an unreadable axis".

    That folds in clips rejected by the hard gate or by a real `False` direction verdict —
    clips whose keep does NOT come back when the label is fixed. The operator was told the
    opposite in the sentence's most emphatic clause, which is the same defect as the wide
    `except ValueError` this branch narrowed in register_audition.py: two different reasons
    for one outcome, reported as one.
    """
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")
    _stub_anchor(tmp_path, qv, monkeypatch)
    camp = _stub_campaign(tmp_path, n=2, intended={"V": -0.9, "A": "angry"})
    rows = [json.loads(l) for l in
            (camp / "qc_measures.jsonl").read_text().splitlines() if l.strip()]
    rows[0]["hard_pass"] = False          # rejected for a reason the label cannot fix
    (camp / "qc_measures.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    _run_verdict(camp, monkeypatch, qv, "--eiv", str(camp / "eiv_scores.jsonl"))
    out = capsys.readouterr().out
    assert "1 clip(s) are held out of keeps.jsonl on that alone" in out, \
        "clip1 is the only one whose keep returns when the label is fixed"
    assert "1 further clip(s) carry an unreadable label AND are rejected independently" in out
    assert "Fixing their labels will NOT make them keep" in out


def test_an_unreadable_label_on_an_unscored_clip_is_a_scoring_repair_first(
        tmp_path, monkeypatch, capsys):
    """The two-bucket split was wrong in BOTH directions, and this is the harder half.

    `only_the_label` asked whether the READABLE axes confirmed. A row whose only intended
    axis is the unreadable one has none, so `all(...)` ran over an empty set and returned
    True — `hard_pass` alone put a clip with no EIV row at all into "would otherwise keep …
    this is a LABELLING repair". Fixing that label cannot make it keep; there is nothing to
    check the fixed label against. The mirror defect swept every UNMEASURED row into "hard
    gate, or a measured direction disagreement", which is NONE IS NOT FALSE (issue #55)
    stated backwards in the file whose docstring insists on it.
    """
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")
    _stub_anchor(tmp_path, qv, monkeypatch)
    camp = _stub_campaign(tmp_path, n=2, scored=1, intended={"A": "angry"})
    _run_verdict(camp, monkeypatch, qv, "--eiv", str(camp / "eiv_scores.jsonl"))
    out = capsys.readouterr().out
    assert "1 clip(s) are held out of keeps.jsonl on that alone" in out, \
        "the scored clip is the only one a label fix would bring back"
    assert "1 further clip(s) carry an unreadable label AND have NO EIV ROW" in out
    assert "Run the SCORING pass over them first" in out, \
        "the operator is being sent to fix a label when the clip was never scored"
    assert "measured direction disagreement" not in out, \
        "an unmeasured axis is not a measured disagreement — issue #55, restated backwards"


def test_a_scored_clip_with_no_phonation_is_not_sent_back_to_the_scoring_pass(
        tmp_path, monkeypatch, capsys):
    """`measured_z is None` has TWO causes and this module is emphatic they differ.

    Both used to be told "score them first". For a clip stage 1 could not measure phonation
    on, that instruction is a loop: re-run the scoring pass, get back the row it already
    had, see no change. The run even says "were scored" about the same clip three lines
    later — the bucket and the summary contradicting each other in one output.
    """
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")
    _stub_anchor(tmp_path, qv, monkeypatch)
    camp = _stub_campaign(tmp_path, n=1, scored=1, intended={"A": "angry"})
    rows = [json.loads(l) for l in
            (camp / "qc_measures.jsonl").read_text().splitlines() if l.strip()]
    rows[0].pop("phonation")              # scored, but stage 1 measured no phonation
    (camp / "qc_measures.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    _run_verdict(camp, monkeypatch, qv, "--eiv", str(camp / "eiv_scores.jsonl"))
    out = capsys.readouterr().out
    assert "were SCORED but carry no phonation measures from stage 1" in out
    assert "Do NOT re-run the scoring pass" in out, \
        "the remedy sends the operator round a loop that cannot terminate"
    assert "Run the SCORING pass over them first" not in out, \
        "this clip already has its EIV row — that is the other cause, with another repair"


def test_the_anchor_is_held_to_the_same_standard_as_the_campaign_file(
        tmp_path, monkeypatch, capsys):
    """main() refuses a campaign eiv file missing a combo head; build_anchors filled the
    same gap with 0.0, which collapses that head's std to the 1e-9 floor and turns its z
    into `x * 1e9` inside the V dot product (issue #54).

    MEASURED on the shipped anchor: the only two unanchored heads carry weight exactly
    0.0, so V is unaffected today — bulk1 is min -2.755 / max 3.272 either way. The
    asymmetry is what is fixed: one weight refit would otherwise turn that live silently.
    """
    SCRIPTS.on_path()
    qv = pytest.importorskip("qc_verdict")

    _stub_anchor(tmp_path, qv, monkeypatch, weights=[0.5, 0.3, 0.0], drop=("Sadness",))
    anchor, heads, w = qv.build_anchors()
    assert anchor["Sadness"] == (0.0, 1.0), \
        "a zero-weight head must not be anchored on a 1e-9 std it could be refit into"
    out = capsys.readouterr().out
    assert "Sadness" in out and "0/40" in out and "weight" in out, \
        "an unanchored head is filled silently, which is how it stays unanchored"

    _stub_anchor(tmp_path, qv, monkeypatch, weights=[0.5, 0.3, -0.2], drop=("Sadness",))
    with pytest.raises(SystemExit) as e:
        qv.build_anchors()
    assert "Sadness" in str(e.value) and "moves V" in str(e.value)


def test_the_sampler_does_not_read_an_unmeasured_axis_as_a_disagreement():
    """audit_sampler's disagreement-first pass took every falsy axis check as a fail, so
    the None that now means UNMEASURED would have sent the ear a clip on a finding nobody
    made — and `not keep` is true of an unmeasured clip too (issue #55)."""
    src = _src("audit_sampler.py")
    assert "if ok is False" in src, \
        "unmeasured axes are being sampled as axis-disagree"
