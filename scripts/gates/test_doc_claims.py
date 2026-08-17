#!/usr/bin/env python3
"""Gate: the notes' corpus/checkpoint numbers must agree with the artifacts on disk.

WHY THIS EXISTS (2026-08-09, owner-commissioned)
------------------------------------------------
This repo catches forked CODE constants and nothing else. `test_skill_files.py` verifies 94
claims against source; `test_vat_dim_seams.py` has 30+; there are parity tests for
`DASH_RUN`, the delivery vocabulary and the device G2P tables, and `test_data_mirrors.py`
for `/data` copies. **Every one of those exists because a value forked once.** Documentation
had no equivalent — a doc fork was caught by a code review, if one happened to run.

The scale of the exposure, measured the day this was written: **30 notes files, ~9,000
lines, 18 of them declaring themselves SSOT or canon for something, and every load-bearing
corpus number restated in 6-8 files.** Consolidation does not fix that on its own. LongCat
is the proof: `23c6af3` consolidated three contradictory bench states into one record and
forbade restating it, and **it did not hold for 24 hours** — CL-L2 found two rows still
restating a status the commit had just centralised. Consolidation is a one-time act; drift
is continuous; only a check is continuous.

**IT COMPARES DOCS TO ARTIFACTS, NOT TO EACH OTHER — that is the whole design.** Two
documents can agree and both be wrong. The 2026-08-09 review found `10,653` in
`training-sources.md` and called it "simply wrong" because differencing two other numbers
gave 10,997; that finding was itself wrong (10,653 is Emilia's TRAIN rows, 10,997 is
train+val, both in `derivation_report.json`), and the fix pass propagated the error. It
survived a review, a fix and a re-read, and fell out the moment a number was checked against
the artifact instead of against another document.

**TRAIN, VAL AND TOTAL ARE SEPARATE FACTS HERE, deliberately**, because conflating them is
precisely the mistake above. A registry that recorded "v5 rows" as one number would have
reproduced it.

WHAT IT DOES NOT COVER — read this before trusting a pass
---------------------------------------------------------
* **Two registries, and each is narrow.** `FACTS` holds corpus/checkpoint numbers (owner
  scope, 2026-08-09). `PROTOCOL_FACTS` holds the workflow lane's own constants (owner,
  2026-08-17). Neither covers statuses, prose, or the sequencing claims that made up most of
  the 2026-08-09 review's product findings.
* **Only phrasings the patterns match.** A number written in a form no pattern recognises is
  invisible here — a silent miss, not an error. When a check goes green it means "no
  RECOGNISED statement disagrees", never "the docs are correct".
* **Historical mentions are exempted by explicit substring**, listed per fact and printed on
  every run, so an exemption cannot quietly become a hiding place.
* **Only the facts whose artifact is on THIS machine.** Each fact names the files it reads;
  a fact whose artifact is absent is skipped and PRINTED, never silently passed. That is
  per-fact deliberately — the prerequisite used to be all-or-nothing in the pytest harness,
  so a machine holding v5 and not v6 checked *nothing* and reported a skip, discarding the
  ten v5 facts it could have enforced (#52). A fact whose artifact IS present and whose key
  is missing is still a failure.

Run:  .venv/bin/python scripts/gates/test_doc_claims.py
Exit: 0 every recognised claim agrees with disk; 1 otherwise.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# one level deeper since #26 step 3 moved the gates into scripts/gates/
REPO = os.path.dirname(os.path.dirname(HERE))
NOTES = os.path.join(REPO, "notes")
DOCS = os.path.join(REPO, "docs")
V5 = os.path.join(REPO, "data", "libritts_r_emilia_vat_v5")
V6 = os.path.join(REPO, "data", "libritts_r_emilia_expressive_vat_v6")
V4 = os.path.join(REPO, "data", "libritts_r_vat_v4")
HOLDOUT = os.path.join(REPO, "data", "libritts_r_holdout_devclean")
WORKFLOW = os.path.join(REPO, "workflow")
CONFIG_ENV = os.path.join(WORKFLOW, "config.env")

# The artifacts themselves, named once. Every fact lists the ones it reads, so a machine
# that holds one corpus and not another enforces what it can (see the docstring, #52).
V5_TRAIN = os.path.join(V5, "train_op.txt")
V5_VAL = os.path.join(V5, "val_op.txt")
V5_REPORT = os.path.join(V5, "derivation_report.json")
V6_REPORT = os.path.join(V6, "derivation_report.json")
V4_TRAIN = os.path.join(V4, "train_op.txt")
V4_VAL = os.path.join(V4, "val_op.txt")
HOLDOUT_FILE = os.path.join(HOLDOUT, "holdout_8w.txt")


def rows(path):
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def report(key, path=V5_REPORT):
    with open(path, encoding="utf-8") as fh:
        node = json.load(fh)
    for part in key.split("."):
        node = node[part]
    return node


def report6(key):
    """v6's derivation report — a SECOND artifact, and the reason for `report`'s `path`.

    Every generation of the corpus restates the same KINDS of number (candidates, keeps,
    what was dropped and why) about different data. Reading them from one report by
    default is convenient and was also how the digit-drop fact below came to claim two
    generations at once.
    """
    return report(key, V6_REPORT)


def drop_count(key, path=V5_REPORT):
    """A `<corpus>.dropped.<reason>` counter, where **absent means zero**.

    ⚠ NOT A CONVENIENCE — the plain reader was wrong here, and wrong in the direction this
    file exists to prevent. Both builders tally drops in a `collections.Counter` and
    serialise it as `dict(dropped)` (`merge_expressive_registers.py:457`,
    `merge_emilia_corpus.py:410`), and a Counter has no entry for a reason that never
    fired. `expressive.dropped.digits` is in today's report only because six rows happened
    to carry digits. Rebuild from inputs carrying none — **the state D-M3 exists to
    produce** — and the key vanishes, `report()` raised `KeyError`, and `main()` printed
    "cannot read the artifact" about an artifact that is present and simply says zero. A
    correct corpus, correct documents, and a red gate: the rule's own success condition
    firing the check (#51).

    Only the FINAL hop defaults. A missing report file still raises `OSError` and a missing
    `dropped` map still raises `KeyError`, because those are genuinely unreadable sources
    and a fact whose source is missing is not a passing fact.
    """
    parent, _, reason = key.rpartition(".")
    node = report(parent, path)
    if not isinstance(node, dict):
        raise KeyError(f"{parent!r} in {os.path.relpath(path, REPO)} is not a drop map "
                       f"(got {type(node).__name__}) — the report schema moved")
    return node.get(reason, 0)


def drop_count6(key):
    return drop_count(key, V6_REPORT)


def comma(n):
    return f"{n:,}"


def setting(key, path=CONFIG_ENV):
    """An integer `KEY=value` from `workflow/config.env`, the file that OWNS the constant.

    ⚠ COMMENT LINES ARE SKIPPED, and that is what makes `config.env` safe to also SCAN.
    The file states several of its own constants in prose directly above the assignment
    ("Per-issue fix-pass cap. Three, unless the owner says otherwise."), so it is both a truth
    source and a scanned document. Reading only real assignments keeps those two roles from
    touching: the prose is checked against the value, never against itself.

    Absent or empty is a `KeyError`, never a default. A protocol fact whose constant has been
    deleted is not a passing fact — the documents would then be restating something nothing
    defines, which is the drift this registry exists to find.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() != key:
                continue
            v = v.strip()
            if not v:
                raise KeyError(f"{key} is empty in {os.path.relpath(path, REPO)}")
            return int(v)
    raise KeyError(f"{key} is not set in {os.path.relpath(path, REPO)}")


# Small integers get written as words in prose at least as often as as digits — three of the
# five live restatements of the pass cap spell it "three". A cap that moves to 4 must turn
# those lines red too, so the accepted spellings include the word and its capitalised form.
# Bounded deliberately: past twelve, prose uses digits, and a longer table would only be a
# place for a mistake to hide.
_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def spellings(truth, words=False):
    """Every form a document may legitimately write `truth` in."""
    out = {comma(truth), str(truth)}
    if words and truth in _WORDS:
        out.add(_WORDS[truth])
        out.add(_WORDS[truth].capitalize())
    return out


# --- the registry -------------------------------------------------------------------
#
# Each fact: how to READ it from disk, WHICH FILES that read needs, and the phrasings that
# STATE it in prose. Patterns carry exactly one capture group — the number as written. Keep
# patterns narrow: a loose one produces false failures, which is how a gate gets switched
# off.
#
# `artifacts` is the list of paths `truth` opens. A fact whose artifact is not on this
# machine is skipped and printed; a fact whose artifact IS here and disagrees is a failure.
# It is per-fact so that a v5-only machine still enforces v5 (#52) — and it must list EVERY
# file the lambda touches, or the fact goes red on a host that simply lacks that corpus.
#
# `exempt` lines are skipped and announced. They exist for deliberate historical mentions
# ("this cell read X until <date>"), which the repo writes on purpose.
FACTS = [
    {
        "name": "v5 TRAIN rows",
        "truth": lambda: rows(V5_TRAIN),
        "artifacts": [V5_TRAIN],
        "scope": r"v5|libritts_r_emilia_vat_v5",
        "patterns": [r"([\d,]{6,}) train / [\d,]+ val", r"([\d,]{6,}) train rows",
                     r"([\d,]{6,}) train \+ [\d,]+ val\)"],
        "exempt": [],
    },
    {
        "name": "v5 VAL rows",
        "truth": lambda: rows(V5_VAL),
        "artifacts": [V5_VAL],
        "scope": r"v5|libritts_r_emilia_vat_v5",
        "patterns": [r"[\d,]{6,} train / ([\d,]+) val", r"[\d,]{6,} train \+ ([\d,]+) val\)"],
        "exempt": [],
    },
    {
        "name": "v5 TOTAL rows",
        "truth": lambda: rows(V5_TRAIN) + rows(V5_VAL),
        "artifacts": [V5_TRAIN, V5_VAL],
        "scope": r"v5|libritts_r_emilia_vat_v5",
        "patterns": [r"\*\*([\d,]{6,})\*\* \| \*\*\+[\d,]+ Emilia"],
        "exempt": [],
    },
    {
        "name": "v5 speakers",
        "truth": lambda: report("n_spks"),
        "artifacts": [V5_REPORT],
        # Only the composite idioms the corpus is actually quoted in. A bare "N speakers"
        # matches every dataset row in dataset-landscape.md, which is how this check first
        # produced 20 false failures — and a noisy gate is a gate someone turns off.
        "scope": r"v5|libritts_r_emilia_vat_v5",
        "patterns": [r"[\d.]+ h(?:ours)? [/·|] \*?\*?([\d,]+) (?:spk|speakers)",
                     r"([\d,]+) speakers\*\*, against"],
        "exempt": [],
    },
    {
        "name": "Emilia keeps (train+val)",
        "truth": lambda: report("emilia.kept"),
        "artifacts": [V5_REPORT],
        "scope": r"Emilia|emilia",
        "patterns": [r"\+?([\d,]+) Emilia-YODAS keeps", r"([\d,]+) keeps of `?libritts"],
        "exempt": ["until 2026-08-09", "10,653 and 10,997"],
    },
    {
        "name": "Emilia TRAIN rows",
        "truth": lambda: report("emilia.train"),
        "artifacts": [V5_REPORT],
        "scope": r"Emilia|emilia",
        "patterns": [r"\*\*([\d,]+) train \+ [\d,]+ val\*\*"],
        "exempt": [],
    },
    {
        "name": "Emilia candidates before filtering",
        "truth": lambda: report("emilia.candidates"),
        "artifacts": [V5_REPORT],
        "scope": r"Emilia|emilia|13,141",
        "patterns": [r"planned ([\d,]+) became", r"([\d,]+) candidates"],
        "exempt": [],
    },
    {
        # ⚠ TWO DIFFERENT DIGIT DROPS EXIST AND THEY ARE STATED IN THE SAME WORDS.
        # This one is v5's Emilia merge — 1,676 of 13,141. The v6 append has its own,
        # 6 of 832 (the fact below). The first version of this entry owned the phrase
        # "dropped on digits" outright, so when v6 landed it claimed BOTH: it read
        # README's v6 count as a wrong Emilia count, and on STATE.md's "6 of the 832
        # dropped on digits" it captured the **832** — the staged total, which is not
        # what that sentence claims — rather than the 6. Two correct documents, a
        # permanently red gate. **That is the failure this whole file exists to
        # prevent**: a check nobody can turn green is a check everybody learns to
        # ignore, and it goes on ignoring the fork it was built to catch.
        #
        # The two facts are kept apart on TWO axes, and both are load-bearing:
        #   * SCOPE — Emilia's own markers only. The bare word "digit" used to be in
        #     here as a catch-all for lines that state the fact without naming Emilia;
        #     it is what swallowed v6. `digit rows` / `13,141` / `D-M3` cover those
        #     lines without reaching v6's phrasings. ⚠ NEVER SCOPE A FACT ON THE VALUE
        #     IT CHECKS: a line held in scope only by the number under test drops out
        #     of scope the moment that number is corrupted, which is the one instant
        #     the check had a job. `13,141` is safe here because it is the CANDIDATE
        #     count and the fact under test is the digit count — two different numbers
        #     on the same line — but every scope on this fact except that one is a
        #     phrase, so a wrong number cannot hide by being wrong.
        #   * CAPTURE — anchored to the number actually being claimed. `(?<!of the )`
        #     refuses the bare "N of the M" idiom (that shape is v6's, below), and
        #     `(?<![\d,])` stops the capture starting INSIDE a longer number, which is
        #     how a lookbehind alone gets fooled: without it, "of the 13,141" merely
        #     shifts the match one digit right and reports "3,141".
        #
        # ⚠ THE REFUSAL ABOVE LEFT A SILENT MISS, AND IT IS CLOSED BY THE LAST PATTERN.
        # `(?<!of the )` made a WRONG v5 count written as "1,677 of the 13,141 dropped on
        # digits" invisible to this entry by construction, while the v6 entries — scoped on
        # the bare rule id `D-M3` at the time — claimed it and went red on a correct v5
        # sentence (#48). Both halves are fixed here: `D-M3` is out of v6's scope, and this
        # entry now accepts the "N of the M" idiom when M is ANCHORED ON 13,141, Emilia's
        # own candidate count. That anchor is what stops the two entries ever claiming one
        # sentence again — v6's staged total is 832 and can never satisfy it — and 13,141 is
        # a legal anchor for the same reason it is a legal scope marker: it is a DIFFERENT
        # number from the one under test, so corrupting the digit count cannot dissolve it.
        # `notes/training-sources.md:133` already writes the neighbouring fact in this idiom
        # ("2,144 of the 13,141 keeps did not make it: 1,676 carry digits"), so regularising
        # the two halves of that one sentence is all it takes. Not hypothetical: one edit.
        "name": "Emilia rows dropped on digits (v5 merge)",
        "truth": lambda: drop_count("emilia.dropped.digits"),
        "artifacts": [V5_REPORT],
        "scope": r"Emilia|emilia|13,141|digit rows|D-M3",
        "patterns": [r"(?<!of the )(?<![\d,])([\d,]+) (?:rows )?dropped on digits",
                     r"(?<![\d,])([\d,]+) digit rows",
                     r"(?<![\d,])([\d,]+) carry digits",
                     r"(?<![\d,])([\d,]+) of the 13,141 (?:rows |keeps )?dropped on digits"],
        "exempt": [],
    },
    {
        # The v6 append's own digit drop — same rule (D-M3), different corpus, and the
        # documents are right to state it separately. Scope carries v6's markers and
        # deliberately NOT `Emilia`; the patterns match only the two idioms v6 is
        # actually written in, and both anchor the capture on the count being claimed
        # rather than on the staged total it is quoted out of.
        #
        # ⚠ `D-M3` IS NOT A v6 MARKER AND IS NO LONGER IN THIS SCOPE (#48). It is the RULE
        # id, and every v5 statement of the v5 digit drop carries it too — so this entry
        # claimed v5 sentences the moment one of them was written in the "N of the M" form
        # these patterns require, which is one wording away from what
        # `notes/training-sources.md:129` says today. The file's old defence — "harmless BY
        # CONSTRUCTION, none of them is written in that form" — was a claim about the
        # CURRENT WORDING, and nothing holds the wording. `notes/STATE.md:313`, the line
        # that motivated `D-M3` because it named no corpus, was given a corpus marker in
        # the same commit instead: the fix for a line with no handle is to give the LINE a
        # handle, never to hand the registry a rule id that spans two generations.
        "name": "v6 append rows dropped on digits",
        "truth": lambda: drop_count6("expressive.dropped.digits"),
        "artifacts": [V6_REPORT],
        "scope": r"v6|expressive|append|clips resolve",
        "patterns": [r"(?<![\d,])([\d,]+) of the [\d,]+ (?:rows )?dropped on digits",
                     r", ([\d,]+) (?:rows )?dropped on digits\)"],
        "exempt": [],
    },
    {
        # Anchoring the fact above onto the 6 would have retired the only check the 832
        # in that sentence was getting — a silent loss of coverage dressed up as a fix.
        # It is a real documented figure with a real artifact behind it, so it gets its
        # own entry and the same sentence now verifies BOTH of its numbers.
        #
        # ⚠ Its scope was `…|832` for one draft, and corrupting the 832 on STATE.md:313
        # made the gate go GREEN — the line's only scope marker was the number under
        # test, so a wrong value removed its own check. Caught by testing the RED
        # direction, which is the only way that class of hole is ever found. Every
        # marker here is now a phrase.
        "name": "v6 append rows staged (candidates)",
        "truth": lambda: report6("expressive.candidates"),
        "artifacts": [V6_REPORT],
        "scope": r"v6|expressive|append|clips resolve",
        "patterns": [r"[\d,]+ of the ([\d,]+) (?:rows )?dropped on digits",
                     r"0 of the ([\d,]+) clips resolve",
                     r"append(?: set)? is settled at \*\*([\d,]+) rows\*\*",
                     r"(?<![\d,])([\d,]+) rows to append",
                     r"(?<![\d,])(\d[\d,]*) rows staged"],
        "exempt": [],
    },
    {
        # STAGED IS NOT KEPT, and the documents forked on exactly that. `notes/README.md:29`
        # said "832 rows appended" beside the totals that refute it — 41,937 + 1,331 − 42,442
        # = 826 — while the root `README.md:75` said 826, and the registry checked only the
        # number the two agreed on (#50). 832 is what was staged; 826 is what survived D-M3.
        # Both are real keys from one builder (`merge_expressive_registers.py:455-456`), so
        # the cure is one more entry, not a choice between them.
        #
        # ⚠ THE PATTERNS ARE DELIBERATELY TIGHT — "N appended" alone reaches
        # `notes/quality-gap-plan.md:224`, where **846** is the correct historical count for
        # a different filter stage (after dedup, before the duration drop). That line is
        # exempted as well as unmatched: two guards, because this fact's whole failure mode
        # is one population's number read as another's, and the exemption is printed on
        # every run so it cannot become a hiding place.
        #
        # ⚠ `(\d[\d,]*)`, NOT `([\d,]+)` — the number must START with a digit. `[\d,]+`
        # matches a BARE COMMA, so the first draft of the first pattern here fired on
        # `notes/CHANGELOG.md:155` ("byte-identical, appended rows carrying a real") and
        # reported *doc says ','*. Caught by running it; it is the same class of red-gate-on-
        # correct-prose this whole file exists to prevent, one character wide.
        "name": "v6 append rows kept (built)",
        "truth": lambda: report6("expressive.kept"),
        "artifacts": [V6_REPORT],
        "scope": r"v6|expressive|append",
        "patterns": [r"(?<![\d,])(\d[\d,]*) appended rows",
                     r"(?<![\d,])(\d[\d,]*) rows appended"],
        "exempt": ["846 appended"],
    },
    {
        "name": "v4 TOTAL rows (the base v5 merges into)",
        "truth": lambda: rows(V4_TRAIN) + rows(V4_VAL),
        "artifacts": [V4_TRAIN, V4_VAL],
        "scope": r"v4|libritts_r_vat_v4|31,44",
        "patterns": [r"v4 rows verbatim.{0,40}?([\d,]{6,})", r"([\d,]{6,}) clips `vat3c"],
        "exempt": [],
    },
    {
        "name": "holdout clips",
        "truth": lambda: rows(HOLDOUT_FILE),
        "artifacts": [HOLDOUT_FILE],
        "scope": r"holdout|dev-clean|dev_clean",
        "patterns": [r"([\d,]+) clips? / [\d.]+ h / \*?\*?40 speakers",
                     r"([\d,]+) clips? / \*\*[\d.]+ h\*\* / 40 speakers",
                     r"of ([\d,]+) clips, never trained on"],
        "exempt": [],
    },
]


# --- the protocol registry ------------------------------------------------------------
#
# SAME MACHINE, DIFFERENT TRUTH SOURCE (owner, 2026-08-17). Every fact above reads a corpus
# ARTIFACT; every fact here reads a CONSTANT from the file that owns it. The scanning half —
# scope, exemptions, capture — is shared, because a second copy of that regex logic is exactly
# the fork this file exists to catch.
#
# WHY IT IS A SEPARATE LIST, measured before it was written. Widening `docs()` to
# `AGENTS.md`, `CLAUDE.md` and `workflow/*.md` enrols **zero** statements into the registry
# above: those five files hold 472 numeric tokens and not one is written in a phrasing any
# corpus fact recognises. The reason is structural rather than incidental — the numbers there
# are protocol constants, not corpus counts — so the file-set widening was never going to be
# enough on its own, and a registry keyed on `derivation_report.json` could not have grown
# into these however many files it scanned.
#
# ⚠ THESE ALL AGREE TODAY. This is prophylactic work and the entries should not pretend
# otherwise. What justifies it is the restatement count: measured by turning `MAX_PASSES` to 4
# and reading the failures, the pass cap is restated SEVEN times across FOUR files —
# `WORKFLOW.md` twice, `DEVELOPER.md` three times, `issue.py`'s module docstring, and
# `config.env`'s own comment one line above the assignment. Nothing compared any two of them.
#
# ⚠ THEY ALSO RUN EVERYWHERE, unlike the corpus facts. `workflow/config.env` is tracked, so
# these are the first facts in this file that a laptop and a CI runner both check for real
# rather than skip.
#
# ⚠ WHAT IS DELIBERATELY ABSENT: the issue BODY cap (200,000) and the tracker's `state`
# vocabulary. Their only authority is the live PocketBase schema, which is not a path on disk
# — so `unreadable()` cannot express the prerequisite, and an offline laptop would report a
# doc defect where there is none. That needs a probe-shaped prerequisite, which is the next
# increment and not a rename of this one.
PROTOCOL_FACTS = [
    {
        # ⚠ SIX INDEPENDENT SPELLINGS, and `config.env`'s own comment is one of them — the
        # tightest possible fork, sitting one line above the assignment it restates.
        # Three of the six write the number as a word, hence `words`.
        "name": "review pass cap (MAX_PASSES)",
        "truth": lambda: setting("MAX_PASSES"),
        "artifacts": [CONFIG_ENV],
        "words": True,
        # Phrases only. `agent_passes` is the field, `counter` is what the map calls it, and
        # `fix-pass cap` is config.env's own wording — none of them is the number under test,
        # so a wrong value cannot drop its own line out of scope.
        "scope": r"agent_passes|counter|fix-pass cap|attempts|passes",
        "patterns": [r"counter is (\d+)",
                     r"`agent_passes` is already (\d+)",
                     r"never past (\d+)",
                     r"(?:own|fresh) (\w+) passes",
                     r"fix-pass cap\. (\w+)"],
        "exempt": [],
    },
    {
        # One restatement today, in REVIEWER.md's field table. It is kept because that
        # sentence also claims the cap is "enforced by the schema" — checked by hand
        # 2026-08-17 and TRUE (`issue_comments.body.max` really is 1500), which means the two
        # numbers must move together or the claim silently becomes false.
        #
        # ⚠ THE PATTERN IS TIGHT ON PURPOSE, and the tightness is the ONLY thing protecting a
        # correct sentence. `config.env` is scanned as well as read, and the comment two lines
        # above this constant says "a reviewer averaged 1839 characters" — a DIFFERENT fact
        # about a different population. A pattern of `([\d,]+) characters` reads it as a wrong
        # cap and turns the gate red on correct prose, which is the noisy-gate failure this
        # file documents at length.
        #
        # ⚠ `[Cc]omment`, NOT `comment`, AND THAT MATTERS MORE THAN IT LOOKS. With the
        # lower-case spelling the 1839 line fell out of scope on CASE ALONE — so the pattern's
        # tightness was untested, the mutation that loosened it left the suite green, and
        # sentence-casing that one word (or writing "the comment cap") would have armed the
        # trap with nobody touching this file. Protection by accident is not protection.
        "name": "comment length cap (COMMENT_MAX)",
        "truth": lambda: setting("COMMENT_MAX"),
        "artifacts": [CONFIG_ENV],
        "scope": r"hard maximum|COMMENT_MAX|[Cc]omment",
        "patterns": [r"hard maximum ([\d,]+) characters"],
        "exempt": [],
    },
]


# The checkpoint a concluded run SELECTED. Not a count, so it is checked differently: the
# file the verdict names must exist, because "warm start from epNNN" is worthless if it does
# not.
#
# ⚠ DERIVED, NOT HARDCODED — and the first version of this got it wrong. It pinned
# `checkpoint_epoch=019.ckpt` and a run directory, which is a DEFAULT THAT ENCODES A
# JUDGMENT ("the current pick") rather than a fact. Judgments rot and the constant cannot
# know it: the day v6 concludes, that line would have been quietly wrong inside the very
# gate written to stop documents being quietly wrong.
#
# The invariant is self-updating instead: every `SELECTED.md` on disk names a checkpoint,
# and that checkpoint must exist. A new run's verdict is picked up with no edit here.
TRAIN_LOGS = "/data/model-training/sonora/logs/train"
CKPT_IN_VERDICT = re.compile(r"checkpoint_epoch=\d+\.ckpt")


def selected_checkpoints():
    """-> [(experiment, checkpoint filename, verdict path)] for every concluded run."""
    out = []
    if not os.path.isdir(TRAIN_LOGS):
        return out
    for exp in sorted(os.listdir(TRAIN_LOGS)):
        verdict = os.path.join(TRAIN_LOGS, exp, "SELECTED.md")
        if not os.path.isfile(verdict):
            continue
        with open(verdict, encoding="utf-8") as fh:
            names = CKPT_IN_VERDICT.findall(fh.read())
        # The heading names the pick; later mentions may reference others (ep039 as the
        # newest, ep009 as the runner-up), so the FIRST is the one the verdict selects.
        if names:
            out.append((exp, names[0], verdict))
    return out


def unreadable(fact, exists=None):
    """The artifacts this fact needs that are not on this machine.

    A named function rather than an inline comprehension so the per-fact prerequisite can
    be TESTED with a synthetic `exists` — "only v6 is missing" is a one-line probe that runs
    on a laptop, where the real answer would be "everything is missing" and the regression
    #52 describes would be unguarded exactly where it is most likely to reappear.

    `exists` resolves at CALL time, not as a default argument, so patching `os.path.exists`
    reaches this too. Bound as a default it would not, and the test that drives it would
    have passed while testing nothing.
    """
    exists = exists or os.path.exists
    return [p for p in fact["artifacts"] if not exists(p)]


def docs():
    """Every file whose corpus numbers this gate is allowed to see.

    `configs/data/*.yaml` was added 2026-08-11 (owner). The v6 config's header block is the
    densest set of corpus numbers in the repo — 826 / 832 / 154 / 82 / 78 / 251 -> 328 /
    69 -> 68 — and it sat entirely outside the registry's field of view, so the same `826`
    and `832` written one directory over in `notes/README.md` were checked in both
    directions while these were not checked at all. That is the #50 failure ("the gate
    could not see it, because it checked the number the two READMEs agreed on") relocated
    rather than closed.

    ⚠ Measured when it was added: this widens the FILE set, not the fact set. Scope alone
    does not enrol a line — the number must also be written in a phrasing some FACT
    recognises — so it moved 32 files / 36 recognised statements to 42 / 37. One statement
    today; the point is that the next config line written in a recognised phrasing is
    enforced without anyone remembering to come back here.

    `AGENTS.md`, `CLAUDE.md`, the workflow lane's prose, `workflow/config.env` and
    `workflow/scripts/issue.py` were added 2026-08-17 (owner), and the same measurement was
    taken first: against the CORPUS registry alone they enrol **zero** statements. They are
    here for `PROTOCOL_FACTS`, which is the registry their numbers actually belong to.

    ⚠ TWO NON-MARKDOWN FILES ARE IN HERE ON PURPOSE. `config.env` states its own constants in
    prose above the assignments, and `issue.py`'s module docstring restates the pass cap
    beside the code that enforces it — the most dangerous restatement in the lane, because it
    is the one a reader trusts most. A claim is a claim wherever it is written; the file
    extension is not the thing being checked.

    ⚠ THIS SET DOES NOT TRAVEL. `workflow/` is meant to be copied wholesale into another repo,
    and this gate is Sonora's. A ported lane has the constants and the prose but no check that
    compares them, which `workflow/WORKFLOW.md` records under "Porting this lane".
    """
    return [p for p in _candidate_docs() if os.path.isfile(p)]


# The files named ONE BY ONE rather than globbed. A glob cannot tell "this repo has no
# CLAUDE.md" from "CLAUDE.md was renamed and nobody noticed", and the second is a silent
# coverage loss of exactly the kind this file exists to make impossible.
NAMED_DOCS = ("README.md", "AGENTS.md", "CLAUDE.md")


def _candidate_docs():
    out = []
    # ⚠ BOTH PROSE DIRECTORIES. `docs/` was split out of `notes/` on 2026-08-17 (M0) to
    # separate policy and canon from what is in flight — and a gate that kept scanning only
    # `notes/` would have silently stopped checking the SEVEN most authoritative files in the
    # repo at the moment they became authoritative. That is the `.gitignore` failure of the
    # same week with a different mechanism: the check does not break, it just stops matching.
    for d in (NOTES, DOCS):
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".md"):
                out.append(os.path.join(d, name))
    out += [os.path.join(REPO, n) for n in NAMED_DOCS]
    if os.path.isdir(WORKFLOW):
        out += [os.path.join(WORKFLOW, n) for n in sorted(os.listdir(WORKFLOW))
                if n.endswith(".md")]
        out.append(CONFIG_ENV)
        out.append(os.path.join(WORKFLOW, "scripts", "issue.py"))
    cfg = os.path.join(REPO, "configs", "data")
    if os.path.isdir(cfg):
        out += [os.path.join(cfg, n) for n in sorted(os.listdir(cfg))
                if n.endswith((".yaml", ".yml"))]
    return out


def absent_docs():
    """Named documents that are not on disk. PRINTED by `main()`, never silently dropped.

    ⚠ `docs()` has to filter — a checkout without one of these must not crash the gate — and a
    filter is how a scanned file quietly stops being scanned. This is the other half: the
    filter is allowed, the silence is not.
    """
    return [p for p in _candidate_docs() if not os.path.isfile(p)]


def scan(facts, files):
    """Run one registry over one file set. -> (failures, skipped, checked, matched, exempted)

    ⚠ SHARED BY BOTH REGISTRIES ON PURPOSE. The scope-then-exempt-then-capture order below is
    not arrangement, it is the accumulated fix list of this file's own bugs — scope first
    (20 false failures), exemptions counted rather than silent, `group(1)` and nothing else.
    A second copy of it for `PROTOCOL_FACTS` would be a fork of the very logic that catches
    forks, and it would drift in the direction nobody looks at.
    """
    failures, skipped, checked, matched, exempted = [], [], 0, 0, 0
    for fact in facts:
        # PER FACT, not per run. The prerequisite used to live in the pytest harness as one
        # all-or-nothing list, so a host with v5 and not v6 — a partial mount, a rollback,
        # ai-lab-0 itself between the two merges — checked NOTHING and called it a skip,
        # dropping ten enforceable v5 facts to protect two unenforceable v6 ones (#52).
        # Coverage was the intersection of every corpus the registry had ever read, and it
        # shrinks with every generation.
        absent = unreadable(fact)
        if absent:
            skipped.append((fact["name"], absent))
            continue
        try:
            truth = fact["truth"]()
        except (OSError, KeyError, ValueError) as e:
            # ValueError joins the pair for the protocol registry: `setting()` ends in
            # `int(v)`, so a constant edited to a non-number is a source this fact cannot
            # read — the same condition as a missing key, and it must report as one rather
            # than crash the run for every other fact behind it.
            failures.append(f"{fact['name']}: cannot read the artifact ({e}). "
                            "A fact whose source is missing is not a passing fact.")
            continue
        want = spellings(truth, fact.get("words", False))
        checked += 1
        for path in files:
            rel = os.path.relpath(path, REPO)
            scope = re.compile(fact["scope"])
            with open(path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    # SCOPE FIRST. A number is only read on a line that is demonstrably
                    # about this corpus; otherwise "110 speakers" in a VCTK row is compared
                    # against v5's 2,500. That produced 20 false failures on the first run.
                    if not scope.search(line):
                        continue
                    if any(x in line for x in fact["exempt"]):
                        exempted += 1
                        continue
                    for pat in fact["patterns"]:
                        for m in re.finditer(pat, line):
                            matched += 1
                            got = m.group(1)
                            if got not in want:
                                failures.append(
                                    f"{rel}:{lineno} — {fact['name']}: doc says {got!r}, "
                                    f"artifact says {comma(truth)!r}\n"
                                    f"      {line.strip()[:110]}")
    return failures, skipped, checked, matched, exempted


def main():
    files = docs()
    failures, skipped, checked, matched, exempted = scan(FACTS, files)
    p_fail, p_skip, p_checked, p_matched, p_exempted = scan(PROTOCOL_FACTS, files)
    failures += p_fail
    skipped += p_skip

    print(f"checked {checked} of {len(FACTS)} corpus/checkpoint facts against the "
          f"artifacts on disk")
    print(f"  {len(files)} documents scanned, {matched} recognised statements, "
          f"{exempted} exempted line(s)")
    # REPORTED SEPARATELY, because they are checked against different sources and a reader
    # has to be able to tell which half a green run is talking about. Folding the counts
    # together would let the corpus half go to zero coverage behind a healthy-looking total.
    print(f"checked {p_checked} of {len(PROTOCOL_FACTS)} protocol constants against the "
          f"files that own them")
    print(f"  {p_matched} recognised statements, {p_exempted} exempted line(s)")
    for path in absent_docs():
        print(f"  ~ NOT SCANNED: {os.path.relpath(path, REPO)} is not in this checkout")
    # PRINTED, NEVER SILENT. "The corpus is not on this machine" is not a finding, but it
    # is not a pass either, and a reduction in coverage that nobody can see is how a gate
    # stops being one.
    for name, absent in skipped:
        print(f"  ~ {name} NOT CHECKED: {os.path.relpath(absent[0], REPO)} is not on "
              f"this machine")

    # ⚠ Checked only where it CAN be: the checkpoint lives on the /data mount, which exists
    # on ai-lab-0 and nowhere else. "The mount is absent" is not a finding — but it must not
    # read as a pass either, so the skip is printed rather than silent. Where the mount IS
    # present, a missing checkpoint is a real failure: the docs name it as v6's warm start.
    if not os.path.isdir("/data/model-training"):
        print("  ~ selected checkpoints NOT CHECKED: /data is not mounted here")
    else:
        picked = selected_checkpoints()
        if not picked:
            print("  ~ no concluded run carries a SELECTED.md")
        for exp, name, verdict in picked:
            hits = glob.glob(os.path.join(TRAIN_LOGS, exp, "runs", "*", "checkpoints", name))
            if hits:
                print(f"  {exp} selected {name} — present")
            else:
                failures.append(
                    f"{verdict} selects {name}, which is not on disk under "
                    f"{TRAIN_LOGS}/{exp}/runs/*/checkpoints/ — the docs name a warm start "
                    f"that does not exist")

    if failures:
        print(f"\nFAIL — {len(failures)} claim(s) disagree with the artifact:\n")
        for f in failures:
            print(f"  {f}")
        print("\nThe artifact is right and the document is wrong, unless the artifact was "
              "rebuilt —\nin which case update the registry in this file IN THE SAME COMMIT.")
        return 1

    print("\nPASS — every recognised claim matches disk.")
    print("⚠ This does NOT mean the docs are correct: only recognised phrasings are seen, "
          "and\n  scope is corpus/checkpoint numbers (owner, 2026-08-09) plus the workflow "
          "lane's own\n  constants (owner, 2026-08-17). A green run means 'nothing recognised "
          "disagrees'.")
    if skipped:
        print(f"⚠ …and {len(skipped)} of {len(FACTS) + len(PROTOCOL_FACTS)} facts were NOT "
              f"CHECKED here (listed\n  above): their artifact is absent, so this run says "
              f"nothing about them either way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
