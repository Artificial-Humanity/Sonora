#!/usr/bin/env python3
"""Gate: the notes' checkable numbers must agree with what they describe on disk — corpus
and checkpoint artifacts, and since 2026-08-19 module-level CODE CONSTANTS too.

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
* **Only checkable numbers** — corpus/checkpoint artifacts (owner scope, 2026-08-09) and,
  since 2026-08-19, module-level code constants via `const()`. ⚠ The registry was widened
  and these four scope statements were not (issue #131), so the gate described a narrower
  job than it did while `AGENTS.md` §5b told agents to use the wider one.
  Not statuses, not prose, not
  the sequencing claims that made up most of the review's product findings.
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
# ⚠ THE M0 SPLIT MOVED SEVEN CANON DOCUMENTS OUT OF THIS GATE'S REACH. `docs()`
# enumerated `notes/` alone, so relocating ARCHITECTURE, model-decisions,
# audiobook-corpus-policy, markup-schema-brief, vat-channels,
# direction-interface-brief and tts-engine-onboarding silently un-enforced every
# claim in them — caught only because `scm.VAT_TOL` then matched nothing and the
# liveness test went red. A move is a scope change even when no line is edited.
DOCS = os.path.join(REPO, "docs")
V5 = os.path.join(REPO, "data", "libritts_r_emilia_vat_v5")
V6 = os.path.join(REPO, "data", "libritts_r_emilia_expressive_vat_v6")
V4 = os.path.join(REPO, "data", "libritts_r_vat_v4")
HOLDOUT = os.path.join(REPO, "data", "libritts_r_holdout_devclean")

# The artifacts themselves, named once. Every fact lists the ones it reads, so a machine
# that holds one corpus and not another enforces what it can (see the docstring, #52).
V5_TRAIN = os.path.join(V5, "train_op.txt")
V5_VAL = os.path.join(V5, "val_op.txt")
V5_REPORT = os.path.join(V5, "derivation_report.json")
V6_REPORT = os.path.join(V6, "derivation_report.json")
# v7 (Phase 1 rung 3). Registered because the rung-3 range shipped the densest block of
# corpus numbers in the repo into a generation the gate had no fact for — it printed
# "checked 16 of 16 ... PASS" while enforcing nothing about any of them (#370).
V7 = os.path.join(REPO, "data", "libritts_r_full_vat_v7")
V7_TRAIN = os.path.join(V7, "train_op.txt")
V7_VAL = os.path.join(V7, "val_op.txt")
V7_SPEAKERS = os.path.join(V7, "speakers.json")
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


def const(module_rel, name):
    """A module-level constant, read by AST from a SOURCE FILE rather than a data artifact.

    ⚠ THE REGISTRY WAS CORPUS-ONLY AND THAT WAS THE GAP (2026-08-19). Every fact here read a
    number off `/data`, so a number that lives in CODE and is restated in prose had no
    mechanism at all — and one had drifted: `docs/markup-schema-brief.md` gave the SCM
    verifier tolerance as `±0.25` in its field-semantics table while §5 item 6 of the SAME
    FILE recorded the owner's same-day amendment to `±0.35`, which is what `scm.VAT_TOL`
    implements. The document contradicted itself for a month and nothing could see it,
    because the gate only ever compared docs to corpora.

    AST rather than import: importing `scm` pulls in `schemas` and the sibling-path setup,
    and a gate that needs the package layout to be correct in order to check a docstring is
    a gate with a second failure mode.

    ⚠ `module_rel` IS REPO-RELATIVE, AND WAS `scripts/lib`-RELATIVE UNTIL 2026-08-21 (#189).
    The hardcoded directory meant this could only ever reach one bucket, so the constants
    worth registering most — `SPEECH_MIN_SECONDS` in `scripts/stages/`, the direction dial in
    `matcha/` — were unreachable, and the feature was wired for the single fact that motivated
    it. Widening a path parameter is exactly the edit that can make a gate pass VACUOUSLY, so
    two behaviours are asserted rather than assumed: a missing FILE and a missing NAME both
    raise, and `main()` turns either into a failure ("a fact whose source is missing is not a
    passing fact"). A registry entry that silently read the wrong file would be worse than no
    entry, because it would report green.
    """
    import ast

    path = os.path.join(REPO, module_rel)
    if not os.path.isfile(path):
        # KeyError, not FileNotFoundError: `main()` catches (OSError, KeyError) and both are
        # already handled — but naming the repo-relative path is what tells the reader the
        # argument changed meaning, rather than that the file was deleted.
        raise KeyError(f"{module_rel} is not a file in this repo (const() takes a "
                       f"REPO-relative path since 2026-08-21, not a scripts/lib one)")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} is not a module-level constant of {module_rel}")


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
        # ⚠ THE FIRST FACT HERE THAT IS NOT A CORPUS NUMBER. Its artifact is a source file,
        # which is the point: a constant restated in prose drifts exactly like a row count,
        # and until 2026-08-19 nothing checked that class. See `const` above for the drift
        # that prompted it.
        "name": "SCM verifier tolerance (scm.VAT_TOL)",
        "truth": lambda: const("scripts/lib/scm.py", "VAT_TOL"),
        "artifacts": [os.path.join(REPO, "scripts", "lib", "scm.py")],
        "scope": r"VAT_TOL|verifier tolerance|within tolerance|vat\.[VAT] claimed",
        "patterns": [r"±\s*([\d.]+)", r"tolerance = ±?([\d.]+)"],
        # The amendment RECORD legitimately names the superseded value; that is history,
        # not a claim about today. Narrow on purpose — a bare "0.25" exemption would blind
        # the fact to the very table cell that was wrong.
        "exempt": ["table above said `±0.25`", "(originally ±0.25)"],
    },
    {
        # ⚠ TWO DIFFERENT GATES SIT AT THE SAME VALUE, AND THEY ARE REGISTERED SEPARATELY.
        # `qc_gate.SPEECH_MIN_SECONDS` is the QC hard gate on MEASURED VAD speech;
        # `book_ingest.MIN_CLIP_SECONDS` (below) gates ESTIMATED speech in the INPUT TEXT
        # before anything is rendered. Both read 4.0 today, which is exactly why one fact
        # scoped to catch every "4 s" sentence would look right: it would compare each
        # document to whichever constant it happened to be built from, and go green by
        # coincidence. The day one moves, the coincidence becomes a false failure against the
        # other. So each fact is scoped by ITS OWN constant name.
        "name": "QC speech floor (qc_gate.SPEECH_MIN_SECONDS)",
        "truth": lambda: const("scripts/stages/qc_gate.py", "SPEECH_MIN_SECONDS"),
        "artifacts": [os.path.join(REPO, "scripts", "stages", "qc_gate.py")],
        # ⚠⚠ BOTH ALTERNATIVES ARE LOAD-BEARING — `floor via Silero` IS NOT REDUNDANT.
        # `delivery-mix-campaign.md:304` names NO constant and is pinned only through it.
        # Measured (#258): deleting it drops this fact from 4 enrolled sites to 3, silently.
        # A comment here previously said "every line is already scope-gated on the constant's
        # name", which invited exactly that deletion as dead code. `test_the_silero_scope_
        # alternative_is_load_bearing` in tests/test_doc_claims_registry.py now fails if it
        # goes — a claim about what is load-bearing needs a guard, not a sentence.
        "scope": r"SPEECH_MIN_SECONDS|floor via Silero",
        # ⚠ THE FLOOR PATTERN ALLOWS UP TO TWO WORDS BEFORE "floor", AND THAT IS NOT COSMETIC.
        # `([\d.]+)\s*s floor` required the words to be adjacent, so "4 s **speech** floor" and
        # "4 s **owner** floor" matched NOTHING — the two sites #256 identified. Adding the
        # constant name to those lines put them in SCOPE and left them unenforced, because
        # scope selects the line and a pattern is still what reads the number. Caught by
        # mutating the constant and seeing only 2 of 4 sites go red; a scope-only fix would
        # have shipped looking done. Bounded at two words, and every line is already gated by
        # this fact's `scope` above, so the widening cannot reach unrelated prose.
        "patterns": [r"Minimum\s+([\d.]+)\s*s\b", r"([\d.]+)\s*s(?:\s+[a-z]+){0,2}\s+floor"],
        "exempt": [],
    },
    {
        "name": "Teacher-bank text floor (book_ingest.MIN_CLIP_SECONDS)",
        "truth": lambda: const("scripts/lib/book_ingest.py", "MIN_CLIP_SECONDS"),
        "artifacts": [os.path.join(REPO, "scripts", "lib", "book_ingest.py")],
        "scope": r"MIN_CLIP_SECONDS",
        "patterns": [r"Minimum clip length is ([\d.]+)\s*s\b"],
        "exempt": [],
    },
    {
        "name": "v5 TRAIN rows",
        "truth": lambda: rows(V5_TRAIN),
        "artifacts": [V5_TRAIN],
        "scope": r"v5|libritts_r_emilia_vat_v5",
        "patterns": [r"([\d,]{6,}) train / [\d,]+ val", r"([\d,]{6,}) train rows",
                     r"([\d,]{6,}) train \+ [\d,]+ val\)"],
        # ⚠ A COMPARISON SENTENCE NAMES BOTH CORPORA ON ONE LINE, and scope matches the LINE
        # while the pattern takes the FIRST number on it. `vat6_finetune.yaml:10` reads
        # "41,937 train rows against v5's 41,138" — correct prose, in which the v5 figure is
        # the second number. Surfaced the moment `configs/experiment/` entered `docs()`
        # (#370), as a RED GATE ON A CORRECT SENTENCE, which is the failure mode this file
        # fears most: the cheap repair is to loosen the pattern, and a loosened pattern stops
        # catching the drift the fact exists for. Exempt the sentence instead — narrowly, on
        # a phrase that only a v6-vs-v5 comparison contains.
        "exempt": ["train rows against v5's"],
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
        #     `(?<![\w,])` stops the capture starting INSIDE a longer number, which is
        #     how a lookbehind alone gets fooled: without it, "of the 13,141" merely
        #     shifts the match one digit right and reports "3,141".
        #
        #     ⚠ THE CLASS IS `\w`, NOT `\d`, AND THAT IS THE WHOLE POINT (2026-09-02).
        #     It was `[\d,]` until a CORRECT sentence in notes/training-sources.md went
        #     red: "`emilia_kept_24k` is what v5–v7 train on". `[\d,]` only refuses a
        #     digit before the capture, so the `7` of `v7` started a match, `train` was
        #     read as the noun, and the gate reported the v7 train filelist held 7 rows
        #     against an artifact holding 336,546. A COUNT IS A STANDALONE TOKEN; a
        #     version tag is not. `\w` says exactly that and costs nothing — every real
        #     count in the tree is preceded by a space, a `*` or a `|`. The narrow class
        #     was wrong for `vat7` too (`t` is not a digit either), so the bug was one
        #     wording away from hitting every fact here, not just this one. Fixed on all
        #     twelve patterns at once rather than the one line that happened to fail.
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
        "patterns": [r"(?<!of the )(?<![\w,])([\d,]+) (?:rows )?dropped on digits",
                     r"(?<![\w,])([\d,]+) digit rows",
                     r"(?<![\w,])([\d,]+) carry digits",
                     r"(?<![\w,])([\d,]+) of the 13,141 (?:rows |keeps )?dropped on digits"],
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
        "patterns": [r"(?<![\w,])([\d,]+) of the [\d,]+ (?:rows )?dropped on digits",
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
                     r"(?<![\w,])([\d,]+) rows to append",
                     r"(?<![\w,])(\d[\d,]*) rows staged"],
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
        "patterns": [r"(?<![\w,])(\d[\d,]*) appended rows",
                     r"(?<![\w,])(\d[\d,]*) rows appended"],
        "exempt": ["846 appended"],
    },
    {
        # ⚠ SCOPE IS `v7|vat7` AND IT MATCHES PER LINE, so a number is only read where the
        # line itself names the corpus. That is why the headline rows in both v7 configs
        # carry a `v7:` marker: without it the most load-bearing line in each file is out of
        # scope and enforced by nothing. Same fix the v6 work applied to notes/STATE.md.
        "name": "v7 train rows",
        "truth": lambda: rows(V7_TRAIN),
        "artifacts": [V7_TRAIN],
        "scope": r"v7|vat7",
        "patterns": [r"(?<![\w,])(\d[\d,]*) (?:train|tr)\b"],
        "exempt": [],
    },
    {
        "name": "v7 val rows",
        "truth": lambda: rows(V7_VAL),
        "artifacts": [V7_VAL],
        "scope": r"v7|vat7",
        "patterns": [r"(?<![\w,])(\d[\d,]*) val\b"],
        "exempt": [],
    },
    {
        "name": "v7 n_spks",
        "truth": lambda: json.loads(open(V7_SPEAKERS, encoding="utf-8").read())["n_spks"],
        "artifacts": [V7_SPEAKERS],
        "scope": r"v7|vat7",
        "patterns": [r"(?<![\w,])(\d[\d,]*) speakers\b"],
        "exempt": [],
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
#
# ⚠ A PICK HAS TWO LEGAL STATES, AND "PRESENT" IS ONLY ONE OF THEM (2026-09-02).
# The rule above read "the file the verdict names must exist, because 'warm start from
# epNNN' is worthless if it does not". That rationale is FALSE, and v5 is the counter-
# example that proved it. `vat5_finetune`'s pick `ep019` was deleted on 2026-08-29 while
# reclaiming disk, and the warm start it seeded was NOT worthless: v6 trained from it and
# `warmstart/vat6_init.ckpt` still holds those weights. Measured rather than assumed —
# `vat6_init` was compared tensor-by-tensor against `ep009`, the only vat5 checkpoint left
# on disk, and 335 of 337 shared same-shape tensors DIFFER. So the donor was one of the
# deleted checkpoints and the document is right about the lineage.
#
# The convention that makes this normal is `scripts/lib/make_warmstart.py`: a pick becomes
# a donor by being STRIPPED to a ~90 MB weights-only file in `warmstart/`, after which the
# 273 MB original with its optimizer state is dead weight. Demanding the fat file forever
# means the gate goes red for doing the routine thing.
#
# So a reclaimed pick is accepted — but only when the verdict DECLARES it, in two lines the
# gate can read:
#
#     RECLAIMED: checkpoint_epoch=019.ckpt
#     LINEAGE: /data/model-training/sonora/warmstart/vat6_init.ckpt
#
# and the named lineage file is on disk. Absence alone is never enough. An accidental
# deletion looks exactly like a reclaimed one ON DISK and differs only in whether anybody
# wrote it down, so the declaration IS the check — without it this degrades to "the file is
# missing, therefore fine", which is the vacuous pass this whole gate exists to refuse.
#
# ⚠ WHAT THIS DOES NOT VERIFY. That the lineage file DESCENDS from the pick. `make_warmstart`
# records no donor in its output, so nothing on disk ties `vat6_init.ckpt` to `ep019` — the
# tensor comparison above can only prove which checkpoints it is NOT. The `LINEAGE:` line is
# trusted prose pointing at a verified-present file. Closing that gap means stamping the
# donor path and hash into the stripped checkpoint at strip time; until then, do not read a
# pass here as a proof of ancestry.
TRAIN_LOGS = "/data/model-training/sonora/logs/train"
CKPT_IN_VERDICT = re.compile(r"checkpoint_epoch=\d+\.ckpt")
RECLAIMED_IN_VERDICT = re.compile(r"^RECLAIMED:\s*(checkpoint_epoch=\d+\.ckpt)\s*$", re.M)
LINEAGE_IN_VERDICT = re.compile(r"^LINEAGE:\s*(\S+)\s*$", re.M)


def selected_checkpoints():
    """-> [(experiment, checkpoint, verdict path, reclaimed, lineage)] per concluded run.

    `reclaimed` is the checkpoint the verdict declares reclaimed (or None), and `lineage`
    the file it says the weights survive in (or None). Both are read here rather than at
    the call site so the whole parse is exercised by one probe.
    """
    out = []
    if not os.path.isdir(TRAIN_LOGS):
        return out
    for exp in sorted(os.listdir(TRAIN_LOGS)):
        verdict = os.path.join(TRAIN_LOGS, exp, "SELECTED.md")
        if not os.path.isfile(verdict):
            continue
        with open(verdict, encoding="utf-8") as fh:
            body = fh.read()
        rec = RECLAIMED_IN_VERDICT.search(body)
        lin = LINEAGE_IN_VERDICT.search(body)
        # The heading names the pick; later mentions may reference others (ep039 as the
        # newest, ep009 as the runner-up), so the FIRST is the one the verdict selects.
        #
        # ⚠ THE `RECLAIMED:` LINE IS REMOVED BEFORE THE PICK IS READ, and that is not
        # tidiness. It names a checkpoint, so wherever it sits above the pick's own first
        # mention it BECOMES `names[0]` — and then "does the reclaim match the pick?" is
        # comparing the line to itself and can never fail. A declaration that authenticates
        # itself is not a check. Reading the pick from the body WITHOUT the declaration
        # keeps the two independent, so the mismatch probe has something real to catch.
        names = CKPT_IN_VERDICT.findall(RECLAIMED_IN_VERDICT.sub("", body))
        if names:
            out.append((exp, names[0], verdict,
                        rec.group(1) if rec else None,
                        lin.group(1) if lin else None))
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
    """
    out = []
    for base in (NOTES, DOCS):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if name.endswith(".md"):
                out.append(os.path.join(base, name))
    out.append(os.path.join(REPO, "README.md"))
    # ⚠ BOTH config groups, not just `data` (#370). `configs/experiment/*.yaml` restates the
    # corpus split, both speaker counts, the hours delta and the data_statistics pair — the
    # vat7 experiment config alone carries five enforceable numbers — and it was outside this
    # set entirely, so registering a v7 FACT still reached none of them. Measured at the time:
    # corrupting `n_spks` in the experiment config left the gate green while the same
    # corruption in the data config failed it. Adding a fact and adding scope are two separate
    # acts and this file has now been bitten by each of them separately.
    for group in ("data", "experiment"):
        cfg = os.path.join(REPO, "configs", group)
        if os.path.isdir(cfg):
            out += [os.path.join(cfg, n) for n in sorted(os.listdir(cfg))
                    if n.endswith((".yaml", ".yml"))]
    return out


def main():
    failures, skipped, checked, matched, exempted = [], [], 0, 0, 0
    files = docs()

    for fact in FACTS:
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
        except (OSError, KeyError) as e:
            failures.append(f"{fact['name']}: cannot read the artifact ({e}). "
                            "A fact whose source is missing is not a passing fact.")
            continue
        want = {comma(truth), str(truth)}
        # ⚠ A FLOAT CONSTANT AND ITS PROSE SPELLING DIFFER, and the mismatch is silent until
        # a fact is registered for one. `SPEECH_MIN_SECONDS = 4.0` is written "4 s" in every
        # note, while `str(4.0)` is "4.0" — so the fact would have called every CORRECT
        # sentence a drift. An integral float admits both spellings; 0.35 is untouched.
        if isinstance(truth, float) and truth.is_integer():
            want |= {comma(int(truth)), str(int(truth))}
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

    print(f"checked {checked} of {len(FACTS)} facts against the "
          f"artifacts on disk")
    print(f"  {len(files)} documents scanned, {matched} recognised statements, "
          f"{exempted} exempted line(s)")
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
        for exp, name, verdict, reclaimed, lineage in picked:
            hits = glob.glob(os.path.join(TRAIN_LOGS, exp, "runs", "*", "checkpoints", name))
            if hits:
                print(f"  {exp} selected {name} — present")
            elif reclaimed is None:
                failures.append(
                    f"{verdict} selects {name}, which is not on disk under "
                    f"{TRAIN_LOGS}/{exp}/runs/*/checkpoints/ — the docs name a warm start "
                    f"that does not exist. If it was reclaimed on purpose, say so in the "
                    f"verdict with a 'RECLAIMED: {name}' line and a 'LINEAGE: <path>' line "
                    f"naming the stripped donor that carries its weights")
            elif reclaimed != name:
                failures.append(
                    f"{verdict} selects {name} but declares 'RECLAIMED: {reclaimed}' — the "
                    f"verdict reclaims a checkpoint it did not pick, so the pick is still "
                    f"unaccounted for")
            elif not lineage:
                failures.append(
                    f"{verdict} declares {name} reclaimed but carries no 'LINEAGE: <path>' "
                    f"line — a reclaimed pick must name the file its weights survive in, or "
                    f"the deletion is indistinguishable from losing the lineage")
            elif not os.path.exists(lineage):
                failures.append(
                    f"{verdict} declares {name} reclaimed into {lineage}, which is not on "
                    f"disk either — the pick and its stated lineage are both gone")
            else:
                print(f"  {exp} selected {name} — reclaimed, lineage in {lineage}")

    if failures:
        print(f"\nFAIL — {len(failures)} claim(s) disagree with the artifact:\n")
        for f in failures:
            print(f"  {f}")
        print("\nThe artifact is right and the document is wrong, unless the artifact was "
              "rebuilt —\nin which case update the registry in this file IN THE SAME COMMIT.")
        return 1

    print("\nPASS — every recognised claim matches disk.")
    print("⚠ This does NOT mean the docs are correct: only recognised phrasings are seen, "
          "and\n  scope is corpus/checkpoint numbers and registered code constants (owner,\n"
          "  2026-08-09; widened 2026-08-19). A green run "
          "means\n  'nothing recognised disagrees'.")
    if skipped:
        print(f"⚠ …and {len(skipped)} of {len(FACTS)} facts were NOT CHECKED here (listed "
              f"above): their\n  artifact is absent, so this run says nothing about them "
              f"either way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
