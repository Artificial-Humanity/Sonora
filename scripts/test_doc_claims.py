#!/usr/bin/env python3
"""Gate: the notes' corpus/checkpoint numbers must agree with the artifacts on disk.

WHY THIS EXISTS (2026-08-09, owner-commissioned)
------------------------------------------------
This repo catches forked CODE constants and nothing else. `test_skill_files.py` verifies 92
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
* **Only corpus/checkpoint numbers** (owner scope, 2026-08-09). Not statuses, not prose, not
  the sequencing claims that made up most of the review's product findings.
* **Only phrasings the patterns match.** A number written in a form no pattern recognises is
  invisible here — a silent miss, not an error. When a check goes green it means "no
  RECOGNISED statement disagrees", never "the docs are correct".
* **Historical mentions are exempted by explicit substring**, listed per fact and printed on
  every run, so an exemption cannot quietly become a hiding place.

Run:  .venv/bin/python scripts/test_doc_claims.py
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
REPO = os.path.dirname(HERE)
NOTES = os.path.join(REPO, "notes")
V5 = os.path.join(REPO, "data", "libritts_r_emilia_vat_v5")
V6 = os.path.join(REPO, "data", "libritts_r_emilia_expressive_vat_v6")
V4 = os.path.join(REPO, "data", "libritts_r_vat_v4")
HOLDOUT = os.path.join(REPO, "data", "libritts_r_holdout_devclean")


def rows(path):
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def report(key, path=os.path.join(V5, "derivation_report.json")):
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
    return report(key, os.path.join(V6, "derivation_report.json"))


def comma(n):
    return f"{n:,}"


# --- the registry -------------------------------------------------------------------
#
# Each fact: how to READ it from disk, and the phrasings that STATE it in prose. Patterns
# carry exactly one capture group — the number as written. Keep patterns narrow: a loose
# one produces false failures, which is how a gate gets switched off.
#
# `exempt` lines are skipped and announced. They exist for deliberate historical mentions
# ("this cell read X until <date>"), which the repo writes on purpose.
FACTS = [
    {
        "name": "v5 TRAIN rows",
        "truth": lambda: rows(os.path.join(V5, "train_op.txt")),
        "scope": r"v5|libritts_r_emilia_vat_v5",
        "patterns": [r"([\d,]{6,}) train / [\d,]+ val", r"([\d,]{6,}) train rows",
                     r"([\d,]{6,}) train \+ [\d,]+ val\)"],
        "exempt": [],
    },
    {
        "name": "v5 VAL rows",
        "truth": lambda: rows(os.path.join(V5, "val_op.txt")),
        "scope": r"v5|libritts_r_emilia_vat_v5",
        "patterns": [r"[\d,]{6,} train / ([\d,]+) val", r"[\d,]{6,} train \+ ([\d,]+) val\)"],
        "exempt": [],
    },
    {
        "name": "v5 TOTAL rows",
        "truth": lambda: (rows(os.path.join(V5, "train_op.txt"))
                          + rows(os.path.join(V5, "val_op.txt"))),
        "scope": r"v5|libritts_r_emilia_vat_v5",
        "patterns": [r"\*\*([\d,]{6,})\*\* \| \*\*\+[\d,]+ Emilia"],
        "exempt": [],
    },
    {
        "name": "v5 speakers",
        "truth": lambda: report("n_spks"),
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
        "scope": r"Emilia|emilia",
        "patterns": [r"\+?([\d,]+) Emilia-YODAS keeps", r"([\d,]+) keeps of `?libritts"],
        "exempt": ["until 2026-08-09", "10,653 and 10,997"],
    },
    {
        "name": "Emilia TRAIN rows",
        "truth": lambda: report("emilia.train"),
        "scope": r"Emilia|emilia",
        "patterns": [r"\*\*([\d,]+) train \+ [\d,]+ val\*\*"],
        "exempt": [],
    },
    {
        "name": "Emilia candidates before filtering",
        "truth": lambda: report("emilia.candidates"),
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
        #     refuses the "N of the M" idiom outright (that is v6's, below), and
        #     `(?<![\d,])` stops the capture starting INSIDE a longer number, which is
        #     how a lookbehind alone gets fooled: without it, "of the 13,141" merely
        #     shifts the match one digit right and reports "3,141".
        "name": "Emilia rows dropped on digits (v5 merge)",
        "truth": lambda: report("emilia.dropped.digits"),
        "scope": r"Emilia|emilia|13,141|digit rows|D-M3",
        "patterns": [r"(?<!of the )(?<![\d,])([\d,]+) (?:rows )?dropped on digits",
                     r"(?<![\d,])([\d,]+) digit rows",
                     r"(?<![\d,])([\d,]+) carry digits"],
        "exempt": [],
    },
    {
        # The v6 append's own digit drop — same rule (D-M3), different corpus, and the
        # documents are right to state it separately. Scope carries v6's markers and
        # deliberately NOT `Emilia`; the patterns match only the two idioms v6 is
        # actually written in, and both anchor the capture on the count being claimed
        # rather than on the staged total it is quoted out of.
        #
        # `D-M3` earns its place: `notes/STATE.md:313` states this fact on a line that
        # names no corpus at all, so the finding id is the only value-independent handle
        # on it. It pulls a few v5 lines into scope, and that is harmless BY
        # CONSTRUCTION rather than by luck — none of them is written in the "N of the M"
        # or parenthetical-breakdown form these patterns require.
        "name": "v6 append rows dropped on digits",
        "truth": lambda: report6("expressive.dropped.digits"),
        "scope": r"v6|expressive|append|D-M3",
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
        "scope": r"v6|expressive|append|D-M3|clips resolve",
        "patterns": [r"[\d,]+ of the ([\d,]+) (?:rows )?dropped on digits",
                     r"0 of the ([\d,]+) clips resolve",
                     r"append(?: set)? is settled at \*\*([\d,]+) rows\*\*",
                     r"(?<![\d,])([\d,]+) rows to append"],
        "exempt": [],
    },
    {
        "name": "v4 TOTAL rows (the base v5 merges into)",
        "truth": lambda: (rows(os.path.join(V4, "train_op.txt"))
                          + rows(os.path.join(V4, "val_op.txt"))),
        "scope": r"v4|libritts_r_vat_v4|31,44",
        "patterns": [r"v4 rows verbatim.{0,40}?([\d,]{6,})", r"([\d,]{6,}) clips `vat3c"],
        "exempt": [],
    },
    {
        "name": "holdout clips",
        "truth": lambda: rows(os.path.join(HOLDOUT, "holdout_8w.txt")),
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


def docs():
    out = []
    for name in sorted(os.listdir(NOTES)):
        if name.endswith(".md"):
            out.append(os.path.join(NOTES, name))
    out.append(os.path.join(REPO, "README.md"))
    return out


def main():
    failures, checked, matched, exempted = [], 0, 0, 0
    files = docs()

    for fact in FACTS:
        try:
            truth = fact["truth"]()
        except (OSError, KeyError) as e:
            failures.append(f"{fact['name']}: cannot read the artifact ({e}). "
                            "A fact whose source is missing is not a passing fact.")
            continue
        want = {comma(truth), str(truth)}
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

    print(f"checked {checked} corpus/checkpoint facts against the artifacts on disk")
    print(f"  {len(files)} documents scanned, {matched} recognised statements, "
          f"{exempted} exempted line(s)")

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
          "and\n  scope is corpus/checkpoint numbers only (owner, 2026-08-09). A green run "
          "means\n  'nothing recognised disagrees'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
