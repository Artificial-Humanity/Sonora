"""The doc-claims REGISTRY, exercised as regex — no corpus, no `/data`, no venv.

WHY THIS FILE EXISTS (2026-08-11, issue #49)
--------------------------------------------
`scripts/gates/test_doc_claims.py` is two things in one file: a set of artifact reads, and a body
of subtle regex — lookbehinds, scope alternations, and an explicit rule ("never scope a fact
on the value it checks") that was enforced by nothing. The artifact half is rightly
data-gated and skips on every machine except ai-lab-0. The regex half needs no artifacts at
all and was guarded **nowhere**, so `pytest -q` was green on every laptop while the part
most likely to rot was never executed. Nothing even imported `FACTS`.

The cost is documented in the gate's own comments. Its first draft scoped the v6 entries on
the literal `832`, so corrupting `832` removed the line from its own scope and **the gate
went green** — a hole found only by running the RED direction by hand, with nothing
committed that re-runs it. Every probe below is that hand-run made durable.

**PROVING A GATE PASSES PROVES NOTHING; PROVE IT FAILS.** Most of what follows asserts that
a corrupted number is still SEEN and still CAPTURED, because a check that stops looking at
the moment its subject goes wrong is worse than no check: it reads as a pass.

The lines are synthetic on purpose — they include wordings the tree does not use today, so
the registry is tested against the phrasings it will meet tomorrow rather than only the ones
it was fitted to.
"""

import importlib.util
import json
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(REPO, "scripts", "gates", "test_doc_claims.py")


def _load_gate():
    """Import the gate as a module. It is a script, not a package member.

    Loaded under a name of its own — NOT `test_doc_claims` — because pytest would otherwise
    try to collect the gate script itself as a test module and run its module-level code
    under a second identity.
    """
    spec = importlib.util.spec_from_file_location("sonora_doc_claims_gate", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load_gate()

# BOTH REGISTRIES. The generic invariants below — one capture group, no scope anchored on the
# value under test, every fact reads something live — are properties of the SCANNER, not of the
# corpus, so a second registry that skipped them would be the untested half all over again.
ALL_FACTS = gate.FACTS + gate.PROTOCOL_FACTS
BY_NAME = {f["name"]: f for f in ALL_FACTS}

PASS_CAP = "review pass cap (MAX_PASSES)"
COMMENT_CAP = "comment length cap (COMMENT_MAX)"

V5_DIGITS = "Emilia rows dropped on digits (v5 merge)"
V6_DIGITS = "v6 append rows dropped on digits"
V6_STAGED = "v6 append rows staged (candidates)"
V6_KEPT = "v6 append rows kept (built)"


def in_scope(name, line):
    return bool(re.search(BY_NAME[name]["scope"], line))


def captures(name, line):
    """Exactly what the gate would read off this line for this fact.

    Mirrors `main()`: scope first, then exemptions, then every pattern.
    """
    fact = BY_NAME[name]
    if not in_scope(name, line):
        return []
    if any(x in line for x in fact["exempt"]):
        return []
    return [m.group(1) for pat in fact["patterns"] for m in re.finditer(pat, line)]


# --- the sentences ------------------------------------------------------------------
#
# Real prose where the tree already carries it, and the one-wording-away variant where the
# point is that it must not matter. Each is named for the trap it holds.

# The wording `notes/training-sources.md` is one routine edit from — the two halves of its
# digit sentence regularised into the same idiom. Feeding this to the registry as it stood
# before #48 produced TWO false failures on a CORRECT v5 sentence, because the v6 entries
# were scoped on the bare rule id `D-M3` and this line carries it.
V5_IN_THE_OF_THE_IDIOM = (
    "1,676 of the 13,141 rows dropped on digits (D-M3 — the tokenizer deletes them)")

# What that file says today.
V5_AS_WRITTEN_TODAY = (
    "2,144 of the 13,141 keeps did not make it: **1,676 carry digits** (D-M3 — the tokenizer")

# `notes/STATE.md`, after #48 gave it the corpus marker it was missing.
V6_STATE_LINE = (
    "   **6 of the 832 dropped on digits (D-M3)** in the v6 append — five correctly (the "
    "audio")

# The same sentence WITHOUT a corpus marker, which is how it read before #48. Kept as a
# regression case: the registry deliberately cannot see it, and that is why the fix had to
# be an edit to the document rather than a rule id in the scope.
V6_STATE_LINE_WITHOUT_A_CORPUS_MARKER = (
    "   **6 of the 832 dropped on digits (D-M3)** — five correctly (the audio")

# The root README's shape — the parenthetical breakdown.
V6_README_LINE = (
    "merged into v6 and BUILT (2026-08-10): **826 appended rows**, 816 of them "
    "delivery-labelled, out of 1,004 eligible keeps (158 already in v5, 14 over-length, "
    "6 dropped on digits).")

# `notes/README.md` after #50 — staged and kept in one sentence, which is the sentence the
# two READMEs used to fork on.
V6_STAGED_AND_KEPT = (
    "set is **832 rows staged, 826 appended rows** — 1,004 eligible, less 158 "
    "duplicate-audio,")

# `notes/CHANGELOG.md:155`. `[\d,]+` matches a BARE COMMA, so a pattern one character too
# loose reads this as the claim "doc says ','" about a v6 count.
COMMA_BEFORE_APPENDED_ROWS = (
    "rather than trusted: donor rows byte-identical, appended rows carrying a real (v6)")

# `notes/quality-gap-plan.md:224`. 846 is CORRECT for its own filter stage — after dedup,
# before the duration drop — and is neither the staged 832 nor the built 826.
HISTORICAL_846 = (
    "**v6 scope: 1,004 eligible keeps before dedup → 846 appended (owner 2026-08-09; dedup")


# --- #48: two generations, one rule id ------------------------------------------------

def test_a_v5_digit_sentence_in_the_of_the_idiom_is_read_by_v5_and_by_nobody_else():
    """The reproduction from #48, in the direction that used to go red.

    Rewording a CORRECT v5 sentence into the "N of the M" idiom made the gate fail on two
    v6 facts — the v6 entries claimed it through `D-M3`, and the v5 entry that owns the
    fact could not see it at all because `(?<!of the )` refuses that idiom outright.
    """
    assert captures(V5_DIGITS, V5_IN_THE_OF_THE_IDIOM) == ["1,676"]
    assert captures(V6_DIGITS, V5_IN_THE_OF_THE_IDIOM) == []
    assert captures(V6_STAGED, V5_IN_THE_OF_THE_IDIOM) == []


def test_a_wrong_v5_digit_count_in_that_idiom_is_still_read():
    """The silent-miss half — the worse direction, and the one a green run hides.

    With the idiom refused and nothing accepting it, `1,677 of the 13,141` was invisible to
    the entry that owns the fact. The 13,141 anchor is what makes it visible without letting
    v6 claim it: v6's staged total is 832 and can never satisfy the pattern.
    """
    wrong = V5_IN_THE_OF_THE_IDIOM.replace("1,676", "1,677")
    assert captures(V5_DIGITS, wrong) == ["1,677"]
    assert captures(V6_DIGITS, wrong) == []


def test_the_capture_never_starts_inside_a_longer_number():
    """`(?<![\\d,])`, deleted, shifts the match one digit right and reports `3,141`.

    A lookbehind on the words alone is trivially evaded by the number that follows them.
    """
    assert "3,141" not in captures(V5_DIGITS, V5_IN_THE_OF_THE_IDIOM)
    naive = re.findall(r"(?<!of the )([\d,]+) (?:rows )?dropped on digits",
                       V5_IN_THE_OF_THE_IDIOM)
    assert naive == ["3,141"], (
        "the naive pattern is supposed to be fooled — if it stopped being, this test no "
        "longer demonstrates what the lookbehind is for")


def test_the_v5_sentence_as_written_today_is_still_read():
    assert captures(V5_DIGITS, V5_AS_WRITTEN_TODAY) == ["1,676"]
    assert captures(V6_DIGITS, V5_AS_WRITTEN_TODAY) == []


def test_a_v6_digit_sentence_is_read_by_v6_and_by_nobody_else():
    assert captures(V6_DIGITS, V6_STATE_LINE) == ["6"]
    assert captures(V6_STAGED, V6_STATE_LINE) == ["832"]
    assert captures(V5_DIGITS, V6_STATE_LINE) == [], (
        "the v5 entry must not read v6's count — that collision is what split these facts")


def test_the_rule_id_alone_no_longer_puts_a_line_in_v6_scope():
    """`D-M3` is the RULE, not the corpus. Every v5 statement of the rule carries it."""
    for entry in (V6_DIGITS, V6_STAGED, V6_KEPT):
        assert "D-M3" not in BY_NAME[entry]["scope"], (
            f"{entry} is scoped on the finding id again — it spans both generations, so "
            f"the entry will claim v5 sentences the moment one is written in v6's idiom")


def test_a_line_with_no_corpus_marker_is_seen_by_nothing_and_that_is_the_deal():
    """The cost of dropping `D-M3`, stated so it cannot be discovered by surprise.

    A sentence naming no corpus is unattributable, and the registry refuses to guess. The
    remedy is a marker on the LINE — which `notes/STATE.md` now carries, asserted below
    against the live file.
    """
    assert captures(V6_DIGITS, V6_STATE_LINE_WITHOUT_A_CORPUS_MARKER) == []
    assert captures(V6_STAGED, V6_STATE_LINE_WITHOUT_A_CORPUS_MARKER) == []


def test_the_live_state_md_still_names_the_corpus_on_its_digit_line():
    """Guards the document half of #48's fix, which the registry half depends on."""
    idiom = re.compile(r"(?<![\d,])\d[\d,]* of the [\d,]+ (?:rows )?dropped on digits")
    with open(os.path.join(REPO, "notes", "STATE.md"), encoding="utf-8") as fh:
        stated = [(n, ln) for n, ln in enumerate(fh, 1) if idiom.search(ln)]
    assert stated, "notes/STATE.md no longer states the v6 digit drop in a checked idiom"
    for lineno, line in stated:
        assert captures(V6_DIGITS, line), (
            f"notes/STATE.md:{lineno} states a digit drop that NO fact reads — it has lost "
            f"its corpus marker, and #48's fix with it:\n  {line.strip()}")


def test_no_digit_sentence_is_ever_claimed_by_both_corpora():
    """The #48 invariant, over every live document.

    Scope overlap is fine — `notes/STATE.md`'s line is inside v5's scope through `D-M3` and
    always will be. What must never happen again is two entries CAPTURING from one sentence,
    which is two facts claiming one number, and a red gate on prose that is right.

    There is NO carve-out. `notes/CHANGELOG.md` used to be the one file allowed zero readers
    — append-only history quoting prior wordings verbatim, which a checker must not force
    anyone to edit. **There is no changelog** (AGENTS.md §4), so the exemption has nothing
    left to exempt. Every **digit-drop sentence** in a scanned file must now have a reader.
    ⚠ NOT every digit sentence: this loop only examines lines matching the idiom compiled below
    — 3 lines across 40 scanned files. `notes/STATE.md`'s "2,500 speakers" is a digit sentence in
    a scanned file and nothing here looks at it. Do not read this as a guarantee that a new digit
    claim would be caught; that is what the registry's own per-fact reader test is for.
    ⚠ If an append-only quoting document is ever reintroduced, the carve-out comes back WITH
    it; do not add one speculatively.
    """
    # BOTH idioms, deliberately. Restricted to "N of the M", this test passed while the
    # bare word `digit` was back in the v5 scope — because the collision that mutation
    # re-creates lands on the PARENTHETICAL form, `README.md`'s "…, 6 dropped on digits)",
    # which the narrow regex never looked at. The hole the sweep misses is the hole.
    idiom = re.compile(
        r"(?<![\d,])\d[\d,]*(?: of the [\d,]+)? (?:rows |keeps )?dropped on digits")
    for path in gate.docs():
        rel = os.path.relpath(path, REPO)
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not idiom.search(line):
                    continue
                readers = [n for n in (V5_DIGITS, V6_DIGITS) if captures(n, line)]
                assert len(readers) <= 1, (
                    f"{rel}:{lineno} is claimed by BOTH {readers} — one sentence, two "
                    f"facts, and one of them must be wrong about it:\n  {line.strip()}")
                assert readers, (
                    f"{rel}:{lineno} states a digit drop that NO fact reads — it names no "
                    f"corpus, so nothing checks it:\n  {line.strip()}\n"
                    "  If this file is append-only history that QUOTES prior wordings, the "
                    "answer is a carve-out (see this test's docstring) rather than editing the "
                    "quote — falsifying the record to satisfy a checker is the worse trade.")


# --- #49: a corrupt value must never delete its own check -----------------------------

# (fact, line, the value the fact reads off it). The RED direction: replace the value with
# a wrong one and the fact must still see the line and still report the wrong value.
CORRUPTIONS = [
    (V5_DIGITS, V5_AS_WRITTEN_TODAY, "1,676"),
    (V5_DIGITS, V5_IN_THE_OF_THE_IDIOM, "1,676"),
    (V6_DIGITS, V6_STATE_LINE, "6"),
    (V6_STAGED, V6_STATE_LINE, "832"),
    (V6_DIGITS, V6_README_LINE, "6"),
    (V6_KEPT, V6_README_LINE, "826"),
    (V6_STAGED, V6_STAGED_AND_KEPT, "832"),
    (V6_KEPT, V6_STAGED_AND_KEPT, "826"),
]


@pytest.mark.parametrize("name,line,value", CORRUPTIONS,
                         ids=[f"{n}|{v}" for n, _l, v in CORRUPTIONS])
def test_a_wrong_number_is_still_in_scope_and_still_reported(name, line, value):
    """NEVER SCOPE A FACT ON THE VALUE IT CHECKS.

    This is the hole the gate's author found by hand and nothing re-ran: with `832` in the
    v6 scope, corrupting `832` dropped the line out of scope and the gate went GREEN at the
    one instant the check had a job.
    """
    assert value in captures(name, line), "the table is stale — the fact no longer reads it"
    corrupt = line.replace(value, "9,999")
    assert in_scope(name, corrupt), (
        f"corrupting {value!r} removed the line from {name}'s scope — the scope is anchored "
        f"on the number under test, so a wrong value deletes its own check")
    assert "9,999" in captures(name, corrupt), (
        f"{name} stopped reading the line once the number was wrong")


def test_no_scope_marker_is_a_bare_number_a_fact_could_be_checking():
    """The structural half of the same rule.

    `13,141` is legal and deliberate: it is Emilia's CANDIDATE count on a line whose fact is
    the DIGIT count, two different numbers in one sentence. Anything else numeric is the
    trap — so the whitelist is explicit and adding to it is a decision, not a slip.
    """
    allowed = {"13,141", "31,44"}
    for fact in ALL_FACTS:
        for alt in fact["scope"].split("|"):
            if re.fullmatch(r"[\d,]+", alt):
                assert alt in allowed, (
                    f"{fact['name']} is scoped on the bare number {alt!r}. If that is the "
                    f"value it checks, a wrong value silently removes its own check.")


# --- #50: staged is not kept ----------------------------------------------------------

def test_staged_and_kept_are_separate_facts_on_one_sentence():
    assert captures(V6_STAGED, V6_STAGED_AND_KEPT) == ["832"]
    assert captures(V6_KEPT, V6_STAGED_AND_KEPT) == ["826"]
    assert captures(V6_DIGITS, V6_STAGED_AND_KEPT) == []


def test_the_kept_count_is_read_from_the_parenthetical_readme_idiom():
    assert captures(V6_KEPT, V6_README_LINE) == ["826"]
    assert captures(V6_DIGITS, V6_README_LINE) == ["6"]
    # The original two-generation collision, and the form it actually took: the bare word
    # `digit` in the v5 scope pulls this line in, and `(?<!of the )` does NOT protect the
    # parenthetical idiom the way it protects "N of the M" — so v5 reads v6's 6 as a wrong
    # Emilia count. That is exactly the red-on-correct-prose failure the split was for.
    assert captures(V5_DIGITS, V6_README_LINE) == [], (
        "the v5 entry is reading v6's parenthetical count — check the v5 scope for a "
        "marker broad enough to reach v6's documents")


def test_a_bare_comma_is_not_a_number():
    """`[\\d,]+` matches "," alone, which turned prose into a claim of `doc says ','`."""
    assert captures(V6_KEPT, COMMA_BEFORE_APPENDED_ROWS) == []


def test_the_historical_846_is_exempted_rather_than_matched():
    """Two guards, because this fact's failure mode IS one population read as another."""
    assert captures(V6_KEPT, HISTORICAL_846) == []
    assert any(x in HISTORICAL_846 for x in BY_NAME[V6_KEPT]["exempt"]), (
        "the exemption stopped covering the line it was written for")


# --- #51: a drop reason that never fired ----------------------------------------------

def _report(tmp_path, payload):
    path = tmp_path / "derivation_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_a_drop_reason_that_never_fired_reads_as_zero(tmp_path):
    """The rule's own SUCCESS condition used to fail the gate.

    The builders serialise `dict(collections.Counter())`, so a reason that never fired has
    no key. Rebuild from inputs carrying no digits — what D-M3 exists to achieve — and the
    reader raised, `main()` reported "cannot read the artifact", and a correct corpus with
    correct documents went red.
    """
    path = _report(tmp_path, {"expressive": {"candidates": 40, "kept": 40, "dropped": {}}})
    assert gate.drop_count("expressive.dropped.digits", path) == 0


def test_a_drop_reason_that_did_fire_is_read_as_written(tmp_path):
    path = _report(tmp_path, {"expressive": {"dropped": {"digits": 6}}})
    assert gate.drop_count("expressive.dropped.digits", path) == 6


def test_a_report_with_no_drop_map_at_all_is_still_a_failure(tmp_path):
    """Zero-defaulting is for the REASON, never for the source. Do not mask a real gap."""
    path = _report(tmp_path, {"expressive": {"candidates": 40}})
    with pytest.raises(KeyError):
        gate.drop_count("expressive.dropped.digits", path)


def test_a_drop_map_that_is_not_a_map_is_a_failure(tmp_path):
    path = _report(tmp_path, {"expressive": {"dropped": 6}})
    with pytest.raises(KeyError):
        gate.drop_count("expressive.dropped.digits", path)


def test_an_unreadable_report_is_still_a_failure(tmp_path):
    with pytest.raises(OSError):
        gate.drop_count("expressive.dropped.digits", str(tmp_path / "nope.json"))


# --- #52: coverage must not vanish silently -------------------------------------------

def test_every_fact_declares_the_artifacts_it_reads():
    for fact in ALL_FACTS:
        assert fact.get("artifacts"), (
            f"{fact['name']} declares no artifacts — it would be checked on a machine that "
            f"cannot read it, and go red for a fact it had no way to check")
        for path in fact["artifacts"]:
            assert os.path.isabs(path), f"{fact['name']}: {path} is not absolute"


def test_a_missing_v6_does_not_take_the_v5_facts_down_with_it():
    """The #52 regression, run with a SYNTHETIC filesystem so it runs everywhere.

    The prerequisite used to be ANDed in the pytest harness, so a host holding v5 and not v6
    checked nothing and called it a skip — the ten checkable facts discarded to protect the
    two that were not.
    """
    only_v6_missing = lambda p: p != gate.V6_REPORT  # noqa: E731
    blocked = [f["name"] for f in gate.FACTS if gate.unreadable(f, only_v6_missing)]
    assert blocked == [V6_DIGITS, V6_STAGED, V6_KEPT], (
        "a missing v6 report must block v6's facts and nothing else")

    readable = [f["name"] for f in gate.FACTS if not gate.unreadable(f, only_v6_missing)]
    assert len(readable) == len(gate.FACTS) - 3
    assert V5_DIGITS in readable


def test_with_nothing_on_disk_every_fact_is_named_rather_than_quietly_dropped(capsys,
                                                                              monkeypatch):
    """A laptop is allowed to check nothing. It is not allowed to be quiet about it."""
    real_exists, real_isdir = os.path.exists, os.path.isdir
    artifacts = {p for f in gate.FACTS for p in f["artifacts"]}
    monkeypatch.setattr(gate.os.path, "exists",
                        lambda p: False if p in artifacts else real_exists(p))
    # The checkpoint half lives on the /data mount and is a separate skip; pin it off so
    # this test says the same thing on ai-lab-0 as it does in CI.
    monkeypatch.setattr(gate.os.path, "isdir",
                        lambda p: False if str(p).startswith("/data/model-training")
                        else real_isdir(p))

    assert gate.main() == 0, "an absent corpus is not a finding"
    out = capsys.readouterr().out
    assert f"checked 0 of {len(gate.FACTS)}" in out
    for fact in gate.FACTS:
        assert f"{fact['name']} NOT CHECKED" in out, (
            f"{fact['name']} was dropped without a word — a coverage reduction nobody can "
            f"see is how a gate stops being one")
    assert "NOT CHECKED here" in out, "the PASS footer must not read as full coverage"


# --- the registry as a whole ----------------------------------------------------------

def test_every_fact_recognises_at_least_one_live_statement():
    """A fact no document states is a fact nobody is checking.

    This is the silent-miss direction: the registry can go on passing while the prose drifts
    into a phrasing no pattern knows. If this fails, coverage was lost — find the reworded
    sentence and either restore the idiom or teach the entry the new one.
    """
    for fact in ALL_FACTS:
        hits = 0
        for path in gate.docs():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    hits += len(captures(fact["name"], line))
        assert hits, (
            f"{fact['name']} matches nothing in the notes — either the documents stopped "
            f"stating it, or they restated it in a phrasing this entry cannot read")


def test_every_pattern_carries_exactly_one_capture_group():
    for fact in ALL_FACTS:
        for pat in fact["patterns"]:
            assert re.compile(pat).groups == 1, (
                f"{fact['name']}: {pat!r} — main() reads group(1) and nothing else")


def test_every_exemption_is_still_earning_its_place():
    """An exemption whose line is gone is a hiding place waiting for a new tenant."""
    for fact in ALL_FACTS:
        for needle in fact["exempt"]:
            found = any(needle in line
                        for path in gate.docs()
                        for line in open(path, encoding="utf-8"))
            assert found, (
                f"{fact['name']} exempts {needle!r}, which no document contains any more")


# --- the protocol registry (2026-08-17) -----------------------------------------------
#
# Same discipline as everything above: the RED direction, made durable. Each of these was run
# by hand against the live tree before it was written down, and the counts below are what that
# run reported rather than what anybody expected — hand-counting said the pass cap was restated
# six times and the gate found seven.
#
# ⚠ THESE RUN EVERYWHERE. `workflow/config.env` is tracked, so unlike the corpus facts there is
# no machine on which this coverage quietly becomes a skip.


def _patched_setting(**overrides):
    """`gate.setting` with some constants forced. Everything else still reads the real file."""
    real = gate.setting

    def fake(key, path=gate.CONFIG_ENV):
        if key in overrides:
            return overrides[key]
        return real(key, path)
    return fake


def _protocol_failures(monkeypatch, **overrides):
    monkeypatch.setattr(gate, "setting", _patched_setting(**overrides))
    failures, _skipped, _checked, _matched, _exempted = gate.scan(
        gate.PROTOCOL_FACTS, gate.docs())
    return failures


def test_a_wrong_pass_cap_is_reported_at_every_file_that_restates_it(monkeypatch):
    """The whole point of the registry, in one assertion.

    Nothing compared any two of the seven statements of this constant before it existed. The
    assertion is on the FILE SET rather than on the count, because a new restatement written
    in a recognised phrasing is ordinary prose and must not fail a test — but a file dropping
    OUT of this set means the entry stopped reading a place the constant is claimed.
    """
    failures = _protocol_failures(monkeypatch, MAX_PASSES=4)
    named = {f.split(":")[0] for f in failures}
    for path in ("workflow/WORKFLOW.md", "workflow/DEVELOPER.md",
                 "workflow/config.env", "workflow/scripts/issue.py"):
        assert path in named, (
            f"a wrong pass cap went unreported in {path} — that file restates the constant, "
            f"so the entry has stopped reading a phrasing it used to read.\n"
            f"reported: {sorted(named)}")


def test_the_word_spellings_are_what_make_three_of_those_seven_visible(monkeypatch):
    """⚠ Half the restatements of a small integer are WORDS, and a digits-only registry is
    blind to them while reporting full coverage of the file they live in."""
    failures = _protocol_failures(monkeypatch, MAX_PASSES=4)
    assert any("'three'" in f for f in failures), "the lower-case word form went unread"
    assert any("'Three'" in f for f in failures), (
        "the capitalised word form went unread — `config.env` starts the sentence with it, "
        "and that is the restatement sitting one line above the assignment it restates")


def test_config_env_is_both_the_truth_and_a_scanned_document(monkeypatch):
    """The tightest fork in the lane: a file restating its own constant in the comment above it.

    This is the one hand-counting missed, and it is the one most likely to be edited without
    thinking — nobody re-reads a comment they are not changing.
    """
    assert gate.CONFIG_ENV in gate.docs(), "config.env stopped being scanned"
    assert gate.setting("MAX_PASSES") == 3
    failures = _protocol_failures(monkeypatch, MAX_PASSES=4)
    assert any(f.startswith("workflow/config.env:") for f in failures)


def test_a_wrong_comment_cap_is_reported(monkeypatch):
    failures = _protocol_failures(monkeypatch, COMMENT_MAX=1200)
    assert any("workflow/REVIEWER.md" in f for f in failures), (
        "REVIEWER.md states the cap AND states that the schema enforces it — the two numbers "
        "must move together or that second claim silently becomes false")


def test_the_reviewer_note_that_the_schema_enforces_the_cap_is_still_there():
    """⚠ CHECKED BY HAND AGAINST THE LIVE TRACKER 2026-08-17 and TRUE: `issue_comments.body`
    really does carry `max=1500`, so the cap survives a writer that bypasses `issue.py`.

    This test guards the SENTENCE, not the schema — a network read does not belong in the
    suite. If the sentence is reworded, the protocol fact above stops reading the only line
    that states the cap and its coverage drops to zero with nothing to show for it.
    """
    text = (open(os.path.join(REPO, "workflow", "REVIEWER.md"), encoding="utf-8").read())
    assert re.search(r"hard maximum [\d,]+ characters", text), (
        "REVIEWER.md no longer states the comment cap in the phrasing the registry reads")


def test_the_1839_in_config_env_is_not_read_as_a_wrong_cap():
    """A DIFFERENT fact about a different population, two lines from the constant.

    `([\\d,]+) characters` reads it as the cap and turns the gate red on correct prose — the
    noisy-gate failure that gets a check switched off.

    ⚠ THIS TEST USED TO PASS FOR THE WRONG REASON, and only a mutation found it. The scope
    said `comment` and the line says `Comment`, so it fell out of scope on CASE and the
    pattern was never reached — loosening the pattern left the whole suite green. The scope
    accepts both spellings now, which is what puts the line in front of the pattern and makes
    the assertion below mean what it says.
    """
    line = ("# Comment length cap. The owner set one because a reviewer averaged 1839 "
            "characters and detail")
    assert in_scope(COMMENT_CAP, line), (
        "the 1839 line is out of this fact's scope, so the assertion below proves nothing "
        "about the pattern — that is exactly how this test passed while enforcing nothing")
    assert captures(COMMENT_CAP, line) == []
    loose = re.findall(r"([\d,]+) characters", line)
    assert loose == ["1839"], (
        "the loose pattern is supposed to be fooled — if it stopped being, this test no "
        "longer demonstrates why the real pattern is narrow")


# --- the constant reader --------------------------------------------------------------

def _config(tmp_path, text):
    p = tmp_path / "config.env"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_a_constant_stated_only_in_a_comment_is_not_read_as_a_value(tmp_path):
    """The half that lets config.env be scanned as well as read.

    If comment lines counted, the file's prose would be checked against itself — doc-vs-doc,
    which this gate exists specifically not to do.
    """
    path = _config(tmp_path, "# MAX_PASSES=9 was the old value\nMAX_PASSES=3\n")
    assert gate.setting("MAX_PASSES", path) == 3


def test_a_missing_constant_is_a_failure_rather_than_a_default(tmp_path):
    with pytest.raises(KeyError):
        gate.setting("MAX_PASSES", _config(tmp_path, "BASE_BRANCH=main\n"))


def test_an_empty_constant_is_a_failure_rather_than_a_default(tmp_path):
    """`REPO_SLUG=` is legitimately empty in this file, so empty must be distinguishable from
    set — and for an integer constant it is a missing value, not a zero."""
    with pytest.raises(KeyError):
        gate.setting("MAX_PASSES", _config(tmp_path, "MAX_PASSES=\n"))


def test_a_non_numeric_constant_is_reported_rather_than_crashing_the_run(tmp_path):
    """`int()` raises ValueError, which `scan()` catches beside OSError and KeyError.

    Uncaught it would abort the whole gate — every other fact behind it losing its check to
    one bad line, which is the all-or-nothing failure #52 was about.
    """
    with pytest.raises(ValueError):
        gate.setting("MAX_PASSES", _config(tmp_path, "MAX_PASSES=three\n"))
    fact = dict(BY_NAME[PASS_CAP], truth=lambda: (_ for _ in ()).throw(ValueError("boom")))
    failures, _s, checked, _m, _e = gate.scan([fact], gate.docs())
    assert checked == 0
    assert len(failures) == 1 and "cannot read the artifact" in failures[0]


def test_the_spellings_helper_adds_words_only_when_asked():
    assert gate.spellings(3) == {"3"}
    assert gate.spellings(3, words=True) == {"3", "three", "Three"}
    assert gate.spellings(1500) == {"1500", "1,500"}
    # Past the table, `words` is inert rather than wrong.
    assert gate.spellings(99, words=True) == {"99"}


def test_the_words_flag_and_the_word_patterns_agree():
    """⚠ EITHER HALF ALONE IS SILENTLY BROKEN, in opposite directions.

    A `(\\w+)` pattern on a fact that does not accept words is PERMANENTLY red — it captures
    "three" and compares it against "3". A `words` flag on a fact with no word-capturing
    pattern is inert: it reads as coverage and buys nothing.
    """
    for fact in ALL_FACTS:
        word_patterns = [p for p in fact["patterns"] if r"(\w+)" in p]
        if word_patterns:
            assert fact.get("words"), (
                f"{fact['name']} captures a word with {word_patterns[0]!r} but does not "
                f"accept word spellings — it can only ever report a mismatch")
        if fact.get("words"):
            assert word_patterns, (
                f"{fact['name']} sets words=True and no pattern can capture one — the flag "
                f"is inert and reads as coverage it does not have")


# --- the scanned file set -------------------------------------------------------------

def test_every_named_document_is_scanned_by_full_path():
    """⚠ FULL PATHS, NOT SUBSTRINGS. A guard written as a substring search over the scanned
    set stays green when its subject MOVES — measured twice on 2026-08-17, in two different
    files, within an hour."""
    scanned = set(gate.docs())
    expected = [os.path.join(REPO, n) for n in gate.NAMED_DOCS]
    expected += [os.path.join(REPO, "workflow", n)
                 for n in ("WORKFLOW.md", "DEVELOPER.md", "REVIEWER.md")]
    expected += [gate.CONFIG_ENV, os.path.join(REPO, "workflow", "scripts", "issue.py")]
    for path in expected:
        assert path in scanned, (
            f"{os.path.relpath(path, REPO)} is no longer scanned — every claim in it has "
            f"silently stopped being checked")


def test_nothing_named_is_quietly_missing():
    """`docs()` filters non-existent files so a checkout cannot crash the gate. That filter is
    exactly how a scanned file stops being scanned, so the absence has to be visible."""
    assert gate.absent_docs() == [], (
        f"named documents are missing from this checkout and would be skipped: "
        f"{[os.path.relpath(p, REPO) for p in gate.absent_docs()]}")


def test_a_renamed_document_is_announced_rather_than_dropped(monkeypatch):
    """The red direction for the filter above."""
    real_isfile = os.path.isfile
    gone = os.path.join(REPO, "CLAUDE.md")
    monkeypatch.setattr(gate.os.path, "isfile",
                        lambda p: False if p == gone else real_isfile(p))
    assert gone not in gate.docs()
    assert gone in gate.absent_docs()


def test_protocol_facts_are_checked_even_when_no_corpus_is_on_disk(monkeypatch):
    """The claim in the gate's own comment — 'these run everywhere' — asserted rather than said.

    Every corpus fact skips on a laptop and in CI. If the protocol facts ever acquire an
    artifact that is not tracked, this is what notices.
    """
    real_exists = os.path.exists
    corpus = {p for f in gate.FACTS for p in f["artifacts"]}
    monkeypatch.setattr(gate.os.path, "exists",
                        lambda p: False if p in corpus else real_exists(p))
    _f, skipped, checked, _m, _e = gate.scan(gate.PROTOCOL_FACTS, gate.docs())
    assert skipped == []
    assert checked == len(gate.PROTOCOL_FACTS)


def test_that_trap_line_is_still_the_one_config_env_actually_carries():
    """The synthetic line above is only worth testing while it matches the real file."""
    text = open(gate.CONFIG_ENV, encoding="utf-8").read()
    assert "averaged 1839 characters" in text, (
        "config.env no longer carries the sentence this trap was built from — re-derive the "
        "trap from the file rather than deleting the test")
