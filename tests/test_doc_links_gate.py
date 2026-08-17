"""The link checker's own logic, exercised with synthetic trees — no siblings, no network.

WHY THIS FILE EXISTS (2026-08-17, M0)
-------------------------------------
`scripts/gates/test_doc_links.py` was written to make the `notes/` -> `notes/` + `docs/` split
safe, and its first run reported **34 dead links that were every one of them correct** — it
resolved cross-repo paths literally, and `../../../Prosodia` from a checkout at
`/data/repos/Sonora` is `/data/Prosodia`, which is nobody's mistake. A gate that noisy gets
switched off, and this repo has the scar tissue to say so: the doc-claims registry's first run
produced 20 false failures from one loose pattern and its comments still carry the warning.

So most of what follows pins the DISTINCTION the fix turns on — in-repo links are resolved,
cross-repo links are checked by name and tail — in both directions, because getting it wrong
either way is silent. Too strict and the gate is ignored; too loose and it finds nothing.

**PROVING A GATE PASSES PROVES NOTHING; PROVE IT FAILS.**
"""

import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(REPO, "scripts", "gates", "test_doc_links.py")


def _load():
    spec = importlib.util.spec_from_file_location("sonora_doc_links_gate", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load()


def tree(tmp_path, files):
    """Build a synthetic repo. `files` maps relative path -> contents."""
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return str(tmp_path)


# --- what counts as a link at all -----------------------------------------------------

def test_a_double_bracket_memory_slug_is_not_a_link():
    """⚠ THE ONE RULE THIS GATE INHERITS RATHER THAN INVENTS.

    `notes/README.md` is the only place it is written down: `[[double-bracket]]` names point at
    the agent's persistent memory and are deliberately unresolvable. A checker that flagged
    them would report dozens of "failures" that are all correct, and be turned off within a
    day. They contain no `](`, so the pattern cannot match one — but that is a PROPERTY of the
    pattern, and properties get edited.
    """
    assert gate.targets("see [[vocoder-not-the-bottleneck]] for the measurement") == []
    assert gate.targets("both [[a]] and [[b]] are slugs") == []


def test_external_and_anchor_targets_are_left_alone():
    """Fetching URLs would make the gate fail offline; anchors are a different check."""
    for line in ("[x](https://example.com/a.md)", "[x](http://example.com)",
                 "[x](mailto:a@b.c)", "[x](#a-section)"):
        assert gate.targets(line) == [], line


def test_an_anchor_on_a_real_file_is_checked_as_the_file():
    assert gate.targets("[x](model-decisions.md#5)") == [("model-decisions.md#5",
                                                          "model-decisions.md")]


# --- in-repo vs cross-repo, the distinction the gate turns on -------------------------

def test_a_link_that_stays_inside_the_repo_is_resolved_here(tmp_path):
    root = tree(tmp_path, {"notes/a.md": "[x](../docs/b.md)\n", "docs/b.md": "hi\n"})
    kind, path, _ = gate.classify(os.path.join(root, "notes", "a.md"), "../docs/b.md", root)
    assert kind == "in"
    assert os.path.exists(path)
    assert gate.dangling(root) == []


def test_a_link_that_stays_inside_and_is_broken_is_a_failure(tmp_path):
    root = tree(tmp_path, {"notes/a.md": "[x](../docs/gone.md)\n", "docs/b.md": "hi\n"})
    bad = gate.dangling(root)
    assert [(r, t) for r, _n, t in bad] == [("notes/a.md", "../docs/gone.md")]


def test_a_link_that_leaves_the_repo_is_never_resolved_by_path(tmp_path):
    """⚠ THE 34-FALSE-FAILURE BUG, pinned.

    Where a sibling checkout sits is not something this tree can know — there are two Sonora
    checkouts on the machine this was written on. Resolving the path asks a question the repo
    has no way to answer and then treats the answer as a defect.
    """
    root = tree(tmp_path, {"notes/a.md": "[x](../../../Prosodia/notes/next-steps.md)\n"})
    kind, name, tail = gate.classify(
        os.path.join(root, "notes", "a.md"), "../../../Prosodia/notes/next-steps.md", root)
    assert (kind, name, tail) == ("cross", "Prosodia", "notes/next-steps.md")
    assert gate.dangling(root) == [], (
        "a cross-repo link was resolved as an in-repo path — this is the bug that reported "
        "34 correct citations as dead")


def test_outbound_links_are_grouped_by_the_sibling_they_name(tmp_path):
    root = tree(tmp_path, {
        "notes/a.md": "[x](../../../Prosodia/notes/p.md) and [y](../../AI-Lab-AMD/notes/q.md)\n",
    })
    out = gate.outbound(root)
    assert set(out) == {"prosodia", "ai-lab-amd"}
    assert out["prosodia"][0][2] == "notes/p.md"


def test_an_outbound_link_is_judged_against_the_sibling_on_disk(tmp_path):
    sib = tmp_path / "Prosodia"
    (sib / "notes").mkdir(parents=True)
    (sib / "notes" / "p.md").write_text("hi", encoding="utf-8")
    root = tree(tmp_path / "repo", {
        "notes/a.md": "[ok](../../Prosodia/notes/p.md)\n[bad](../../Prosodia/notes/nope.md)\n"})
    links = gate.outbound(root)["prosodia"]
    bad = gate.outbound_dangling(str(sib), links)
    assert [t for _r, _n, t, _raw in bad] == ["notes/nope.md"]


def test_the_tail_is_matched_case_sensitively(tmp_path):
    """⚠ A REAL FIND, NOT A HYPOTHETICAL. `notes/high-ambition-6…` cited
    `Prosodia/Docs/defensive-publication-expressive-control.md` against a repo whose directory
    is `docs/`. Case-folding the tail to make the gate green would have hidden it.
    """
    sib = tmp_path / "Prosodia"
    (sib / "docs").mkdir(parents=True)
    (sib / "docs" / "d.md").write_text("hi", encoding="utf-8")
    root = tree(tmp_path / "repo", {"notes/a.md": "[x](../../Prosodia/Docs/d.md)\n"})
    bad = gate.outbound_dangling(str(sib), gate.outbound(root)["prosodia"])
    assert [t for _r, _n, t, _raw in bad] == ["Docs/d.md"]


# --- the inbound half ------------------------------------------------------------------

def test_an_inbound_link_is_matched_on_its_tail_not_its_prefix(tmp_path):
    """A sibling writes `../../Sonora/github/notes/x.md`; the prefix encodes ITS layout."""
    root = tree(tmp_path / "repo", {"notes/x.md": "hi\n"})
    sib = tmp_path / "Prosodia"
    sib.mkdir()
    (sib / "s.md").write_text(
        "[a](../../Sonora/github/notes/x.md) [b](../../Sonora/anywhere/notes/gone.md)\n",
        encoding="utf-8")
    bad = gate.inbound_dangling(str(sib), root)
    assert [t for _r, _n, t in bad] == ["notes/gone.md"]


def test_an_inbound_link_into_docs_is_checked_too(tmp_path):
    """The split created a second destination for inbound links.

    ⚠ ASSERTING THE EMPTY LIST IS NOT ENOUGH, and a mutation proved it: narrowing the inbound
    pattern back to `notes/` only left this test GREEN, because a link nobody looks at reports
    no failure in exactly the same way a link that resolves does. The RED direction is what
    distinguishes "checked and fine" from "never examined", so it goes first.
    """
    root = tree(tmp_path / "repo", {"docs/c.md": "hi\n"})
    sib = tmp_path / "Prosodia"
    sib.mkdir()

    (sib / "broken.md").write_text("[a](../../Sonora/x/docs/gone.md)\n", encoding="utf-8")
    bad = gate.inbound_dangling(str(sib), root)
    assert [t for _r, _n, t in bad] == ["docs/gone.md"], (
        "a dead inbound link into docs/ was not reported — the inbound pattern has stopped "
        "recognising the directory the split created")

    (sib / "broken.md").unlink()
    (sib / "ok.md").write_text("[a](../../Sonora/x/docs/c.md)\n", encoding="utf-8")
    assert gate.inbound_dangling(str(sib), root) == []


# --- absence is a skip, never a pass ---------------------------------------------------

def test_a_sibling_that_is_not_checked_out_is_reported_not_assumed(monkeypatch):
    monkeypatch.setenv(gate.SIBLING_ENV, "/nonexistent/Prosodia")
    found = [p for _w, p in gate.sibling_paths()]
    assert found == [None], "an absent sibling must be kept and reported, not dropped"


def test_the_env_var_overrides_the_defaults(monkeypatch, tmp_path):
    (tmp_path / "Elsewhere").mkdir()
    monkeypatch.setenv(gate.SIBLING_ENV, str(tmp_path / "Elsewhere"))
    assert [os.path.basename(p) for _w, p in gate.sibling_paths()] == ["Elsewhere"]


# --- the live tree ---------------------------------------------------------------------

def test_the_live_repo_has_no_in_repo_dead_links():
    """The acceptance criterion for the split itself: 224 relative links were rewritten and
    this is what says they landed."""
    bad = gate.dangling()
    assert bad == [], "\n".join(f"{r}:{n} -> {t}" for r, n, t in bad)


def test_both_prose_directories_are_scanned():
    """⚠ FULL PATHS, NOT SUBSTRINGS — a guard written as a substring stays green when its
    subject moves, which is how three checks went quiet in this repo on 2026-08-17."""
    scanned = set(gate.markdown_files(REPO))
    for rel in ("notes/README.md", "docs/README.md", "docs/ARCHITECTURE.md",
                "notes/STATE.md", "workflow/WORKFLOW.md"):
        assert os.path.join(REPO, rel) in scanned, f"{rel} is not scanned"


@pytest.mark.parametrize("rel", ["docs/ARCHITECTURE.md", "docs/vat-channels.md",
                                 "docs/model-decisions.md", "docs/markup-schema-brief.md",
                                 "docs/direction-interface-brief.md",
                                 "docs/tts-engine-onboarding.md",
                                 "docs/audiobook-corpus-policy.md"])
def test_the_canon_landed_where_the_plan_said(rel):
    assert os.path.isfile(os.path.join(REPO, rel))
    assert not os.path.isfile(os.path.join(REPO, rel.replace("docs/", "notes/")))


def test_the_high_ambition_series_did_not_move():
    """⚠ Prosodia links to these BY NAME across repo boundaries; moving them breaks links no
    checker in this repo would ever see. `notes/README.md` states the constraint and the
    measurement backed it: the rejected 16-file draft would have broken 21 inbound links."""
    for name in ("high-ambition-index.md", "high-ambition-1-matcha-actor.md",
                 "high-ambition-2-dramatic-reader.md",
                 "high-ambition-6-audience-conveyance-stt.md",
                 "high-ambition-7-singing.md"):
        assert os.path.isfile(os.path.join(REPO, "notes", name)), name
        assert not os.path.isfile(os.path.join(REPO, "docs", name)), (
            f"{name} moved to docs/ — Prosodia cites it by name and nothing here would notice")


# --- doc paths built in CODE, which no link checker can see -----------------------------

def test_every_doc_path_constructed_in_python_resolves():
    """⚠ THE CLASS THE LINK CHECKER CANNOT REACH, and it bit during this very split.

    `tests/test_delivery_channel.py` built its target as `REPO / "notes" / "ARCHITECTURE.md"`
    — a path join, not a markdown link — so the relink pass that rewrote 81 links walked
    straight past it and the test failed with a FileNotFoundError after the move. A link
    checker reads prose; this reads the other half.

    Both spellings the tree uses are covered. `os.path.join(REPO, "notes")` with no filename
    is not — that is a directory constant, and `scripts/gates/test_doc_claims.py` legitimately
    holds one for each prose directory.
    """
    import re as _re
    import subprocess

    joined = _re.compile(
        r'os\.path\.join\(\s*REPO\s*,\s*"([a-z]+)"\s*,\s*"([^"]+\.md)"\s*\)')
    divided = _re.compile(r'REPO\s*/\s*"([a-z]+)"\s*/\s*"([^"]+\.md)"')

    tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO,
                             capture_output=True, text=True).stdout.split()
    bad = []
    for rel in tracked:
        path = os.path.join(REPO, rel)
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(lines, 1):
            for pat in (joined, divided):
                for d, name in pat.findall(line):
                    if not os.path.isfile(os.path.join(REPO, d, name)):
                        bad.append(f"{rel}:{lineno} — builds {d}/{name}, which does not exist")
    assert not bad, "\n".join(bad)
