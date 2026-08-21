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
import shutil
import subprocess
import sys

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
    """Build a synthetic repo. `files` maps relative path -> contents.

    ⚠ IT IS A REAL GIT REPO WITH THE FILES STAGED, and that is not ceremony. The gate scans
    `git ls-files`, so a fixture that only wrote to disk would exercise a code path production
    never takes — and would have to be served by a working-tree fallback whose behaviour then
    differs from the real gate's. `git add` needs no identity configured (only `commit` does),
    so this stays hermetic.
    """
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True,
                   capture_output=True, text=True)
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
                 "[x](mailto:a@b.c)", "[x](#a-section)",
                 "[x](file:///Users/someone/Projects/thing.rs)"):
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


def test_the_scan_is_every_tracked_markdown_except_workflow():
    """⚠ FULL PATHS, NOT SUBSTRINGS — a guard written as a substring stays green when its
    subject moves, which is how three checks went quiet in this repo on 2026-08-17.

    ⚠ The set widened on 2026-08-21 (#261): it was `notes/`+`docs/`+`workflow/`+3 root files,
    38 of 53 tracked markdown files, and **7 dead links were living in one of the 15 it never
    opened**. Both halves of the gate now scan `repo_markdown()`. `workflow/` is excluded by
    owner ruling — the review lane is retired and a dead link in `REVIEWER.md` must not fail
    a Sonora merge.
    """
    scanned = set(gate.repo_markdown(REPO))
    for rel in ("notes/README.md", "docs/README.md", "docs/ARCHITECTURE.md", "notes/STATE.md",
                # the four that the old PROSE_DIRS set never opened — #260 lived in the last
                "README-Matcha.md", "audition/README.md", "scripts/README.md",
                "scripts/teacher_audition/README.md"):
        assert os.path.join(REPO, rel) in scanned, f"{rel} is not scanned"
    for rel in ("workflow/WORKFLOW.md", "workflow/REVIEWER.md", "workflow/DEVELOPER.md"):
        assert os.path.join(REPO, rel) not in scanned, (
            f"{rel} IS scanned — the retired lane is back in a Sonora merge gate")


def test_the_scan_reads_the_index_not_the_working_tree(tmp_path):
    """An untracked markdown file must not be scanned: reporting on a file no clone has is
    the working-tree-vs-index confusion this repo has already paid for.

    ⚠ Runs in the HERMETIC fixture, not the live checkout. The first version of this wrote a
    probe into the repo root and removed it in a `finally` (#268) — a test that mutates the
    tree it is asserting about, with a window where a concurrent `git status` sees junk, while
    ignoring the `tmp_path` it had already asked for. `tree()` builds a real index; use it.
    """
    root = tree(tmp_path, {"notes/a.md": "prose\n"})
    tracked = os.path.join(root, "notes", "a.md")
    stray = os.path.join(root, "notes", "untracked.md")
    open(stray, "w").close()
    scanned = set(gate.repo_markdown(root))
    assert tracked in scanned, "the tracked file was not scanned"
    assert stray not in scanned, "an untracked .md was scanned — repo_markdown reads the tree"


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

    ⚠ AST, NOT TEXT-SCANNING, and the first draft proved why within a minute: written as two
    regexes over raw lines, it flagged the DOCSTRING THREE LINES ABOVE — prose that quotes the
    old code is not the old code. That is the third time this exact shape has cost something in
    two days (`tests/test_schemas.py` hit it with docstrings, then again with f-strings, which
    tokenize as FSTRING_* rather than STRING on 3.12+). A parser cannot be fooled by a quote of
    itself, and M4 of notes/quality-mechanisms-plan.md is about making this the easy path
    rather than the one you reach for after being bitten.
    """
    import ast
    import subprocess

    def built_paths(tree):
        """-> [(lineno, dir, filename)] for repo-relative doc paths built in this module."""
        found = []
        for node in ast.walk(tree):
            # os.path.join(REPO, "docs", "X.md")
            if isinstance(node, ast.Call):
                f = node.func
                if (isinstance(f, ast.Attribute) and f.attr == "join"
                        and len(node.args) == 3
                        and isinstance(node.args[0], ast.Name) and node.args[0].id == "REPO"
                        and all(isinstance(a, ast.Constant) and isinstance(a.value, str)
                                for a in node.args[1:])):
                    d, name = node.args[1].value, node.args[2].value
                    if name.endswith(".md"):
                        found.append((node.lineno, d, name))
            # REPO / "docs" / "X.md" — parses as ((REPO / "docs") / "X.md"), so the constants
            # come off right-to-left and `ast.walk` also visits the inner node, which has only
            # one constant and is skipped by the length test.
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                parts, cur = [], node
                while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
                    if isinstance(cur.right, ast.Constant) and isinstance(cur.right.value, str):
                        parts.append(cur.right.value)
                    cur = cur.left
                if (isinstance(cur, ast.Name) and cur.id == "REPO" and len(parts) == 2
                        and parts[0].endswith(".md")):
                    found.append((node.lineno, parts[1], parts[0]))
        return found

    tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO,
                             capture_output=True, text=True).stdout.split()
    bad, seen = [], 0
    for rel in tracked:
        try:
            with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
        except (UnicodeDecodeError, OSError, SyntaxError):
            continue
        for lineno, d, name in built_paths(tree):
            seen += 1
            if not os.path.isfile(os.path.join(REPO, d, name)):
                bad.append(f"{rel}:{lineno} — builds {d}/{name}, which does not exist")
    assert not bad, "\n".join(bad)
    # ⚠ A guard that matches nothing is not a passing guard. If the tree stops building doc
    # paths in code this can go, but it must be a decision rather than a silent drift to zero.
    assert seen, "this guard now examines NO path in the tree — it has stopped binding"


# --- `<file>.md §N` citations (#181, #185) ---------------------------------------------
#
# ⚠ THESE ARE NOT LINKS, WHICH IS THE ENTIRE DEFECT. Every one of the 19 real instances is
# spelled `[todo.md §6](todo.md)`: the link target is the FILE and the section lives only in
# the label. The link resolves, nothing goes red, and the reader lands on the right document
# to hunt for a section that is not there. So these tests exercise the LABEL, and the link
# tests above are untouched by them.

def test_a_section_citation_into_a_file_with_that_heading_is_fine(tmp_path):
    root = tree(tmp_path, {
        "notes/a.md": "see [t.md §2](t.md)\n",
        "notes/t.md": "# T\n\n## 1 · one\n\n## 2 · two\n",
    })
    dangling, examined, skipped = gate.section_citations(root)
    assert dangling == [] and examined == 1 and skipped == []


def test_a_section_citation_into_a_missing_section_is_reported(tmp_path):
    """The #181 shape: the target file HAS numbered headings, just not that one."""
    root = tree(tmp_path, {
        "notes/a.md": "see [t.md §6](t.md)\n",
        "notes/t.md": "# T\n\n## 1 · one\n\n## 2 · two\n",
    })
    dangling, examined, skipped = gate.section_citations(root)
    (rel, lineno, target, n, available), = dangling
    assert examined == 1 and skipped == []
    assert (os.path.basename(rel), lineno, target, n) == ("a.md", 1, "t.md", "6")
    assert available == ["1", "2"], "the message must name what the file DOES have"


def test_a_section_citation_into_a_file_with_no_numbers_is_reported(tmp_path):
    """The #185 shape, and the larger of the two: nine citations named `model-decisions.md §5`
    in a file whose seven sections are all NAMED. `available` is empty, and the gate's message
    turns that into "NO numbered headings" — a different remedy from a wrong number, because
    the fix is to cite by name rather than to pick a different digit."""
    root = tree(tmp_path, {
        "notes/a.md": "see [t.md §5](t.md)\n",
        "notes/t.md": "# T\n\n## Why not Kokoro\n\n## The base model\n",
    })
    dangling, _examined, _skipped = gate.section_citations(root)
    (_rel, _lineno, _target, n, available), = dangling
    assert n == "5" and available == []


def test_scripts_assets_is_scanned_for_citations_though_not_for_links(tmp_path):
    """⚠ TWO OF THE SIX `todo.md` DEFECTS WERE IN SHIPPED DIRECTOR SKILL FILES under
    `scripts/assets/`, which is outside PROSE_DIRS. A guard blind to the files that motivated
    it is not a guard — so this check, and ONLY this check, is widened to reach them. The link
    half must stay where it was: changing which paths it resolves is a scope change, and this
    repo has already paid for one of those made silently."""
    root = tree(tmp_path, {
        "scripts/assets/director_skills/x.md": "traps in [t.md §6](../../../notes/t.md)\n",
        "notes/t.md": "# T\n\n## 1 · one\n",
    })
    found, _examined, _skipped = gate.section_citations(root)
    assert len(found) == 1 and "director_skills" in found[0][0]


def test_a_citation_naming_a_file_this_repo_does_not_hold_is_left_to_the_link_half(tmp_path):
    """Reporting it here would duplicate the dead-link failure under a worse message — one
    that talks about sections when the actual problem is the filename."""
    root = tree(tmp_path, {"notes/a.md": "see [nowhere.md §3](nowhere.md)\n"})
    dangling, examined, skipped = gate.section_citations(root)
    assert dangling == [] and examined == 0
    # ⚠ #255: it used to vanish here. Skipping is still right — this half cannot resolve a
    # name the repo does not hold — but it must be COUNTED, because the caller cannot
    # otherwise tell "nothing dangles" from "nothing was looked at".
    assert skipped == [("notes/a.md", 1, "nowhere.md", "3")]


def test_a_shared_basename_resolves_as_a_union(tmp_path):
    """`README.md` exists in both `notes/` and `docs/`, and a citation carries no path. The
    union is deliberately permissive: this check should fail on a section that exists NOWHERE
    under that name, never on the ambiguity of which copy was meant."""
    root = tree(tmp_path, {
        "notes/a.md": "see [README.md §4](README.md)\n",
        "notes/README.md": "# N\n\n## 1 · one\n",
        "docs/README.md": "# D\n\n## 4 · four\n",
    })
    dangling, examined, _skipped = gate.section_citations(root)
    assert dangling == [] and examined == 1


def test_the_examined_count_is_what_the_floor_must_watch(tmp_path):
    """⚠ #255's secondary: a floor on FILES SCANNED cannot see a citation scan that reads
    nothing. This branch wrote that lesson down for the shell heredocs — "all 14 could be
    enumerated while `_embedded_python` silently matched none of them" — and then put the
    floor on the outer population here anyway. Files and examined-citations move
    independently, so the floor watches the inner one."""
    root = tree(tmp_path, {
        "notes/a.md": "prose with no citations at all\n",
        "notes/b.md": "more prose\n",
        "notes/t.md": "# T\n\n## 1 · one\n",
    })
    dangling, examined, skipped = gate.section_citations(root)
    assert dangling == [] and skipped == []
    assert examined == 0, "three files scanned, zero citations read — a file floor sees 3"


# --- main(), executed end to end (#267) --------------------------------------------------
#
# ⚠ THE OBVIOUS TEST IS THE WRONG ONE, AND THAT IS WHY THIS CLASS HAD NO COVERAGE. `main()`
# takes no `root`, and every helper defaults `root=REPO` bound at import — so monkeypatching
# `gate.REPO` leaves half the inputs reading the LIVE checkout and the test passes for the
# wrong reason. The gate is a standalone program with an exit code (that is how pipelines use
# it), so it is run as one, inside a fixture repo, with `REPO` deriving from `__file__` the way
# production does.
#
# Until 2026-08-21 the `depth` kind, the `KINDS` table, the unrouted-kind catch and the
# coverage floor had ZERO automated coverage. Every one of them was verified by a hand-run
# mutation that nothing preserved — which is exactly how this branch kept shipping guards that
# were green for the wrong reason.

def _gate_repo(tmp_path, files):
    """A fixture repo with the real gate script installed at its real path. -> (root, run)."""
    root = tree(tmp_path, files)
    gates = os.path.join(root, "scripts", "gates")
    os.makedirs(gates, exist_ok=True)
    shutil.copy(GATE, os.path.join(gates, "test_doc_links.py"))
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)

    def run():
        # SONORA_SIBLING_REPOS -> a path that cannot exist, so the sibling half is a printed
        # skip and never reaches out to the real Prosodia/AI-Lab-AMD checkouts.
        env = dict(os.environ, SONORA_SIBLING_REPOS=os.path.join(root, "no-such-sibling"))
        return subprocess.run([sys.executable, os.path.join(gates, "test_doc_links.py")],
                              cwd=root, capture_output=True, text=True, env=env)
    return root, run


def test_main_is_green_on_a_clean_fixture(tmp_path):
    _root, run = _gate_repo(tmp_path, {
        "notes/a.md": "see [t.md](t.md) and [t.md §1](t.md)\n",
        "notes/t.md": "# T\n\n## 1 · one\n",
    })
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


def test_main_reports_a_dead_link_under_the_link_heading(tmp_path):
    _root, run = _gate_repo(tmp_path, {"notes/a.md": "[x](gone.md)\n", "notes/t.md": "# T\n"})
    r = run()
    assert r.returncode == 1
    assert "dead link(s) — the target file does not exist" in r.stdout
    assert "restore the file" in r.stdout


def test_main_reports_a_dangling_section_with_the_opposite_remedy(tmp_path):
    """⚠ The remedy is the assertion. For a dangling `§N` the link RESOLVES, so "fix the link"
    is the wrong move — that mismatch was #259."""
    _root, run = _gate_repo(tmp_path, {
        "notes/a.md": "see [t.md §6](t.md)\n",
        "notes/t.md": "# T\n\n## 1 · one\n",
    })
    r = run()
    assert r.returncode == 1
    assert "the LINK RESOLVES, the section does not" in r.stdout
    assert "Do NOT 'fix the link'" in r.stdout


def test_main_reports_a_wrong_depth_link_and_names_the_second_cause(tmp_path):
    """The `depth` kind (#261), including the harder case the first tell missed (#265): a link
    at the wrong depth whose target has ALSO moved. `notes/` exists here; `notes/gone.md`
    does not, and the message must say so rather than falling through to a sibling skip."""
    _root, run = _gate_repo(tmp_path, {
        "scripts/teacher_audition/README.md":
            "[a](../../../notes/t.md)\n[b](../../../notes/gone.md)\n",
        "notes/t.md": "# T\n",
    })
    r = run()
    assert r.returncode == 1
    assert "WRONG DEPTH" in r.stdout
    assert "README.md:1" in r.stdout and "README.md:2" in r.stdout, (
        "both links must be caught — the tell is the NAME colliding with a local directory, "
        "not the tail resolving:\n" + r.stdout)
    assert "does not exist either" in r.stdout, "the second cause is not named for line 2"
    assert "no checkout of it on this machine" not in r.stdout, (
        "a wrong-depth link was reported as an absent sibling — the #265 miss")


def test_the_live_citation_count_holds():
    """The RATCHET that the gate deliberately does not carry (#267).

    A partial blinding — 37 citations read dropping to 5 — is invisible to the gate's
    "read nothing" check, and rightly so: the gate must run on any tree, and a magnitude from
    THIS repo baked into it makes every smaller checkout red. The magnitude is a property of
    this tree, so it is asserted here, against this tree, where it can be re-derived.

    ⚠ Floor, not equality: adding documents must not fail this. 37 examined of 42 found,
    measured 2026-08-21.
    """
    _dangling, examined, skipped = gate.section_citations(REPO)
    assert examined >= 30, (
        f"only {examined} §N citation(s) are read in this repo, from the 37 measured on "
        f"2026-08-21. Either SECTION_CITE stopped matching or the scanned set shrank — "
        f"re-derive this floor deliberately rather than lowering it to fit.")
    assert examined + len(skipped) >= 35, (
        f"only {examined + len(skipped)} citation(s) were FOUND at all (42 measured) — "
        f"suspect the regex before believing the citations went away.")
