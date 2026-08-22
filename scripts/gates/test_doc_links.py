#!/usr/bin/env python3
"""Gate: every relative link in the prose resolves — in this repo, and from its siblings.

WHY THIS EXISTS (2026-08-17, owner-commissioned as M0 of notes/quality-mechanisms-plan.md)
------------------------------------------------------------------------------------------
The `notes/` -> `notes/` + `docs/` split is the occasion for this file, not its product. Before
the split, `notes/` was one flat directory and **224 links inside it were bare filenames** —
`[STATE.md](STATE.md)` — which resolve only because every file sits together. Splitting the
directory is exactly what stops that being true, and a broken markdown link does not error: it
renders, and goes nowhere. That is the silent-degrade shape this repo keeps meeting, so the
move could not be done by hand and cannot be left unguarded afterwards.

**THE CROSS-REPO HALF IS NOT AN EXTRA — IT IS WHERE THE ROT ALREADY WAS.** Measured the day
this was written: Prosodia carries 37 links into Sonora's notes, and **10 of them were already
dead** — `notes/archive/` was removed on 2026-08-02 and neither repo noticed for a fortnight.
A checker that only looked at Sonora's own outbound links would have reported a clean tree on
the day that number was 10, which is the definition of a check pointed away from the failure.

⚠ `[[double-bracket]]` NAMES ARE NOT LINKS. They are the agent's persistent-memory slugs and
are deliberately unresolvable; `notes/README.md` is the only place that rule is written down,
which makes it part of this file's specification. They contain no `](`, so the link pattern
below cannot match one — but that is a property of the pattern, and
`tests/test_doc_links_gate.py` asserts it rather than trusting it.

WHAT IT DOES NOT COVER — read this before trusting a pass
---------------------------------------------------------
* ⚠ **THE FILE SET — read this first, because it is what a green run is scoped to.** Both
  halves scan `repo_markdown()`: every TRACKED `.md` except `workflow/`. That is 50 of the
  repo's 53 today. It was `notes/`+`docs/`+`workflow/`+3 root files — 38 of 53 — while this
  banner said "every relative link this repo owns resolves", and **7 dead links were living
  in one of the 15 files it never opened** (#261). Untracked markdown is deliberately not
  read: a scan that reports on files no clone has is the working-tree-vs-index confusion
  again.
* **Relative links only.** `http(s)://` and `mailto:` are not checked; a 404 on the public
  internet is not this gate's business and would make it fail offline.
* **Inline `[label](target)` only.** A bare backticked path such as `` `docs/x.md` `` is not
  a link and is never checked, in any file. Four live stale ones were found by hand (#260).
* **Existence, not correctness.** `#anchor` fragments are stripped and NOT verified — a link
  to a real file's vanished section still passes.
* **A CROSS-REPO LINK IS CHECKED BY ITS TAIL, NEVER BY ITS PATH — in both directions.**
  Prosodia writes `../../Sonora/github/notes/x.md` and Sonora writes
  `../../../Prosodia/notes/y.md`; both encode where the author's checkouts happened to sit,
  and **two Sonora checkouts exist on this machine**. Resolving those paths literally reported
  **34 dead links on the first run, every one of them a correct citation** — the checkout is
  at `/data/repos/Sonora`, so `../../../Prosodia` resolves to `/data/Prosodia`, which is
  nobody's mistake. That is the false-failure class this repo has learned to fear: a noisy
  gate is a gate someone switches off. What is checkable, and what actually matters, is
  whether the NAMED REPO still holds the named file.
  ⚠ The sibling directory is matched case-INSENSITIVELY (a checkout may be cloned under any
  name) while the tail is matched EXACTLY, which is what caught
  `Prosodia/Docs/...` against a repo whose directory is `docs/`.
* **A sibling that is not checked out is a printed skip, never a pass.** "Prosodia is not on
  this machine" is not a finding.

Run:  .venv/bin/python scripts/gates/test_doc_links.py
Exit: 0 every relative link resolves; 1 otherwise.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# ⚠ THERE IS EXACTLY ONE SCAN SET AND IT IS `repo_markdown()`. `PROSE_DIRS`, `ROOT_DOCS` and
# `markdown_files()` were deleted on 2026-08-21 (#266): after both halves moved to
# `repo_markdown()` they had NO callers, while `markdown_files`'s docstring still said "Every
# markdown file this gate reads" and its comment named callers that no longer existed — and
# the tuple still listed `workflow/`, which the owner excluded. A second scan set is a second
# thing to keep in sync; this gate has now had three and been wrong about which was live.

# ⚠⚠ THIS REPO'S SCAN IS EVERY TRACKED `.md`, MINUS `workflow/` — one set for BOTH halves.
#
# It was `PROSE_DIRS + ROOT_DOCS` for links and a different, wider set for `§N` citations, and
# BOTH were wrong in the same direction. Measured (#261): the link half read **38 of the
# repo's 53** markdown files while printing "every relative link this repo owns resolves", and
# **7 dead links were living in one of the 15 it never opened** (`scripts/teacher_audition/
# README.md`, every link at the wrong depth). The narrowness was deliberate and documented —
# beside `CITATION_DIRS`, which is not where a reader is told to look. The "WHAT IT DOES NOT
# COVER" list did not mention the file set at all.
#
# ⚠ The asymmetry was its own defect. The §N half had already been widened to `scripts/assets/`
# on the reasoning that "a guard blind to the files that motivated it is not a guard" — and
# that is exactly what the link half's blind spot then produced (#260). Two halves of one gate
# disagreeing about which files exist is a second thing to keep in sync and nobody was.
#
# ⚠ `workflow/` IS EXCLUDED, both halves, by owner ruling 2026-08-21: the review lane is
# retired and this gate is scoped to Sonora's own code and docs. A dead link or a `§N` inside
# `REVIEWER.md` is not a Sonora defect and must not fail a Sonora merge. The honest cost is
# stated at `section_citations`: `CLAUDE.md:21` cites a `REVIEWER.md §0` that does not exist,
# and this gate steps over it deliberately.
_SCAN_EXCLUDE = ("workflow/",)

# This checker and the file that tests it — see `citation_sources`. Both quote `§N` citations
# as examples; scanning them reports the explanation as the defect.
_CITATION_EXEMPT = ("scripts/gates/test_doc_links.py", "tests/test_doc_links_gate.py")


def _tracked_top_dirs(root=REPO):
    """Top-level directory names git actually carries. -> sorted list."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                         capture_output=True, text=True, check=True)
    return sorted({rel.split("/", 1)[0] for rel in out.stdout.split("\0")
                   if rel and "/" in rel})


def citation_sources(root=REPO):
    """Files a `<file>.md §N` citation can appear in: tracked `.md` AND `.py`. -> abs paths.

    ⚠ DELIBERATELY A DIFFERENT SET FROM `repo_markdown()`, AND THE REASON MATTERS — this gate
    has been burned by a second scan set before (#266), so the split is stated rather than
    left to be discovered. The LINK half must stay `.md`-only: `](...)` in Python is not a
    markdown link. A `§N` CITATION is just prose, and prose lives in comments and docstrings
    as readily as in documents.

    Measured (#275): two live `notes/todo.md § 8` citations survived #181's sweep in
    `matcha/cli.py` and `scripts/lib/derive_vat_corpus.py` — #181's exact defect, structurally
    invisible to a `.md`-only scan. Fixing the two instances without widening the scan leaves
    the class alive, which is this branch's standing lesson: a guard beats a sweep.

    ⚠ **THE CHECKER AND ITS OWN TESTS ARE EXEMPT, and the cost is real.** Both quote
    citations as EXAMPLES — `[todo.md §6](todo.md)` appears in this file's comments and in
    `tests/test_doc_links_gate.py`'s fixtures precisely to describe the defect being caught.
    Measured: without the exemption they contribute 6 of 8 "dangling" hits, all false. A gate
    that fires on the prose explaining it is one somebody switches off. The cost, stated
    rather than discovered: a GENUINE dangling citation in either file goes unseen. That is
    the same trade the doc-claims gate's `exempt` list makes for deliberate historical
    mentions, and it is narrow — two files, both of which exist to describe this check.
    """
    out = subprocess.run(["git", "ls-files", "-z", "*.md", "*.py"], cwd=root,
                         capture_output=True, text=True, check=True)
    return sorted(
        os.path.join(root, rel) for rel in out.stdout.split("\0")
        if rel and not rel.startswith(_SCAN_EXCLUDE) and rel not in _CITATION_EXEMPT
    )


def repo_markdown(root=REPO):
    """Every tracked `.md` in this repo except the excluded prefixes. -> sorted abs paths.

    ⚠ TRACKED, not on-disk: a scan that reads untracked files reports on things no clone has,
    which is the working-tree-vs-index confusion this repo has already paid for elsewhere.
    """
    out = subprocess.run(["git", "ls-files", "-z", "*.md"], cwd=root,
                         capture_output=True, text=True, check=True)
    return sorted(
        os.path.join(root, rel) for rel in out.stdout.split("\0")
        if rel and not rel.startswith(_SCAN_EXCLUDE)
    )

# ⚠ A markdown inline link, and NOTHING ELSE. `[[memory-slug]]` carries no `](` and so cannot
# match; reference-style links (`[x][ref]`) are not used in this tree and are not handled.
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)")

# Not our business: the public internet, mail, in-page anchors, and absolute file URLs.
#
# ⚠ `file://` IS HERE BECAUSE IT WAS MEASURED ELSEWHERE, not on principle. Prosodia's
# `notes/out-of-bounds-references.md` deliberately records `file:///Users/...` paths from a
# Mac; run against that tree, this gate reported five of them as dead links. Sonora carries
# none today, which is exactly why it belongs in the list now — a scheme is not a relative
# path, and the first one written here would otherwise arrive as a false failure on a gate
# nobody was expecting to argue with.
EXTERNAL = ("http://", "https://", "mailto:", "file://", "#")

# Sibling checkouts whose links point INTO this repo. Colon-separated, `~` allowed, relative
# paths resolve from the repo root — the same shape as `SIBLING_REPO_CANDIDATES` in
# workflow/config.env, which already solved this problem for the reviewer.
# ⚠ CANDIDATES, NOT REQUIREMENTS. Each one that is absent is printed and skipped.
SIBLING_ENV = "SONORA_SIBLING_REPOS"
SIBLING_DEFAULTS = (
    "../Prosodia",
    "~/Projects/Artificial-Humanity/Prosodia",
    "../AI-Lab-AMD",
    "~/Projects/Artificial-Humanity/AI-Lab-AMD",
)

# What a sibling's link into this repo looks like once the checkout-dependent prefix is
# discarded: everything from the last `notes/` or `docs/` segment onward.
INBOUND = re.compile(r"/Sonora/.*?/((?:notes|docs)/[^)\s]+\.md)")


# ⚠ A `§N` CITATION IS NOT A LINK, AND THAT IS THE WHOLE PROBLEM. Every one of the 19 this
# check was written for is spelled `[todo.md §6](todo.md)` — the LINK TARGET IS THE FILE and
# the section lives only in the label, so the link resolves, nothing renders red, and the
# reader lands on the right document to hunt for a section that is not there. The existing
# half of this gate says so in its own "what it does not cover" list: `#anchor` fragments are
# stripped and not verified. This is the same blind spot approached from the label side.
#
# Measured 2026-08-20/21 (#181, #185): 19 dangling citations across 13 files. Six named a
# `todo.md` section that has never existed; NINE named `model-decisions.md §5` in a file with
# ZERO numbered headings; two named a `todo.md` section that exists but is the wrong one. Two
# of the six were in SHIPPED director skill files under `scripts/assets/`.
SECTION_CITE = re.compile(r"([A-Za-z0-9_.-]+\.md)\s*§+\s*(\d+)")
# ⚠ THE `§` IS OPTIONAL IN THE HEADING TOO. `docs/vat-channels.md` writes `## §1 Mechanism`,
# and requiring the digit to follow the hashes directly made that file read as having NO
# numbered sections — so a CORRECT citation to its §1 would have been failed, with the remedy
# "cite by name" (#272). A guard that is wrong about the target is worse than one that is
# silent about it: this one hands the reader an incorrect instruction.
NUMBERED_HEADING = re.compile(r"^#+\s+§?\s*(\d+)[.)]?\s", re.M)


def section_citations(root=REPO):
    """-> (dangling, examined, skipped) for the `<file>.md §N` citations in this repo.

    `dangling` is [(rel, lineno, target, n, available)] — the failures. `examined` counts the
    citations actually CHECKED. `skipped` is [(rel, lineno, target, n)] for citations whose
    target file is outside the scanned set, which nothing checks (#255).

    ⚠ **THE THREE ARE RETURNED TOGETHER ON PURPOSE.** An empty `dangling` means "nothing
    dangles" and "there was nothing to read" identically, and the caller cannot tell them
    apart from the failures alone. This gate already treats an absent sibling as a printed
    skip rather than a pass; a citation nobody checked deserves the same.

    ⚠ **CROSS-REPO `§N` IS DELIBERATELY NOT CHECKED, and that is a scope call, not an
    oversight.** Four of the five skipped citations point into Prosodia. Resolving them would
    mean walking a sibling checkout for a bare basename and failing on ANOTHER repo's heading
    numbers — and the sibling is absent on most machines, so the common case would be a skip
    anyway. It is now REPORTED instead, which is what #255 identified as missing. If the owner
    wants it enforced, it is a follow-up with its own design, not a widened constant.

    Basenames are resolved as a UNION when several files share one (`README.md` exists in both
    `notes/` and `docs/`). Deliberately permissive: this should fail on a section that exists
    NOWHERE under that name, never on the ambiguity of which copy was meant.
    """
    files = citation_sources(root)
    # ⚠ TARGETS ARE MARKDOWN; SOURCES ARE WIDER. A `.py` file can CONTAIN a citation but can
    # never BE one's target, so the heading map is built from `repo_markdown()` alone.
    by_name = {}
    for path in repo_markdown(root):
        by_name.setdefault(os.path.basename(path), []).append(path)

    headings = {}
    for name, paths in by_name.items():
        nums = set()
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                nums |= set(NUMBERED_HEADING.findall(fh.read()))
        headings[name] = nums

    out, examined, skipped = [], 0, []
    for path in files:
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                for target, n in SECTION_CITE.findall(line):
                    # ⚠ A TARGET OUTSIDE THIS SET IS NOT CHECKED BY ANYTHING, AND SAYING SO
                    # IS THE POINT. This used to `continue` under a comment claiming the LINK
                    # half would catch it. Measured (#255): 5 of 38 live citations land here,
                    # and FOUR are Prosodia targets whose links RESOLVE — so the link half is
                    # green, this half defers to it, and nobody looks at the section at all.
                    # The hand-off is only real when the file is MISSING. Now counted and
                    # printed, so a green run cannot be read as full coverage.
                    if target not in headings:
                        skipped.append((rel, lineno, target, n))
                        continue
                    examined += 1
                    if n not in headings[target]:
                        out.append((rel, lineno, target, n, sorted(headings[target], key=int)))
    return out, examined, skipped


def targets(text):
    """-> [(raw target, path with any #anchor stripped)] for the checkable links on a line."""
    out = []
    for raw in LINK.findall(text):
        if raw.startswith(EXTERNAL):
            continue
        path = raw.split("#", 1)[0]
        if not path:            # a bare `#anchor` that survived the prefix test
            continue
        out.append((raw, path))
    return out


def classify(containing_file, target, root=REPO):
    """Where a link points. -> ("in", abspath) | ("cross", sibling name, tail) | ("skip", ...)

    ⚠ THE SPLIT IS ON WHETHER THE LINK LEAVES THE REPO, not on how many `../` it has. A link
    that stays inside is resolvable here and is checked here. A link that escapes names
    ANOTHER repository, and where that repository sits on any given machine is not something
    this tree can know — so the path is discarded and the name is kept.
    """
    resolved = os.path.normpath(os.path.join(os.path.dirname(containing_file), target))
    if resolved == root or resolved.startswith(root + os.sep):
        return ("in", resolved, "")
    parts = [p for p in target.split("/") if p not in ("..", ".", "")]
    if len(parts) < 2:
        # It escapes the repo and names no sibling directory — nothing to check against.
        return ("skip", target, "")
    return ("cross", parts[0], "/".join(parts[1:]))


def dangling(root=REPO):
    """In-repo relative links that do not resolve. -> [(rel, lineno, target)]

    Cross-repo links are deliberately NOT here; `outbound_dangling` handles them by name.
    """
    bad = []
    for path in repo_markdown(root):
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                for raw, target in targets(line):
                    kind, a, _b = classify(path, target, root)
                    if kind == "in" and not os.path.exists(a):
                        bad.append((rel, lineno, raw))
    return bad


def outbound(root=REPO):
    """This repo's links INTO other repos, grouped by the sibling they name.

    -> {sibling name lowered: [(rel, lineno, tail, raw)]}
    """
    out = {}
    for path in repo_markdown(root):
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                for raw, target in targets(line):
                    kind, name, tail = classify(path, target, root)
                    if kind == "cross":
                        out.setdefault(name.lower(), []).append((rel, lineno, tail, raw))
    return out


def outbound_dangling(sibling_root, links):
    """Which of `links` the sibling on disk cannot satisfy.

    ⚠ The TAIL is compared exactly — case included. `Prosodia/Docs/x.md` against a checkout
    whose directory is `docs/` is a real dead link on a case-sensitive filesystem, and the
    only reason to be lenient here would be to make a red gate green.
    """
    return [(rel, lineno, tail, raw) for rel, lineno, tail, raw in links
            if not os.path.exists(os.path.join(sibling_root, tail))]


def sibling_paths():
    """-> [(candidate as written, absolute path or None)] — absent ones kept, to be printed."""
    raw = os.environ.get(SIBLING_ENV)
    cands = raw.split(":") if raw else list(SIBLING_DEFAULTS)
    out = []
    for c in cands:
        if not c.strip():
            continue
        p = os.path.expanduser(c.strip())
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(REPO, p))
        out.append((c.strip(), p if os.path.isdir(p) else None))
    return out


def inbound_dangling(sibling_root, root=REPO):
    """A sibling's links INTO this repo that this repo can no longer satisfy.

    Compares the repo-relative TAIL only — see the module docstring on why the prefix is not
    checkable. Scans the sibling's whole tree rather than its prose directories, because its
    layout is its own business and may not match ours.
    """
    bad = []
    for dirpath, dirnames, names in os.walk(sibling_root):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "target", "node_modules", ".venv")]
        for name in sorted(names):
            if not name.endswith(".md"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    lines = fh.readlines()
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(lines, 1):
                for tail in INBOUND.findall(line):
                    if not os.path.exists(os.path.join(root, tail)):
                        bad.append((os.path.relpath(path, sibling_root), lineno, tail))
    return bad


def main():
    # ⚠ (KIND, text). THREE UNRELATED FAILURES SHARED ONE HEADER AND ONE REMEDY (#259): a
    # dead link, a `§N` whose link RESOLVES, and a coverage floor where nothing in the docs is
    # broken at all. All three printed as "N dead link(s)" under "fix the link or restore the
    # file" — wrong for the second, meaningless for the third. Each kind now carries its own
    # count and its own remedy.
    failures = []

    scanned = repo_markdown(REPO)
    bad = dangling()
    print(f"checked the relative links in {len(scanned)} markdown files")
    for rel, lineno, raw in bad:
        failures.append(("link", f"{rel}:{lineno} — link target does not exist: {raw}"))

    stale, examined, skipped = section_citations()
    cited_files = citation_sources(REPO)
    print(f"checked {examined} `<file>.md §N` citation(s) in {len(cited_files)} markdown "
          f"files (.md + .py; the link half above is .md only)")

    # ⚠ PRINTED, NEVER SILENT — the same treatment an absent sibling gets above. Grouped by
    # target so five lines do not read as five unrelated problems.
    if skipped:
        by_target = {}
        for rel, lineno, target, n in skipped:
            by_target.setdefault(target, []).append(f"{rel}:{lineno} §{n}")
        print(f"  ~ {len(skipped)} citation(s) NOT CHECKED — target outside the scanned set, "
              f"so nothing verifies the section:")
        for target in sorted(by_target):
            print(f"       {target}: {', '.join(sorted(by_target[target]))}")

    # ⚠⚠ ZERO CITATIONS IS NOT A FAILURE, AND THIS GATE CANNOT ADJUDICATE IT.
    #
    # This has now been wrong twice in the same place. First the floor counted FILES, so
    # neutering SECTION_CITE left rc 0 (#255). Then it counted CITATIONS with a live-repo
    # magnitude (`< 25`), which failed every small tree (#267). The rewrite moved the
    # threshold to 1 — and **1 is still a magnitude** (#269): a markdown tree with clean docs
    # and no `§N` citation at all is a legitimate state, and the gate called it a breach,
    # with a remedy telling the reader to suspect a regex that is working.
    #
    # The gate genuinely CANNOT distinguish "this tree has no citations" from "SECTION_CITE
    # is broken" — both produce zero, and no portable threshold separates them. So it says
    # so, and does not fail. The repo-specific ratchet that CAN tell (because it knows this
    # tree has 37) lives in the suite: `test_the_live_citation_count_holds`. Printed, never
    # silent, exactly as an absent sibling is treated.
    if cited_files and not examined:
        # ⚠ TWO DIFFERENT ZEROS. "Found none" and "found some and skipped every one" have
        # different causes and different remedies, and the first version of this notice gave
        # the regex diagnosis to both — printing "Either this tree has none, or SECTION_CITE
        # is broken" three lines below the list of citations it had just found (#273).
        if skipped:
            print(f"  ~ 0 of {len(skipped)} citation(s) could be checked: every target is "
                  f"outside the scanned\n    set (listed above). SECTION_CITE is working — "
                  f"the targets are the problem.")
        else:
            print(f"  ~ 0 citations FOUND across {len(cited_files)} file(s). Either this tree "
                  f"has none, or\n    SECTION_CITE is broken — THIS GATE CANNOT TELL. The "
                  f"ratchet that can is tests/test_doc_links_gate.py.")
    for rel, lineno, target, n, available in stale:
        have = ", ".join(f"§{x}" for x in available) if available else "NO numbered headings"
        failures.append(("section",
            f"{rel}:{lineno} — cites {target} §{n}, which does not exist ({target} has: {have})\n"
            f"      ⚠ The LINK still resolves — the target is the file and the section is only "
            f"the label,\n      so nothing goes red and the reader lands on the right document "
            f"to hunt for a\n      section that is not there. Retarget it by NAME "
            f"(`§ Some Heading`) if the file is unnumbered."))

    # ⚠ PRINTED, NEVER SILENT, exactly as the doc-claims gate treats an absent corpus. A
    # sibling that is not checked out reduces coverage, and a coverage reduction nobody can
    # see is how a gate stops being one.
    out = outbound()
    found = {}
    for written, path in sibling_paths():
        if path is None:
            continue
        found.setdefault(os.path.basename(path).lower(), path)

    for name in sorted(out):
        path = found.get(name)
        n = len(out[name])
        if path is None:
            # ⚠⚠ ONE `../` TOO MANY LOOKS EXACTLY LIKE A LINK TO A SIBLING REPO, and until
            # 2026-08-21 it was skipped as one. `../../../notes/todo.md` from a two-deep file
            # leaves the repo, so `classify()` reads `notes` as a SIBLING NAME; no checkout
            # called `notes` exists, so it printed "NOT CHECKED" and the run stayed green.
            # That is how the 7 dead links in `scripts/teacher_audition/README.md` survived —
            # and widening the file set (#261) did NOT catch them, because the file set was
            # never the reason they were invisible. Measured: re-adding the depth bug after
            # the widening still produced a clean PASS.
            #
            # The tell is precise: the named sibling is absent AND the tail resolves inside
            # THIS repo. That is not a cross-repo link at all; it is a local link written at
            # the wrong depth, and it is reported as its own kind with its own remedy.
            # ⚠ THE SIBLING NAME IS THE FIRST PATH COMPONENT, so the local candidate is
            # `<name>/<tail>`, not `<tail>`. My first attempt checked `REPO/tail` — i.e.
            # `todo.md` at the repo root — found nothing, and reported a clean PASS on the
            # very mutation it was written for. Matched case-insensitively against real
            # top-level entries because `classify` lowercases the sibling name.
            # ⚠⚠ THE TELL IS THE NAME COLLIDING WITH A LOCAL DIRECTORY — NOT THE TAIL
            # RESOLVING. Requiring `<name>/<tail>` to exist caught 6 of 7 and missed the one
            # link this whole thread started from (#265): line 73 was at the wrong depth AND
            # pointed at a file that had moved to `docs/`, so the tail did not resolve and it
            # fell through to a "no checkout of it on this machine" skip — telling the reader
            # to check out a repo that is this repo's own directory.
            #
            # A "sibling" with no checkout whose name IS a directory here is a depth error
            # whatever the tail does. Whether the tail then resolves only changes the message:
            # one problem or two. Measured false-positive risk: with siblings forced absent,
            # 0 of 36 real cross-repo links are flagged — real sibling names (`Prosodia`,
            # `AI-Lab-AMD`) are not directories in this repo.
            # ⚠ TRACKED DIRECTORIES, AND `isdir` — `os.listdir` alone reads the WORKING TREE,
            # so an untracked directory that happens to share a sibling's name flipped a
            # printed skip into a hard FAIL, and a top-level FILE named like a sibling was
            # reported as "IS a directory in this repo" (#271). The tell is about the repo's
            # own shape, which is an index question.
            local_dir = next(
                (d for d in _tracked_top_dirs()
                 if d.lower() == name and os.path.isdir(os.path.join(REPO, d))), None)
            depth_bugs = [] if local_dir is None else list(out[name])
            for rel, lineno, tail, raw in depth_bugs:
                failures.append(("depth", f"{rel}:{lineno} — {raw}\n"
                                 f"      leaves the repo and names a sibling '{name}' that is "
                                 f"not checked out,\n      but '{local_dir}' IS a directory in "
                                 f"this repo. One `../` too many."
                                 + ("" if os.path.exists(os.path.join(REPO, local_dir, tail))
                                    else f"\n      ⚠ AND `{local_dir}/{tail}` does not exist "
                                         f"either — the target moved or was deleted, so this "
                                         f"link\n      needs BOTH the depth and the path "
                                         f"fixed.")))
            remaining = n - len(depth_bugs)
            if remaining:
                print(f"  ~ {remaining} outbound link(s) to '{name}' NOT CHECKED: no checkout "
                      f"of it on this machine")
            continue
        bad_out = outbound_dangling(path, out[name])
        print(f"  -> {name}: {n} outbound link(s), {len(bad_out)} unresolved")
        for rel, lineno, tail, raw in bad_out:
            failures.append(("link", f"{rel}:{lineno} — {raw}\n"
                             f"      '{name}' is checked out here and has no {tail}"))

    # ⚠ INBOUND IS REPORTED, NOT FAILED, AND THE ASYMMETRY IS DELIBERATE.
    # An outbound dead link is THIS repo's defect and a commit here fixes it. An inbound one
    # is a defect in the SIBLING'S files: no commit in this repo can turn it green. Failing on
    # it would block work here on another repo's state and produce exactly what the doc-claims
    # gate warns about at length — "a check nobody can turn green is a check everybody learns
    # to ignore, and it goes on ignoring the fork it was built to catch". So it is loud, it is
    # counted, and it does not fail.
    reported = 0
    for name, path in sorted(found.items()):
        inbound = inbound_dangling(path)
        reported += len(inbound)
        print(f"  <- {name}: {len(inbound)} link(s) INTO this repo do not resolve "
              f"(reported, not failed — they are {name}'s files to fix)")
        for rel, lineno, tail in inbound:
            print(f"       {rel}:{lineno} -> {tail}")
    if not found:
        print("  ~ no sibling checkout was found; the cross-repo half of this gate did NOT run")

    # Each kind gets its own heading and its own remedy. ⚠ The remedy is the part that was
    # actually wrong before: "fix the link or restore the file" is the OPPOSITE of what a
    # dangling `§N` needs (the link resolves — the label is wrong), and it names no action at
    # all for a coverage breach, where the docs are fine and the SCAN went blind.
    KINDS = [
        ("link", "dead link(s) — the target file does not exist",
         "A broken markdown link does not error, it renders. Fix the link or restore the "
         "file —\nand if the file was deliberately deleted, the citing line is what needs "
         "rewriting."),
        ("section", "dangling `§N` citation(s) — the LINK RESOLVES, the section does not",
         "Do NOT 'fix the link': it already works. Retarget the label — `§ Some Heading` if "
         "the\ntarget file has no numbered sections, or the section it actually meant."),
        ("depth", "link(s) written at the WRONG DEPTH — they escape the repo",
         "These are not cross-repo links. Count the `../` against the file's own depth: a\n"
         "file two directories deep reaches the repo root with `../../`, not `../../../`."),
    ]
    if failures:
        for kind, heading, remedy in KINDS:
            items = [text for k, text in failures if k == kind]
            if not items:
                continue
            print(f"\nFAIL — {len(items)} {heading}:\n")
            for text in items:
                print(f"  {text}")
            print(f"\n{remedy}")
        # ⚠ A kind added above and forgotten here would print nothing and still return 1 —
        # a failure with no message. ⚠⚠ THE FIRST VERSION OF THIS CATCH PRINTED THE KIND NAME
        # AND SWALLOWED THE TEXT (#263), under a comment claiming the case was handled: the
        # RUN got a message, the FAILURE still had none, and the reader learned a label
        # instead of what broke. It also counted KINDS, so forty unrouted failures read as
        # "1". Both fixed — the count is of failures, and every text is printed.
        routed = {k for k, _h, _r in KINDS}
        unrouted = [(k, text) for k, text in failures if k not in routed]
        if unrouted:
            kinds = sorted({k for k, _ in unrouted})
            print(f"\nFAIL — {len(unrouted)} failure(s) whose kind has no heading "
                  f"({', '.join(kinds)}). Add each to KINDS with its own remedy:\n")
            for kind, text in unrouted:
                print(f"  [{kind}] {text}")
        return 1

    print("\nPASS — every relative link this repo owns resolves.")
    # ⚠ THE NUMBERS ARE INTERPOLATED, NOT WRITTEN DOWN. This banner said "except
    # `<file>.md §N`, whose NUMBER is checked" — unconditional, and true of 33 of 38 (#255).
    # A banner asserting coverage the run did not have is AGENTS.md §5b's subject, and a
    # hardcoded count here would drift the way every count on this branch already has.
    print("⚠ Existence only: `#anchor` fragments are stripped and not verified, and external "
          "URLs\n  are not fetched. A green run does not mean a citation points at the right "
          "section.")
    print(f"⚠ `<file>.md §N` is the one exception and it is PARTIAL: {examined} of "
          f"{examined + len(skipped)} citation(s) had their\n  number checked against the "
          f"target's headings (#181, #185). The other {len(skipped)} name a file outside\n"
          f"  the scanned set and are listed above — nothing verifies those. A citation by "
          f"NAME is\n  never verified, and cross-repo `§N` is not checked at all.")
    if reported:
        print(f"⚠ …and {reported} link(s) INTO this repo are dead. They are not this repo's "
              f"to fix,\n  so they do not fail — but nothing else in either tree is looking "
              f"at them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
