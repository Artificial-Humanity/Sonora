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
* **Relative links only.** `http(s)://` and `mailto:` are not checked; a 404 on the public
  internet is not this gate's business and would make it fail offline.
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
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# The prose directories, and the root anchors. `workflow/` is included because the lane is
# copied wholesale into other repos and a dead link travels with it.
PROSE_DIRS = ("notes", "docs", "workflow")
ROOT_DOCS = ("README.md", "AGENTS.md", "CLAUDE.md")

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


def markdown_files(root, dirs=PROSE_DIRS, roots=ROOT_DOCS):
    """Every markdown file this gate reads, as absolute paths."""
    out = []
    for d in dirs:
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        for dirpath, _dirnames, names in os.walk(full):
            out += [os.path.join(dirpath, n) for n in sorted(names) if n.endswith(".md")]
    out += [os.path.join(root, n) for n in roots if os.path.isfile(os.path.join(root, n))]
    return sorted(out)


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
    for path in markdown_files(root):
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
    for path in markdown_files(root):
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
    failures = []

    scanned = markdown_files(REPO)
    bad = dangling()
    print(f"checked the relative links in {len(scanned)} markdown files")
    for rel, lineno, raw in bad:
        failures.append(f"{rel}:{lineno} — link target does not exist: {raw}")

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
            print(f"  ~ {n} outbound link(s) to '{name}' NOT CHECKED: no checkout of it "
                  f"on this machine")
            continue
        bad_out = outbound_dangling(path, out[name])
        print(f"  -> {name}: {n} outbound link(s), {len(bad_out)} unresolved")
        for rel, lineno, tail, raw in bad_out:
            failures.append(f"{rel}:{lineno} — {raw}\n"
                            f"      '{name}' is checked out here and has no {tail}")

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

    if failures:
        print(f"\nFAIL — {len(failures)} dead link(s):\n")
        for f in failures:
            print(f"  {f}")
        print("\nA broken markdown link does not error, it renders. Fix the link or restore "
              "the file —\nand if the file was deliberately deleted, the citing line is what "
              "needs rewriting.")
        return 1

    print("\nPASS — every relative link this repo owns resolves.")
    print("⚠ Existence only: `#anchor` fragments are stripped and not verified, and external "
          "URLs\n  are not fetched. A green run does not mean a citation points at the right "
          "section.")
    if reported:
        print(f"⚠ …and {reported} link(s) INTO this repo are dead. They are not this repo's "
              f"to fix,\n  so they do not fail — but nothing else in either tree is looking "
              f"at them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
