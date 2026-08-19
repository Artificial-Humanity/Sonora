"""`.gitignore` must not eat a source directory — enforced, not asserted.

WHY THIS FILE EXISTS (2026-08-18, issue #98)
--------------------------------------------
The root `.gitignore` came from the standard Python template, where `lib/` means the top-level
setuptools build directory. Unanchored, it also matches `scripts/lib/` — a first-class source
directory — so every NEW file added there was skipped by `git add` **with no message and no
entry in `git status`**. Measured twice on 2026-08-17: `scripts/lib/find_changeset.py` existed
in no commit at all, and `scripts/lib/schemas.py` was invisible to a guard that enumerates via
`git ls-files`. Existing files stayed tracked, which is what kept it hidden — the rule only
ever ate the *next* one.

⚠ **THE FIRST FIX ANCHORED TWO ENTRIES AND CLAIMED TO HAVE ANCHORED ALL OF THEM.** Its comment
read "EVERY DIRECTORY NAME FROM THAT TEMPLATE IS NOW ANCHORED" while `dist/` sat two lines
above `/build/`, still bare — a false claim written into a comment while fixing an issue about
a false claim, and it survived a full review pass because `git check-ignore` was not on the
reviewer's allowlist. **That is what this file replaces: a sentence nobody could falsify.**
"""

import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GITIGNORE = os.path.join(REPO, ".gitignore")

# ⚠ RECURSIVE BY DESIGN. These SHOULD match at any depth, and anchoring them is the opposite
# error. The list is explicit so that adding to it is a decision rather than a slip — exactly
# the shape the doc-claims gate uses for its exemptions.
RECURSIVE_BY_DESIGN = {
    "__pycache__/",
    "gradio_cached_examples/", "synth_output/",
    "model_tflite/", "model_e2e_tflite/", "model_e2e_test_tflite/",
    # Downloaded INTO build_matcha.py's work directory (scripts/litert_export/
    # build_matcha.py:66), not to the repo root, so anchoring them would stop them
    # matching where they actually land.
    "generator_v1", "g_02500000",
}

# The source directories that must never be swallowed, whatever the pattern list says.
#
# ⚠ NOT ONLY THE PYTHON PACKAGES (issue #106). `scripts/assets` holds `register_lexicon.json`
# — which this branch's whole schema layer loads — and `configs/` holds the training YAML.
# An ignore rule that swallowed a tracked ASSET rather than a module was invisible to the
# older list, and the probe below only ever asked about a `.py` file, so it was invisible
# twice over. The bucket ROOTS are here too: an unanchored pattern matching `scripts` or
# `matcha` outright is the worst case of all and was not probed.
SOURCE_DIRS = ["scripts", "scripts/lib", "scripts/gates", "scripts/tools", "scripts/stages",
               "scripts/assets", "scripts/assets/director_skills", "configs",
               "matcha", "matcha/models", "matcha/utils", "tests", "workflow",
               "workflow/scripts", "audition/app", "notes"]

# ⚠ ONE PROBE PER EXTENSION, because a rule can be extension-specific. `*.onnx`, `*.cover`
# and `*.py,cover` are all live in this .gitignore, so "a new .py file is safe here" says
# nothing about a new .json asset in the directory that holds the register lexicon.
PROBE_NAMES = ["__probe_new_file__.py", "__probe_new_file__.json",
               "__probe_new_file__.md", "__probe_new_file__.yaml"]


def _entries():
    """Every unanchored pattern in .gitignore that can match at depth, with its line number.

    ⚠ THE TRAILING SLASH IS NOT WHAT MAKES A PATTERN DANGEROUS (issue #106). Until
    2026-08-18 this collected only `t.endswith("/")`, which discarded every bare name before
    any check ran — and a bare name is the WIDER form: `logs` matches a directory named
    `logs` at any depth AND a file named `logs` at any depth, where `logs/` matches only the
    directory. The commit that introduced this file anchored `outputs` and `logs`, both bare;
    reverting either one passed every test in it.

    A pattern containing a `/` anywhere is already anchored to this directory by git's own
    rules (`matcha/utils/monotonic_align/core.c`), so it is not collected.
    """
    out = []
    with open(GITIGNORE, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            t = line.strip()
            if not t or t.startswith("#") or t.startswith("!"):
                continue
            # ⚠ DOT-PREFIXED ENTRIES ARE TOOL METADATA AND ARE RECURSIVE ON PURPOSE.
            # `.vscode/`, `.tox/`, `.eggs/`, `.idea/` — a nested one is still editor or tool
            # state and should still be ignored. The distinction is not cosmetic: a SOURCE
            # directory never starts with a dot, so this exempts exactly the set that cannot
            # be the failure this file exists to prevent. Found by running the check, which
            # is why the rule is here rather than in a hand-written exemption list.
            if t.startswith("."):
                continue
            if t.startswith("/") or "*" in t:
                continue
            if "/" in t.rstrip("/"):      # already anchored by git: the pattern has a path
                continue
            out.append((n, t))
    return out


def test_every_bare_directory_pattern_is_anchored_or_declared_recursive():
    """The structural rule. An unanchored `foo` or `foo/` matches at any depth."""
    stray = [(n, t) for n, t in _entries() if t not in RECURSIVE_BY_DESIGN]
    assert not stray, (
        "unanchored directory pattern(s) in .gitignore — each matches at ANY depth, so a "
        "source directory of that name would be silently untrackable:\n"
        + "\n".join(f"  .gitignore:{n}  {t!r}  → write it as /{t}" for n, t in stray)
        + "\n\nIf one is recursive on purpose, add it to RECURSIVE_BY_DESIGN with a reason.")


def test_the_recursive_list_has_not_gone_stale():
    """⚠ An exemption whose entry is gone is a hiding place waiting for a new tenant.

    The doc-claims gate makes the same argument about its own exemptions, and for the same
    reason: a list nobody prunes stops describing the file it exempts from.
    """
    present = {t for _n, t in _entries()}
    dead = RECURSIVE_BY_DESIGN - present
    assert not dead, (
        f".gitignore no longer contains {sorted(dead)}, so exempting them exempts nothing — "
        f"remove them from RECURSIVE_BY_DESIGN")


def test_no_source_directory_is_ignored():
    """⚠ THE BEHAVIOURAL CHECK, and the one the pattern rule above cannot replace.

    `git check-ignore` asks git itself, so it survives any pattern this repo has not thought
    of — a negation, a nested `.gitignore`, a global excludes file. It is the direct falsifier
    that was missing when the first fix's false claim went through.
    """
    probes = [os.path.join(d, name) for d in SOURCE_DIRS for name in PROBE_NAMES]
    r = subprocess.run(["git", "check-ignore", "--no-index", "-v", *probes],
                       cwd=REPO, capture_output=True, text=True)
    # exit 0 = at least one path IS ignored, which is the failure here.
    assert r.returncode != 0, (
        "a NEW file in a source directory would be silently ignored by git:\n" + r.stdout)


def test_the_probe_would_actually_detect_an_ignored_directory():
    """⚠ PROVE THE INSTRUMENT FIRES. A `check-ignore` invocation that silently stopped
    matching would make the test above pass while checking nothing — which is the exact
    failure mode this repo keeps paying for."""
    r = subprocess.run(["git", "check-ignore", "--no-index", "-v",
                        "build/__probe__.py"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and "build" in r.stdout, (
        "git check-ignore did not report /build/ as ignored — the probe is not working, so "
        "test_no_source_directory_is_ignored is not either")


def test_every_probed_directory_exists():
    """⚠ A PROBE FOR A DIRECTORY THAT IS NOT THERE PASSES AND PROVES NOTHING.

    `git check-ignore --no-index` matches patterns against a path; it does not care whether
    the path exists. So a SOURCE_DIRS entry that is renamed, or added in anticipation of a
    branch that has not landed, keeps the test green while silently covering one directory
    fewer. Caught on 2026-08-18 by listing `docs` here while it existed only on another
    branch — the same shape as `RECURSIVE_BY_DESIGN` going stale, which the test above
    already guards in the other direction.
    """
    missing = [d for d in SOURCE_DIRS if not os.path.isdir(os.path.join(REPO, d))]
    assert not missing, (
        "SOURCE_DIRS names directories that do not exist, so they are probed and prove "
        f"nothing: {missing}. Remove them, or land the branch that creates them.")
