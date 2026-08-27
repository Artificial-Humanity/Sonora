"""Every lane COMMAND must be runnable from something the developer reads.

This is the inverse of `tests/test_reviewer_write_path.py`. That file asserts everything
**documented** can be **run** (#318, #321, #328 — three instructions the harness refused).
This one asserts everything **runnable** is **documented**, and it exists because the same
seam has now been found from the other side.

⚠ THE INSTANCE, AND THERE IS EXACTLY ONE — stated as one on purpose. `workflow/scripts/
review_cycle.sh` is a 469-line driver that runs the whole review loop to convergence. It was
named in three files (`CLAUDE.md`, `AGENTS.md`, `REVIEWER.md`) and **by no line addressed to
the role that runs it**: two of those are warnings about not inferring your role from the
invocation, the third tells the *reviewer* that a machine greps its summary. So the string was
present in the developer's auto-loaded context continuously, and read every time as describing
a mode it was not in. Six reviews were driven by hand in one session while it sat there.

⚠⚠ WHY THE OBVIOUS ASSERTION IS THE WRONG ONE, measured before this was written. "Named in
DEVELOPER.md" flags `full_review.sh` too — and that is a FALSE POSITIVE: `full_review.sh` has
a runnable invocation block in `WORKFLOW.md`, which `DEVELOPER.md` §3 explicitly says
"outranks this file wherever the two differ". Its affordance is real; only its *location*
differs. Shipping a guard justified by two instances, one of which was fine, is how a guard
gets switched off the first time it goes red — and then it is not there for the real one.

So the property tested is **"is the reader ever shown a line they could copy"**, not "does the
name occur" — the axis `#333` died on, where a keyword match went green on the one section
that was never wrong. A fenced code block in the developer's reading set is the closest
CHEAPLY CHECKABLE proxy for an affordance.

⚠ Its limits, stated rather than glossed: it is still a text-shape check on prose and is one
refactor of the code-fence convention away from being wrong; and it is LESS sensitive than the
name check, which is the direction that loses findings. Accepted deliberately — the population
is six enumerated files, not open-ended prose, so a miss here is one of six things and the
person adding the seventh is the one who has to notice.

Owed to FerroStep's resident, who found the instance, proposed the guard, and then argued my
stricter version down to this one with a measurement.
"""

import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
# The developer's reading set: its own persona, plus the map that outranks it (DEVELOPER.md §3).
DEV_DOCS = ["workflow/DEVELOPER.md", "workflow/WORKFLOW.md"]


def _lane_scripts():
    out = subprocess.run(["git", "ls-files", "workflow/scripts/*"],
                         cwd=REPO, capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split() if p and "__pycache__" not in p]


def _is_command(rel):
    """A COMMAND is something a person runs; a LIBRARY is something a command imports.

    ⚠ The executable bit does NOT discriminate — measured 2026-08-27, all six lane scripts are
    mode 0755, including the library. What does: `merge_floor.py` carries none of these entry
    markers and every other file carries one or two. It is imported at `merge_branch.sh` as
    the single copy of the severity-floor rule, so flagging it would be a guard going red on
    correct code — the exact failure this file's docstring is about.
    """
    text = (REPO / rel).read_text(encoding="utf-8")
    return bool(re.search(r'if __name__ == ["\']__main__|^set -euo|^usage\(\)', text, re.M))


def _fenced_blocks(rel):
    text = (REPO / rel).read_text(encoding="utf-8")
    return "\n".join(re.findall(r"```[a-z]*\n(.*?)```", text, re.S))


def test_the_enumeration_is_not_empty():
    """⚠ Floors first. If `git ls-files` stopped matching, every assertion below would pass
    over an empty list and this file would report green while checking nothing."""
    scripts = _lane_scripts()
    assert len(scripts) >= 5, (
        f"only {len(scripts)} lane scripts enumerated: {scripts}. The listing is probably "
        f"broken, not the lane.")
    commands = [s for s in scripts if _is_command(s)]
    assert len(commands) >= 4, (
        f"only {len(commands)} of {len(scripts)} classified as commands: {commands}. If the "
        f"entry-marker heuristic stopped matching, this file checks nothing.")
    # And the discriminator must still discriminate — if EVERYTHING is a command, it is not
    # separating anything and `merge_floor.py` has silently rejoined the population.
    assert len(commands) < len(scripts), (
        f"every lane script classified as a command ({commands}). `merge_floor.py` is a "
        f"library; if it now looks like a command the marker heuristic has broken open.")


def test_every_lane_command_has_a_runnable_line_in_the_developers_docs():
    """The invariant. See the module docstring for why it is fences and not names."""
    docs = {d: _fenced_blocks(d) for d in DEV_DOCS}
    assert any(docs.values()), "no fenced code blocks parsed from the developer's docs at all"

    missing = []
    for rel in _lane_scripts():
        if not _is_command(rel):
            continue
        name = pathlib.PurePosixPath(rel).name
        if not any(name in body for body in docs.values()):
            missing.append(name)

    assert not missing, (
        "lane command(s) with no runnable line anywhere the developer reads:\n"
        + "\n".join(f"  {m}" for m in missing)
        + f"\n\nSearched fenced code blocks in: {', '.join(DEV_DOCS)}.\n"
          "⚠ Being MENTIONED is not enough and is the whole point — `review_cycle.sh` was "
          "named in three files and run by nobody, because no line showed a reader a command "
          "they could copy. Add an invocation, do not add a sentence.")
