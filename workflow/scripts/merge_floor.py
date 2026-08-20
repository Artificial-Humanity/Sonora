#!/usr/bin/env python3
"""The merge floor's one decision: does this issue ride, or does it block?

⚠⚠ THE THRESHOLD IS NOT SET HERE. It is `MERGE_SEVERITY_FLOOR` in `workflow/config.env`, and
that is the only place it may be set (owner, 2026-08-20). This module reads it.

WHY. The rule was previously spelled out in prose in eight places — scripts, personas, the
workflow map, a manifest, test docstrings — and three separate sweeps failed to find them all;
the guard written to end the sweeps then shipped green while blind to a ninth. The owner's
ruling: settle it everywhere it surfaces, AND make it reconfigurable from one place. So the
value lives in config, the code reads it, and the prose points at it instead of repeating it.

⚠ THIS MODULE EXISTS SO THE RULE CAN BE TESTED WITHOUT THE TRACKER. The first proof was live
fixture issues, and that proof is not repeatable: `escalated` is a one-way door only the
owner's `user_decision` clears, so a fixture escalated to prove "escalated blocks" cannot be
cleaned up by the agent that filed it. It had to be deleted server-side with its comments.

⚠ AND IT IS THE ONE COPY OF THE LOGIC. `merge_branch.sh` imports it rather than restating it;
a floor implemented twice disagrees with itself on the day it matters.

Everything below fails CLOSED, deliberately:

  * an UNGRADED issue blocks. Every issue filed before 2026-08-20 has `severity=""`, and a
    floor that waves through what it cannot grade is not a floor;
  * an UNRECOGNISED severity blocks, so widening the PocketBase field cannot silently widen
    what merges;
  * an UNRECOGNISED or MISSING floor setting blocks EVERYTHING. There is no "off": a typo in
    config must not open the gate;
  * an UNKNOWN STATE blocks. Both inputs are whitelists — an earlier version blacklisted only
    `escalated`, so `rides("frobnicated", "low")` was True (#204). A guard with one blacklist
    in it has a hole shaped like the future;
  * ESCALATED blocks at ANY severity. Severity says how bad a finding is; `escalated` says a
    human is waiting, and the floor does not get to overrule the second with the first.
"""
import os

# Ordered weakest → strongest. Position IS the comparison, so do not reorder; add new values
# at the correct rank and they become comparable everywhere at once.
LADDER = ("low", "medium", "high", "critical")

# `closed` never reaches the floor (the tracker query excludes it) and `escalated` must block,
# so these two are the whole rideable set.
RIDEABLE_STATES = frozenset({"open", "review"})

_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "config.env")


def floor_setting(path=None):
    """`MERGE_SEVERITY_FLOOR` as written in config.env — raw, unvalidated, "" if absent.

    Parsed rather than sourced: config.env is documented as plain KEY=value that both the
    shell and `issue.py` read, and importing a shell to read one value would make this module
    depend on a shell being present to answer a question about a string.
    """
    try:
        with open(path or _CONFIG, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("MERGE_SEVERITY_FLOOR="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        return ""
    return ""


# ⚠ A DISTINCT SENTINEL, NOT `None`. With `floor=None` meaning "read config", a caller doing
# `rides(s, sev, floor=cfg.get("MERGE_SEVERITY_FLOOR"))` on a config MISSING that key passes
# None and silently gets the configured floor instead of a refusal — a fail-open path reached
# by ordinary-looking code. `None` is now an unusable value like any other, and only omitting
# the argument reads config. Caught by this module's own tests.
_READ_CONFIG = object()


def rides(state, severity, floor=_READ_CONFIG):
    """True if this unclosed issue may ride past the floor and let the branch land."""
    raw = floor_setting() if floor is _READ_CONFIG else floor
    setting = (raw or "").strip().lower()
    if setting not in LADDER:
        # ⚠ Unset, misspelled, or a value this ladder has never heard of. Block everything —
        # the alternative is a config typo quietly turning the gate off.
        return False
    if (state or "").strip().lower() not in RIDEABLE_STATES:
        return False
    sev = (severity or "").strip().lower()
    if sev not in LADDER:
        return False
    return LADDER.index(sev) < LADDER.index(setting)
