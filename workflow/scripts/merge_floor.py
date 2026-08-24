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
  * a HALTED state blocks at ANY severity. Severity says how bad a finding is; halted says a
    human is waiting, and the floor does not get to overrule the second with the first.
    ⚠ WHICH states are halted or terminal is the LANE DEFINITION's fact, not this module's
    (phase 2): `rideable_states()` derives the whitelist from workflow/sonora-lane.json, and
    an unreadable definition blocks everything.
"""
import os

# Ordered weakest → strongest. Position IS the comparison, so do not reorder; add new values
# at the correct rank and they become comparable everywhere at once.
LADDER = ("low", "medium", "high", "critical")

_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "config.env")
_DEFINITION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "sonora-lane.json")


def rideable_states(path=None):
    """States an unclosed issue may RIDE in: declared in the lane definition, and neither
    `halted` nor `terminal` there.

    ⚠ DERIVED, NOT DECLARED (phase 2, 2026-08-24). This was `frozenset({"open", "review"})`
    beside a comment explaining why `escalated` and `closed` are excluded — a second copy of
    state semantics the definition already carries, which would disagree with it the day a
    state is added. workflow/sonora-lane.json is the one copy; this module reads it.

    ⚠ FAILS CLOSED like everything else here: a missing, unreadable or malformed definition
    returns the EMPTY set, so every state blocks — a gate that cannot read its own rules
    must not guess them.
    """
    import json
    try:
        with open(path or _DEFINITION, encoding="utf-8") as fh:
            d = json.load(fh)
        states = set(d.get("states") or [])
        blocked = set(d.get("halted") or []) | set(d.get("terminal") or [])
        return frozenset((s or "").strip().lower() for s in states - blocked)
    except (OSError, ValueError, TypeError, AttributeError):
        return frozenset()


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


def rides(state, severity, floor=_READ_CONFIG, states=_READ_CONFIG):
    """True if this unclosed issue may ride past the floor and let the branch land."""
    raw = floor_setting() if floor is _READ_CONFIG else floor
    setting = (raw or "").strip().lower()
    if setting not in LADDER:
        # ⚠ Unset, misspelled, or a value this ladder has never heard of. Block everything —
        # the alternative is a config typo quietly turning the gate off.
        return False
    allowed = rideable_states() if states is _READ_CONFIG else states
    if (state or "").strip().lower() not in allowed:
        return False
    sev = (severity or "").strip().lower()
    if sev not in LADDER:
        return False
    return LADDER.index(sev) < LADDER.index(setting)
