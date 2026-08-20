#!/usr/bin/env python3
"""The merge floor's one decision: does this issue ride, or does it block?

⚠ THIS EXISTS AS A MODULE SO IT CAN BE TESTED WITHOUT THE TRACKER. The rule was first
proven with live fixture issues filed into PocketBase, and that proof is not repeatable:
`escalated` is a one-way door that only the owner's `user_decision` clears, so a fixture
escalated to prove "escalated blocks" cannot be cleaned up by the agent that filed it. It
had to be deleted server-side along with its comments. A rule this small should not need a
write to a shared service to check.

⚠ IT IS ALSO THE ONE COPY. `merge_branch.sh` imports it rather than restating it; a floor
implemented twice is a floor that disagrees with itself on the day it matters.

The rule (owner, ratified 2026-08-19; built 2026-08-20):

    MEDIUM and above blocks. LOW rides to a follow-up branch.

with two additions that the ruling implies but does not say, both fail-closed:

  * UNGRADED blocks. Every issue filed before 2026-08-20 has `severity=""` — 111 of them at
    the time this was written — and a floor that waves through what it cannot grade is not a
    floor. An unrecognised value blocks too, so widening the field in PocketBase cannot
    silently widen what merges.
  * ESCALATED blocks at ANY severity. Severity says how bad a finding is; `escalated` says a
    human is waiting. They are different questions and the second one is not the floor's to
    overrule.

⚠ BOTH INPUTS ARE WHITELISTS, AND THAT IS THE WHOLE DESIGN. The first version blacklisted
`escalated` and whitelisted severity, so an unrecognised SEVERITY blocked (right) while an
unrecognised STATE rode (wrong) — `rides("frobnicated", "low")` returned True. That inverted
the fail-closed property `merge_branch.sh` claims for its own filter, which says in terms that
a fifth state must block "until someone decides what it means". Caught in review as #204. A
guard with one blacklist in it has a hole shaped like the future.
"""

# Only `low` rides. Everything else — including a value added to the PocketBase field later
# and never taught to this file — falls through to blocking, which is the safe direction.
RIDES = frozenset({"low"})

# ⚠ AND ONLY THESE STATES. `closed` never reaches here (the query excludes it) and `escalated`
# must block, so these two are the whole rideable set. A state added to the PocketBase field
# later is unknown to this file and must block until someone teaches it — same reasoning as
# RIDES, and the reason this is a set rather than a `!= "escalated"`.
RIDEABLE_STATES = frozenset({"open", "review"})


def rides(state, severity):
    """True if this unclosed issue may ride past the floor and let the branch land."""
    if (state or "").strip().lower() not in RIDEABLE_STATES:
        return False
    return (severity or "").strip().lower() in RIDES
