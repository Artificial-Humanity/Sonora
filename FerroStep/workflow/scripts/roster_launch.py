"""How an agent in the roster is LAUNCHED: which model, at which effort.

⚠⚠ THE VALUES ARE NOT SET HERE. They are the `model:` and `effort:` keys on an agent's entry
in `FerroStep/config.yaml` — the FerroStep roster — and that is the only place they
may be set (owner, 2026-09-02: the reviewer runs Fable 5.1 at high). This module reads them.

WHY A READER AND NOT A DEFAULT. Until 2026-09-02 `request_review.sh` and `review_cycle.sh`
each carried `MODEL="opus"` / `EFFORT="xhigh"` — two copies of one setting, in scripts, where
a change to either leaves the other looking correct. That is the exact shape the severity
floor died of in eight places (`merge_floor.py`, and the ruling in config.env). So the value
lives beside the identity it launches, and the script reads it.

⚠ `ferrostep agent-env` DOES NOT EMIT THESE. The roster crate reads `name`, `email` and
`persona` and tolerates keys it does not know ("the file is the deployment's own" — its own
test says so), which is what makes it safe to put them there. But tolerated is not read: the
launcher has to ask this module, with the roster path `agent-env` emitted as AGENT_ROSTER.

⚠ LIMIT. That path is the roster `agent-env` READ, which for an entry inherited from a
workspace-level roster is not necessarily the file the entry CAME FROM. Sonora declares its
own reviewer, so the two are the same file here; a repo that inherits its reviewer would need
this module to walk the chain, and it does not.

⚠ WHAT WOULD END THIS MODULE, written here so it can be. FerroStep's resident may take the
same setting into the roster crate (owner, 2026-09-02: she is authorised, and who makes the
deployed change does not matter). The day `ferrostep agent-env` emits AGENT_MODEL and
AGENT_EFFORT itself, the launcher should eval those and this file should be deleted, not kept
as a second reader — check `ferrostep agent-env --agent reviewer` for the two names before
touching either side.

Everything below fails CLOSED, deliberately, in the same spirit as merge_floor.py:

  * a MISSING key is a refusal, never a built-in default — a fallback here would be the
    second copy this module exists to remove, and it would win silently on the day the
    roster's key is misspelt;
  * an UNRECOGNISED effort is a refusal, so the launcher does not hand `claude` a word it
    will reject an hour into someone's afternoon, or one it silently maps.
"""
import shlex

# The claude CLI's ladder, verbatim (`claude --help`, 2026-09-02). Position is not compared;
# membership is. Extend it when the CLI does, not before.
EFFORTS = ("low", "medium", "high", "xhigh", "max")

KEYS = ("model", "effort")


class LaunchError(ValueError):
    """The roster cannot say how to launch this agent. The message names what is missing."""


def launch_settings(roster_path, title):
    """`{"model": ..., "effort": ...}` for `title`, from the roster at `roster_path`.

    Raises LaunchError on an unreadable roster, an unknown title, a missing key, an empty
    value or an effort outside EFFORTS. Never returns a default.
    """
    import yaml
    try:
        with open(roster_path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as e:
        raise LaunchError(f"cannot read the roster at {roster_path}: {e}") from e
    agents = (doc or {}).get("agents") if isinstance(doc, dict) else None
    if not isinstance(agents, dict) or title not in agents:
        raise LaunchError(f"the roster at {roster_path} has no agent titled '{title}'")
    entry = agents[title] or {}
    out = {}
    for key in KEYS:
        value = entry.get(key) if isinstance(entry, dict) else None
        if value is None or not str(value).strip():
            raise LaunchError(
                f"agent '{title}' in {roster_path} has no `{key}:` — set it there; the "
                f"launcher has no default on purpose (see FerroStep/workflow/scripts/roster_launch.py)")
        out[key] = str(value).strip()
    if out["effort"] not in EFFORTS:
        raise LaunchError(
            f"agent '{title}' in {roster_path} has effort '{out['effort']}', which is not one "
            f"of {', '.join(EFFORTS)}")
    return out


def shell_lines(roster_path, title):
    """The same, as `AGENT_MODEL=...` / `AGENT_EFFORT=...` lines for a shell to eval.

    Quoted with shlex so a value can never become shell syntax. Mirrors the shape of
    `ferrostep agent-env`, which is what the launcher already evals.
    """
    s = launch_settings(roster_path, title)
    return "\n".join(f"AGENT_{k.upper()}={shlex.quote(v)}" for k, v in s.items()) + "\n"
