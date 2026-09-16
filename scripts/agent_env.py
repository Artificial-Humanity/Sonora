# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["pyyaml"]
# ///
"""Resolve an agent identity from `roster.yaml` and emit it for `eval` into a shell.

    AGENT_ENV="$(.venv/bin/python scripts/agent_env.py)"   # non-zero rc = refused; stop
    eval "$AGENT_ENV"
    git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" commit -m "…"

⚠⚠ THE ASSIGNMENT-THEN-EVAL SPLIT IS LOAD-BEARING, AND THIS SCRIPT IS BUILT TO MAKE IT WORK.
`eval "$(cmd)"` is status 0 even when `cmd` exits 1, because eval reports the status of the
text it ran and a refusal emits no text — `eval ""` is 0. Measured 2026-08-24. So a caller
that collapses the two steps loses every refusal below, silently, and then runs `git -c
user.name="" ...`.

That is why this script **emits NOTHING on any failure path** and exits non-zero with the
reason on stderr. If it ever grows a partial-output failure mode, the split stops protecting
anyone.

⚠ WHY THIS EXISTS AT ALL. Until 2026-09-15 identity resolved through `ferrostep agent-env`
against `FerroStep/config.yaml`. The owner is dismantling the Ferro projects to rethink them,
so the resolver came home and the roster came with it. The rule it serves is unchanged and is
not FerroStep's: **this repo's configured git identity is the OWNER's, deliberately**, so a
forgotten `-c` pair does not error — it commits an agent's work under their name, and nothing
downstream says so. `AGENTS.md` §1 and `docs/personas/DEVELOPER.md` §1 carry the check to run
afterwards.

⚠ EMITTED VALUES ARE SHELL-QUOTED with `shlex.quote`. A name with a space in it is ordinary
(`name: Les Paul`), and unquoted it would assign `AGENT_NAME=Les` and then try to run `Paul`.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
ROSTER = REPO / "roster.yaml"

# Every key an entry must carry. `persona` is included because a roster entry that cannot say
# which file describes the role is not a complete identity — and a missing persona path is the
# kind of gap that reads as "not applicable" rather than as an error.
REQUIRED = ("name", "email", "persona")


def die(msg):
    """Refuse: nothing on stdout, reason on stderr, non-zero exit. See the module docstring."""
    sys.stderr.write("agent_env: %s\n" % msg)
    raise SystemExit(1)


def load(path=ROSTER):
    try:
        import yaml
    except ImportError:
        die("PyYAML is not installed in this interpreter (%s). AGENTS.md §3: run host "
            "scripts through .venv/bin/python." % sys.executable)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        die("cannot read the roster at %s: %s" % (path, e))
    try:
        doc = yaml.safe_load(text)
    except Exception as e:                        # noqa: BLE001 — the reason is the point
        die("%s is not valid YAML: %s" % (path, e))
    if not isinstance(doc, dict) or not isinstance(doc.get("agents"), dict) or not doc["agents"]:
        die("%s declares no agents" % path)
    return doc


def resolve(doc, title=None, path=ROSTER):
    """The entry for `title`, or for `default_agent` when no title is given."""
    agents = doc["agents"]
    if title is None:
        title = doc.get("default_agent")
        if not title:
            die("no --agent given and %s sets no `default_agent`" % path)
    if title not in agents:
        die("no agent titled %r in %s — it has: %s"
            % (title, path, ", ".join(sorted(agents))))
    entry = agents[title]
    if not isinstance(entry, dict):
        die("the %r entry in %s is not a mapping" % (title, path))
    missing = [k for k in REQUIRED if not str(entry.get(k) or "").strip()]
    if missing:
        # ⚠ A REFUSAL, NEVER A DEFAULT. A script-side default here would be a second copy of
        # the setting, and it would win silently over the roster it was meant to back up.
        die("the %r entry in %s is missing: %s" % (title, path, ", ".join(missing)))
    return title, entry


def emit(title, entry):
    persona = str(entry["persona"]).strip()
    # ⚠ Checked, not assumed. A roster pointing at a persona that has moved is exactly the
    # drift this file centralises identity to prevent, and the failure would otherwise surface
    # as an agent reading nothing rather than as a bad path.
    #
    # ⚠⚠ THE PATH MUST BE REPO-RELATIVE, AND `REPO / persona` DOES NOT ENFORCE THAT.
    # `pathlib` DISCARDS the left operand when the right is absolute, so `REPO / "/etc/passwd"`
    # is `/etc/passwd` — it exists, `is_file()` is true, and a leading-slash typo in the roster
    # resolved to a file outside the repo with rc=0. Measured 2026-09-15: it emitted
    # `AGENT_PERSONA=/etc/hostname` and reported success. Rejected explicitly rather than
    # relying on the join.
    if os.path.isabs(persona):
        die("the %r entry names persona %r as an ABSOLUTE path; roster personas are "
            "repo-relative (a leading slash makes the repo root vanish)" % (title, persona))
    resolved = (REPO / persona).resolve()
    if not resolved.is_relative_to(REPO.resolve()):
        die("the %r entry names persona %r, which resolves outside the repo (%s)"
            % (title, persona, resolved))
    if not resolved.is_file():
        die("the %r entry names persona %r, which is not a file under %s" % (title, persona, REPO))
    return "\n".join(
        "%s=%s" % (k, shlex.quote(str(v)))
        for k, v in (("AGENT_TITLE", title),
                     ("AGENT_NAME", str(entry["name"]).strip()),
                     ("AGENT_EMAIL", str(entry["email"]).strip()),
                     ("AGENT_PERSONA", persona))
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default=None, metavar="TITLE",
                    help="roster title; defaults to the roster's `default_agent`")
    ap.add_argument("--roster", default=str(ROSTER),
                    help=argparse.SUPPRESS)        # tests only; not a lane knob
    args = ap.parse_args(argv)
    path = pathlib.Path(args.roster)
    title, entry = resolve(load(path), args.agent, path)
    sys.stdout.write(emit(title, entry) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
