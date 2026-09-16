"""`scripts/agent_env.py` — the resolver every commit's attribution depends on.

⚠⚠ THE PROPERTY UNDER TEST IS NOT "IT RESOLVES". It is that a refusal emits **zero bytes on
stdout** and exits non-zero. `eval "$(cmd)"` reports the status of the text it ran, and a
refusal that emits nothing makes `eval ""` — which is status 0. So a caller who collapses

    AGENT_ENV="$(… agent_env.py)"   # rc checked here
    eval "$AGENT_ENV"

into one step loses every refusal silently and then runs `git -c user.name="" …`. The split
is what protects them, and the split only works if this script never half-emits.

⚠ This replaced `ferrostep agent-env` on 2026-09-15, when the Ferro projects were dismantled
for a ground-up rethink. The rule it serves is older and is not FerroStep's: the repo's
configured git identity is the OWNER's, deliberately, so a forgotten `-c` pair commits an
agent's work under their name and nothing downstream says so.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

# ⚠ A PLAIN IMPORT, NOT `importorskip`. `pyproject.toml` declares PyYAML as "CORE, NOT
# TRANSITIVE, AND NOT OPTIONAL", and `tests/test_commit_hygiene.py` hard-fails without it.
# Skipping here would have given two files two answers to the same missing dependency, and
# the file that vanishes silently is this one — the eval-trap guard.
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "agent_env.py"
ROSTER = REPO / "roster.yaml"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=str(REPO))


def write(tmp_path, doc):
    p = tmp_path / "roster.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return str(p)


def good(persona="docs/personas/DEVELOPER.md"):
    return {"default_agent": "developer",
            "agents": {"developer": {"name": "Ozzy", "email": "o@x.io", "persona": persona}}}


# --------------------------------------------------------------------------- floors
def test_the_real_roster_resolves_and_names_a_persona_that_exists():
    r = run()
    assert r.returncode == 0, r.stderr
    env = dict(l.split("=", 1) for l in r.stdout.strip().splitlines())
    assert set(env) == {"AGENT_TITLE", "AGENT_NAME", "AGENT_EMAIL", "AGENT_PERSONA"}
    assert (REPO / env["AGENT_PERSONA"].strip("'\"")).is_file()


def test_the_shipped_roster_is_what_the_repo_actually_uses():
    """⚠ A test that only drove synthetic rosters would pass with `roster.yaml` deleted."""
    assert ROSTER.is_file(), "roster.yaml is missing — nothing resolves an identity"
    doc = yaml.safe_load(ROSTER.read_text(encoding="utf-8"))
    assert doc["default_agent"] in doc["agents"], "default_agent names no entry"


# --------------------------------------------------------- the property the split rests on
@pytest.mark.parametrize("case,doc,agent", [
    ("unknown title",      good(),                                        "nosuch"),
    ("no agents",          {"default_agent": "developer", "agents": {}},  None),
    ("no default_agent",   {"agents": {"developer": {"name": "O", "email": "e", "persona": "x"}}}, None),
    ("missing email",      {"default_agent": "d", "agents": {"d": {"name": "O", "persona": "x"}}}, None),
    ("blank name",         {"default_agent": "d", "agents": {"d": {"name": "  ", "email": "e", "persona": "x"}}}, None),
    ("entry not a mapping", {"default_agent": "d", "agents": {"d": "Ozzy"}},                       None),
    ("persona not on disk", good("docs/personas/GONE.md"),                None),
])
def test_every_refusal_emits_nothing_on_stdout(tmp_path, case, doc, agent):
    """⚠⚠ THE CENTRAL TEST. Non-zero is not enough — a refusal that printed a partial
    assignment would still be eval'd by a caller, and `AGENT_NAME=` is worse than no output.
    """
    args = ["--roster", write(tmp_path, doc)] + (["--agent", agent] if agent else [])
    r = run(*args)
    assert r.returncode != 0, "%s: exited 0\nstdout=%r" % (case, r.stdout)
    assert r.stdout == "", "%s: emitted %r on stdout — a caller would eval it" % (case, r.stdout)
    assert r.stderr.strip(), "%s: refused with no reason on stderr" % case


def test_a_malformed_roster_is_refused_not_crashed(tmp_path):
    p = tmp_path / "roster.yaml"
    p.write_text("default_agent: [unclosed\n", encoding="utf-8")
    r = run("--roster", str(p))
    assert r.returncode != 0 and r.stdout == ""
    assert "valid YAML" in r.stderr, r.stderr


# ------------------------------------------------------------------- quoting, and why
def test_a_name_with_a_space_survives_eval(tmp_path):
    """⚠ `name: Les Paul` unquoted assigns AGENT_NAME=Les and then tries to RUN `Paul`."""
    doc = good(); doc["agents"]["developer"]["name"] = "Les Paul"
    r = run("--roster", write(tmp_path, doc))
    assert r.returncode == 0, r.stderr
    probe = subprocess.run(["bash", "-c", 'eval "$1"; printf %s "$AGENT_NAME"', "_", r.stdout],
                           capture_output=True, text=True)
    assert probe.stdout == "Les Paul", (probe.stdout, probe.stderr, r.stdout)


def test_the_output_is_evalable_and_sets_all_four(tmp_path):
    r = run("--roster", write(tmp_path, good()))
    probe = subprocess.run(
        ["bash", "-c", 'eval "$1"; printf "%s|%s|%s|%s" "$AGENT_TITLE" "$AGENT_NAME" "$AGENT_EMAIL" "$AGENT_PERSONA"',
         "_", r.stdout], capture_output=True, text=True)
    assert probe.stdout == "developer|Ozzy|o@x.io|docs/personas/DEVELOPER.md", probe.stdout


def test_the_eval_trap_is_real_and_the_split_is_what_catches_it(tmp_path):
    """⚠ POSITIVE CONTROL for this file's whole premise, asserted rather than asserted-about.

    Collapsed, a refusal reports SUCCESS. Split, it reports failure. If this ever stops being
    true the docstrings above are decoration.
    """
    bad = write(tmp_path, {"default_agent": "d", "agents": {}})
    cmd = '%s %s --roster %s 2>/dev/null' % (sys.executable, SCRIPT, bad)
    collapsed = subprocess.run(["bash", "-c", 'eval "$(%s)"; echo $?' % cmd],
                               capture_output=True, text=True)
    assert collapsed.stdout.strip() == "0", "the trap did not reproduce: %r" % collapsed.stdout
    split = subprocess.run(["bash", "-c", 'AE="$(%s)"; echo $?' % cmd],
                           capture_output=True, text=True)
    assert split.stdout.strip() != "0", "the split failed to catch the refusal"
