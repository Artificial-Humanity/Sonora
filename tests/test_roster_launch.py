"""The reviewer's model and effort come from ITS ROSTER ENTRY, and from nowhere else.

Owner, 2026-09-02: Janis runs Fable 5.1 at high. The values are `model:` / `effort:` on the
reviewer entry in FerroStep/config.yaml; `FerroStep/workflow/scripts/roster_launch.py` reads them and
`request_review.sh` evals what it emits. Two things are asserted here:

  * the reader is a READER — a temp roster's values come back verbatim, quoted for a shell;
  * it has NO DEFAULT — a missing key, an empty value, an effort off the CLI's ladder or an
    unknown title each REFUSE. A default would be the second copy the module was written to
    remove, and the one that wins silently when the roster's key is misspelt.

⚠ The real FerroStep/config.yaml is checked too, but only for SHAPE (present, on the ladder), never for
a value: a test that says "the model is X" is a copy of X that goes stale on the next owner
change, which is the drift this whole arrangement exists to end.
"""
import importlib.util
import pathlib
import shlex

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
# Loaded by path, as test_merge_floor.py does: a hand `sys.path.insert` leaks into every
# later module (tests/test_stage_coverage.py polices exactly that).
_SPEC = importlib.util.spec_from_file_location(
    "roster_launch", REPO / "FerroStep" / "workflow" / "scripts" / "roster_launch.py")
rl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rl)

HEAD = """default_agent: developer
agents:
  developer:
    name: PJ
    email: pj@example.invalid
    persona: FerroStep/personas/DEVELOPER.md
  reviewer:
    name: J
    email: j@example.invalid
    persona: FerroStep/personas/REVIEWER.md
"""


def _roster(tmp_path, extra):
    p = tmp_path / "config.yaml"
    p.write_text(HEAD + "".join(f"    {k}: {v}\n" for k, v in extra.items()))
    return str(p)


def test_a_complete_entry_is_read_verbatim(tmp_path):
    p = _roster(tmp_path, {"model": "some-model-id", "effort": "max"})
    assert rl.launch_settings(p, "reviewer") == {"model": "some-model-id", "effort": "max"}


def test_shell_lines_are_evaluable_and_quoted(tmp_path):
    """A value with shell metacharacters must arrive as data, not syntax."""
    p = _roster(tmp_path, {"model": "'x; rm -rf /'", "effort": "low"})
    out = rl.shell_lines(p, "reviewer")
    assert out == f"AGENT_MODEL={shlex.quote('x; rm -rf /')}\nAGENT_EFFORT=low\n"


@pytest.mark.parametrize("extra, missing", [
    ({"effort": "high"}, "model"),                 # no model at all
    ({"model": "m"}, "effort"),                    # no effort at all
    ({"model": "''", "effort": "high"}, "model"),  # present but empty
])
def test_a_missing_key_refuses_and_names_itself(tmp_path, extra, missing):
    """THE RED ONE. There is no default to fall back to, and the message says which key."""
    p = _roster(tmp_path, extra)
    with pytest.raises(rl.LaunchError, match=f"`{missing}:`"):
        rl.launch_settings(p, "reviewer")


def test_an_effort_off_the_ladder_refuses(tmp_path):
    p = _roster(tmp_path, {"model": "m", "effort": "extreme"})
    with pytest.raises(rl.LaunchError, match="extreme"):
        rl.launch_settings(p, "reviewer")


def test_an_unknown_title_refuses(tmp_path):
    p = _roster(tmp_path, {"model": "m", "effort": "high"})
    with pytest.raises(rl.LaunchError, match="no agent titled 'auditor'"):
        rl.launch_settings(p, "auditor")


def test_an_unreadable_roster_refuses(tmp_path):
    with pytest.raises(rl.LaunchError, match="cannot read"):
        rl.launch_settings(str(tmp_path / "absent.yaml"), "reviewer")


def test_the_ladder_is_the_cli_s():
    """`claude --help` lists exactly these (2026-09-02). If the CLI grows a rung, add it HERE;
    the roster's comment beside `effort:` points at this tuple rather than restating it."""
    assert rl.EFFORTS == ("low", "medium", "high", "xhigh", "max")


def test_the_real_reviewer_entry_resolves():
    """Shape only — see the module docstring for why no value is asserted."""
    s = rl.launch_settings(str(REPO / "FerroStep" / "config.yaml"), "reviewer")
    assert s["model"] and s["effort"] in rl.EFFORTS
