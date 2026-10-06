"""No Sonora code reaches Ollama any more (Notes/Sonora/lemonade-migration-design.md).

Gemma is served by Sonora's own llama.cpp server (scripts/lib/gemma_server.py) through
scripts/lib/gemma_client.py. This scans every TRACKED .py and .sh file under scripts/,
matcha/ and tests/ for the old server's name or port.
"""

import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
SELF = pathlib.Path(__file__).resolve()
PATTERN = re.compile(r"ollama|11434", re.IGNORECASE)


def _sources():
    listed = subprocess.run(["git", "ls-files", "--", "scripts", "matcha", "tests"],
                            cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [REPO / p for p in listed.split()
            if p.endswith((".py", ".sh")) and (REPO / p).resolve() != SELF]


def offenders(paths):
    return [f"{p.name}:{i}" for p in paths
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace")
                                     .splitlines(), 1)
            if PATTERN.search(line)]


def test_the_scan_covers_the_code():
    # An empty or mis-rooted scan would pass vacuously.
    assert len(_sources()) >= 100


def test_no_code_names_the_old_server():
    assert offenders(_sources()) == []


def test_a_planted_reference_is_caught(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text('URL = "http://localhost:11434/api/chat"\n# an Ollama call\n')
    assert offenders([planted]) == ["planted.py:1", "planted.py:2"]
