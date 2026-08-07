"""Every copy of our code on /data must match the repo it came from.

WHY THIS EXISTS (2026-08-06). The suspicion was that the LiteRT export harness at
`/data/toolchain/litert-conversion/` was unbacked — outside git, one disk failure from
gone. It is not: it was migrated into `scripts/litert_export/` on 2026-07-22 and every
script is tracked. The real problem was the opposite shape, and worse for being invisible.

`/data` holds *working copies*, and two of them had DRIFTED from the tracked originals:

  * `convert_vat.py` on /data was three weeks stale. The repo copy gained a
    `detect_vat_dim` seam guard on 2026-08-04 — recorded in notes/todo.md as landed —
    and the harness that actually runs never received it. A guard nobody has watched
    fail is a guess; a guard that is not installed is not a guard.
  * `README.md` on /data still documented a bare-`pip` install, predating the uv standard.

Backup was never the risk. Divergence was, and nothing detected it: both directories look
healthy, the scripts run, and the only symptom is that a fix you believe shipped did not.

The other three mirrors (the audition app, the dashboard, the teacher-audition renderers)
were checked at the same time and were all in sync — so this test locks in a property
that currently holds, rather than papering over a known break.

Skipped anywhere /data is not mounted, which is every machine but ai-lab-0.
"""

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PROJECTS = REPO.parent.parent          # …/Artificial-Humanity
DATA = pathlib.Path("/data")

pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="/data is not mounted here")

# (repo-relative source dir, /data working dir, glob) — the source of truth is ALWAYS the
# repo. A /data copy that is newer is not authoritative, it is unreviewed: it has no
# history, no diff and no changelog entry.
MIRRORS = [
    (REPO / "scripts/litert_export",
     DATA / "toolchain/litert-conversion", "*.py"),
    (REPO / "scripts/litert_export",
     DATA / "toolchain/litert-conversion", "*.md"),
    (REPO / "scripts/synthesis/teacher_audition",
     DATA / "toolchain/teacher-audition", "*.py"),
    (REPO / "scripts/synthesis/teacher_audition",
     DATA / "toolchain/teacher-audition", "*.sh"),
    (PROJECTS / "AI-Lab-AMD/audition/app",
     DATA / "services/audition/app", "*.py"),
    (PROJECTS / "AI-Lab-AMD/dashboard",
     DATA / "services/dashboard", "*.html"),
    (PROJECTS / "AI-Lab-AMD/dashboard/scripts",
     DATA / "services/dashboard/scripts", "*.sh"),
]


def _pairs():
    for src_dir, data_dir, glob in MIRRORS:
        if not src_dir.is_dir() or not data_dir.is_dir():
            continue
        for src in sorted(src_dir.glob(glob)):
            yield src, data_dir / src.name


def test_there_is_something_to_check():
    """A gate that cannot fail is not a gate: if every path went missing (a layout change,
    a rename), the parametrised test below would silently collect nothing and pass."""
    assert sum(1 for _ in _pairs()) >= 15


@pytest.mark.parametrize("src,mirror", list(_pairs()),
                         ids=lambda p: getattr(p, "name", str(p)))
def test_data_copy_matches_the_repo(src, mirror):
    if not mirror.exists():
        pytest.skip(f"{mirror} not deployed on this host")
    assert src.read_bytes() == mirror.read_bytes(), (
        f"{mirror}\n  has drifted from the tracked original\n  {src}\n"
        "  The repo is authoritative. Diff them, decide which change is wanted, and if the\n"
        "  /data side is right then COMMIT it there rather than copying it back — an\n"
        "  untracked edit has no history, no review and no changelog entry.\n"
        "  This is exactly how convert_vat.py's detect_vat_dim guard failed to ship."
    )
