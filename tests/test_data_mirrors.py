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
# NOTE the LiteRT harness is deliberately absent. It no longer has a /data copy at all:
# `SONORA_LITERT_WORK` split the data from the code on 2026-08-06, so the scripts execute
# from the repo and only checkpoints, graphs, artifacts and the venv remain on /data. That
# is the preferred resolution — the copy is gone rather than policed. See
# `test_litert_harness_has_no_code_copy` below, which guards the property directly.
# The audition entry became INTRA-repo on 2026-08-08 when the app moved here from
# AI-Lab-AMD. That it had ever been a cross-repo hop was the symptom worth noticing: this
# repo owned the test for another repo's code, because the code was a Sonora domain surface
# filed in a machine-blueprint repo. The dashboard hops remain cross-repo and correctly so —
# a lab status page really is the box's.
MIRRORS = [
    (REPO / "scripts/synthesis/teacher_audition",
     DATA / "toolchain/teacher-audition", "*.py"),
    (REPO / "scripts/synthesis/teacher_audition",
     DATA / "toolchain/teacher-audition", "*.sh"),
    (REPO / "audition/app",
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
    assert sum(1 for _ in _pairs()) >= 12


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


def test_litert_harness_has_no_code_copy():
    """The LiteRT harness must EXECUTE from the repo, not from a copy on /data.

    Inverted on purpose: the other mirrors are checked for agreement, this one is checked
    for absence. The copy existed only because every script read and wrote next to its own
    source, so running from the checkout would have dumped ~400 MB of graphs into the
    working tree; `SONORA_LITERT_WORK` split those apart and the copy was retired. If a
    `.py` reappears here, someone has re-created the exact arrangement that let
    convert_vat.py run three weeks stale.
    """
    work = DATA / "toolchain/litert-conversion"
    if not work.is_dir():
        pytest.skip("harness work dir not present on this host")
    stray = sorted(p.name for p in work.glob("*.py"))
    assert not stray, (
        f"code has reappeared in {work}: {stray}\n"
        "  Run the harness from the repo instead: scripts/litert_export/run.sh <script.py>\n"
        "  /data holds the venv, checkpoints and artifacts — not our source (AGENTS.md §6)."
    )


# --- The delivery vocabulary must not fork again (2026-08-08) --------------------------

def test_the_audition_app_does_not_redeclare_the_delivery_lanes():
    """It imports the contract; it must never carry a literal copy of the closed set.

    Until this app moved into Sonora it lived in AI-Lab-AMD and wrote the five lanes out in
    full — in a different order from `matcha.delivery.DELIVERY_LANES`, in a different
    repository, where no import could reach them. That is the review's most-repeated defect
    class (B-L5's MAX_REF_EXCURSION written three times, D-L2's two disagreeing z-guards),
    and "which number means Newscaster" is the rule that must not fork: getting it wrong
    produces fluent, confidently mis-delivered audio rather than an error.

    Guarded by text rather than by value, because a re-declared tuple that happens to AGREE
    today would pass a value check and drift tomorrow.
    """
    src = (REPO / "audition/app/main.py").read_text(encoding="utf-8")
    body = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    # Pins the binding the PICKER actually uses. This asserted on
    # `DELIVERY_LANES = _delivery.DELIVERY_LANES` until 2026-08-10, and by then that line
    # had no readers left — the picker had moved to the active set, so the guard was
    # keeping a string alive rather than proving a fact (issue #17). Someone could have
    # hardcoded `DELIVERY = ("", "Dialogue", …)` and this would still have passed.
    assert "ACTIVE_DELIVERY_LANES = getattr(_delivery," in body, \
        "the app must take its assignable lanes from matcha.delivery, not declare them"
    assert "*sorted(ACTIVE_DELIVERY_LANES)" in body, \
        "the picker must be composed from ACTIVE_DELIVERY_LANES, not from a literal"
    for lane in ("Dialogue", "Newscaster", "Documentary"):
        assert f'"{lane}"' not in body, \
            f"{lane!r} is written literally in main.py — the vocabulary has forked again"


def test_the_delivery_contract_is_deployed_beside_the_app():
    """The app RAISES without it, so a forgotten copy must fail loudly rather than silently.

    `deploy.sh audition` copies `matcha/delivery.py` into the deployed `app/_contract/`
    after its `rsync --delete` — the same pattern as the device G2P's exported
    `g2p_contractions.json`, and for the same reason: a hand-synced table is a defect of
    omission on a delay.
    """
    deployed = DATA / "services/audition/app"
    if not deployed.is_dir():
        pytest.skip("audition is not deployed on this machine")
    asset = deployed / "_contract/delivery.py"
    assert asset.is_file(), (
        "the deployed app has no _contract/delivery.py — it will refuse to start. "
        "Run scripts/deploy.sh audition.")
    # ACTIVE_DELIVERY_LANES, not DELIVERY_LANES: a contract copy predating the Documentary
    # retirement satisfies the weaker check while being exactly the stale asset that makes
    # the app raise at import (issue #16). The test has to fail on the same input the app
    # fails on, or it certifies a deploy that cannot start.
    asset_text = asset.read_text(encoding="utf-8")
    assert "DELIVERY_LANES" in asset_text
    assert "ACTIVE_DELIVERY_LANES" in asset_text, (
        "the deployed _contract/delivery.py predates RETIRED_LANES — the app will refuse "
        "to start. Run scripts/deploy.sh audition (it copies the contract, not just code).")
