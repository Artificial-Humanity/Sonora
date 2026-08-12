"""C-M8: one definition of "ear-confirmed (reader, title)".

There were two, at opposite thresholds:

  * `pick_audit_subset` required ALL THREE of gender/age/accent before it would stop
    force-queuing a clip from the pair;
  * `stage_pool` tested the truthiness of the attribute dict, so ANY ONE was enough to
    fold the whole title into the corpus as machine-written keeps, unheard.

A pair sitting on two of three was therefore both "confirmed enough to fold three hundred
clips" and "still owed an ear pass" — and `learn()` PRODUCES that shape on purpose: when
the votes disagree within a title it writes `{attr}_CONFLICT` INSTEAD of `{attr}`. So the
one title we have positive evidence is internally inconsistent is precisely the one that
arrives partial, and the looser reader folded it. C-M7 closed the same hole on the hint
path; this is the fold path, which is the consequential one.

Everything here runs against tmp_path.
"""

import csv
import json
import pathlib
import sys

import pytest
from scripts_layout import SCRIPTS  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
SYNTH = SCRIPTS
SCRIPTS.on_path()
rp = pytest.importorskip("reader_profile")

FULL = {"gender": "female", "age": "adult", "accent": "american", "clips_seen": 4}
PARTIAL = {"gender": "female", "age": "adult", "clips_seen": 4}
CONFLICTED = {"gender": "female", "age": "adult",
              "accent_CONFLICT": {"american": 2, "british": 2}, "clips_seen": 4}


def _profiles(title_entry):
    return {"A Reader": {"titles": {"A Book": dict(title_entry)}, "hint": {}}}


# --- the predicate itself -------------------------------------------------------------


@pytest.mark.parametrize("entry,expected", [
    (FULL, True), (PARTIAL, False), (CONFLICTED, False), ({}, False),
])
def test_all_three_or_it_is_not_confirmed(entry, expected):
    assert rp.is_confirmed_pair(_profiles(entry), "A Reader", "A Book") is expected


def test_a_conflicted_attribute_is_not_an_attribute():
    """`learn()` withholds the value and records the disagreement instead, so the plain
    key is simply absent. The explicit exclusion is belt-and-braces for the day the merge
    becomes per-attribute and a stale value could sit beside a fresh conflict marker."""
    mixed = dict(CONFLICTED, accent="american")     # the shape that merge would produce
    assert "accent" not in rp.confirmed_attrs(_profiles(mixed), "A Reader", "A Book")
    assert rp.is_confirmed_pair(_profiles(mixed), "A Reader", "A Book") is False


def test_an_unknown_pair_is_not_confirmed_and_does_not_raise():
    prof = _profiles(FULL)
    assert rp.is_confirmed_pair(prof, "A Reader", "Another Book") is False
    assert rp.is_confirmed_pair(prof, "Someone Else", "A Book") is False
    assert rp.confirmed_attrs(prof, "", "") == {}


# --- the two readers now agree --------------------------------------------------------


@pytest.mark.parametrize("entry", [FULL, PARTIAL, CONFLICTED, {}])
def test_the_stager_and_the_auditor_answer_identically(entry, tmp_path, monkeypatch):
    """THE REGRESSION. Whatever the profile looks like, folding and force-queueing must
    reach the same verdict. They disagreed on exactly the middle two cases."""
    sp = pytest.importorskip("stage_pool")
    pa = pytest.importorskip("pick_audit_subset")

    path = tmp_path / "reader_profiles.json"
    path.write_text(json.dumps(_profiles(entry)), encoding="utf-8")
    monkeypatch.setattr(sp, "PROFILES", path)
    monkeypatch.setattr(pa, "PROFILES_PATH", str(path))

    rec = {"reader": "A Reader", "book": "A Book"}
    stager_says = bool(sp.confirmed_tags(rec))
    auditor_says = ("A Reader", "A Book") in pa._confirmed_reader_titles()
    assert stager_says == auditor_says == (entry is FULL)


def test_the_attribute_tuple_has_one_definition():
    """Two copies of `("gender", "age", "accent")` are two chances to add a fourth
    attribute in one place only."""
    sp = pytest.importorskip("stage_pool")
    assert sp.ATTRS is rp.ATTRS
    src = (SYNTH / "pick_audit_subset.py").read_text(encoding="utf-8")
    assert '"gender", "age", "accent"' not in src
    assert "reader_profile.is_confirmed_pair" in src


# --- what it means in the staging run -------------------------------------------------


RATINGS_HEADER = ["campaign", "id", "engine", "status", "score", "note", "link",
                  "gender", "age", "accent", "delivery"]


@pytest.fixture()
def campaign(tmp_path, monkeypatch):
    sp = pytest.importorskip("stage_pool")
    datasets = tmp_path / "datasets"
    cdir = datasets / "camp"
    cdir.mkdir(parents=True)
    with (cdir / "librivox_manifest.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(4):
            fh.write(json.dumps({
                "id": f"clip{i}", "book": "A Book", "reader": "A Reader",
                "section": 1, "sentence_index": i, "wav": f"clip{i}.wav",
                "seconds": 8.0,
            }) + "\n")
    # QC-M2: staging REFUSES a campaign with no measures. Written clean, so what these
    # tests are about — the confirmation predicate — is what decides the fold.
    with (cdir / "qc_measures.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(4):
            fh.write(json.dumps({
                "id": f"clip{i}", "hard_pass": True, "worst_pause": 0.4,
                "head_words_lost": 0,
                "gates": {"asr_ok": True, "tail_ok": True, "pause_ok": True,
                          "speech_ok": True, "length_ok": True},
            }) + "\n")
    ratings = datasets / "ratings.csv"
    with ratings.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=RATINGS_HEADER).writeheader()
    profiles = datasets / "reader_profiles.json"

    monkeypatch.setattr(sp, "DATASETS", datasets)
    monkeypatch.setattr(sp, "RATINGS", ratings)
    monkeypatch.setattr(sp, "PROFILES", profiles)
    monkeypatch.setattr(sp, "LEDGER", datasets / "books_ledger.json")

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["stage_pool", "--campaign", "camp", *argv])
        return sp.main()

    return sp, profiles, ratings, run


def _rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_a_conflicted_title_is_no_longer_folded_unheard(campaign, capsys):
    """The live path. A title whose ear pass disagreed with itself on accent used to be
    staged as keeps carrying gender and age, with the disputed attribute blank and nobody
    listening — the exact outcome the conflict marker was added to prevent."""
    _sp, profiles, ratings, run = campaign
    profiles.write_text(json.dumps(_profiles(CONFLICTED)), encoding="utf-8")

    assert run("--stage", "4", "--apply") == 1
    assert _rows(ratings) == []
    out = capsys.readouterr().out
    assert "no FULLY confirmed" in out
    assert "still owed: accent" in out
    assert "_CONFLICT" in out            # points at why, not just that


def test_a_fully_confirmed_title_still_folds(campaign):
    """The fix must not stop the mechanism it guards: full confirmation still propagates
    all three attributes without a per-clip audition."""
    _sp, profiles, ratings, run = campaign
    profiles.write_text(json.dumps(_profiles(FULL)), encoding="utf-8")

    assert run("--stage", "4", "--apply") == 0
    rows = _rows(ratings)
    assert len(rows) == 4
    assert all(r["status"] == "keep" and r["score"] == "" for r in rows)
    assert all(r["gender"] == "female" and r["accent"] == "american" for r in rows)


def test_seed_ear_seeds_a_partial_pair_instead_of_declaring_it_done(campaign, capsys):
    """`--seed-ear` skipped any pair with one attribute set, so the deadlock it exists to
    break stayed shut for the partial case: nothing seeded, nothing stageable, and the
    printed reason was "already ear-confirmed"."""
    _sp, profiles, ratings, run = campaign
    profiles.write_text(json.dumps(_profiles(PARTIAL)), encoding="utf-8")

    assert run("--seed-ear", "--apply") == 0
    rows = _rows(ratings)
    assert len(rows) == 1
    assert rows[0]["status"] == "unaudited"
    assert rows[0]["gender"] == ""          # a seed carries no tags, by design
    assert "still owed: accent" in capsys.readouterr().out


def test_seed_ear_still_skips_a_pair_that_is_genuinely_done(campaign, capsys):
    _sp, profiles, ratings, run = campaign
    profiles.write_text(json.dumps(_profiles(FULL)), encoding="utf-8")

    assert run("--seed-ear", "--apply") == 0
    assert _rows(ratings) == []
    assert "already ear-confirmed" in capsys.readouterr().out


# --- QC-M2: what the fold does when the instrument is absent or merely advisory --------


def _measures(cdir, rows):
    with (cdir / "qc_measures.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_a_campaign_with_no_qc_measures_is_refused_not_folded(campaign, tmp_path):
    """QC-M2. `qc_flagged` returned an empty set when the file was missing, with no
    warning — so a pooled campaign staged before anyone ran `qc_gate` folded EVERY clip
    as a machine-written `keep` with zero flags. "QC follows every generation pass"
    defeated by file absence.

    This is the missing-file-treated-as-pass shape the repo has closed twice already (the
    empty-glob 0/0 report, `qc_gate`'s empty-measures refusal), and here it fails in the
    direction that costs a corpus rather than a run: the fold writes no ratings row, so
    there is nothing to notice afterwards.
    """
    sp, _profiles, ratings, run = campaign
    (sp.DATASETS / "camp" / "qc_measures.jsonl").unlink()
    with pytest.raises(SystemExit) as e:
        run("--stage", "4", "--apply")
    assert "qc_measures.jsonl" in str(e.value)
    assert _rows(ratings) == [], "nothing may be staged on an unmeasured campaign"


def test_a_mid_clip_stall_reaches_the_ear_through_the_pooled_lane(campaign):
    """The dead-air ADVISORY band (1.4–2.5 s) could not queue a pooled clip at all: this
    lane tested failing gates plus the head advisory, and stopped there. So a 2.0 s
    mid-clause stall folded unheard here, while the identical clip in the synth lane
    reached `qc_flags.txt` via `register_audition._qc_triage`. One clip, two lanes,
    opposite fates — and this is the lane that folds without a ratings row.
    """
    sp, profiles, ratings, run = campaign
    profiles.write_text(json.dumps(_profiles(FULL)), encoding="utf-8")
    cdir = sp.DATASETS / "camp"
    clean = {"gates": {"asr_ok": True, "tail_ok": True, "pause_ok": True},
             "hard_pass": True, "head_words_lost": 0}
    _measures(cdir, [
        {"id": "clip0", "worst_pause": 0.4, **clean},
        {"id": "clip1", "worst_pause": 2.0, **clean},   # in the band; every gate passes
        {"id": "clip2", "worst_pause": 0.5, **clean},
        {"id": "clip3", "worst_pause": 0.3, **clean},
    ])
    assert "clip1" in sp.qc_flagged(cdir)

    assert run("--stage", "4", "--apply") == 0
    by_id = {r["id"]: r for r in _rows(ratings)}
    assert by_id["clip1"]["status"] == "unaudited", "the stall folded unheard"
    assert by_id["clip1"]["score"] == ""
    # ...and the clips with nothing to hear still fold, or the band would cost the corpus.
    assert by_id["clip0"]["status"] == "keep"


def test_a_pause_inside_ordinary_phrasing_does_not_queue(campaign):
    """The band has to have a bottom, or every clip is an audition."""
    sp, _profiles, _ratings, _run = campaign
    cdir = sp.DATASETS / "camp"
    clean = {"gates": {"asr_ok": True, "pause_ok": True}, "hard_pass": True,
             "head_words_lost": 0}
    _measures(cdir, [{"id": "clip0", "worst_pause": 1.39, **clean},
                     {"id": "clip1", "worst_pause": 1.4, **clean}])
    assert sp.qc_flagged(cdir) == {"clip1"}


def test_a_row_written_before_the_pause_measure_says_so(campaign, capsys):
    """`head_flagged` can recompute from text when a row predates its measure; a pause
    cannot be recovered from text at all. 2,030 of the 2,188 pooled clips on disk are in
    exactly that state, so silence here would claim every clip was checked on a campaign
    where the check never ran."""
    sp, _profiles, _ratings, _run = campaign
    cdir = sp.DATASETS / "camp"
    _measures(cdir, [{"id": "clip0", "gates": {"asr_ok": True}, "hard_pass": True,
                      "head_words_lost": 0}])          # no worst_pause at all
    assert sp.qc_flagged(cdir) == set()
    assert "advisory band is NOT covered" in capsys.readouterr().out


def test_the_pause_threshold_has_one_definition():
    """It was written out three times; `scripts/test_skill_files` watched two of them, and
    the third lane did not carry the number at all — which is why the band could not queue
    a pooled clip."""
    sp = pytest.importorskip("stage_pool")
    ra = pytest.importorskip("register_audition")
    import synth_common
    assert ra.PAUSE_FLAG_SECONDS is synth_common.PAUSE_FLAG_SECONDS
    assert "PAUSE_FLAG_SECONDS = 1.4" not in (SYNTH / "register_audition.py").read_text()
    assert sp.synth_common.pause_flagged({"worst_pause": 1.4})
