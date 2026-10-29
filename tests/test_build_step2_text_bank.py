"""Phase 1S Step 2b's text bank is the pre-registered one (Option A): 36,000 globally distinct lines at
4.8-9.0 s, 90 speakers of 360 primary and 40 reserve lines, Step 1c-i's constants untouched, and
the length bounds are parameters that default to 1c-i's."""
import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()

import build_step2_text_bank as s2  # noqa: E402
import probe_text_bank as ptb  # noqa: E402

CPS = ptb.CHARS_PER_SEC


def _row(k):
    return {"id": "2b-%04d" % (k + 1), "text": "Sentence %d." % k}


def test_the_step2b_constants_are_the_registered_option_a_ones():
    assert s2.MIN_CLIP_SECONDS == 4.8
    assert s2.MAX_CLIP_SECONDS == 9.0
    assert (s2.N_SPEAKERS, s2.LINES_PER_SPEAKER, s2.RESERVE_PER_SPEAKER) == (90, 360, 40)
    assert s2.TARGET == 36000
    assert s2.DEFAULT_SEED == 20261029
    assert s2.MAX_PER_BOOK == 800
    assert s2.ID_PREFIX != ptb.ID_PREFIX


def test_step_1ci_constants_are_frozen_and_unchanged():
    assert ptb.MIN_CLIP_SECONDS == 4.0
    assert ptb.MAX_CLIP_SECONDS == 8.0
    assert ptb.BANK_SIZE == 300
    assert ptb.MAX_PER_BOOK == 30


def test_length_bounds_default_to_1ci_and_accept_step2b_bounds():
    # 1c-i's defaults: 56 characters (4.0 s) through 112 (8.0 s).
    assert ptb.within_length("x" * 56) and ptb.within_length("x" * 112)
    assert not ptb.within_length("x" * 113)
    # Step 2b's bounds at 14 chars/s: 68 (4.857 s) through 126 (9.0 s), both edges included.
    assert ptb.within_length("x" * 68, s2.MIN_CLIP_SECONDS, s2.MAX_CLIP_SECONDS)
    assert ptb.within_length("x" * 126, s2.MIN_CLIP_SECONDS, s2.MAX_CLIP_SECONDS)
    assert not ptb.within_length("x" * 67, s2.MIN_CLIP_SECONDS, s2.MAX_CLIP_SECONDS)
    assert not ptb.within_length("x" * 127, s2.MIN_CLIP_SECONDS, s2.MAX_CLIP_SECONDS)


def test_sentence_rejection_reports_the_step2b_length_reason():
    # A complete declarative sentence of 66 characters (4.71 s): too short for Step 2b, and
    # long enough for 1c-i's 4.0 s floor.
    text = "She folded the letter twice, then set it on the table by the lamp."
    assert 56 <= len(text) < 68
    assert ptb.sentence_rejection(text, s2.MIN_CLIP_SECONDS, s2.MAX_CLIP_SECONDS) == "too_short"
    assert ptb.sentence_rejection(text) is None


def test_the_assignment_gives_every_speaker_400_lines_and_no_line_twice():
    rows = [_row(k) for k in range(s2.TARGET)]
    spk = s2.speaker_files(rows)
    assert all(len(spk[s]) == s2.SPEAKER_LINES for s in range(s2.N_SPEAKERS))
    ids = [r["id"] for s in range(s2.N_SPEAKERS) for r in spk[s]]
    assert len(ids) == len(set(ids)) == s2.TARGET


def test_each_speaker_has_360_primary_then_40_reserve_in_speaker_order():
    spk = s2.speaker_files([_row(k) for k in range(s2.TARGET)])
    for s in range(s2.N_SPEAKERS):
        rows = spk[s]
        roles = [r["role"] for r in rows]
        assert roles.count("primary") == 360 and roles.count("reserve") == 40
        assert roles == ["primary"] * 360 + ["reserve"] * 40
        assert [r["speaker_line_index"] for r in rows] == list(range(1, 401))
        assert all(r["speaker_index"] == s for r in rows)


def test_line_k_goes_to_speaker_k_mod_90():
    rows = [_row(k) for k in range(s2.TARGET)]
    spk = s2.speaker_files(rows)
    assert spk[0][0]["id"] == rows[0]["id"]
    assert spk[5][2]["id"] == rows[2 * s2.N_SPEAKERS + 5]["id"]


def test_the_assignment_refuses_a_bank_of_the_wrong_size():
    with pytest.raises(ValueError, match="36000"):
        s2.assign_speakers([_row(k) for k in range(10)])


def test_the_pool_guard_refuses_an_under_full_bank_by_name():
    cands = [{"book": "b%d" % (i % 3), "author": "a%d" % (i % 3), "text": "Line %d." % i}
             for i in range(20)]
    with pytest.raises(ptb.BankError):
        ptb.select_lines(cands, n=s2.TARGET, seed=s2.DEFAULT_SEED, max_per_book=s2.MAX_PER_BOOK)


def test_write_refuses_to_replace_a_different_bank_and_keeps_an_identical_one(tmp_path):
    import json

    rows = [dict(_row(k), book="b", title="t", source_url="u", est_seconds=5.0, chars=60)
            for k in range(s2.TARGET)]
    spk = s2.speaker_files(rows)
    meta = {"sha256": "x"}
    out = tmp_path / "bank"
    s2.write(out, rows, spk, meta)
    assert len(list((out / "speakers").glob("spk_*.jsonl"))) == s2.N_SPEAKERS
    first = json.loads((out / "speakers" / "spk_00.jsonl").read_text().splitlines()[0])
    assert first["speaker_index"] == 0 and first["role"] == "primary"
    assert first["speaker_line_index"] == 1
    s2.write(out, rows, spk, meta)                       # identical: accepted
    other = [dict(r, text=r["text"] + " x") for r in rows]
    with pytest.raises(SystemExit):
        s2.write(out, other, s2.speaker_files(other), meta)
