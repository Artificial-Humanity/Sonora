"""The pure half of the baseline hum bench (`scripts/lib/baseline_bench.py`).

The bench asks whether stock Matcha-TTS (`matcha_vctk`, the root of every Sonora
checkpoint) hums as much as ours. Each helper here decides something the listening result
rests on, so each is pinned: the stock model's input ids, which speakers are compared, who
is excluded, and how the ratings turn into the pre-registered outcome.
"""

import pytest

from scripts_layout import SCRIPTS

SCRIPTS.on_path()
import baseline_bench as bb  # noqa: E402


# ---------------------------------------------------------------- the stock symbol table

def test_the_original_table_differs_from_ours_only_in_its_last_two_slots():
    ours = bb.ours_symbols()
    orig = bb.original_matcha_symbols()
    assert len(orig) == len(ours) == 178
    assert orig[:176] == ours[:176]
    assert orig[176:] == ["'", "ᵻ"] and ours[176:] == ["ᵻ", "ᵊ"]


def test_encoding_follows_upstream_last_wins_for_the_duplicated_apostrophe():
    # Upstream builds `{s: i for i, s in enumerate(symbols)}`, so its duplicate "'" maps to
    # the LATER slot. Our table would give the punctuation slot and read `ᵻ` one off.
    ids = bb.encode_original("'ᵻa")
    assert ids[0] == 176 and ids[1] == 177
    assert ids[2] == bb.ours_symbols().index("a")


def test_the_derived_table_is_upstream_byte_for_byte():
    # sha256 of "".join(symbols) from upstream Matcha-TTS matcha/text/symbols.py
    import hashlib
    got = hashlib.sha256("".join(bb.original_matcha_symbols()).encode()).hexdigest()
    assert got == bb.UPSTREAM_SYMBOLS_SHA256


def test_encoding_refuses_a_symbol_the_stock_model_never_had():
    with pytest.raises(ValueError, match="ᵊ"):
        bb.encode_original("ᵊ")


# ---------------------------------------------------------------- speaker matching

def test_matching_spreads_over_hnr_and_takes_the_nearest_unused_voice():
    vctk = {"p1": 1.0, "p2": 2.0, "p3": 3.0, "p4": 4.0, "p5": 5.0}
    libri = {10: 1.05, 11: 2.9, 12: 3.02, 13: 5.1, 14: 0.99}
    got = bb.match_speakers(vctk, libri, n=3, exclude=set(), max_gap=0.2)
    assert [m["vctk"] for m in got] == ["p1", "p3", "p5"]
    assert [m["libri"] for m in got] == [14, 12, 13]


def test_matching_never_reuses_a_voice_and_honours_the_exclusion():
    vctk = {"p1": 3.0, "p2": 3.0}
    libri = {10: 3.0, 11: 3.1, 12: 3.05}
    got = bb.match_speakers(vctk, libri, n=2, exclude={10}, max_gap=0.2)
    assert [m["libri"] for m in got] == [12, 11]


def test_matching_refuses_a_pair_too_far_apart_to_call_matched():
    with pytest.raises(SystemExit, match="p1"):
        # both inside the shared range, but nothing sits near p1
        bb.match_speakers({"p1": 3.0, "p2": 5.0}, {10: 1.0, 11: 5.0}, n=2,
                          exclude=set(), max_gap=0.25)


def test_matching_spreads_over_the_range_both_corpora_share():
    # p0 is rougher than any LibriTTS-R voice and p9 cleaner: no match exists for either,
    # so the spread runs over the overlap rather than refusing on the extremes.
    vctk = {"p0": 0.05, "p1": 1.0, "p2": 2.0, "p3": 3.0, "p9": 9.0}
    libri = {10: 1.0, 11: 2.0, 12: 3.0}
    got = bb.match_speakers(vctk, libri, n=3, exclude=set(), max_gap=0.2)
    assert [m["vctk"] for m in got] == ["p1", "p2", "p3"]


def test_prior_speakers_reads_every_bench_file_shape():
    docs = [{"items": {"s1": {"spk": 1}, "s2": {"spk": 2}}},
            {"items": [{"spk": 3}, {"note": "no speaker"}]},
            {"items": {"item_00": {"spk": "4"}}},
            {"items": {"item_01": {"spk": "p326"}}},   # a VCTK name is not an index
            {"rule": "no items at all"}]
    assert bb.prior_speakers(docs) == {1, 2, 3, 4}


# ---------------------------------------------------------------- the pre-registered test

def test_sign_flip_is_exact():
    assert bb.sign_flip_p([1] * 12) == pytest.approx(2 / 4096)
    assert bb.sign_flip_p([0, 0, 0]) == 1.0
    assert bb.sign_flip_p([1, -1]) == 1.0


def test_sign_flip_refuses_a_sample_too_large_to_enumerate():
    with pytest.raises(ValueError):
        bb.sign_flip_p([1] * 25)


@pytest.mark.parametrize("ours_p, stock, d, expected", [
    (0.01, (0.9, 0.01), (0.1, 0.70), "matcha"),
    (0.01, (0.1, 0.60), (1.2, 0.01), "lineage"),
    (0.01, (0.8, 0.02), (0.9, 0.03), "both"),
    (0.01, (0.3, 0.30), (0.4, 0.30), "inconclusive"),
    (0.30, (0.9, 0.01), (0.1, 0.70), "invalid"),
    # a stock gap significant in the WRONG direction is not "stock hums"
    (0.01, (-0.9, 0.01), (1.5, 0.01), "lineage"),
])
def test_outcome_follows_the_preregistered_rule(ours_p, stock, d, expected):
    got = bb.outcome(ours=(1.5, ours_p), stock=stock, matched=d)
    assert got == expected


@pytest.mark.parametrize("rt_means, readable", [
    ((1.0, 1.5), True),
    ((1.0, 2.0), False),    # the round trips sit a point apart
    ((4.0, 3.5), False),    # one round trip is already near the top of the scale
])
def test_the_matched_contrast_is_read_only_on_comparable_round_trips(rt_means, readable):
    got = bb.outcome(ours=(1.5, 0.01), stock=(0.1, 0.6), matched=(1.2, 0.01),
                     rt_means=rt_means)
    assert got == ("lineage" if readable else "inconclusive")


def test_flips_balance_within_each_family_and_oppose_within_each_match():
    import random
    flips = bb.balanced_flips(12, random.Random(3141))
    assert len(flips) == 12
    assert sum(st for st, _o in flips) == 6 and sum(o for _s, o in flips) == 6
    assert all(st != o for st, o in flips)


def test_closest_duration_takes_the_nearest_then_the_first_name():
    cands = [("b", 4.0), ("a", 4.0), ("c", 3.1)]
    assert bb.closest_duration(cands, 3.9) == "a"
    assert bb.closest_duration(cands, 3.0) == "c"
