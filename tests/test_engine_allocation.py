"""Guards for the three-layer engine allocation (ref_select, 2026-08-02).

These tables encode measurements, and a weight table is exactly the kind of thing
that gets hand-edited under time pressure and silently stops meaning what it says.
A share nudged to zero retires an engine from a lane and nothing anywhere notices;
a channel quietly added to ENGINE_CHANNELS re-opens a routing path the direction-
relay audit closed on source evidence.

Run: .venv/bin/python -m pytest tests/test_engine_allocation.py
"""
import os
import sys

import pytest

from scripts_layout import SCRIPTS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS.on_path()
import ref_select as rs  # noqa: E402
# TAKEN FROM THE CONTRACT, not restated. This file carried its own literal copy until
# 2026-08-11, and when `Documentary` was retired out of `NARRATION_LANES` the copy went on
# parametrizing the narration guards over a lane no builder may target — asserting that a
# mix exists for exactly the lane that must not have one. A test with its own vocabulary
# fails on the change instead of on the defect.
from matcha.delivery import NARRATION_LANES, RETIRED_LANES  # noqa: E402

ALL_LANES = NARRATION_LANES + ("Dialogue", "Speech")


def test_every_lane_mix_is_internally_consistent():
    """The import-time validator must actually be running."""
    rs._validate_mixes()


@pytest.mark.parametrize("lane", ALL_LANES)
def test_allocation_sums_exactly(lane):
    """Largest-remainder, so counts sum to n rather than drifting per lane."""
    for n in (1, 7, 100, 137):
        assert sum(rs.allocate_engines(n, lane=lane).values()) == n


@pytest.mark.parametrize("lane", NARRATION_LANES)
def test_no_narration_lane_collapses_to_one_voice(lane):
    """Layer 3. A teacher corpus that becomes one engine's timbre has failed its job.

    This is the guard that makes "allocate by measured keep rate" safe: without it
    the reliability ranking hands qwen the corpus, which is the natural thing to
    build and the wrong thing to want.
    """
    alloc = rs.allocate_engines(100, lane=lane)
    assert max(alloc.values()) <= rs.MAX_SHARE_NARRATION * 100 + 1
    assert sum(1 for v in alloc.values() if v > 0) >= 4, (
        f"{lane} draws on {sum(1 for v in alloc.values() if v > 0)} engines")


@pytest.mark.parametrize("lane", NARRATION_LANES)
def test_every_surviving_engine_keeps_a_share(lane):
    """Thin cells are how a lane earns evidence; 0% can never produce the data
    that would raise it. orpheus on Newscaster (5%, no data at all) is the deliberate
    case — a floor share placed TO generate the verdicts that would move it. It used to
    be named as moss_vg on Documentary (56%), a lane this test no longer covers."""
    alloc = rs.allocate_engines(100, lane=lane)
    for engine in rs.ENGINE_MIX:
        if engine in rs.SET_ASIDE:
            continue
        assert alloc.get(engine, 0) > 0, f"{engine} is starved in {lane}"


def test_set_aside_engines_never_allocated():
    for lane in ALL_LANES:
        alloc = rs.allocate_engines(1000, lane=lane)
        for engine in rs.SET_ASIDE:
            assert alloc.get(engine, 0) == 0


@pytest.mark.parametrize("lane", RETIRED_LANES)
def test_a_retired_lane_has_no_mix_and_cannot_be_allocated(lane):
    """Retirement has to reach the ALLOCATOR, not just the label (issue #12).

    Deleting `ENGINE_MIX_BY_LANE["Documentary"]` on its own would have made this quieter
    rather than safer: `mix_for_lane` falls back to the flat ENGINE_MIX for any key it
    does not know, so a stale campaign builder would have kept dealing five engines into
    a lane no row may be labelled with, and the failure would have surfaced at the corpus
    merge with the renders already bought. The refusal is the guard; the missing table is
    only bookkeeping.
    """
    assert lane not in rs.ENGINE_MIX_BY_LANE
    for spelling in (lane, lane.lower()):
        with pytest.raises(ValueError, match="RETIRED"):
            rs.mix_for_lane(spelling)
        with pytest.raises(ValueError, match="RETIRED"):
            rs.allocate_engines(10, lane=spelling)


@pytest.mark.parametrize("lane", RETIRED_LANES)
@pytest.mark.parametrize("pad", ["{} ", " {}", " {} ", "\t{}", "{}\t", "\n{}\n"])
def test_a_padded_retired_lane_is_refused_in_both_spellings(lane, pad):
    """Issue #46: the guard normalised differently from the contract, and fell OPEN.

    `mix_for_lane` built its own lowercase table and never stripped, while
    `matcha.delivery.check_assignable` — the function it defers to for the refusal and its
    message — opens with `str(lane).strip()`. The two therefore disagreed on any lane with
    surrounding whitespace, and the disagreement resolved toward allocating:
    `mix_for_lane(" Documentary")` returned the flat ENGINE_MIX and
    `allocate_engines(10, lane=" Documentary")` dealt {qwen 3, chatterbox 3, zonos 2,
    orpheus 1, moss_vg 1} into the retired lane.

    A padded lane is a realistic input, not an invented one: lanes arrive from JSON
    manifests and CSV cells, and `stage_pool` strips `r.get("delivery")` for exactly this
    reason. Both spellings are covered because the capitalised one reaches
    `_RETIRED_LANES` directly while the lowercase one has to survive the table lookup
    first, and only one normalisation step protects both.
    """
    for spelling in (pad.format(lane), pad.format(lane.lower())):
        with pytest.raises(ValueError, match="RETIRED"):
            rs.mix_for_lane(spelling)
        with pytest.raises(ValueError, match="RETIRED"):
            rs.allocate_engines(10, lane=spelling)


@pytest.mark.parametrize("pad", ["{} ", " {}", " {} ", "\t{}", "{}\t"])
def test_a_padded_live_lane_still_resolves_to_its_own_mix(pad):
    """The other half of #46, and the reason the fix is a normalisation rather than a
    second membership test: whitespace used to send a LIVE lane to the flat mix too.

    ` Newscaster` missed the lowercase table, missed `ENGINE_MIX_BY_LANE`, and fell back —
    silently rendering the lane with the flat mix instead of its measured one, which is a
    render-mix defect with no error and no symptom until someone compares the manifest to
    the weights. Whitespace-only stays `unknown`, which is the flat mix by contract.
    """
    for lane in ("Newscaster", "Dialogue", "Neutral"):
        for spelling in (pad.format(lane), pad.format(lane.lower())):
            assert rs.mix_for_lane(spelling) is rs.ENGINE_MIX_BY_LANE[lane]
    assert rs.mix_for_lane("   ") is rs.ENGINE_MIX


def test_the_lane_table_is_derived_from_the_vocabulary():
    """#46's latent half. The five-entry lowercase table was another restatement of the
    delivery vocabulary in the repo whose stated core defect class is a forked one — and
    it fails in the direction that matters: a lane appended to `DELIVERY_LANES` and later
    retired would not be in the table, would therefore not match `_RETIRED_LANES` after
    the lookup, and would fall silently back to the flat mix instead of raising."""
    from matcha.delivery import DELIVERY_LANES
    assert rs._LANE_BY_LOWER == {ln.lower(): ln for ln in DELIVERY_LANES}


def test_lane_names_resolve_both_conventions():
    """`lane="dialogue"` is the pre-2026-08-02 caller convention and must survive."""
    assert rs.mix_for_lane("dialogue") is rs.ENGINE_MIX_BY_LANE["Dialogue"]
    assert rs.mix_for_lane("Dialogue") is rs.ENGINE_MIX_BY_LANE["Dialogue"]
    # An unknown lane falls back rather than raising: a bank builder passing a
    # typo should render with the flat mix, not crash mid-campaign.
    assert rs.mix_for_lane("Nonsense") is rs.ENGINE_MIX
    assert rs.mix_for_lane(None) is rs.ENGINE_MIX


def test_prose_channel_is_only_the_instruct_engines():
    """Layer 1, the finding that matters most.

    vibevoice's `voice_design` is read by OUR casting code and never reaches the
    model — it looks like an instruction slot and is not one. Listing it here as
    prose would silently restore the routing the relay audit closed.
    """
    prose = {e for e, ch in rs.ENGINE_CHANNELS.items() if "prose" in ch}
    assert prose == {"qwen", "moss_vg", "moss85"}
    assert "prose" not in rs.ENGINE_CHANNELS["vibevoice"]
    assert "prose" not in rs.ENGINE_CHANNELS["chatterbox"]


def test_numeric_prosody_is_zonos_alone():
    """chatterbox's `exaggeration` is a rate PROFILE, not a scale you can direct
    with, so it is `coarse` — the distinction is the whole point of the field."""
    numeric = {e for e, ch in rs.ENGINE_CHANNELS.items() if "numeric" in ch}
    assert numeric == {"zonos"}


def test_channel_veto_drops_engines_that_cannot_hear_the_direction():
    kept, dropped = rs.route_engines_for({"prose"}, list(rs.ENGINE_MIX))
    assert set(kept) == {"qwen", "moss_vg"}
    assert {e for e, _ in dropped} == {"chatterbox", "zonos", "orpheus"}


def test_described_delivery_plus_reference_identity_is_unservable():
    """A real capability gap, asserted so it is noticed if it ever closes.

    No engine in the portfolio can BOTH take a described delivery and clone a
    reference voice. Any line needing both must be split across two renders or
    have one requirement dropped — and if a future engine serves both, this test
    failing is the notification.
    """
    kept, _ = rs.route_engines_for({"prose", "reference"}, list(rs.ENGINE_CHANNELS))
    assert kept == [], f"portfolio changed — {kept} now serves both"


def test_casting_ban_outranks_channel_veto():
    """An engine excluded for both reasons should report the stronger one."""
    kept, dropped = rs.route_engines(
        "middle-aged man", ["vibevoice", "zonos"], requires={"prose"})
    assert kept == []
    why = dict(dropped)
    assert "set aside" in why["vibevoice"]      # not "no prose channel"
    assert "no prose channel" in why["zonos"]


def test_legacy_two_argument_call_is_unaffected():
    kept, _ = rs.route_engines("young man, bright", list(rs.ENGINE_MIX))
    assert set(kept) == set(rs.ENGINE_MIX)
