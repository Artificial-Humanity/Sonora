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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "synthesis"))

import ref_select as rs  # noqa: E402

NARRATION_LANES = ("Neutral", "Documentary", "Newscaster")
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
    that would raise it. moss_vg on Documentary (56%) is the deliberate case."""
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
