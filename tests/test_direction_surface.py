"""The directable surface — is every dimension the model consumes actually vettable?

The standing rule is that every new model capability ships with a Vocalizer dial in the
same phase. That rule WAS a test, but the test named one dropdown, so it could only fail
for the capability in flight when it was written. It never fired for the speaker
dimension, which went 247 -> 2500 speakers with a dial still defaulting to a
247-era id and silently clamping out-of-range requests to a different valid voice.

These tests iterate `matcha.direction.DIRECTABLE` instead of naming widgets, so a
dimension added without a control fails here rather than being noticed by ear later.

Torch-free and Gradio-free on purpose: it reads vocalizer.py as text, the same crudeness
`test_delivery_channel` already used. The value is in the SET being complete, not in the
matching being clever — a source-token match cannot prove the dial WORKS, only that a
capability did not ship with no control at all, which is the failure this guards.
"""

import pathlib
import sys
import tokenize

import pytest
from scripts_layout import SCRIPTS  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS.on_path()
delivery = pytest.importorskip("matcha.delivery")
direction = pytest.importorskip("matcha.direction")


def code_only(path):
    """Source with comments and string literals blanked, layout preserved.

    Matching source TEXT cannot by itself tell code from prose ABOUT code, and that is not
    hypothetical: the first version of `test_the_speaker_dial_rejects_out_of_range...`
    failed against a correct file, because `bounded_speaker`'s docstring quotes the very
    clamp it replaced. A guard that a fix's own documentation can trip is a guard that
    gets silenced rather than believed — and the same hazard sits under every "must NOT
    appear" assertion of this style, including the one this file inherited.

    Blanking in place rather than dropping tokens keeps offsets and adjacency intact, so
    `"spk_input = gr.Number("` still matches as a contiguous string. It also makes the
    "must appear" assertions stricter for free: a control that exists only in a comment no
    longer counts as shipped.
    """
    grid = [list(line) for line in path.read_text(encoding="utf-8").splitlines(keepends=True)]
    with open(path, "rb") as handle:
        for tok in tokenize.tokenize(handle.readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                continue
            (start_row, start_col), (end_row, end_col) = tok.start, tok.end
            for row in range(start_row, end_row + 1):
                line = grid[row - 1]
                lo = start_col if row == start_row else 0
                hi = end_col if row == end_row else len(line)
                for i in range(lo, min(hi, len(line))):
                    if line[i] != "\n":
                        line[i] = " "
    return "".join("".join(row) for row in grid)


#: The vetting surface as CODE — see `code_only`. Every assertion below matches this, not
#: the raw file, so neither a comment nor a docstring can satisfy or break a check.
VOCALIZER = code_only(REPO / "vocalizer.py")

ALL = direction.DIRECTABLE
IDS = [d.name for d in ALL]


@pytest.mark.parametrize("dim", ALL, ids=IDS)
def test_every_directable_dimension_has_a_control(dim):
    """The standing rule, as a check rather than a memory."""
    assert dim.control in VOCALIZER, (
        f"{dim.name} is declared directable but has no control on the vetting surface. "
        f"Expected to find {dim.control!r} in vocalizer.py. A conditioning channel with "
        "no dial is one whose failure mode we learn about from a training run instead of "
        "from a listen."
    )


@pytest.mark.parametrize("dim", ALL, ids=IDS)
def test_no_dimension_uses_a_control_shape_that_misrepresents_it(dim):
    """A slider over categories invites interpolation; a dropdown over a z-score quantises
    it. The shape of the control is part of the claim it makes about the dimension."""
    for wrong in dim.forbidden:
        assert wrong not in VOCALIZER, f"{dim.name}: {wrong!r} misrepresents a {dim.kind} dimension — {dim.why}"


def test_the_declaration_covers_every_conditioning_channel():
    """THE ONE THAT MATTERS FOR THE NEXT CAPABILITY.

    If `vat_dim` grows and no dimension claims the new channels, this fails on the host
    before anything is trained. That is the property the old delivery-specific test could
    not have: it asserted a dropdown existed, which stays true no matter how many
    undeclared channels appear beside it.

    The gated emotion block (todo.md §3) would append as channels 8+; span-scoped delivery
    appends its own. Either one lands here as a declaration or it does not land.
    """
    assert direction.CHANNELS == tuple(range(delivery.VAT_DIM)), (
        f"channels {sorted(set(range(delivery.VAT_DIM)) - set(direction.CHANNELS))} are in "
        f"the conditioning vector but claimed by no dimension in matcha/direction.py. "
        "Widening vat_dim without declaring what the new channels MEAN is how a channel "
        "ships with no way to vet it."
    )


def test_the_delivery_block_does_not_absorb_appended_channels():
    """GUARD THE GUARD, for the flaw this file shipped with for an hour.

    The first version declared delivery as `VAT_BASE_DIM .. VAT_DIM` — the tail of the
    vector. Same numbers today, but it made `test_the_declaration_covers_every_conditioning
    _channel` a tautology: whatever anyone appended, delivery already claimed it, so
    coverage could never fail. Every test passed, including the coverage one, which is
    precisely why review did not catch it and a negative test did.

    Reverting to the tail form is therefore silent unless something simulates a wider
    vector and checks that delivery's claim does NOT grow with it.
    """
    import importlib

    original = delivery.VAT_DIM
    delivery.VAT_DIM = delivery.VAT_BASE_DIM + delivery.DELIVERY_DIM + 3
    try:
        widened = importlib.reload(direction)
        assert len(widened.dimension("delivery_lane").channels) == delivery.DELIVERY_DIM, (
            "delivery absorbed appended channels — it is claiming the tail of the vector "
            "rather than its own block, which makes the coverage check unfailable"
        )
        assert widened.CHANNELS != tuple(range(delivery.VAT_DIM)), (
            "the declaration auto-followed the width; coverage cannot fail"
        )
    finally:
        delivery.VAT_DIM = original
        importlib.reload(direction)


def test_no_two_dimensions_claim_the_same_channel():
    """Overlap means two dials write the same slot and the last one wins, silently."""
    claimed = [c for d in ALL for c in d.channels]
    assert len(claimed) == len(set(claimed)), f"channel claimed twice in {claimed}"


def test_delivery_is_declared_from_its_own_module_not_restated():
    """The vocabulary has exactly one owner. This module refers to it; it must not carry a
    second copy that can drift — "which number means Newscaster" is the rule that must not
    fork."""
    dim = direction.dimension("delivery_lane")
    # Its OWN block, not "to the end of the vector" — see the note in direction.py. The
    # tail form is the same numbers today and makes the coverage test unfailable.
    assert dim.channels == tuple(
        range(delivery.VAT_BASE_DIM, delivery.VAT_BASE_DIM + delivery.DELIVERY_DIM))
    assert len(dim.channels) == len(delivery.DELIVERY_LANES)
    src = (REPO / "matcha" / "direction.py").read_text(encoding="utf-8")
    for lane in delivery.DELIVERY_LANES:
        assert f'"{lane}"' not in src, (
            f"{lane!r} is spelled out in direction.py; the lanes belong to "
            "matcha/delivery.py and must be referenced, never restated."
        )


def test_speaker_is_bounded_per_checkpoint_and_not_by_a_constant():
    """Its range is `n_spks`, which changes with the corpus. A constant bound would be
    correct for exactly one generation — which is how it went unbounded through two."""
    dim = direction.dimension("spk_input")
    assert dim.channels == (), "speaker is an embedding lookup, not a conditioning channel"
    assert direction.speaker_bound(2500) == (0, 2499)
    assert direction.speaker_bound(247) == (0, 246)
    assert direction.speaker_bound(1) == (0, 0), "single-speaker ckpts have exactly id 0"


def test_the_speaker_dial_rejects_out_of_range_rather_than_clamping():
    """The defect this whole file exists because of. `render` used to clamp with
    `min(int(spk_id), n_spks - 1)`, so a request for speaker 5000 on a 2500-speaker
    checkpoint rendered speaker 2499 and reported success — a confident verdict about a
    voice nobody selected."""
    assert "min(int(spk_id), n_spks - 1)" not in VOCALIZER, "the silent speaker clamp is back"
    assert "bounded_speaker(spk_id, n_spks)" in VOCALIZER
    assert "def bounded_speaker(" in VOCALIZER


def test_the_stale_speaker_default_is_gone_from_both_surfaces():
    """245 was a valid LibriTTS-R val speaker under 247 rows and is an unrelated voice
    under 2500. A default that means someone different after each re-derivation is not a
    default."""
    assert "spk_id = 245" not in VOCALIZER, "HTTP API still defaults to the 247-era speaker"
    assert "value=245" not in VOCALIZER, "the Gradio dial still defaults to the 247-era speaker"


def test_the_comment_stripper_actually_strips(tmp_path):
    """Guard the guard. If `code_only` silently returned the raw source, every "must NOT
    appear" assertion above would keep passing while checking nothing — and the one that
    would have caught it is the one that was already failing for this exact reason."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        'x = 1  # min(int(spk_id), n_spks - 1)\n'
        '"""A docstring naming spk_input = gr.Slider( which does not exist."""\n'
        'spk_input = gr.Number(0)\n',
        encoding="utf-8",
    )
    stripped = code_only(sample)
    assert "min(int(spk_id)" not in stripped, "comment survived"
    assert "gr.Slider(" not in stripped, "docstring survived"
    assert "spk_input = gr.Number(" in stripped, "real code was destroyed"
    assert stripped.count("\n") == 3, "layout not preserved"


def test_the_stripper_changes_what_the_guards_see():
    """That stripping is not a no-op for THIS file, so the checks above are reading code
    rather than raw text.

    Deliberately not an assertion about any particular phrase. The first version of this
    test pinned a clamp expression quoted in `bounded_speaker`'s docstring, and then failed
    the moment that docstring was reworded — a guard that breaks when someone improves a
    comment is one that gets deleted rather than fixed.
    """
    raw = (REPO / "vocalizer.py").read_text(encoding="utf-8")
    assert raw != VOCALIZER, "nothing was stripped — the guards are matching prose too"
    assert len(raw) == len(VOCALIZER), "blanking must preserve layout, not delete"


def test_render_controls_are_not_confused_with_directed_ones():
    """`temperature`, `length_scale`, `n_timesteps` and `guidance` shape how a conditioning
    vector is SOLVED; they are not things a Director says. Declaring them here would make
    "is every directed dimension controllable" unanswerable."""
    assert not {"temperature", "length_scale", "n_timesteps", "guidance"} & set(IDS)
