"""The directable surface — every dimension a Director can address, declared once.

WHY THIS FILE EXISTS
--------------------
The standing rule is that **every new model capability ships with a Vocalizer dial in the
same phase**: a capability with no control cannot be vetted, and an unvetted conditioning
channel is one whose failure mode we learn about from a training run instead of from a
listen.

That rule was already a test — `test_the_vocalizer_ships_a_dial_in_the_same_phase` — but
it asserted *one specific dropdown* existed, so it only ever guarded the capability that
happened to be in flight when it was written. It could not fail for a channel it did not
name.

And it did not. Between them, `matcha/delivery.py` owns channels 3-7, V/A/T owns 0-2 by
convention, and the speaker index lives somewhere else entirely as `n_spks`/`spk_emb`.
Nothing enumerated them together, so:

  * `n_spks` went 247 -> 2500 in the v5 corpus and NOTHING failed. The Vocalizer's speaker
    dial still defaulted to 245 — a valid near-final index under 247 speakers, an
    unrelated voice under 2500 — and was unbounded on both surfaces. The Gradio
    `gr.Number` had no maximum, and the HTTP API read it with a bare `int(voice)` while
    every other control went through `bounded()`. That is precisely the defect E-M2 closed
    for V/A/T, still open for speaker, because no declaration said speaker was a directable
    dimension that needed the same treatment.

So the set is declared HERE, and the parity test iterates the declaration rather than a
hand-written list of widgets. Adding a dimension without a control now fails a test
instead of being noticed later by ear.

WHAT THIS FILE IS NOT
---------------------
It is NOT a second definition of any vocabulary. `matcha/delivery.py` remains the single
owner of the delivery lanes and of `VAT_DIM`; this module imports those and refers to
them. Two implementations of one rule drifting apart is the review's recurring finding,
and "which number means Newscaster" must not fork. Nothing here restates a lane name, a
width, or a bound that another module owns.

It is also NOT the render controls. `n_timesteps`, `temperature`, `length_scale` and
`guidance` shape HOW a given conditioning vector is solved; they are not things the
Director says. They are bounded by `CONTROL_BOUNDS` in vocalizer.py and are out of scope
here — mixing them in would make "is every directed dimension controllable" unanswerable.

ADDING A DIMENSION
------------------
Append a `Dimension` here in the same commit that teaches the model to consume it, and add
the matching control to vocalizer.py. `CHANNELS` must stay in step with `delivery.VAT_DIM`
— the coverage test fails if a channel exists that no dimension claims, which is what
makes a future channel block (the gated emotion group, ARCHITECTURE §1 / todo.md §3)
impossible to ship silently. Span-scoped delivery, when it unparks, is a NEW dimension
here rather than an edit to the delivery entry: it conditions a span, not a clip.
"""

from dataclasses import dataclass

from matcha import delivery


@dataclass(frozen=True)
class Dimension:
    """One thing a Director can say, and the control that must exist to vet it.

    `control` and `forbidden` are source tokens matched against vocalizer.py rather than
    imported widgets, because the test that reads them is torch-free and Gradio-free and
    runs on the host where `make test` runs. Crude, and deliberately the same crudeness
    the delivery test already used — the value is in the SET being complete, not in the
    matching being clever.
    """

    name: str
    kind: str                  # "continuous" | "categorical" | "identity"
    channels: tuple            # vat channel indices this occupies; () if not a vat channel
    control: str               # source token that MUST appear in vocalizer.py
    forbidden: tuple           # control forms that would misrepresent the dimension
    bound: str                 # where its range comes from — static, or resolved per checkpoint
    why: str                   # why this control shape and not another


#: Channels 0-2. Continuous, and bounded by DERIVATION rather than by taste: V/A/T are
#: per-speaker z-scores clamped at 2 sigma when the corpus is built, so +/-1 is already the
#: edge of the trained range. A slider is right because the axis is genuinely continuous.
_VAT = tuple(
    Dimension(
        name=name,
        kind="continuous",
        channels=(index,),
        control=f"{name} = gr.Slider(",
        forbidden=(f"{name} = gr.Dropdown(",),
        bound="CONTROL_BOUNDS",
        why=("continuous and real-valued; a dropdown would quantise an axis the corpus "
             "carries as a z-score"),
    )
    for index, name in enumerate(("valence", "energy", "tension"))
)

#: Channels 3-7. Categorical one-hot, vocabulary owned by `matcha.delivery`. A slider here
#: would invite interpolating between Newscaster and Dialogue, which has no meaning and
#: which `delivery.lane_of_vector` refuses outright.
#:
#: Its block is `VAT_BASE_DIM .. VAT_BASE_DIM + DELIVERY_DIM` and deliberately NOT
#: `VAT_BASE_DIM .. VAT_DIM`. The two are the same numbers today and they are not the same
#: claim: "to the end of the vector" would make delivery the owner of every channel anyone
#: ever appends, and the coverage test below a tautology that cannot fail. Written as its
#: own block, an appended group (the gated emotion channels, todo.md §3) lands unclaimed
#: and coverage fails — which is the entire point of declaring any of this. Caught by a
#: negative test, not by review: the tautological version passed everything.
_DELIVERY = Dimension(
    name="delivery_lane",
    kind="categorical",
    channels=tuple(range(delivery.VAT_BASE_DIM,
                         delivery.VAT_BASE_DIM + delivery.DELIVERY_DIM)),
    control="delivery_lane = gr.Dropdown(",
    forbidden=("delivery_lane = gr.Slider(",),
    bound="delivery.DELIVERY_LANES",
    why=("categorical modes of address; interpolation between lanes is meaningless and is "
         "refused when read back"),
)

#: NOT a vat channel — a lookup into the speaker embedding table, sized by the checkpoint.
#: Its bound therefore cannot be a constant: it is whatever `n_spks` the loaded checkpoint
#: reports, which is why it never made it into `CONTROL_BOUNDS` and why it went unbounded
#: for two corpus generations. Adjacency is meaningless (index 244 and 245 are unrelated
#: people), so a slider would imply an ordering the table does not have.
_SPEAKER = Dimension(
    name="spk_input",
    kind="identity",
    channels=(),
    control="spk_input = gr.Number(",
    forbidden=("spk_input = gr.Slider(",),
    bound="n_spks (per checkpoint)",
    why=("an index into a per-checkpoint embedding table, not a scale; neighbouring ids "
         "are unrelated voices, so it is bounded at load time and not by a constant"),
)

DIRECTABLE = _VAT + (_DELIVERY, _SPEAKER)

#: Every vat channel claimed by some dimension. The coverage test compares this against
#: `delivery.VAT_DIM`, so widening the conditioning vector without declaring what the new
#: channels MEAN fails on the host, before anything is trained.
CHANNELS = tuple(sorted(c for d in DIRECTABLE for c in d.channels))


def dimension(name):
    """-> the declared `Dimension` called `name`. Raises rather than returning None.

    A caller asking for a dimension that does not exist has a stale mental model, and
    handing back None lets it proceed with one.
    """
    for entry in DIRECTABLE:
        if entry.name == name:
            return entry
    raise ValueError(
        f"no directable dimension named {name!r}. The declared set is "
        f"{[d.name for d in DIRECTABLE]} — adding one is a contract change (see the "
        "module docstring and ARCHITECTURE.md §1)."
    )


def speaker_bound(n_spks):
    """-> (lo, hi) inclusive for a speaker id under a checkpoint with `n_spks` speakers.

    Single-speaker checkpoints have exactly one valid id, 0. Returning a range rather than
    a maximum keeps the caller from having to know that the table is zero-based.
    """
    return (0, max(0, int(n_spks) - 1))
