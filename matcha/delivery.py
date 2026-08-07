"""The delivery channel — Director↔Actor contract v2, §1 of notes/ARCHITECTURE.md.

ONE definition of how a delivery lane becomes numbers. Everything that produces or
consumes the 4th..8th conditioning channels goes through here: the corpus derivation, the
CLI, the Vocalizer, the export converter and the mobile host's config.json. The review's
recurring finding is two implementations of one rule drifting apart (B-L5's duplicated
`MAX_REF_EXCURSION`, D-L2's two disagreeing z-guards) — and "which number means
Newscaster" is precisely the rule that must not fork, because getting it wrong produces
fluent, confident, wrongly-delivered audio rather than an error.

WHY ONE-HOT AND NOT A SCALAR (owner call, 2026-08-07)
-----------------------------------------------------
`notes/todo.md` §1 proposed `vat_dim: 4` — the five lanes on a single ordered channel.
That encoding asserts the lanes lie on one continuum, and the corpus documentation says
they do not: `seed_delivery.py` records that **Dialogue vs Neutral is a property of the
TEXT** ("is this a character speaking?") while **Newscaster vs Documentary is a property
of the RENDER** ("an engine can deliver either from the same line"). Those are two
different kinds of distinction, and no ordering of the five is faithful to both.

A single channel also cannot cleanly carry `unknown`. The contract pins `unknown ≡ zero
vector ≡ v1 behavior`, and on one channel zero sits in the middle of the range — so the
five real lanes would have to be spaced asymmetrically around a hole, and "no label" would
be numerically adjacent to whatever lanes flank it. With one-hot, `unknown` is all five
channels at zero: not a value on the axis but the absence of one, which is what it means.

The cost is four extra input channels on a 1×1 conv — about a thousand parameters — and it
buys an encoding that makes no claim the data does not support.

    vat = [V, A, T, d_Dialogue, d_Neutral, d_Documentary, d_Newscaster, d_Speech]

`unknown` is the v1 vector padded with zeros, so a v1 consumer that knows nothing about
delivery produces exactly the conditioning it always did. That is the compatibility
property the contract asks for, and it is worth more than the four channels.

EMBODIMENT IS NOT A LANE (owner, 2026-08-02, ARCHITECTURE §1)
An embodiment clip is narration in which the narrator voices a character mid-sentence and
returns. Its delivery does not have a value; it CHANGES partway through. Such clips stay
delivery-blank — which here means `unknown`, all zeros — rather than acquiring a sixth
label that would teach the model embodiment is a uniform manner of speaking. Span-scoped
delivery is a later phase; the clip-level channel ships first.

Vocabulary changes are contract changes and require an owner call.
"""

# Order is the wire format. Appending is a contract change; REORDERING silently
# reinterprets every checkpoint and every filelist ever written, so it is worse than
# appending and there is no situation in which it is the right edit.
DELIVERY_LANES = ("Dialogue", "Neutral", "Documentary", "Newscaster", "Speech")

# The label `unknown` carries in ratings.csv and the corpus filelists: a blank cell.
# `seed_delivery.py` deliberately leaves the ear's cases blank rather than guessing, and
# every LibriTTS clip is blank because that corpus predates the axis entirely.
DELIVERY_UNKNOWN = ""

# Which lanes are NARRATION — a subset of the vocabulary, not a second copy of it, and a
# property of the vocabulary rather than of any render mix. It lived in `book_ingest` and
# again in `ref_select`; those two import each other, so neither could own it without a
# cycle. It belongs here, where the vocabulary is.
NARRATION_LANES = ("Neutral", "Documentary", "Newscaster")

VAT_BASE_DIM = 3                                  # valence, energy, tension
DELIVERY_DIM = len(DELIVERY_LANES)
VAT_DIM = VAT_BASE_DIM + DELIVERY_DIM             # 8


def delivery_index(lane):
    """-> position of `lane` in the one-hot block, or None for unknown.

    Raises on an unrecognised NON-EMPTY label rather than treating it as unknown. A typo
    ("Newscasterr", "dialogue") would otherwise become a silently unconditioned clip that
    still trains, and the corpus balance report would count it as unlabelled — a defect
    with no symptom. Blank is the only spelling of "no label".
    """
    if lane is None:
        return None
    lane = str(lane).strip()
    if lane == DELIVERY_UNKNOWN:
        return None
    if lane not in DELIVERY_LANES:
        raise ValueError(
            f"unknown delivery lane {lane!r}. The vocabulary is closed at "
            f"{list(DELIVERY_LANES)} (+ blank for unknown) — see ARCHITECTURE.md §1, "
            "where changing it is a contract change requiring an owner call. If this is "
            "an embodiment clip, it is deliberately BLANK, not a sixth lane."
        )
    return DELIVERY_LANES.index(lane)


def delivery_onehot(lane):
    """-> a `DELIVERY_DIM`-long list of floats. Unknown is all zeros."""
    vec = [0.0] * DELIVERY_DIM
    index = delivery_index(lane)
    if index is not None:
        vec[index] = 1.0
    return vec


def vat_vector(valence, energy, tension, lane=None):
    """-> the full `VAT_DIM` conditioning vector for one clip.

    The ONE place V/A/T and delivery are concatenated. Callers that build the list
    themselves are how the slot map drifts.
    """
    return [float(valence), float(energy), float(tension)] + delivery_onehot(lane)


def lane_of_vector(vec):
    """Inverse of `vat_vector`'s delivery block -> lane name, or `DELIVERY_UNKNOWN`.

    For reading a filelist or a manifest back. Refuses a vector that is not a valid
    one-hot: more than one lane set means two labels were merged somewhere upstream, and
    a fractional value means someone interpolated a categorical.
    """
    block = list(vec[VAT_BASE_DIM:VAT_DIM])
    if len(block) != DELIVERY_DIM:
        raise ValueError(
            f"expected {VAT_DIM} channels, got {len(vec)} — the filelist and the model "
            "config disagree on vat_dim (see notes/todo.md §1)")
    hot = [i for i, v in enumerate(block) if v]
    if not hot:
        return DELIVERY_UNKNOWN
    if len(hot) > 1 or any(v not in (0.0, 1.0) for v in block):
        raise ValueError(
            f"delivery block {block} is not one-hot. Delivery is CATEGORICAL: more than "
            "one lane set means two labels were merged upstream, and a fractional value "
            "means someone interpolated between categories, which has no meaning here."
        )
    return DELIVERY_LANES[hot[0]]
