"""The delivery channel — Director↔Actor contract v2, §1 of docs/ARCHITECTURE.md.

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

**THE 8 IS NOT THE EIGHT VAD OCTANTS.** It is 3 + 5: three continuous channels plus five
one-hot lanes. The collision is a coincidence and a trap, because our continuous triple
genuinely is VAD-adjacent — the affective-computing octants are the sign combinations of
a 3-axis space (2³ = 8 regions of ±valence/±arousal/±dominance), a partition of ONE
continuous space. Nothing here is a partition of anything: channels 0-2 are continuous and
real-valued, channels 3-7 are a categorical block, and no arithmetic relates the two
groups. Asked and answered on the day the width landed, recorded here because the
inference is an easy one to make from `vat_dim: 8` alone.

Our third continuous axis is **tension** (pressed ↔ breathy phonation), not dominance —
see vat-channels.md § T. It was rescoped LAX ↔ TIGHT on 2026-07-20 precisely because the
VAD reading of it was wrong.

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
#
# THIS IS NOT AN EMOTION TAXONOMY, and if you came here to add one, read todo.md § 3
# first. A categorical emotion block (the standard 3-continuous + 8-Plutchik hybrid) is an
# OPEN question, gated on whether Phase 1's corpus growth fixes the valence channel — the
# current diagnosis is a data limit, not a representation limit, and adding channels
# before that read would confound the two. If it is ever taken it APPENDS as channels 8+,
# a separate group with its own name, not extra entries here: these five are modes of
# ADDRESS (who is being spoken to), which is orthogonal to affect.
DELIVERY_LANES = ("Dialogue", "Neutral", "Documentary", "Newscaster", "Speech")

# Lanes that keep their CHANNEL but may no longer be ASSIGNED. `Documentary` was retired
# into `Neutral` (owner, 2026-08-10): **154 clips in `ratings.csv`, and all 154 went to
# `Neutral`** — Newscaster is 87 before and after. Every one of them was synthetic, so the
# lane was a property of the render brief, not of a manner of speaking.
#
# ⚠ Corrected 2026-08-11. This comment used to say "82 clips split bimodally toward
# Neutral and Newscaster". Both halves were wrong: 82 is the lane's share of the v6 APPEND
# SET (a subset), not the lane, and nothing split — measured against
# `ratings.csv.bak-20260810-documentary-retire`, Neutral goes 524 → 678 (+154) and every
# other lane is unchanged. v6's derivation report says the same from the other side:
# Newscaster stays 76 across the retirement while Neutral goes 251 → 328.
#
# It stays inside `DELIVERY_LANES` because ORDER IS THE WIRE FORMAT. Removing it would
# renumber Newscaster and Speech and silently reinterpret every filelist and every
# checkpoint written at `vat_dim` 8 — `ep019` warm-starts from one. So the channel stays
# and the label retires, and those are two different statements that need two different
# names here. Without this constant "Documentary is retired" is a fact about `ratings.csv`
# that no build can check: `lane not in DELIVERY_LANES` is False for it, so a guard
# written that way passes a retired lane straight through to the one-hot.
RETIRED_LANES = ("Documentary",)

# What a NEW row may be labelled. Producers validate against this; READERS
# (`delivery_index`, `delivery_onehot`, `lane_of_vector`) deliberately stay on the full
# vocabulary, because v5's filelists and ep019's weights both carry the retired channel
# and must keep decoding.
ACTIVE_DELIVERY_LANES = tuple(ln for ln in DELIVERY_LANES if ln not in RETIRED_LANES)

# The label `unknown` carries in ratings.csv and the corpus filelists: a blank cell.
# `seed_delivery.py` deliberately leaves the ear's cases blank rather than guessing, and
# every LibriTTS clip is blank because that corpus predates the axis entirely.
DELIVERY_UNKNOWN = ""

# Which lanes are NARRATION — a subset of the vocabulary, not a second copy of it, and a
# property of the vocabulary rather than of any render mix. It lived in `book_ingest` and
# again in `ref_select`; those two import each other, so neither could own it without a
# cycle. It belongs here, where the vocabulary is.
#
# A RETIRED LANE FILTERS OUT OF IT (Documentary, 2026-08-11 — issue #12). Narration lanes
# are lanes new lines are DIRECTED INTO, so a lane no new row may carry cannot be one, and
# deriving the tuple rather than restating it means retiring a narration lane later cannot
# leave a stale member behind.
#
# THIS IS AN ACCEPTED BEHAVIOUR CHANGE, not bookkeeping — owner call, and recorded here
# because it is not a consequence the retirement forces on its own. Two live rules read
# this tuple, and both stop applying to Documentary:
#   * `ref_select.MAX_SHARE_NARRATION` — the 30% per-engine cap that stops a narration
#     lane collapsing into one engine's timbre. Dropping a lane from the narration set
#     therefore reshapes future campaign balance, which is a render-mix decision the
#     retirement did not itself argue for.
#   * `book_ingest.build_direction` — zonos's emotion vector is forced off and
#     chatterbox's exaggeration capped only on a narration lane.
# Both are now unreachable in practice: `ENGINE_MIX_BY_LANE` carries no Documentary mix,
# `ref_select.mix_for_lane` refuses the lane outright, and no bank builder targets it. The
# change is recorded rather than silent because a reader finding a narration rule that no
# longer fires deserves to find the decision, not infer it.
NARRATION_LANES = tuple(
    ln for ln in ("Neutral", "Documentary", "Newscaster") if ln not in RETIRED_LANES)

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


def check_assignable(lane):
    """-> the normalised lane, or raises if it may not be put on a NEW row.

    Deliberately NOT folded into `delivery_index`. Indexing is a READ of the wire format
    and has to keep working for a retired lane; assignment is a WRITE and has to refuse
    one. A corpus builder calls this, a filelist reader does not.

    Raising rather than dropping is the point: these rows are ear-certified, and a lane
    that is no longer assignable is a stale metadata source, which is a fact about the
    INPUT. Dropping would move the append count with no record — the exact failure the
    surrounding merge discipline exists to prevent.
    """
    if lane is None:
        return DELIVERY_UNKNOWN
    lane = str(lane).strip()
    if lane == DELIVERY_UNKNOWN:
        return lane
    if lane in RETIRED_LANES:
        raise ValueError(
            f"delivery lane {lane!r} is RETIRED and cannot be assigned to a new row. "
            f"Its channel remains in the wire format at position "
            f"{DELIVERY_LANES.index(lane)} so existing filelists and checkpoints still "
            f"decode, but the assignable vocabulary is {list(ACTIVE_DELIVERY_LANES)} "
            f"(+ blank for unknown). Rows still carrying it mean the metadata source is "
            f"stale — re-read it rather than reassigning the lane here."
        )
    delivery_index(lane)                       # same message for an unrecognised lane
    return lane


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
            "config disagree on vat_dim (see scripts/gates/test_vat_dim_seams.py)")
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
