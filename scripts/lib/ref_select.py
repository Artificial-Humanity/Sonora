"""Reference-clip casting for cloning engines (VibeVoice) — v3d-proven logic.

Given a direction design text + intended VAT, selects an audited keep from the
certified dataset (gender parse + intended-VAT proximity + duration window +
casting-faithful engine preference). This is the production home of the logic
piloted in make_v3d_bank.py.

The v3d/v3e limitation this was written under — "the pool is own-synthesis keeps
whose heritage skews young; age fidelity is bounded by the pool, not the cloning
engine" — is RESOLVED as of 2026-08-02. The upgrade this docstring asked for
(real-speech pools) landed: `reference-pool-v2` is 243 LibriTTS-R references, one
per speaker, merged with the 193 synthetic clips whose expressive registers still
have no real-audio equivalent. Built by `build_reference_pool.py`.
"""
import json
import math
import re
from pathlib import Path

# v2 (2026-08-02). v1 was 193 clips, ALL synthetic, from moss85/qwen/dia/longcat —
# and three of those four engines are retired, so we were cloning voices for the
# current five from clips made by engines we no longer use. Of 16 narration-register
# references only 4 came from a live engine, and the narration lanes exhausted the
# male side mid-build. v2 takes neutral_narration from 16 to 259 and its gender
# balance from 6 F / 10 M to 120 F / 123 M. v1 is untouched on disk.
POOL_PATH = Path("/data/model-training/datasets/reference-pool-v2/metadata.jsonl")
POOL_ROOT = POOL_PATH.parent
ACOUSTICS_PATH = POOL_ROOT / "pool_acoustics.json"
# Lower is preferred. `libritts-r` is NEGATIVE, and that is the half of the pool
# switch that actually does the work: an unlisted engine defaults to 0.2, so real
# speech would have ranked BELOW moss85 (0.0) and longcat (0.05) — neither is a live
# engine. ("retired" here previously; longcat is BENCHED, not retired — the distinction
# and the condition that would end it live in
# notes/teacher-tts-audition-shortlist.md § Benched engines. Do not restate a status here.)
# Measured over 18 narration casts against the merged pool: unlisted gave
# moss85 9 / longcat 3 / libritts-r 6; at -0.10 it gives libritts-r 18. Building
# the pool without this line leaves two thirds of narration casting on engines we
# do not use. Real recorded speech is the better voice reference on its face — the
# synthetic entries are kept for registers real audio does not yet cover, not
# because they are preferable.
ENGINE_PREF = {"libritts-r": -0.10, "moss85": 0.0, "longcat": 0.05,
               "qwen": 0.15, "dia": 0.3}

# Age is carried by the reference's acoustics (v3d/v3e finding: renders copy the
# reference F0 register within ~2%), so the design's age band maps to an F0
# percentile target WITHIN gender. Crude but directionally correct until the
# measured casting norms land (casting-attribute-norms brief).
AGE_BANDS = [
    (r"\b(child|little (girl|boy)|kid)\b",          0.95),
    (r"\b(teen|adolescent|girlish|boyish)\b",       0.80),
    # 0.70 = the "adult" band. `young` is the classic trigger, but designs routinely say
    # "adult male" or give a decade ("mid-30s", "in her twenties") and those matched
    # NOTHING before 2026-07-27 — 23 of 39 historical rows recorded no age band despite
    # naming one in plain English.
    (r"\b(young|adult|twenties|thirties|20s|30s)\b",  0.70),
    (r"\b(middle.?aged|matronly|mature|forties|fifties|40s|50s)\b", 0.30),
    (r"\b(elderly|old (woman|man|lady)|aged|weathered|grandmother|grandfather)\b", 0.10),
]
AGE_WEIGHT = 0.8

_pool = None
_acoustics = None
_f0_pct = None


def _load_acoustics():
    global _acoustics, _f0_pct
    if _f0_pct is None:
        _acoustics = json.loads(ACOUSTICS_PATH.read_text()) if ACOUSTICS_PATH.exists() else {}
        _f0_pct = {}
        by_gender = {}
        for k in _load_pool():
            a = _acoustics.get(k["file"])
            if a:
                by_gender.setdefault(k.get("gender", "?")[:1].upper(), []).append((a["f0_median"], k["file"]))
        for g, vals in by_gender.items():
            vals.sort()
            n = max(len(vals) - 1, 1)
            for i, (_, f) in enumerate(vals):
                _f0_pct[f] = i / n
    return _f0_pct


# Owner taxonomy (casting-attribute-norms-brief.md): child / teen / adult /
# middle-aged / elderly. The 0.70 band is named "adult" to match the brief — note the
# TRIGGER WORD in a design is still "young" (that is what the regex matches and what a
# director writes); only the recorded band name is "adult".
AGE_BAND_NAMES = {0.95: "child", 0.80: "teen", 0.70: "adult", 0.30: "middle-aged", 0.10: "elderly"}


def design_age_target(design: str):
    d = (design or "").lower()
    for pat, target in AGE_BANDS:
        if re.search(pat, d):
            return target
    return None  # unspecified: no age term applied


def design_age_band(design: str):
    """Canonical age label for training attribution (owner taxonomy).

    Returns "" when the design names no age at all. Previously this returned "adult"
    for BOTH an explicit young/adult design and an unmarked one — and audit_sampler
    skips "adult" as "no age claim to betray", so explicitly-aged designs were being
    silently exempted from the age-mismatch check along with the genuinely unmarked
    (found 2026-07-27). An unmarked design has no claim; an explicit one does.
    """
    t = design_age_target(design)
    return AGE_BAND_NAMES[t] if t is not None else ""


def _load_pool():
    global _pool
    if _pool is None:
        _pool = [json.loads(l) for l in POOL_PATH.open()]
    return _pool


def design_gender(design: str) -> str:
    d = (design or "").lower()
    return "F" if re.search(r"\b(female|woman|maternal|girl)\b", d) else "M"


def _vat(v):
    """The three axes as floats, or **None** where the axis carries no usable number.

    ⚠ `None` MEANS "NO INFORMATION" AND THE RANKING MUST SKIP IT — it is not `0.0`, and the
    difference is the whole point. This returned `0.0` for an unknown axis until 2026-08-18,
    and the docstring here claimed that "contributes nothing to the distance". It did not:
    `select_reference` computes `(want[a] - kv[a]) ** 2`, so a `0.0` target contributes
    `kv[a] ** 2` — the LARGEST term for the most strongly-labelled candidates — and
    minimising it selects the reference closest to neutral. Measured on the 436-clip pool
    for a tag with V known and A/T absent: the top ten averaged |A|+|T| of **0.252**, against
    **0.219** for the pool at large, so the ranking preferred references FLATTER than average
    on the two axes nobody had said anything about (issue #103).

    ⚠ DELEGATES TO `schemas.coerce_axis`, WHICH IS THE SINGLE DEFINITION of what counts as a
    number. This function kept its own `isinstance(c, (int, float))` test, which is the exact
    test issue #58 was filed against: it reads `"0.7"` — a numeric string, which #58 ruled IS
    a label — as no-information (issue #104). `qc_verdict.coerce_axis` was rewritten into a
    one-line delegation nine commits earlier for this reason; a third copy went in anyway.

    ⚠ A PRESENT-BUT-NULL KEY IS THE TRAP, and `.get(k, 0)` does not catch it (issue #92).
    `{"V": None}.get("V", 0)` is `None`, not `0` — the default only fires when the key is
    MISSING. `intended.V/A/T` may be null since 2026-08-17, so `want[a] - kv[a]` in
    `select_reference` raised `TypeError`. The three synth callers catch
    `(LookupError, ValueError)`, which does not include `TypeError`, so one null axis killed
    that engine's entire remaining bank rather than one clip.

    The alias chain is a fall-through, not a first-key-wins: an unreadable `V` still lets
    `valence` answer, which is what the `isinstance` version did and is worth keeping.

    ⚠ IT TAKES THE AXIS DICT, NOT THE RECORD THAT CONTAINS ONE, and being handed the
    wrong one of those is a silent no-op rather than an error — so it is refused here
    (issue #107). `make_v3d_bank` kept a private `keep_vat` that took a keeps RECORD and
    did the `["intended_vat"]` itself; `98734f9` pointed the alias at this function to stop
    the fork duplicating helpers, and left both call sites passing records. Every axis then
    read as absent, for all 193 keeps, and a constant cancels out of a distance comparison
    — so the ranking silently stopped happening and a bank still came out. **A shared
    helper is only shared if its argument is the same thing.**
    """
    if isinstance(v, dict) and "intended_vat" in v:
        raise TypeError(
            "_vat was handed a record, not an axis dict: it carries an 'intended_vat' key, "
            "so you almost certainly want _vat(rec['intended_vat']). Passing the record "
            "reads every axis as absent and the caller ranks on a constant (issue #107).")

    def _n(*candidates):
        for c in candidates:
            n = schemas.coerce_axis(c)
            if n is not None:
                return n
        return None
    return {"V": _n(v.get("V"), v.get("valence")),
            "A": _n(v.get("A"), v.get("arousal"), v.get("energy")),
            "T": _n(v.get("T"), v.get("tension"))}


def _vat_distance(want, kv):
    """Euclidean distance over the axes BOTH sides label, rescaled to three axes.

    ⚠ THE RESCALE IS WHAT MAKES CANDIDATES COMPARABLE. Summing over only the shared axes
    would hand a systematic advantage to whichever side labelled fewer of them — two axes of
    error always sum to less than three. `* 3 / len(axes)` restores the missing terms at the
    mean squared error of the ones present, which is the standard treatment and, more
    usefully here, is **exactly identity when all three axes are present**: measured across
    the 197 in-window female clips, ordering unchanged and max score delta 2.2e-16. So this
    does not recalibrate the ranking against `ENGINE_PREF` or `AGE_WEIGHT`.

    ⚠⚠ "NO SHARED AXIS" IS TWO DIFFERENT SITUATIONS AND THIS RETURNED `0.0` FOR BOTH
    (issue #114). `0.0` is the FLOOR of a quantity `select_reference` MINIMISES, so it is not
    a neutral value — it is the best score available. Handing it to one candidate does not
    drop the term out of the ranking; it exempts that candidate from a cost every other
    candidate still pays. The two cases:

      * **the TARGET labels nothing** — every candidate scores `0.0`, they all tie, and the
        term really does drop out. That is correct and is what the sentence meant.
      * **the target labels something this candidate does not** — only that candidate scores
        `0.0`, so a reference clip with no labels at all TIES A PERFECT MATCH and BEATS every
        labelled clip that is off by any amount. Measured: want (0.8, 0.8, 0.65) against a
        perfect match scores 0.0, against an unlabelled candidate 0.0, against a close-but-
        inexact candidate 0.4924.

    This is #103 one level up, in the function written to fix #103: an unknown replaced by a
    number that happens to mean "ideal". The direction of the bias even inverted — before
    `9bb3607` an unlabelled candidate scored ‖want‖ and was PENALISED in proportion to how
    expressive the target was; then it was REWARDED in the same proportion.

    An unrankable candidate now costs the worst its labelled axes could justify. Finite on
    purpose, so it is still selectable when the pool offers nothing else — quietly shrinking
    the pool is the failure this module keeps paying for — but never preferred over a clip
    that actually carries labels.
    """
    wanted = [a for a in "VAT" if want.get(a) is not None]
    if not wanted:
        return 0.0
    axes = [a for a in wanted if kv.get(a) is not None]
    if not axes:
        return math.sqrt(sum(max(abs(want[a] - schemas.AXIS_MIN),
                                 abs(want[a] - schemas.AXIS_MAX)) ** 2 for a in wanted)
                         * 3.0 / len(wanted))
    return math.sqrt(sum((want[a] - kv[a]) ** 2 for a in axes) * 3.0 / len(axes))


# Engines that must not receive bright / teen female casting. **EMPTY as of 2026-07-29**,
# and deliberately kept as a mechanism rather than deleted.
#
# Chatterbox was listed here for a few hours, on the theory that its voice-split made
# bright/teen female casting unusable. A blind 16-clip audition then showed median pitch
# does not predict the split at all (227 Hz split, 335 Hz did not) — pitch EXCURSION does.
# So the constraint belongs at casting granularity, not engine granularity:
# `synth_chatterbox.MAX_REF_EXCURSION` refuses swooping references while leaving
# steady-but-high ones available, which keeps 61% of the female pool against 38% under
# the engine-level ban.
#
# Bright/teen female therefore renders on Chatterbox again. Zonos and Qwen remain good
# homes for it (Qwen only with an instruct that holds it out of "fairyland" — see
# director_skills/qwen.md), but they are no longer the ONLY homes.
NO_BRIGHT_FEMALE = {}
BRIGHT_FEMALE_ENGINES = ("zonos", "qwen", "chatterbox")


def is_bright_female(design: str) -> bool:
    """True when a casting call asks for a young/bright/high female voice.

    This is the casting shape that breaks Chatterbox. Deliberately generous: it fires on
    the age words AND on the timbre words a director actually writes, because a design
    reading "young woman, bright open timbre" names no age band the regex table catches
    but is exactly the case at issue.
    """
    if design_gender(design) != "F":
        return False
    t = design_age_target(design)
    if t is not None and t >= 0.70:          # child / teen / young-adult bands
        return True
    return bool(re.search(r"\b(bright|light|airy|girlish|youthful|high|sweet)\b",
                          (design or "").lower()))


# Engines restricted to NARRATION only — they must not take `Dialogue` delivery.
# Owner decision 2026-07-29, from stress-v1 plus the full audit history.
#
# These two are long-form multi-speaker models, and given emotional dialogue they do not
# merely voice the line — they infer a SCENE and populate it. Measured in stress-v1: a
# crowd cheering behind "We did it!", a piano behind a flood warning, a second actor
# taking the second half of "oh, the third door!". The owner's summary is the mechanism:
# VibeVoice "can be extremely good but it also ad libs and takes enormous liberties."
# Hand it narration and there is no scene to infer, so it simply reads.
#
# The keep rates split hard along exactly that line, over the whole audit history:
#   vibevoice   Dialogue 54%   Neutral 94%   Newscaster 100%   Documentary 100%
#   dia         Dialogue 43%   Neutral 88%
#
# ⚠ Do NOT read this as "drop these engines". They carry the corpus's narration mix:
# dropping them outright loses 70 registers, halves Neutral, takes Newscaster 8 -> 1 and
# pushes the corpus from 82% to 89% Dialogue. Restricting them instead retains 90% of
# keeps AND improves the mix to 79% Dialogue — better balance than doing nothing.
# (stress-v1 measured 6/10 and 8/10 failures, but that campaign was 59/60 Dialogue, i.e.
# these engines judged solely on their worst mode. The rates are real for dialogue and
# do not generalise.)
NARRATION_ONLY = {
    "vibevoice": "ad-libs scenes on dialogue (crowd/music/second actor); 54% vs 94% neutral",
    "dia": "degrades on dialogue (screech, grinding, mechanical laughter); 43% vs 88% neutral",
}
# "Speech" (added 2026-07-29: public address — rally, sermon, toast) routes with
# Dialogue: it is performed character speech to an audience, and an audience is
# exactly what VibeVoice's scene-staging invents (the stress-v1 triumphant line got
# a crowd cheer). Untested directly — reassess if these engines are reinstated.
DIALOGUE_DELIVERIES = ("Dialogue", "Speech")

# Engines set aside entirely. Owner decision 2026-07-29, superseding the narration-only
# restriction above (which is kept because the analysis behind it stays true and the rule
# is what we reinstate if these come back).
#
# The narration-only compromise assumed the surviving engines could not cover what these
# two contribute. The audit history says otherwise: on `Neutral` delivery the survivors
# keep 49/52 = 94%, against VibeVoice's own 45/48 = 94% — same rate, comparable n. They
# have simply not been ASKED to narrate. And the "VibeVoice carries Newscaster" worry
# rested on 7 clips corpus-wide, which is not a lane anyone has built yet.
#
# The owner's argument is the decisive one, and it is precedent, not speculation: before
# the Gemma director skill files existed, Chatterbox and Qwen both looked narrow, and both
# turned out not to be. Qwen went from a naive mean of 2.0 to 5.0 on the same day the
# grounded instruct was written. Judging what the remaining engines cannot do BEFORE
# writing them a narration skill file would repeat exactly that error.
#
# Reversible on purpose: clear an entry and the engine returns under NARRATION_ONLY.
SET_ASIDE = {
    "vibevoice": "set aside 2026-07-29 — ad-libs scenes; survivors cover narration at 94%",
    "dia": "set aside 2026-07-29 — dialogue degradation; 88% neutral rests on only 17 clips",
}

# --- Render allocation (owner 2026-07-31) -------------------------------------
# A TIER IS A TAG ON AN ENGINE, NOT AN ALLOCATION BUCKET (owner, 2026-07-31). It
# says how much verification that engine needs — nothing about how much of the
# corpus it deserves. Two engines can share a tier and warrant very different
# shares; one engine can be a whole tier and warrant a small one. Keep the two
# ideas in separate tables and neither can quietly redefine the other.
#
# HOW MANY lines each engine gets. Deliberately PER ENGINE, not per audit tier.
# The owner caught the flaw in a tier-based split: the standard tier holds exactly
# one engine, so "50% trusted / 30% standard / 20% scrutinized" hands orpheus 30% —
# a larger single share than either trusted engine's 25% — and orpheus has the
# thinnest evidence base of the five. Tiers govern how much gets AUDITIONED
# (pick_audit_subset.TIERS); this governs how much gets RENDERED. Conflating them
# lets audit policy silently reshape the corpus.
#
# There was nowhere in the tree for this before, which is the real bug: the
# delivery-v1-narration mix (52% trusted / 9% standard / 39% scrutinized) was never
# decided, it accumulated. Bank builders must read this rather than hand-rolling a
# split, the same reason route_engines exists.
#
# Shares are ordered by measured PRODUCTION failure rate (probe and stress campaigns
# excluded — they are adversarial by construction and would libel every engine):
#   chatterbox 0/38 · qwen recent 0/37, 1/83, 2/20 · orpheus 1/19 · moss_vg 16/68 · zonos 20/64
ENGINE_MIX = {
    "qwen":       0.275,   # trusted; earned across many campaigns
    "chatterbox": 0.275,   # trusted, but PROVISIONAL — the whole claim is 38 clips in
                           # one campaign, and its SPLIT reverb is ear-only (four
                           # detectors failed). Watch its 3% tail for two batches.
    "zonos":      0.200,   # NORMAL tier since 2026-08-04, and the ONLY engine with
                           # numeric prosody dials (pitch_std / speaking_rate) — a
                           # capability the delivery-as-FiLM-channel work needs. The
                           # old 31% failure rate averaged two populations; conditioned
                           # correctly it measured 93.7%, then 37/38 by ear on r2.
    "orpheus":    0.150,   # standard. A step up from 8.8%, NOT the 30% a tier-based
                           # split implied: 5% rests on n=19, and its probe record is
                           # 29%. It earns more by surviving one round at this share.
    "moss_vg":    0.100,   # scrutinized and reduced: it overlaps qwen's instruct-driven
                           # niche while failing 24% to qwen's recent ~1%.
}


# Dialogue draws from a DIFFERENT mix than narration (owner 2026-08-01, on ebooks with
# no LibriVox audio: "we may want to lean on our most directable models for these ...
# where we are dealing more dramatic and quoted content").
#
# The reason is structural, not preference. A narration line needs one steady voice; a
# quoted line needs per-line CASTING and DELIVERY — which character, in what state —
# and that can only be relayed to an engine that HAS an instruction slot. The
# direction-relay audit (2026-07-25) established the slots are unevenly distributed, and
# routing dialogue by the flat mix spends 35% of it on the two engines that can hear the
# least of what the Director says.
#
# Narration keeps ENGINE_MIX: breadth of voice matters more there, and zonos's numeric
# prosody dials still serve the delivery-as-FiLM-channel work.
#
# ENGINE MIX IS NOT A LANE BUDGET (owner 2026-08-01: "I don't expect these lanes to be
# synchronous in count. Dialog and Narration very well should be the largest categories,
# by far"). These weights say WHICH ENGINES render a line of a given kind; they say
# nothing about how many lines of that kind exist. Lane SIZE comes from the ratified
# delivery mix (Dialogue 50 / Neutral 30 / Documentary 8 / Newscaster 6 / Speech 6) and
# from the source text, which for a novel is overwhelmingly narration and quoted speech.
# Measured 2026-08-02: Dialogue 578 of 1,071 fold-eligible keeps is ON SHAPE, not
# oversupplied — held as the 50% anchor the corpus completes at 1,156, needing
# +81 Neutral, +35 Documentary and no further dialogue at all. Newscaster (84/69)
# and Speech (69/69) CLOSED when newscaster-v1 landed, so the two remaining gaps are
# both narration. Live table: notes/delivery-mix-campaign.md.
ENGINE_MIX_DIALOGUE = {
    "qwen":       0.450,   # richest instruct slot by a distance (12 references in its
                           # skill file vs 1-2 for most), and the measured gold standard.
                           # Casting + delivery + accent all reach it as TEXT.
    "chatterbox": 0.350,   # no text-instruction slot, but it carries voice IDENTITY
                           # through reference audio — which is what keeps a character
                           # stable across 200 pages. Different directability, equally
                           # needed for a full cast.
    "orpheus":    0.120,   # tag-based (<laugh>, <sigh>): a narrow channel, but a
                           # genuinely dialogue-shaped one.
    "moss_vg":    0.080,   # has instruct slots and they are real; held low because the
                           # stochastic triad (early-EOS, radio drift, IVR cadence) is
                           # worst exactly where dialogue is most exposed.
    "zonos":      0.000,   # OUT of the dialogue lane. Its emotion vector must be
                           # switched off entirely to render cleanly (unconditional_keys
                           # + {"emotion"}), so the one channel dialogue most needs is
                           # the one channel we must disable. Keeps its 20% of narration.
                           # ⚠ Measured 2026-08-02: zonos WITH the vector keeps 74%
                           # (32/43) on Dialogue — bad on narration (8%), only mediocre
                           # here. Held at 0 anyway because Dialogue is AT TARGET and
                           # 74% is the worst of the four; revisit if the lane reopens.
}


# --- Layers 2 + 3: measured per-lane weights, with a diversity floor -----------
# Built 2026-08-02 from every HEARD verdict on a survivor engine in a PRODUCTION
# campaign (probe/stress/revisit excluded — adversarial by construction), and — the
# part that changes the answer — with each engine scored on the interface it was
# actually given.
#
#   keep rate (n)   Dialogue    Neutral   Documentary  Newscaster
#   qwen             81% (161)  100% (38)   100% (19)   100% (27)
#   chatterbox         —        100% (35)   100% ( 7)      —
#   zonos  emotion OFF  —        91% (34)   100% (14)    94% (31)
#   zonos  w/ vector   74% (43)    8% (12)    25% ( 8)      —
#   orpheus            —         95% (19)      —           —
#   moss_vg          89% (19)    74% (35)    56% (18)     95% (21)
#
# TWO THINGS THIS OVERTURNS.
#
# 1. Zonos's headline 31% failure rate was two populations averaged. Conditioned
#    correctly it is a 93.7% engine (74/79) across all three narration lanes. The
#    old flat ENGINE_MIX priced it on the average of a bug we have since fixed —
#    the exact error the onboarding rule names: an engine's failure rate is a
#    property of the interface until proven otherwise.
# 2. moss_vg is not uniformly weak; it is lane-shaped. Worst on Documentary (56%),
#    near-best on Newscaster (95%) — because its documented failure mode (radio-
#    timbre drift, IVR cadence) stops reading as a defect when the target register
#    IS a broadcast read. A single global share cannot express that, and the flat
#    10% both overpaid it on Documentary and underpaid it on Newscaster.
#
# WHY THIS IS NOT "ALLOCATE BY KEEP RATE". Ranking share on reliability alone hands
# the corpus to qwen, and a TEACHER corpus that collapses to one engine's timbre and
# prosody has failed at its actual job. The objective is usable diversity per unit
# of the owner's listening time, so:
#   * MIN_SHARE keeps every eligible engine present — thin cells are how a lane
#     earns evidence, and a 0% share can never produce the data that would raise it.
#   * MAX_SHARE_NARRATION caps any one engine, so no narration lane can become a
#     single voice however well that engine scores.
# Cells marked (no data) are floor shares placed deliberately TO generate evidence.
MIN_SHARE = 0.05
MAX_SHARE_NARRATION = 0.30

ENGINE_MIX_BY_LANE = {
    # All five clear 74%+; zonos, chatterbox and qwen are the reliable core.
    "Neutral": {
        "qwen":       0.25,   # 100% (38)
        "chatterbox": 0.25,   # 100% (35)
        "zonos":      0.25,   # 91% (34) with emotion off — was priced at 8%
        "orpheus":    0.15,   # 95% but n=19; mid share until the n grows
        "moss_vg":    0.10,   # 74% (35) — the weakest here, held at low share
    },
    # NO `Documentary` ENTRY, deliberately (2026-08-11, issue #12). The lane is RETIRED
    # (matcha.delivery.RETIRED_LANES): its channel stays in the wire format so old
    # filelists decode, but no new row may be labelled with it — so there is nothing for
    # a mix to allocate. Its measured weights are not lost; they are in the keep-rate
    # table above, which is a record of what was heard and stays true. Removing the block
    # is not the whole guard: falling back to the flat ENGINE_MIX would let a stale bank
    # builder keep dealing engines into the lane, so `mix_for_lane` refuses it by name.
    #
    # moss_vg's STRONG lane, and the clearest case for lane-conditioning at all.
    "Newscaster": {
        "qwen":       0.30,   # 100% (27)
        "zonos":      0.30,   # 94% (31) with emotion off
        "moss_vg":    0.25,   # 95% (21) — 2.5x its flat share, earned in-lane
        "chatterbox": 0.10,   # no data in this lane
        "orpheus":    0.05,   # no data in this lane
    },
    # Deliberately the ratified dialogue mix, near-unchanged: this lane is AT
    # TARGET, so it is the wrong place to spend renders chasing evidence.
    "Dialogue": ENGINE_MIX_DIALOGUE,
}

# Speech routes with Dialogue (DIALOGUE_DELIVERIES) and has NO per-engine evidence
# — not one heard verdict splits by engine there. It is at target (69/69), so the
# honest thing is to inherit rather than invent weights for it.
ENGINE_MIX_BY_LANE["Speech"] = ENGINE_MIX_BY_LANE["Dialogue"]

# Which lanes are narration — a SUBSET of the vocabulary, written out here AND in
# book_ingest. Same B-L5 shape: one gets a new lane, the other does not, and an engine is
# directed as narration in one file and as dialogue in the next. It lives in
# matcha.delivery because these two modules import each other, so neither can own it.
import sys as _sys  # noqa: E402
# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from matcha.delivery import (DELIVERY_LANES as _DELIVERY_LANES,  # noqa: E402
                            NARRATION_LANES as _NARRATION_LANES,
                            RETIRED_LANES as _RETIRED_LANES, check_assignable)
import schemas  # noqa: E402  -- the single definition of what counts as an axis number

# The pre-2026-08-02 caller convention was lowercase lane names, and it still has to
# resolve. DERIVED from the vocabulary rather than typed out as a five-entry table, for
# the reason the contract module exists (issue #46): a hand-written copy is a second
# delivery vocabulary, and it fails in the direction that matters — a lane appended to
# DELIVERY_LANES and later retired would miss the table, therefore miss _RETIRED_LANES,
# and fall silently back to the flat mix instead of raising.
_LANE_BY_LOWER = {ln.lower(): ln for ln in _DELIVERY_LANES}


def mix_for_lane(lane):
    """Resolve a delivery lane to its mix. Unknown lanes fall back to ENGINE_MIX;
    a RETIRED lane raises.

    The fallback is for a TYPO: a bank builder that misspells a lane should render with
    the flat mix rather than crash mid-campaign. A retired lane is not a typo — it is a
    lane we know about and know no new row may carry, so falling back would deal engines
    into it and buy renders that nothing may then label. Allocation is an ASSIGNMENT, and
    assignment refuses (matcha.delivery.check_assignable). Deleting the lane's weight
    table alone would have made the failure quieter, not louder.
    """
    # NORMALISE EXACTLY AS THE CONTRACT DOES (issue #46). `check_assignable`, the function
    # this defers to for the refusal and its message, opens with `str(lane).strip()`. A
    # guard that normalises differently from the function it defers to disagrees with it,
    # and this disagreement fell OPEN: `mix_for_lane(" Documentary")` returned the flat mix
    # and `allocate_engines(10, lane=" Documentary")` dealt five engines into the retired
    # lane — the exact outcome the refusal below exists to prevent, reached by a leading
    # space. Lanes arrive from JSON manifests and CSV cells (`stage_pool` strips
    # `r.get("delivery")` for precisely this reason), so a padded lane is a real input.
    lane = "" if lane is None else str(lane).strip()
    if not lane:
        return ENGINE_MIX
    key = _LANE_BY_LOWER.get(lane.lower(), lane)
    # The lowercase spelling is normalised FIRST, so `lane="documentary"` is refused too.
    if key in _RETIRED_LANES:
        check_assignable(key)          # raises, with the one contract message
    return ENGINE_MIX_BY_LANE.get(key, ENGINE_MIX)


def _validate_mixes():
    """Fail at import if a lane table breaks its own invariants.

    A weight table is exactly the kind of thing that gets edited by hand under time
    pressure and silently stops meaning what it says — a share nudged to 0 quietly
    retires an engine, and nothing anywhere would notice.
    """
    problems = []
    for lane, mix in ENGINE_MIX_BY_LANE.items():
        total = sum(mix.values())
        if abs(total - 1.0) > 1e-6:
            problems.append(f"{lane}: weights sum to {total:.3f}, not 1.0")
        for engine, w in mix.items():
            if engine in SET_ASIDE and w > 0:
                problems.append(f"{lane}: {engine} is SET_ASIDE but has share {w}")
            if 0 < w < MIN_SHARE - 1e-9:
                problems.append(f"{lane}: {engine} at {w} is under MIN_SHARE "
                                f"{MIN_SHARE} — use 0 to retire it, or the floor")
            if lane in _NARRATION_LANES and w > MAX_SHARE_NARRATION + 1e-9:
                problems.append(f"{lane}: {engine} at {w} exceeds "
                                f"MAX_SHARE_NARRATION {MAX_SHARE_NARRATION}")
    if problems:
        raise ValueError("ENGINE_MIX_BY_LANE is inconsistent:\n  "
                         + "\n  ".join(problems))


_validate_mixes()


# --- Layer 1: what direction channel each engine actually HAS ------------------
# The direction-relay audit (2026-07-25) established that the engines' input slots
# are unevenly distributed, and that table has lived in markdown ever since — which
# means a line whose delivery must be DESCRIBED could be routed to an engine with
# nowhere to put the description, and the only symptom is a clip that is worse than
# expected. `build_direction` documents the same contracts, verified against each
# model's shipped API; this is that knowledge made checkable.
#
#   prose      a natural-language instruction reaches the model
#   numeric    continuous prosody dials (a real, repeatable pacing/pitch control)
#   coarse     a knob, but not a scale you can direct with — chatterbox's
#              `exaggeration` is a rate profile, not an emotion selector
#   reference  voice IDENTITY comes from a reference clip
#   voice_set  a closed set of named voices, and nothing else
#   tags       closed inline markup
#   none       nothing is directable; casting is the whole decision
ENGINE_CHANNELS = {
    "qwen":       {"prose"},
    "moss_vg":    {"prose"},
    "moss85":     {"prose"},
    "chatterbox": {"coarse", "reference"},
    "zonos":      {"numeric", "reference"},
    "orpheus":    {"voice_set", "tags"},
    # vibevoice takes text + a reference wav and has NO instruct slot — its
    # `voice_design` is consumed by OUR casting code, never by the model. That is
    # exactly the trap this table exists to make visible.
    "vibevoice":  {"reference"},
    "dia":        {"punctuation"},
    "longcat":    {"reference"},
}


def route_engines_for(requires, engines):
    """Drop engines that lack a channel this line REQUIRES. Returns (kept, dropped).

    `requires` is a set of channel names the line's direction genuinely needs — not
    a wish list. Ask for "prose" only when the delivery must be described (an
    accent, a named emotional state, a situational frame); asking for it on every
    line would collapse the portfolio to qwen and moss_vg and throw away the
    reference-cloned identity that keeps a character stable across a book.
    """
    kept, dropped = [], []
    for e in engines:
        have = ENGINE_CHANNELS.get(e, set())
        missing = set(requires or ()) - have
        if missing:
            dropped.append((e, f"no {'/'.join(sorted(missing))} channel — "
                               f"has {'/'.join(sorted(have)) or 'nothing'}"))
        else:
            kept.append(e)
    return kept, dropped


def allocate_engines(n_lines, mix=None, lane=None):
    """Split n_lines across engines by a mix. Returns {engine: count}.

    `lane` selects a measured per-delivery-lane mix (see ENGINE_MIX_BY_LANE);
    "dialogue" and the ASSIGNABLE delivery names all resolve — a retired lane raises,
    because dealing engines into one buys renders no row may be labelled with. An
    explicit `mix` still wins, so callers can override either; a caller that states the
    weights itself is not asking this function to resolve a lane at all, and skips that
    refusal with it.

    Largest-remainder, so the counts always sum to exactly n_lines rather than
    drifting by a clip or two per lane the way independent rounding does.
    """
    if mix is None and lane:
        mix = mix_for_lane(lane)
    mix = mix or ENGINE_MIX
    mix = {e: w for e, w in mix.items() if w > 0}
    total = sum(mix.values())
    exact = {e: n_lines * w / total for e, w in mix.items()}
    out = {e: int(v) for e, v in exact.items()}
    for e in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True):
        if sum(out.values()) >= n_lines:
            break
        out[e] += 1
    return out


def route_engines(design: str, engines, delivery: str = None, requires=None):
    """Drop engines that must not take this casting. Returns (kept, [(engine, why), ...]).

    Bank builders call this instead of hand-maintaining per-engine line lists, so the
    routing rule lives in one place and shows up in the build log rather than in
    somebody's memory of a Slack thread.

    `delivery` is the mix-balance axis from ratings.csv (Dialogue / Neutral / Newscaster /
    Documentary). Passing it enforces NARRATION_ONLY; omitting it leaves that rule off, so
    existing two-argument callers are unaffected.

    `requires` is the CHANNEL veto (see ENGINE_CHANNELS): the set of direction
    channels this line genuinely needs. It is checked last so that a casting ban
    still reports the casting reason — an engine excluded for both should say the
    stronger thing. Omitting it leaves the rule off, same as `delivery`.
    """
    kept, dropped = [], []
    bright = is_bright_female(design)
    channel_veto = {}
    if requires:
        _, lacking = route_engines_for(requires, engines)
        channel_veto = dict(lacking)
    for e in engines:
        why = SET_ASIDE.get(e)
        if why is None and bright:
            why = NO_BRIGHT_FEMALE.get(e)
        if why is None and delivery in DIALOGUE_DELIVERIES:
            why = NARRATION_ONLY.get(e)
        if why is None:
            why = channel_veto.get(e)
        (dropped.append((e, why)) if why else kept.append(e))
    return kept, dropped


# Reference clips measured to break cloning engines, with the evidence. This is the
# instrument that works: across every campaign the SPECIFIC CLIP predicts failure far
# better than any threshold on its pitch statistics. rev_05_whimsy_CBX2 rendered clean
# at score 5 from a 301 Hz reference while these failed repeatedly from 242-266 Hz.
# Shared by synth_chatterbox and synth_zonos — the failures cross engines.

# B-L5: ONE definition. This was written out separately in synth_chatterbox and
# synth_zonos (240.0 both) and again in build_reference_pool as MAX_EXCURSION, with
# qc_artifacts and the comment above pointing at "synth_chatterbox.MAX_REF_EXCURSION" as
# though it were canonical. Three copies of a number that encodes a measured finding — the
# chatterbox voice-split tracks pitch EXCURSION, not median (blind 16-clip audition,
# 2026-07-29) — is three chances to re-tune one and not the others, and the symptom would
# be an engine quietly casting references another engine refuses. It belongs beside
# REF_BLACKLIST because it is the same kind of thing: a casting-granularity guard learned
# from heard failures.
MAX_REF_EXCURSION = 240.0

REF_BLACKLIST = {
    # 6 failures across chatterbox AND zonos, revisit-v1 + retake-v1 (2026-07-28/29).
    "tr_victory_whimsical_01_brightF_s1234",
    # split in 3/3 seeds at every exaggeration tried, plus rev_12 twice.
    "tr_embodiment_02_narratorF_s5150",
    # 6 of 7 renders doubled across every seed and parameter cell in the probe.
    "protective_urgency_00_urgentF_s1234",
    # Destabilized an entire Zonos narration group across TWO configurations
    # (delivery-v1-narration mike|Neutral, 2026-07-30/31): with a neutral emotion
    # vector, 7/7 audited clips dropped for clause-boundary pauses + rushed
    # resumption; with emotion truly unconditional the signature persisted at
    # reduced severity (4/8 QC-flagged, dead-air 42-44%). Swapping ONLY the ref
    # (-> neutral_narration_03_narratorF) took the group to 8/8 objectively clean.
    # Its sister refs 00_lottery and 02_lottery are fine — it's this clip.
    "neutral_narration_01_lottery_s1234",
}


# --- the bright-reference policy (B-M9) ----------------------------------------------
#
# `BRIGHT_REF_POLICY` lived in synth_chatterbox, read "exclude", was written into every
# chatterbox manifest row as provenance — and was never consulted by anything. Worse, the
# value it recorded did not describe what the code did. What ran was neither documented
# policy but a hybrid: `select_reference(max_excursion=MAX_REF_EXCURSION)` dropped
# references above the excursion ceiling (that half IS "exclude") and a separate block
# then clamped exaggeration for anything at or above 80% of it. So every chatterbox clip
# ever rendered carries a provenance field asserting a policy that was not applied.
#
# All three are named here, beside MAX_REF_EXCURSION for B-L5's reason: it is casting
# policy learned from heard failures, and a copy per renderer is how the duplicated
# constant happened the first time.
#
#   "exclude"        drop refs above MAX_REF_EXCURSION, never damp. Guaranteed clean;
#                    costs ~60% of the female pool — no bright or teen female casting.
#   "damp"           cast the whole pool, clamp exaggeration on bright refs instead.
#                    Keeps the range at the price of a residual artifact the owner rates
#                    as needing nitpicking to find.
#   "exclude+damp"   both. WHAT HAS ACTUALLY BEEN RUNNING, and therefore the default:
#                    every existing chatterbox clip was rendered under it, and changing
#                    the rendering silently while fixing a provenance bug would be its
#                    own defect.
#
# Which of the three is right is the owner's call — it trades casting range against a
# known artifact in TRAINING data, and these clips train Sonora, so a nitpick-level split
# is still something the student can learn.
BRIGHT_REF_POLICIES = ("exclude", "damp", "exclude+damp")
BRIGHT_REF_POLICY = "exclude+damp"
BRIGHT_REF_EXAG = 0.3
# Damping engages at 80% of the exclusion ceiling. Named rather than inlined so the
# hybrid's two thresholds are visibly ONE relationship.
BRIGHT_REF_DAMP_FRACTION = 0.8

if BRIGHT_REF_POLICY not in BRIGHT_REF_POLICIES:
    raise ValueError(f"BRIGHT_REF_POLICY must be one of {BRIGHT_REF_POLICIES}")


def bright_ref_selection_ceiling(policy=None):
    """-> the `max_excursion` to hand `select_reference`, or None for no ceiling."""
    policy = policy or BRIGHT_REF_POLICY
    return None if policy == "damp" else MAX_REF_EXCURSION


def bright_ref_exaggeration(exag, ref_excursion, policy=None):
    """-> (exaggeration to render at, damped?) under `policy`.

    `ref_excursion` is None for a PINNED reference, which is never damped: pinning exists
    so a probe can render deliberately outside the guards, and a probe whose parameters
    are silently rewritten measures nothing. Re-running artifact-probe-chatterbox would
    otherwise report exaggeration 0.3 for a bank line that plainly reads 0.5, and
    skip-if-exists would hide the change until someone cleared the directory.
    """
    policy = policy or BRIGHT_REF_POLICY
    if policy == "exclude" or ref_excursion is None:
        return exag, False
    if ref_excursion >= MAX_REF_EXCURSION * BRIGHT_REF_DAMP_FRACTION and exag > BRIGHT_REF_EXAG:
        return BRIGHT_REF_EXAG, True
    return exag, False


def pinned_reference(ref_id: str):
    """Same return shape as select_reference, for a reference chosen BY ID.

    Casting is bypassed entirely — no gender parse, no VAT proximity, no duration
    window, no F0 ceiling. For probes that must render a specific clip, and for
    bank authors who have already decided.
    """
    for k in _load_pool():
        if k["id"] == ref_id:
            meta = {"id": k["id"], "register": k["register"], "engine": k["engine"],
                    "gender": k["gender"], "score": 0.0, "pinned": True}
            pct = _load_acoustics().get(k["file"])
            if pct is not None:
                meta["ref_f0_pct"] = round(pct, 2)
            f0 = ref_f0(k["file"])
            if f0 is not None:
                meta["ref_f0_hz"] = round(f0, 1)
            return (str(POOL_ROOT / k["file"]), k["text"], meta)
    raise LookupError(f"no pool clip with id={ref_id}")


def ref_f0(file: str):
    """Median F0 (Hz) of a pool clip, or None if unmeasured."""
    _load_acoustics()                      # populates _acoustics as a side effect
    a = _acoustics.get(file)
    return a["f0_median"] if a else None


def ref_excursion(file: str):
    """Pitch excursion (p90-p10 F0, Hz) of a pool clip, or None if unmeasured.

    Blind test 2026-07-29: this predicts the Chatterbox voice-split better than median
    F0 does. Of 16 clips auditioned blind, the four with audible layering had excursions
    of 240/247/287/291 Hz; the eight clean sub-215 Hz clips ran 104-189; and the
    HIGHEST-median reference in the set (364 Hz) was clean, with an excursion of 264.
    Median F0 did not order the bright band at all — 227 Hz split and 335 Hz did not.

    Mechanistically this is the owner's account of the defect: the voice splits at
    emphasis peaks, so what matters is how far the pitch travels, not where it sits.
    Lilty/singsong/whimsical references are the dangerous ones (three of the four
    failures), and a steady-but-high voice is comparatively safe.
    """
    _load_acoustics()
    a = _acoustics.get(file)
    return a.get("f0_excursion") if a else None


def select_reference(design: str, intended: dict, used: set | None = None,
                     max_f0: float | None = None, exclude: set | None = None,
                     max_excursion: float | None = None):
    """Returns (ref_wav_path, ref_text, ref_meta). `used` biases toward variety.

    `exclude` is a set of pool ids to drop outright — for individual clips measured
    to render badly on a given engine (synth_chatterbox.REF_BLACKLIST).

    `max_f0` is a HARD ceiling on the reference's median F0. NOTHING SETS IT TODAY:
    Chatterbox used to, and a controlled probe refuted the ceiling that motivated it
    (see synth_chatterbox.py — the highest-pitched reference proved the cleanest).
    The parameter is kept because the mechanism is sound and cheap to re-enable if a
    future engine genuinely shows a pitch ceiling; do not re-enable it for Chatterbox
    without new probe evidence. A clip whose F0 was never measured is kept rather
    than dropped — pool_acoustics.json does not cover the pool completely, and
    silently shrinking the pool would be worse than letting an unmeasured clip through.
    """
    used = used or set()
    exclude = exclude or set()
    want, g = _vat(intended), design_gender(design)
    age_target = design_age_target(design)
    best, best_score = None, 1e9
    for k in _load_pool():
        if k.get("gender", "")[:1].upper() != g:
            continue
        if k["id"] in exclude:
            continue
        if not (4.0 <= float(k.get("duration", 0)) <= 10.0):   # owner floor 2026-07-25
            continue
        if max_f0 is not None:
            f0 = ref_f0(k["file"])
            if f0 is not None and f0 >= max_f0:
                continue
        if max_excursion is not None:
            exc = ref_excursion(k["file"])
            if exc is not None and exc >= max_excursion:
                continue
        kv = _vat(k["intended_vat"])
        score = _vat_distance(want, kv)
        score += ENGINE_PREF.get(k.get("engine"), 0.2)
        if age_target is not None:
            pct = _load_acoustics().get(k["file"])
            if pct is not None:
                score += AGE_WEIGHT * abs(pct - age_target)
        if k["file"] in used:
            score += 0.5
        if score < best_score:
            best, best_score = k, score
    if best is None:
        raise LookupError(f"no reference for gender={g}"
                          + (f" under max_f0={max_f0} Hz" if max_f0 else ""))
    used.add(best["file"])
    meta = {"id": best["id"], "register": best["register"], "engine": best["engine"],
            "gender": best["gender"], "score": round(best_score, 3)}
    pct = _load_acoustics().get(best["file"])
    if pct is not None:
        meta["ref_f0_pct"] = round(pct, 2)   # age evidence: within-gender F0 percentile
    f0 = ref_f0(best["file"])
    if f0 is not None:
        meta["ref_f0_hz"] = round(f0, 1)     # artifact evidence: see max_f0 above
    exc = ref_excursion(best["file"])
    if exc is not None:
        meta["ref_excursion_hz"] = round(exc, 1)
    return (str(POOL_ROOT / best["file"]), best["text"], meta)
