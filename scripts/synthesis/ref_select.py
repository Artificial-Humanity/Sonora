"""Reference-clip casting for cloning engines (VibeVoice) — v3d-proven logic.

Given a direction design text + intended VAT, selects an audited keep from the
certified dataset (gender parse + intended-VAT proximity + duration window +
casting-faithful engine preference). This is the production home of the logic
piloted in make_v3d_bank.py.

Known limitation (v3d/v3e finding, 2026-07-23): the current reference pool is
own-synthesis keeps whose heritage skews young — age fidelity is bounded by the
pool, not the cloning engine (measured: render-vs-reference dF0 ~ +1-3%).
Real-speech pools + measured age norms (casting-attribute-norms brief) are the
upgrade path; swap POOL_PATH when they land.
"""
import json
import math
import re
from pathlib import Path

POOL_PATH = Path("/data/model-training/datasets/sonora-expressive-registers/v1/metadata.jsonl")
POOL_ROOT = POOL_PATH.parent
ACOUSTICS_PATH = POOL_ROOT / "pool_acoustics.json"
ENGINE_PREF = {"moss85": 0.0, "longcat": 0.05, "qwen": 0.15, "dia": 0.3}

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
    return {"V": v.get("V", v.get("valence", 0)),
            "A": v.get("A", v.get("arousal", v.get("energy", 0))),
            "T": v.get("T", v.get("tension", 0))}


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
    "zonos":      0.200,   # scrutinized, but the ONLY engine with numeric prosody
                           # dials (pitch_std / speaking_rate) — a capability the
                           # delivery-as-FiLM-channel work needs, so it keeps the
                           # larger scrutinized share despite a 31% failure rate.
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
# Measured 2026-08-01: Dialogue 579 of 919 keeps is ON SHAPE, not oversupplied — held as
# the 50% anchor the corpus completes at 1158, needing +80 Neutral, +35 Documentary,
# +61 Newscaster, +61 Speech and no further dialogue at all.
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
}


def allocate_engines(n_lines, mix=None, lane=None):
    """Split n_lines across engines by a mix. Returns {engine: count}.

    lane="dialogue" selects ENGINE_MIX_DIALOGUE; anything else keeps ENGINE_MIX. An
    explicit `mix` still wins, so callers can override either.

    Largest-remainder, so the counts always sum to exactly n_lines rather than
    drifting by a clip or two per lane the way independent rounding does.
    """
    if mix is None and lane == "dialogue":
        mix = ENGINE_MIX_DIALOGUE
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


def route_engines(design: str, engines, delivery: str = None):
    """Drop engines that must not take this casting. Returns (kept, [(engine, why), ...]).

    Bank builders call this instead of hand-maintaining per-engine line lists, so the
    routing rule lives in one place and shows up in the build log rather than in
    somebody's memory of a Slack thread.

    `delivery` is the mix-balance axis from ratings.csv (Dialogue / Neutral / Newscaster /
    Documentary). Passing it enforces NARRATION_ONLY; omitting it leaves that rule off, so
    existing two-argument callers are unaffected.
    """
    kept, dropped = [], []
    bright = is_bright_female(design)
    for e in engines:
        why = SET_ASIDE.get(e)
        if why is None and bright:
            why = NO_BRIGHT_FEMALE.get(e)
        if why is None and delivery in DIALOGUE_DELIVERIES:
            why = NARRATION_ONLY.get(e)
        (dropped.append((e, why)) if why else kept.append(e))
    return kept, dropped


# Reference clips measured to break cloning engines, with the evidence. This is the
# instrument that works: across every campaign the SPECIFIC CLIP predicts failure far
# better than any threshold on its pitch statistics. rev_05_whimsy_CBX2 rendered clean
# at score 5 from a 301 Hz reference while these failed repeatedly from 242-266 Hz.
# Shared by synth_chatterbox and synth_zonos — the failures cross engines.
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
        score = math.sqrt(sum((want[a] - kv[a]) ** 2 for a in "VAT"))
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
