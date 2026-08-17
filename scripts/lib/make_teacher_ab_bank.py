#!/usr/bin/env python3
"""Build the MOSS-VoiceGenerator vs Qwen3-TTS-VoiceDesign paired audition bank.

Owner ask 2026-07-25: re-audition the two INSTRUCTION-FOLLOWING teachers head to
head before committing render time, on the suspicion they deserve more of it than
VibeVoice has been getting. 20 items, as varied as possible, identical text AND
identical direction across both engines so the only free variable is the engine.

Design notes:
  * MOSS arm is MOSS-VoiceGenerator, NOT the 8.5B flagship. The flagship is an
    un-SFT'd base model whose card does not list `instruction` as an input; that
    is the source of the prompt-read-aloud leakage the owner found. VoiceGenerator
    is the instruction-trained sibling.
  * One merged instruct string per item — the only channel either model reads.
    Qwen's `generate_voice_design()` has no `voice_description` parameter.
  * Every instruction anchors the voice (age + gender + timbre) before the
    delivery imperative. Prior audition lesson: unanchored descriptions drift
    cartoonish under high arousal.
  * Every text is >= MIN_CLIP_CHARS (the 4 s floor, owner 2026-07-25).
  * Items 17-20 are deliberate ACCENT PROBES. Neither model documents accent
    control (MOSS maintainers state it is unsupported; Qwen mentions accent only
    as cloning drift), so these four settle empirically whether accent belongs in
    direction at all or moves wholly to casting.

Usage:
    python scripts/lib/make_teacher_ab_bank.py --out <bank.json>
"""
import argparse
import json
import os
import sys
import urllib.request

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import schemas  # noqa: E402  (the validated loaders; see scripts/lib/schemas.py)
from ref_select import route_engines
from book_ingest import (MIN_CLIP_CHARS, MIN_CLIP_SECONDS, DIRECTOR_SYSTEM,
                         MODEL, OLLAMA, _merge, _extract_json)

CAMPAIGN = "teacher-ab-v1"

ITEMS = [   # (key, expected_register, casting/scene brief for Gemma, text)
    ("victory", "victory_peak",
     "young adult man, bright open timbre, full-throated. Deliver at a near-shout, elated and disbelieving, riding the energy up through the last word",
     "We did it! After all of it, every single one of them said we couldn't, and we did it anyway!"),

    ("grief", "grief_herenow",
     "middle-aged man, low and slightly hoarse timbre. Deliver quietly and slowly, voice heavy and close to breaking, pausing where the breath fails",
     "I keep setting a place for her at the table. I know. I know she isn't coming. I do know that."),

    ("threat", "threat_warning",
     "middle-aged man, dark resonant timbre, controlled. Deliver as a pressed, quiet menace through the teeth — never raise the volume, let the stillness carry it",
     "You will want to think very carefully about the next thing you say to me. Take your time."),

    ("narration", "neutral_narration",
     "middle-aged woman, warm even timbre, unhurried. Deliver as a settled audiobook narrator, even pacing, no editorialising",
     "The road out of the valley climbed for three miles before it turned, and from the turn you could see the whole of the town below."),

    ("tenderness", "tenderness",
     "young adult woman, soft breathy timbre. Deliver gently and very close, almost under the breath, unhurried and warm",
     "Hey. Hey, look at me. You're alright. I've got you, and I'm not going anywhere tonight."),

    ("whimsy", "singsong_whimsy",
     "elderly woman, light papery timbre, playful. Deliver with a lilting singsong, savouring the odd words, as if letting a child in on a secret",
     "And the third door — oh, the third door! — was the one nobody sensible ever, ever opened."),

    ("oratory", "impassioned_oratory",
     "middle-aged man, resonant baritone, projecting. Deliver with rising oratorical weight and rhythm, hammering the repeated word, as if to a packed hall",
     "We asked for patience. We asked for time. We asked, and we asked, and we are done asking."),

    ("arrogance", "arrogance",
     "young adult man, smooth bright timbre, lazy delivery. Deliver with unhurried, smug superiority, savouring your own cleverness",
     "Oh, you practised for this? That's genuinely adorable. I'd hate to tell you how little I did."),

    ("resignation", "melancholic_resignation",
     "elderly man, thin dry timbre, worn. Deliver flatly and slowly, all the fight long gone out of it",
     "You do what you like with the house. I stopped having opinions about it a very long time ago."),

    ("goodnews", "excited_good_news",
     "young adult woman, bright quick timbre. Deliver fast and breathless with delight, tumbling over the words in a rush to get it out",
     "Pick up, pick up — they said yes! They said yes to all of it, the whole thing, starting Monday!"),

    ("urgency", "protective_urgency",
     "middle-aged woman, firm clear timbre. Deliver low, fast and absolutely level — urgent command, no panic in it",
     "Get down. Stay quiet. Whatever happens, do not leave this room until I come back for you."),

    ("rationality", "dismissive_rationality",
     "middle-aged woman, cool precise timbre. Deliver with clipped, patient condescension, as if explaining something obvious for the third time",
     "No. That isn't what the report says, and it isn't what it said the last two times you asked."),

    ("curiosity", "polite_curiosity",
     "young adult man, light clear timbre. Deliver with genuine, slightly tentative interest, lifting at the questions",
     "Sorry, may I ask — how long have you been doing this? It's just, you make it look very easy."),

    ("devotion", "fierce_devotion",
     "young adult woman, strong slightly rough timbre. Deliver with fierce, unwavering conviction, leaning hard into every stressed word",
     "They can take the rest of it. All of it. They are not taking him, and they are not taking you."),

    ("resentment", "simmering_resentment",
     "middle-aged woman, tight constrained timbre. Deliver with held-back anger, tight and level, the pressure audible under every word",
     "No, it's fine. It's completely fine. I only rearranged my entire year around it, so it's fine."),

    ("mystic", "playful_mystic",
     "elderly man, dry crackling timbre, conspiratorial. Deliver slowly and teasingly, as if the listener already half knows and you are enjoying the wait",
     "There are older things than the wood, boy. They were here first, and they are patient about it."),

    ("accent_rp", "composed_reassurance",
     "middle-aged woman, refined British RP accent, clear cool timbre. Deliver with composed, unhurried reassurance in a crisp British Received Pronunciation accent",
     "Everything is quite in hand, I assure you. Do sit down, and we shall sort it out together."),

    ("accent_south", "reminiscent_gossip",
     "middle-aged woman, warm Southern American drawl, easy timbre. Deliver with an unhurried Southern American drawl, warm and conspiratorial, stretching the vowels",
     "Well now, I'm not one to gossip, honey, but you did not hear a single word of this from me."),

    ("accent_scots", "observational_neutral",
     "middle-aged man, Scottish accent, plain steady timbre. Deliver plainly and steadily in a Scottish accent, matter-of-fact, no performance",
     "The ferry goes at six, and if it's blowing hard from the west it doesn't go at all that day."),

    ("accent_none", "casual_social",
     "middle-aged man, General American accent, easy relaxed timbre. Deliver casually and easily in a General American accent, like talking across a kitchen table",
     "You want the short version or the long one? Because the long one takes us into next week."),

]

# Dia's ONLY Gemma-drivable channel. nari-labs document exactly these 21
# non-verbal tags and warn that "overusing or using unlisted non-verbals may
# cause weird artifacts" — the set is soft-closed (the encoder is a byte
# tokenizer, so unlisted tags are accepted silently and misbehave). Everything
# else about Dia (length window, trailing speaker tag, audio-prompt cloning) is
# pipeline mechanics, not something the director writes.
DIA_TAGS = ("(laughs)", "(clears throat)", "(sighs)", "(gasps)", "(coughs)",
            "(singing)", "(sings)", "(mumbles)", "(beep)", "(groans)",
            "(sniffs)", "(claps)", "(screams)", "(inhales)", "(exhales)",
            "(applause)", "(burps)", "(humming)", "(sneezes)", "(chuckle)",
            "(whistles)")

DIA_TAG_SYSTEM = (
    "You choose non-verbal audio tags for a TTS line. Reply with ONLY compact "
    "minified JSON: {\"tags\": [...]}. Each entry must be an EXACT string from "
    "this closed list, or return an empty list if none genuinely fits:\n"
    + ", ".join(DIA_TAGS) +
    "\nChoose at most 2. Prefer an empty list: the engine's own guidance is to "
    "use these sparingly, and unlisted or overused tags cause artifacts. The tag is "
    "placed INSIDE the line at a natural clause boundary, never before the first "
    "word — a leading bare tag is realised with no duration constraint and produces "
    "extreme, non-human artefacts (owner audit 2026-07-26)."
)

ENGINES = [("moss_vg", "MVG"), ("qwen", "QWN"), ("vibevoice", "VV"), ("dia", "DIA")]


def _ollama(system, user, model, url, retries=3, num_predict=400):
    for _ in range(retries):
        body = json.dumps({
            "model": model, "stream": False, "think": False,
            "options": {"num_predict": num_predict, "temperature": 0.2},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                content = json.load(r)["message"]["content"]
            d = _extract_json(content)
            if d is not None:
                return d
        except Exception as e:
            print(f"    retry ({e!r})", flush=True)
    return None


SKILL_DIR = os.path.join(_SONORA_REPO, "scripts", "assets", "director_skills")

# Engine is PINNED per line here, so the director does not choose one — it writes
# direction for a named target using that target's skill file. Identical text and
# identical casting brief across arms; the direction itself is deliberately
# tailored, because a lowest-common-denominator string would handicap every engine
# at once and tell us nothing about which deserves render time.
# ⚠ ONE VALIDATED LOADER, shared with `book_ingest`. The bare `json.load(...)["lexicon"]`
# this replaces failed loudly rather than silently — but it raised `KeyError: 'lexicon'` or a
# `JSONDecodeError` at import time, naming neither the file nor what it was for, and it
# accepted an empty, unsorted or duplicated vocabulary without a word.
LEXICON = schemas.load_register_lexicon()

# TWO PASSES, deliberately. V/A/T and register are properties of the LINE;
# voice_design and instruct are properties of the ENGINE. Asking one call for all
# of it let the labels drift with the target — a first run produced 54 distinct
# register labels for 20 items and identical V/A/T on only 4 of 20 (owner finding
# 2026-07-25). Splitting the calls makes that structurally impossible.
LINE_SYSTEM = (
    "You are the Emotional Director for an audiobook TTS pipeline. You read one "
    "line and label it. The label describes the LINE ONLY — never a voice, never "
    "a rendering engine.\n"
    "Output ONLY compact minified JSON, no markdown, with EXACTLY these keys:\n"
    '{"valence": float in [-1,1], "arousal": float in [-1,1], '
    '"tension": float in [-1,1], "register": string}\n'
    "Valence = pleasant(+)/unpleasant(-); Arousal = energy; Tension = "
    "held/threat/unease.\n"
    "`register` MUST be copied EXACTLY from this controlled lexicon. Do not invent "
    "a label, do not alter spelling, do not compose a new one. Pick the closest "
    "fit:\n" + ", ".join(LEXICON) + "\nJSON only."
)

TARGETED_SYSTEM = (
    "You are the Casting and Delivery Director for an audiobook TTS pipeline. You "
    "are given a casting brief and one line, and you write direction for ONE NAMED "
    "TTS ENGINE. A skill file for that engine follows; it describes what that "
    "engine can and cannot actually be told. FOLLOW IT EXACTLY — writing direction "
    "the engine cannot act on is worse than writing none.\n"
    "Do not emit valence, arousal, tension or register — those are already fixed "
    "for this line and are not yours to set.\nJSON only."
)

# The output schema is ENGINE-SHAPED, because the engines are. qwen and moss_vg
# read exactly one string, so asking for two fields made the director write its
# whole 12-key block twice — which both wasted the token budget (5 of 20 qwen
# calls truncated into invalid JSON) and produced duplicated instructions.
# vibevoice genuinely has two fields: design is casting input, instruct is an
# audit note the model never sees.
SCHEMA = {
    "qwen":      '{"instruct": string}',
    "moss_vg":   '{"instruct": string}',
    "vibevoice": '{"voice_design": string, "instruct": string}',
}


def load_skill(engine):
    """Load a per-engine director adapter.

    Refuses WIP files: `director_skills/sonora.md` is written ahead of the interface
    it describes and must never reach a live director pass. Encoding the refusal here
    rather than trusting nobody sets engine="sonora" is the same discipline the relay
    audit was written to enforce.
    """
    path = os.path.join(SKILL_DIR, f"{engine}.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if "WIP — NOT LOADED" in text or "WIP - NOT LOADED" in text:
        raise RuntimeError(
            f"{engine}.md is marked WIP and must not be loaded into a director pass")
    return text


def label_line(text, model, url):
    """Pass 1 — engine-agnostic. V/A/T + a register from the controlled lexicon."""
    d = _ollama(LINE_SYSTEM, f"The line:\n\u201c{text}\u201d", model, url,
                num_predict=300)
    if d is None:
        return None
    for k in ("valence", "arousal", "tension", "register"):
        if k not in d:
            print(f"    line pass missing key {k}")
            return None
    if d["register"] not in LEXICON:
        print(f"    off-lexicon register {d['register']!r} — coercing to neutral")
        d["register"] = "neutral"
    return d


def direct(brief, text, engine, model, url):
    """Pass 2 — per engine. Casting/delivery only, governed by the skill file."""
    system = (TARGETED_SYSTEM
              + "\nOutput ONLY compact minified JSON, no markdown, with EXACTLY "
              + "these keys:\n" + SCHEMA[engine] + "\n"
              + "\n===== SKILL FILE: " + engine + " =====\n" + load_skill(engine))
    user = (f"Target engine: {engine}\n\n"
            f"Casting and delivery brief from the producer:\n{brief}\n\n"
            f"The line to be performed:\n\u201c{text}\u201d")
    d = _ollama(system, user, model, url, num_predict=900)
    if d is None:
        return None
    required = ("voice_design", "instruct") if engine == "vibevoice" else ("instruct",)
    for k in required:
        if k not in d:
            print(f"    director missing key {k}")
            return None
    return d


def pick_dia_tags(text, model, url):
    d = _ollama(DIA_TAG_SYSTEM, f"The line:\n\u201c{text}\u201d", model, url,
                num_predict=120)
    tags = (d or {}).get("tags") or []
    return [t for t in tags if t in DIA_TAGS][:2]



def _place_tags(text, tags):
    """Insert non-verbal tags at the first clause boundary, not before word one.

    A leading bare tag has no preceding speech to bound it, and Dia realises it
    without duration constraint — `[S1] (sighs) I keep setting a place…` produced an
    "extreme, non-human" sigh and the worst tail in teacher-ab-v1 (2026-07-26).
    nari-labs' own examples place tags mid-utterance.
    """
    if not tags:
        return text
    blob = " ".join(tags)
    for mark in (". ", ", ", "! ", "? "):
        i = text.find(mark)
        if 0 < i < len(text) - len(mark):
            cut = i + len(mark)
            return text[:cut] + blob + " " + text[cut:]
    return text + " " + blob      # single-clause line: trail it rather than lead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ollama", default=OLLAMA)
    args = ap.parse_args()

    short = [k for k, _r, _b, t in ITEMS if len(t) < MIN_CLIP_CHARS]
    if short:
        raise SystemExit(f"texts below the {MIN_CLIP_SECONDS}s floor "
                         f"({MIN_CLIP_CHARS} chars): {short}")

    lines, misses = [], []
    for idx, (key, expected_register, brief, text) in enumerate(ITEMS):
        print(f"[{idx + 1}/{len(ITEMS)}] {key}", flush=True)
        # ---- pass 1: label the LINE once. Shared verbatim by every arm. ----
        lab = label_line(text, args.model, args.ollama)
        if lab is None:
            print("    SKIPPED (line pass failed)")
            continue
        register = lab["register"]
        # ⚠ THE SECOND WRITER OF `intended`, and it was left behind (issue #93). The whole
        # argument for validating at the writer is "one place instead of seven readers" —
        # which only holds if every writer goes through it. `float(lab["valence"])` raises
        # KeyError on an absent axis and ValueError on an unreadable one, both of which kill
        # the campaign mid-run, and it accepts an out-of-range value that is silently clamped
        # downstream. `intended_vat` is the one definition of what a legal axis is.
        intended = {k: (round(v, 2) if v is not None else None)
                    for k, v in schemas.intended_vat(lab).items()}
        if register != expected_register:
            misses.append((key, expected_register, register))
        print(f"    line: {register}  V/A/T {intended['V']}/{intended['A']}/{intended['T']}",
              flush=True)

        tags = pick_dia_tags(text, args.model, args.ollama)
        dia_text = _place_tags(text, tags)

        # ---- pass 2: casting/delivery per engine, governed by its skill file ----
        for engine, suffix in ENGINES:
            row = {
                "id": f"tab_{idx:02d}_{key}_{suffix}",
                "engine": engine,
                "register": register,
                "expected_register": expected_register,
                "intended": intended,
                "seed": args.seed,
                "text": text,
                "pair_key": key,
                "probe": "accent" if key.startswith("accent_") else "register",
            }
            if engine == "dia":
                # Dia takes no direction; its skill file exists to say so and to
                # govern tag choice (done once, above).
                row["direction"] = {"render_text": f"[S1] {dia_text} [S1]",
                                    "temperature": 1.8, "guidance": 3.0,
                                    "dia_tags": tags}
                lines.append(row)
                continue

            d = direct(brief, text, engine, args.model, args.ollama)
            if d is None:
                print(f"    {engine}: SKIPPED (director failed)")
                continue
            # Routing is checked HERE, not at the top of the loop: the rule reads the
            # voice_design the director just wrote, which does not exist until now.
            # Empty today (Chatterbox's bright-female ban was withdrawn 2026-07-29 when
            # its guard moved to pitch excursion), wired so the next such finding is one
            # dict entry in ref_select rather than an edit to every builder.
            _kept, _dropped = route_engines(d.get("voice_design", ""), [engine])
            if not _kept:
                for _e, _why in _dropped:
                    print(f"    {_e}: ROUTED AWAY — {_why}")
                continue
            if engine == "vibevoice":
                # design verbatim so ref_select can parse gender + age band;
                # instruct is carried for the audit card only — never sent.
                row["direction"] = {"design": d["voice_design"],
                                    "instruct": d["instruct"]}
            else:
                # single-string engines: exactly what the director wrote
                row["direction"] = {"instruct": d["instruct"]}
            lines.append(row)

    bank = {"campaign": CAMPAIGN, "version": "1.0",
            "director": args.model,
            "note": ("Fresh Gemma direction per line; engine pinned so all arms "
                     "share identical text and identical direction. Dia's arm "
                     "carries director-chosen non-verbal tags from the closed set."),
            "lines": lines}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    secs = [len(t) / 14.0 for _k, _r, _b, t in ITEMS]
    tagged = sum(1 for l in lines if l["engine"] == "dia" and l["direction"].get("dia_tags"))
    print(f"\nwrote {args.out}")
    print(f"  {len(lines) // len(ENGINES)} items x {len(ENGINES)} engines = {len(lines)} renders")
    print(f"  est. length {min(secs):.1f}-{max(secs):.1f}s (floor {MIN_CLIP_SECONDS}s)")
    print(f"  dia lines carrying a non-verbal tag: {tagged}")
    print(f"  director register mismatches vs expectation: {len(misses)}")
    for k, exp, got in misses:
        print(f"      {k}: expected {exp}, lexicon pick {got}")


if __name__ == "__main__":
    main()
