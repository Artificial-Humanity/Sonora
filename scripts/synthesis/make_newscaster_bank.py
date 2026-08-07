# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""newscaster-v1 bank — the last starved delivery lane.

Newscaster stands at 8 keeps against a 60 target and is the only lane with no real-audio
source at any bandwidth. Measured 2026-08-01: a public-domain Universal Newsreel carries
99.9% of its energy below 4.2 kHz against LibriTTS-R's 10.7 kHz, so period broadcast audio
cannot enter a 24 kHz corpus. Worse than a quality problem — delivery is the 4th FiLM
channel, so a lane where every clip is band-limited and no other lane is teaches the model
that *Newscaster means muffled*. Modern broadcast audio is copyrighted. That leaves
synthesis, which the campaign brief already specified: **author bulletin copy**
(notes/delivery-mix-campaign.md § Newscaster).

TWO RULES THIS FILE EXISTS TO HOLD
----------------------------------
1. **Newscaster is a VOCAL style, never a medium.** No instruct here says "broadcast",
   "radio", "on air" or "bulletin". Both the Qwen and MOSS skill files call this out
   because "like a news broadcast" is the obvious phrase and it is exactly the one that
   asks for old-radio EQ artifacts instead of a delivery — the same band-limiting defect
   that disqualified the real recordings, reproduced on purpose. "News anchor" is a
   persona and is fine; "as heard on the radio" is a medium and is not.
2. **The copy is authored, generic, and about nothing.** No real person, organisation,
   place or event appears in any line. Partly it keeps the text period-neutral, and partly
   a corpus published CC-BY-4.0 should not ship 78 fabricated news reports that read as
   real reporting about real places. The register is the point; the facts are furniture.

CASTING
-------
Three anchors, biased male (the pool skews Female 391/280 and the campaign asks casting to
converge). Each anchor is ONE voice: a pinned reference for the cloning engines and frozen
voice keys for the instruct engines, per the standing narrator-identity discipline. Anchors
exist so the lane is not one speaker wide.

Engines follow the campaign's Newscaster row — zonos (pitch_std 35-45, rate 15-16, emotion
OMITTED), qwen (the 12-key instruct), moss_vg (one persona sentence). Chatterbox and
orpheus are not in that row and are not used.

    .venv/bin/python scripts/synthesis/make_newscaster_bank.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

DATASETS = pathlib.Path("/data/model-training/datasets")
CAMPAIGN = "newscaster-v1"
MIN_CHARS = 92          # Zonos rate-16 floor; also keeps clips over the 4 s speech floor

# Authored bulletin copy. Generic by construction — no real entity is named anywhere.
# Written for news REGISTER: front-loaded fact, subordinate detail, attribution late.
LINES = [
    "Crews worked through the night to clear the coast road after heavy rain brought down a section of the embankment.",
    "The harbour authority says ferry services will run to a reduced timetable until the repairs are completed.",
    "A fault at the eastern pumping station has been traced to a failed valve, and engineers expect supply to be restored by evening.",
    "Traffic on the northern approach is moving slowly this morning following an earlier collision near the junction.",
    "The regional council has approved funding for a new footbridge to replace the crossing closed last winter.",
    "Forecasters expect the cold front to clear the inland valleys by midday, with brighter conditions following behind it.",
    "Overnight temperatures fell below freezing across the upland districts, and motorists are advised to allow extra time.",
    "The reservoir has returned to seasonal levels after three weeks of steady rainfall across the catchment.",
    "A weather warning remains in place for the exposed coastal strip, where gusts are expected to reach gale force.",
    "Snow is falling on the higher passes, and the transport authority has begun gritting the main routes.",
    "The harvest is running two weeks ahead of schedule, according to figures released by the agricultural board.",
    "Growers in the river valley report the driest spring in a decade, though yields so far appear largely unaffected.",
    "Fishing quotas for the coming season have been set slightly below last year's levels, and the fleet has been notified.",
    "The livestock market recorded its strongest trading day of the year, with prices up across most categories.",
    "A shortage of seasonal labour is being felt across the orchard districts as the picking season begins.",
    "Researchers at the coastal station have recorded the return of a seabird colony absent from the cliffs for forty years.",
    "A survey of the estuary has identified three species not previously documented in these waters.",
    "The observatory reports that conditions should be favourable for viewing the meteor shower later this week.",
    "Excavation at the hillside site has uncovered foundations thought to predate the settlement by several centuries.",
    "A long-running study of the woodland has found that tree cover has increased by nearly a fifth since the survey began.",
    "The restoration of the town hall clock is complete, and the mechanism was wound this morning for the first time in years.",
    "Preparations are under way for the summer festival, which organisers say will be the largest held so far.",
    "The library's reading rooms reopen next week following the refurbishment of the upper floor.",
    "A collection of glass plate negatives donated to the museum has been catalogued and placed on public display.",
    "The choral society will perform in the cathedral on Saturday, marking the fiftieth anniversary of its founding.",
    "Trading closed higher across the board, with the strongest gains recorded in transport and construction.",
    "The manufacturing index rose for a third consecutive month, though analysts caution the increase remains modest.",
    "Unemployment in the region has fallen to its lowest level in four years, according to figures published this morning.",
    "The cost of freight on the northern route has eased considerably following the reopening of the tunnel last month.",
    "A shipping company has confirmed it will add two vessels to the coastal service before the end of the year.",
    "The hospital trust reports that waiting times have shortened following the opening of the new assessment unit.",
    "Health officials are reminding residents that the seasonal vaccination programme begins on Monday.",
    "A review of ambulance response times has recommended additional cover for the outlying districts.",
    "The water authority has lifted the advisory notice issued last week after repeated testing returned clear results.",
    "Dental services in the coastal towns will be expanded under a scheme announced by the health board this morning.",
    "The inquiry into the warehouse fire has concluded, and its findings will be published in full next month.",
    "A court has ordered the demolition of the derelict mill following a lengthy dispute over its ownership.",
    "The tribunal upheld the appeal, and the disputed boundary will now be reassessed by an independent surveyor.",
    "Police are appealing for witnesses following a break-in at commercial premises on the industrial estate.",
    "Charges have been dropped in the case brought against the haulage firm after new evidence was submitted.",
    "Polling stations across the district open at seven o'clock, and counting is expected to begin shortly after midnight.",
    "Turnout in the local elections was slightly higher than at the previous contest four years ago.",
    "The committee has published its recommendations, and members will vote on them at the end of the month.",
    "A public consultation on the proposed boundary changes closes at the end of the week, and responses may be submitted online.",
    "The chamber sat late into the evening before adjourning without reaching a decision, and will resume on Thursday.",
    "Work begins next week on the resurfacing of the ring road, and diversions will be signposted throughout.",
    "The disused railway line is to be converted into a walking and cycling route linking the two villages.",
    "A new water treatment works has come into service, replacing a facility that had operated for over sixty years.",
    "The power station will be taken offline for scheduled maintenance during the quietest weeks of the year.",
    "Fibre connections have now reached the last of the outlying hamlets under the rural connection programme.",
    "The regional side secured promotion with a draw away from home, finishing the season three points clear.",
    "A record entry is expected for the coastal race, which returns to the calendar after a two-year absence.",
    "The swimming club has qualified four athletes for the national championships later this summer.",
    "Rain forced an early close to play, with the match to resume tomorrow morning if conditions allow.",
    "The stadium redevelopment has been approved, and the first phase is scheduled to begin in the autumn.",
    "School enrolment across the district has risen for the second year running, prompting a review of capacity.",
    "The technical college will offer three new apprenticeship routes from September, with places allocated by application.",
    "Examination results published this morning show improvement across most subjects, with the sharpest gains in the sciences.",
    "A programme placing working scientists in classrooms has been extended to a further dozen schools.",
    "The teachers' association has welcomed the settlement reached after several months of negotiation.",
    "Applications for the winter heating allowance open on Monday and close at the end of the following month.",
    "The recycling centre has extended its opening hours in response to sustained demand from residents across the district.",
    "A scheme offering interest-free loans for insulation work has been taken up by more than a thousand households.",
    "The bus service to the outlying villages will run on Sundays for the first time from next month.",
    "Residents are reminded that the pedestrian crossing on the high street will be closed for two days this week.",
    "The lifeboat was launched twice overnight, and both vessels were escorted safely back to harbour.",
    "Firefighters brought the blaze under control within two hours, and no injuries were reported.",
    "A coastguard helicopter assisted in the search before the walkers were located in good health.",
    "The mountain rescue team has issued advice ahead of what is expected to be a busy holiday weekend.",
    "Flood defences held overnight, and the emergency centre stood down its volunteers this morning.",
    "The population census returns show the district growing more slowly than the national average.",
    "A report published today finds that household recycling rates have risen for the fifth consecutive year.",
    "Figures released this morning indicate that visitor numbers over the summer exceeded those of the previous season.",
    "The survey found that most respondents supported the proposals, though a substantial minority did not.",
    "Statistics released by the transport authority show journey times on the coastal route have shortened.",
    "That is the news at this hour, and the next full summary follows immediately after the weather and travel report.",
    "More on that story, and the rest of the day's news, in the full summary at the top of the hour.",
    "The full forecast is next, and we will bring you further details on that story as they reach us this afternoon.",
]

# Three anchors, biased male. Frozen voice identity per anchor — one voice, many bulletins.
ANCHORS = [
    {"key": "anchorM1", "gender": "M",
     "design": "middle-aged man, level and authoritative, unhurried but brisk",
     "qwen": ("gender: Male. pitch: Medium own voice, stable. speed: Brisk, clipped. "
              "volume: Moderate. age: Adult. clarity: High, crisp diction. "
              "fluency: Highly fluent. accent: American English. texture: Firm and clear. "
              "emotion: Detached urgency. tone: Authoritative, informative. "
              "personality: Professional, direct."),
     "moss": ("A composed adult male news anchor's voice, firm and clear in American "
              "English, delivering short factual items at a brisk, even pace."),
     "pitch_std": 38.0, "rate": 15.0},
    {"key": "anchorM2", "gender": "M",
     "design": "older man, dry and precise, measured authority",
     "qwen": ("gender: Male. pitch: Low own voice, stable. speed: Brisk, clipped. "
              "volume: Moderate. age: Older adult. clarity: High, crisp diction. "
              "fluency: Highly fluent. accent: British English. texture: Dry and resonant. "
              "emotion: Detached urgency. tone: Authoritative, informative. "
              "personality: Professional, direct."),
     "moss": ("An older male news anchor's voice, dry and precise in British English, "
              "reading short factual items with unhurried authority."),
     "pitch_std": 35.0, "rate": 15.0},
    {"key": "anchorF1", "gender": "F",
     "design": "adult woman, crisp and even, professional detachment",
     "qwen": ("gender: Female. pitch: Medium own voice, stable. speed: Brisk, clipped. "
              "volume: Moderate. age: Adult. clarity: High, crisp diction. "
              "fluency: Highly fluent. accent: American English. texture: Clear and even. "
              "emotion: Detached urgency. tone: Authoritative, informative. "
              "personality: Professional, direct."),
     "moss": ("A composed adult female news anchor's voice, clear and even in American "
              "English, delivering short factual items at a brisk, steady pace."),
     "pitch_std": 42.0, "rate": 16.0},
]

# Newscaster is flat-affect authority: neutral valence, mild arousal, mild tension.
INTENDED = {"V": 0.0, "A": 0.2, "T": 0.2}

# Campaign brief's Newscaster row. Chatterbox/orpheus are not on it.
MIX = {"zonos": 0.40, "qwen": 0.35, "moss_vg": 0.25}

# VAT PROXIMITY IS NOT ENOUGH TO CAST AN ANCHOR.
# select_reference matches on gender, duration and VAT distance, and for Newscaster's
# (V 0.0, A 0.2, T 0.2) the nearest pool clips are all register `arrogance` — smug reads.
# On the cloning engines the reference's PROSODY carries the delivery, so the anchors would
# have come out smug: correct by the numbers, wrong in the ear. Restrict the pool to
# registers whose read is already flat-affect authority, then let VAT choose within that.
REF_REGISTERS = {"neutral_narration", "matter_of_fact", "composed_reassurance",
                 "condescending_authority", "formal_narrative"}
BANNED = ("broadcast", "radio", "on air", "on the air", "microphone", "studio",
          "bulletin", "newsreel", "transmission", "airwaves")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATASETS / CAMPAIGN))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    import ref_select

    short = [t for t in LINES if len(t) < MIN_CHARS]
    if short:
        print(f"!! {len(short)} lines under {MIN_CHARS} chars — Zonos rate-16 floor:")
        for t in short:
            print(f"   {len(t):>3}  {t}")
        return 1
    if len(set(LINES)) != len(LINES):
        print("!! duplicate lines"); return 1

    # The medium rule, enforced rather than remembered — on the instructs AND the copy.
    for a in ANCHORS:
        for field in ("design", "qwen", "moss"):
            low = a[field].lower()
            hit = [w for w in BANNED if w in low]
            if hit:
                print(f"!! {a['key']}.{field} names a medium {hit} — describe the speaker")
                return 1

    alloc = ref_select.allocate_engines(len(LINES), mix=MIX)
    print(f"{len(LINES)} authored lines -> {alloc}")

    # One pinned reference per anchor: a lane must not be one voice wide, and a voice
    # must not wander between clips.
    off_register = {k["id"] for k in ref_select._load_pool()
                    if k.get("register") not in REF_REGISTERS}
    print(f"  reference pool: {len(ref_select._load_pool()) - len(off_register)} clips in "
          f"{sorted(REF_REGISTERS)} (excluded {len(off_register)})")
    refs = {}
    for a in ANCHORS:
        # Taken refs go into EXCLUDE, not just `used`. `used` only biases toward variety,
        # and against a 17-clip register-filtered pool that bias lost: anchorM1 and
        # anchorM2 were handed the identical reference, which would have made two of the
        # three anchors one voice and defeated the point of having three.
        #
        # B-L8: there was also a `used` set here, accumulating the same ids that go into
        # `exclude`. Since exclude ⊇ used at every iteration, the variety bias could only
        # ever apply to references already dropped outright — dead code that read as a
        # second, weaker guard. Removed rather than kept: two mechanisms where one is inert
        # is how a later reader concludes the bias is doing work it is not.
        wav, text, meta = ref_select.select_reference(
            a["design"], INTENDED, exclude=off_register | set(refs.values()))
        rid = meta.get("id") if isinstance(meta, dict) else None
        if not rid:
            print(f"  !! no distinct reference left for {a['key']} in the allowed "
                  f"registers — widen REF_REGISTERS or drop an anchor")
            return 1
        refs[a["key"]] = rid
        print(f"  {a['key']}: pinned ref {rid} ({meta.get('register')})")

    order = [e for e, n in sorted(alloc.items(), key=lambda kv: -kv[1]) for _ in range(n)]
    lines = []
    for i, (text, engine) in enumerate(zip(LINES, order)):
        a = ANCHORS[i % len(ANCHORS)]
        kept, dropped = ref_select.route_engines(a["design"], [engine], "Newscaster")
        if not kept:
            print(f"  routed out: {engine} for {a['key']} ({dropped})")
            continue
        if engine == "zonos":
            # emotion OMITTED, per the skill file's Newscaster row. A neutral emotion
            # VECTOR still conditions the render — the measured root cause of the zonos
            # narration failures on 2026-07-30.
            direction = {"design": a["design"], "pitch_std": a["pitch_std"],
                         "speaking_rate": a["rate"]}
        elif engine == "qwen":
            direction = {"instruct": a["qwen"]}
        else:
            direction = {"instruct": a["moss"]}
        lines.append({
            "id": f"news_{i:04d}_{a['key']}_{engine[:3].upper()}",
            "engine": engine, "register": "newscaster", "intended": dict(INTENDED),
            "seed": 7100 + i, "text": text, "direction": direction,
            "intended_delivery": "Newscaster", "anchor": a["key"],
            "ref_id": refs[a["key"]] if engine in ("zonos", "chatterbox") else None,
            "source_ref": {"kind": "authored", "index": i},
        })

    out = pathlib.Path(args.out)
    bank = {"version": "newscaster-1", "campaign": CAMPAIGN,
            "license_note": "Copy authored for this corpus; no real entity is named.",
            "note": "Newscaster is a vocal style, never a medium — see module docstring.",
            "lines": lines}
    print(f"\n{len(lines)} lines across {len({l['anchor'] for l in lines})} anchors")
    for e in sorted({l["engine"] for l in lines}):
        print(f"  {e}: {sum(1 for l in lines if l['engine'] == e)}")
    if not args.apply:
        print("\nDRY RUN — pass --apply to write")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "bank.json").write_text(json.dumps(bank, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"wrote {out / 'bank.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
