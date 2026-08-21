# High-ambition goals — index

The series spans **two repositories**, which is deliberate but was undiscoverable: each note
carried a "Sequence: N of 5" header written when there were five, and no note listed where
its siblings actually live. **Seven exist** (7 added 2026-08-09). This index is the map.

⚠ **7 was promoted from a roadmap, and the promotion is the point** (owner, 2026-08-09).
It lived as `singing-roadmap.md` with a phase marked "immediate, ~zero cost", which made a
long-range ambition read as due work. Filing it here fixes the sequencing claim: singing is
wanted, and it is wanted *after* the model reliably delivers the five spoken lanes.

**The split is by subject, not by accident.** A goal lives with the thing it changes: goals
that change the MODEL or the CORPUS live in `Sonora/github/notes`; goals that change the
PRODUCT — what the app does with a model it already has — live in `Prosodia/notes`.

| # | goal | home | status |
|---|---|---|---|
| 1 | [Matcha-TTS directable production actor](high-ambition-1-matcha-actor.md) | Sonora | **live** — the actor everything else builds on |
| 2 | [Dramatic Reader & full-cast audiobooks](high-ambition-2-dramatic-reader.md) | Sonora | live — needs 1 first |
| 3 | [Child voices](../../../Prosodia/notes/high-ambition-3-child-voices.md) | Prosodia | live — casting/voicing range |
| 4 | [Multilingual G2P](../../../Prosodia/notes/high-ambition-4-multilingual-g2p.md) | Prosodia | live — after English production quality |
| 5 | StyleTTS2-Lite custom model | git history (deleted 2026-08-02) | **RETIRED 2026-07-29** — the quality-ceiling escape hatch is now a scaled flow-matching backbone; decision record in [model-decisions.md §5](../docs/model-decisions.md) |
| 6 | [Audience: conveyance-aware STT](high-ambition-6-audience-conveyance-stt.md) | Sonora | live — the reverse lane; perceive prosody rather than dictate it |
| 7 | [Singing](high-ambition-7-singing.md) | Sonora | **added 2026-08-09** — down the road, behind the quality ladder. ⚠ *Unrequested* singing is a DEFECT and stays one |

⚠ **Cloning real people is not a focus of the owner's** (2026-08-20). Their words, verbatim,
because the distinctions are the whole point: *"It's not forbidden and it's not off the table.
It's just not a focus of mine."* **Three separate statements — not forbidden, not off the table,
not a focus — and none may be upgraded into the others.** The standing is *unprioritised
attention*, the weakest and most reversible form: if it ever becomes a focus that is a change of
interest, not a reversal, and it needs no repeal.

⚠ **"Not a focus" is the owner's; "not a goal" is this repo's own record** — separate claims,
and the second must not be quoted as the first. As a matter of record: **no goal for cloning has
ever been filed in this series.** That is a fact about the series, checkable here; it is not a
stronger version of what the owner said. Recorded because a reader scanning seven goals for
cloning cannot otherwise tell "ruled out" from "nobody has raised it".

⚠ **Neither of the above is a policy.** The likeness *policy* that exists is
[audiobook-corpus-policy.md](../docs/audiobook-corpus-policy.md), scoped to the personally-acquired
audiobook corpus. It is a **separate instrument with a narrower subject** — a corpus boundary,
not a statement about this series — though it does record a compatible intent of the owner's
(*"I want dynamic voicing from the model"*). **Do not read it as creating or removing a goal**,
in either direction.

Three things a reader should know before following the numbers:

- **5 is retired, not pending** — its design note was deleted with the notes `archive/`
  on 2026-08-02 (git history keeps it; the decision survives in model-decisions.md §5).
  Any "5 of 5" phrasing in older notes predates that and predates 6 existing at all.
- **7 carries a defect on the other side of the same behaviour.** Unrequested singing is
  what benched VibeVoice and Dia; requested singing is goal 7. Progress on the goal is not
  evidence about those engines, which were benched for *uncontrollability* rather than for
  inability — a capability you cannot withhold is not a capability.
- **3 and 4 sit in Prosodia but are arguably model-side work.** That is the one place the
  Sonora/Prosodia boundary is genuinely blurred: child voices and multilingual G2P both
  change the model, not the app. Left where they are because they have accumulated their own
  history there — but if the series is ever reorganised, those two are the ones to move.
