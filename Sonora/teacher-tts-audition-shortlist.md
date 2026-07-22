# Large Teacher-TTS Audition Shortlist (2026-07-17)

_Owner intent: audition very large, high-quality open TTS models on ai-lab-0 (Strix Halo,
up to ~112 GB allocatable unified memory, with headroom reserved for the concurrent
scoring/recording pipeline) as candidate **synthetic-data teachers** — especially for the scarce
expressive tails (V−: grief/anger; genuine pressed strain) that even Emilia undersamples — and
as quality benchmarks. License requirement follows the wall: outputs are only trainable if the
model's license/ToS permits it — Apache-2.0/MIT weights run locally carry no output
restrictions; NC or custom licenses need reading first._

## Shortlist (all fit trivially; fp16 sizes — largest ≈ 17 GB, leaving room to run
teacher + EIV scorer + mining instruments concurrently)

| # | Model | Params | License | Why audition |
|---|---|---|---|---|
| 1 | **OpenMOSS MOSS-TTS** | 8.5B | **Apache-2.0** ✓ | The flagship-size teacher: high-expressiveness, stable long-form, multi-speaker dialogue, voice/character **design** (natural-language voice creation — no fixed voices). ~17 GB fp16. |
| 2 | **Qwen3-TTS-12Hz-1.7B-VoiceDesign** (+ CustomVoice sibling) | 1.9B | **Apache-2.0** ✓ | Description-based voice design + fine-grained control — the "dynamic voicing, avoid 'voices'" philosophy as a product; 3-s cloning in the CustomVoice variant (we'd use design, not cloning). Massive ecosystem (vLLM-Omni, GGUF). Fast enough for bulk synthesis. |
| 3 | **Boson Higgs-TTS-3** | 4.7B | **NC — verdict rendered 2026-07-17** ✗teacher / ✓benchmark | License is literally named `boson-higgs-tts-3-research-and-non-commercial-license`. Per the wall (audeering precedent: NC excluded even as pipeline tool): NO outputs into training or calibration. Retained as the **benchmark bar** — its VAD conveyance is what our renders get judged against in audits. |
| 4 | Orpheus-3B (Canopy) | 3B | Apache-2.0 ✓ | Emotion tags (`<laugh>`, `<sigh>`, guided emotion); proven, simple. |
| 5 | Chatterbox-Turbo (Resemble) | 0.5B | MIT ✓ | **Emotion-exaggeration dial** — maps directly onto our A/T axes; small enough to be a cheap bulk generator. |

| 6 | Zonos-v0.1-transformer (Zyphra) | 1.6B | Apache-2.0 ✓ | **Explicit emotion conditioning vector** (happiness/sadness/fear/anger/surprise/neutral) + pitch/rate — the closest native V/A/T interface in the field. v0.1 quality uneven; must earn its place by ear. Transformer variant only (hybrid needs Mamba kernels, ROCm-dicey). |
| 7 | Dia-1.6B (Nari) | 1.6B | Apache-2.0 ✓ | Dialogue-native, nonverbals ([laughs]/sighs/coughs), wildly expressive; run-to-run voice instability is near-feature for identity-free delivery data. Feeds the parked expressive-fine-tune stage too. |

Excluded on license (verified live tags 2026-07-17): Mistral Voxtral-4B-TTS (CC-BY-NC),
Spark-TTS (CC-BY-NC-SA), F5/MaskGCT lineage (NC weights), **Fish Audio / OpenAudio s1-mini
(CC-BY-NC-SA)**, **ChatTTS (CC-BY-NC)** — all benchmark/audition-only, never teachers.

**ElevenLabs (owner inquiry 2026-07-18) — EXCLUDED, all plan tiers.** The Creator-plan
"commercial license" covers publishing output (podcasts, audiobooks), NOT training. The
Prohibited Use Policy (elevenlabs.io/use-policy, verified 2026-07-18) bans our use case
three ways with no paid-tier exemption: §9(k) output as ML training input, §9(l) output in
any training dataset, §9(j) developing competing models. Stricter than NC — a flat
contractual ban. Never a teacher, never a benchmark render source (generating benchmark
clips would still be §9(j)-adjacent; Higgs fills that role legally).

Late-round scouting via tts-bench capabilities table (owner link, 2026-07-17; licenses
verified against primary sources same day):
* **IndexTTS-2** (1.5B, emotion-reference control — tantalizing fit for our V/A/T philosophy)
  — **EXCLUDED**: bilibili Model Use License §3(c): "You may not Use the bilibili indextts2
  or any Derivative Work to improve any AI model, except for the bilibili indextts2 itself,
  its Derivative Works, or non-commercial AI models" — and §1.5 defines model OUTPUTS as
  Derivative Works. Training Sonora on its outputs = prohibited. Textbook wall case.
* **Step-Audio-EditX** (3B, affect EDITING — would manufacture within-speaker VAT contrast
  pairs by construction) — **HOLD/unverified**: README licenses only the CODE as Apache-2.0;
  the HF weights repo carries no license tag and no LICENSE file. Weights terms unstated =
  fails the wall until StepFun clarifies. Worth a repo discussion ask; the capability is
  uniquely valuable to us.
* **Maya1** (3B, description + inline tags, no cloning) — **ELIGIBLE**: `license:apache-2.0`
  verified on the weights repo. Audition candidate if the field wants another
  description-driven engine.
* **LongCat-AudioDiT-3.5B** (Meituan; owner flagged 2026-07-17) — **ELIGIBLE, auditioning**:
  `license:mit` verified on the official repo — the freest license in the field. Non-AR
  diffusion DiT in waveform-latent space; SOTA voice-cloning SIM on Seed (beats MOSS-TTS).
  No instruct/tag channel — control = prompt audio (timbre + style transfer). Candidate role:
  **affect-transfer fidelity engine** — clone from LABELED SYNTHETIC references (our own
  co-lead renders; never a real person's voice) onto new text, inheriting the reference's
  V/A/T labels at cloning-grade quality. Audition = 4 standard renders + 2 transfer tests
  (MOSS grief/victory anchors). Bonus kinship: flow/diffusion + CFG-style guidance (APG),
  architecturally our side of the family. **OWNER VERDICTS (2026-07-17) — TRANSFER TEST
  PASSED:** standard renders "unimpressive with little conveyance" (longform: good clear
  voice), but BOTH transfers landed: victory_transfer "one of the better ones... the bar
  patron who just saw his team win the cup" (a new everyday-authentic V+ register vs the
  field's battle cries), grief_transfer "old BW Noir but it does it well (7/10)." Affect
  rides through the cloning mechanism; the engine is nothing alone and valuable as an
  amplifier. Note: transfer PRESERVES affect but RE-RENDERS character (deep-serious anchor →
  noir; shout anchor → bar patron) — register drift is a diversity feature under QC, and
  labels still inherit from the anchor. **ROLE CONFIRMED: affect-transfer diversifier — not
  a generator share but a multiplier STAGE: validated co-lead performances × transfer =
  register-varied expansion of the synthetic corpus, anchor labels inherited, blind QC gate
  on every output. Standard (promptless) mode unused.**
* **DramaBox** (Resemble, 3.3B DiT on LTX-2.3 audio branch; owner demoed, "impressive") —
  **fails the wall as a teacher**: LTX-2 Community License. Not NC — but conditional:
  royalty-free only under $10M revenue, commercial entities barred from using it "to train,
  improve, or fine-tune any other machine learning model," distributed outputs must carry
  machine-generated disclosure, plus an acceptable-use rider. More permissive than Higgs's
  license, still not CC-BY-4.0-or-freer — anything it touched would carry lineage
  encumbrance. Benchmark/audition-only unless the owner deliberately breaks the
  unencumbered-by-construction guarantee (not recommended). OWNER DECISION (2026-07-17):
  **on ice.** Owner notes personal use is legal ($0 revenue, under the LTX-2 threshold);
  acknowledged that published-Sonora lineage is the actual constraint. Back-pocket option
  if ever revisited: DramaBox as teacher for a PRIVATE-LINEAGE experiment only
  (/data/private-corpus firewall, never published/merged) — fully legal and fully in-policy.

**MOSS-TTS 8.5B flagship auditioned 2026-07-17** (same Apache-2.0 as sibling): rendered the
four canonical lines with the SAME anchored instructions as the 2.1B — the flagship's
processor DOES accept `instruction=` (undocumented in README, present in
processing_moss_tts.py), plus extra dials the sibling lacks: `quality`, `sound_event`,
`ambient_sound`, `language`, `tokens` (duration), and reference-audio cloning. **OWNER
VERDICTS (2026-07-17):** Longform "very good... as good as any other male longform voice."
Threat "very good... stone cold... black and white noir Chicago gangster movie" — a NEW
menace register (noir gangster) for the spectrum. Grief: Australian-accent drift, subtle,
FAILS the self-evident test ("not obvious without knowing it's a depiction of grief"). 
Victory: "up there with the best female voice depictions" — but the anchored instruction
said DEEP-VOICED ADULT MAN: the flagship gender-flipped it. **Reading: the 8.5B has richer
raw registers but WEAKER instruct adherence than the 2.1B sibling (gender flip on victory,
accent drift on grief — the same anchoring that held rock-solid on the 2.1B). It does NOT
displace the co-anchor. Role: situational register donor (noir threat; strong female
victory) + its unique dials (duration control, sound_event/ambient, cloning) may serve the
pipeline; anchor-adherence caveat noted for any batch use.**

## What "teacher" means here (three escalating uses)

1. **Benchmark**: A/B references for our human audits (zero risk, immediate).
2. **Director vocabulary study**: prompt them across V/A/T-style instructions, score outputs
   with our instruments — how do strong models realize "tense", "grieving", "elated"?
3. **Synthetic tail manufacture** (the strategic one): prompt Apache/MIT teachers to generate
   the scarce extremes (V− above all) → score with the standing instruments → same mining
   criteria and human audit as Emilia keeps → merge as a labeled minority slice. Kokoro-lesson
   applied in reverse: we'd distill *expressiveness we ask for*, not neutrality we inherit —
   and only from permissively-licensed weights run on our own hardware. EIV taught us the
   caution: synthetic-trained systems drift from reality, so synthetic keeps face the same
   real-data-calibrated gates as everything else, and stay a bounded minority of the corpus.

## Audition field (owner call, 2026-07-17; widened same day)

**MOSS-TTS, Qwen3-TTS-VoiceDesign, Zonos, Dia, Chatterbox-Turbo** (teacher candidates,
Apache/MIT, on disk) + **Higgs** (benchmark bar, NC, non-competing, on disk). **Orpheus:
approval-gated on HF (discovered 2026-07-17) — owner access request pending; joins when granted.**

## Audition plan

Run each on ROCm (PyTorch works; Qwen3-TTS also via vLLM-Omni/GGUF), one fixed audition script:
the "We won!" line, a grief line, a threat-whisper line (high-T/low-A — the contract's hardest
case), a long-form paragraph; plus each model's native emotion-control mechanism swept. Score
everything with phonation/EIV/LUFS + human ears. Verdict per model: quality ceiling, emotion
range, controllability, RTF on this box, output-license status.

**Status:** MOSS-TTS + Qwen3-TTS-VoiceDesign downloaded to the reference library (teacher candidates). Higgs license read 2026-07-17: research/non-commercial -> benchmark-only, never a teacher.

Linked from: [audiobook-corpus-policy.md](audiobook-corpus-policy.md) (the license logic) ·
[emilia-mining-plan.md](emilia-mining-plan.md) · [STATE.md](STATE.md) §3.

## Audition results (owner ears, 2026-07-17—)

* **Chatterbox** — exaggeration dial is REAL and monotonic on arousal (0.25 deadpan/sarcastic →
  0.5 mild → 1.0 genuinely happy) but it is an *intensity* knob, not an affect knob: grief
  rendered as casual chat (FAIL — can't reach negative-valence/low-energy), threat came out
  comic-villain not menacing (partial), long-form clean neutral. **Role: candidate A+ (and
  playful/sarcastic) manufacturer; not a V− source.** Bonus finding: exag 0.25 ≈ usable
  sarcasm register.
* **Qwen3-TTS-VoiceDesign** — **best affect control heard so far, and the first model to
  produce REAL GRIEF** (owner: "truly sounds like a man in a deep state of grief", 8.5/10
  intensity — the V−/low-A quadrant everything else missed; intensity tunable via instruct
  wording). Victory emotion "fantastic" and threat "very good" BUT voice design drifts
  tinny/cartoonish under high-arousal instructs (same description, different apparent speaker
  per line). For teacher use the drift is acceptable — identity is scrambled on ingest and
  voice variety helps generalization; anchored descriptions / CustomVoice presets exist if
  stability is ever needed. Long-form: audiobook-grade neutral, slightly breathy, poetry-apt.
  **Provisional role: lead V− (and general instruct-affect) manufacturer.**
* **MOSS-VoiceGenerator** — **strongest all-rounder so far**: best celebratory male voice yet
  (note: wants rising energy into the final word, fixable via instruct), grief AS GOOD as
  Qwen's but darker/contained (complementary flavor — Lethal-Weapon revelation vs Qwen's
  breakdown), **best threat of the field** (Godfather-style pressed menace, not cartoon), deep
  clear neutral long-form. **Voice anchoring in the instruction PREVENTED the cartoon drift**
  Qwen showed — recipe generalizes. ~12 it/s on ROCm (RTF ≈ 2-3× — fine for batch synthesis).
  **Provisional roles: co-lead V− (dark flavor), lead T+ synthesis, strong V+.**

**OWNER CALL (2026-07-17): Qwen3-TTS-VoiceDesign and MOSS-VoiceGenerator on EQUAL FOOTING as
co-lead synthesis teachers** — "one is more whimsical and airy, the other more deep and serious.
Both have their place." The synthesis lane is a portfolio, not a single engine: complementary
palettes = broader coverage of voice/affect space per emotion, which is what the V (and T)
training slices need for generalization. Chatterbox retained for the A+/sarcasm dial — owner
addendum: its Vizzini-style theatrical menace is not a miss but a *specialized register* with
its place ("nuanced and perhaps a bit more specialized than a generic use case") — the portfolio
principle extends to narrow colors: no failed renders, only registers of different breadth.
Zonos/Dia verdicts pending; Higgs benchmarks last.

* **Dia** — rendered via the transformers-native `nari-labs/Dia-1.6B-0626` checkpoint (the
  original repo's config is dia-package format; the 0626 repack avoids the descript-audio-codec
  dependency tree entirely — plain `transformers` + `soundfile` in a rocm/pytorch container).
  Five clips: four canonical + a `(laughs)` nonverbal-tag variant. Known quirks observed at
  render time: improvised a 15.8 s continuation on the 2 s victory line; voice identity varies
  run to run (no control channel). **OWNER VERDICTS (2026-07-17): the field's dark horse.**
  Victory: "the voice is fantastic. Best yet" BUT hallucinated background vocalizing (intended
  crowd, landed "between that and a car engine... running without oil") — subtract the extras
  and it wins the category. Victory + `(laughs)` tag: "best so far... the WWE days with the
  Rock screaming aloud about an amazing victory" — the field's V+/A+ ceiling (owner note: the
  "nonverbal" label is a misnomer — the tag made it MORE verbal). Grief: "best female
  depiction so far and it's very good" — a third grief register: "grief at the therapist's
  office six months after the death," processed and believable, vs Qwen's breakdown and MOSS's
  dark containment ("not every depiction should be grief at the breaking point"). Threat:
  subtle, believable-but-B-movie; inter-clause pause a touch long. Longform: female, "possibly
  the best... my choice for a standard neutral audiobook narrator," pitch-inflection among the
  best of the field. **Role implications: earns a slice — top-end V+/A+ (Rock-scream ceiling),
  female neutral narration + female grief (fills the gender-coverage gap Zonos exposed). Costs
  that shape the slice: no instruct channel (conditioning = script text + tags only, so V
  labels are weaker than instruct-conditioned engines), uncontrollable voice identity, and it
  IMPROVISES (continuations, background sfx) — synthesis QC gate mandatory (DNSMOS +
  duration-vs-text sanity check would have caught both victory artifacts).**
* **Higgs TTS 3 (BENCHMARK ONLY — NC)** — official runtime is CUDA-only (SGLang-Omni); ROCm
  path = community pure-PyTorch repack `multimodalart/higgs-audio-v3-tts-4b-transformers`
  (trust_remote_code, transformers>=5.5, `generate_speech()`, 24 kHz; audio decode via
  `bosonai/higgs-audio-v2-tokenizer`, auto-fetched). Control = native inline tags:
  `<|emotion:elation|>` victory, `<|emotion:sadness|>` grief, `<|style:whispering|><|emotion:fear|>`
  threat, longform untagged. Fast: 4 clips in ~2 min. License wall: clips set the bar only —
  never train, never calibrate. **OWNER VERDICTS (2026-07-17):** Victory: teenage-girl voice
  ("I'd still use it in a Disney film... 15 year old girls need to celebrate victories too") —
  as good as Dia's male victory, different take. Grief: female, softer, "fantastic" — and it
  passes the strictest test in the field: **affect self-evident without context** ("obvious
  without the 'grief' keyword," where Dia's needed scene awareness). Threat: raspy male, the
  field's most DIRECTLY threatening — "a thug in an alley with a knife pointed at my ribs."
  Longform: ties Dia for best female audiobook narrator ("I could listen to this one and Dia's
  10x each and still be at a loss").

**BENCHMARK CONCLUSION (2026-07-17).** The NC 4B sets the bar and the permissive field ROUGHLY
MEETS IT: longform is a dead tie with Dia, victory ties Dia's ceiling, and where Higgs wins
(self-evident grief, direct street menace) it wins on *register*, not fidelity — no quality
ceiling problem in the portfolio. Two things Higgs contributes without a byte of its audio
touching the lineage:
1. **The QC bar**: "affect obvious without the keyword" becomes the acceptance standard for
   synthesized training clips — blind-listen (and blind-instrument) verification, no scene
   context allowed. (The principle is borrowed; Higgs audio itself still never calibrates
   any detector.)
2. **The menace spectrum is now mapped**: Chatterbox Vizzini (theatrical) → Zonos council
   chamber (calm coercion) → MOSS Godfather (pressed power) → Higgs alley thug (direct).
   Threat is not one register; the T+/V− synthesis slice should sample the spectrum.

## FINAL TEACHER PORTFOLIO — OWNER-RATIFIED 2026-07-17

"I'd say keep Dia, Qwen, Moss 8.5 and Longcat as second stage."

* **Stage 1 — generators**: **Dia-1.6B-0626** (V+/A+ ceiling, female narration/grief, coached
  direct menace; QC gate mandatory, temp ≥1.3), **Qwen3-TTS VoiceDesign** (instruct affect
  virtuoso: here-and-now grief, warning threat, whimsical register), **MOSS-TTS 8.5B**
  (noir threat, top female victory, male longform; instruct-ADHERENCE caveat → its labels
  come from post-hoc verification, not instruct intent). Working split Dia 40 / Qwen 30 /
  MOSS-8.5B 30 until material needs say otherwise.
* **Stage 2 — transfer multiplier**: **LongCat-AudioDiT-3.5B** — validated stage-1
  performances as anchors, register-varied expansion, anchor labels inherited, blind QC on
  every output. Synthetic anchors only (never a real person's voice).
* **Retired with credit**: MOSS-VG 2.1B (superseded by flagship on owner's ear; its
  anchored-instruct recipe remains the template), Chatterbox and Zonos (owner: "probably
  aren't worth keeping... our higher quality models can do what they can do" — their
  registers stay documented for recall: sarcasm/A+ dial via script-craft on the ratified
  three, implication menace via scene staging; coached-Dia proved registers are reachable
  by direction).
* **Benchmark shelf (never train/calibrate)**: Higgs TTS 3, DramaBox (on ice).
* **Every clip passes the blind QC bar**: affect obvious without the keyword; DNSMOS +
  duration-vs-text sanity; labels confirmed by instrument (EIV + phonation measures), which
  also neutralizes the 8.5B drift caveat by construction.

### Superseded proposal (v2, pre-ratification — kept for the record)

Owner's closing calls (2026-07-17): "if I had to choose only one, I'd choose Higgs but with
Dia as a big loss. Higgs and Dia both are a power pair." **License wall: Higgs is NC and can
never be a teacher** — so Dia carries the pair's weight alone and Higgs remains the measuring
stick only. Per-scenario picks below follow the owner's take-some-not-all review.

**Scenario × engine assignment** (registers in parentheses are owner's words):

| Scenario | Engines that made the cut |
|---|---|
| Victory / V+A+ | **Dia lead** (male "best yet" + Rock-scream ceiling; QC kills crowd-hallucination takes) · Qwen ("Poppy from Trolls" — whimsical/children's-fantasy register, voicing re-do worth trying) · Chatterbox HARD PASS |
| Grief / V− | **Three permissive registers**: Dia (processed, "therapist's office six months after") · MOSS ("vengeful") · Qwen ("here and now... dying in my arms" — on par with the best) |
| Threat / T+ | MOSS ("warning" — lead) · **Dia coached (direct menace — GAP CLOSED, see experiment below)** · Qwen (warning, semi-cartoonish but ranks highly) · Chatterbox (theatrical, niche) · Zonos passes entirely (owner's pre-stated condition met, pending confirm) |
| Longform / neutral | **Dia female lead** (ties the NC benchmark) · MOSS + Qwen best male options |

**Proposed shares v2** (mix of synthesized material; Dia elevated by the closing calls):
MOSS 30% · Dia 30% · Qwen 25% · Chatterbox 10% · Zonos 5%.

**Dia menace-coaching experiment (2026-07-17, coach_dia_threat.py — 6 takes, owner-judged).**
Result: **Dia CAN reach direct threat**, two working recipes:
* *Beat punctuation* ("Don't. Even. Breathe. ... they will hear us.", temp 1.3): "good...
  a convincing direct threat" (not Higgs-raspy, but direct).
* *Scene staging* (two-speaker mugging script; the victim's fear directs the menace): works
  AND fixes the improvisation problem — both staged takes self-terminated cleanly while every
  bare single-line take ran to the token cap. Voice lottery gave a young female mugger
  ("Dakota Fanning... but it's still good") on seed 42; seed 7 monotone open, "far better"
  from the threat line onward.
Hard engine limits learned:
* **Temperature floor ≈ 1.3**: temp 1.1 and 1.2 collapse to WHITE NOISE (t1/t3 — nothing
  there). Never batch-synth Dia below 1.3; the QC gate's DNSMOS check catches collapses.
* **High guidance (6.0) distorts register, not intensity**: canonical line came out as "a
  mother warning her child of danger... as gentle as she can," pauses too long. Keep gscale 3.
  (Accidental find: that *protective-urgency* register — the canonical line's literal
  semantics — may itself be worth a corpus slice.)
Portfolio effect: Zonos's 5% folds into Dia (threat volume) → **proposed shares v3: MOSS 30 ·
Dia 35 · Qwen 25 · Chatterbox 10**, pending owner confirmation of the Zonos pass.

Grief taxonomy now four registers deep (processed / vengeful / here-and-now / Higgs's
self-evident-soft, the last as QC bar only) — the V− slice samples the range so "sad" never
collapses to one depth. Embodiment (narrator-voice-within-voice) stays with the instruct
engines (MOSS, Qwen). Orpheus joins only if HF approval lands; shares rebalance after its
audition.
* **Zonos** — control interface is the closest analogue to ours (numeric 8-way emotion vector,
  no prompt engineering) but the affect it delivers is muted: victory "very low energy...
  almost anti-excited" (FAIL, Chatterbox-like), grief ≈ 5/10 ("got a C hoping for a B" —
  down-tempo audible but shallow; notably the field's FIRST female grief voice), long-form
  clear but with "moments of slight robotics" in inflection. One real find: threat lands as
  *calm coercion* — "a threat relayed in a political setting... a council chamber" (owner) —
  a colder register than MOSS's Godfather press or Chatterbox's Vizzini theater. Also the
  slowest renderer of the field (~14 min for 4 clips on ROCm). **Role: niche at most — the
  calm-coercion register and female-voice coverage are the only cards it holds that the
  co-leads don't; its scriptable emotion vector is attractive for batch work but the affect
  ceiling is too low to lead anything.** CORRECTION (owner, 2026-07-17): "bad example" in the
  threat verdict was a typo — the threat is GOOD, register refined to *implication menace*:
  "this man is implying bad things might happen to my family when I least expect it, if I
  catch his drift." Everything else (victory/grief/longform) is a pass; even threat becomes a
  full pass if a coached Dia can reach the register.

**OWNER CALL (2026-07-17): provisional synthesis mix — 40% Qwen / 45% MOSS / 15% Chatterbox**
(from the first three auditions; Zonos/Dia may adjust the split). Ratios are of synthesized
training *material*, not of engines' worth.

**NEW REQUIREMENT (owner, 2026-07-17): narrator embodiment / voice-within-a-voice.** Real
narration shifts voice to portray characters — e.g. the *Beartown* audiobook's female narrator
dropping her pitch to play a teenage hockey player. The model must learn *a voice emulating
another voice* as an expressive move, not a speaker change. Implications:

* This is exactly what speaker-as-vector buys us (ARCHITECTURE.md contract): a narrator doing
  a boy's voice is a *excursion in speaker space within one utterance's identity*, not a
  discrete ID swap. Discrete-speaker models (Kokoro-style) cannot represent it; our contract
  already can.
* Training material: instruct-controlled teachers can manufacture it with generation-conditioned
  labels — "a female narrator lowering her voice to imitate a teenage boy, then returning to
  her own voice." Qwen (VoiceDesign) and MOSS (anchored instructions) both take this kind of
  prompt; Chatterbox cannot (no instruct channel) — another reason the mix leans Qwen/MOSS.
* Audition follow-up: add an embodiment line to the stress script when the synthesis pipeline
  is built, to verify teachers can actually do the within-line register shift rather than
  averaging the two voices.
