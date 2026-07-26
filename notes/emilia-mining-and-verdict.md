# Emilia-YODAS mining — campaign and verdict

> **Consolidated 2026-07-26** from `emilia-mining-and-verdict.md` + `emilia-mining-and-verdict.md`.
> Both campaigns are closed. The verdict is what anyone needs to find, so it leads; the plan
> follows as the evidence behind it. The plan previously read as a live workstream and was the
> file with the obvious name — that was the reason to merge.

## Verdict: NO MERGE (2026-07-21)

The pre-registered human gate fired and failed the mining thesis. The tail was **not** merged;
vat3 trained on LibriTTS-R alone. Valence depth moved to story-driven data instead — LibriVox
quote mining plus directed teacher synthesis.

**Standing lesson:** instrument-percentile mining of in-the-wild audio concentrates whatever
fools the instruments. A percentile is not a label.

---

_The pre-registered human gate on the Emilia-YODAS mined keeps
([emilia-mining-and-verdict.md](emilia-mining-and-verdict.md) §4: "Human spot-audit before anything trains")
fired, and it failed the campaign. This note is the formal record: what the audit found, why the
mining thesis broke, and where the v1.1 valence-depth work goes instead._

## The audit

48 clips (12 per criteria group, seed 1234, drawn by `mine_emilia_keeps.py`), registered as
Auditions campaign `audit-emilia-keeps-v1` (staged `audit-sets/emilia-keeps-v1/`, provenance in
its `manifest.json`), owner-rated 2026-07-21. Semantics: Keep 4–5 = clip perceptually matches its
mining label; 1–2/Drop = mismatch; 3 = can't tell.

| Group | Criterion | Match (4–5) | Score-1s |
|---|---|---|---|
| arousal_high | EIV-Arousal > p95 | 7/12 (58%) | 0 |
| tight | T_full > p90 | 5/12 (42%) | 1 |
| valence_low | V_combo < p5 | 5/12 (42%) | 3 |
| valence_high | V_combo > p95 | **2/12 (17%)** | 4 |
| **overall** | | **19/48 (40%)** | **8** |

Baseline for calibration: the same valence instrument scored **62%** on in-domain LibriTTS-R
(`audit-valence-v1`). Emilia mining performs *worse* than the corpus it was meant to enrich.

## Two failure modes, both owner-observed

1. **Domain mismatch.** The keeps are news broadcasts, technical webinars, and similar — YouTube-CC
   speech is rhetorical/presentational, not story-driven. "Excited anchor" tops the arousal
   percentile without containing narrative emotional conveyance. None of the sample was
   storytelling. For an audiobook actor this is the wrong register of expressiveness even where
   the acoustics are genuine.
2. **The mining enriched for synthetic speech.** All 8 score-1 clips are old-TTS output (obviously
   electronically generated to a listener), and **7 of 8 sit in the valence extremes** (4/12 of
   valence_high). Mechanism: legacy TTS voices are acoustic outliers on exactly the proxies the
   mining percentile-ranked (phonation composite, EIV head responses), so "top 5% valence"
   partially selected "most robotic." DNSMOS cannot catch this — clean synthetic speech scores
   well on noise metrics. Score-1 ids (labeled examples for a future detector):
   `Thi_10_EN_jjUQN8KP-5U_W000246`, `Vhi_01_EN_k9mwo5BQjMg_W000512`,
   `Vhi_02_EN_kCLt51_HD5k_W000013`, `Vhi_03_EN_U0O9MYXofFk_W000139`,
   `Vhi_10_EN_4vPGJ7ZQQFE_W000306`, `Vlo_00_EN_Ct065n9TxSM_W000063`,
   `Vlo_04_EN_sVEER8q4Biu_W000137`, `Vlo_05_EN_W0khlbmQL1c_W000062`.

## Ruling

* **The v1.1 "Emilia continuation" as planned is OFF.** No Emilia clip merges into
  `libritts_r_vat_v3` on this audit. Training valence on 17%-match data — partly synthetic —
  would inject noise and teach old-TTS artifacts as "emotion."
* **The assets keep.** 13,141 keeps stand transcoded to 24 kHz with per-clip caption WER
  (`emilia_kept_24k/`, `asr.jsonl`: median WER 0.045, 29.9 h at ≤ 0.2; Sonora `e2aa88d`). Mining
  scores, provenance, and this audit are all retained — nothing needs re-running if a future,
  properly-filtered subset earns a second look.
* **Emilia's only surviving candidacy is a tension/arousal *texture* supplement** (the strongest
  original rationale: real aspiration that LibriTTS-R restoration strips; A-high was the least-bad
  group at 58%). Deliberately deferred — it would require a synthetic-speech detector plus a
  fresh audit of the filtered subset, and it is not on the v1.1 critical path.
* **Lesson, standing:** instrument-percentile mining of in-the-wild audio *concentrates* whatever
  fools the instruments. Any future wild-source campaign gates on a synthetic-speech detector and
  a human audit of each criteria tail *before* bulk processing.

## Where valence depth comes from instead (v1.1 rescope)

Owner insight (2026-07-21, during vat3 auditioning): the best real valence examples are narrators
performing **quoted character dialogue** — story-shaped emotion, in-domain by construction. Two
lanes, both on standing infrastructure:

1. **LibriVox real audio via the book-prose lane.** SE/PG text × LibriVox narration
   ([book-prose-synthesis-lane]): SCM quote/cast-map tags locate emotional dialogue from the
   *text* side; force-alignment recovers the corresponding audio. Public domain, story-driven,
   and label candidates can be cross-checked (text sentiment × acoustic instruments agreeing =
   high-confidence valence labels).
2. **Own teacher-synthesis renders with directed valence.** Engines with a REAL instruction
   slot — Qwen and MOSS-VoiceGenerator (`moss_vg`) — render what the director asks, so the
   valence label is exact *by construction* — no instrument in the loop —
   *(caveat added 2026-07-25 after the direction-relay audit: this holds only for engines whose
   relay is verified. VibeVoice takes a casting call, not a performance note; Dia takes only
   inline text and its 21 non-verbal tags; and the `moss85` flagship accepts no `instruction`
   at all. At the time this was written, none of the three named engines was receiving the
   voice design.)*
   QC-gated and owner-auditioned through the existing workflow. Already the designated source for
   breathy/strained tension vocabulary (T-audit ruling); valence extremes join that charter.

Linked from: [emilia-mining-and-verdict.md](emilia-mining-and-verdict.md) · [STATE.md](STATE.md) ·
[vat-channels.md](vat-channels.md) ·
`Sonora/huggingface/vat3-24k/README.md` (whose "Next (v1.1)" section this supersedes)


---

# The campaign as executed (evidence behind the verdict)

*Retained for the mining criteria, the retained-asset paths, and the labelled score-1 examples,
which are genuinely useful if a synthetic-speech detector is ever built. **This is a closed
campaign — nothing below is a live plan.***

_Owner call (2026-07-17): the first 3-channel VAT run waits for Emilia-derived expressive tails
("let's give Emilia her turn"). Trigger: the tension-v2 within-voice audit — breathy end
validated, pressed end "better but not 'pressed'" (nasal-tinged) — because polite narration
simply doesn't contain genuine strain, and no label formula can synthesize what the data lacks.
Same logic applies to |V| extremes ("We won!" conveyance). This plan bounds the campaign._

## Source & license

`amphion/Emilia-Dataset` on HF (gated — one-time terms acceptance required, owner action):
* `Emilia/` — original 101k h, **CC-BY-NC-4.0 — NEVER TOUCHED** (license wall).
* `Emilia-YODAS/` — 114k h from YouTube-CC sources, **CC-BY-4.0 — our lane.** EN portion:
  thousands of ~1.09 GB webdataset tars (≥ 1 TB total). Per-utterance mp3 + JSON (transcript,
  speaker cluster, DNSMOS).

## Strategy: sample → score → keep extremes (never bulk-download)

1. **Probe** (~2 tars): confirm format; score with the standing instruments (phonation
   composite + EIV heads via `scripts/eiv_score.py` + LUFS) and measure *tail richness* — what
   fraction of clips land beyond the LibriTTS-R per-speaker 2σ bands for T+, |V|, A. This
   yields hours-of-extremes per tar, which sizes the whole campaign.
2. **Iterate**: random tars in bounded batches (disk is fine: 3.6 TB free; keep ≤ 200 GB of
   tars on disk at a time, delete after scoring — scores and kept clips are the assets).
   Keep a clip when it clears pre-registered mining criteria (draft, to be pinned after the
   probe): T+ tail (pressed candidates), V extremes both signs, A+ tail; DNSMOS/quality floor;
   duration 1–16 s; English.
3. **Stop rule**: target ≈ **20–40 h of kept extremes** (comparable to a meaningful fraction of
   the 45.7 h base corpus) or K consecutive tars adding < 1% new keeps.
4. **Human spot-audit** (~50 kept clips) before anything trains — the same gate every label has
   passed or failed honestly so far.

## Design decision needed before training (not before mining): speakers

The model has a fixed 247-slot speaker table; Emilia clips arrive from unknown, many-speaker
clusters. Options when the enriched corpus is assembled:
* **A (recommended): extend the table** — keep only Emilia speaker-clusters with enough kept
  clips (e.g., ≥ 20), append N new rows (fresh embeddings, warm-started rest of model). Simple,
  matches current architecture; casting grid grows.
* B: train Emilia clips with a shared "expressive" pseudo-speaker — pollutes identity space,
  rejected unless A yields too few per-cluster clips.
* C: speaker encoder — the sonora-mid item; out of scope for this stage.

## Pipeline (all standing instruments, no new licenses)

tar → extract → filter (duration/lang/quality) → phonation measures (`derive_vat_corpus`
functions) → EIV heads (`eiv_score.py`: Soft_vs._Harsh, Arousal + the 9 combo heads) → LUFS →
mining criteria → kept clips copied to `/data/model-training/datasets/emilia_kept/` with a
provenance manifest (tar, key, license note) → phonemize (op_g2p; Emilia transcripts are ASR —
digits/OOV expected, validate hard) → merge into `libritts_r_vat_v3` filelists with per-speaker
z over the *combined* corpus.

## Probe results (2026-07-17, 2 tars, 10k sampled clips, DNSMOS ≥ 3.0)

Fractions of Emilia clips beyond LibriTTS-R's global p97.5/p2.5 (LibriTTS-anchored scales):
**T pressed+ 33.3%**, V+ 9.5%, **V− 0.3% (the scarce tail — drives tar count)**, A+ 6.7%;
union keep 41% ≈ 23 minable hours per full tar. Reading: the corpora barely overlap — in-the-wild
speech is wholesale brighter/harsher than studio narration (partly genuine, partly recording
conditions; per-speaker/cluster z at labeling time re-centers this). Speaker-table extension
viable: 251 clusters with ≥5 keeps in the 20% sample alone.

## Mining criteria v1 (pre-registered 2026-07-17, post-probe)

* Quality: DNSMOS ≥ 3.3, duration 1–16 s, EN.
* Keep on ANY of (Emilia-internal percentiles, computed over the batch): T_full > p90;
  V_combo > p95; **V_combo < p5 (keep the whole thin tail)**; EIV-Arousal > p95.
* Per-speaker-cluster cap 40 keeps (diversity); provenance manifest per keep (tar, id, license).
* Stop rule: ≥ 30 h kept AND ≥ 3 h in the V− tail, hard cap 10 tars this batch.
* Standing human spot-audit (~50 clips) before anything merges into `libritts_r_vat_v3`.

## Campaign results (2026-07-17, COMPLETE — stop rule satisfied at 9/10 tars)

66,217 quality-filtered clips scored (DNSMOS ≥ 3.3) → **12,707 keeps, 31.6 h**: T+ 13.7 h,
V+ 8.9 h, **V− 8.1 h** (floor 3 h), A+ 8.8 h. 2,515 speaker clusters; **139 with ≥ 20 keeps**
(speaker-table extension viable). Thresholds stable between batches. Keeps + provenance
manifest + 48-clip owner-audit sample: `/data/model-training/datasets/emilia_kept/`.
Raw tar extracts deleted after keeping (re-downloadable). **Owner spot-audit (2026-07-17): T+ VALIDATED** ("much closer... nothing nasaly"; known
in-the-wild impurities: occasional overlapping speakers/slur — accepted weak-label noise);
**A+ VALIDATED** (evangelist/shopping-host energy); **V± FAILED as mined** — global
LibriTTS-anchored scales selected *timbre* (deep-voiced men scored "negative", crisp didactic
speech "positive"), not affect. **Criteria amended (v1.1): V mining uses WITHIN-CLUSTER z**
(clip must be dark/bright *for its own speaker*; clusters ≥ 10 clips) — the same load-bearing
per-speaker-normalization lesson the label pipeline already learned. Re-mined totals: **33.1 h**
(T+ 13.6, V+ 8.0, V− 7.1, A+ 8.8), 137 clusters ≥ 20 keeps. **V-tail re-audit (owner, 2026-07-17): FAILED AGAIN, differently** — timbre confound gone
(voices vary), but both tails surface *professional registers* (news field reports, webinars,
tour guides, documentary narration), not affect. Diagnosis: base-rate problem — in-the-wild
speech is overwhelmingly informative; a d'≈0.9 detector's top percentiles there are mostly
engaged-neutral false positives. **Consequence (final): v3 merge takes Emilia T+ and A+ keeps
only (both owner-validated); mined V± keeps dropped from enrichment. Valence enrichment
reassigns to the teacher-synthesis lane** (teacher-tts-audition-shortlist.md), where the label
is generation-conditioned ground truth ("this is grief" because we instructed grief — Qwen
already demonstrated it), verified by instrument + ear — structurally stronger than weak-label
mining for the one dimension acoustics cannot measure.**

## Status

* 2026-07-17: gate accepted (owner); probe done; criteria pre-registered; **campaign complete
  (results above)**. Launch kit for the 3-channel run stays staged — it fires when
  `libritts_r_vat_v3` exists and passes its gates.

Linked from: [vat-channels.md](vat-channels.md) ·
[vat-channels.md](vat-channels.md) · [STATE.md](STATE.md) §3 ·
[ARCHITECTURE.md](ARCHITECTURE.md) §2 (corpus accumulation as standing activity).

## OUTCOME (2026-07-21): audit failed the campaign — NO MERGE

The pre-registered spot-audit (48 clips, campaign `audit-emilia-keeps-v1`) came back 19/48
(40%) overall, valence_high 2/12, with 8 clips identified as old-TTS synthetic speech
concentrated in the valence extremes (the mining enriched for what fools the instruments).
Full verdict and the v1.1 rescope: [emilia-mining-and-verdict.md](emilia-mining-and-verdict.md).
