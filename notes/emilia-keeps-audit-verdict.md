# Emilia Keeps Audit — Verdict: NO MERGE (2026-07-21)

_The pre-registered human gate on the Emilia-YODAS mined keeps
([emilia-mining-plan.md](emilia-mining-plan.md) §4: "Human spot-audit before anything trains")
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
2. **Own teacher-synthesis renders with directed valence.** Dia/Qwen/MOSS render what the
   director asks, so the valence label is exact *by construction* — no instrument in the loop —
   QC-gated and owner-auditioned through the existing workflow. Already the designated source for
   breathy/strained tension vocabulary (T-audit ruling); valence extremes join that charter.

Linked from: [emilia-mining-plan.md](emilia-mining-plan.md) · [STATE.md](STATE.md) ·
[vat-corpus-decision-brief.md](vat-corpus-decision-brief.md) ·
`Registry/Sonora/vat3-24k/README.md` (whose "Next (v1.1)" section this supersedes)
