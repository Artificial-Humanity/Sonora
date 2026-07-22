# Multilingual Dataset Sources — Survey for the Multilanguage Phase

_Drafted 2026-07-18. **This is an UNVERIFIED survey**, not a cleared list. Every entry below still
owes a per-repo license check and a per-split sample-rate/quality check before a single file enters
any pipeline — exactly the discipline [dataset-landscape.md](dataset-landscape.md) applies to
English. Do not trust the parentheticals here as clearance; they are leads to verify._

## Governing constraints (unchanged, cross-lingual)

The multilanguage phase does **not** relax either wall:

1. **License wall — CC-BY-4.0 or freer, no NC/ND anywhere in the lineage**
   ([open-decision-licensing.md](../Prosodia/open-decision-licensing.md) tightening #3;
   [ARCHITECTURE.md](ARCHITECTURE.md) §2). Note the sharp edge: **CC-BY-SA is arguably *not*
   "freer" than CC-BY** — share-alike adds a copyleft obligation. Any SA-licensed subset must be
   split out and treated as excluded until we decide share-alike is acceptable in the public
   lineage (it currently is not established that it is). CC0 is freer; that's clean.
2. **24 kHz quality bar for the fine-tune.** Almost every large multilingual corpus is **16 kHz**
   (ASR heritage). 16 kHz sources are *pretraining-scale only* — same verdict MLS-English already
   earned. Bandwidth extension / restoration (LibriTTS-R was the "-R" restoration of LibriTTS) is a
   separate research bet, not an assumption.

## 🎯 Primary: Multilingual LibriSpeech (MLS)

The natural first target — same LibriVox provenance as the English backbone, so the license story
is the most tractable of the multilingual options.

| Split | Reported license | Notes to verify |
|---|---|---|
| MLS **English** | CC-BY-4.0 (verified 2026-07-13 via `parler-tts/mls_eng`) | Already in the English landscape; **16 kHz**, pretraining-scale only |
| MLS **German, Dutch, French, Spanish, Italian, Portuguese, Polish** | Corpus-level CC-BY-4.0 (MLS is released CC-BY-4.0 as a whole — **verify the specific HF repo you pull each split from carries it through**) | All **16 kHz**; hours are very uneven across languages (German/Spanish deep, Polish/Portuguese thin) |

- Canonical source: OpenSLR SLR94 (the original MLS release). HF mirrors exist per language
  (`facebook/multilingual_librispeech` and community re-hosts) — **the license/quality guarantee
  lives with the specific repo you download from, not with "MLS" the name.** Re-verify per repo.
- Provenance is the strength: LibriVox (public-domain audio) + Project Gutenberg (public-domain
  text), the same clean lineage that makes the English side publishable. This is why MLS ranks
  above web-scraped multilingual sets.
- Weakness: read-audiobook narration → prosodically narrow in *every* language, the same tail
  problem called out for English. Multilingual expressivity tails are an open sourcing question
  (there is no clean Emilia-YODAS equivalent surveyed yet — flag for a later pass).

## 🌐 MLCommons sources

Two distinct things wear the MLCommons name; keep them separate.

1. **Common Voice** (Mozilla, stewarded under the MLCommons umbrella) — **CC0** (public-domain
   dedication). License-perfect. Massive language coverage, crowd-read short utterances.
   - We already ingest a *derivative*: **GLOBE V2** (CC0) is Common Voice-derived and is in the
     English cleared list for accents. For multilanguage, Common Voice itself is the direct source.
   - Caveats to verify: variable recording quality (consumer mics), sample rates vary by clip and
     are often low; short contextless utterances (range, not narrative continuity); per-language
     hours extremely uneven. Curation/filtering is mandatory, not optional.
2. **MLCommons People's Speech** — ~30k h, **English**, **mixed CC-BY-4.0 + CC-BY-SA-4.0**. The
   **SA portion trips the share-alike edge above** and must be split out, not ingested wholesale.
   ASR-grade audio; sample rate/quality need per-clip checking before any TTS use. (Listed as an
   English vetting-pass candidate in [dataset-landscape.md](dataset-landscape.md); repeated here
   because it arrives via the MLCommons question.)

## 🔭 Other multilingual candidates to survey later (NOT yet checked)

- **VoxPopuli** (European Parliament recordings; permissive-leaning license — **verify**, 16 kHz,
  parliamentary register is its own prosodic bias) ·
- **CML-TTS** (Multilingual LibriSpeech-derived, explicitly built *for TTS* — check license + rate;
  potentially higher-value than raw MLS if it clears) ·
- **Emilia non-English subsets** — only the **YODAS** portion is CC-BY (the original subset is
  CC-BY-NC-4.0; the exact same misread trap flagged for English — verify per language) ·
- Fleurs, BibleTTS, and other OpenSLR entries — per-language license lottery; verify each.

## 🧭 Recommended posture when the phase starts

1. **Lead with MLS non-English splits + Common Voice (CC0).** Cleanest provenance; establishes the
   multilingual pipeline on defensible license ground before touching anything murkier.
2. **Treat everything as 16 kHz pretraining-scale** until proven otherwise; do not assume a
   multilingual 24 kHz fine-tune corpus exists — sourcing or restoring one is its own line item.
3. **Split share-alike out.** People's Speech CC-BY-SA and any SA multilingual subset stay excluded
   until/unless the license wall is deliberately revisited for SA — an owner decision, not a
   default.
4. **Record source + license + sample rate per split in the registry model card**, same promotion
   convention as English, so the Apache "for everyone" claim stays auditable across languages.

Cross-refs: [dataset-landscape.md](dataset-landscape.md) (English SSOT) ·
[open-decision-licensing.md tightening #3](../Prosodia/open-decision-licensing.md) ·
[audiobook-corpus-policy.md](audiobook-corpus-policy.md) (the private-lineage boundary, unchanged
cross-lingually) · [STATE.md](STATE.md).
