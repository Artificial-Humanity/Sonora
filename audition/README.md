# Dataset Auditions

Web rating surface for the Sonora **expressive-registers** dataset. Replaces the
sandboxed Excel workflow: serves each clip's audio inline (`audio/wav`, no download
dialog, plays on phone/iPad over Tailscale) and edits `ratings.csv` directly as the
**single source of truth** the training pipeline reads.

- **Live URL:** https://audition.ai-lab-0.mcfarlin.family (Caddy → `localhost:8095`)
- **Service:** `audition` in the `ai-lab` compose (CPU-only, `python:3.12-slim`,
  runs as `1000:1002` = `lmcfarlin:datashare` so ledger writes keep NFS ownership)
- **Data:** `/data/model-training/datasets/sonora-expressive-registers/`
  — `ratings.csv` (SSOT), `ratings_history.csv` (append-only log), `reroll_queue.csv`
  (Phase 2 work queue)

## Rating vocabulary (v3)

The four UI actions are exactly the v3 score codes (see `synthesis-pipeline.md` in the
`Sonora-GH` sibling repo's `notes/`):

| Action        | Writes score | status      | Notes |
|---------------|--------------|-------------|-------|
| **Keep 1–5**  | `1`…`5`      | `keep`      | quality → training exposure weight |
| **Drop**      | `x`          | `dropped`   | comment kept ("drop because…") |
| **Retake**    | `0`          | `reroll`    | + row appended to `reroll_queue.csv` |
| **Recategorize** | `x1`…`x5` | `relabeled` | sets `register` to the true one; stamps `[was: <old>]` in note |

Comment field is always available. Every save also appends to
`ratings_history.csv` (re-rating is lossless). Keyboard: `1`–`5` keep, `d` drop,
`r` retake, `c` recategorize, `space` play, `j`/`k` next/prev card.

## Filters & re-rating

- **To rate** (`todo`) — clips not yet rated (unaudited).
- **Re-rate** (`rerate`) — *every* already-rated clip, so any rating can be revised at
  any time (scale is strictly 1–5; no former-9 special-casing). Each card shows its
  current score ("Currently: Keep ★3") so nothing is lost when you reconsider.
- **All** (`all`) — the whole dataset.

Every list is **paginated 20 clips per page** (`page` / `page_size`, server-clamped);
`[`/`]` (or `n`/`p`) page, and `j`/`k` roll onto the next/prev page at a list edge.

## Endpoints

- `GET /api/clips?filter=todo|rerate|all&register=<r>&page=<n>&page_size=<20>` — clip page
- `GET /api/stats` — status/score distribution
- `GET /audio?path=<link>` — streams the wav (path confined to the data root)
- `POST /api/rate` — `{id, action, score?, register?, note}` → rewrites the row atomically
- `GET /api/anchors` — the reference set, joined to live ratings (see below)
- `POST /api/anchor` — `{score, id?, why?}` → set a reference from a clip; `id: null` clears
- `GET /` — the audition UI (static)

## Anchor exemplars — what each number sounds like (2026-08-08)

**The scale had saturated.** 799 of 1,219 scored keeps are 5s (66%); on identical text, 46
of 62 controlled groups have three or more *different* engines all at 5; and `librivox`
**real human audio** means exactly 5.00 across its 43 keeps. The top of the scale is
"indistinguishable from a human read" and most of the corpus is sitting on it — which is why
mean score ranks `chatterbox` above `qwen`, an inversion produced by compression rather than
by quality. Never rank engines by mean score; keep RATE survives this, means do not.

Owner, after the equal-loudness re-listen: *"Qwen relays human-like prosody in those cases
where it scored a 5 that makes me rethink 5's given out to others."*

A saturated scale is not fixed by adding scale points. It is fixed by a **fixed reference**,
which is standard MOS practice — without one, 5 drifts per session, per engine, and per how
good the last clip happened to be.

- **Reference bar**, sticky under the header, one slot per scale point with a play button.
  Descending, because the question you actually have is "is this a 5?"
- **⚓ on every clip** makes it the reference for a score, with a one-line *why*. In situ,
  because that is the only moment the judgement exists.
- **`Shift`+`1`–`5`** plays a reference. Deliberately not a bare digit — those already rate,
  and a mis-key that silently re-rates a clip is the worst collision on this surface.

**Anchors are the ear's, never a measure's.** Nothing computes one; a computed anchor would
re-anchor the scale to whatever the measure already believes. Ships with exactly one entry —
the exemplar the owner named by hand — and the other four **unset rather than guessed**.

**State lives in `anchors.json`, not in a clip column**, and that is a scar: the
Qwen/VibeVoice A/B parked prior scores in `note`, which this app overwrites when you type,
and 17 of 33 were lost. Hand-editable; `id` must exist in `ratings.csv`; deleting an entry
unsets it. Two failure modes are shown rather than hidden — an anchor whose clip was
**re-rated** renders as drift (it does not silently re-point), and one whose clip has left
`ratings.csv` renders as **broken**.

## Local dev

```bash
cd app
AUDITION_DATA_ROOT=/data/model-training/datasets \
  uv run --with fastapi --with 'uvicorn[standard]' \
  python -m uvicorn main:app --port 8095 --reload
```

## Deploy

1. **App code (absolute path — NOT pure GitOps):** the container binds
   `/data/services/audition/app` (absolute), because Portainer deploys the stack from its
   own checkout and a relative `./audition/app` bind resolves to an empty dir → crash
   loop. Sync code changes to that path:
   `sudo rsync -a --delete AI-Lab-AMD/audition/app/ /data/services/audition/app/`
2. **Compose (GitOps):** commit and push to `main` — Portainer redeploys the `ai-lab`
   stack and (re)creates the `audition` container against the absolute bind.
3. **Caddy route:** `sudo cp AI-Lab-AMD/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy`
4. **Dashboard tile:** `sudo rsync -av --delete --exclude '.git*' --exclude 'AGENTS.md' --exclude 'status.json' AI-Lab-AMD/dashboard/ /data/services/dashboard/`
   (the `Dataset Auditions` tile + regrouped sections are already in `index.html`).

---

## Re-rolls — mark, don't fire (owner decision 2026-07-18)

**The app never triggers synthesis.** Retake is a *mark*, not a job. This was a
deliberate call: the value of a Retake is the owner's freeform comment, and that
comment is an instruction needing judgment ("half-step down in pitch" = voice-design
change; "replace the words with a psalm, fix the distortion" = text+quality change;
"monotone and lazy" = maybe just an unlucky seed). An auto-trigger could only bump the
seed and would faithfully reproduce the same defect — burning GPU. So interpretation
stays human-in-the-loop, and the web app stays a dumb, robust, phone-friendly rating
surface with no GPU coupling.

**The hand-off is `reroll_queue.csv`.** Every Retake appends a row:
`queued, campaign, id, engine, register, link, tweak(comment)`. The `id` alone recovers
everything else (original text, direction, seed) from the campaign bank/manifests — so
this row is a complete re-roll request.

**Re-rolls are a batch step, run in a working session (Claude drives the pipeline):**

1. Owner audits, hits Retake + note whenever a take is close-but-wrong. Nothing runs.
2. The queue accumulates marks (no live linking, no GPU contention with `sonora_training`).
3. Periodically, process the queue: for each mark, read the note and decide **seed
   re-roll vs. directed change**, apply the actual fix (new seed and/or edited
   `direction`), render the one line via `synth_<engine>.py` (or LongCat
   `batch_inference.py` for transfers), QC-gate it, and the new take enters the audit
   queue as a normal `unaudited` clip — its provenance noting `reroll of <old id>`.
4. The original row stays at score `0` as the audit trail; if the new take passes it
   becomes the keeper.

Because a re-roll is just a new seed → new `id` → new wav, the fresh take shows up as an
ordinary clip; the app needs no relinking or new-file logic. **Default:** re-rolls land
in the same campaign dir (they look like extra seeds); switch to an isolated `rerolls/`
tree only if stricter provenance separation is wanted. No `docker.sock` mount, no synth
worker in this service.

---

## Audit sets — `audit-*` campaigns (owner, 2026-07-19)

**The app is the de facto auditioning surface for EVERY audit set** — calibration audits
included — because phone-anywhere beats desk sessions. Convention:

- Clips are staged under `DATA_ROOT/audit-sets/<set-name>/` (self-describing filenames +
  `manifest.json` with provenance) and registered as `unaudited` rows with campaign
  **`audit-<name>`** and `register` = the expected perceptual class (e.g. `pressed`/
  `breathy`/`neutral`, `valence_high`/`valence_low`).
- **Scoring semantics for audit rows:** Keep 4–5 = clip perceptually matches its label;
  Keep 1–2 or Drop = mismatch (note welcome); Keep 3 = can't tell. Retake/Recategorize
  are not meaningful here.
- **`audit-*` campaigns are NEVER promoted to v1 by the fold** — they are calibration
  evidence, not dataset clips. The fold reads them out as per-class agreement rates and
  archives the rows.

First sets (2026-07-19): `audit-tension-v2` (does +T sound strained, −T breathy, at
matched loudness?) and `audit-valence-v1` (does +V sound positive, −V negative?) — both
50 clips drawn from the `libritts_r_vat_v2` corpus, the pre-training human gate from
vat-corpus-decision-brief step 3.

## Processing a rated batch (the fold, first run 2026-07-19)

When the owner has rated a batch, a working session folds the ratings into the dataset:

1. **Rename on tag mismatch (standing rule, owner 2026-07-19).** Filenames imply tags
   (`<register>_<NN>_<voice>_s<seed>`; voice's last letter = gender). When the audited
   truth contradicts them, the files are renamed so the dataset stays in good order:
   - register relabel → register prefix rewritten (`fierce_devotion_00_…` →
     `grand_oratory_00_…`);
   - gender contradiction → the true gender letter is **appended** to the voice label
     (`smugM` rated Female → `smugMF`): last letter rules the tag, the design voice stays
     readable, and no collision with the real sibling voice (`smugF`) is possible;
   - rated Undefined against an explicit M/F suffix → suffix stripped (`bigM` → `big`).
   Renames propagate to every copy of the wav (v1 + staging), `ratings.csv` (with a
   `[renamed from: …]` note), `v1/metadata.jsonl`, the campaign `*_manifest.jsonl`
   (+ `renamed_from` field), keeps/QC artifacts, LongCat `anchor` refs, and
   `reroll_queue.csv`. A per-fold map is kept as `renames_<date>.csv` in the data root.
2. **Drops** (`x`): wav moved to `_dropped/` (single surviving copy; duplicates deleted),
   row removed from `ratings.csv` (`ratings_history.csv` keeps the record), v1 metadata
   row + audio evicted.
3. **Keeps/relabels**: `owner_audit {score, date, note}` + `gender` folded into
   `v1/metadata.jsonl`; certified keeps still in staging are promoted into `v1/audio` +
   metadata (enriched from keeps/qc/manifest records). Retakes (score `0`) keep their
   rows and any v1 presence as the audit trail until the re-roll lands.
4. **Reroll queue**: deduped (newest row per id wins), rows for ids no longer marked
   `reroll` pruned. Rendering the retakes stays a separate GPU working session.

## Scoring semantics for synth/actor campaigns (owner ruling 2026-07-22)

Two rating regimes exist in the logged data — read tallies accordingly:

- **Through quote-pilot-v3b** (quote-pilot-v1/v2/v3/v3b): score = **prosodic/VAT relay
  ONLY**. Instruction-following (casting accuracy: gender/age/voice match to the design)
  was deliberately NOT scored — it lives in the notes and the audited gender column
  instead. These keeps are valid training material: a prosody-only Keep-5 is direct human
  validation of the clip's V/A/T label; descriptive metadata must reflect the ACTUAL
  portrayal (audited gender/age), not the intended design.
- **From quote-pilot-v3c onward**, actor-evaluation campaigns score **holistically**:
  VAT relay AND instruction-following as a whole. Direction text is shown on each card
  (design/instruct/intended) so the deviation is visible while rating.
