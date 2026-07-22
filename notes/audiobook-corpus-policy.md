# Personally-Acquired Audiobooks — Corpus Policy (2026-07-17, owner-initiated)

_Owner intent: explore using a legally purchased, many-narrator audiobook collection to improve
emotional delivery — explicitly NOT to clone narrator voices ("my general intent is to avoid
'voices'... I want dynamic voicing from the model"). This brief pins the policy so the boundary
is canon before any audiobook file touches the pipeline. Not legal advice; it is the project's
own risk posture._

## The two legal layers (they do not collapse into one)

1. **Voice likeness / right of publicity** — reproducing a specific narrator's timbre. The
   owner's no-cloning intent addresses this layer, and the architecture can enforce it
   (identity and delivery are separate inputs; see below).
2. **Copyright in the recordings** — training weights on purchased audio is a derivative-use
   question that is *unsettled law*, contested in active litigation, and jurisdiction-dependent.
   A purchase licenses listening, not training. **"Avoids voice cloning" does not clear this
   layer.** DRM'd purchases add a further access-circumvention issue before training even
   starts.

## The deciding constraint: our own license wall

ARCHITECTURE §2: every training input in the **public Sonora lineage** is CC-BY-4.0 or freer —
the wall that makes the HF-published models publishable, applied without exception all week
(Expresso, audeering, CREMA-D all excluded over smaller doubts). Purchased audiobooks can never
enter the public lineage. No exception exists or will be requested.

## Sanctioned uses (decreasing safety; 1–2 are unconditionally green)

1. **Measurement / analysis (RECOMMENDED).** Score the collection with the standing instruments
   (phonation, LUFS, EIV heads, chunk statistics) to study how professional narrators move
   through V/A/T space: calibration anchors for label semantics, Director-side pacing/direction
   research, "what does masterful delivery look like in our measurement space." No weights
   derive from the audio; nothing enters any training set.
2. **Benchmark / audit reference.** Professional deliveries as the A/B bar rendered output is
   judged against in human audits. Zero training surface.
3. **Private-lineage fine-tune (permitted with the firewall below, owner's personal risk
   call).** Local experiments only. Contains — does not eliminate — layer-2 risk.

## Firewall rules for any private-lineage experiment (absolute)

* **Never published**: no checkpoint, export, sample, or derived statistic-of-audio leaves the
  machine — not to HF, not to the public registry, not to the Space, not into git.
* **Never merged**: private branches never warm-start, fine-tune, or blend into the public
  lineage. One-way street: public → private allowed, private → public never.
* **Paths**: audiobook audio and all derivatives live only under
  `/data/private-corpus/` (excluded from restic's cloud backup and from every filelist the
  public pipeline reads). Registry entries, if any, live in a local `Registry-private/` that
  has no remote.
* **No selectable identities**: audiobook clips never receive usable speaker slots; if trained
  at all, speaker conditioning is discarded/held-out so the model cannot render "narrator X."
* **Marking**: every private artifact carries `PRIVATE-LINEAGE` in its name and README.

## The dynamic-voicing reframe

"Avoid voices" is an architecture goal, not a data acquisition problem. Kokoro's fixed voices
were baked artifacts; Sonora's are points in a continuous speaker space (mini: 247-vector table
+ interpolation; mid: speaker encoder) directed by continuous V/A/T. Dynamic voicing = identity
interpolation + Director-driven delivery, and the audiobooks' highest-value, lowest-risk
contribution to it is **use 1**: teaching the Director's vocabulary from measurements, with zero
copyrighted samples in any model.

**Status:** policy adopted 2026-07-17. Default posture = uses 1–2; any use-3 experiment is a
deliberate owner call per instance, under the firewall.

Linked from: [ARCHITECTURE.md](ARCHITECTURE.md) §2 · [STATE.md](STATE.md) §3 ·
[emilia-mining-plan.md](emilia-mining-plan.md) (the license-clean expressive-data lane).
