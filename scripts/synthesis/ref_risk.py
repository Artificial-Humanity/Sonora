"""Rejection rate by CONFIGURATION — to target audit sampling and grow REF_BLACKLIST.

MEASURED FINDING (2026-07-29, 523 rendered clips joined to verdicts): the REFERENCE CLIP
dominates everything else. Per-reference rejection ranges from 4% to 100% — a 25x spread —
while Chatterbox's `exaggeration` moves it only 41%->53% across its whole range. An
engine-level rate is therefore close to meaningless: Orpheus reads 36% rejected, but that
is `tara` at 90% dragging up a `jess` that is 0 for 24.

Use this two ways:
  1. BLACKLIST. A reference with a high rejection rate over enough uses belongs in
     ref_select.REF_BLACKLIST. The two already there measure 15/15 and 16/19 here —
     they were added on 6 and 4 observations, and the fuller data confirms them.
  2. SAMPLE. Audit heavily where the configuration is risky or the reference is new and
     unproven; audit lightly where a configuration has a long clean record. Sampling every
     clip equally spends most of the listening on clips that do not fail.

⚠ Rates are HISTORICAL and mostly predate the 2026-07-29 guards (excursion ceiling,
blacklist, `tara` ban, rate clamp). Re-measure after a fresh campaign before trusting them
as current — several of these configurations can no longer be produced.

Joins ratings.csv verdicts to the render manifests so the risk is keyed on what was
actually done (voice, reference, parameters), not on which engine did it. Engine-level
rates are misleading: Orpheus reads 36% rejected, but that is almost entirely `tara`,
while `jess` is 0 for 19.
"""
import csv
import glob
import json
import os
from collections import defaultdict

RAT = "/data/model-training/datasets/sonora-expressive-registers/ratings.csv"
verdict = {}
for r in csv.DictReader(open(RAT)):
    verdict[r["id"]] = r["status"] in ("dropped", "reroll")

rows = []
for mf in glob.glob("/data/model-training/datasets/*/audio/*_manifest.jsonl") + \
          glob.glob("/data/model-training/datasets/book-prose/*/audio/*_manifest.jsonl"):
    for line in open(mf, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("id") in verdict:
            rows.append(r)

print(f"joined {len(rows)} rendered clips to verdicts\n")


def table(title, rows, keyfn):
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        k = keyfn(r)
        if k is None:
            continue
        agg[k][verdict[r["id"]]] += 1
    out = [(k, v[1], v[0] + v[1]) for k, v in agg.items() if v[0] + v[1] >= 4]
    if not out:
        return
    print(f"== {title}")
    for k, bad, n in sorted(out, key=lambda x: -(x[1] / x[2])):
        bar = "#" * int(20 * bad / n)
        print(f"   {str(k):26s} {bad:3d}/{n:3d} {bad/n:5.0%} {bar}")
    print()


orp = [r for r in rows if r.get("engine") == "orpheus"]
table("ORPHEUS by voice", orp, lambda r: r.get("direction", {}).get("voice"))

cbx = [r for r in rows if r.get("engine") == "chatterbox"]


def exc_band(r):
    e = r.get("ref", {}).get("ref_excursion_hz")
    if e is None:
        return None
    return "excursion >=240" if e >= 240 else ("excursion 190-240" if e >= 190 else "excursion <190")


table("CHATTERBOX by reference pitch excursion", cbx, exc_band)
table("CHATTERBOX by exaggeration", cbx,
      lambda r: f"exag {r.get('applied_exaggeration', r.get('direction', {}).get('exaggeration'))}")

zon = [r for r in rows if r.get("engine") == "zonos"]
table("ZONOS by speaking_rate", zon,
      lambda r: f"rate {r.get('applied_speaking_rate', r.get('direction', {}).get('speaking_rate'))}")
table("ZONOS by pitch_std", zon, lambda r: f"pitch_std {r.get('direction', {}).get('pitch_std')}")

# the single strongest predictor found all session: the individual reference clip
table("ANY CLONING ENGINE by reference clip (>=4 uses)", rows,
      lambda r: (r.get("ref", {}) or {}).get("id"))
