# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""reader_profile — learn a LibriVox reader's casting attributes once, apply them everywhere.

Owner observation 2026-08-01, on auditing the first force-aligned book: "since this is a
single author book, the tags I added for accent, age, etc can be systematically applied
to all other clips from the same title."

The propagation unit is **(reader, title)** — not the reader alone. Owner 2026-08-01:
"I don't mind if we re-verify these tags for another title by the same reader name. Some
names may belong to multiple readers and, more likely, longtime readers might fall under
both 'adult' and 'middle age'."

Both hazards are real and they are different:

  * **Name collision** — a LibriVox display name is not an identity. Two people can
    share one, and nothing in the catalogue disambiguates them.
  * **Age drift** — a reader with a fifteen-year catalogue genuinely IS "adult" in the
    early recordings and "middle-aged" in the late ones. That is a true fact about the
    recordings, not an inconsistent tag, and a global per-reader profile would have
    flagged it as a conflict or silently overwritten it.

So: attributes propagate FREELY within a title (one session, one person, one age), and a
reader's other titles supply only a PRE-FILL HINT that still needs one ear confirmation
per title. That is the cheap half of the win — the auditor verifies a suggestion instead
of entering three values from scratch — without asserting an identity we cannot check.

This is the real-audio counterpart of the synthesis lane's casting-intent pre-fill
(register_audition, owner standard 2026-07-24): the auditor should VERIFY an attribute,
never re-enter it. Delivery is deliberately NOT propagated — it is the mix-balance axis,
a per-clip judgement about how a given passage was read, not a property of the reader.

    --learn     derive profiles from already-audited clips, write reader_profiles.json
    --apply     fill EMPTY attribute cells in ratings.csv from those profiles

`--apply` never overwrites a value the ear has set; it only fills blanks. ratings.csv is
a live writer (the audition app), so every write is mtime-guarded.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_common  # noqa: E402

DATASETS = pathlib.Path("/data/model-training/datasets")
RATINGS = DATASETS / "sonora-expressive-registers/ratings.csv"
PROFILES = DATASETS / "reader_profiles.json"
ATTRS = ("gender", "age", "accent")     # reader properties. NOT delivery.
MIN_AGREE = 0.80                        # modal value must hold this share to be trusted

# Provenance marker written into the note column by --apply, naming exactly which
# cells this script filled. Without it, learn() re-read its own propagations as
# fresh agreeing votes: four propagated copies out-voted one genuine later ear
# disagreement at MIN_AGREE=0.80, so _CONFLICT never fired and the profile
# echoed itself past the very signal the (reader, title) design exists to surface.
AUTO_PREFIX = "auto-attrs:"
# Written by stage_pool for machine-folded rows; those were never heard either.
FOLD_MARKER = "folded: staged unheard"


def _auto_attrs(row):
    """Attributes in this row that a machine wrote, not the ear."""
    note = row.get("note") or ""
    if FOLD_MARKER in note:
        return set(ATTRS)
    for part in note.split(";"):
        part = part.strip()
        if part.startswith(AUTO_PREFIX):
            return {a for a in part[len(AUTO_PREFIX):].split("+") if a}
    return set()


def _merge_auto_note(note, attrs):
    """Add/extend the auto-attrs marker in a note, preserving anything else."""
    kept, existing = [], set()
    for part in note.split(";"):
        part = part.strip()
        if not part:
            continue
        if part.startswith(AUTO_PREFIX):
            existing |= {a for a in part[len(AUTO_PREFIX):].split("+") if a}
        else:
            kept.append(part)
    merged = sorted(existing | set(attrs))
    kept.append(AUTO_PREFIX + "+".join(merged))
    return "; ".join(kept)


def _ear_set(row):
    """True when a human actually audited this row.

    An ear pass is evidenced by a score. `status != unaudited` was too weak: it
    admitted deferred and dropped rows, and every row --apply had already
    written.
    """
    if (row.get("status") or "") == "unaudited":
        return False
    return bool((row.get("score") or "").strip())


def clip_meta() -> dict[str, tuple[str, str]]:
    """clip id -> (reader, title), from every librivox manifest on disk."""
    out: dict[str, tuple[str, str]] = {}
    for man in DATASETS.rglob("librivox_manifest.jsonl"):
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") and rec.get("reader"):
                out[rec["id"]] = (rec["reader"], rec.get("book") or "?")
    return out


def _title_of(cid: str, manifests: dict) -> str:
    return manifests.get(cid, ("", ""))[1]


def learn(rows: list[dict], meta: dict[str, tuple[str, str]]) -> dict:
    """Profiles keyed reader -> title -> attrs, learned only from AUDITED clips."""
    per: dict[str, dict[str, dict[str, collections.Counter]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(collections.Counter)))
    for r in rows:
        reader, title = meta.get(r["id"], (None, None))
        if not reader or not _ear_set(r):
            continue
        auto = _auto_attrs(r)
        for a in ATTRS:
            if r.get(a) and a not in auto:
                per[reader][title][a][r[a]] += 1
    profiles: dict[str, dict] = {}
    for reader, titles in per.items():
        entry: dict[str, object] = {"titles": {}}
        for title, attrs in titles.items():
            t: dict[str, object] = {}
            seen = 0
            for a in ATTRS:
                c = attrs.get(a)
                if not c:
                    continue
                val, n = c.most_common(1)[0]
                total = sum(c.values())
                seen = max(seen, total)
                if n / total >= MIN_AGREE:
                    t[a] = val
                    t[f"{a}_agreement"] = round(n / total, 3)
                else:
                    # Disagreement WITHIN one title is a real inconsistency (same
                    # session, same person, same age), unlike disagreement across
                    # titles, which is expected.
                    t[f"{a}_CONFLICT"] = dict(c)
            t["clips_seen"] = seen
            entry["titles"][title] = t
        # Cross-title summary: a HINT only. Where a reader's titles disagree the value
        # is recorded as varies-by-title rather than resolved — see age drift above.
        hint: dict[str, object] = {}
        for a in ATTRS:
            vals = {t.get(a) for t in entry["titles"].values() if t.get(a)}
            if len(vals) == 1:
                hint[a] = vals.pop()
            elif len(vals) > 1:
                hint[a] = None
                hint[f"{a}_varies_by_title"] = sorted(v for v in vals if v)
        entry["hint"] = hint
        profiles[reader] = entry
    return profiles


def _title_is_settled(confirmed, attr):
    """C-M7: is this title's own evidence for `attr` good enough to accept a hint on top?

    The gate was `elif hint and confirmed` — the truthiness of the whole title entry. But a
    title whose ear pass DISAGREED WITH ITSELF still produces an entry: `learn()` writes
    `{attr}_CONFLICT` in place of `{attr}`, and `clips_seen` is set either way. So the
    entry is truthy, the hint fires, and a cross-title guess is written into the one title
    we have positive evidence is inconsistent — exactly the case the conflict marker exists
    to stop. It is then recorded in `note` as machine-written and thereafter looks settled.

    Disagreement WITHIN one title is a real inconsistency (same session, same person, same
    age). Disagreement ACROSS titles is expected, and is what a hint is for. Conflating
    them is the bug.
    """
    if not confirmed:
        return False                      # no ear pass on this title at all
    return f"{attr}_CONFLICT" not in confirmed


def _fill(rows, hdr, meta, profiles):
    """Propagate ear-confirmed attributes onto unfilled cells. -> True if anything moved."""
    filled = collections.Counter()
    hinted = collections.Counter()
    for r in rows:
        reader, title = meta.get(r["id"], ("", ""))
        entry = profiles.get(reader) or {}
        confirmed = ((entry.get("titles") or {}).get(title)) or {}
        wrote = []
        for a in ATTRS:
            if a not in hdr or r.get(a):
                continue
            if confirmed.get(a):
                # Same title, ear-confirmed: propagate freely.
                r[a] = confirmed[a]
                filled[a] += 1
                wrote.append(a)
            elif (entry.get("hint") or {}).get(a) and _title_is_settled(confirmed, a):
                # Cross-title hint, but only once THIS title has a CLEAN ear pass —
                # otherwise a name collision or an age shift would propagate unchecked
                # into a title nobody has heard, or into one that disagreed with itself.
                r[a] = entry["hint"][a]
                hinted[a] += 1
                wrote.append(a)
        if wrote and "note" in hdr:
            # Record WHICH cells were machine-written, so a later learn()
            # pass does not read them back as independent ear agreement.
            r["note"] = _merge_auto_note(r.get("note") or "", wrote)

    if not (filled or hinted):
        print("  nothing to fill")
        return False
    if filled:
        print("  filled (same title, ear-confirmed): "
              + ", ".join(f"{n} {a}" for a, n in filled.items()))
    if hinted:
        print("  filled (cross-title hint, title already ear-checked): "
              + ", ".join(f"{n} {a}" for a, n in hinted.items()))
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--ratings", default=str(RATINGS))
    args = ap.parse_args()
    if not (args.learn or args.apply):
        ap.error("pass --learn and/or --apply")

    rpath = pathlib.Path(args.ratings)
    meta = clip_meta()
    print(f"{len(meta)} clips carry a reader across all librivox manifests")

    with rpath.open(newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        hdr, rows = rdr.fieldnames or [], list(rdr)

    profiles = json.loads(PROFILES.read_text(encoding="utf-8")) if PROFILES.is_file() else {}

    if args.learn:
        learned = learn(rows, meta)
        for reader, entry in learned.items():
            prev = profiles.get(reader, {})
            merged_titles = {**(prev.get("titles") or {}), **entry["titles"]}
            profiles[reader] = {**prev, **entry, "titles": merged_titles}
            print(f"  {reader}")
            for title, t in sorted(entry["titles"].items()):
                bits = ", ".join(f"{a}={t[a]}" for a in ATTRS if a in t) or "(nothing yet)"
                print(f"    {title}: {bits}  ({t['clips_seen']} audited clips)")
                for a in ATTRS:
                    if f"{a}_CONFLICT" in t:
                        print(f"      !! {a} disagrees WITHIN this title: "
                              f"{t[f'{a}_CONFLICT']} — not propagated")
            for a in ATTRS:
                if entry["hint"].get(f"{a}_varies_by_title"):
                    print(f"    ~ {a} varies across titles "
                          f"{entry['hint'][f'{a}_varies_by_title']} — expected for a "
                          f"long-running reader (or two people sharing a name); "
                          f"re-verified per title, never merged")
        PROFILES.write_text(json.dumps(profiles, indent=1, ensure_ascii=False),
                            encoding="utf-8")
        print(f"  wrote {PROFILES} ({len(profiles)} readers)")

    if args.apply:
        # C-M6: the shared transaction, not this script's own read/stamp/write flavour.
        with synth_common.ratings_transaction(rpath, tag="reader") as (_h, live_rows):
            if not _fill(live_rows, hdr, meta, profiles):
                raise synth_common.DryRun
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
