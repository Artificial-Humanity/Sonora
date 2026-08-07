# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""stage_pool — separate the segmented POOL from what is STAGED for training.

Owner design 2026-08-01: "We should separate out our dataset into the full pool and what
is staged for training. If we're already in here, we might as well break up the recording
into trainable segments and add them all to the pool. We should choose selections
linearly, from the start of the book reading up to our required number, record the range
and leave the rest in our pool."

WHY THE SPLIT EXISTS
--------------------
Segmenting is the expensive, one-time part (fetch, ASR anchor, CTC refine); selecting is
cheap and reversible. Doing all the segmenting once and staging a slice means the corpus
can grow later without re-running a single alignment.

It also fixes a real balance hazard. *Uneasy Money* alone is 6h41m across 25 sections. Fed
in whole it would dominate a corpus of ~800 keeps with ONE reader, ONE accent and ONE
delivery — and because `derive_vat_corpus` computes V/A/T as PER-SPEAKER z-scores, a
speaker holding most of the corpus effectively defines the population everyone else is
measured against. The pool keeps that material available without letting it set the
population.

SELECTION IS LINEAR AND RECORDED
--------------------------------
Clips are taken in READING ORDER (section, then sentence index) from where the last
staging run stopped, never sampled at random. Owner rationale 2026-08-01: **"there may be
value in training on continuous readings."**

That is a substantive bet, not a convenience. Contiguous clips preserve what isolated
sentences destroy:

  * **Declination and discourse arcs.** Pitch and energy drift across a paragraph and
    reset at its boundary. Randomly sampled sentences carry that drift as unexplained
    variance; contiguous ones carry it as structure. This is also the cleanest available
    test of the unexplained duration/arousal correlation (r ~ -0.42) — with continuous
    reading order we can finally ask whether it is declination or content.
  * **The north-star "hush".** A narrator lowering his voice as the mice creep past the
    Grand High Witch is a prosodic contour ACROSS utterances. A corpus of independent
    sentences contains no example of one.
  * **Verifiability.** "The range" is one pair of offsets. Anyone can check what was
    taken, resume from it, or undo it — which random sampling with a seed only
    approximates.

Every run appends to `staging_log.json`.

TAGS AND THE EAR
----------------
Staged clips inherit the ear-confirmed (reader, title) attributes
([[new-reader-ear-rule]]) and enter as `keep` WITHOUT a per-clip audition — the owner's
call for a known quantity: "we don't need to spend as much time on known quantities like
the chosen Libri recordings."

That relaxation applies ONLY to clips with no detected issue. Any clip carrying a QC
finding still enters as `unaudited` and reaches the ear, per the standing rule that every
QC failure is auditioned regardless of source or tier ([[qc-gate-mandatory]]). A book is
a known quantity; a flagged clip inside it is not.

Delivery is the exception to "known quantity", because it is not a property of the
recording the way gender and accent are — it is how a given passage was read, and in a
novel it varies from paragraph to paragraph. So a title that is attribute-confirmed but
not delivery-marked stages under `--delivery-ear`, which takes the same linear range and
enters it unaudited for the lane call. Without it those clips fold as keeps with delivery
blank: real speech in the corpus, invisible to the lane table.

Usage:
    .venv/bin/python scripts/synthesis/stage_pool.py --campaign librivox-v1 --status
    .venv/bin/python scripts/synthesis/stage_pool.py --campaign librivox-v1 --stage 250 --apply
    .venv/bin/python scripts/synthesis/stage_pool.py --campaign librivox-v2 --stage 30 \
        --delivery-ear --apply
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import synth_common  # noqa: E402

DATE = datetime.date.today().isoformat()   # was hard-coded "2026-08-01"

# C-M10 floor. A title-level delivery mark propagates to EVERY clip in the book, so the
# sample that certifies it has to look like the book. Chosen to reject both real samples
# to date (12 contiguous clips in one section; 30 contiguous in one section of 25) while
# passing an honest spread.
MIN_DELIVERY_CLIPS = 8
MIN_DELIVERY_SECTIONS = 3
MIN_SECTION_SPREAD = 0.25

DATASETS = pathlib.Path("/data/model-training/datasets")
RATINGS = DATASETS / "sonora-expressive-registers/ratings.csv"
PROFILES = DATASETS / "reader_profiles.json"
LEDGER = DATASETS / "books_ledger.json"
ATTRS = ("gender", "age", "accent")
# Written into the note column of every machine-folded row. It is the ONLY
# marker that separates a folded keep from an ear-scored one, so consumers that
# count ear evidence match on it (see reader_profile.learn, gate_calibration,
# stage_pool.--mark-delivery).
FOLD_NOTE = "folded: staged unheard under group certification"
LANES = ("Newscaster", "Documentary", "Neutral", "Dialogue", "Speech")


def ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.is_file() else {}


def ledger_key_for(rec: dict, led: dict) -> str:
    """The ledger key for a manifest record, '' if none.

    The manifest's own `ledger_key` is NOT canonical. librivox_fetch writes `lv:<slug>`,
    while book_router keys the ledger by Gutenberg etext-id when one exists — Dickens's
    *Speeches* is `pg:824` in the ledger and `lv:speeches-literary-and-social-...` in the
    manifest, so a direct lookup misses. The librivox project URL is carried by both and
    identifies the same project either way, so it is the reliable join.

    (The right long-term fix is for librivox_fetch to write the canonical key; until then
    every consumer joining these two files needs this.)
    """
    k = rec.get("ledger_key") or ""
    if k in led:
        return k
    url = (rec.get("librivox_url") or "").rstrip("/")
    if url:
        for key, e in led.items():
            if key != "_doc" and (e.get("url") or "").rstrip("/") == url:
                return key
    return ""


def homogeneous_delivery(rec: dict, led: dict | None = None) -> str:
    """Title-level delivery, or '' — see --mark-delivery for why this exists.

    Delivery is normally a PER-CLIP ear judgement and never propagates: it is the
    mix-balance axis, a statement about how a passage was read, and it genuinely varies
    across a novel. A collection of speeches is a different object. Every section of
    Dickens's *Speeches: Literary and Social* is a speech; the title itself carries the
    delivery, and asking the ear the same question 300 times spends it on something the
    source already answers. Owner agreed to this amendment 2026-08-02.

    It is an explicit MARK, never an inference. Nothing here guesses homogeneity from
    a title or a genre — `--mark-delivery` writes it, and only after the ear has already
    said the same thing about a clip from that title.
    """
    led = led if led is not None else ledger()
    return (led.get(ledger_key_for(rec, led)) or {}).get("delivery_homogeneous") or ""


def load_pool(campaign_dir: pathlib.Path) -> list[dict]:
    """Every segmented clip, in reading order."""
    pool: list[dict] = []
    for man in sorted(campaign_dir.rglob("librivox_manifest.jsonl")):
        for line in man.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    pool.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    seen, uniq = set(), []
    for r in pool:                       # last record per id wins, same as qc_gate
        if r.get("id"):
            seen.add(r["id"])
    by_id = {r["id"]: r for r in pool if r.get("id")}
    uniq = list(by_id.values())
    uniq.sort(key=lambda r: (r.get("book") or "", int(r.get("section") or 0),
                             int(r.get("sentence_index") or 0)))
    return uniq


def qc_flagged(campaign_dir: pathlib.Path) -> set[str]:
    """Ids with any failing gate — these keep their ear pass."""
    out: set[str] = set()
    qc = campaign_dir / "qc_measures.jsonl"
    if not qc.is_file():
        return out
    for line in qc.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("id") and not all((r.get("gates") or {}).values()):
            out.add(r["id"])
    return out


def confirmed_tags(rec: dict) -> dict:
    """Ear-confirmed attributes for this clip's (reader, title), or {}."""
    if not PROFILES.is_file():
        return {}
    prof = json.loads(PROFILES.read_text(encoding="utf-8"))
    entry = prof.get(rec.get("reader") or "") or {}
    t = (entry.get("titles") or {}).get(rec.get("book") or "") or {}
    return {a: t[a] for a in ATTRS if t.get(a)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True, help="dir under datasets/, e.g. librivox-v1")
    ap.add_argument("--stage", type=int, default=0, help="how many clips to stage")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--thin-coverage", action="store_true",
                    help="allow --mark-delivery below the coverage floor, for a title that "
                         "is homogeneous by construction (a speech collection). Recorded "
                         "in the ledger as thin_override so the exception is auditable.")
    ap.add_argument("--mark-delivery", metavar="LANE", choices=LANES,
                    help="mark every title in this campaign as delivery-homogeneous in "
                         "LANE, so staging propagates it. Refused unless the ear has "
                         "already said the same about a clip from that title.")
    ap.add_argument("--seed-ear", action="store_true",
                    help="register the minimum needed to unlock a new (reader, title): "
                         "one mid clip per unconfirmed pair, plus every QC-flagged clip")
    ap.add_argument("--delivery-ear", action="store_true",
                    help="stage as usual, but enter every clip unaudited so the ear "
                         "supplies the delivery lane. For a title whose attributes are "
                         "confirmed but whose delivery is not, and is not homogeneous.")
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
    args = ap.parse_args()

    cdir = DATASETS / args.campaign
    log_path = cdir / "staging_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else \
        {"campaign": args.campaign, "runs": []}
    staged_ids = {i for run in log["runs"] for i in run["ids"]}

    pool = load_pool(cdir)
    flagged = qc_flagged(cdir)
    unstaged = [r for r in pool if r["id"] not in staged_ids]

    by_book = collections.Counter(r.get("book") or "?" for r in pool)
    print(f"POOL   {len(pool)} clips  "
          f"({sum(r.get('seconds', 0) for r in pool)/3600:.2f} h)")
    for b, n in by_book.most_common():
        print(f"         {b}: {n}")
    print(f"STAGED {len(staged_ids)} across {len(log['runs'])} run(s)")
    print(f"POOLED {len(unstaged)} still available")
    if args.mark_delivery:
        # The mark PROPAGATES a judgement the ear has already made; it never asserts a new
        # one. So it is checked against every audited clip from the title, and a single
        # disagreement refuses the whole mark — disagreement is the evidence that the
        # title is not homogeneous, which is exactly the thing being claimed.
        # "Heard" means a human actually listened: an ear score is present and
        # the row is not a machine-written fold. Accepting any status other
        # than unaudited let deferred, dropped, reroll and — worst — the rows
        # this same command had previously folded all vote, so a staged run
        # could re-certify the very mark that produced it.
        with RATINGS.open(newline="", encoding="utf-8") as fh:
            heard = {r["id"]: (r.get("delivery") or "").strip()
                     for r in csv.DictReader(fh)
                     if (r.get("score") or "").strip()
                     and FOLD_NOTE not in (r.get("note") or "")
                     and (r.get("status") or "") in ("keep", "relabeled")}
        by_title: dict[str, dict] = {}
        for r in pool:
            by_title.setdefault(r.get("book") or "?", r)
        led = ledger()
        ok = True
        coverage: dict[str, dict] = {}
        for title, rec in sorted(by_title.items()):
            rows = [r for r in pool if (r.get("book") or "?") == title]
            voted = [r for r in rows if heard.get(r["id"])]
            said = {heard[r["id"]] for r in voted}
            secs_heard = {int(r.get("section") or 0) for r in voted}
            secs_total = {int(r.get("section") or 0) for r in rows}
            spread = len(secs_heard) / max(len(secs_total), 1)
            coverage[title] = {"clips_heard": len(voted), "clips_total": len(rows),
                               "sections_heard": sorted(secs_heard),
                               "sections_total": len(secs_total),
                               "section_spread": round(spread, 3)}
            key = ledger_key_for(rec, led)
            if not said:
                print(f"  !! {title}: no audited clip yet — audition one first")
                ok = False
            elif said != {args.mark_delivery}:
                print(f"  !! {title}: the ear said {sorted(said)}, not "
                      f"{{{args.mark_delivery}}} — not homogeneous, refusing")
                ok = False
            # C-M10: unanimity was the WHOLE test — any sample, any size, any distribution.
            # One clip certified a title-level delivery that then propagates to every clip
            # in the book. Both real samples to date are exactly the degenerate shape:
            # librivox-v1's 12 audited clips are one contiguous run in section 2, and
            # librivox-v2's 30 are one run in section 1 of a 25-section novel. It was safe
            # only because the one title actually marked (`pg:824`, *Speeches*) is
            # homogeneous by construction — a property of that book, not of this check.
            #
            # A contiguous run says nothing about a novel: delivery is the mix-balance
            # axis and genuinely varies across one. Demand clips, and demand they be spread.
            elif (len(voted) < MIN_DELIVERY_CLIPS
                  or len(secs_heard) < min(MIN_DELIVERY_SECTIONS, len(secs_total))
                  or spread < MIN_SECTION_SPREAD) and not args.thin_coverage:
                print(f"  !! {title}: coverage too thin to certify a TITLE-level mark — "
                      f"{len(voted)} clip(s) (need {MIN_DELIVERY_CLIPS}) across "
                      f"{len(secs_heard)} of {len(secs_total)} section(s) "
                      f"(need {min(MIN_DELIVERY_SECTIONS, len(secs_total))}, "
                      f"spread {spread:.0%} < {MIN_SECTION_SPREAD:.0%}).")
                print(f"     Sections heard: {sorted(secs_heard)}. A contiguous run says "
                      "nothing about a novel. Audition across the book, or pass "
                      "--thin-coverage if this title is homogeneous by construction "
                      "(a speech collection); the reason is recorded either way.")
                ok = False
            elif not key:
                print(f"  !! {title}: no ledger entry matches "
                      f"{rec.get('librivox_url') or rec.get('ledger_key')} to mark")
                ok = False
            else:
                print(f"  {title}: ear agrees ({args.mark_delivery}) -> "
                      f"{key}.delivery_homogeneous")
        if not ok:
            return 1
        if not args.apply:
            print("\nDRY RUN — pass --apply to write the ledger")
            return 0
        # A-H3/C-M5: re-read under a lock and touch only these keys, rather than writing
        # back the snapshot read at startup. `ledger_key_for` needs the pre-read `led` to
        # resolve its key, so resolve first, then apply to the current contents.
        marks = {ledger_key_for(rec, led): (args.mark_delivery, coverage[title])
                 for title, rec in by_title.items()}

        def _mark(current):
            for k, (value, cov) in marks.items():
                entry = current.setdefault(k, {})
                entry["delivery_homogeneous"] = value
                entry["delivery_marked"] = DATE
                # C-M10: the evidence travels WITH the mark. A title-level delivery
                # propagates to every clip in the book, so "how much of the book did the
                # ear actually hear, and from where" is the first thing anyone auditing
                # this decision needs, and it was recorded nowhere.
                entry["delivery_coverage"] = dict(cov, thin_override=bool(args.thin_coverage))

        synth_common.update_json(LEDGER, _mark)
        print(f"\nmarked {len(by_title)} title(s) in {LEDGER}")
        return 0

    if args.status or (not args.stage and not args.seed_ear):
        for run in log["runs"]:
            # `range` is None for a run that was not a contiguous take — seed-ear cherry-
            # picks, so it records None on purpose (see the write below). The printer has
            # to handle that. It did not, and --status crashed outright on any campaign
            # that had ever been seeded.
            rng_txt = (f"reading-order range {run['range'][0]}..{run['range'][1]}"
                       if run.get("range") else "no contiguous range")
            print(f"  run {run['run']}: {run['count']} clips, "
                  f"{rng_txt} [{run.get('kind', 'stage')}] ({run['date']})")
        return 0

    if args.seed_ear:
        # THE DEADLOCK THIS BREAKS
        # ------------------------
        # stage_pool refuses to stage a (reader, title) with no ear-confirmed attributes
        # ([[new-reader-ear-rule]]). pick_audit_subset can only queue clips that are
        # ALREADY rows in ratings.csv. And stage_pool is the only thing that writes rows
        # for this lane. So a brand-new book could never reach the ear at all: the first
        # audition was unreachable by construction. Found on the first Speech-lane book,
        # 2026-08-02 — the pool/staging split created it and nothing had exercised it yet.
        #
        # The seed is deliberately the MINIMUM that unlocks propagation: one clip per
        # unconfirmed pair, plus every QC-flagged clip (which is owed an ear regardless of
        # source or tier, [[qc-gate-mandatory]]). No tags are written — the whole point is
        # that the ear supplies them.
        pairs: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
        for r in unstaged:
            pairs[(r.get("reader") or "?", r.get("book") or "?")].append(r)
        seed: list[dict] = []
        for pair, recs in sorted(pairs.items()):
            if confirmed_tags(recs[0]):
                print(f"  {pair[0]} / {pair[1]}: already ear-confirmed, no seed needed")
                continue
            mid = recs[len(recs) // 2]        # mid, not an edge — edges are the ragged ones
            seed.append(mid)
            print(f"  {pair[0]} / {pair[1]}: {len(recs)} pooled -> seeding 1 ({mid['id']})")
        extra = [r for r in unstaged if r["id"] in flagged and r not in seed]
        for r in extra:
            print(f"  + QC-flagged, owed an ear regardless: {r['id']}")
        take = seed + extra
        if not take:
            print("nothing to seed — every pair is confirmed and no clip is QC-flagged")
            return 0
    else:
        take = unstaged[: args.stage]
    if not take:
        print("nothing left to stage")
        return 0
    order = {r["id"]: i for i, r in enumerate(pool)}
    rng = [order[take[0]["id"]], order[take[-1]["id"]]]
    n_flag = sum(1 for r in take if r["id"] in flagged)
    # Tags are resolved PER CLIP, not once from take[0]. A staging run can span two books
    # (the pool is sorted by book, then reading order), and one book's confirmed profile
    # must never be written onto another's clips.
    tags_of = {r["id"]: confirmed_tags(r) for r in take}
    led_cache = ledger()
    if args.seed_ear:
        print(f"\nseeding {len(take)} clips for the ear — NO tags, all unaudited")
        print("  this is the minimum that unlocks propagation; it is not a staging run")
    else:
        untagged = [r for r in take if not tags_of[r["id"]]]
        print(f"\nstaging {len(take)} clips, reading-order {rng[0]}..{rng[1]}")
        shown = {k: v for k, v in list(tags_of.values())[0].items()} if tags_of else {}
        print(f"  tags from the ear-confirmed profile: {shown or '(NONE — refusing)'}")
        if args.delivery_ear:
            # WHY THIS MODE EXISTS
            # --------------------
            # Confirmation is not one axis. `confirmed_tags` answers gender/age/accent;
            # delivery is answered either per clip by the ear or, for a title the owner
            # has marked, by `homogeneous_delivery`. A (reader, title) can be fully
            # confirmed on the first and silent on the second — which is precisely what
            # a re-cut produces, since the ear's verdicts are keyed on clip id and the
            # ids do not survive re-segmentation, while the profile is keyed on the pair
            # and does.
            #
            # Staging such a title normally folds every clip as a keep with delivery
            # BLANK, which is a keep the lane table cannot count. This mode stages the
            # same linear range but hands it to the ear instead, so the delivery call
            # gets made. Attributes are still pre-filled from the confirmed profile —
            # the ear is being asked one question, not four.
            marked = {t for t in {r.get("book") or "?" for r in take}
                      if homogeneous_delivery(next(r for r in take
                                                   if (r.get("book") or "?") == t),
                                              led_cache)}
            if marked:
                print(f"  !! already delivery-homogeneous, nothing to ask: "
                      f"{sorted(marked)}")
                print("     stage without --delivery-ear and the mark propagates")
                return 1
            print(f"  all {len(take)} enter as UNAUDITED — the ear supplies delivery")
        else:
            print(f"  {len(take)-n_flag} enter as keep; {n_flag} carry a QC finding -> unaudited")
        if untagged:
            pairs = sorted({(r.get('reader') or '?', r.get('book') or '?')
                            for r in untagged})
            print(f"  !! {len(untagged)} clips have no confirmed (reader, title) profile:")
            for p in pairs:
                print(f"       {p[0]} / {p[1]}")
            print("     seed the ear first:  --seed-ear --apply   ([[new-reader-ear-rule]])")
            return 1
    if not args.apply:
        print("\nDRY RUN — pass --apply to write")
        return 0

    before = RATINGS.stat().st_mtime_ns
    with RATINGS.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        hdr, rows = rd.fieldnames or [], list(rd)
    have = {r["id"] for r in rows}
    added = 0
    for rec in take:
        if rec["id"] in have:
            continue
        row = {k: "" for k in hdr}
        unaudited = args.seed_ear or args.delivery_ear or rec["id"] in flagged
        # A folded clip is a keep — that is the point of folding — but NOBODY
        # HEARD IT, so it must not carry an ear score. It used to be written
        # `score=5`, which is a fabricated ear verdict under the v4 rule that
        # score is vocals/prosody by ear only, and three consumers then trusted
        # it: gate_calibration counted it as an ear keep, audit_sampler counted
        # it as scored history (retiring novelty sampling early), and
        # --mark-delivery accepted it as "the ear said". Blank score + a note
        # keeps the clip in the dataset while making its provenance legible.
        row.update({
            "campaign": f"book-{args.campaign}", "id": rec["id"],
            "engine": rec.get("engine", "librivox"),
            "status": "unaudited" if unaudited else "keep",
            "score": "",
            "note": "" if unaudited else FOLD_NOTE,
            "link": f"../{args.campaign}/audio/{rec['wav']}",
        })
        if not args.seed_ear:
            row.update({a: v for a, v in tags_of[rec["id"]].items() if a in hdr})
            lane = homogeneous_delivery(rec, led_cache)
            if lane and "delivery" in hdr:
                row["delivery"] = lane
        rows.append(row)
        added += 1
    if RATINGS.stat().st_mtime_ns != before:
        print("ABORT: ratings.csv changed under us (live writer)")
        return 1
    fd, tmp = tempfile.mkstemp(dir=str(RATINGS.parent), suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    if RATINGS.stat().st_mtime_ns != before:
        os.unlink(tmp)
        print("ABORT: ratings.csv changed during write")
        return 1
    shutil.copymode(str(RATINGS), tmp)
    os.replace(tmp, str(RATINGS))

    log["runs"].append({
        "run": len(log["runs"]) + 1, "date": DATE, "count": len(take),
        # A seed is NOT a contiguous reading-order range — it is one clip per pair plus
        # the QC-flagged. Recording a range for it would misrepresent what was taken and
        # break the "resume from where staging stopped" contract.
        # delivery-ear IS a contiguous take — it stages the same linear range a normal
        # run would, and only changes who answers the delivery question. So it keeps its
        # range and stays resumable; only seed-ear, which cherry-picks, records None.
        "kind": "seed-ear" if args.seed_ear
                else "delivery-ear" if args.delivery_ear else "stage",
        "range": None if args.seed_ear else rng,
        "qc_flagged": n_flag,
        "tags": None if args.seed_ear else {i: t for i, t in tags_of.items() if t},
        "ids": [r["id"] for r in take],
    })
    # A-H3: staging_log truncation forgets staged ranges, which double-stages clips.
    synth_common.write_json_atomic(log_path, log)
    left = len(pool) - len(staged_ids) - len(take)
    if args.seed_ear:
        print(f"\nseeded {added} clips into ratings.csv as unaudited; {left} still pooled")
        print("  audition them, then:  reader_profile.py --learn --apply")
        print("  then stage the rest:  stage_pool.py --campaign <c> --stage N --apply")
    elif args.delivery_ear:
        print(f"\nstaged {added} new rows into ratings.csv as unaudited; "
              f"{left} left in the pool")
        print("  audition them for the delivery lane; attributes are already filled in")
    else:
        print(f"\nstaged {added} new rows into ratings.csv; {left} left in the pool")
    print(f"  recorded in {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
