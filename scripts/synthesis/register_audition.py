"""register_audition — bridge book-prose synthesis output into the audition queue.

The Dataset Auditions app (audition.ai-lab-0:8095) is ratings.csv-driven: a clip
only appears in the review `todo` queue if a row exists in ratings.csv. The
book-prose lane (book_ingest.py -> synth_{dia,qwen,moss85}.py) stops at producing
wavs + per-engine manifests and never touches ratings.csv, so rendered books never
reach the vetting surface. This script closes that gap.

It walks a book's audio/ dir, reads the *_manifest.jsonl files, and appends one
`unaudited` row per rendered clip to ratings.csv (SSOT). Idempotent: clips whose id
is already in ratings.csv are skipped, so it's safe to re-run per book (or after a
partial render adds more clips).

Rows conform to the CSV's *actual on-disk header* (not an assumed schema), and the
`link` is written relative to RATINGS_DIR so the app's DATA_ROOT confinement passes.

The gender column is pre-populated from casting intent so the auditor only has to
correct it, never enter it (owner standard 2026-07-24): manifest `intended_gender`
when present (vibevoice), else parsed from the `direction.design` / `direction.instruct`
casting text (qwen, moss85). Engines with no casting slot (dia) and deliberately
non-gendered castings (children, non-human entities) are left blank — the app shows
those as Undefined for the auditor to set by ear.

Run:
  .venv/bin/python scripts/synthesis/register_audition.py \
      --book /data/model-training/datasets/book-prose/apple-cart [--dry-run]
  # every book under the book-prose root:
  .venv/bin/python scripts/synthesis/register_audition.py --all [--dry-run]
  # or point straight at a manifests dir (what synth_bank.sh passes as its out_dir):
  .venv/bin/python scripts/synthesis/register_audition.py --audio-dir <out_dir> [--dry-run]
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_common  # noqa: E402

# Mirror the audition app's path resolution (same env overrides).
DATA_ROOT = Path(os.environ.get(
    "AUDITION_DATA_ROOT", "/data/model-training/datasets")).resolve()
RATINGS_DIR = Path(os.environ.get(
    "AUDITION_RATINGS_DIR", str(DATA_ROOT / "sonora-expressive-registers"))).resolve()
RATINGS_CSV = RATINGS_DIR / "ratings.csv"
BOOK_PROSE_ROOT = Path(os.environ.get(
    "BOOK_PROSE_ROOT", str(DATA_ROOT / "book-prose"))).resolve()

# Fallback header if ratings.csv doesn't exist yet (matches app's CSV_FIELDS).
DEFAULT_FIELDS = ["campaign", "id", "engine", "register",
                  "gender", "score", "note", "status", "link"]


def _read_header_and_ids():
    """Return (fieldnames, set-of-existing-ids). Empty defaults if no CSV yet."""
    if not RATINGS_CSV.is_file():
        return DEFAULT_FIELDS, set()
    with open(RATINGS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or DEFAULT_FIELDS
        ids = {r.get("id", "") for r in reader}
    return list(fields), ids


def _manifest_rows(audio: Path):
    """Yield (record, wav_path) for every manifest line in an audio/manifests dir."""
    manifests = sorted(audio.glob("*_manifest.jsonl"))
    if not manifests:
        print(f"  ! no *_manifest.jsonl under {audio}", file=sys.stderr)
    for mf in manifests:
        with open(mf, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ! {mf.name}:{lineno} bad JSON ({e}); skipped",
                          file=sys.stderr)
                    continue
                yield rec, audio / rec.get("wav", "")


# "female" must be tried before "male" (substring); word-boundaries keep
# "human"/"woman-like" phrasing from mismatching.
_FEMALE = re.compile(r"\b(female|woman|girl|feminine)\b", re.I)
_MALE = re.compile(r"\b(male|man|boy|masculine)\b", re.I)


def _predicted_gender(rec) -> str:
    """Casting-intent gender for a manifest record; '' when there is none.

    Priority: explicit intended_gender (vibevoice) > direction.design (qwen) >
    direction.instruct (moss85 embeds the design phrase there). Blank means the
    engine has no casting slot (dia) or the casting is deliberately non-gendered
    (child / non-human) — the app then shows Undefined for the auditor to set.
    """
    g = rec.get("intended_gender") or ""
    if g in ("Male", "Female"):
        return g
    d = rec.get("direction") or {}
    for text in (d.get("design"), d.get("instruct")):
        if not text:
            continue
        if _FEMALE.search(text):
            return "Female"
        if _MALE.search(text):
            return "Male"
    return ""


def _under_data_root(p: Path) -> bool:
    """True if p resolves inside DATA_ROOT (the app confines served audio to it)."""
    p = p.resolve()
    return DATA_ROOT == p or DATA_ROOT in p.parents


def _build_row(rec, wav: Path, book_slug: str, fields):
    """Map a manifest record to a ratings row dict keyed by the on-disk header."""
    link = os.path.relpath(wav.resolve(), RATINGS_DIR)  # e.g. ../book-prose/<b>/audio/x.wav
    full = {
        "campaign": rec.get("campaign") or f"book-{book_slug}",
        "id": rec.get("id", ""),
        "engine": rec.get("engine", ""),
        "register": rec.get("register", ""),
        "gender": _predicted_gender(rec),  # pre-filled from casting intent; auditor corrects
        # Same verify-don't-enter standard as gender (owner 2026-07-24): when the bank
        # states the lane it was DIRECTED toward (delivery-mix campaigns, 2026-07-30),
        # pre-fill it. The auditor's dropdown correction is then itself signal — a
        # Neutral-directed clip re-tagged Newscaster is an engine drifting broadcast.
        "delivery": rec.get("intended_delivery", ""),
        "score": "",
        "note": "",
        "status": "unaudited",  # -> lands in the `todo` filter
        "link": link,
    }
    return {k: full.get(k, "") for k in fields}


# --- QC triage (2026-07-31) ---------------------------------------------------
# Renders used to reach the queue with no objective QC unless somebody ran qc_gate
# by hand, and nothing in synth_bank.sh did. The cost was measured the same day:
# windfairies_nar_0042_neu_MOS was queued at 3.1 s against a 7.9 s floor (ASR WER
# 0.72 — 11 of 40 words) and spent an audition slot proving what the instrument
# already knew.
#
# OWNER RULE 2026-07-31: **every QC failure is auditioned, at every tier, on every
# engine.** An earlier draft of this file auto-rerolled "obviously broken" clips to
# save the ear a slot. That was overruled, and the data says why: the owner scored
# the-return_nar_0051_doc_MOS a 5 with no note while it rendered 20 of its 47 words,
# because MOSS ended cleanly mid-sentence. The ear cannot hear text that is missing —
# so the instrument's job is to TELL the ear what to check, not to decide alone.
# Auto-rerolling would also have hidden the failure from the person calibrating on it.
#
# A QC failure therefore never changes a clip's fate here. It is queued like any other
# clip, with the finding written into the note so the auditor knows what to look for,
# and its id written to qc_flags.txt so pick_audit_subset routes it to the ear however
# thin that engine's sampling tier is.
MIN_SPEECH_SECONDS = 4.0        # owner floor 2026-07-25; keep-rate cliff is exactly here
# Mirrors qc_gate.PAUSE_FLAG_MAX — the advisory band for internal dead air, where
# the clip earns an audition rather than a rejection. Kept in sync by
# test_skill_files.py so the two cannot drift apart silently.
PAUSE_FLAG_SECONDS = 1.4
# Head truncation is measured but NOT gated (C-M4) — see qc_gate's uncalibrated block.
# This is the advisory count at which the auditor is told to listen to the opening. It is
# a REPORTING threshold, not a gate: the cost of it being wrong is an auditor reading one
# extra line, and the whole reason it exists is to accumulate the ear labels a real
# threshold has to be chosen from. Three matches TAIL_WORDS_MIN, so one dropped article
# is not called a truncation at either end.
#
# Imported rather than re-declared since 2026-08-07, when `stage_pool.qc_flagged` gained
# the same test: the number that decides whether a clip is QUEUED and the number that
# decides what the auditor is TOLD have to be one number, or the queue fills with clips
# whose note says nothing is wrong.
HEAD_WORDS_FLAG = synth_common.HEAD_WORDS_FLAG


def _tail_beyond(canonical: str, heard: str) -> str:
    """The words ASR heard AFTER the canonical text ran out.

    Used to pre-fill the suspected missing line when a reader's edition carries text
    ours does not. Matching is on normalised words so punctuation and casing cannot
    hide the join; the returned string is the raw ASR tail, for the owner to verify and
    correct rather than transcribe from scratch.
    """
    import difflib
    import re as _re

    def norm(t):
        return [w for w in _re.sub(r"[^a-z0-9\s]", " ", t.lower()).split() if w]

    a, b = norm(canonical), norm(heard)
    if not a or not b:
        return ""
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    blocks = [bl for bl in sm.get_matching_blocks() if bl.size]
    if not blocks:
        return ""
    last = blocks[-1]
    if last.a + last.size < len(a) - 2:      # canonical text itself was not finished
        return ""
    tail = b[last.b + last.size:]
    if len(tail) < 3:
        return ""
    words = heard.split()
    return " ".join(words[-len(tail):]) if len(words) >= len(tail) else " ".join(tail)


def _qc_triage(qc_path):
    """Read qc_measures.jsonl -> ({id: note}, [flagged ids]). Never changes status."""
    notes, flagged = {}, []
    for line in qc_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        cid, g = r.get("id"), r.get("gates") or {}
        if not cid:
            continue
        why = []
        if not g.get("asr_ok", True):
            heard = len((r.get("asr_hyp") or "").split())
            want = len(str(r.get("text", "")).split())
            # Direction matters: short means text is MISSING, long means the engine
            # improvised or repeated a tail, and the two send the ear looking for
            # opposite things. Saying "check completeness" on an over-run points the
            # auditor at the wrong end of the clip.
            if heard < want * 0.9:
                what = (f"transcribed only {heard} words of {want} — "
                        f"CHECK THE END, text may be MISSING")
            elif heard > want * 1.1:
                # An over-run has TWO causes and they need opposite responses. On a
                # SYNTHESIS clip the engine improvised. On REAL AUDIO a high align_score
                # alongside high WER means the alignment was good and the reader simply
                # spoke words the canonical edition does not contain — measured
                # 2026-08-01 on uneasy-money_lv002_0232, which the owner scored 5/5 and
                # annotated by hand. That is a TEXT gap, not a clip defect, and no
                # alignment work can fix it.
                # Owner rule 2026-08-01: "rule in these possibilities and always subject
                # suspected clips for review. I can then attempt to fill the gap." So
                # name the cause and PRE-FILL the suspected missing words from ASR —
                # verify-don't-enter, the same standard as the attribute columns.
                score = r.get("align_score")
                if score is not None and score >= 0.75:
                    extra = _tail_beyond(r.get("text", ""), r.get("asr_hyp") or "")
                    what = ("alignment is GOOD (align_score "
                            f"{score:.2f}) but {heard} words were spoken against {want} "
                            "in the source — the reader's EDITION LIKELY CONTAINS TEXT "
                            "OURS DOES NOT. Not a clip defect. Suspected missing text, "
                            f"from ASR, please verify: \u201c{extra}\u201d"
                            if extra else
                            "alignment is GOOD (align_score "
                            f"{score:.2f}) but {heard} words were spoken against {want} "
                            "in the source — the reader's EDITION LIKELY CONTAINS TEXT "
                            "OURS DOES NOT. Not a clip defect; please supply the extra "
                            "text.")
                else:
                    what = (f"transcribed {heard} words but the passage has {want} — "
                            f"CHECK FOR AN IMPROVISED OR REPEATED TAIL")
            else:
                what = (f"{heard} words vs {want} in the passage — wording differs; "
                        f"check for misreadings")
            why.append(f"ASR WER {r.get('asr_wer'):.2f}, {what}")
        # tail_ok is the gate WER cannot substitute for: a read that simply stops
        # scores a small global error rate and is still unusable. Report it in its own
        # words, and before the duration message, because it names WHERE to listen.
        if not g.get("tail_ok", True):
            lost, frac = r.get("tail_words_lost"), r.get("tail_lost_frac") or 0
            why.append(f"stops early — last {lost} words ({frac*100:.0f}% of the passage) "
                       f"never spoken. LISTEN TO THE END")
        # The other end (C-M4). Advisory, not a gate: nothing has calibrated a head
        # threshold, and nothing CAN until the ear has described the failure at least
        # once — searching every drop note in ratings.csv for a phrase naming a late
        # start returns nothing, because no auditor has ever been pointed at it. This
        # line is how those labels start existing. Measured over 3,189 clips: 19 drop
        # three or more opening words and 8 of those passed every gate.
        head_lost = r.get("head_words_lost") or 0
        if head_lost >= HEAD_WORDS_FLAG:
            frac = r.get("head_lost_frac") or 0
            # REWORDED 2026-08-08 after the first four verdicts came back. It used to say
            # "starts late — first N words never spoken", and that was false in three of
            # the four: two were keeps at 5 ("I hear all of the words in the printed
            # text") and one was dropped for shrillness. What the measure detects is that
            # ASR could not ALIGN the opening, which has several causes and only one of
            # them is truncation — so the note now names the alternatives instead of
            # asserting the rarest one. Telling an auditor what to conclude is how a
            # measure launders itself into a finding.
            why.append(f"ASR could not match the first {head_lost} words "
                       f"({frac*100:.0f}% of the passage). LISTEN TO THE OPENING — it may "
                       f"be a late start, words slurred or run together, or nothing at all "
                       f"(ASR is weakest on the first word, which has no left context). "
                       f"MEASURED BUT NOT GATED: say what you actually hear")
        # Internal dead air. Named before the duration line for the same reason
        # tail_ok is: it tells the auditor WHERE in the clip to listen. The
        # advisory band exists because a long pause is sometimes a real dramatic
        # choice — 3.3% of ear-keeps land in it — so this queues the clip rather
        # than condemning it. Past the hard cap the gate has already failed it.
        pause = r.get("worst_pause")
        if pause is not None and pause >= PAUSE_FLAG_SECONDS:
            verdict = ("exceeds the hard cap" if not g.get("pause_ok", True)
                       else "suspicious but within the cap")
            why.append(f"{pause:.1f}s of silence mid-clip ({verdict}) — "
                       f"LISTEN TO THE MIDDLE for a stall or a rushed resumption")
        speech = r.get("speech_dur")
        if speech is not None and speech < MIN_SPEECH_SECONDS:
            why.append(f"only {speech:.1f}s of speech (floor {MIN_SPEECH_SECONDS}s)")
        if not g.get("duration_ok", True) and len(why) == 0:
            why.append("duration outside the expected band for this passage length")
        if not g.get("length_ok", True):
            why.append(f"{r.get('duration', 0):.0f}s exceeds the 30s training cap — the read "
                       f"may be fine; the LINE needs splitting, not a reroll")
        if not g.get("dnsmos_ok", True):
            why.append(f"DNSMOS {r.get('dnsmos_ovr', 0):.2f} below floor")
        if not g.get("measures_ok", True):
            why.append("phonation measures unavailable")
        if why:
            notes[cid] = "QC: " + "; ".join(why)
            flagged.append(cid)
    return notes, flagged


def register_audio_dir(audio: Path, slug: str, existing_ids, fields, dry_run: bool,
                       qc=None):
    """Return list of new row-dicts for one manifests dir; reports skips/missing."""
    new_rows, seen = [], set(existing_ids)
    added = skipped = missing = outside = qc_noted = 0
    for rec, wav in _manifest_rows(audio):
        cid = rec.get("id", "")
        if not cid:
            print(f"  ! manifest record without id in {slug}; skipped",
                  file=sys.stderr)
            continue
        if cid in seen:
            skipped += 1
            continue
        if not wav.is_file():
            print(f"  ! wav missing for {cid}: {wav}; skipped", file=sys.stderr)
            missing += 1
            continue
        if not _under_data_root(wav):
            # The audition app only serves audio under DATA_ROOT; a row pointing
            # outside it would 404 in the UI, so don't queue an unplayable clip.
            print(f"  ! {cid} wav is outside DATA_ROOT ({wav}); not queued",
                  file=sys.stderr)
            outside += 1
            continue
        row = _build_row(rec, wav, slug, fields)
        if qc and cid in qc:
            # Note only — status stays `unaudited`. The finding guides the ear; it
            # never decides for it (owner rule, see _qc_triage).
            row["note"] = qc[cid]
            qc_noted += 1
        new_rows.append(row)
        seen.add(cid)          # guard against dup ids across manifests
        added += 1
    gendered = sum(1 for r in new_rows if r.get("gender"))
    verb = "would add" if dry_run else "queued"
    extra = "".join([f", {missing} missing-wav" if missing else "",
                     f", {outside} outside-data-root" if outside else "",
                     f", {qc_noted} carrying a QC finding" if qc_noted else ""])
    print(f"  {slug}: {verb} {added} for the ear ({gendered} gender-prefilled), "
          f"skipped {skipped} (already present){extra}")
    return new_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", action="append", default=[], metavar="DIR",
                    help="book-prose book dir (repeatable); scans DIR/audio. "
                         "e.g. .../book-prose/apple-cart")
    ap.add_argument("--audio-dir", action="append", default=[], metavar="DIR",
                    help="a manifests dir directly (repeatable); slug = its parent "
                         "name. This is what synth_bank.sh passes as its out_dir.")
    ap.add_argument("--all", action="store_true",
                    help=f"process every book under {BOOK_PROSE_ROOT}")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be added without writing")
    ap.add_argument("--qc", metavar="QC_MEASURES.JSONL", default=None,
                    help="qc_gate output. Clips failing ASR or the 4s speech floor are "
                         "registered as `reroll` instead of reaching the ear; every "
                         "other gate failure is still queued and also written to "
                         "<campaign>/qc_flags.txt for pick_audit_subset --flags.")
    args = ap.parse_args()

    qc_map, advisory = None, []
    if args.qc:
        qc_path = Path(args.qc)
        if not qc_path.is_file():
            ap.error(f"--qc file not found: {qc_path}")
        qc_map, advisory = _qc_triage(qc_path)
        print(f"QC triage: {len(qc_map)} clips carry a QC finding — all queued for the "
              f"ear regardless of engine tier")
        if advisory and not args.dry_run:
            # Append-dedup, not clobber. qc_engine_defects.py --append-flags
            # writes the same file for the scrutinized engines, so whichever
            # ran second used to erase the other's ids — and pick_audit_subset
            # --flags then never saw them, quietly breaking "every QC failure
            # is auditioned" for exactly the engines that need it most.
            flags = qc_path.parent / "qc_flags.txt"
            have = set()
            if flags.is_file():
                have = {ln.strip() for ln in flags.read_text(encoding="utf-8").splitlines()
                        if ln.strip()}
            new = [i for i in advisory if i not in have]
            with open(flags, "a", encoding="utf-8") as f:
                for i in new:
                    f.write(i + "\n")
            print(f"wrote {flags} ({len(new)} new ids, "
                  f"{len(advisory) - len(new)} already present)")

    # Build the list of (audio_dir, slug) targets from whichever flags were given.
    targets = []
    if args.all:
        for d in sorted(BOOK_PROSE_ROOT.iterdir()):
            if (d / "audio").is_dir():
                targets.append((d / "audio", d.name))
    for b in args.book:
        bd = Path(b).resolve()
        targets.append((bd / "audio", bd.name))
    for a in args.audio_dir:
        ad = Path(a).resolve()
        targets.append((ad, ad.parent.name))  # e.g. .../apple-cart/audio -> apple-cart
    if not targets:
        ap.error("give --book <dir>, --audio-dir <dir> (both repeatable), or --all")

    fields, existing_ids = _read_header_and_ids()
    print(f"ratings.csv: {RATINGS_CSV} ({len(existing_ids)} existing ids, "
          f"{len(fields)}-col header)")

    all_new = []
    for audio, slug in targets:
        if not audio.is_dir():
            print(f"  ! {audio} is not a dir; skipped", file=sys.stderr)
            continue
        all_new += register_audio_dir(audio, slug, existing_ids, fields, args.dry_run)

    if not all_new:
        print("nothing to add.")
        return
    if args.dry_run:
        print(f"[dry-run] {len(all_new)} rows would be appended to ratings.csv")
        return

    append_guarded(all_new, fields)


def append_guarded(all_new, fields, attempts=5):
    """Append rows, verifying they survived a concurrent app write.

    Public on purpose: this is the ONE guarded writer for ratings.csv, and every
    script that appends to it must come through here. audit_sampler used to
    hand-roll a bare append and silently lost rows.

    ⚠ The audition app is a LIVE WRITER and saves via read-all -> modify -> write
    tmp -> os.replace (audition/app/main.py `_write_rows`). The replace is atomic, so a
    plain O_APPEND here can never corrupt the file — but rows appended inside the app's
    read/write window are SILENTLY LOST when the swap lands. An auditor saving one rating
    at the wrong moment would drop a whole campaign's queue with no error anywhere.

    So: append, then read back and confirm our ids are actually on disk. If the file was
    replaced underneath us, re-append only what went missing. Verification is the guard —
    checking mtime beforehand only narrows the window, it does not close it.
    """
    new_file = not RATINGS_CSV.is_file()
    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    pending = list(all_new)
    for attempt in range(1, attempts + 1):
        with open(RATINGS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
                new_file = False
            w.writerows(pending)
        _, on_disk = _read_header_and_ids()
        missing = [r for r in pending if r["id"] not in on_disk]
        if not missing:
            print(f"appended {len(all_new)} unaudited rows to ratings.csv"
                  + (f" (took {attempt} attempts; app wrote concurrently)" if attempt > 1 else ""))
            return
        print(f"  ! {len(missing)} row(s) lost to a concurrent app write; "
              f"retrying ({attempt}/{attempts})", file=sys.stderr)
        pending = missing
    print(f"ERROR: {len(pending)} row(s) still missing after {attempts} attempts. "
          f"Is someone auditioning right now? Re-run when the app is idle.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
