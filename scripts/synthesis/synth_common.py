"""Shared harness pieces for the synth_*.py renderers.

Eight renderers grew from one hand-copied template, so a bug fixed in one
survived in the others (the explicit-keys manifest bug was fixed in six and
still lived in two). Anything that every renderer must get identically right
belongs here instead.

What lives here so far:

`write_wav_atomic` — every renderer wrote its wav in place and keyed its
resume on the wav merely *existing*. A crash inside the write (OOM, a killed
container, a full disk) therefore left a truncated wav that every later run
skipped, that no manifest row described, and that silently vanished from the
campaign while the file count still looked right. A corrupt header could also
take down the loudnorm pass for the whole batch. Writing to a temp file in the
same directory and renaming is atomic on POSIX, so a clip is either wholly
there or wholly absent.

`update_json` — the ledger/log files (`books_ledger.json`, `staging_log.json`)
were read once at startup, held across minutes of network work, then written
back whole. Two overlapping runs therefore each wrote a snapshot taken before
the other started, and the later one silently erased the earlier one's entries
(A-H3). The write was also a bare `write_text`, so a crash mid-write left a
truncated JSON file that nothing could parse. This does the read-modify-write
INSIDE an exclusive lock, immediately before the write, and renames into place.

`ratings_transaction` — ratings.csv is the ear-verdict SSOT and the audition
app is a live writer. Six scripts each grew their own mtime stamp, in four
flavours, and one had none. This takes an flock (which serialises our scripts
against each other, something mtime cannot do) AND re-checks mtime inside it
(which is the only thing that catches the app, since the app takes no lock).

`attempt_seed` — a re-run under skip-if-exists is the retry mechanism, and it
re-seeded identically every time, so a DETERMINISTIC failure could never
converge: the same seed reproduces the same malformed generation forever. The
count has to persist because a failure leaves nothing behind to count.

`rebuild_used_set` — the variety-bias `used` set starts empty on a resume,
because already-rendered jobs skip before select_reference() ever runs. A
rerolled clip then casts as if the reference pool were untouched, and the
casting is not reproducible from the bank. Rebuilding from the manifest rows
that already exist restores it.
"""

import contextlib
import csv
import datetime as _dt
import fcntl
import json
import os
import re
import shutil


@contextlib.contextmanager
def _exclusive(path):
    """flock an adjacent `.lock` file for the duration of a read-modify-write.

    The lock is a SIDECAR rather than the file itself: the update replaces the target
    by rename, so a lock held on the old inode would protect nothing once the rename
    lands. Advisory locks only bind processes that ask for them, which is every writer
    that goes through `update_json` — the point is to make that the only way to write.
    """
    lock_path = f"{path}.lock"
    with open(lock_path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _atomic_write(path, emit, mode="w", **open_kw):
    """tmp + fsync + rename, and clean the tmp up when any of that fails.

    `write_wav_atomic` and `save_via_atomic` already swept their tmp on failure; these two
    did not, so an interrupted write left a `.name.tmp.json` beside the real file. That is
    not merely litter — these files sit in dataset directories that other tools glob, and
    a half-written twin of the ledger is exactly the kind of thing a later `rglob("*.json")`
    picks up and reads as real.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    tmp = _tmp_path(directory, os.path.basename(path))
    try:
        with open(tmp, mode, **open_kw) as fh:
            emit(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def write_json_atomic(path, obj, *, indent=1):
    """Write JSON that is either wholly the new content or wholly the old."""
    _atomic_write(path, lambda fh: json.dump(obj, fh, indent=indent, ensure_ascii=False),
                  encoding="utf-8")


def write_text_atomic(path, text, *, encoding="utf-8"):
    """Same guarantee as `write_json_atomic`, for the plain-text siblings."""
    _atomic_write(path, lambda fh: fh.write(text), encoding=encoding)


def update_json(path, mutate, *, default=dict, indent=1):
    """Re-read, apply `mutate`, and write back — all under one exclusive lock.

    `mutate(obj)` edits in place; its return value is ignored unless it returns a dict,
    which then replaces the object. Returns the object that was written.

    THE POINT IS THE RE-READ. Callers legitimately load these files at startup to decide
    what work to do, and that snapshot is minutes stale by the time the work finishes.
    Writing it back is what loses a concurrent run's entries. Mutating the *current*
    contents here means a stale in-memory copy can no longer overwrite anything but the
    keys the caller actually touched.
    """
    with _exclusive(path):
        try:
            with open(path, encoding="utf-8") as fh:
                obj = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            obj = default()
        replaced = mutate(obj)
        if isinstance(replaced, dict):
            obj = replaced
        write_json_atomic(path, obj, indent=indent)
        return obj


# ------------------------------------------------------------- utterance completeness
#
# ONE definition of "is this a whole utterance". There were two: book_ingest's and
# qc_passages', the latter carrying a comment saying it was "kept in sync" — and they had
# already drifted, qc_passages accepting guillemets and trailing spaces that book_ingest
# rejected. The stated reason for duplicating was that importing book_ingest drags in its
# epub/pysbd chain; this module has no such chain (json, os, fcntl), so that reason is
# gone rather than argued with.
_TERMINALS = ".!?…"
_CLOSERS = "”\"')]»’ "
_OPENERS = "“\"'([«‘ "


def is_complete_utterance(text):
    """Is this text a whole utterance a listener could judge a performance of?

    THE GATE THAT MATTERS FOR AUDITABILITY (owner, 2026-07-28). A clip that is merely
    short stays judgeable — the sentence is still there. A clip that is INCOMPLETE
    destroys the judgement itself: prosody lives in the arc of a finished utterance, so a
    fragment ending on a comma has no terminal contour to assess, and no amount of vocal
    quality rescues it.

    This is why the gate is completeness and NOT a token-count floor. audit-markup-v0 read
    its <=6-token failures as a LENGTH problem and prescribed a floor; length was the
    correlate, not the cause. A floor rejects good short-but-whole clips and passes every
    long fragment.

    The rating vocabulary cannot see this defect: under v4 the score means vocals and
    prosody only, so fragments were rated 4-5 for a fine-sounding voice and are
    statistically invisible in ratings.csv. 79 reached the audit surface across two
    independent lanes. The instrument will not catch this; the gate has to.
    """
    t = (text or "").strip()
    if not t:
        return False
    core = t.rstrip(_CLOSERS).rstrip()
    if not core or core[-1] not in _TERMINALS:
        return False        # ends on a comma, dash, conjunction — mid-utterance
    head = t.lstrip(_OPENERS).lstrip()
    return bool(head) and (head[0].isupper() or head[0].isdigit())


def split_sentences(text):
    """Sentence split that knows about abbreviations (A-H5).

    pysbd, not a regex. The real-audio lane used `(?<=[.!?])["”’')\\]]*\\s`, which splits
    on any period followed by whitespace — so "Mr. Smith went home." becomes "Mr." and
    "Smith went home.", and BOTH halves are shipped as clips. The first is a fragment; the
    second starts mid-sentence. That is the completeness rule broken by the very step that
    is supposed to produce whole utterances, and the audit score cannot see it.

    pysbd is imported lazily so that merely importing this module stays free for the
    callers that only want the atomic writers.
    """
    import pysbd

    seg = pysbd.Segmenter(language="en", clean=False)
    return [s.strip() for s in seg.segment(text) if s.strip()]


# ------------------------------------------------------------- Gutenberg boilerplate
#
# A-M8. Project Gutenberg wraps every edition in a header (title/author/release date/
# "Produced by …") and a footer that is the FULL Project Gutenberg License — some four
# thousand words of terms of use. The plaintext lane in `librivox_fetch` cut it on the
# `*** START/END OF …` markers; the **epub** lane in `book_ingest` did not cut it at all.
# Its only filter was a filename SKIP list, and that list is Standard Ebooks' vocabulary
# (`titlepage`, `imprint`, `colophon`, `uncopyright`) — PG names its documents nothing of
# the sort, and often puts the entire book INCLUDING the boilerplate in one document, so
# no filename rule could have worked.
#
# The result is licence prose parsed as prose: paragraphs of terms of use split into
# sentences, directed by Gemma, and rendered by a teacher engine as though they were the
# novel. `is_complete_utterance` passes them — they are grammatical sentences.
#
# And it makes a claim in the record false. `book_ingest.text_provenance` stamps every PG
# bank with "PG header/footer stripped, so the Project Gutenberg License does not attach",
# which is the same class of defect as A-M6 (SE CC0 claimed for PG text): metadata
# asserting a step that no code performed, propagating into every derived clip's paper
# trail where a later audit reads it as fact.
#
# ONE implementation, here, for both lanes.

PG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
PG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)

# Phrases that occur in PG boilerplate and effectively never in the prose of a novel.
# Used only to DETECT residue after the markers have been applied — never to filter, so a
# false positive costs a loud refusal rather than a silently deleted paragraph of the book.
_PG_RESIDUE = (
    "project gutenberg literary archive foundation",
    "project gutenberg-tm",
    "project gutenberg™",
    "www.gutenberg.org",
    "gutenberg.org/license",
    "terms of the project gutenberg",
    "full project gutenberg license",
    "redistributing project gutenberg",
    "start of the project gutenberg ebook",
    "end of the project gutenberg ebook",
)


def strip_pg_boilerplate_text(text):
    """Cut PG header/footer from a plaintext edition, on the `*** START/END ***` markers."""
    m = PG_START_RE.search(text)
    if m:
        text = text[m.end():]
    m = PG_END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text


def strip_pg_boilerplate_paragraphs(paras):
    """Cut PG header/footer from an epub's flat paragraph list.

    The markers can sit anywhere: their own paragraph, or inline at the head of the first
    real one. So the marker's paragraph is cut AT the marker rather than dropped whole —
    dropping it whole would lose the opening sentence of the book on the editions that
    inline it, which is a silent one-sentence truncation nothing downstream could see.
    """
    out, started = [], None
    for i, p in enumerate(paras):
        m = PG_START_RE.search(p)
        if m:
            started = i
            tail = p[m.end():].strip()
            out = [tail] if tail else []
            continue
        m = PG_END_RE.search(p)
        if m:
            head = p[: m.start()].strip()
            if head:
                out.append(head)
            return out
        out.append(p)
    # No START marker anywhere means this is not a PG-wrapped document (a Standard Ebooks
    # epub, a hand-supplied file); returning the input unchanged is correct, not a miss.
    return out if started is not None else list(paras)


def pg_boilerplate_residue(paras):
    """-> the paragraphs that still look like PG boilerplate after stripping.

    Detection, deliberately separate from removal. PG has changed its wrapper format
    several times and older editions predate the `***` markers entirely, so the cut above
    cannot be assumed to have worked — and the failure it would otherwise have is silent.
    The caller refuses; it does not filter. A paragraph of a novel that genuinely says
    "www.gutenberg.org" does not exist, but if one did, deleting it quietly would be the
    worse outcome.
    """
    hits = []
    for p in paras:
        low = (p or "").lower()
        if any(marker in low for marker in _PG_RESIDUE):
            hits.append(p)
    return hits


# ------------------------------------------------------------- books_ledger identity
#
# A-M9 / A-M10. One book has TWO possible ledger keys — `pg:<etext_id>` when the Gutenberg
# edition is known, `lv:<librivox-slug>` when it is not — and which one it gets depends on
# whether the LibriVox API happened to return `url_text_source` on the pass that created
# it. Nothing reconciled them, so:
#
#   * `book_router`'s already-in-ledger check only ever built a `pg:` key. A LibriVox book
#     without an etext id was checked against NOTHING, so every re-route re-created it and
#     reset `status` to "pending force-align ingest" — regressing a book that was already
#     fetched and aligned, silently, because the router prints SKIP only for what it found.
#   * The same book could land under BOTH keys across two passes (API flaky once), giving
#     two ledger entries and two states for one book.
#   * `librivox_fetch` derived its on-disk directory from `key.replace(":", "_")`, so
#     `--key pg:6684` wrote `pg_6684` and `--url <librivox>` wrote
#     `lv_uneasy-money-by-p-g-wodehouse` — the same book downloaded twice into two trees
#     by invocation style alone.
#   * And it updated `status` only under `if args.key and args.key in ledger`, so a fetch
#     driven by `--url` left the ledger saying "pending force-align ingest" forever.
#
# The identity of a book is not one key; it is the SET of keys it could have. These two
# functions are that rule, in one place, for both scripts.


def etext_id_from(url_text_source):
    """-> the Gutenberg etext id in a text-source URL, or None.

    Moved here from `book_router` when `librivox_fetch` needed it too (A-M10): the fetch
    lane cannot resolve a book's canonical key without it, and it is the same rule.
    """
    if not url_text_source:
        return None
    m = re.search(r"(?:ebooks/|etext/|files/)(\d+)", url_text_source)
    return m.group(1) if m else None


def parse_playtime(value):
    """LibriVox playtime -> seconds, or None when it says nothing usable.

    A-M1. The API is not consistent about this field: most projects give a count of
    seconds as a string ("2105"), some give "hh:mm:ss" or "mm:ss", and `totaltime` is
    always the colon form. The aligner did `float(sec.get("playtime") or 0)` inside a
    try/except that swallowed the failure and moved on, so a colon-formatted section
    simply VANISHED from the duration map — and the cumulative fraction that drives the
    text split was then computed over a subset, putting the surviving sections' windows in
    the wrong place and leaving gaps the missing ones fell into. The windows stopped
    tiling, quietly.

    Returns None rather than 0.0 for "unknown": zero is a real duration and the caller has
    to tell the difference. All three books on disk today use the seconds form, so this is
    the format we would meet on the next project rather than a defect in the current data.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        try:
            parts = [float(p) for p in text.split(":")]
        except ValueError:
            return None
        seconds = 0.0
        for part in parts:            # h:m:s or m:s, most significant first
            seconds = seconds * 60 + part
        return seconds if seconds > 0 else None
    try:
        seconds = float(text)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _url_parts(url):
    """-> (host, [path segments]) lowercased, or ("", []) for nothing usable."""
    if not url:
        return "", []
    import urllib.parse

    p = urllib.parse.urlparse(url)
    return p.netloc.lower(), [s for s in p.path.strip("/").lower().split("/") if s]


def ledger_key_candidates(*, url=None, etext_id=None):
    """Every key this book could legitimately be filed under, canonical first.

    Three live schemes, all present in the ledger today:

      `pg:<etext_id>`        the Gutenberg text — leads, because it identifies the TEXT,
                             which survives a re-recording and is what the LibriTTS
                             overlap check matches on
      `se:<author>/<title>`  a Standard Ebooks edition (15 hand-authored entries use it)
      `lv:<project-slug>`    a LibriVox recording whose text source is unresolved

    The last entry in the SE list is a LEGACY form and is deliberately still probed. The
    router's old fallback was `"lv:" + <last path segment>` applied to whatever URL it
    held, including Standard Ebooks ones — which is why five SE books sit in the ledger
    today under `lv:` keys (`lv:walden`, `lv:conan-stories`,
    `lv:the-voyage-of-the-beagle`, `lv:up-from-slavery`,
    `lv:the-autobiography-of-benjamin-franklin`, all `lane=synthesize`). Probing the old
    spelling keeps those findable instead of orphaning them behind a corrected key. They
    are not rewritten here: re-keying live ledger data is a migration, and a migration is
    the owner's call.
    """
    keys = []
    if etext_id:
        keys.append(f"pg:{etext_id}")
    host, parts = _url_parts(url)
    if "standardebooks.org" in host:
        if len(parts) >= 3 and parts[0] == "ebooks":
            keys.append(f"se:{parts[1]}/{parts[2]}")
        if parts:
            keys.append(f"lv:{parts[-1]}")          # legacy mis-key, probe only
    elif "librivox.org" in host and parts:
        keys.append(f"lv:{parts[-1]}")
    return keys


def resolve_ledger_key(ledger, *, url=None, etext_id=None):
    """-> (key, entry) for this book, preferring a key the ledger ALREADY uses.

    Returns `(None, None)` when the book cannot be identified at all. `entry` is None for
    a book that is genuinely new — the caller distinguishes "new" from "already recorded",
    which is exactly the distinction the router was unable to make.
    """
    candidates = ledger_key_candidates(url=url, etext_id=etext_id)
    for key in candidates:
        entry = (ledger or {}).get(key)
        if isinstance(entry, dict):
            return key, entry
    return (candidates[0] if candidates else None), None


def _tmp_path(directory, basename):
    """Hidden sibling temp name that keeps the original extension last."""
    stem, ext = os.path.splitext(basename)
    return os.path.join(directory, f".{stem}.tmp{ext or '.wav'}")


def write_wav_atomic(path, data, samplerate, *, subtype=None):
    """Write a wav that is either complete or not present at all.

    Same signature shape as soundfile.write, with the temp+rename around it.
    """
    import soundfile as sf

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    # Same directory, so the rename is a metadata operation on one filesystem.
    # The extension has to stay LAST: soundfile infers the container format
    # from it, and a name ending in ".tmp" raises instead of writing a wav.
    tmp = _tmp_path(directory, os.path.basename(path))
    try:
        if subtype is None:
            sf.write(tmp, data, samplerate)
        else:
            sf.write(tmp, data, samplerate, subtype=subtype)
        os.replace(tmp, path)
    except BaseException:
        # Includes KeyboardInterrupt/SystemExit: a half-written temp file must
        # never be left where a later glob could find it.
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    return path


def save_via_atomic(save_fn, path, *args, **kwargs):
    """Atomic wrapper for renderers whose library owns the file write.

    Dia's processor.save_audio takes a path and writes it itself, so the temp
    file has to be handed to the library and renamed afterwards.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = _tmp_path(directory, os.path.basename(path))
    try:
        save_fn(*args, tmp, **kwargs)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    return path


class DryRun(Exception):
    """Raised inside `ratings_transaction` to leave without writing. Never escapes."""


@contextlib.contextmanager
def ratings_transaction(path, *, tag="edit", backup=True, dry_run=False):
    """Read ratings.csv, yield (fieldnames, rows), write it back — no window (C-M6/D-M5).

    `ratings.csv` is the ear-verdict SSOT and the Dataset Auditions app is a LIVE WRITER,
    so a script that reads it, thinks, and writes back can erase an edit committed in
    between. That happened on 2026-07-26 (an owner-set accent value), which is why six
    scripts each grew an mtime stamp — in four different flavours, the widest of which
    (`seed_delivery`) stamps before serializing ~1,500 rows and never re-checks. One had
    none at all (`tag_spike`, D-M5).

    Two guards, because they cover different writers:

      * an **flock** on a sidecar, which serialises OUR scripts against each other —
        mtime stamping cannot do this, it can only notice afterwards that it lost;
      * an **mtime re-check inside the lock**, because the app does not take the lock, so
        it is the only thing that can catch it. Checked immediately before the write, so
        the vulnerable window is the write itself rather than the whole run.

    Mutate `rows` in place; the write happens on clean exit. Raise, and nothing is written.

    `dry_run=True` reads and yields exactly as normal and then does NOT write. Every one of
    the six converted scripts has a report-only mode, and without this each would either
    have to rebuild the read itself (which is the duplication being removed) or write a
    byte-identical file — taking a backup and moving the mtime for a run that was supposed
    to change nothing, which is precisely the kind of surprise this file is about.
    """
    path = os.fspath(path)
    with _exclusive(path):
        mtime_at_read = os.stat(path).st_mtime_ns
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames, rows = reader.fieldnames, list(reader)

        try:
            yield fieldnames, rows
        except DryRun:
            return

        if dry_run:
            return

        if os.stat(path).st_mtime_ns != mtime_at_read:
            raise SystemExit(
                f"ABORTED: {path} changed while this was running — the audition app\n"
                "  committed an edit. Writing now would erase it. Nothing was modified;\n"
                "  re-run once the app is idle."
            )
        if backup:
            stamp = _dt.datetime.now().strftime("%Y%m%d")
            bak = f"{path}.bak-{stamp}-{tag}"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
        directory = os.path.dirname(os.path.abspath(path)) or "."
        tmp = _tmp_path(directory, os.path.basename(path))
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)


def attempt_seed(out_dir, job_id, base_seed):
    """-> (seed, attempt). Perturbs the seed once per attempt at this clip (B-M5).

    The renderers seed with `torch.manual_seed(job["seed"])` and treat "run the script
    again under skip-if-exists" as the retry mechanism. That works for a STOCHASTIC failure
    and is useless against a deterministic one: the same seed reproduces the same malformed
    audio-code stream forever, so a clip that hit MOSS's `split_with_sizes` off-by-one is
    stuck, and every re-run burns the same GPU time to fail identically.

    The attempt count has to persist, because the failure leaves nothing behind — no wav,
    no manifest row — so a fresh process cannot tell a first try from a fiftieth. It lives
    in `<out>/.attempts.json`, written through `update_json` so two renderers sharing an
    output directory cannot lose each other's counts.

    The seed is RETURNED rather than applied so the caller can record it in the manifest:
    a clip rendered on attempt 3 was rendered with a different seed than the bank says, and
    that has to be recoverable.
    """
    path = os.path.join(out_dir, ".attempts.json")
    attempts = {}

    def bump(current):
        current[job_id] = int(current.get(job_id, 0)) + 1
        attempts.update(current)

    update_json(path, bump)
    attempt = attempts[job_id] - 1          # 0 on the first try
    return base_seed + attempt, attempt


def rebuild_used_set(manifest_path, key="ref_id"):
    """Reference ids already spent in this campaign, from the manifest.

    Returns an empty set when the manifest does not exist yet (a fresh run).
    Malformed lines are skipped rather than fatal: the manifest is append-only
    and a torn last line from an earlier crash should not block a resume.
    """
    used = set()
    if not os.path.isfile(manifest_path):
        return used
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = rec.get(key) or (rec.get("direction") or {}).get(key)
            if val:
                used.add(val)
    return used


def rendered_ids(manifest_path):
    """Clip ids that already have a manifest row.

    Stronger resume key than "the wav exists": a wav with no manifest row is
    an incomplete render, not a finished one.
    """
    done = set()
    if not os.path.isfile(manifest_path):
        return done
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id"):
                done.add(rec["id"])
    return done
