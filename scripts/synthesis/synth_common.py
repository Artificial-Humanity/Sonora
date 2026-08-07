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

`rebuild_used_set` — the variety-bias `used` set starts empty on a resume,
because already-rendered jobs skip before select_reference() ever runs. A
rerolled clip then casts as if the reference pool were untouched, and the
casting is not reproducible from the bank. Rebuilding from the manifest rows
that already exist restores it.
"""

import contextlib
import fcntl
import json
import os


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


def write_json_atomic(path, obj, *, indent=1):
    """Write JSON that is either wholly the new content or wholly the old."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    tmp = _tmp_path(directory, os.path.basename(path))
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_text_atomic(path, text, *, encoding="utf-8"):
    """Same guarantee as `write_json_atomic`, for the plain-text siblings."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    tmp = _tmp_path(directory, os.path.basename(path))
    with open(tmp, "w", encoding=encoding) as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


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
