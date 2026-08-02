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

`rebuild_used_set` — the variety-bias `used` set starts empty on a resume,
because already-rendered jobs skip before select_reference() ever runs. A
rerolled clip then casts as if the reference pool were untouched, and the
casting is not reproducible from the bank. Rebuilding from the manifest rows
that already exist restores it.
"""

import json
import os


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
