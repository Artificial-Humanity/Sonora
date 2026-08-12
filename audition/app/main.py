"""Dataset Auditions — rating surface for the Sonora expressive-registers dataset.

Serves clip audio inline (audio/wav, no browser download dialog, plays on the
iPad too) and makes ratings.csv the single editable source of truth. Replaces the
sandboxed Excel workflow.

Rating vocabulary (v4, owner 2026-07-26 — see notes/synthesis-pipeline.md):
    Keep 1-5      -> score "1".."5",  status keep
    Drop          -> score "x",       status dropped
    Retake        -> score "0",       status reroll   (+ re-roll queue; Phase 2 fires it)

**The 1-5 score means VOCAL AND PROSODIC QUALITY ONLY.** It is a single axis, not a
composite verdict. Categorisation — register, gender, accent — is corrected with the
per-attribute dropdowns and never folded into the number. So a clip with excellent
delivery but the wrong register scores high AND gets its register fixed; the two facts
are recorded separately because they are separate facts.

Why it matters downstream: under the old composite scoring, a low number could mean bad
audio OR a mislabel, so scores were not comparable across clips and could not be used as
a quality signal. They now can. Anything that reads `score` as quality — gate
calibration, keep-rate comparisons between engines — is only valid for v4-era rows
(campaign teacher-ab-v1 onward).

    (Recategorize was retired 2026-07-26 — the per-attribute dropdowns do that job
     precisely, and enforce the controlled register lexicon at the point of judgement.
     Existing "relabeled" rows keep their status; nothing new produces it.)

CSV is rewritten atomically on every save; every change is also appended to
ratings_history.csv (append-only audit log).
"""
import csv
import datetime
import fcntl
import importlib.util
import json
import os
import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


def _load_delivery_contract():
    """`matcha.delivery` — the single owner of the delivery vocabulary — loaded by PATH.

    This app runs in a `python:3.12-slim` container whose only dependencies are fastapi and
    uvicorn, and the deploy copies `audition/app/` alone, so the matcha package is not
    importable here and never will be. The vocabulary therefore travels WITH the deployment
    as an exported asset, exactly the way the device G2P's contraction tables ship as
    `g2p_contractions.json` rather than being transcribed by hand — D-C1 was a defect of
    omission, and a hand-synced copy is the same defect on a delay.

    Loaded by file path rather than as `matcha.delivery` because `import matcha` would drag
    in a package this container does not have. The module itself has ZERO imports (149 lines
    of constants and small pure functions), so loading it standalone is safe.

    RAISES if the asset is missing, and does not fall back to a literal. A fallback would be
    a second copy of a closed set — the precise thing this function exists to delete — and
    the app failing to start is a loud, immediate, obviously-correct failure, where wrong
    delivery labels are silent and reach the corpus.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "_contract" / "delivery.py",          # deployed: rsync'd by deploy.sh
        here.parent.parent / "matcha" / "delivery.py",  # dev: running from the repo
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("_delivery_contract", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError(
        "delivery contract not found — looked for:\n  "
        + "\n  ".join(str(p) for p in candidates)
        + "\nThis app does not carry its own copy of the delivery vocabulary on purpose "
          "(matcha/delivery.py owns it). Redeploy with scripts/deploy.sh audition."
    )


_delivery = _load_delivery_contract()
DELIVERY_UNKNOWN = _delivery.DELIVERY_UNKNOWN

# A BARE ATTRIBUTE READ HERE DEFEATS THE LOADER'S OWN CONTRACT (issue #16).
# `_load_delivery_contract()` goes to real trouble to turn a MISSING contract into an
# actionable RuntimeError naming `scripts/deploy.sh audition`, on the stated reasoning
# that failing to start must be "a loud, immediate, obviously-correct failure". A contract
# that is PRESENT but STALE — an app deployed without its `_contract/` asset, a partial
# rsync, a rollback of one and not the other — became newly reachable when the picker
# started needing `ACTIVE_DELIVERY_LANES`, and `_delivery.ACTIVE_DELIVERY_LANES` would
# have died with
#     AttributeError: module '_delivery_contract' has no attribute 'ACTIVE_DELIVERY_LANES'
# which names neither the file nor the fix. Same failure class, opposite failure quality.
#
# `DELIVERY_LANES` is deliberately NOT bound here any more (issue #17). After the picker
# moved to the active set it had exactly one reference — this module — kept alive only by
# a string match in tests/test_data_mirrors.py. A guard that asserts on a line with no
# readers proves nothing about behaviour: the picker could have been hardcoded and the
# assertion would still pass. The mirror test now pins the binding the picker ACTUALLY
# uses.
ACTIVE_DELIVERY_LANES = getattr(_delivery, "ACTIVE_DELIVERY_LANES", None)
if ACTIVE_DELIVERY_LANES is None:
    raise RuntimeError(
        "the delivery contract beside this app is STALE: it has no "
        "ACTIVE_DELIVERY_LANES, so the picker cannot know which lanes are assignable. "
        "It predates the Documentary retirement (RETIRED_LANES, 2026-08-10). The app "
        "refuses to start rather than offer a vocabulary it cannot verify. "
        "Fix: scripts/deploy.sh audition — which copies matcha/delivery.py to "
        "app/_contract/delivery.py, not just the app code.")

# --- Paths (overridable via env for local dev) --------------------------------
DATA_ROOT = Path(os.environ.get("AUDITION_DATA_ROOT", "/data/model-training/datasets")).resolve()
RATINGS_DIR = Path(os.environ.get(
    "AUDITION_RATINGS_DIR", str(DATA_ROOT / "sonora-expressive-registers"))).resolve()
RATINGS_CSV = RATINGS_DIR / "ratings.csv"
HISTORY_CSV = RATINGS_DIR / "ratings_history.csv"
REROLL_QUEUE = RATINGS_DIR / "reroll_queue.csv"
BOOK_QUEUE = DATA_ROOT / "book_queue.txt"   # book-prose conversion intake (host runner reads it)
STATIC_DIR = Path(__file__).parent / "static"

BOOK_QUEUE_HEADER = (
    "# book_queue.txt — book-prose conversion intake (the source router's inbox).\n"
    "#   One book URL per line; optional \"| note\". Lines starting with '#' are ignored;\n"
    "#   a leading '* ' marks a processed book (verdict appended after ' → ').\n"
)

CSV_FIELDS = ["campaign", "id", "engine", "register", "gender", "age", "accent",
              "delivery", "score", "note", "status", "link"]
GENDERS = ("Male", "Female", "Undefined")
# Provisional accent vocabulary (owner 2026-07-26). The casting-attribute-norms brief
# calls accent an "open list" with no exhaustive set attempted, so this is a starter
# taxonomy meant to be edited here in one place as auditing reveals what we actually
# need. Blank = not assessed, which is the default and must stay distinguishable from
# a deliberate "unclear".
# Sorted at definition so additions stay ordered without anyone remembering to —
# blank ("not assessed") stays first, and the two catch-alls stay pinned at the bottom
# where they belong in a picker.
_ACCENT_TAIL = ("Other", "Unclear")
ACCENTS = ("",) + tuple(sorted((
    "US - General", "US - Midwestern", "US - Southern", "US - Transatlantic",
    "US - African American",
    "British RP", "British other", "Scottish", "Irish", "Australian", "Indian",
))) + _ACCENT_TAIL

# Owner age taxonomy (casting-attribute-norms-brief.md): five bands, deliberately ours
# rather than inherited. Ordered youngest-first — alphabetical would be meaningless for
# an ordinal scale. Blank = not assessed; "Unclear" = assessed and indeterminate.
AGES = ("", "child", "teen", "adult", "middle-aged", "elderly", "Unclear")

# Delivery style (owner 2026-07-27) — orthogonal to `register`, which is the LINE's
# emotional colour. A clip can be a newscaster-styled anything. Added because
# "newscaster" had appeared in 17 free-text notes across 4 campaigns, making it more
# frequent than accent was when that column was requested.
#   Dialogue    — sounds like an actor speaking a character's line. Owner: "the most
#                 valuable for prosody", so it is the one worth counting.
#   Documentary — narrator or broadcast documentary; no need to split the two.
#   Neutral     — plain reading, also covers straight narration.
#   Newscaster  — the broadcast-reading prior. Observed ONLY on engines where direction
#                 cannot reach the model (vibevoice, the moss85 base).
#   Speech      — public address: a speaker performing TO AN AUDIENCE (rally, sermon,
#                 toast, courtroom), projected rather than conversational. Added
#                 2026-07-29 (owner) when the oratory registers (`grand_oratory`,
#                 `impassioned_oratory`) made clear this is a mode of address, not an
#                 emotional colour — a defiant line delivered to a crowd is Speech,
#                 the same line hissed at one person is Dialogue.
# IMPORTED, NOT RE-DECLARED — closed 2026-08-08 when this app moved into Sonora.
#
# This tuple used to be written out here in full, in a different order from
# `matcha.delivery.DELIVERY_LANES`, in a different repository. `matcha/delivery.py` is the
# single owner of the vocabulary by contract ("vocabulary changes are contract changes and
# require an owner call"), and a second copy of a closed set is the exact defect class the
# review kept finding — B-L5's MAX_REF_EXCURSION written out three times, D-L2's two
# disagreeing z-guards. "Which number means Newscaster" must not fork, because getting it
# wrong produces fluent, confidently mis-delivered audio rather than an error.
#
# The repo boundary is what made it un-importable, which is the concrete reason the move
# was worth doing rather than a tidiness argument.
#
# ONE entry is the APP's and stays here: `""` is DELIVERY_UNKNOWN — not assessed, the blank
# cell the corpus reads as the all-zero block, and a legitimate value rather than a gap
# (embodiment clips are delivery-blank BY RULE).
#
# `"Unclear"` was REMOVED 2026-08-10 (owner: "it really should just be our settled four").
# It read as a peer of the lanes and was not one: `delivery_index()` refuses any value
# outside `DELIVERY_LANES`, so a clip marked Unclear could never be built — it would abort
# the merge at `check_assignable`. It was a trap in a dropdown, not a rating state, and
# nothing ever carried it (0 of 1,802 rows). The accent column keeps its own "Unclear",
# where assessed-and-indeterminate IS a real distinct answer and must not be collapsed
# into blank; delivery has no such need because a lane it does not fit is simply unlabelled.
#
# `Documentary` is gone for the other reason: it is RETIRED (RETIRED_LANES), still inside
# DELIVERY_LANES because channel position is the wire format, but no longer assignable. The
# picker therefore offers ACTIVE_DELIVERY_LANES, never DELIVERY_LANES — offering a retired
# lane invites a label the build will refuse.
# SORTED FOR DISPLAY, and that is not a second fork — the contract owns the SET, this owns
# the PRESENTATION. `DELIVERY_LANES` is in wire order (channel position IS the wire format,
# so it can never be re-sorted there). This picker has shown them alphabetically across
# 1,802 rated rows, and silently swapping two adjacent entries in a dropdown someone uses
# at speed is how a mis-click becomes a mislabel — the kind of defect the score cannot
# detect. Sorting reproduces the existing order exactly and stays correct if a lane is ever
# added.
DELIVERY = (DELIVERY_UNKNOWN, *sorted(ACTIVE_DELIVERY_LANES))

# Marker stage_pool.py writes into the note column of a machine-folded row: a clip
# staged into the dataset on its GROUP's certification, which nobody listened to.
# Folded rows deliberately carry a BLANK score (they used to carry a fabricated 5,
# which is a verdict no ear gave), and a blank score would otherwise read as
# "unaudited" below and queue all of them — defeating the entire point of folding.
# They are treated like deferred rows: reachable, never in the pending queue.
FOLD_MARKER = "folded: staged unheard"

class _RatingsLock:
    """Serialize every read-modify-write of ratings.csv — against our own threads AND
    against the scripts (QC-M5).

    The threading lock was here from the start and covers uvicorn's workers. It cannot see
    another PROCESS, and several scripts write this file constantly — `scripts/tools/`'s
    `stage_pool`, `pick_audit_subset` and `sweep_dropped`, plus `scripts/stages/`'s
    `register_audition`. Those scripts protect themselves against the app, by two DIFFERENT
    mechanisms — the `scripts/tools/` three take `synth_common.ratings_transaction`, which
    re-checks the mtime inside its own lock and aborts if the app moved the file, while
    `register_audition` uses its own `append_guarded`: append, then read back and confirm the
    ids are on disk, re-appending whatever went missing. Verification is the guard there;
    checking mtime beforehand only narrows the window. Nothing protected the
    reverse direction: a script transaction committing inside this app's read -> replace
    window was silently overwritten. The window is milliseconds; the loss is a whole
    staging run's rows or a select pass's defers, with no error anywhere.

    So the app takes the SAME flock the scripts take. It shares the filesystem with them,
    which is the only thing both sides can agree on.

    ⚠ The lock path must stay byte-identical to `synth_common._exclusive`'s — `<path>.lock`
    — or the two sides take different locks and neither notices. It is reimplemented here
    rather than imported because this app runs in a `python:3.12-slim` container with only
    fastapi and uvicorn, where `synth_common` is not importable and never will be; that is
    the same constraint the delivery vocabulary works around, and the same answer applies:
    a test holds the two in step (`tests/test_ratings_transaction.py`).

    Sidecar, not the file itself: both sides replace the target by rename, and a lock held
    on the old inode would protect nothing once the rename lands.
    """

    def __init__(self, path):
        self._path = path
        self._thread_lock = threading.Lock()
        self._fh = None

    def __enter__(self):
        self._thread_lock.acquire()
        try:
            self._fh = open(f"{self._path}.lock", "a+", encoding="utf-8")
            fcntl.flock(self._fh, fcntl.LOCK_EX)
        except Exception:
            self._thread_lock.release()
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            raise
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
        finally:
            self._fh = None
            self._thread_lock.release()
        return False


_lock = _RatingsLock(RATINGS_CSV)  # serialize CSV read-modify-write

app = FastAPI(title="Dataset Auditions")


# --- CSV helpers --------------------------------------------------------------
def _read_rows():
    with open(RATINGS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _on_disk_fields():
    """The header ratings.csv actually has, with anything the app needs appended.

    QC-L4: this used to write exactly `CSV_FIELDS`, so ANY column a script added would be
    silently dropped the next time an auditor saved a rating — the app is the file's most
    frequent writer, so "next time" is minutes. Harmless today (the live header matches
    the 12 fields, verified), but it makes the app the one writer that does not honor the
    on-disk header. `register_audition._read_header_and_ids` has read it since it was
    written, precisely so a schema change does not need a coordinated deploy.

    The union, not the file's header alone: a column the app knows about must still appear
    on a sheet that predates it, or a rating written today would have nowhere to go.
    Order is preserved — the on-disk columns keep their positions and new ones append —
    because a reordered header is a diff nobody can read.
    """
    with open(RATINGS_CSV, newline="", encoding="utf-8") as f:
        on_disk = csv.DictReader(f).fieldnames or []
    return list(on_disk) + [c for c in CSV_FIELDS if c not in on_disk]


def _write_rows(rows):
    fields = _on_disk_fields()
    tmp = RATINGS_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    os.replace(tmp, RATINGS_CSV)  # atomic


def _append(path, header, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def _resolve_audio(link: str) -> Path:
    """Resolve a ratings link (relative to RATINGS_DIR) and confine to DATA_ROOT."""
    p = (RATINGS_DIR / link).resolve()
    if DATA_ROOT not in p.parents and p != DATA_ROOT:
        raise HTTPException(400, "path escapes data root")
    if not p.is_file():
        raise HTTPException(404, "audio not found")
    return p


def _gender_from_id(clip_id: str) -> str:
    """Presumed gender from the voice-label suffix in the id (…intimateM_s1234 -> Male)."""
    m = re.search(r"_([A-Za-z0-9]+)_s\d+$", clip_id or "")
    if m:
        last = m.group(1)[-1]
        if last == "M":
            return "Male"
        if last == "F":
            return "Female"
    return "Undefined"


# --- API ----------------------------------------------------------------------
# Director-direction sidecar (owner ask 2026-07-22): synth campaigns write
# *_manifest.jsonl next to the wavs with the full direction per clip id; show
# it on the card so contradictions (casting vs render, register vs delivery)
# can be spotted and noted during audition. Cached per directory by mtime;
# rows whose dir has no manifests (real-audio audit sets) get None.
_dir_cache: dict = {}


def _direction_for(row):
    link = row.get("link") or ""
    try:
        wav_dir = (RATINGS_DIR / link).resolve().parent
        if not str(wav_dir).startswith(str(DATA_ROOT)):
            return None
        manifests = sorted(wav_dir.glob("*_manifest.jsonl"))
        if not manifests:
            return None
        stamp = tuple((str(m), m.stat().st_mtime) for m in manifests)
        cached = _dir_cache.get(wav_dir)
        if not cached or cached[0] != stamp:
            byid = {}
            for m in manifests:
                with open(m, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        d = rec.get("direction") or {}
                        byid[rec.get("id")] = {
                            "design": d.get("design"),
                            "instruct": d.get("instruct"),
                            "render_text": d.get("render_text"),
                            "intended": rec.get("intended"),
                            "text": rec.get("text"),
                            "ref": rec.get("ref"),
                        }
            _dir_cache[wav_dir] = (stamp, byid)
        found = _dir_cache[wav_dir][1].get(row.get("id"))
        return _annotate_relay(found, row.get("engine", "")) if found else None
    except Exception:
        return None


# Which direction fields each renderer ACTUALLY passes to the TTS model.
# Grounded in the renderer call sites, not in what the director emitted:
#   synth_vibevoice.py  -> model gets "Speaker 0: {text}" + a reference wav.
#                          design is read ONLY by ref_select.select_reference()
#                          for a gender regex + age band; instruct is unused.
#   synth_qwen.py       -> instruct ONLY. `generate_voice_design()` has no
#                          `voice_description` parameter (verified in qwen-tts
#                          0.0.5/0.1.0/0.1.1) — it lands in **kwargs and is
#                          silently dropped, so the `except TypeError` fallback
#                          is dead code and design never reaches the model.
#   synth_moss85.py     -> instruct -> instruction (+ quality). design is unused;
#                          book_ingest pre-merges design into instruct, but the
#                          quote-pilot/director-bench builders did not. NB the
#                          8.5B flagship is an un-SFT'd base model whose card
#                          does not even list `instruction` as an input — those
#                          fields are untrained prompt text there.
#   synth_dia.py        -> render_text only; design/instruct never read.
# "casting" = shapes reference selection but is not spoken direction.
RELAY = {
    "vibevoice": {"design": "casting", "instruct": "no", "render_text": "no"},
    "qwen":      {"design": "no", "instruct": "yes", "render_text": "no"},
    # moss_vg = MOSS-VoiceGenerator, the instruction-TUNED sibling; it is the
    # engine that actually reads `instruction`. moss85 is the un-SFT'd flagship
    # kept only for cloning work. Distinct keys — neither name prefixes the other.
    "moss_vg":   {"design": "no", "instruct": "yes", "render_text": "no"},
    "moss85":    {"design": "no", "instruct": "yes", "render_text": "no"},
    "dia":       {"design": "no", "instruct": "no", "render_text": "yes"},
    # Revisit-list engines (interface studies 2026-07-28). None takes prose; each is
    # driven by parameters, a closed tag set, or reference selection.
    #   chatterbox -> exaggeration + cfg_weight (a PAIR) & 34 inline tags; voice = ref clip
    #   zonos      -> 8-float emotion SHAPE + pitch_std + speaking_rate; voice = ref clip
    #   orpheus    -> a voice from a closed 8; 8 inline tags. `-ft` only.
    #   longcat    -> nothing at all; reference selection is the whole decision
    "chatterbox": {"design": "casting", "instruct": "no", "render_text": "yes"},
    "zonos":      {"design": "casting", "instruct": "no", "render_text": "no"},
    "orpheus":    {"design": "no", "instruct": "no", "render_text": "yes"},
    "longcat":    {"design": "casting", "instruct": "no", "render_text": "yes"},
}


def _annotate_relay(direction: dict, engine: str) -> dict:
    """Tag each direction field with whether the engine actually received it.

    The manifest records the director's full output; renderers consume only a
    subset. Showing the whole thing undifferentiated invites scoring a model
    down for direction it was never given (owner finding 2026-07-25).
    """
    eng = (engine or "").strip().lower()
    for key, val in RELAY.items():
        if eng.startswith(key):
            return {**direction, "relay": val, "relay_engine": key}
    return {**direction, "relay": None, "relay_engine": eng}


@app.get("/api/clips")
def clips(filter: str = "todo", register: str | None = None,
          campaign: str | None = None, delivery: str | None = None,
          page: int = 1, page_size: int = 20):
    """List clips, paginated.

    filter:
        todo / unaudited -> not yet rated (status unaudited or empty score)
        rerate           -> every already-rated clip, for top-to-bottom recalibration
                            (the card shows its existing score so nothing is lost)
        deferred         -> registered but PARKED out of the audit queue by the
                            risk-stratified sampling protocol (owner 2026-07-30:
                            batches are now too large to audition exhaustively).
                            pick_audit_subset.py selects the high-risk subset and
                            defers the rest; a failed group promotes its deferred
                            clips back to unaudited. Deferred clips appear only
                            here and under `all`, never in todo or rerate.
        folded           -> staged into the dataset unheard, on the strength of
                            their GROUP's certification (stage_pool.py). They
                            carry a blank score because no ear ever gave one —
                            so they would otherwise flood todo, which is the
                            opposite of what folding is for. Like deferred:
                            here and under `all` only.
        all              -> everything
    """
    rows = _read_rows()
    matched = []
    for r in rows:
        score, status = r.get("score", ""), r.get("status", "")
        is_deferred = status == "deferred"
        is_folded = FOLD_MARKER in (r.get("note") or "")
        is_unaudited = ((status == "unaudited" or score == "")
                        and not is_deferred and not is_folded)
        is_rated = not is_unaudited and not is_deferred and not is_folded
        if filter in ("todo", "unaudited") and not is_unaudited:
            continue
        if filter == "rerate" and not is_rated:
            continue
        if filter == "deferred" and not is_deferred:
            continue
        if filter == "folded" and not is_folded:
            continue
        if register and r.get("register") != register:
            continue
        # campaign filter: a restart leaves hundreds of superseded clips
        # queued, and they drown the campaign actually being audited.
        if campaign and r.get("campaign") != campaign:
            continue
        # delivery filter: added 2026-08-10 to work a single lane end-to-end. The
        # case in hand is Documentary, which is being retired — 154 clips, each
        # reassigned on the ear, and all 154 landed on Neutral (Newscaster is 87
        # before and after). ⚠ Corrected 2026-08-11: this said "82 clips split
        # bimodally toward Neutral and Newscaster" — 82 is the lane's share of the v6
        # append set, not the lane, and the sweep was not a split.
        # Without this you cannot reach a lane except by remembering which
        # campaigns fed it. `_none` matches delivery-blank clips, which are otherwise
        # unreachable: blank is a legitimate value (embodiment clips are blank BY
        # RULE), so "has no delivery" is a real query, not an empty filter.
        if delivery and (r.get("delivery") or "_none") != delivery:
            continue
        matched.append({**r, "_rated": is_rated, "_unaudited": is_unaudited,
                        "_folded": is_folded,
                        "gender": r.get("gender") or _gender_from_id(r.get("id", "")),
                        "direction": _direction_for(r)})
    registers = sorted({r.get("register", "") for r in rows if r.get("register")})
    campaigns = sorted({r.get("campaign", "") for r in rows if r.get("campaign")})

    page_size = max(1, min(page_size, 200))
    pages = max(1, (len(matched) + page_size - 1) // page_size)
    page = max(1, min(page, pages))
    start = (page - 1) * page_size
    return {
        "clips": matched[start:start + page_size],
        "registers": registers,
        "campaigns": campaigns,
        "accents": list(ACCENTS),
        "ages": list(AGES),
        "deliveries": list(DELIVERY),
        "lexicon": _lexicon(),
        "page": page,
        "pages": pages,
        "page_size": page_size,
        "matched": len(matched),   # clips in this filter (across all pages)
        "total": len(rows),        # whole dataset
    }


@app.get("/api/stats")
def stats():
    rows = _read_rows()
    from collections import Counter
    by_status = Counter(r.get("status", "") for r in rows)
    scored = Counter(r.get("score", "") for r in rows if r.get("score", "") not in ("", None))
    return {"total": len(rows), "by_status": dict(by_status), "by_score": dict(scored)}


@app.get("/audio")
def audio(path: str):
    return FileResponse(_resolve_audio(path), media_type="audio/wav")


class Rating(BaseModel):
    id: str
    action: str            # keep | drop | retake
    score: int | None = None   # 1-5 for keep
    note: str = ""


def _apply(row: dict, r: Rating):
    a = r.action
    if a == "keep":
        if r.score not in range(1, 6):
            raise HTTPException(422, "keep needs score 1-5")
        row["score"], row["status"] = str(r.score), "keep"
    elif a == "drop":
        row["score"], row["status"] = "x", "dropped"
    elif a == "retake":
        row["score"], row["status"] = "0", "reroll"
    else:
        raise HTTPException(422, f"unknown action {a}")
    row["note"] = r.note
    return row


@app.post("/api/rate")
def rate(r: Rating):
    with _lock:
        rows = _read_rows()
        idx = next((i for i, row in enumerate(rows) if row.get("id") == r.id), None)
        if idx is None:
            raise HTTPException(404, f"unknown id {r.id}")
        row = rows[idx]
        _apply(row, r)
        _write_rows(rows)
        today = datetime.date.today().isoformat()
        _append(HISTORY_CSV, ["date", "campaign", "id", "score", "note"],
                [today, row.get("campaign", ""), row["id"], row["score"], row.get("note", "")])
        if r.action == "retake":
            _append(REROLL_QUEUE,
                    ["queued", "campaign", "id", "engine", "register", "link", "tweak"],
                    [datetime.datetime.now().isoformat(timespec="seconds"),
                     row.get("campaign", ""), row["id"], row.get("engine", ""),
                     row.get("register", ""), row.get("link", ""), r.note])
        return {"ok": True, "row": row}


class GenderUpdate(BaseModel):
    id: str
    gender: str


class AccentUpdate(BaseModel):
    id: str
    accent: str


class AgeUpdate(BaseModel):
    id: str
    age: str


class DeliveryUpdate(BaseModel):
    id: str
    delivery: str


class NoteUpdate(BaseModel):
    id: str
    note: str


@app.post("/api/note")
def set_note(n: NoteUpdate):
    """Persist a comment independent of any rating action (owner data-loss
    2026-07-22: notes typed AFTER the score click were silently dropped —
    the note only rode along with /api/rate)."""
    with _lock:
        rows = _read_rows()
        idx = next((i for i, row in enumerate(rows) if row.get("id") == n.id), None)
        if idx is None:
            raise HTTPException(404, f"unknown id {n.id}")
        rows[idx]["note"] = n.note
        _write_rows(rows)
        return {"ok": True, "id": n.id, "note": n.note}


@app.post("/api/gender")
def set_gender(g: GenderUpdate):
    """Persist a corrected speaker gender (independent of the rating action)."""
    if g.gender not in GENDERS:
        raise HTTPException(422, f"gender must be one of {GENDERS}")
    with _lock:
        rows = _read_rows()
        idx = next((i for i, row in enumerate(rows) if row.get("id") == g.id), None)
        if idx is None:
            raise HTTPException(404, f"unknown id {g.id}")
        rows[idx]["gender"] = g.gender
        _write_rows(rows)
        return {"ok": True, "id": g.id, "gender": g.gender}


# --- Book-prose conversion queue (book_queue.txt) -----------------------------
# The audition container has no docker/GPU, so it never runs synthesis — it only
# manages the queue file. A separate host-side runner processes pending lines and
# marks them '* ' when done. Status here is read straight from that '* ' marker.
def _parse_book_line(raw: str):
    """One queue line -> entry dict, or None for a blank/comment line."""
    s = raw.rstrip("\n")
    t = s.strip()
    if not t or t.startswith("#"):
        return None
    converted = t.startswith("* ")
    body = t[2:].strip() if converted else t
    verdict = ""
    if "→" in body:                      # ' → <verdict> (<date>)'
        body, verdict = body.split("→", 1)
        verdict = verdict.strip()
    url, _, note = body.partition("|")
    return {"url": url.strip(), "note": note.strip(),
            "status": "converted" if converted else "pending",
            "verdict": verdict, "raw": s}


def _read_books():
    if not BOOK_QUEUE.exists():
        return []
    with open(BOOK_QUEUE, encoding="utf-8") as f:
        return [e for e in (_parse_book_line(l) for l in f) if e]


def _write_book_text(text: str):
    tmp = BOOK_QUEUE.with_suffix(".txt.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, BOOK_QUEUE)               # atomic


@app.get("/api/books")
def list_books():
    books = _read_books()
    return {"books": books,
            "pending": sum(1 for e in books if e["status"] == "pending"),
            "converted": sum(1 for e in books if e["status"] == "converted")}


class BookAdd(BaseModel):
    url: str
    note: str = ""


@app.post("/api/books")
def add_book(b: BookAdd):
    url = b.url.strip()
    if not re.match(r"^https?://\S+$", url):
        raise HTTPException(422, "url must be a valid http(s) link")
    note = b.note.strip()
    with _lock:
        if any(e["url"] == url for e in _read_books()):
            raise HTTPException(409, "already in the queue")
        data = BOOK_QUEUE.read_text(encoding="utf-8") if BOOK_QUEUE.exists() else BOOK_QUEUE_HEADER
        if data and not data.endswith("\n"):
            data += "\n"
        data += (f"{url} | {note}" if note else url) + "\n"
        _write_book_text(data)
    return {"ok": True, "url": url, "note": note, "status": "pending"}


@app.delete("/api/books")
def delete_book(url: str):
    """Remove a *pending* book from the queue (converted lines are kept as record)."""
    with _lock:
        if not BOOK_QUEUE.exists():
            raise HTTPException(404, "queue is empty")
        removed = False
        out = []
        for raw in BOOK_QUEUE.read_text(encoding="utf-8").splitlines(keepends=True):
            e = _parse_book_line(raw)
            if e and e["url"] == url:
                if e["status"] == "converted":
                    raise HTTPException(409, "already converted — kept as record")
                removed = True
                continue
            out.append(raw)
        if not removed:
            raise HTTPException(404, "url not in queue")
        _write_book_text("".join(out))
    return {"ok": True, "removed": url}


@app.post("/api/accent")
def set_accent(a: AccentUpdate):
    """Persist an audited accent, independent of the rating action.

    Accent is a CASTING attribute, not direction — no engine we run accepts it as an
    instruction — so this records what was actually heard, which is the only way to
    build the measurable norms the casting brief asks for. Blank means "not assessed";
    "Unclear" means assessed and indeterminate. Do not collapse the two.
    """
    if a.accent not in ACCENTS:
        raise HTTPException(422, f"accent must be one of {ACCENTS}")
    with _lock:
        rows = _read_rows()
        idx = next((i for i, row in enumerate(rows) if row.get("id") == a.id), None)
        if idx is None:
            raise HTTPException(404, f"unknown id {a.id}")
        rows[idx]["accent"] = a.accent
        _write_rows(rows)
        return {"ok": True, "id": a.id, "accent": a.accent}


def _lexicon(min_keeps: int = 3):
    """Controlled register lexicon, derived live from the ratings SSOT.

    Retiring Recategorize (2026-07-26) moved register correction to a dropdown, which
    is the first time the audit surface can ENFORCE the lexicon rather than accept free
    text. Pre-enforcement, ratings.csv had drifted to 138 distinct labels.

    Derived here rather than read from scripts/assets/register_lexicon.json because
    that file lives outside the container's mounts (/app and /data only) — and because
    the promotion rule is identical, so both stay consistent by construction: a label
    is in the lexicon once it has `min_keeps` certified keeps. Recomputed per request
    so a label promoted mid-session becomes selectable without a restart.
    """
    from collections import Counter
    try:
        counts = Counter(r.get("register", "") for r in _read_rows()
                         if r.get("status") == "keep" and r.get("register"))
    except Exception:
        return []
    return sorted(k for k, v in counts.items() if v >= min_keeps)


class RegisterUpdate(BaseModel):
    id: str
    register: str


@app.post("/api/age")
def set_age(a: AgeUpdate):
    """Persist an audited age band — the owner's most frequent freehand note.

    Five bands from casting-attribute-norms-brief.md, ours by design rather than
    inherited. Like accent this records what was actually HEARD, which is the only
    route to measurable casting norms. Note ref_select's F0 bands use "young" where
    this taxonomy says "adult" — the brief's five names are canonical here.
    """
    if a.age not in AGES:
        raise HTTPException(422, f"age must be one of {AGES}")
    with _lock:
        rows = _read_rows()
        idx = next((i for i, row in enumerate(rows) if row.get("id") == a.id), None)
        if idx is None:
            raise HTTPException(404, f"unknown id {a.id}")
        rows[idx]["age"] = a.age
        _write_rows(rows)
        return {"ok": True, "id": a.id, "age": a.age}


@app.post("/api/delivery")
def set_delivery(d: DeliveryUpdate):
    """Persist the audited delivery style — orthogonal to register and to accent."""
    if d.delivery not in DELIVERY:
        raise HTTPException(422, f"delivery must be one of {DELIVERY}")
    with _lock:
        rows = _read_rows()
        idx = next((i for i, row in enumerate(rows) if row.get("id") == d.id), None)
        if idx is None:
            raise HTTPException(404, f"unknown id {d.id}")
        rows[idx]["delivery"] = d.delivery
        _write_rows(rows)
        return {"ok": True, "id": d.id, "delivery": d.delivery}


@app.post("/api/register")
def set_register(u: RegisterUpdate):
    """Persist a corrected register. Replaces the Recategorize action."""
    lex = _lexicon()
    if lex and u.register not in lex:
        raise HTTPException(422, f"register must be in the controlled lexicon ({len(lex)} labels)")
    with _lock:
        rows = _read_rows()
        idx = next((i for i, row in enumerate(rows) if row.get("id") == u.id), None)
        if idx is None:
            raise HTTPException(404, f"unknown id {u.id}")
        was = rows[idx].get("register", "")
        rows[idx]["register"] = u.register
        if was and was != u.register:
            note = rows[idx].get("note", "")
            rows[idx]["note"] = f"[was: {was}] {note}".strip()
        _write_rows(rows)
        return {"ok": True, "id": u.id, "register": u.register, "was": was}


# --- Anchor exemplars ---------------------------------------------------------
#
# WHY THIS EXISTS (owner, 2026-08-08). After re-listening to Qwen and VibeVoice at equal
# loudness the owner wrote: *"Qwen relays human-like prosody in those cases where it scored
# a 5 that makes me rethink 5's given out to others."* That measures. Over the audited
# corpus **799 of 1,219 scored keeps are 5s — 66%**; on identical text, 46 of 62 controlled
# groups have three or more DIFFERENT engines all at 5; and `librivox` REAL HUMAN audio
# means exactly 5.00. The top of the scale is "indistinguishable from a human read" and
# most of the corpus is sitting on it, which is why mean score ranks `chatterbox` above
# `qwen`.
#
# A saturated scale is not fixed by adding scale points. It is fixed by giving the ear a
# fixed reference, which is standard MOS practice and the one thing this app never had:
# with no anchor, "5" drifts per session, per engine, and per how good the last clip was.
#
# THE ANCHORS ARE THE OWNER'S EAR, NOT A COMPUTED CHOICE. Nothing here picks one from a
# measure. `POST /api/anchor` sets an anchor from the clip being auditioned — in situ,
# which is the only moment the judgement is actually available. Two seeds ship because
# they need no guessing: the definitional 5 (a real human recording, since that is what
# the top of the scale MEANS) and the exemplar the owner named by hand.
#
# A SIDECAR, NOT A COLUMN — and that is a scar, not a preference. The Qwen/VibeVoice A/B
# parked each clip's prior score in `note`, which this app OVERWRITES when the owner types;
# 17 of 33 were lost, and the comparison survived only because a transaction backup existed.
# Anchors are app-owned state about the *scale*, not about a clip, so they live in their own
# file and no clip row is touched.
ANCHORS_JSON = RATINGS_DIR / "anchors.json"
ANCHOR_SCORES = ("1", "2", "3", "4", "5")


def _read_anchors() -> dict:
    try:
        with open(ANCHORS_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data.get("anchors") or {}


def _write_anchors(anchors: dict):
    """Same atomicity as the ratings CSV: wholly the new file or wholly the old."""
    tmp = ANCHORS_JSON.with_suffix(".json.tmp")
    payload = {
        "_comment": ("Anchor exemplars for the 1-5 audition scale — what each number "
                     "SOUNDS like. Written by the app (POST /api/anchor) from the clip "
                     "being auditioned. Safe to hand-edit; `id` must exist in "
                     "ratings.csv. Deleting an entry unsets that anchor."),
        "anchors": anchors,
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, ANCHORS_JSON)


@app.get("/api/anchors")
def anchors():
    """The reference set, joined to live ratings so a stale entry cannot play silently.

    `missing` is reported rather than dropped: an anchor whose clip has vanished from
    ratings.csv is a broken reference, and a reference bar that quietly shows four of five
    anchors is exactly the kind of failure-by-omission that would go unnoticed for weeks.
    """
    stored = _read_anchors()
    if not stored:
        return {"anchors": {}, "missing": [], "scores": list(ANCHOR_SCORES)}
    by_id = {r.get("id"): r for r in _read_rows()}
    out, missing = {}, []
    for score, a in stored.items():
        row = by_id.get(a.get("id"))
        if row is None:
            missing.append({"score": score, "id": a.get("id")})
            continue
        out[score] = {**a, "link": row.get("link", ""), "engine": row.get("engine", ""),
                      "register": row.get("register", ""),
                      # The anchor's CURRENT score, which is not necessarily the score it
                      # anchors: re-rating a clip does not silently re-point the anchor.
                      "current_score": row.get("score", ""),
                      "drifted": row.get("score", "") != score}
    return {"anchors": out, "missing": missing, "scores": list(ANCHOR_SCORES)}


class AnchorUpdate(BaseModel):
    score: str                 # "1".."5" — the scale point this clip exemplifies
    id: str | None = None      # omit to CLEAR the anchor for that score
    why: str = ""              # the owner's one line on what makes it that number


@app.post("/api/anchor")
def set_anchor(a: AnchorUpdate):
    """Make the clip being auditioned the reference for a scale point, or clear one."""
    if a.score not in ANCHOR_SCORES:
        raise HTTPException(422, f"score must be one of {ANCHOR_SCORES}")
    with _lock:
        anchors = _read_anchors()
        if a.id is None:
            anchors.pop(a.score, None)
            _write_anchors(anchors)
            return {"ok": True, "score": a.score, "cleared": True}
        row = next((r for r in _read_rows() if r.get("id") == a.id), None)
        if row is None:
            raise HTTPException(404, f"unknown id {a.id}")
        anchors[a.score] = {
            "id": a.id,
            "why": a.why,
            "set_at": datetime.datetime.now().isoformat(timespec="seconds"),
            # Recorded at the moment of anchoring so a later re-rating is VISIBLE as drift
            # rather than rewriting history.
            "score_when_set": row.get("score", ""),
            "engine_when_set": row.get("engine", ""),
        }
        _write_anchors(anchors)
        return {"ok": True, "score": a.score, "id": a.id}


# NB: this catch-all mount must stay LAST. Routes registered after it are shadowed —
# Starlette matches in order, and StaticFiles only permits GET/HEAD, so a POST added
# below returns 405 rather than reaching its handler (hit 2026-07-26).
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
