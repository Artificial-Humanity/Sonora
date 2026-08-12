# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy", "torch", "torchaudio", "soundfile", "librosa>=0.10", "numba>=0.60",
#   "faster-whisper",
# ]
# ///
"""librivox_align — cut a REAL-AUDIO book into clips whose text is CANONICAL.

Stage B of the force-align lane. `librivox_fetch.py` puts a book's section MP3s and its
Gutenberg source text on disk; this turns them into audition-ready clips.

HOW THIS OBEYS [[force-align-first-dataprep]]
---------------------------------------------
The rule is: force-align canonical text to audio; ASR is fallback-only. This aligner
uses ASR, and it is worth being exact about where, because the distinction is the whole
design:

  * The TEXT WE EMIT is always the Gutenberg text. Not one word of ASR output reaches a
    clip's transcript. If ASR mishears "Bill" as "Bell", the clip still says "Bill".
  * ASR is used ONLY as an INDEX — to answer "roughly where in this 40-minute chapter
    are we?" — because a CTC alignment trellis over a whole chapter is not tractable
    (an hour of audio against a chapter of text is billions of cells).
  * Precise word boundaries come from `torchaudio.functional.forced_align`, a CTC forced
    aligner, run on SHORT WINDOWS between ASR anchors, against the CANONICAL tokens.

So ASR chooses the windows; canonical text chooses the words; CTC chooses the boundaries.
An ASR error costs at worst a slightly misplaced window edge, never a wrong transcript.

WHAT COMES OUT
--------------
Clips obeying the standing corpus constraints:
  * complete utterances only — cut on sentence boundaries, never mid-sentence
    ([[completeness-over-length]]: "short-but-whole is ratable, incomplete is not")
  * >= MIN_SECONDS of speech ([[min-clip-length-4s]])
  * <= MAX_SECONDS (22.0, matching derive_vat_corpus)
  * a per-clip alignment score; anything below --min-score is dropped rather than
    shipped, because a confidently-wrong alignment is worse than a missing clip.

Written as <out>/audio/<id>.wav plus a `librivox_manifest.jsonl` in the campaign shape,
so `qc_gate.py` and `register_audition.py` accept the output unchanged.

Runs in a container (see librivox_align.sh) — the host has no torch and no ffmpeg.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import re
import sys
import unicodedata

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import synth_common  # noqa: E402
from synth_common import (  # noqa: E402
    is_complete_utterance,
    split_sentences as _shared_split_sentences,
    write_wav_atomic,
)

SAMPLE_RATE = 16000          # the aligner's rate; clips are written at OUT_SR
OUT_SR = 24000               # the corpus interchange rate (model-decisions.md)
MIN_SECONDS = 4.0            # wall-clock floor
MIN_SPEECH_SECONDS = 4.0     # [[min-clip-length-4s]] — the owner floor is SPEECH,
                             # not duration: a 4 s clip holding 2.5 s of speech and
                             # 1.5 s of room tone does not clear it.
MAX_SECONDS = 22.0           # matches derive_vat_corpus.MAX_SECONDS
PAD_SECONDS = 0.12           # breath room around a cut
EDGE_GUARD = 0.02            # gap held against the neighbouring sentence
ANCHOR_WINDOW_S = 30.0       # max audio per CTC call

_FA_CACHE = None             # (model, labels, device) — see refine()

# SENT_END is retired -- see split_sentences (A-H5). Kept out of the module rather than
# left lying around: a naive splitter sitting next to the real one is an invitation.


# ------------------------------------------------------------------ text
def norm_word(w: str) -> str:
    w = unicodedata.normalize("NFKD", w).lower()
    return re.sub(r"[^a-z0-9]", "", w)


def split_sentences(text: str) -> list[str]:
    """A-H5: pysbd, shared with book_ingest via synth_common.

    This was `SENT_END`, a regex splitting on any [.!?] followed by whitespace. It has no
    idea what an abbreviation is, so "Mr. Smith went home." became TWO clips: "Mr." and
    "Smith went home." — a fragment, and a sentence starting mid-utterance. Both were
    then aligned, cut and shipped, and the v4 rating vocabulary (vocals and prosody only)
    cannot see the defect. The book lane has used pysbd since it was written; the
    real-audio lane, which produces the clips we most want to be whole, used neither that
    nor is_complete_utterance.
    """
    return _shared_split_sentences(re.sub(r"\s+", " ", text).strip())


HEADING_RE = re.compile(r"(?im)^[ \t]*(chapter|section|book|part)\s+([IVXLC]+|\d+)\b.*$")
MIN_CHAPTER_CHARS = 1200      # below this, consecutive "headings" are a contents list
MAX_HEADING_CHARS = 70        # a heading is a short line, not a sentence that starts with one


def find_headings(text: str) -> list[re.Match]:
    """Chapter headings that are plausibly real (A-H1).

    The old pattern was `^\\s*(chapter|section)\\s+([IVXLC]+|\\d+)\\b.*$` with no further
    validation, and it matched three things that are not headings:

      1. **Table-of-contents entries.** A ToC lists every chapter, so `heads` came back
         with roughly twice as many entries as the book has chapters and the first half
         were all inside the ToC. Slicing on those returns a few lines of contents —
         which contains almost none of the words the reader says.
      2. **Hard-wrapped prose.** Gutenberg plaintext wraps at ~70 columns, so
         "...as I explained in the last\\nchapter I was unwilling..." puts `chapter I` at
         the start of a line and matches. The heading then lands in the middle of a
         paragraph.
      3. **Front-matter noise** — "BOOK I", "PART II" — interleaved with real chapters.

    Two cheap filters kill all three. A heading occupies a SHORT line (a wrapped prose
    line is near the wrap width and keeps going), and it is followed by a substantial
    body (a ToC entry is followed by the next entry a line later). Note `^[ \\t]*` rather
    than `^\\s*`: `\\s` matches newlines, which let the pattern skip blank lines and start
    matching mid-paragraph.
    """
    out = []
    for m in HEADING_RE.finditer(text):
        line = m.group(0).strip()
        if len(line) > MAX_HEADING_CHARS:
            continue                       # (2) a prose line that happens to begin "chapter I"
        tail = line[m.end(2) - m.start():].strip(" .:—-–")
        if tail[:1].islower():
            continue                       # (2) "chapter I explained the matter to him"
        out.append(m)
    # (1)/(3): drop any heading whose body is too short to be a chapter. Walk from the end
    # so a ToC entry is measured against the NEXT entry, which is one line away.
    kept: list[re.Match] = []
    for i, m in enumerate(out):
        nxt = out[i + 1].start() if i + 1 < len(out) else len(text)
        if nxt - m.start() >= MIN_CHAPTER_CHARS:
            kept.append(m)
    return kept


def _proportional(text: str, index: int, n_sections: int,
                  durations: dict[int, float] | None, pad_factor: float) -> str:
    """SPLIT BY DURATION, NOT BY SECTION INDEX.

    An even index split assumes every section is the same length. That holds for a novel —
    it is why *Uneasy Money* aligned at 94% — and fails completely for a collection.
    Dickens's *Speeches* opens with a 73-minute editorial introduction (sections 1-2) and
    then has speeches as short as 2.6 minutes; an even split put section 3's window at
    3.3% of the text when the words were near 10.8%, so the anchor pass located **0%** of
    the heard words and every clip was dropped.

    Cumulative playtime fraction fixes it, resting on the assumption the even split was
    already making implicitly and more weakly: that reading pace is roughly constant
    within one reader. Where durations are missing we fall back to the old even split.
    """
    total = sum(durations.values()) if durations else 0.0
    if durations and total > 0 and index in durations:
        before = sum(d for i, d in durations.items() if i < index)
        f0, f1 = before / total, (before + durations[index]) / total
    else:
        per = 1.0 / max(n_sections, 1)
        f0, f1 = (index - 1) * per, index * per

    span = f1 - f0
    # Pad relative to this section's own span, with a floor: a 2.6-minute section inside
    # an 11-hour book has a span so small that a purely relative pad cannot absorb any
    # drift in reading pace.
    pad = max(span * pad_factor, 0.01)
    a = max(0, int((f0 - pad) * len(text)))
    b = min(len(text), int((f1 + pad) * len(text)))
    return text[a:b]


def section_durations(book: dict) -> dict[int, float]:
    """Per-section playtime in seconds, COMPLETE or empty. A-M1.

    The old version had two silent failure modes, and they compounded:

    1. `float(sec.get("playtime") or 0)` inside a `try/except: continue`. LibriVox is not
       consistent about this field — most projects give seconds as a string, some give
       "hh:mm:ss" — so a colon-formatted section was dropped from the map entirely. The
       cumulative fraction in `_proportional` was then computed over a SUBSET: surviving
       sections got windows sized against a smaller total, the dropped ones fell back to
       the even split, and the windows no longer tiled. Nothing detects that; the coverage
       gate sees a bad slice as a bad alignment.
    2. `if not all(durations.values()): durations = {}` — one zero playtime anywhere threw
       away every duration in the book and reverted ALL sections to the even split. That
       is precisely the split that located **0%** of the heard words in Dickens's
       *Speeches* and dropped every clip, and it happened without a word of output.

    So: parse both formats, fill what is missing from the residual of `totaltime` when
    that is possible, and when it is not, fall back loudly rather than silently. The
    return is all-or-nothing on purpose — a partial map is the case that produces windows
    which do not tile, and `_proportional` cannot tell a partial map from a complete one.
    """
    sections = book.get("sections") or []
    parsed: dict[int, float | None] = {}
    for sec in sections:
        try:
            index = int(sec["index"])
        except (TypeError, ValueError, KeyError):
            continue
        parsed[index] = synth_common.parse_playtime(sec.get("playtime"))

    if not parsed:
        return {}
    missing = sorted(i for i, d in parsed.items() if not d)
    if not missing:
        return {i: d for i, d in parsed.items()}

    # Fill from the residual: `totaltime` is the whole work, so what the known sections do
    # not account for belongs to the unknown ones. Spread evenly among them — which is the
    # even split's own assumption, but applied ONLY where we have no better information
    # instead of thrown over the entire book.
    total = synth_common.parse_playtime(book.get("totaltime"))
    known = sum(d for d in parsed.values() if d)
    residual = (total - known) if total else 0.0
    if total and residual > 0:
        share = residual / len(missing)
        print(f"  !! {len(missing)} section(s) have no usable playtime "
              f"({', '.join(str(i) for i in missing[:8])}"
              f"{'…' if len(missing) > 8 else ''}) — imputing {share:.0f}s each from "
              f"totaltime ({total:.0f}s residual {residual:.0f}s)", file=sys.stderr)
        return {i: (d if d else share) for i, d in parsed.items()}

    print(f"  !! {len(missing)} of {len(parsed)} section(s) have no usable playtime and "
          "totaltime cannot fill the gap — falling back to an EVEN split for the whole "
          "book. That is the split that located 0% of the heard words in Dickens's "
          "Speeches; check coverage on this run.", file=sys.stderr)
    return {}


def chapter_slices(text: str, index: int, n_sections: int,
                   durations: dict[int, float] | None = None) -> list[tuple[str, str]]:
    """Ordered candidate slices as (strategy, text) — try them until one anchors.

    A-H1's second half. There used to be ONE slice: headings if they looked usable,
    otherwise proportional. When that slice was wrong the coverage gate rejected the
    section and moved on, so a book whose headings misled the slicer lost **every** clip
    while each individual decision looked defensible in the log.

    Retrying is nearly free, which is what makes this worth doing: the ASR pass depends
    only on the audio, so a second candidate costs one more `difflib` run over a few
    thousand words and no GPU work at all.

    Ordering is most-specific-first. Headings, when they are real, beat any proportional
    guess; a wide proportional window beats a narrow one only if the narrow one failed,
    because a wider window admits more wrong-chapter text for the matcher to trip on.
    """
    out: list[tuple[str, str]] = []

    heads = find_headings(text)
    # Headings are only usable if they map ~1:1 onto audio sections. `len(heads) >=
    # n_sections` was the old test, and it is exactly wrong for a single-file
    # multi-chapter book: 40 chapters in 3 audio files satisfies it, then hands section 1
    # the text of chapter 1 alone — about 10% of what is actually read, so every clip in
    # the section is dropped. Off-by-a-couple is tolerated for front matter.
    if heads and abs(len(heads) - n_sections) <= 2 and 0 <= index - 1 < len(heads):
        a = heads[index - 1].start()
        b = heads[index].start() if index < len(heads) else len(text)
        out.append(("headings", text[a:b]))

    out.append(("duration", _proportional(text, index, n_sections, durations, 0.35)))
    out.append(("duration-wide", _proportional(text, index, n_sections, durations, 1.50)))
    # Last resort. For a one-section book this IS the right slice; for a long book it is
    # slow but correct, and being slow on the final attempt beats dropping the section.
    out.append(("whole-text", text))

    seen, uniq = set(), []
    for name, chunk in out:
        if chunk and chunk not in seen:
            seen.add(chunk)
            uniq.append((name, chunk))
    return uniq


# ------------------------------------------------------------------ audio
def load_audio(path: pathlib.Path, sr: int):
    import librosa
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return wav


# ------------------------------------------------------------------ ASR anchors
def asr_words(wav, sr: int, model_size: str) -> list[tuple[str, float, float]]:
    """Coarse (word, start, end) from ASR. INDEX ONLY — never emitted as text."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="auto", compute_type="int8")
    segs, _ = model.transcribe(wav, language="en", word_timestamps=True, vad_filter=True)
    out = []
    for s in segs:
        for w in (s.words or []):
            nw = norm_word(w.word)
            if nw:
                out.append((nw, float(w.start), float(w.end)))
    return out


def anchor(canon: list[str],
           heard: list[tuple[str, float, float]]) -> list[tuple[int, float, float]]:
    """Map canonical word index -> (start, END), from high-confidence matching blocks.

    Carrying the END time matters: a clip's closing boundary is the end of its last
    word, not the start. Using start times for both edges cuts the final word almost
    entirely, which reads as a truncated tail.
    """
    sm = difflib.SequenceMatcher(a=canon, b=[h[0] for h in heard], autojunk=False)
    anchors: list[tuple[int, float, float]] = []
    for blk in sm.get_matching_blocks():
        if blk.size < 4:                   # short blocks match by luck; ignore them
            continue
        for k in range(blk.size):
            w = heard[blk.b + k]
            anchors.append((blk.a + k, w[1], w[2]))
    anchors.sort()
    # enforce monotonic time; a non-monotonic anchor is a mis-match
    clean: list[tuple[int, float, float]] = []
    for idx, t0, t1 in anchors:
        if not clean or (idx > clean[-1][0] and t0 >= clean[-1][1]):
            clean.append((idx, t0, t1))
    return clean


def dedup_manifest(path):
    """Collapse the manifest to one record per id, last write wins.

    The manifest is opened in append mode and flushed per clip so a mid-book
    crash cannot orphan wavs. The cost is that a re-run after such a crash
    appends a second record for every clip it redoes — with possibly different
    t0/t1/text if the args changed, while only the last one matches the wav on
    disk. qc_gate already had to grow last-record-wins dedup to survive this
    (416 records over 301 wavs); every other jsonl consumer was still exposed.
    Collapsing here, at the one place that writes the file, fixes it for all of
    them. Runs on clean exit only, so the crash-safety of the append is intact.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return 0
    records = {}
    dupes = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn last line from an earlier crash
            key = rec.get("id")
            if key is None:
                continue
            if key in records:
                dupes += 1
            records[key] = rec
    if dupes:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for key in sorted(records):
                f.write(json.dumps(records[key], ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        print(f"   manifest: collapsed {dupes} duplicate record(s) -> "
              f"{len(records)} unique clips", flush=True)
    return dupes


def interp_time(anchors: list[tuple[int, float, float]], idx: int, which: int = 1) -> float | None:
    """which=1 -> word start, which=2 -> word end."""
    if not anchors:
        return None
    lo, hi = 0, len(anchors) - 1
    if idx <= anchors[0][0]:
        return anchors[0][which]
    if idx >= anchors[-1][0]:
        return anchors[-1][which]
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if anchors[mid][0] <= idx:
            lo = mid
        else:
            hi = mid
    a0, a1 = anchors[lo], anchors[hi]
    i0, i1 = a0[0], a1[0]
    t0, t1 = a0[which], a1[which]
    if i1 == i0:
        return t0
    return t0 + (t1 - t0) * (idx - i0) / (i1 - i0)


# ------------------------------------------------------------------ CTC refine
def group_token_spans(token_spans, tokens, t0: float, ratio: float):
    """Per-token CTC spans -> one (start, end) time span per WORD.

    Split out of `refine()` so it can be tested without torch, torchaudio, a GPU or the
    1.18 GB MMS_FA bundle — none of which the host venv has, and the host venv is where
    `make test` runs. `token_spans` needs only `.start` / `.end` in frames.

    `tokens` is the per-word list of token ids, so `len(tokens[i])` says how many of the
    flat token spans belong to word `i`. That mapping is the entire fix for A-M2: the
    previous code inferred it by counting non-blank FRAMES, and a token occupies a run of
    frames, not one.
    """
    spans, pos = [], 0
    for tok in tokens:
        group = token_spans[pos:pos + len(tok)]
        pos += len(tok)
        if not group:
            spans.append(None)
            continue
        spans.append((t0 + group[0].start * ratio, t0 + group[-1].end * ratio))
    return spans


def refine(wav, sr, words: list[str], t0: float, t1: float):
    """Precise word spans for `words` inside [t0,t1] via CTC forced alignment.

    Returns (spans, score) or (None, 0.0) if the bundle is unavailable or the span is
    too long to align in one call.
    """
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA as BUNDLE
    except Exception:  # noqa: BLE001 - refinement is optional, anchors still work
        return None, 0.0
    if t1 - t0 > ANCHOR_WINDOW_S * 1.5 or not words:
        return None, 0.0
    a, b = int(max(0, t0) * sr), int(min(len(wav) / sr, t1) * sr)
    if b - a < sr * 0.2:
        return None, 0.0
    # Cache the model. It is 1.18 GB; rebuilding and moving it to the GPU once per
    # SENTENCE (which the first version did) dominates runtime by orders of magnitude.
    global _FA_CACHE
    if _FA_CACHE is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _FA_CACHE = (BUNDLE.get_model().to(device), BUNDLE.get_labels(), device)
    model, labels, device = _FA_CACHE
    dic = {c: i for i, c in enumerate(labels)}
    tokens = [[dic[c] for c in w if c in dic] for w in words]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None, 0.0
    flat = [t for tok in tokens for t in tok]
    with torch.inference_mode():
        wave = torch.tensor(wav[a:b], dtype=torch.float32, device=device).unsqueeze(0)
        emission, _ = model(wave)
        try:
            aligned, scores = torchaudio.functional.forced_align(
                emission, torch.tensor([flat], dtype=torch.int32, device=device), blank=0)
        except Exception:  # noqa: BLE001 - text longer than the emission, etc.
            return None, 0.0
    ratio = (b - a) / emission.size(1) / sr

    # A-M2. `forced_align` returns ONE ENTRY PER FRAME, so a single target token occupies
    # a RUN of frames — `[0,0,t1,t1,t1,0,t2,0,0,t3,t3]` for three tokens. The previous
    # walk counted non-blank FRAMES against `len(tok)` CHARACTERS and stopped there, which
    # is only correct if every character happens to occupy exactly one frame. It never
    # does. The error compounds along the sentence: each word consumes the frames of the
    # word before it, and the whole sentence is exhausted after `len(flat)` non-blank
    # frames out of the many more that are really there.
    #
    # THIS IS THE TAIL TRUNCATION, not a CTC failure. The caller carries a comment
    # blaming "CTC may fail to align the LAST words of a sentence" and a measured example
    # ("...a chill wind blew [through the world.]", 3 words lost, tripped tail_ok). The
    # arithmetic explains it exactly: the last word's end lands on the `len(flat)`-th
    # non-blank frame, which is systematically early. The anchor-time `max()` there masks
    # the symptom and stays as a guard, but it was compensating for this.
    #
    # `merge_tokens` is the API for exactly this: it collapses the frame-level alignment
    # into one span per TARGET TOKEN, with frame boundaries and a per-token score.
    merge = getattr(torchaudio.functional, "merge_tokens", None)
    if merge is None:
        # Refinement is optional by contract, and a wrong span is worse than none:
        # anchors alone still produce a correct (if looser) cut.
        return None, 0.0
    try:
        token_spans = merge(aligned[0], scores[0])
    except Exception:  # noqa: BLE001 - stay on the anchor path rather than guess
        return None, 0.0
    if len(token_spans) != len(flat):
        # One span per target token is the contract. If that ever stops holding, the
        # grouping below would silently mis-attribute every word from that point on.
        return None, 0.0

    spans = group_token_spans(token_spans, tokens, t0, ratio)
    score = float(scores.exp().mean()) if scores is not None and scores.numel() else 0.0
    return spans, score


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-dir", required=True, help="output of librivox_fetch.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", help="1-based, e.g. '1' or '1-3' (default: all present)")
    ap.add_argument("--asr-model", default="small.en")
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--no-refine", action="store_true", help="anchors only, skip CTC")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (smoke test)")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf

    bdir = pathlib.Path(args.book_dir)
    book = json.loads((bdir / "book.json").read_text(encoding="utf-8"))
    full_text = (bdir / "text/source.txt").read_text(encoding="utf-8")
    out = pathlib.Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    # The manifest lives BESIDE the wavs, not above them: qc_gate.py and
    # register_audition.py both discover clips by globbing *_manifest.jsonl in the
    # directory that holds the audio, which is the convention synth_bank.sh writes.
    man = (out / "audio" / "librivox_manifest.jsonl").open("a", encoding="utf-8")

    present = sorted((bdir / "audio").glob("*.mp3"))
    if args.sections:
        keep = set()
        for part in args.sections.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                keep.update(range(int(a), int(b) + 1))
            else:
                keep.add(int(part))
        present = [p for p in present if int(p.stem) in keep]

    slug = re.sub(r"[^a-z0-9]+", "-", (book.get("title") or "book").lower()).strip("-")
    # Reader per section. For real audio the READER is the casting intent: gender, age
    # and accent are properties of the person at the microphone, constant across every
    # clip they narrate — in this book and in any other. Carrying it in the manifest is
    # what lets register_audition pre-fill those columns instead of asking the ear to
    # re-enter the same three values a thousand times.
    readers = {int(sec["index"]): sec.get("reader")
               for sec in (book.get("sections") or []) if sec.get("index")}
    # Per-section playtime drives the text split (see chapter_slices). This must cover
    # EVERY section in the book, not just the ones being aligned now — the fraction is
    # cumulative over the whole work, so a --sections 3-5 run still needs to know how
    # long sections 1 and 2 were.
    durations = section_durations(book)
    n_written = n_incomplete = 0

    for mp3 in present:
        idx = int(mp3.stem)
        print(f"\n== section {idx}: {mp3.name} ==", flush=True)
        wav = load_audio(mp3, SAMPLE_RATE)
        dur = len(wav) / SAMPLE_RATE
        print(f"   audio {dur/60:.1f} min", flush=True)

        # The LibriVox API returns num_sections as a STRING ("25"), not an int.
        try:
            n_sec = int(book.get("num_sections") or 0) or len(present)
        except (TypeError, ValueError):
            n_sec = len(present)
        # ASR depends on the AUDIO only, so it runs once and every candidate slice is
        # scored against the same heard words. This is what makes the A-H1 fallback cheap:
        # a retry is one more difflib pass, not another transcription.
        heard = asr_words(wav, SAMPLE_RATE, args.asr_model)

        # Coverage is measured against the HEARD words, not the canonical slice. The
        # proportional slices deliberately over-cover, so "fraction of canonical words
        # anchored" is low by construction and says nothing about alignment quality. The
        # question that matters is the reverse: did we locate what the reader actually
        # said inside the canonical text?
        best = None
        for strategy, chap in chapter_slices(full_text, idx, n_sec, durations):
            sents = split_sentences(chap)
            canon_words, sent_of = [], []
            for si, s in enumerate(sents):
                for w in s.split():
                    nw = norm_word(w)
                    if nw:
                        canon_words.append(nw)
                        sent_of.append(si)
            anchors = anchor(canon_words, heard)
            cover = len(anchors) / max(len(heard), 1)
            print(f"   [{strategy}] {len(sents)} sentences, {len(canon_words)} words "
                  f"-> {len(anchors)} anchors ({cover:.0%} of {len(heard)} heard words "
                  f"located)", flush=True)
            if best is None or cover > best[0]:
                best = (cover, strategy, sents, canon_words, sent_of, anchors)
            if cover >= 0.60:
                break

        cover, strategy, sents, canon_words, sent_of, anchors = best
        if cover < 0.60:
            # Every strategy failed, so this is the edition or the audio, not the slicer.
            # Naming the best attempt is what tells those apart in the log.
            print(f"   !! too little of the audio was found in this text — best was "
                  f"{strategy} at {cover:.0%}, below the 60% gate. Wrong edition, or a "
                  f"section whose text is not in this source; skipping",
                  file=sys.stderr)
            continue
        if strategy != "headings":
            print(f"   using the {strategy} slice", flush=True)
        # Trim to the region the audio actually covers; everything outside it is padding
        # from the slice and would otherwise get clamped, nonsense timings.
        lo_idx, hi_idx = anchors[0][0], anchors[-1][0]
        print(f"   spoken region: canonical words {lo_idx}-{hi_idx} "
              f"of {len(canon_words)}", flush=True)

        # sentence -> (first word idx, last word idx)
        bounds: dict[int, list[int]] = {}
        for wi, si in enumerate(sent_of):
            b = bounds.setdefault(si, [wi, wi])
            b[1] = wi

        for si, sent in enumerate(sents):
            if si not in bounds:
                continue
            w0, w1 = bounds[si]
            if w0 < lo_idx or w1 > hi_idx:
                continue          # outside the audio's span — slice padding
            t0 = interp_time(anchors, w0, which=1)   # start of the first word
            t1 = interp_time(anchors, w1, which=2)   # END of the last word
            if t0 is None or t1 is None or t1 <= t0:
                continue
            # Per-clip score. NEVER the section-level coverage — that is a property of
            # the slice, not of this sentence. Prefer the CTC score; fall back to local
            # anchor density over this sentence's own word range.
            local = sum(1 for a in anchors if w0 <= a[0] <= w1) / max(w1 - w0 + 1, 1)
            score = local
            if not args.no_refine:
                spans, sc = refine(wav, SAMPLE_RATE,
                                   [canon_words[i] for i in range(w0, w1 + 1)],
                                   max(0.0, t0 - 0.5), t1 + 0.5)
                if spans:
                    real = [s for s in spans if s]
                    if real:
                        # This was diagnosed as "CTC fails to align the LAST words of a
                        # sentence", measured as "...a chill wind blew [through the
                        # world.]" losing 3 words and tripping tail_ok. The real cause
                        # was A-M2 inside `refine` — the per-word grouping counted frames
                        # against characters, so the sentence ran out after `len(flat)`
                        # non-blank frames and every span landed early. Reproduced on a
                        # real alignment: the end came out at frame 6 of 13, losing 54%.
                        #
                        # Fixed at the source (2026-08-07, `merge_tokens`). These two
                        # bounds STAY: CTC genuinely can drop an edge word, and holding
                        # the interpolated anchor time as a floor for the end (and a
                        # ceiling for the start) costs nothing when the spans are right.
                        t0 = min(t0, real[0][0])
                        t1 = max(t1, real[-1][1])
                        score = sc if sc > 0 else local
            # Structural clamp: a clip may NEVER reach into the next sentence. CTC can
            # overshoot into the search padding, and the earlier max() above (added on a
            # wrong truncation diagnosis, kept because it still guards a real edge case)
            # can extend the end. Measured failure: a clip whose canonical text was read
            # perfectly still scored WER 0.38 because it ran on into "He was generally
            # to be found at the pen and ink cl...". Over-run reads as a transcript
            # error even when the alignment is good.
            # Pad FIRST, clamp LAST. The other order is self-defeating: clamping
            # to nxt_start - 0.02 and then adding PAD_SECONDS back put the end
            # 0.10 s *inside* the next sentence on every clip with a neighbour,
            # so essentially the whole book lane carried audio its transcript
            # lacked — at both edges, and invisible to WER because the extra
            # words are real speech the reference text simply does not contain.
            t0 = max(0.0, t0 - PAD_SECONDS)
            t1 = min(dur, t1 + PAD_SECONDS)

            nxt = bounds.get(si + 1)
            if nxt:
                nxt_start = interp_time(anchors, nxt[0], which=1)
                if nxt_start is not None and nxt_start > t0:
                    t1 = min(t1, nxt_start - EDGE_GUARD)
            # The start edge had no clamp at all, so the pad (and the
            # min(t0, real[0][0]) anchor pull above) reached back into the
            # previous sentence unchecked.
            prv = bounds.get(si - 1)
            if prv:
                prv_end = interp_time(anchors, prv[1], which=2)
                if prv_end is not None and prv_end < t1:
                    t0 = max(t0, prv_end + EDGE_GUARD)
            secs = t1 - t0
            if secs < MIN_SECONDS or secs > MAX_SECONDS or score < args.min_score:
                continue
            # A-H5: completeness, the gate the book lane has and this one did not.
            # Cheap and before the audio work on purpose. With pysbd above this should
            # rarely fire, which is the point — it is the check that the splitter is
            # doing its job, not a substitute for it.
            if not is_complete_utterance(sent):
                n_incomplete += 1
                continue
            # Speech floor. Measured with the energy gate (librosa.effects.split at
            # top_db=35) rather than ASR word spans: word spans absorb the silence around
            # each word and over-estimate, which let a clip through here at ">=4.0 s" that
            # qc_gate then measured at 3.8 s. Two different measures of one owner rule is
            # one measure too many.
            #
            # qc_gate's `speech_ok` became a HARD gate on the SILERO VAD figure on
            # 2026-08-07 (owner's call), so this is no longer literally the same
            # instrument. It stays the energy gate on purpose, and the divergence is
            # measured rather than assumed: mean VAD/energy ratio 1.008 over 150 clips,
            # one clip in 150 changing side at the 4 s floor. This is an INGEST
            # pre-filter — its job is to not cut a clip that admission will reject — and
            # at 0.7% disagreement it still does that. If that number moves, this is the
            # line to change.
            seg = wav[int(t0 * SAMPLE_RATE):int(t1 * SAMPLE_RATE)]
            import librosa as _lb
            speech = float(sum(e - st for st, e in
                               _lb.effects.split(seg, top_db=35))) / SAMPLE_RATE
            if speech < MIN_SPEECH_SECONDS:
                continue
            clip = wav[int(t0 * SAMPLE_RATE):int(t1 * SAMPLE_RATE)]
            if clip.size < SAMPLE_RATE * MIN_SECONDS:
                continue
            import librosa
            clip24 = librosa.resample(clip, orig_sr=SAMPLE_RATE, target_sr=OUT_SR)
            cid = f"{slug}_lv{idx:03d}_{si:04d}"
            write_wav_atomic(str(out / "audio" / f"{cid}.wav"), clip24, OUT_SR)
            man.write(json.dumps({
                "id": cid, "wav": f"{cid}.wav", "engine": "librivox",
                "text": sent, "source": "real-audio/force-align",
                "book": book.get("title"), "ledger_key": book.get("ledger_key"),
                "librivox_url": book.get("librivox_url"),
                "section": idx, "sentence_index": si,
                "reader": readers.get(idx),
                "t0": round(t0, 3), "t1": round(t1, 3), "seconds": round(secs, 3),
                "speech_seconds": round(speech, 3),
                "align_score": round(float(score), 4),
                "anchor_coverage_asr": round(cover, 4),
                "local_anchor_density": round(local, 4),
                "sr": OUT_SR, "license": "LibriVox public domain",
                "text_source": book.get("url_text_source"),
            }, ensure_ascii=False) + "\n")
            man.flush()
            n_written += 1
            if args.limit and n_written >= args.limit:
                print(f"\n   --limit {args.limit} reached", flush=True)
                man.close()
                dedup_manifest(out / "audio" / "librivox_manifest.jsonl")
                print(f"\n{n_written} clips -> {out}")
                return 0
    man.close()
    dedup_manifest(out / "audio" / "librivox_manifest.jsonl")
    if n_incomplete:
        # Reported, not silent: with pysbd upstream this should be near zero, so a
        # non-trivial count means the splitter is struggling with this edition.
        print(f"\n   {n_incomplete} sentence(s) rejected as incomplete utterances")
    if n_written == 0:
        # Exiting 0 here made a book whose every section was gate-skipped
        # indistinguishable from a successful run without reading the log.
        print(f"\nNO CLIPS WRITTEN from {bdir} — every section was skipped "
              f"(coverage gate, score floor, or duration filters).")
        return 1
    print(f"\n{n_written} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
