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
import pathlib
import re
import sys
import unicodedata

SAMPLE_RATE = 16000          # the aligner's rate; clips are written at OUT_SR
OUT_SR = 24000               # the corpus interchange rate (model-decisions.md)
MIN_SECONDS = 4.0            # wall-clock floor
MIN_SPEECH_SECONDS = 4.0     # [[min-clip-length-4s]] — the owner floor is SPEECH,
                             # not duration: a 4 s clip holding 2.5 s of speech and
                             # 1.5 s of room tone does not clear it.
MAX_SECONDS = 22.0           # matches derive_vat_corpus.MAX_SECONDS
PAD_SECONDS = 0.12           # breath room around a cut
ANCHOR_WINDOW_S = 30.0       # max audio per CTC call

_FA_CACHE = None             # (model, labels, device) — see refine()

SENT_END = re.compile(r'(?<=[.!?])["”’\')\]]*\s')


# ------------------------------------------------------------------ text
def norm_word(w: str) -> str:
    w = unicodedata.normalize("NFKD", w).lower()
    return re.sub(r"[^a-z0-9]", "", w)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    out, start = [], 0
    for m in SENT_END.finditer(text):
        out.append(text[start:m.end()].strip())
        start = m.end()
    if start < len(text):
        out.append(text[start:].strip())
    return [s for s in out if s]


def chapter_slice(text: str, index: int, n_sections: int,
                  durations: dict[int, float] | None = None) -> str:
    """Best-effort chapter extraction.

    LibriVox sections usually map to chapters, but the mapping is not guaranteed and the
    Gutenberg text's chapter headings vary wildly. We try headings first and fall back to
    a proportional split, which is good enough because the ASR anchor pass re-finds the
    true position inside whatever slice we hand it.

    SPLIT BY DURATION, NOT BY SECTION INDEX. An even index split assumes every section is
    the same length. That holds for a novel — it is why *Uneasy Money* aligned at 94% —
    and fails completely for a collection. Dickens's *Speeches* opens with a 73-minute
    editorial introduction (sections 1-2) and then has speeches as short as 2.6 minutes;
    an even split put section 3's window at 3.3% of the text when the words were near
    10.8%, so the anchor pass located **0%** of the heard words and every clip was
    dropped. The 35% padding is not remotely enough to bridge that.

    Cumulative playtime fraction fixes it, and it rests on the assumption the even split
    was already making implicitly and more weakly: that reading pace is roughly constant
    within one reader. Where durations are missing we fall back to the old even split.
    """
    heads = list(re.finditer(r"(?im)^\s*(chapter|section)\s+([IVXLC]+|\d+)\b.*$", text))
    if len(heads) >= n_sections and 0 <= index - 1 < len(heads):
        a = heads[index - 1].start()
        b = heads[index].start() if index < len(heads) else len(text)
        return text[a:b]

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
    pad = max(span * 0.35, 0.01)
    a = max(0, int((f0 - pad) * len(text)))
    b = min(len(text), int((f1 + pad) * len(text)))
    return text[a:b]


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
    spans, cur, k = [], 0, 0
    frames = aligned[0].tolist()
    # walk token runs back to per-word time spans
    pos = 0
    for tok in tokens:
        need = len(tok)
        start_f = end_f = None
        got = 0
        while pos < len(frames) and got < need:
            if frames[pos] != 0:
                if start_f is None:
                    start_f = pos
                end_f = pos
                got += 1
            pos += 1
        if start_f is None:
            spans.append(None)
        else:
            spans.append((t0 + start_f * ratio, t0 + (end_f + 1) * ratio))
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
    # Per-section playtime drives the text split (see chapter_slice). This must cover
    # EVERY section in the book, not just the ones being aligned now — the fraction is
    # cumulative over the whole work, so a --sections 3-5 run still needs to know how
    # long sections 1 and 2 were.
    durations: dict[int, float] = {}
    for sec in (book.get("sections") or []):
        try:
            durations[int(sec["index"])] = float(sec.get("playtime") or 0)
        except (TypeError, ValueError, KeyError):
            continue
    if durations and not all(durations.values()):
        durations = {}                     # a zero anywhere makes the cumsum a lie
    n_written = 0

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
        chap = chapter_slice(full_text, idx, n_sec, durations)
        sents = split_sentences(chap)
        canon_words, sent_of = [], []
        for si, s in enumerate(sents):
            for w in s.split():
                nw = norm_word(w)
                if nw:
                    canon_words.append(nw)
                    sent_of.append(si)
        print(f"   canonical slice: {len(sents)} sentences, {len(canon_words)} words",
              flush=True)

        heard = asr_words(wav, SAMPLE_RATE, args.asr_model)
        anchors = anchor(canon_words, heard)
        # Coverage is measured against the HEARD words, not the canonical slice.
        # chapter_slice() deliberately over-covers (an even split plus 35% padding), so
        # "fraction of canonical words anchored" is low by construction and says nothing
        # about alignment quality. The question that matters is the reverse: did we
        # locate what the reader actually said inside the canonical text?
        cover = len(anchors) / max(len(heard), 1)
        print(f"   ASR {len(heard)} words -> {len(anchors)} anchors "
              f"({cover:.0%} of HEARD words located in canonical text)", flush=True)
        if cover < 0.60:
            print("   !! too little of the audio was found in this text — wrong chapter "
                  "slice or wrong edition; skipping", file=sys.stderr)
            continue
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
                        # CTC may fail to align the LAST words of a sentence, and
                        # real[-1][1] is then the end of the last word it DID align —
                        # silently truncating the tail. Measured: "...a chill wind blew
                        # [through the world.]" lost 3 words and tripped tail_ok.
                        # The interpolated anchor time is the floor for the end boundary
                        # (and the ceiling for the start), so unaligned edge words are
                        # still inside the clip.
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
            nxt = bounds.get(si + 1)
            if nxt:
                nxt_start = interp_time(anchors, nxt[0], which=1)
                if nxt_start is not None and nxt_start > t0:
                    t1 = min(t1, nxt_start - 0.02)
            t0 = max(0.0, t0 - PAD_SECONDS)
            t1 = min(dur, t1 + PAD_SECONDS)
            secs = t1 - t0
            if secs < MIN_SECONDS or secs > MAX_SECONDS or score < args.min_score:
                continue
            # Speech floor. Measured with EXACTLY the instrument qc_gate uses
            # (librosa.effects.split at top_db=35) rather than ASR word spans: word
            # spans absorb the silence around each word and over-estimate, which let a
            # clip through here at ">=4.0 s" that qc_gate then measured at 3.8 s. Two
            # different measures of one owner rule is one measure too many.
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
            sf.write(str(out / "audio" / f"{cid}.wav"), clip24, OUT_SR)
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
                print(f"\n{n_written} clips -> {out}")
                return 0
    man.close()
    print(f"\n{n_written} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
