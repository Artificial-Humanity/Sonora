#!/usr/bin/env python3
"""Stage a blind ear test over CORPUS SOURCE AUDIO — no model, no vocoder, no render.

WHY THIS EXISTS (2026-09-19)
----------------------------
Every ear test so far has compared two things the model produced, which can only ever
answer "which checkpoint". The complaint that survived all of them is a robotic hum, and
`vat7_select` showed it is speaker-bound: index 4899 drew it in 8 of 8 notes while 4896,
the adjacent speaker by row count, read as "mostly human". Both are ranks 1 and 2 of 5,332
(`measure_corpus_speaker_balance.py`), so quantity does not explain it.

That leaves one question no render can answer: **is the hum already in the training
audio?** If it is, the model is faithful and the defect is upstream of training — which
makes the corpus, not the ladder, the thing to change. This stages the clips that ask it.

THE SETS ARE THE ARGUMENT, and two of the three exist to make the first one falsifiable.

`--contrast A:B` is the question — one clip from each speaker, side by side.

`--null SPK` is **the negative control, and it is the set that decides whether to believe
the others.** Both sides are the same speaker, different clips. A listener who reliably
calls one side cleaner HERE is hearing clip-to-clip variation, not speaker character, and
a contrast result is then worth nothing. This set is expected to come out a tie and is a
finding if it does not.

A second `--contrast` against a speaker from a different source partition is the anchor:
it says whether any separation found is about these two voices or about the partition they
came from.

⚠ LOUDNESS IS MATCHED, AND THAT IS NOT OPTIONAL. LibriTTS-R is not loudness-normalised
(median −18.16 LUFS against the expressive bank's −23.00), so per-clip level varies freely
and the louder side of a pair reads as the better one. This project has already had a
verdict run backwards on exactly that confound. Every served clip is normalised to
`--lufs`; the ORIGINALS ARE NOT TOUCHED, because the corpus's loudness is deliberate and a
second pass over it would damage it — these are throwaway copies under the test directory.

⚠ CLIPS COME FROM THE FILELIST, NOT THE DATASET DIRECTORY. What matters is what the model
was trained on, and the filelist is the only record of that. A clip on disk that no row
references was never seen.

⚠ THE KEY IS WRITTEN OUTSIDE THE TEST DIRECTORY, for the reason `ear_bench.Bench.write`
gives: the app bind-mounts the test directory whole, so a key inside it would be in the
container and the blinding would rest on the app not opening a path it can reach.

Run:  .venv/bin/python scripts/tools/build_source_audio_eartest.py \
          --corpus data/libritts_r_full_vat_v7 \
          --out /data/model-training/sonora/eartest/source_audio \
          --contrast 4896:4899 --pairs 12 \
          --contrast 183:4899  --pairs 6 \
          --null 4896 --null 4899 --null-pairs 3
Exit: 0 staged; 2 the input is not something this can stage (unknown speaker, a speaker
      with too few eligible clips, an unreadable source file, a malformed --contrast).
"""
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "soundfile", "pyloudnorm"]
# ///
import argparse
import collections
import hashlib
import json
import os
import random
import sys

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO,):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pyloudnorm  # noqa: E402
import soundfile as sf  # noqa: E402

from matcha.delivery import VAT_DIM  # noqa: E402

# ⚠ THE SAME FUNCTION AS `ear_bench.opaque`, DUPLICATED ON PURPOSE. That module imports
# torch at import time and the host venv deliberately has none, so this tool cannot reach
# it. `tests/test_build_source_audio_eartest.py` lifts ear_bench's definition out of the
# source with `ast` and asserts the two agree, so the copy cannot drift silently.
def opaque(pair_key, side_key, salt):
    return hashlib.sha1(f"{salt}|{pair_key}|{side_key}".encode()).hexdigest()[:16]


def _refuse(msg):
    print("REFUSED: %s" % msg, file=sys.stderr)
    raise SystemExit(2)


def read_filelist(corpus, name="train_op.txt"):
    """speaker index -> [audio paths], from the filelist rather than the dataset tree."""
    path = os.path.join(corpus, name)
    if not os.path.exists(path):
        _refuse("%s holds no %s" % (corpus, name))
    by_spk = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 4:
                _refuse("%s line %d states %d fields and a filelist row has 4"
                        % (path, n, len(parts)))
            if len(parts[3].split(",")) != VAT_DIM:
                _refuse("%s line %d is not a %d-wide filelist" % (path, n, VAT_DIM))
            by_spk[int(parts[1])].append(parts[0])
    return by_spk


def eligible(paths, min_seconds):
    """Paths whose audio is at least `min_seconds` long, with their duration.

    The floor is the owner's standing one: no clip under 4 s of speech enters a bank. A
    two-second clip does not carry enough of a voice to judge its recording character.
    """
    out = []
    for p in paths:
        try:
            info = sf.info(p)
        except Exception as e:                                  # noqa: BLE001
            _refuse("%s could not be read (%s) — the filelist references audio that is "
                    "not on this box" % (p, type(e).__name__))
        secs = info.frames / float(info.samplerate)
        if secs >= min_seconds:
            out.append((p, secs))
    return out


def load_normalised(path, lufs):
    """Audio at `lufs`, mono, with its original sample rate. Originals are never written.

    ⚠ The gain is peak-limited. A quiet clip can need +12 dB to reach the target, which
    clips on write and manufactures exactly the harshness this test is trying to hear.
    When the ceiling binds, the clip is reported rather than silently served quieter.
    """
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    loudness = pyloudnorm.Meter(sr).integrated_loudness(x)
    if not np.isfinite(loudness):
        _refuse("%s has no measurable loudness — it is probably silence" % path)
    gain = 10.0 ** ((lufs - loudness) / 20.0)
    peak = float(np.max(np.abs(x))) or 1.0
    limited = min(gain, 0.99 / peak)
    return (x * limited).astype("float32"), sr, loudness, limited < gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-out", default="")
    ap.add_argument("--contrast", action="append", default=[],
                    help="A:B — one clip from each speaker, repeatable")
    ap.add_argument("--null", action="append", type=int, default=[],
                    help="SPK — both sides the same speaker (the negative control)")
    ap.add_argument("--pairs", action="append", type=int, default=[],
                    help="pairs for each --contrast, in the same order")
    ap.add_argument("--null-pairs", type=int, default=3)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--lufs", type=float, default=-23.0)
    ap.add_argument("--salt", default="source-audio-v1")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not args.contrast and not args.null:
        _refuse("nothing to stage — pass at least one --contrast or --null")
    if args.pairs and len(args.pairs) != len(args.contrast):
        _refuse("%d --pairs values for %d --contrast sets; give one per contrast or none"
                % (len(args.pairs), len(args.contrast)))
    if args.contrast and not args.null:
        print("⚠ NO --null SET. Without a same-speaker control there is nothing to say "
              "whether a contrast result is speaker character or clip-to-clip variation.",
              file=sys.stderr)

    by_spk = read_filelist(args.corpus)
    rng = random.Random(args.seed)

    pool = {}
    def _pool(spk):
        if spk not in pool:
            if spk not in by_spk:
                _refuse("the corpus train split holds no speaker %d — its indices run "
                        "%d to %d" % (spk, min(by_spk), max(by_spk)))
            pool[spk] = eligible(by_spk[spk], args.min_seconds)
        return pool[spk]

    # ⚠ ALLOCATION IS GLOBAL, NOT PER-SET. A speaker named in two contrasts and a null
    # set gets sampled three times, and independent draws from one pool collide — the
    # first real invocation of this tool put five clips on the board twice. Taking from a
    # shrinking pool is what makes the repeat guard below unreachable rather than a
    # tripwire that fires on ordinary use.
    taken = set()
    def _take(spk, n, what):
        free = [c for c in _pool(spk) if c[0] not in taken]
        if len(free) < n:
            _refuse("speaker %d has %d unallocated clips of at least %.1fs and %s asks "
                    "for %d more (%d of its %d are already on the board) — lower the pair "
                    "counts or --min-seconds"
                    % (spk, len(free), args.min_seconds, what, n,
                       len(_pool(spk)) - len(free), len(_pool(spk))))
        picks = rng.sample(free, n)
        taken.update(p for p, _ in picks)
        return picks

    plan = []      # (set_name, pair_key, [(side_key, spk, path), ...])
    truth = {}     # internal set name -> what it really is. NEVER SERVED.
    for i, spec in enumerate(args.contrast):
        if spec.count(":") != 1:
            _refuse("--contrast %r is not A:B" % spec)
        try:
            a, b = (int(x) for x in spec.split(":"))
        except ValueError:
            _refuse("--contrast %r does not name two speaker indices" % spec)
        n = args.pairs[i] if args.pairs else 8
        name = "contrast_%d_vs_%d" % (a, b)
        pa = _take(a, n, name)
        pb = _take(b, n, name)
        truth[name] = {"kind": "contrast", "speakers": [a, b]}
        # ⚠⚠ SIDES ARE COIN-FLIPPED PER PAIR. Without this, speaker `a` is side A in every
        # item of the set — the app's own hint tells the listener "which clip is A changes
        # from item to item", so a listener who noticed the consistency would be unblinded
        # by a promise the manifest was breaking. Caught after the first staging run.
        # ⚠ EXACTLY HALF EACH WAY, not a coin per pair. A fair coin gave 8/4 on the
        # first build, and the tally is by SPEAKER — so any side preference in the
        # listener leaks straight into the speaker that happened to sit on side A more
        # often. Balancing the assignment removes the concern instead of assuming the
        # listener has no side bias.
        flip = [True] * (n // 2) + [False] * (n - n // 2)
        rng.shuffle(flip)
        for j in range(n):
            sides = ([("A", b, pb[j][0]), ("B", a, pa[j][0])] if flip[j]
                     else [("A", a, pa[j][0]), ("B", b, pb[j][0])])
            plan.append((name, "%s_%02d" % (name, j), sides))

    for spk in args.null:
        n = args.null_pairs
        name = "null_%d" % spk
        picks = _take(spk, 2 * n, name)
        truth[name] = {"kind": "null", "speakers": [spk]}
        for j in range(n):
            plan.append((name, "%s_%02d" % (name, j),
                         [("A", spk, picks[2 * j][0]), ("B", spk, picks[2 * j + 1][0])]))

    # ⚠ NO CLIP MAY SERVE TWICE. The same file on both sides of a pair is an automatic
    # tie that reads as a real "no difference"; the same file in two pairs makes those
    # judgements non-independent, which the sign test assumes they are not.
    used = collections.Counter(p for _, _, sides in plan for _, _, p in sides)
    repeats = [p for p, c in used.items() if c > 1]
    if repeats:
        _refuse("%d source clips would be served more than once (e.g. %s) — judgements "
                "would not be independent" % (len(repeats), repeats[0]))

    # ⚠⚠ THE SETS ARE RENAMED AND GIVEN ONE IDENTICAL BRIEF, AND THAT IS THE BLINDING.
    # An earlier build served `contrast_4896_vs_4899` and `null_4896` as tab names with
    # briefs to match, which hands the listener two things they must not have. The
    # speaker numbers: the owner already knows 4899 as the voice they condemned, so
    # naming it sets the expectation the test exists to check. And WHICH SET IS THE
    # CONTROL: a listener told "both sides are the same speaker here" can answer "no
    # difference" without listening, and a control answered reflexively measures nothing.
    # Every set now looks the same and asks the same question. `truth` stays in the key.
    rng.shuffle(order := sorted(truth))
    label_of = {name: "set_%d" % (i + 1) for i, name in enumerate(order)}
    ASK = ("These are RECORDINGS, not model output. Ignore what is said, ignore the "
           "reading, and ignore which voice you would rather listen to. Judge the "
           "RECORDING only: does one side sound like a microphone in a room while the "
           "other sounds like something a machine has been through — thin, hollow, "
           "metallic, a hum under the voice? Loudness is matched across every clip, so a "
           "level difference is not the thing to hear. 'No difference' is a real answer "
           "and it is the right one more often than you might expect.")
    sets = {label_of[n]: {
        "title": "Set %d — source recordings" % (i + 1),
        "ask": ASK,
        "labels": {"A": "A is the cleaner recording", "same": "No difference",
                   "B": "B is the cleaner recording"},
    } for i, n in enumerate(order)}
    # ⚠ THE PAIR KEY IS SERVED AS THE ITEM ID, so it is relabelled with the rest.
    # `contrast_4896_vs_4899_00` on screen would defeat everything above it.
    plan = [(label_of[s], pk.replace(s, label_of[s], 1), sides)
            for s, pk, sides in plan]

    out = args.out
    os.makedirs(os.path.join(out, "clips"), exist_ok=True)
    os.makedirs(os.path.join(out, "verdicts"), exist_ok=True)

    key, served, limited, rates = {}, [], [], set()
    for set_name, pair_key, sides in plan:
        # ⚠ THE BLIND FIELDS ARE PART OF THE MANIFEST SHAPE, not decoration. The app's
        # renderer does `it.vat.join(...)` on every item, so a manifest without `vat` threw
        # in the browser and left a blank page — the listener sees nothing load and no
        # error. Every sibling bench writes these three; this one did not.
        item = {"id": pair_key, "set": set_name, "text": "(source recording)",
                "spk": "(blind)", "vat": [], "delivery_ui": "(blind)"}
        for side_key, spk, path in sides:
            name = opaque(pair_key, side_key, args.salt)
            wav, sr, loud, hit = load_normalised(path, args.lufs)
            rates.add(sr)
            if hit:
                limited.append(path)
            sf.write(os.path.join(out, "clips", "%s.wav" % name), wav, sr, "PCM_24")
            key[name] = {"label": "spk%d" % spk, "pair": pair_key, "side": side_key,
                         "source": path, "measured_lufs": round(loud, 2)}
            item[side_key] = name
        served.append(item)

    if len(rates) != 1:
        _refuse("the staged clips carry %d different sample rates (%s) — the app serves "
                "them as-is and a rate change is audible as a character change"
                % (len(rates), sorted(rates)))

    manifest = {"test": os.path.basename(out.rstrip("/")), "sample_rate": rates.pop(),
                "sets": sets, "items": served}
    with open(os.path.join(out, "items.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    key_out = args.key_out or os.path.join(
        os.path.dirname(out.rstrip("/")), "_keys", "%s.key.json" % manifest["test"])
    os.makedirs(os.path.dirname(key_out), exist_ok=True)
    with open(key_out, "w", encoding="utf-8") as f:
        json.dump({"test_dir": out, "salt": args.salt,
                   "sets": {label_of[n]: truth[n] for n in truth},
                   "clips": key}, f, indent=2)

    per_set = collections.Counter(i["set"] for i in served)
    print("STAGED %d pairs / %d clips into %s" % (len(served), len(key), out))
    for k in sorted(per_set):
        # ⚠ To STDERR, and only here. Whoever runs this already knows the design; the
        # listener must not, and stdout is what tends to get pasted into a message.
        print("   %-8s %2d pairs" % (k, per_set[k]))
        t = {label_of[n]: truth[n] for n in truth}[k]
        print("            = %s %s" % (t["kind"], t["speakers"]), file=sys.stderr)
    print("KEY    %s   (outside the served directory)" % key_out)
    if limited:
        print("⚠ %d clips hit the peak ceiling before reaching %.1f LUFS and are quieter "
              "than the target: %s" % (len(limited), args.lufs, limited[:3]))
    # ⚠ Checked against `truth`, not against the served names — the relabelling above
    # renamed every set to set_N, so a prefix test on the served name always fired.
    if not any(t["kind"] == "null" for t in truth.values()):
        print("⚠ staged with NO null control — a contrast result here is uninterpretable")


if __name__ == "__main__":
    main()
