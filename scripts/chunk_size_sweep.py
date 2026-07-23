"""Chunk-size sweep: find the passage length where quality degrades.

The Director will chunk long text before handing it to the Actor; this sweep
determines the chunk-size value (north-star follow-on to §7; see
notes/model-size-target-decision.md, "Director-side chunking"). It
renders one fixed fresh prose passage at cumulative prefix lengths — token
buckets bracketing the 256-token litert export ceiling from well below to
~16x above — through the torch pipeline (NOT the fixed-shape tflite graphs,
so we measure the model's intrinsic degradation, unclipped), then scores
each render for intelligibility (faster-whisper WER), speaking-rate drift,
and within-render loudness drift. WAVs are kept for the human audit that
makes the final call.

Prefix design: every bucket starts with the same sentences, so buckets
differ only in how much text follows — degradation shows up as the delta.
The passage is original prose (not LibriTTS text), digit-free, phonemized
here with the same op_g2p lane used to build the training filelists.

Token counts are post-intersperse (2*ipa+1), the same unit as the litert
MAX_TEXT=256 ceiling and the corpus-measured 37.4 tokens/s speaking rate.

Fully CPU — same throwaway-container recipe as the gate watchers:
    uv pip install soundfile librosa pyloudnorm faster-whisper ai-edge-litert

Usage:
    python scripts/chunk_size_sweep.py --checkpoint <ckpt> --out <dir>
        [--buckets 128,256,512,1024,2048,4096] [--speakers 245,65,93]
"""

import argparse
import json
import math
import os
import sys

HIFIGAN = os.environ.get("SONORA_HIFIGAN_DIR", "/data/model-training/vocoder/hifi-gan")
sys.path.insert(0, HIFIGAN)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import torch

# Original prose composed for this sweep (public-domain-equivalent: written
# for this repo). Constraints: digit-free (op_g2p does not expand digits),
# plain vocabulary, varied sentence lengths, coherent narrative so the human
# audit hears natural long-form reading rather than a sentence salad.
SENTENCES = [
    "The lighthouse keeper woke before dawn, as he had done every morning for thirty years, give or take the mornings he preferred not to count.",
    "He put the kettle on the small iron stove and watched the window while the water warmed.",
    "Outside, the sea was the color of slate, and the gulls hung in the wind like scraps of paper.",
    "There was a ship due in from the south that week, and he liked to have the lamp cleaned early when a ship was due.",
    "He carried his tea up the spiral stair, pausing at the middle landing where the wall held a small square window no bigger than a book.",
    "From that window he could see the whole curve of the bay, the fishing boats still dark at their moorings, and the town beyond them just beginning to show its lights.",
    "The lamp room smelled of oil and brass polish, a smell he had long ago stopped noticing except on cold mornings, when it seemed sharper.",
    "He worked slowly and without hurry, wiping each pane of the great lens until it threw small rainbows on the floor.",
    "People in the town believed the work must be lonely, and they said so whenever he came down for supplies.",
    "He had given up explaining that the light itself was company, that a thing which needed you every single day could not help but become a companion.",
    "By the time the sun cleared the headland, the lamp was ready, the log was written, and the kettle was cold again downstairs.",
    "He made a second pot and took it out to the rail, where the wind had softened and the slate sea had turned to hammered silver.",
    "The morning boat from the town came out on Tuesdays, bringing bread, letters, and whatever news the postmaster thought worth repeating.",
    "This particular Tuesday it brought a passenger as well, a young woman in a gray coat who stepped onto the landing with the ease of somebody who had done it before.",
    "She was the schoolteacher's daughter, home from the city, and she had come to ask him about the storm of nineteen years past, for a book she said she was writing.",
    "He remembered the storm the way other men remember a war, which is to say completely, and in an order all its own.",
    "They sat in the kitchen while the wind picked at the shutters, and he told it from the beginning, from the strange yellow calm of that afternoon to the first long swell that came in without any wind behind it.",
    "The barometer had fallen so fast he thought the instrument was broken, and he tapped the glass twice before he believed it.",
    "By dark the sea was standing up in ranks against the tower, and every seventh wave or so came aboard the rock entire, green and heavy, and pulled at the door like a living thing.",
    "He kept the lamp burning through all of it, not out of bravery, he said, but because keeping it burning was simpler than deciding anything else.",
    "Somewhere in the worst hour a schooner had gone past to the north, running blind, and he had watched her lights appear and vanish between the seas.",
    "She made the harbor, he told the young woman, though her captain never afterward could say exactly how.",
    "The young woman wrote quickly in a notebook, stopping now and then to ask him what a word meant, and he found that he enjoyed the asking more than the telling.",
    "When the boat came back for her that evening, she promised to send him a copy of the book if it ever became one.",
    "He said he would put it on the shelf beside the almanac, which was the only other book to survive his housekeeping, and she laughed and said the almanac would be glad of the company.",
    "The winter that followed was a quiet one, mild and gray, with fogs that lasted for days and made the horn more useful than the light.",
    "He minded the fog less than most keepers did, since the horn had a rhythm to it, and a man can live with anything that has a rhythm.",
    "In the spring a crew came out to look at the tower, engineers from the mainland with instruments in leather cases and a great many opinions.",
    "They measured the crack above the storeroom door, which had been there longer than he had, and pronounced it stable, which he could have told them for considerably less money.",
    "One of them, the youngest, stayed an extra hour to see the lamp lit, and went down the stair afterward without saying much, which the keeper took as the highest compliment the profession allowed.",
    "The book arrived in the autumn, wrapped in brown paper, with the storm in the middle chapter and his own name spelled correctly, which surprised him.",
    "She had written it plainly and gotten the sea right, which almost nobody did, and he read the chapter three times the first night.",
    "He wrote her a letter saying so, the first letter he had written in a decade, and the writing of it took him the better part of an evening.",
    "Years later, when the light was changed over to run itself, and the service pensioned him ashore, he took the book, the almanac, and the small square window's view in his memory, and very little else.",
    "The cottage they gave him in the town looked away from the sea, which he suspected was somebody's idea of kindness.",
    "He planted a garden because his neighbor insisted, and found to his annoyance that he liked it.",
    "Cabbages, it turned out, also needed you every single day, and were similarly poor conversationalists.",
    "On clear evenings he walked to the harbor wall and watched his old light come on across the water, exactly on time, with nobody in it.",
    "It burned as well as ever, he admitted, though he privately held that it burned without style.",
    "The schoolteacher's daughter visited when she came home, bringing new books and arguments, and they drank tea in the garden among the disapproving cabbages.",
    "She was working on a longer book now, about the whole coast, and she wanted the names of every keeper he had ever shared a watch with.",
    "He gave her the names slowly, one at a time, each with its boat and its station and its particular weather, and she wrote them all down.",
    "Some of the names had no one else left to say them, and they both understood that this was the true business of the afternoon.",
    "When the light in the tower failed one night in his ninth year ashore, the whole town knew within the hour, and a boat went out in heavy weather to see to it.",
    "They found a relay burned through, replaced it, and were home by morning, and the newspaper gave the event four sentences.",
    "He read the four sentences several times and then went out to the harbor wall, although it was raining, and stood there until the beam swung around again.",
    "Habits of attention, he told the schoolteacher's daughter afterward, do not retire when you do.",
    "She put that sentence in her book, in the last chapter, and this time she sent him two copies, one to read and one to keep clean.",
    "He read them both, naturally, because a book which needed him twice was twice the company.",
    "The garden did well that year, the mild year, and the almanac was right about the frosts, and the light across the water kept its rhythm every night, all night, whether anyone watched it or not.",
]

CKPT_DEFAULT = ("/data/model-training/sonora/logs/train/derisk_energy/runs/"
                "2026-07-15_00-20-31/checkpoints/checkpoint_epoch=099.ckpt")


def load_acoustic(path):
    from matcha.models.matcha_tts import MatchaTTS

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MatchaTTS(**dict(ckpt["hyper_parameters"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def load_vocoder(ckpt_path, config_path):
    from env import AttrDict
    from models import Generator

    h = AttrDict(json.load(open(config_path)))
    g = Generator(h)
    g.load_state_dict(torch.load(ckpt_path, map_location="cpu")["generator"])
    g.eval()
    g.remove_weight_norm()
    return g, h


def build_prefixes(g2p, targets):
    """Cumulative sentence prefixes hitting each post-intersperse token target."""
    from matcha.text import text_to_sequence

    ipa_cache = [g2p.phonemize(s) for s in SENTENCES]
    for i, ipa in enumerate(ipa_cache):
        bad = g2p.validate(ipa)
        if bad:
            raise SystemExit(f"sentence {i} has out-of-vocab chars: {bad}")

    prefixes = []
    for target in sorted(targets):
        text_parts, ipa_parts, tokens = [], [], 0
        for sent, ipa in zip(SENTENCES, ipa_cache):
            text_parts.append(sent)
            ipa_parts.append(ipa)
            seq, _ = text_to_sequence(" ".join(ipa_parts), ["no_cleaners"])
            tokens = 2 * len(seq) + 1  # post-intersperse, the ceiling's unit
            if tokens >= target:
                break
        if tokens < target:
            raise SystemExit(f"passage too short for target {target} "
                             f"(reached {tokens} tokens)")
        prefixes.append({"target": target, "tokens": tokens,
                         "text": " ".join(text_parts),
                         "ipa": " ".join(ipa_parts),
                         "n_sentences": len(text_parts)})
    return prefixes


def rms_db_thirds(wav):
    thirds = np.array_split(wav.astype(np.float64), 3)
    return [round(20 * math.log10(max(np.sqrt((t ** 2).mean()), 1e-9)), 2)
            for t in thirds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocoder",
                    default="/data/model-training/vocoder/cp_hifigan_24k/g_02510000")
    ap.add_argument("--vocoder-config",
                    default=os.path.join(HIFIGAN, "config_24k_80band.json"))
    ap.add_argument("--buckets", default="128,256,512,1024,2048,4096",
                    help="post-intersperse token targets (256 = litert ceiling)")
    ap.add_argument("--speakers", default="245,65,93")
    ap.add_argument("--n-timesteps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.667)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-wer", action="store_true")
    args = ap.parse_args()

    from matcha.text import text_to_sequence
    from matcha.text.op_g2p import OpenPhonemizerG2P
    from matcha.utils.utils import intersperse

    targets = [int(t) for t in args.buckets.split(",")]
    speakers = [int(s) for s in args.speakers.split(",")]

    g2p = OpenPhonemizerG2P()
    prefixes = build_prefixes(g2p, targets)
    for p in prefixes:
        print(f"bucket {p['target']}: {p['tokens']} tokens, "
              f"{p['n_sentences']} sentences, {len(p['text'].split())} words")

    model = load_acoustic(args.checkpoint)
    vocoder, h = load_vocoder(args.vocoder, args.vocoder_config)
    os.makedirs(args.out, exist_ok=True)

    whisper = None
    if not args.skip_wer:
        from eval_harness import Whisper, wer
        whisper = Whisper()

    fps = h.sampling_rate / 256.0  # mel frames per second (hop 256)
    rows = []
    for spk_id in speakers:
        spks = torch.tensor([spk_id], dtype=torch.long)
        for p in prefixes:
            seq, _ = text_to_sequence(p["ipa"], ["no_cleaners"])
            x = torch.tensor(intersperse(seq, 0), dtype=torch.long)[None]
            x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long)
            torch.manual_seed(args.seed)
            with torch.no_grad():
                out = model.synthesise(x, x_lengths, n_timesteps=args.n_timesteps,
                                       temperature=args.temperature, spks=spks,
                                       length_scale=args.length_scale,
                                       vat=torch.zeros(1, 3))
                wav = vocoder(out["mel"]).squeeze().numpy()
            dur = len(wav) / h.sampling_rate
            fname = f"chunk_spk{spk_id}_t{p['target']}.wav"
            sf.write(os.path.join(args.out, fname), wav, h.sampling_rate)
            row = {"wav": fname, "spk": spk_id, "target": p["target"],
                   "tokens": p["tokens"], "n_sentences": p["n_sentences"],
                   "words": len(p["text"].split()),
                   "mel_frames": int(out["mel"].shape[-1]),
                   "seconds": round(dur, 2),
                   "tokens_per_sec": round(p["tokens"] / dur, 2),
                   "rms_db_thirds": rms_db_thirds(wav)}
            if whisper:
                hyp = whisper.transcribe(os.path.join(args.out, fname))
                row["wer"] = round(wer(p["text"], hyp), 4)
            rows.append(row)
            print(f"  spk {spk_id} t={p['target']}: {dur:.1f}s "
                  f"{row.get('wer', 'n/a')} WER  rms/3rds {row['rms_db_thirds']}")

    report = {"checkpoint": args.checkpoint, "vocoder": args.vocoder,
              "buckets": targets, "speakers": speakers, "seed": args.seed,
              "n_timesteps": args.n_timesteps, "temperature": args.temperature,
              "length_scale": args.length_scale,
              "note": ("post-intersperse tokens; 256 = litert MAX_TEXT ceiling; "
                       "WER on full render vs full reference text"),
              "rows": rows}
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'spk':>4} {'tokens':>7} {'secs':>7} {'tok/s':>6} {'WER':>7}  rms thirds")
    for r in rows:
        print(f"{r['spk']:>4} {r['tokens']:>7} {r['seconds']:>7.1f} "
              f"{r['tokens_per_sec']:>6.1f} {str(r.get('wer', 'n/a')):>7}  "
              f"{r['rms_db_thirds']}")
    print(f"\nreport + {len(rows)} WAVs -> {args.out}")


if __name__ == "__main__":
    main()
