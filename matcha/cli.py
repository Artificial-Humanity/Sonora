import argparse
import datetime as dt
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

# Configure safe globals for PyTorch >= 2.6 loading older Lightning checkpoints
if hasattr(torch.serialization, "add_safe_globals"):
    try:
        import omegaconf
        import collections
        torch.serialization.add_safe_globals([
            omegaconf.dictconfig.DictConfig,
            omegaconf.listconfig.ListConfig,
            omegaconf.nodes.AnyNode,
            omegaconf.nodes.StringNode,
            omegaconf.nodes.IntegerNode,
            omegaconf.nodes.FloatNode,
            omegaconf.nodes.BooleanNode,
            omegaconf.base.ContainerMetadata,
            omegaconf.base.Node,
            collections.defaultdict,
        ])
    except ImportError:
        pass

from matcha import delivery
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir, intersperse

MATCHA_URLS = {
    "matcha_ljspeech": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt",
    "matcha_vctk": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_vctk.ckpt",
}

VOCODER_URLS = {
    "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",  # Old url: https://drive.google.com/file/d/14NENd4equCBLyyCSke114Mv6YR_j_uFs/view?usp=drive_link
    "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",  # Old url: https://drive.google.com/file/d/1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW/view?usp=drive_link
}

MULTISPEAKER_MODEL = {
    "matcha_vctk": {"vocoder": "hifigan_univ_v1", "speaking_rate": 0.85, "spk": 0, "spk_range": (0, 107)}
}

SINGLESPEAKER_MODEL = {"matcha_ljspeech": {"vocoder": "hifigan_T2_v1", "speaking_rate": 0.95, "spk": None}}


def plot_spectrogram_to_numpy(spectrogram, filename):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title("Synthesised Mel-Spectrogram")
    fig.canvas.draw()
    plt.savefig(filename)


def process_text(i: int, text: str, device: torch.device):
    """LEGACY LANE ONLY — espeak via `english_cleaners2`.

    Kept for the pre-VAT LJSpeech checkpoints, which were trained on espeak phonemes and
    are only auditable through them. Everything since `derisk_energy` trains on op_g2p
    IPA against the locked 178-symbol vocab, so calling this on a Sonora checkpoint feeds
    it a phoneme distribution it never saw. Use `process_text_for_lane`, which picks.

    Needs the `espeak` extra (GPL `phonemizer`), deliberately not installed by default.
    """
    print(f"[{i}] - Input text: {text}")
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    print(f"[{i}] - Phonetised text: {x_phones[1::2]}")

    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}


_G2P = None


def get_g2p():
    """Cached op_g2p. Construction loads a 275k dictionary and a TFLite graph."""
    global _G2P
    if _G2P is None:
        from matcha.text.op_g2p import OpenPhonemizerG2P

        _G2P = OpenPhonemizerG2P()
    return _G2P


def detect_lane(checkpoint_path):
    """Read the text/vocoder lane off the checkpoint itself.

    Returns `(lane, n_spks, vat_dim)` with lane in {"vat", "legacy"}.

    THIS LIVES HERE SO THERE IS EXACTLY ONE COPY. vocalizer.py imports it rather than
    keeping its own — the review's recurring finding is two implementations of one rule
    drifting apart (B-L5's duplicated MAX_REF_EXCURSION, D-L2's two disagreeing z-score
    guards), and "which phonemes does this checkpoint expect" is precisely the rule that
    must not fork: getting it wrong produces confident, fluent, wrong audio rather than
    an error.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt.get("hyper_parameters") or {})
    has_vat = any("vat" in k or "film" in k for k in ckpt["state_dict"])
    n_spks = int(hp.get("n_spks") or 1)
    vat_dim = int(hp.get("vat_dim") or 0)
    return ("vat" if (has_vat or n_spks > 1) else "legacy"), n_spks, vat_dim


def process_text_for_lane(i: int, text: str, device: torch.device, lane: str):
    """Text -> ids, phonemized the way the loaded checkpoint was trained."""
    if lane != "vat":
        return process_text(i, text, device)

    print(f"[{i}] - Input text: {text}")
    # Partial D-M3, at the one boundary this function owns. The tokenizer DELETES digits
    # rather than expanding them: "I have 3 cats" phonemizes to "aɪ hæv kæts" and
    # synthesises cleanly, so the defect is inaudible in the output and invisible in the
    # logs. `g2p.validate()` does not catch it — the result contains no illegal symbols,
    # it is simply missing a word. Refusing here costs nothing and removes the silent
    # case. NOTE: this does NOT fix D-M3, which is live in the corpus derivation lane
    # (Emilia YODAS captions carry digits); see notes/training-sources.md, the D-M3 rows.
    digits = sorted({c for c in text if c.isdigit()})
    if digits:
        raise ValueError(
            f"text contains digits {digits}, which this lane silently DROPS rather than "
            "expanding — write them out as words (\"three\", not \"3\")"
        )
    g2p = get_g2p()
    ipa = g2p.phonemize(text)
    bad = g2p.validate(ipa)
    if bad:
        # Loud, because the alternative is silent corruption: unknown symbols would map
        # to whatever the vocab does with them and still synthesise something.
        #
        # ValueError, NOT SystemExit: this function is shared with the Vocalizer's HTTP
        # API, and SystemExit is a BaseException that `except Exception` does not catch —
        # one bad character in a request would have taken the uvicorn worker down instead
        # of returning an error. `cli()` converts it to a clean exit for terminal use.
        raise ValueError(
            f"out-of-vocab characters after G2P: {bad} "
            "(digits are NOT expanded on this lane — write numbers out as words)"
        )
    seq, _ = text_to_sequence(ipa, ["no_cleaners"])
    x = torch.tensor(intersperse(seq, 0), dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    print(f"[{i}] - Phonetised text: {ipa}")
    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": ipa}


def get_texts(args):
    if args.text:
        texts = [args.text]
    else:
        with open(args.file, encoding="utf-8") as f:
            texts = f.readlines()
    return texts


def assert_required_models_available(args):
    save_dir = get_user_data_dir()
    # E-H2. This read `not hasattr(args, "checkpoint_path") and args.checkpoint_path is
    # None`, which argparse makes DEAD: the attribute always exists, so the first
    # conjunct is always False and the else branch always ran. `--checkpoint_path
    # /data/.../vat3_ep099.ckpt` therefore still downloaded the upstream LJSpeech
    # checkpoint over the network before ignoring it. (Had the attribute somehow been
    # missing, the second conjunct would have raised AttributeError reading it — the
    # condition is wrong in both directions.)
    if args.checkpoint_path is not None:
        model_path = Path(args.checkpoint_path)
    else:
        model_path = save_dir / f"{args.model}.ckpt"
        assert_model_downloaded(model_path, MATCHA_URLS[args.model])

    # Same for the vocoder: a local one is a path, not a name in the download table.
    if getattr(args, "vocoder_path", None):
        vocoder_path = Path(args.vocoder_path)
        if not vocoder_path.exists():
            raise SystemExit(f"[!] no vocoder at {vocoder_path}")
    else:
        vocoder_path = save_dir / f"{args.vocoder}"
        assert_model_downloaded(vocoder_path, VOCODER_URLS[args.vocoder])
    return {"matcha": model_path, "vocoder": vocoder_path}


def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def load_vocoder(vocoder_name, checkpoint_path, device):
    print(f"[!] Loading {vocoder_name}!")
    vocoder = None
    if vocoder_name in ("hifigan_T2_v1", "hifigan_univ_v1"):
        vocoder = load_hifigan(checkpoint_path, device)
    else:
        raise NotImplementedError(
            f"Vocoder {vocoder_name} not implemented! define a load_<<vocoder_name>> method for it"
        )

    denoiser = Denoiser(vocoder, mode="zeros")
    print(f"[+] {vocoder_name} loaded!")
    return vocoder, denoiser


# Where the promoted 24 kHz fine-tune lives. Overridable so the eval container and the
# Vocalizer agree without either hardcoding a second copy of the path.
VOC24K_CKPT = os.environ.get("SONORA_VOC24K", "/data/model-training/vocoder/cp_hifigan_24k/g_02510000")
VOC24K_CONFIG = os.environ.get(
    "SONORA_VOC24K_CONFIG", "/data/model-training/vocoder/cp_hifigan_24k/config.json"
)


def load_vocoder_24k(device=torch.device("cpu")):
    """The Sonora 24 kHz HiFi-GAN. Returns `(vocoder, sampling_rate)`.

    Separate from `load_vocoder` because this one carries its own config JSON rather than
    the hardcoded upstream `v1` dict — the band count and sampling rate differ, and
    loading a 24 kHz checkpoint under the 22.05 kHz config is the kind of mistake that
    yields audio rather than an exception.

    No denoiser: it was fitted against the upstream vocoder's bias and confirmed
    unnecessary here (this vocoder is perceptually transparent — mel L1 = 10.2% of one
    mel_std, measured 2026-08-06 by copy-synthesis).
    """
    import json

    with open(VOC24K_CONFIG, encoding="utf-8") as f:
        h = AttrDict(json.load(f))
    g = HiFiGAN(h).to(device)
    g.load_state_dict(torch.load(VOC24K_CKPT, map_location=device)["generator"])
    g.eval()
    g.remove_weight_norm()
    return g, h.sampling_rate


def load_matcha(model_name, checkpoint_path, device):
    print(f"[!] Loading {model_name}!")
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device, weights_only=False)
    _ = model.eval()

    print(f"[+] {model_name} loaded!")
    return model


def to_waveform(mel, vocoder, denoiser=None, denoiser_strength=0.00025):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser is not None:
        audio = denoiser(audio.squeeze(), strength=denoiser_strength).cpu().squeeze()

    return audio.cpu().squeeze()


def save_to_folder(filename: str, output: dict, folder: str, sample_rate: int = 22050):
    # `sample_rate` was hardcoded 22050. The Sonora lane is 24 kHz, so every file it wrote
    # was tagged 22.05 kHz and played back ~9% slow and flat — audible, but easy to hear
    # as "the model sounds sluggish" rather than as a header bug.
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    plot_spectrogram_to_numpy(np.array(output["mel"].squeeze().float().cpu()), f"{filename}.png")
    np.save(folder / f"{filename}", output["mel"].cpu().numpy())
    sf.write(folder / f"{filename}.wav", output["waveform"], sample_rate, "PCM_24")
    return folder.resolve() / f"{filename}.wav"


def synthesis_conditioning(args, device):
    """The VAT-lane kwargs for `model.synthesise`, empty on the legacy lane."""
    if args.lane != "vat":
        return {}
    kwargs = {"guidance": float(args.guidance)}
    if args.vat_vector is not None:
        kwargs["vat"] = torch.tensor([args.vat_vector], dtype=torch.float32, device=device)
    return kwargs


def parse_vat(args):
    """`--vat V,A,T` (+ `--delivery LANE`) -> a validated list, or None on the legacy lane.

    Contract v2 made the conditioning vector V/A/T followed by a one-hot delivery block.
    `--vat` stays THREE numbers and `--delivery` names a lane, because the alternative is
    asking a human to type five zeros in the right order — and a delivery block typed by
    hand is a delivery block that will one day be typed wrong, silently, into a channel
    whose failure mode is a fluent render in the wrong manner.

    Passing the full width is still accepted: the export lane and the seam tests drive
    raw vectors, and refusing them would mean two ways to describe one tensor.
    """
    if args.vat is None and not getattr(args, "delivery", None):
        return None
    if args.lane != "vat":
        raise SystemExit("[!] --vat needs a VAT checkpoint; this one has no conditioning trunk")

    values = [0.0, 0.0, 0.0]
    if args.vat is not None:
        try:
            values = [float(v) for v in args.vat.split(",")]
        except ValueError:
            raise SystemExit(f"[!] --vat wants comma-separated numbers, got {args.vat!r}") from None

    width = args.vat_dim or delivery.VAT_DIM
    lane = (getattr(args, "delivery", None) or "").strip()

    if len(values) == delivery.VAT_BASE_DIM and width > delivery.VAT_BASE_DIM:
        # The ordinary path: three numbers plus a lane name.
        try:
            values = delivery.vat_vector(*values, lane)
        except ValueError as exc:
            raise SystemExit(f"[!] {exc}") from None
    elif lane:
        raise SystemExit(
            f"[!] --delivery cannot be combined with a full {len(values)}-channel --vat: "
            "the vector already carries a delivery block, and there would be two answers "
            "to which lane this render is in.")

    if len(values) != width:
        raise SystemExit(f"[!] this checkpoint takes {width} VAT channels, got {len(values)}")

    # Same bound the Vocalizer's sliders and its HTTP API enforce. Out-of-range values do
    # not synthesise "more" of the channel; they drive the FiLM trunk off the manifold and
    # the result is confident, fluent and wrong. The delivery block is checked separately
    # — it is CATEGORICAL, so [-1, 1] is the wrong question and 0.5 is not a half-lane.
    out = [v for v in values[:delivery.VAT_BASE_DIM] if not -1.0 <= v <= 1.0]
    if out:
        raise SystemExit(f"[!] VAT values must be within [-1, 1]; out of range: {out}")
    if width > delivery.VAT_BASE_DIM:
        try:
            delivery.lane_of_vector(values)
        except ValueError as exc:
            raise SystemExit(f"[!] {exc}") from None
    return values


def validate_args(args):
    assert (
        args.text or args.file
    ), "Either text or file must be provided Matcha-T(ea)TTS need sometext to whisk the waveforms."
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.steps > 0, "Number of ODE steps must be greater than 0"

    if args.checkpoint_path is None:
        # When using pretrained models
        if args.model in SINGLESPEAKER_MODEL:
            args = validate_args_for_single_speaker_model(args)

        if args.model in MULTISPEAKER_MODEL:
            args = validate_args_for_multispeaker_model(args)
    else:
        # When using a custom model. The VAT lane picks its own 24 kHz vocoder from the
        # checkpoint, so the "did you mean univ_v1?" nudge is noise there.
        if args.lane != "vat" and args.vocoder != "hifigan_univ_v1":
            warn_ = "[-] Using custom model checkpoint! I would suggest passing --vocoder hifigan_univ_v1, unless the custom model is trained on LJ Speech."
            warnings.warn(warn_, UserWarning)
        if args.speaking_rate is None:
            args.speaking_rate = 1.0

    if args.batched:
        assert args.batch_size > 0, "Batch size must be greater than 0"
    assert args.speaking_rate > 0, "Speaking rate must be greater than 0"

    return args


def validate_args_for_multispeaker_model(args):
    if args.vocoder is not None:
        if args.vocoder != MULTISPEAKER_MODEL[args.model]["vocoder"]:
            warn_ = f"[-] Using {args.model} model! I would suggest passing --vocoder {MULTISPEAKER_MODEL[args.model]['vocoder']}"
            warnings.warn(warn_, UserWarning)
    else:
        args.vocoder = MULTISPEAKER_MODEL[args.model]["vocoder"]

    if args.speaking_rate is None:
        args.speaking_rate = MULTISPEAKER_MODEL[args.model]["speaking_rate"]

    spk_range = MULTISPEAKER_MODEL[args.model]["spk_range"]
    if args.spk is not None:
        assert (
            args.spk >= spk_range[0] and args.spk <= spk_range[-1]
        ), f"Speaker ID must be between {spk_range} for this model."
    else:
        available_spk_id = MULTISPEAKER_MODEL[args.model]["spk"]
        warn_ = f"[!] Speaker ID not provided! Using speaker ID {available_spk_id}"
        warnings.warn(warn_, UserWarning)
        args.spk = available_spk_id

    return args


def validate_args_for_single_speaker_model(args):
    if args.vocoder is not None:
        if args.vocoder != SINGLESPEAKER_MODEL[args.model]["vocoder"]:
            warn_ = f"[-] Using {args.model} model! I would suggest passing --vocoder {SINGLESPEAKER_MODEL[args.model]['vocoder']}"
            warnings.warn(warn_, UserWarning)
    else:
        args.vocoder = SINGLESPEAKER_MODEL[args.model]["vocoder"]

    if args.speaking_rate is None:
        args.speaking_rate = SINGLESPEAKER_MODEL[args.model]["speaking_rate"]

    if args.spk != SINGLESPEAKER_MODEL[args.model]["spk"]:
        warn_ = f"[-] Ignoring speaker id {args.spk} for {args.model}"
        warnings.warn(warn_, UserWarning)
        args.spk = SINGLESPEAKER_MODEL[args.model]["spk"]

    return args


@torch.inference_mode()
def cli():
    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="matcha_ljspeech",
        help="Model to use",
        choices=MATCHA_URLS.keys(),
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the custom model checkpoint",
    )

    parser.add_argument(
        "--vocoder",
        type=str,
        default=None,
        help="Vocoder to use (default: will use the one suggested with the pretrained model))",
        choices=VOCODER_URLS.keys(),
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--spk", type=int, default=None, help="Speaker ID")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )
    parser.add_argument(
        "--speaking_rate",
        type=float,
        default=None,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of ODE steps  (default: 10)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "--denoiser_strength",
        type=float,
        default=0.00025,
        help="Strength of the vocoder bias denoiser (default: 0.00025)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )
    parser.add_argument("--batched", action="store_true", help="Batched inference (default: False)")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size only useful when --batched (default: 32)"
    )
    # --- Sonora lane (E-H2) -------------------------------------------------------
    parser.add_argument(
        "--vocoder_path",
        type=str,
        default=None,
        help="Path to a local vocoder checkpoint. Skips the download table entirely; "
        "implied for VAT checkpoints, which need the 24 kHz fine-tune",
    )
    parser.add_argument(
        "--vat",
        type=str,
        default=None,
        metavar="V,A,T",
        help="Conditioning as comma-separated floats in [-1, 1], e.g. --vat 0.4,-0.2,0.0 "
        "(per-speaker z-scores clamped at 2 sigma in derivation, so beyond +/-1 is "
        "outside the trained range). Default: zeros, i.e. neutral. Name the delivery "
        "lane with --delivery rather than appending its one-hot block by hand.",
    )
    parser.add_argument(
        "--delivery",
        type=str,
        default=None,
        choices=list(delivery.DELIVERY_LANES),
        metavar="LANE",
        help="Delivery lane (contract v2): "
        + " | ".join(delivery.DELIVERY_LANES)
        + ". Omit for `unknown`, which is the all-zero block and renders exactly as a "
        "v1 checkpoint did. Embodiment clips are deliberately unknown, not a sixth lane.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (1.0 = off). Above 1 wants --steps 25+",
    )

    args = parser.parse_args()

    # Which lane, decided by the checkpoint rather than by a flag the caller can get
    # wrong. A VAT checkpoint gets op_g2p phonemes and the 24 kHz vocoder; a legacy
    # LJSpeech one keeps espeak and 22.05 kHz. Before this, everything got espeak and
    # 22.05 kHz, so every Sonora checkpoint was fed phonemes it had never been trained
    # on and then written out ~9% slow.
    args.lane, args.n_spks, args.vat_dim = "legacy", 1, 0
    if args.checkpoint_path is not None:
        args.lane, args.n_spks, args.vat_dim = detect_lane(args.checkpoint_path)
        if args.lane == "vat" and args.vocoder is None and args.vocoder_path is None:
            args.vocoder_path = VOC24K_CKPT
        elif args.vocoder is None and args.vocoder_path is None:
            # Custom legacy checkpoint and no vocoder named: keep upstream's default
            # rather than leaving None to fall through the download table as a filename.
            args.vocoder = "hifigan_T2_v1"
    # When --checkpoint_path is absent the pretrained-model validators below choose the
    # matching vocoder, so leave args.vocoder alone here.

    args = validate_args(args)
    args.vat_vector = parse_vat(args)
    device = get_device(args)
    print_config(args)
    paths = assert_required_models_available(args)

    if args.checkpoint_path is not None:
        print(f"[🍵] Loading custom model from {args.checkpoint_path}")
        paths["matcha"] = args.checkpoint_path
        args.model = "custom_model"

    model = load_matcha(args.model, paths["matcha"], device)
    if args.lane == "vat":
        vocoder, args.sample_rate = load_vocoder_24k(device)
        denoiser = None
        print(f"[+] Sonora lane: {args.n_spks} speakers, vat_dim={args.vat_dim}, "
              f"{args.sample_rate} Hz")
    else:
        vocoder, denoiser = load_vocoder(args.vocoder, paths["vocoder"], device)
        args.sample_rate = 22050

    texts = get_texts(args)

    if args.spk is not None and args.n_spks > 1 and not 0 <= args.spk < args.n_spks:
        raise SystemExit(f"[!] --spk {args.spk} is outside this checkpoint's 0..{args.n_spks - 1}")
    spk = torch.tensor([args.spk], device=device, dtype=torch.long) if args.spk is not None else None
    try:
        if len(texts) == 1 or not args.batched:
            unbatched_synthesis(args, device, model, vocoder, denoiser, texts, spk)
        else:
            batched_synthesis(args, device, model, vocoder, denoiser, texts, spk)
    except ValueError as e:
        # G2P vocab failures surface as ValueError so the shared path stays usable from a
        # long-running server; a terminal user wants a message, not a traceback.
        raise SystemExit(f"[!] {e}") from None


class BatchedSynthesisDataset(torch.utils.data.Dataset):
    def __init__(self, processed_texts):
        self.processed_texts = processed_texts

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        return self.processed_texts[idx]


def batched_collate_fn(batch):
    x = []
    x_lengths = []

    for b in batch:
        x.append(b["x"].squeeze(0))
        x_lengths.append(b["x_lengths"])

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x_lengths = torch.concat(x_lengths, dim=0)
    return {"x": x, "x_lengths": x_lengths}


def batched_synthesis(args, device, model, vocoder, denoiser, texts, spk):
    total_rtf = []
    total_rtf_w = []
    processed_text = [process_text_for_lane(i, text, "cpu", args.lane) for i, text in enumerate(texts)]
    dataloader = torch.utils.data.DataLoader(
        BatchedSynthesisDataset(processed_text),
        batch_size=args.batch_size,
        collate_fn=batched_collate_fn,
        num_workers=8,
    )
    for i, batch in enumerate(dataloader):
        i = i + 1
        start_t = dt.datetime.now()
        b = batch["x"].shape[0]
        output = model.synthesise(
            batch["x"].to(device),
            batch["x_lengths"].to(device),
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk.expand(b) if spk is not None else spk,
            length_scale=args.speaking_rate,
            **synthesis_conditioning(args, device),
        )

        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser, args.denoiser_strength)
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * args.sample_rate / (output["waveform"].shape[-1])
        print(f"[🍵-Batch: {i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-Batch: {i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)
        for j in range(output["mel"].shape[0]):
            base_name = f"utterance_{j:03d}_speaker_{args.spk:03d}" if args.spk is not None else f"utterance_{j:03d}"
            length = output["mel_lengths"][j]
            new_dict = {"mel": output["mel"][j][:, :length], "waveform": output["waveform"][j][: length * 256]}
            location = save_to_folder(base_name, new_dict, args.output_folder, args.sample_rate)
            print(f"[🍵-{j}] Waveform saved: {location}")

    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")


def unbatched_synthesis(args, device, model, vocoder, denoiser, texts, spk):
    total_rtf = []
    total_rtf_w = []
    for i, text in enumerate(texts):
        i = i + 1
        base_name = f"utterance_{i:03d}_speaker_{args.spk:03d}" if args.spk is not None else f"utterance_{i:03d}"

        print("".join(["="] * 100))
        text = text.strip()
        text_processed = process_text_for_lane(i, text, device, args.lane)

        print(f"[🍵] Whisking Matcha-T(ea)TS for: {i}")
        start_t = dt.datetime.now()
        output = model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk,
            length_scale=args.speaking_rate,
            **synthesis_conditioning(args, device),
        )
        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser, args.denoiser_strength)
        # RTF with HiFiGAN
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * args.sample_rate / (output["waveform"].shape[-1])
        print(f"[🍵-{i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-{i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)

        location = save_to_folder(base_name, output, args.output_folder, args.sample_rate)
        print(f"[+] Waveform saved: {location}")

    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")


def print_config(args):
    print("[!] Configurations: ")
    print(f"\t- Model: {args.model}")
    print(f"\t- Vocoder: {args.vocoder}")
    print(f"\t- Temperature: {args.temperature}")
    print(f"\t- Speaking rate: {args.speaking_rate}")
    print(f"\t- Number of ODE steps: {args.steps}")
    print(f"\t- Speaker: {args.spk}")


def get_device(args):
    if torch.cuda.is_available() and not args.cpu:
        print("[+] GPU Available! Using GPU")
        device = torch.device("cuda")
    else:
        print("[-] GPU not available or forced CPU run! Using CPU")
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    cli()
