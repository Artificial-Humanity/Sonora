"""Sonora Vocalizer: browser audition bench for any training checkpoint.

The standing vetting surface for the model (owner call, 2026-07-16): every new
model capability or control ships with a dial here in the same phase, so
outputs stay auditable by ear at the current feature set.

Two checkpoint lanes, detected from the checkpoint itself:

* VAT/multi-speaker (derisk and later): op_g2p phonemes (no_cleaners, same
  lane as the training filelists), speaker id + V/A/T conditioning, vocoded
  with the promoted 24 kHz HiFi-GAN fine-tune.
* legacy LJSpeech: espeak text lane (matcha.cli.process_text), single
  speaker, hifigan_T2_v1 at 22.05 kHz — kept so old checkpoints stay
  auditable.

CPU-only on purpose: training owns the GPU.
"""

import os
import glob
import argparse
import tempfile
import io
from pathlib import Path
import starlette.templating
original_template_response = starlette.templating.Jinja2Templates.TemplateResponse
def patched_template_response(self, *args, **kwargs):
    if args and isinstance(args[0], str):
        name = args[0]
        context = args[1] if len(args) > 1 else {}
        request = context.get("request")
        new_args = (request, name, context) + args[2:]
        return original_template_response(self, *new_args, **kwargs)
    return original_template_response(self, *args, **kwargs)
starlette.templating.Jinja2Templates.TemplateResponse = patched_template_response

import gradio as gr
import gradio.networking
gradio.networking.url_ok = lambda *args, **kwargs: True
import json
import soundfile as sf
import torch

from matcha.cli import (
    get_device,
    load_matcha,
    load_vocoder,
    process_text,
    to_waveform,
)
from matcha.text import text_to_sequence
from matcha.utils.utils import intersperse, plot_tensor

# We run on CPU to avoid GPU conflicts with the training run
device = torch.device("cpu")

VOC24K_CKPT = os.environ.get(
    "SONORA_VOC24K", "/data/model-training/vocoder/cp_hifigan_24k/g_02510000")
VOC24K_CONFIG = os.environ.get(
    "SONORA_VOC24K_CONFIG",
    "/data/model-training/vocoder/hifi-gan/config_24k_80band.json")

# Loaded-state registry (one checkpoint + its lane's vocoder at a time)
current_checkpoint = None
model = None
vocoder = None
denoiser = None
lane = None          # "vat" | "legacy"
n_spks = 1
sample_rate = 22050
_g2p = None


def get_g2p():
    global _g2p
    if _g2p is None:
        from matcha.text.op_g2p import OpenPhonemizerG2P
        _g2p = OpenPhonemizerG2P()
    return _g2p


def get_checkpoints():
    # Scan for all ckpt files in checkpoints subfolders without doing a full recursive search
    ckpts = glob.glob("/workspace/logs/train/*/runs/*/checkpoints/*.ckpt")

    # De-duplicate and sort by modification time (newest first)
    ckpts = list(set(ckpts))
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts


def load_vocoder_24k():
    from matcha.hifigan.env import AttrDict
    from matcha.hifigan.models import Generator

    h = AttrDict(json.load(open(VOC24K_CONFIG)))
    g = Generator(h)
    g.load_state_dict(torch.load(VOC24K_CKPT, map_location="cpu")["generator"])
    g.eval()
    g.remove_weight_norm()
    return g, h.sampling_rate


def ensure_model_loaded(checkpoint_path):
    global current_checkpoint, model, vocoder, denoiser, lane, n_spks, sample_rate
    if current_checkpoint == checkpoint_path:
        return
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    has_vat = any("vat" in k or "film" in k for k in ckpt["state_dict"])
    n_spks = int(hp.get("n_spks") or 1)

    model = load_matcha("custom", checkpoint_path, device)

    if has_vat or n_spks > 1:
        lane = "vat"
        vocoder, sample_rate = load_vocoder_24k()
        denoiser = None
    else:
        lane = "legacy"
        from matcha.utils.utils import get_user_data_dir
        save_dir = Path(get_user_data_dir())
        vocoder_path = save_dir / "hifigan_T2_v1"
        if not vocoder_path.exists():
            from matcha.cli import assert_model_downloaded, VOCODER_URLS
            assert_model_downloaded(vocoder_path, VOCODER_URLS["hifigan_T2_v1"])
        vocoder, denoiser = load_vocoder("hifigan_T2_v1", vocoder_path, device)
        sample_rate = 22050
    current_checkpoint = checkpoint_path
    print(f"  lane={lane} n_spks={n_spks} sr={sample_rate}")


def encode_text(text):
    """Text -> padded id tensor for the loaded lane."""
    if lane == "vat":
        g2p = get_g2p()
        ipa = g2p.phonemize(text)
        bad = g2p.validate(ipa)
        if bad:
            raise ValueError(f"out-of-vocab characters after G2P: {bad} "
                             "(digits are not expanded — write numbers out)")
        seq, _ = text_to_sequence(ipa, ["no_cleaners"])
        x = torch.tensor(intersperse(seq, 0), dtype=torch.long, device=device)[None]
        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
        return x, x_lengths
    out = process_text(1, text, device)
    return out["x"], out["x_lengths"]


def render(checkpoint_path, text, n_timesteps, temperature, length_scale,
           spk_id, valence, energy, tension, guidance=1.0):
    ensure_model_loaded(checkpoint_path)
    with torch.no_grad():
        x, x_lengths = encode_text(text)
        kwargs = {}
        if n_spks > 1:
            kwargs["spks"] = torch.tensor(
                [max(0, min(int(spk_id), n_spks - 1))], dtype=torch.long)
        else:
            kwargs["spks"] = None
        if lane == "vat":
            kwargs["vat"] = torch.tensor(
                [[float(valence), float(energy), float(tension)]])
            kwargs["guidance"] = float(guidance)
        output = model.synthesise(
            x, x_lengths,
            n_timesteps=n_timesteps,
            temperature=temperature,
            length_scale=length_scale,
            **kwargs,
        )
        if denoiser is not None:
            waveform = to_waveform(output["mel"], vocoder, denoiser)
        else:
            waveform = vocoder(output["mel"]).squeeze()
    return output, waveform


def synthesize(checkpoint_path, text, n_timesteps, temperature, length_scale,
               spk_id, valence, energy, tension, guidance=1.0):
    if not checkpoint_path:
        return "No checkpoint selected", None, None, ""
    try:
        output, waveform = render(checkpoint_path, text, n_timesteps,
                                  temperature, length_scale,
                                  spk_id, valence, energy, tension, guidance)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            sf.write(fp.name, waveform.cpu().numpy(), sample_rate, "PCM_24")
        mel_plot = plot_tensor(output["mel"].squeeze().cpu().numpy())
        info = (f"lane={lane} · {sample_rate} Hz · speakers={n_spks}"
                + (" · V/A/T active (derisk ckpt: only energy is trained) · "
                   "guidance >1 wants ≥25 ODE steps"
                   if lane == "vat" else " · V/A/T + speaker + guidance ignored"))
        return None, fp.name, mel_plot, info
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return str(e), None, None, ""


def refresh_checkpoints():
    ckpts = get_checkpoints()
    return gr.update(choices=ckpts, value=ckpts[0] if ckpts else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    cli_args = parser.parse_args()

    initial_ckpts = get_checkpoints()

    with gr.Blocks(title="🎙️ Sonora Vocalizer") as demo:
        gr.Markdown("# 🎙️ Sonora Vocalizer")
        gr.Markdown("Audition any training checkpoint on the CPU. VAT/multi-speaker "
                    "checkpoints get the 24 kHz vocoder, a speaker picker, "
                    "valence/energy/tension direction, and CFG amplification; "
                    "legacy LJSpeech checkpoints render exactly as before.")

        with gr.Row():
            with gr.Column(scale=2):
                checkpoint_dropdown = gr.Dropdown(
                    choices=initial_ckpts,
                    value=initial_ckpts[0] if initial_ckpts else None,
                    label="Training Checkpoint (.ckpt)",
                    interactive=True
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Rescan Checkpoints")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    value="The secret of getting ahead is getting started.",
                    lines=3,
                    label="Text to Synthesize"
                )

                with gr.Row():
                    n_timesteps = gr.Slider(
                        label="Number of ODE steps",
                        minimum=1, maximum=100, step=1, value=10)
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1, maximum=1.0, step=0.05, value=0.667)
                    length_scale = gr.Slider(
                        label="Length Scale (Speaking Rate)",
                        minimum=0.5, maximum=2.0, step=0.05, value=1.0)

                with gr.Row():
                    spk_input = gr.Number(
                        label="Speaker id (multi-speaker ckpts)",
                        value=245, precision=0, minimum=0)
                    valence = gr.Slider(
                        label="Valence", minimum=-1.0, maximum=1.0,
                        step=0.05, value=0.0)
                    energy = gr.Slider(
                        label="Energy", minimum=-1.0, maximum=1.0,
                        step=0.05, value=0.0)
                    tension = gr.Slider(
                        label="Tension", minimum=-1.0, maximum=1.0,
                        step=0.05, value=0.0)
                    guidance = gr.Slider(
                        label="Guidance (CFG ×, needs ≥25 ODE steps)",
                        minimum=1.0, maximum=4.0, step=0.25, value=1.0)

                synth_btn = gr.Button("🔊 Synthesize Speech", variant="primary")
                error_box = gr.Textbox(label="Error Status", visible=False)
                lane_info = gr.Markdown("")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
                mel_spectrogram_output = gr.Image(label="Mel Spectrogram", show_label=True)

        refresh_btn.click(fn=refresh_checkpoints, outputs=checkpoint_dropdown)

        def on_synth(checkpoint, text, steps, temp, length, spk, v, a, t, s):
            err, audio, mel, info = synthesize(checkpoint, text, steps, temp,
                                               length, spk, v, a, t, s)
            if err:
                return gr.update(value=err, visible=True), None, None, info
            return gr.update(visible=False), audio, mel, info

        synth_btn.click(
            fn=on_synth,
            inputs=[checkpoint_dropdown, text_input, n_timesteps, temperature,
                    length_scale, spk_input, valence, energy, tension, guidance],
            outputs=[error_box, audio_output, mel_spectrogram_output, lane_info]
        )

    # Setup FastAPI app
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    import uvicorn

    app = FastAPI(title="Sonora OpenAI-Compatible TTS API")

    @app.get("/v1/models")
    async def list_models():
        ckpts = get_checkpoints()
        model_list = []
        for ckpt in ckpts:
            name = os.path.basename(ckpt)
            model_list.append({
                "id": name,
                "object": "model",
                "created": int(os.path.getmtime(ckpt)),
                "owned_by": "sonora"
            })
        return {"data": model_list}

    @app.get("/v1/audio/voices")
    @app.get("/v1/voices")
    async def list_voices():
        if current_checkpoint and n_spks > 1:
            return {"voices": [str(i) for i in range(n_spks)]}
        return {"voices": ["default"]}

    @app.post("/v1/audio/speech")
    async def text_to_speech(request: Request):
        """OpenAI-ish TTS. Extra optional fields beyond input/model/voice:
        valence, energy, tension (floats in [-1, 1]; VAT ckpts only);
        guidance (CFG scale, default 1 = off); ode_steps (defaults 10, or 25
        when guidance > 1 — amplification needs the finer solve)."""
        try:
            body = await request.json()
            input_text = body.get("input", "")
            model_name = body.get("model", "")
            voice = body.get("voice", "")

            ckpts = get_checkpoints()
            checkpoint_path = None
            for ckpt in ckpts:
                if os.path.basename(ckpt) == model_name or ckpt == model_name:
                    checkpoint_path = ckpt
                    break
            if not checkpoint_path and ckpts:
                checkpoint_path = ckpts[0]
            if not checkpoint_path:
                return StreamingResponse(io.BytesIO(b"Error: No checkpoints found"), status_code=400)

            try:
                spk_id = int(voice)
            except (TypeError, ValueError):
                spk_id = 245  # a known-good LibriTTS-R val speaker

            s = float(body.get("guidance", 1.0))
            _, waveform = render(
                checkpoint_path, input_text,
                n_timesteps=int(body.get("ode_steps", 25 if s > 1.0 else 10)),
                temperature=0.667, length_scale=1.0,
                spk_id=spk_id,
                valence=float(body.get("valence", 0.0)),
                energy=float(body.get("energy", 0.0)),
                tension=float(body.get("tension", 0.0)),
                guidance=s,
            )

            buffer = io.BytesIO()
            sf.write(buffer, waveform.cpu().numpy(), sample_rate,
                     format="WAV", subtype="PCM_24")
            buffer.seek(0)
            return StreamingResponse(buffer, media_type="audio/wav")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return StreamingResponse(io.BytesIO(f"Error: {str(e)}".encode()), status_code=500)

    # Mount Gradio interface to FastAPI
    app = gr.mount_gradio_app(app, demo, path="/")

    uvicorn.run(app, host=cli_args.host, port=cli_args.port)


if __name__ == "__main__":
    main()
