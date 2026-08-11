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
import soundfile as sf
import torch

# Lane detection, phonemization and the 24 kHz vocoder loader all live in matcha.cli and
# are imported, not re-implemented. This module carried its own copy of each until
# 2026-08-06; `SONORA_VOC24K_CONFIG` in particular defaulted to a different path than the
# CLI's (`hifi-gan/config_24k_80band.json` vs the copy sitting beside the checkpoint).
# Those two files are byte-identical — checked, not assumed — so consolidating on the
# colocated one changes nothing today, which is exactly the kind of near-miss that stops
# being harmless the moment one of them is edited.
from matcha import delivery, direction
from matcha.cli import (
    detect_lane,
    get_device,
    load_matcha,
    load_vocoder,
    load_vocoder_24k,
    process_text_for_lane,
    to_waveform,
)
from matcha.utils.utils import plot_tensor

# We run on CPU to avoid GPU conflicts with the training run
device = torch.device("cpu")

# Loaded-state registry (one checkpoint + its lane's vocoder at a time)
current_checkpoint = None
model = None
vocoder = None
denoiser = None
lane = None          # "vat" | "legacy"
n_spks = 1
sample_rate = 22050
# The loaded checkpoint's OWN conditioning width. Contract v2 made the production width 8
# (3 V/A/T + 5 one-hot delivery lanes), but a pre-v2 checkpoint is 3 and must still render
# — so the delivery dial is driven by what THIS checkpoint accepts, not by the contract.
ckpt_vat_dim = 0

# UI-only label for DELIVERY_UNKNOWN. The contract spells "no lane" as the empty string,
# which a dropdown cannot display or let you re-select, so the UI shows this word and
# `on_synth` maps it back. It is deliberately NOT a member of the lane vocabulary —
# `delivery_index` would refuse it, which is the correct behaviour if it ever leaks past
# the mapping instead of silently rendering an unconditioned clip.
UNKNOWN_UI = "unknown"


def get_checkpoints():
    # Scan for all ckpt files in checkpoints subfolders without doing a full recursive search
    ckpts = glob.glob("/workspace/logs/train/*/runs/*/checkpoints/*.ckpt")

    # De-duplicate and sort by modification time (newest first)
    ckpts = list(set(ckpts))
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts


def ensure_model_loaded(checkpoint_path):
    global current_checkpoint, model, vocoder, denoiser, lane, n_spks, sample_rate
    global ckpt_vat_dim
    if current_checkpoint == checkpoint_path:
        return
    print(f"Loading checkpoint: {checkpoint_path}")
    # Lane detection lives in matcha.cli — one implementation, shared with the CLI. This
    # file used to carry its own copy, which is the drift the review keeps finding.
    lane, n_spks, ckpt_vat_dim = detect_lane(checkpoint_path)

    model = load_matcha("custom", checkpoint_path, device)

    if lane == "vat":
        vocoder, sample_rate = load_vocoder_24k(device)
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
    """Text -> padded id tensor for the loaded lane.

    The op_g2p encoding used to be written out here as well as in matcha.cli. One copy
    now, in cli, so the CLI and this bench cannot disagree about what phonemes a
    checkpoint expects — a disagreement that would not raise, it would just synthesise.
    """
    out = process_text_for_lane(1, text, device, lane)
    return out["x"], out["x_lengths"]


# The control contract, in ONE place (E-M2). These are exactly the bounds the Gradio
# sliders enforce; before 2026-08-06 the HTTP API enforced nothing at all and passed
# whatever a caller sent straight into the model. V/A/T are per-speaker z-scores clamped
# at 2σ during derivation, so ±1 is already the edge of the trained range — a request for
# valence=50 does not produce more emotion, it produces a FiLM activation the trunk has
# never seen, and the failure is silent: plausible-sounding audio off the manifold. The
# UI could not send that and the API could, which is the whole finding.
CONTROL_BOUNDS = {
    "valence": (-1.0, 1.0),
    "energy": (-1.0, 1.0),
    "tension": (-1.0, 1.0),
    "guidance": (1.0, 4.0),
    "temperature": (0.1, 1.0),
    "length_scale": (0.5, 2.0),
    "ode_steps": (1, 100),
}


class ClientError(ValueError):
    """A bad request, not a server fault — mapped to 400 rather than 500."""


def bounded(body, name, default, cast=float):
    """Read one control off the request body and hold it to the contract.

    Rejects rather than silently clamping: a caller asking for valence=50 has a bug or a
    wrong mental model, and quietly rendering valence=1 teaches them the request worked.
    """
    raw = body.get(name, default)
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        raise ClientError(f"{name!r} must be a number, got {raw!r}") from None
    lo, hi = CONTROL_BOUNDS[name]
    if not lo <= value <= hi:
        raise ClientError(
            f"{name}={value:g} is outside the supported range [{lo:g}, {hi:g}]. "
            "V/A/T are per-speaker z-scores clamped at 2 sigma in derivation, so values "
            "beyond +/-1 are outside anything the model was trained on."
        )
    return value


def bounded_speaker(spk_id, n_spks_):
    """Hold a speaker id to the LOADED CHECKPOINT's embedding table.

    Not in `CONTROL_BOUNDS` because its bound is not a constant — it is whatever `n_spks`
    this checkpoint reports. That is exactly why it escaped E-M2: the sweep that gave
    every other control a bound could only see the ones with static ranges, and speaker
    stayed unchecked across two corpus generations while `n_spks` went 247 -> 2500.

    Rejects rather than clamping, and the clamp is the reason this exists. `render` used
    to hold the id down to the table's last row, so asking a 2500-speaker checkpoint for
    speaker 5000 rendered speaker 2499 and reported success. On a VETTING surface that is
    the worst available outcome: not an error, but a confident verdict about a voice
    nobody selected. Same argument as `bounded`'s, with more at stake, because a wrong
    V/A/T is audible as wrongness and a wrong speaker is simply a different person.
    """
    try:
        value = int(spk_id)
    except (TypeError, ValueError):
        # Named voices (`alloy`, `nova`, …) land here, and that is deliberate: the old
        # code mapped ANY unparseable voice to speaker 245 and rendered, which on a
        # vetting surface is a confident verdict about a voice nobody selected. Say so,
        # though — an OpenAI-shaped client sends a name, and "must be a whole number" on
        # its own reads as a bug in the caller's serializer rather than a deliberate
        # difference in this API (CL-L1).
        raise ClientError(
            f"speaker id must be a whole number, got {spk_id!r}. This API has no named "
            f"voices: ids are indices into the loaded checkpoint's embedding table "
            f"(see GET /v1/voices). Omit `voice` for the default."
        ) from None
    lo, hi = direction.speaker_bound(n_spks_)
    if not lo <= value <= hi:
        raise ClientError(
            f"speaker id {value} is outside this checkpoint's table [{lo}, {hi}] — it "
            f"has {n_spks_} speaker(s). Ids are indices into THIS checkpoint's embedding "
            "table, not corpus speaker numbers, and they are re-assigned every time the "
            "corpus is re-derived: id 245 was a LibriTTS-R val speaker under 247 "
            "speakers and is an unrelated voice under 2500."
        )
    return value


def render(checkpoint_path, text, n_timesteps, temperature, length_scale,
           spk_id, valence, energy, tension, guidance=1.0, delivery_lane=""):
    ensure_model_loaded(checkpoint_path)
    with torch.no_grad():
        x, x_lengths = encode_text(text)
        kwargs = {}
        if n_spks > 1:
            # Bounded, NOT clamped — see `bounded_speaker`. What this replaced held an
            # out-of-range id down to the last row of the table, turning it into a
            # different valid voice and reporting success.
            kwargs["spks"] = torch.tensor(
                [bounded_speaker(spk_id, n_spks)], dtype=torch.long)
        else:
            kwargs["spks"] = None
        if lane == "vat":
            # Contract v2. Built by `matcha.delivery`, never spelled out here — the
            # Vocalizer is the VETTING surface, so a dial that disagreed with the corpus
            # encoding would produce exactly the wrong evidence about the channel.
            #
            # Sized to the CHECKPOINT, not to the current contract: a 3-channel checkpoint
            # predates delivery and must still render, and the trunk's width guard would
            # otherwise refuse it with a shape error a listener cannot act on.
            if ckpt_vat_dim and ckpt_vat_dim <= delivery.VAT_BASE_DIM:
                vec = [float(valence), float(energy), float(tension)][:ckpt_vat_dim]
            else:
                vec = delivery.vat_vector(valence, energy, tension, delivery_lane)
            kwargs["vat"] = torch.tensor([vec])
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
               spk_id, valence, energy, tension, guidance=1.0, delivery_lane=""):
    if not checkpoint_path:
        return "No checkpoint selected", None, None, ""
    try:
        output, waveform = render(checkpoint_path, text, n_timesteps,
                                  temperature, length_scale,
                                  spk_id, valence, energy, tension, guidance,
                                  delivery_lane)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            sf.write(fp.name, waveform.cpu().numpy(), sample_rate, "PCM_24")
        mel_plot = plot_tensor(output["mel"].squeeze().cpu().numpy())
        # Name the delivery lane that ACTUALLY ran, including when the checkpoint has no
        # delivery channel. A dial that silently does nothing is how a vetting surface
        # produces a confident wrong verdict about a capability.
        if lane != "vat":
            extra = " · V/A/T + speaker + guidance ignored"
        elif ckpt_vat_dim and ckpt_vat_dim <= delivery.VAT_BASE_DIM:
            extra = (f" · V/A/T active · delivery IGNORED (this checkpoint is "
                     f"{ckpt_vat_dim}-channel, predating contract v2) · "
                     "guidance >1 wants ≥25 ODE steps")
        else:
            extra = (f" · V/A/T active · delivery={delivery_lane or 'unknown'} · "
                     "guidance >1 wants ≥25 ODE steps")
        info = f"lane={lane} · {sample_rate} Hz · speakers={n_spks}" + extra
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
                    # A Number and not a Slider: this is an INDEX into the checkpoint's
                    # embedding table, and neighbouring ids are unrelated people, so a
                    # slider would imply an ordering the table does not have.
                    #
                    # No `maximum` here on purpose. The bound is n_spks, which is not known
                    # until a checkpoint loads — and a maximum baked in at build time is
                    # the same defect as the 245 default it replaces: correct for one
                    # corpus and quietly wrong for the next. `bounded_speaker` rejects
                    # out-of-range at render and names the live range, and the info line
                    # reports `speakers=` after every synth.
                    spk_input = gr.Number(
                        label="Speaker id (index into THIS checkpoint's table — see "
                              "speakers= in the info line)",
                        value=0, precision=0, minimum=0)
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

                with gr.Row():
                    # Contract v2's delivery channel gets a dial in the same phase it
                    # ships — the standing rule ([[vocalizer-vetting-surface]]). A
                    # capability with no control here cannot be vetted, and an unvetted
                    # conditioning channel is one whose failure mode we learn about from
                    # a training run instead of from a listen.
                    #
                    # A DROPDOWN, not a slider: delivery is categorical. A slider would
                    # invite interpolating between Newscaster and Dialogue, which has no
                    # meaning and which `lane_of_vector` refuses outright.
                    #
                    # FLAT STRING choices, and they have to be. `(label, value)` tuple
                    # choices are a Gradio 4 feature; this runs 3.43.2, where the tuple
                    # itself becomes the value — so every lane pick reached
                    # `delivery_index` as "('Newscaster', 'Newscaster')" and the closed
                    # vocabulary raised, making the dial unusable for ALL five lanes
                    # while the HTTP lane (which passes plain strings) stayed fine.
                    # `UNKNOWN_UI` is a display label only; `on_synth` maps it back to
                    # DELIVERY_UNKNOWN, since "" cannot be shown in a dropdown.
                    delivery_lane = gr.Dropdown(
                        label="Delivery lane (contract v2; 'unknown' ≡ v1)",
                        choices=[UNKNOWN_UI] + list(delivery.DELIVERY_LANES),
                        value=UNKNOWN_UI, interactive=True)

                synth_btn = gr.Button("🔊 Synthesize Speech", variant="primary")
                error_box = gr.Textbox(label="Error Status", visible=False)
                lane_info = gr.Markdown("")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
                mel_spectrogram_output = gr.Image(label="Mel Spectrogram", show_label=True)

        refresh_btn.click(fn=refresh_checkpoints, outputs=checkpoint_dropdown)

        def on_synth(checkpoint, text, steps, temp, length, spk, v, a, t, s, d):
            # Tolerate a tuple as well as the sentinel: if this is ever run on Gradio 4,
            # tuple choices become legal again and would arrive as (label, value). Taking
            # the last element is correct for both, so a version bump cannot silently
            # re-break the dial the way the 4-vs-3 mismatch did.
            if isinstance(d, (tuple, list)):
                d = d[-1]
            if d == UNKNOWN_UI:
                d = delivery.DELIVERY_UNKNOWN
            err, audio, mel, info = synthesize(checkpoint, text, steps, temp,
                                               length, spk, v, a, t, s, d)
            if err:
                return gr.update(value=err, visible=True), None, None, info
            return gr.update(visible=False), audio, mel, info

        synth_btn.click(
            fn=on_synth,
            inputs=[checkpoint_dropdown, text_input, n_timesteps, temperature,
                    length_scale, spk_input, valence, energy, tension, guidance,
                    delivery_lane],
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

    def _delivery_of(body):
        lane = (body.get("delivery") or "").strip()
        try:
            delivery.delivery_index(lane)
        except ValueError as exc:
            raise ClientError(str(exc)) from None
        return lane

    @app.post("/v1/audio/speech")
    async def text_to_speech(request: Request):
        """OpenAI-ish TTS. Extra optional fields beyond input/model/voice:
        valence, energy, tension (floats in [-1, 1]; VAT ckpts only);
        delivery (contract v2 lane name, or omitted/"" for unknown ≡ v1);
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

            # Default 0, not 245. 245 was "a known-good LibriTTS-R val speaker" when the
            # table had 247 rows; under v5's 2500 it is an unrelated voice, and a default
            # that silently means someone different after each re-derivation is not a
            # default. 0 is the only id valid for every multi-speaker checkpoint. Range is
            # enforced in `render` via `bounded_speaker`, which needs the loaded
            # checkpoint's n_spks and so cannot run before the model is resolved.
            #
            # CL-L1: `None` is ABSENT, not a value. `body.get("voice", "")` returns None
            # for an explicit `"voice": null`, and `str(None).strip()` is the truthy
            # `"None"` — so a client that sent null got a 400 reading "got None" while a
            # client that omitted the field entirely got the default. Two spellings of
            # "no preference", two different answers, and the error named a Python repr
            # rather than anything the caller wrote.
            spk_id = voice if voice is not None and str(voice).strip() else 0

            s = bounded(body, "guidance", 1.0)
            _, waveform = render(
                checkpoint_path, input_text,
                n_timesteps=bounded(body, "ode_steps", 25 if s > 1.0 else 10, int),
                temperature=bounded(body, "temperature", 0.667),
                length_scale=bounded(body, "length_scale", 1.0),
                spk_id=spk_id,
                valence=bounded(body, "valence", 0.0),
                energy=bounded(body, "energy", 0.0),
                tension=bounded(body, "tension", 0.0),
                guidance=s,
                # Validated by `matcha.delivery`, which raises ValueError on an
                # unrecognised lane. Wrapped as ClientError so a typo'd lane is a 400
                # like every other bad control value, not a 500 that reads as an outage —
                # and NOT silently treated as unknown, which would render a neutral clip
                # while the caller believed it had asked for Newscaster.
                delivery_lane=_delivery_of(body),
            )

            buffer = io.BytesIO()
            sf.write(buffer, waveform.cpu().numpy(), sample_rate,
                     format="WAV", subtype="PCM_24")
            buffer.seek(0)
            return StreamingResponse(buffer, media_type="audio/wav")
        except ClientError as e:
            # 400, not 500: the request was wrong, the server is fine. Returned before
            # the generic handler so a bad control value cannot be read as an outage.
            return StreamingResponse(io.BytesIO(f"Error: {e}".encode()), status_code=400)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return StreamingResponse(io.BytesIO(f"Error: {str(e)}".encode()), status_code=500)

    # Mount Gradio interface to FastAPI
    app = gr.mount_gradio_app(app, demo, path="/")

    uvicorn.run(app, host=cli_args.host, port=cli_args.port)


if __name__ == "__main__":
    main()
