import os
import glob
import argparse
import tempfile
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

from matcha.cli import (
    get_device,
    load_matcha,
    load_vocoder,
    process_text,
    to_waveform,
)
from matcha.utils.utils import plot_tensor

# Define arguments namespace
args = argparse.Namespace(
    cpu=True,
    model="matcha_ljspeech",
    vocoder="hifigan_T2_v1",
)

# We run on CPU to avoid GPU conflicts with the training run
device = torch.device("cpu")

# Track loaded model and vocoder
current_checkpoint = None
model = None
vocoder = None
denoiser = None

def get_checkpoints():
    # Scan for all ckpt files in checkpoints subfolders without doing a full recursive search
    ckpts = glob.glob("/workspace/logs/train/*/runs/*/checkpoints/*.ckpt")
    
    # De-duplicate and sort by modification time (newest first)
    ckpts = list(set(ckpts))
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts

def synthesize(checkpoint_path, text, n_timesteps, temperature, length_scale):
    global current_checkpoint, model, vocoder, denoiser
    
    if not checkpoint_path:
        return "No checkpoint selected", None, None
        
    try:
        # Load model if it's different
        if current_checkpoint != checkpoint_path:
            print(f"Loading checkpoint: {checkpoint_path}")
            model = load_matcha("custom", checkpoint_path, device)
            # Load the corresponding LJ Speech vocoder
            # Note: since the training is on LJ Speech, we use hifigan_T2_v1
            # We can download it locally or load it from the cache
            from matcha.utils.utils import get_user_data_dir
            save_dir = Path(get_user_data_dir())
            vocoder_path = save_dir / "hifigan_T2_v1"
            if not vocoder_path.exists():
                from matcha.cli import assert_model_downloaded, VOCODER_URLS
                assert_model_downloaded(vocoder_path, VOCODER_URLS["hifigan_T2_v1"])
            
            vocoder, denoiser = load_vocoder("hifigan_T2_v1", vocoder_path, device)
            current_checkpoint = checkpoint_path
            
        # Process text and run inference under no_grad to prevent autograd tracking
        with torch.no_grad():
            output_text = process_text(1, text, device)
            
            # Synthesize mel
            output = model.synthesise(
                output_text["x"],
                output_text["x_lengths"],
                n_timesteps=n_timesteps,
                temperature=temperature,
                spks=None,
                length_scale=length_scale,
            )
            
            # Waveform generation
            waveform = to_waveform(output["mel"], vocoder, denoiser)
        
        # Save to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            sf.write(fp.name, waveform.cpu().numpy(), 22050, "PCM_24")
            
        # Plot mel spectrogram
        mel_plot = plot_tensor(output["mel"].squeeze().cpu().numpy())
        
        return None, fp.name, mel_plot
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return str(e), None, None

def refresh_checkpoints():
    ckpts = get_checkpoints()
    return gr.update(choices=ckpts, value=ckpts[0] if ckpts else None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    cli_args = parser.parse_args()
    
    # Get initial checkpoints list
    initial_ckpts = get_checkpoints()
    
    with gr.Blocks(title="🍵 Sonora (Matcha-TTS) Playpen") as demo:
        gr.Markdown("# 🍵 Sonora (Matcha-TTS) Playpen")
        gr.Markdown("Select any training checkpoint and enter text to synthesize speech on the CPU in real-time.")
        
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
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=10
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=0.667
                    )
                    length_scale = gr.Slider(
                        label="Length Scale (Speaking Rate)",
                        minimum=0.5,
                        maximum=2.0,
                        step=0.05,
                        value=0.95
                    )
                    
                synth_btn = gr.Button("🔊 Synthesize Speech", variant="primary")
                error_box = gr.Textbox(label="Error Status", visible=False)
                
            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
                mel_spectrogram_output = gr.Image(label="Mel Spectrogram", show_label=True)
                
        # Link callbacks
        refresh_btn.click(fn=refresh_checkpoints, outputs=checkpoint_dropdown)
        
        def on_synth(checkpoint, text, steps, temp, length):
            err, audio, mel = synthesize(checkpoint, text, steps, temp, length)
            if err:
                return gr.update(value=err, visible=True), None, None
            else:
                return gr.update(visible=False), audio, mel
                
        synth_btn.click(
            fn=on_synth,
            inputs=[checkpoint_dropdown, text_input, n_timesteps, temperature, length_scale],
            outputs=[error_box, audio_output, mel_spectrogram_output]
        )
        
    demo.launch(server_name=cli_args.host, server_port=cli_args.port)

if __name__ == "__main__":
    main()
