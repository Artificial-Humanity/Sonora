import os
import subprocess
import sys
import io

# Ensure datasets and soundfile are installed in the container python environment
try:
    import datasets
except ImportError:
    print("Installing datasets package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    import datasets

try:
    import soundfile as sf
except ImportError:
    print("Installing soundfile package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
    import soundfile as sf

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

def main():
    output_dir = "/data/model-training/datasets/expresso"
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")

    print("Loading Expresso dataset from Hugging Face...")
    # Load the train split of Expresso
    dataset = datasets.load_dataset("ylacombe/expresso", split="train")
    
    print("Casting audio column to disable auto-decoding...")
    dataset = dataset.cast_column("audio", datasets.Audio(decode=False))

    print(f"Loaded {len(dataset)} examples. Resampling and exporting to 24 kHz mono...")
    
    # Resampler: 48kHz -> 24kHz
    resampler = T.Resample(orig_freq=48000, new_freq=24000)
    
    metadata_lines = []
    
    for i, item in enumerate(tqdm(dataset)):
        audio_data = item['audio']
        raw_bytes = audio_data.get('bytes')
        
        # Decode using soundfile from raw bytes
        if raw_bytes is not None:
            array, sr = sf.read(io.BytesIO(raw_bytes))
        else:
            path = audio_data.get('path')
            array, sr = sf.read(path)
        
        # Audio preprocessing
        audio_tensor = torch.from_numpy(array).float()
        if audio_tensor.ndim == 2:
            # If stereo (shape [samples, channels]), convert to mono by taking the mean across channels
            audio_tensor = audio_tensor.mean(dim=1, keepdim=True).t()
        elif audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        # Resample to 24kHz if the native sampling rate differs
        if sr != 24000:
            if sr == 48000:
                audio_tensor = resampler(audio_tensor)
            else:
                dyn_resample = T.Resample(orig_freq=sr, new_freq=24000)
                audio_tensor = dyn_resample(audio_tensor)
                
        # Save audio using soundfile
        speaker = item.get('speaker_id', 'unknown').lower()
        clip_id = f"{speaker}_{i:06d}"
        filename = f"{clip_id}.wav"
        filepath = os.path.join(wav_dir, filename)
        
        audio_np = audio_tensor.squeeze(0).numpy()
        sf.write(filepath, audio_np, 24000)
        
        # Extract transcript and style/emotion keys
        text = item.get('text', '').strip()
        style = item.get('style', 'neutral').lower()
        
        # Record relative path from the Sonora data folder's perspective
        rel_path = f"data/expresso/wavs/{filename}"
        metadata_lines.append(f"{rel_path}|{speaker}|{style}|{text}")
        
    print(f"Writing metadata to {metadata_path}...")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for line in metadata_lines:
            f.write(line + "\n")
            
    print("Success! Expresso dataset downloaded and preprocessed to 24kHz mono.")

if __name__ == "__main__":
    main()
