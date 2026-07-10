import os
import random

def main():
    base_dir = "/home/lmcfarlin/Projects/Artificial-Humanity/Sonora/data/LJSpeech-1.1"
    metadata_path = os.path.join(base_dir, "metadata.csv")
    train_path = os.path.join(base_dir, "train.txt")
    val_path = os.path.join(base_dir, "val.txt")

    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found. Please verify the dataset symlink.")
        return

    print("Parsing metadata.csv...")
    lines_formatted = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                clip_id = parts[0]
                # Prefer normalized text (third column) if available, fallback to raw text (second column)
                text = parts[2] if len(parts) >= 3 and parts[2].strip() else parts[1]
                
                # Format expected by the datamodule: data/LJSpeech-1.1/wavs/{clip_id}.wav|text
                formatted_line = f"data/LJSpeech-1.1/wavs/{clip_id}.wav|{text}"
                lines_formatted.append(formatted_line)

    print(f"Loaded {len(lines_formatted)} rows. Splitting...")
    # Shuffle with a fixed seed for reproducibility
    random.seed(1234)
    random.shuffle(lines_formatted)

    # 95% train / 5% validation split
    split_idx = int(len(lines_formatted) * 0.95)
    train_lines = lines_formatted[:split_idx]
    val_lines = lines_formatted[split_idx:]

    print(f"Writing {len(train_lines)} training rows to {train_path}...")
    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    print(f"Writing {len(val_lines)} validation rows to {val_path}...")
    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    print("Success! Dataset filelists prepared.")

if __name__ == "__main__":
    main()
