#!/bin/bash
# LongCat-AudioDiT-3.5B audition: 4 standard TTS + 2 affect-transfer clones
# (synthetic MOSS anchors as prompts). MIT license — fully wall-clean.
set -e
cd /audition/longcat_src
M=/data/models/meituan-longcat/LongCat-AudioDiT-3.5B
O=/audition/out/longcat
mkdir -p $O

VICTORY="We won! We actually won the championship!"
GRIEF="She's gone. I keep setting two cups out in the morning, and then I remember."
THREAT="Don't move. Don't even breathe. If you make a sound, they will hear us."
LONGFORM="The lighthouse keeper woke before dawn, as he had done every morning for thirty years. He put the kettle on the small iron stove and watched the window while the water warmed. Outside, the sea was the color of slate, and the gulls hung in the wind like scraps of paper."

python inference.py --model_dir $M --guidance_method apg --text "$VICTORY"  --output_audio $O/victory.wav
python inference.py --model_dir $M --guidance_method apg --text "$GRIEF"    --output_audio $O/grief.wav
python inference.py --model_dir $M --guidance_method apg --text "$THREAT"   --output_audio $O/threat.wav
python inference.py --model_dir $M --guidance_method apg --text "$LONGFORM" --output_audio $O/longform.wav

python inference.py --model_dir $M --guidance_method apg --text "$GRIEF" \
  --prompt_text "The house is so quiet now. I don't know what to do with all this silence." \
  --prompt_audio /audition/out/anchors/anchor_grief.wav \
  --output_audio $O/grief_transfer.wav
python inference.py --model_dir $M --guidance_method apg --text "$VICTORY" \
  --prompt_text "Yes! Yes! I can't believe it, we actually did it!" \
  --prompt_audio /audition/out/anchors/anchor_victory.wav \
  --output_audio $O/victory_transfer.wav

echo "LONGCAT-RENDERS-DONE"
