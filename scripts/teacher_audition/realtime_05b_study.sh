#!/bin/bash
set -e
cd /work/VibeVoice-ms
uv pip install -q --python /opt/venv/bin/python -e . soundfile
OUT=/data/model-training/datasets/sonora-expressive-registers/vibevoice-realtime-explore
mkdir -p $OUT
for f in /work/rt_texts/*.txt; do
  id=$(basename "$f" .txt)
  start=$(date +%s.%N)
  python demo/realtime_model_inference_from_file.py --model_path /data/models/microsoft/VibeVoice-Realtime-0.5B --txt_path "$f" --speaker_name Carter > /tmp/rt_$id.log 2>&1 || { echo "$id FAILED"; tail -3 /tmp/rt_$id.log; continue; }
  end=$(date +%s.%N)
  wav=$(find . /tmp -name '*.wav' -newermt "@$start" 2>/dev/null | grep -vi asr_demo | head -1)
  [ -n "$wav" ] && cp "$wav" $OUT/${id}_RT05.wav
  echo "$id wall=$(echo "$end $start" | awk '{printf "%.1f", $1-$2}')s wav=$wav"
  grep -Ei 'rtf|real.?time|latency|generat' /tmp/rt_$id.log | tail -2 || true
done
echo RT2-DONE
