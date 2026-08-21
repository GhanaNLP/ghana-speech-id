#!/usr/bin/env bash
# Retire the CPU decode and its chained job; decode all English on the GPU in one pass so
# the 8.5k / 24k / 50k conditions are subsamples of a single decode rather than three runs.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
SELF=$$
for pat in rephonemise_english after_decode; do
  for pid in $(pgrep -f "$pat"); do
    [ "$pid" = "$SELF" ] && continue
    [ "$pid" = "$PPID" ] && continue
    kill "$pid" 2>/dev/null && echo "killed $pid ($pat)"
  done
done
sleep 2
echo "remaining decode procs: $(pgrep -fc 'rephonemise_english|after_decode' || echo 0)"

SRC='/mnt/volume_d2wey28/hf_cache/hub/datasets--ghanaopendata--ghana-english-speech-ipa/snapshots/*/English_eng/*.parquet'
echo "=== GPU decode: all English, audio cropped to the Ghanaian duration distribution ==="
.venv-gpu/bin/python -u scripts/decode_gpu.py \
    --shards "$SRC" --audio-col audio \
    --crop-to data/ipa_text.parquet \
    --out data/english_ipa_gh.parquet 2>&1 | tail -25
echo "ENGLISH GPU DECODE DONE"
