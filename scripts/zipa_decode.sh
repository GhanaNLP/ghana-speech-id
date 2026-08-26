#!/usr/bin/env bash
# Decode one variant's train and eval sets. Resumable: parts-dir skips finished shards.
set -uo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id
M=${1:?variant}
W=${2:-5}
EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)

echo "########## $M: train, $W workers ##########"
for i in $(seq 0 $((W-1))); do
  .venv/bin/python -u scripts/decode_base.py --model "$M" --provider cpu --threads 4 \
      --batch 16 --audio-root /mnt/volume_d2wey28/data/ghana-speech \
      --keep-ids data/ipa_text.parquet --parts-dir "data/parts_${M}_train" \
      --shard-index "$i" --shard-count "$W" --out "/tmp/${M}_tr_$i.parquet" \
      > "logs/${M}_w$i.log" 2>&1 &
done
wait
echo "########## $M: eval ##########"
for i in $(seq 0 $((W-1))); do
  .venv/bin/python -u scripts/decode_base.py --model "$M" --provider cpu --threads 4 \
      --batch 16 --audio-root "${EV%/}" --keep-ids "" --exclude-prefix bible_ \
      --parts-dir "data/parts_${M}_eval" --shard-index "$i" --shard-count "$W" \
      --out "/tmp/${M}_ev_$i.parquet" > "logs/${M}_ev$i.log" 2>&1 &
done
wait
echo "DECODE $M DONE"
