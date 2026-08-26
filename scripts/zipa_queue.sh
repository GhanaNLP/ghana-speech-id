#!/usr/bin/env bash
# Remaining ZIPA variants, in order. Each decode is resumable, so a crash costs one shard.
cd /mnt/volume_d2wey28/projects/ghana-speech-id

# zipa-large train was at 97% when the last run died; let its worker finish first
while pgrep -f "decode_base.py --model zipa-large " > /dev/null 2>&1; do sleep 60; done

for M in zipa-large zipa-small-fp16 zipa-large-fp16; do
  echo "################ $M ################"
  bash scripts/zipa_decode.sh "$M" 5
  bash scripts/zipa_finish.sh "$M"
done
echo "ZIPA QUEUE DONE"
