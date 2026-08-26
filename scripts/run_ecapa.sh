#!/usr/bin/env bash
# Frozen ECAPA embeddings as a front end, waiting for the GPU.
set -uo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id
export HF_HOME=/mnt/volume_d2wey28/hf-cache
PY=.venv-sb/bin/python
EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)

while pgrep -f "decode_phone.py" > /dev/null 2>&1; do sleep 60; done
while pgrep -f setup_sb > /dev/null 2>&1; do sleep 20; done
echo "=== GPU free, speechbrain ready ==="

echo "########## eval embeddings, full length ##########"
$PY -u scripts/extract_ecapa.py --audio-root "${EV%/}" --keep-ids "" \
    --exclude-prefix bible_ --parts-dir data/parts_ecapa_eval \
    --out data/ecapa_eval_full.parquet

echo "########## eval embeddings, 1.6 s ##########"
$PY -u scripts/extract_ecapa.py --audio-root "${EV%/}" --keep-ids "" \
    --exclude-prefix bible_ --truncate 1.6 --parts-dir data/parts_ecapa_eval16 \
    --out data/ecapa_eval_16.parquet

echo "########## training embeddings ##########"
$PY -u scripts/extract_ecapa.py --audio-root /mnt/volume_d2wey28/data/ghana-speech \
    --keep-ids data/ipa_text.parquet --parts-dir data/parts_ecapa_train \
    --out data/ecapa_train.parquet

echo "########## head ##########"
$PY -u scripts/train_ecapa_head.py --train data/ecapa_train.parquet \
    --eval-full data/ecapa_eval_full.parquet --eval-short data/ecapa_eval_16.parquet \
    --tag ecapa_mlp256
echo "ECAPA DONE"
