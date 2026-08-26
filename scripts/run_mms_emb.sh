#!/usr/bin/env bash
# MMS-LID-4017 features -> our own head. GPU; one pass yields both feature sets.
set -euo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id
export HF_HOME=/mnt/volume_d2wey28/hf-cache
PY=.venv-phone/bin/python
EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)

echo "########## eval, full length ##########"
$PY -u scripts/extract_mms.py --audio-root "${EV%/}" --keep-ids "" \
    --exclude-prefix bible_ --parts-dir data/parts_mms_eval --out data/mms_eval_full.parquet
echo "########## eval, 1.6 s ##########"
$PY -u scripts/extract_mms.py --audio-root "${EV%/}" --keep-ids "" \
    --exclude-prefix bible_ --truncate 1.6 --parts-dir data/parts_mms_eval16 \
    --out data/mms_eval_16.parquet
echo "########## training corpus ##########"
$PY -u scripts/extract_mms.py --parts-dir data/parts_mms_train --out data/mms_train.parquet

for col in emb logits; do
  echo; echo "########## head on $col ##########"
  .venv-sb/bin/python -u scripts/train_ecapa_head.py --train data/mms_train.parquet \
      --eval-full data/mms_eval_full.parquet --eval-short data/mms_eval_16.parquet \
      --feature-col "$col" --tag "mms_${col}_mlp256"
done
echo "MMS EMB DONE"
