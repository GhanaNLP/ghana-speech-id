#!/usr/bin/env bash
# Qwen3-ASR encoder embeddings, CPU int8 -- the configuration that would be served.
#
# sherpa-onnx exposes transcripts only, so embeddings come from running conv_frontend and
# encoder through onnxruntime directly. That is still deployable: the C++ library in this
# repo already links onnxruntime, so the chain is conv -> encoder -> pooled -> head in one
# runtime, with no sherpa dependency at all on this route.
#
# The decoder is never loaded. 44 MB + 182 MB against the 350 MB omniASR front end.
# -e as well: a failed extraction should stop the run, not fall through to a
# head that then complains about a missing file
set -euo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id
export HF_HOME=/mnt/volume_d2wey28/hf-cache
PY=.venv-phone/bin/python
EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)

echo "########## eval, full length ##########"
$PY -u scripts/extract_qwen.py --audio-root "${EV%/}" --keep-ids "" \
    --exclude-prefix bible_ --threads 4 --parts-dir data/parts_qwen_eval \
    --out data/qwen_eval_full.parquet

echo "########## eval, 1.6 s ##########"
$PY -u scripts/extract_qwen.py --audio-root "${EV%/}" --keep-ids "" \
    --exclude-prefix bible_ --threads 4 --truncate 1.6 --parts-dir data/parts_qwen_eval16 \
    --out data/qwen_eval_16.parquet

echo "########## training corpus ##########"
$PY -u scripts/extract_qwen.py --threads 4 --parts-dir data/parts_qwen_train \
    --out data/qwen_train.parquet

echo "########## head ##########"
.venv-sb/bin/python -u scripts/train_ecapa_head.py --train data/qwen_train.parquet \
    --eval-full data/qwen_eval_full.parquet --eval-short data/qwen_eval_16.parquet \
    --tag qwen_mlp256
echo "QWEN DONE"
