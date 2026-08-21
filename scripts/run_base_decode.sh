#!/usr/bin/env bash
# Full base-model decode: training corpus, English, evaluation set.
# fp32 on CUDA at ~111x realtime; int8 on CUDA is 7x, see decode_base.py.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
V=$PWD/.venv-sherpa-gpu/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d $V/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}
export HF_HOME=/mnt/volume_d2wey28/hf-cache
PY=.venv-sherpa-gpu/bin/python
M=${1:-300m}

echo "########## 1/3 training corpus ##########"
$PY -u scripts/decode_base.py --model "$M" --provider cuda --batch 64 \
    --audio-root /mnt/volume_d2wey28/data/ghana-speech \
    --keep-ids data/ipa_text.parquet \
    --out "data/base_${M}_train.parquet"

echo; echo "########## 2/3 English ##########"
$PY -u scripts/decode_base.py --model "$M" --provider cuda --batch 64 \
    --audio-root "/mnt/volume_d2wey28/hf_cache/hub/datasets--ghanaopendata--ghana-english-speech-ipa/snapshots" \
    --keep-ids "" \
    --out "data/base_${M}_english.parquet"

echo; echo "########## 3/3 evaluation set ##########"
EVAL=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/* 2>/dev/null | head -1)
$PY -u scripts/decode_base.py --model "$M" --provider cuda --batch 64 \
    --audio-root "$EVAL" --keep-ids "" \
    --out "data/base_${M}_eval.parquet"

echo "BASE DECODE DONE ($M)"
