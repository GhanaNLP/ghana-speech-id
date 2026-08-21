#!/usr/bin/env bash
cd /mnt/volume_d2wey28/projects/ghana-speech-id
SELF=$$
for pid in $(pgrep -f "stage4.sh"); do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] || kill "$pid" 2>/dev/null
done
V=$PWD/.venv-sherpa-gpu/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d $V/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}
export HF_HOME=/mnt/volume_d2wey28/hf-cache
EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)
echo "########## eval decode ##########"
.venv-sherpa-gpu/bin/python -u scripts/decode_base.py --model 300m --provider cuda \
    --batch 64 --audio-root "${EV%/}" --keep-ids "" --exclude-prefix bible_ \
    --out data/base_300m_eval.parquet
echo "EVAL DECODE DONE"
setsid nohup bash scripts/stage4.sh > logs/stage4.log 2>&1 < /dev/null &
disown
echo "stage4 relaunched"
