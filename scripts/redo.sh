#!/usr/bin/env bash
# Redo only the two stages that failed. The 740 h training decode succeeded and is kept.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
SELF=$$
for pid in $(pgrep -f "stage4.sh"); do
  [ "$pid" = "$SELF" ] && continue
  [ "$pid" = "$PPID" ] && continue
  kill "$pid" 2>/dev/null && echo "paused stage4 ($pid)"
done
V=$PWD/.venv-sherpa-gpu/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d $V/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}
export HF_HOME=/mnt/volume_d2wey28/hf-cache
PY=.venv-sherpa-gpu/bin/python

EN=$(ls -d /mnt/volume_d2wey28/hf_cache/hub/datasets--ghanaopendata--ghana-english-speech-ipa/snapshots/*/ | head -1)
echo "########## English (path fixed: $EN) ##########"
$PY -u scripts/decode_base.py --model 300m --provider cuda --batch 64 \
    --audio-root "${EN%/}" --keep-ids "" --out data/base_300m_english.parquet

EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)
echo; echo "########## eval (id synthesised: $EV) ##########"
$PY -u scripts/decode_base.py --model 300m --provider cuda --batch 64 \
    --audio-root "${EV%/}" --keep-ids "" --out data/base_300m_eval.parquet

echo "REDO DONE"
setsid nohup bash scripts/stage4.sh > logs/stage4.log 2>&1 < /dev/null &
disown
echo "stage4 relaunched"
