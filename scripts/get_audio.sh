#!/usr/bin/env bash
# The training audio has to come back: re-transcribing with a different front-end needs
# waveforms, and the original pull deliberately took only the text columns.
set -uo pipefail
P=/mnt/volume_d2wey28/projects/ghana-speech-id
export HF_HOME=/mnt/volume_d2wey28/hf-cache
export TMPDIR=/mnt/volume_d2wey28/tmp
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$P"
. .venv/bin/activate   # hf lives in the venv, not on PATH
df -h /mnt/volume_d2wey28 | tail -1
python -m huggingface_hub.commands.huggingface_cli download ghananlpcommunity/ghana-speech-ipa --repo-type dataset \
  --local-dir data/ghana-speech-ipa-audio --max-workers 12 2>&1 | tail -3
echo "=== size ==="
du -sh data/ghana-speech-ipa-audio
find data/ghana-speech-ipa-audio -name "*.parquet" | wc -l
echo "AUDIO DOWNLOAD DONE"
