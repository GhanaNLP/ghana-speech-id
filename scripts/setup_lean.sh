#!/usr/bin/env bash
# Lean provisioning: the head is a text classifier over IPA units, so no audio and no
# fairseq2 are needed for training. Audio only comes back for the WaxalNLP OOD eval.
set -euo pipefail
P=/mnt/volume_d2wey28/projects/ghana-speech-id
export HF_HOME=/mnt/volume_d2wey28/hf-cache
export TMPDIR=/mnt/volume_d2wey28/tmp
mkdir -p "$P"/{data,logs,scripts,out} "$TMPDIR"
cd "$P"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip -q install --upgrade pip
pip -q install "numpy<2" "pyarrow>=15" pandas scikit-learn scipy \
  "huggingface_hub[cli,hf_transfer]>=0.30" datasets soundfile tqdm
echo "=== deps ok ==="
python - <<'PY'
import pyarrow, sklearn, numpy, huggingface_hub as hh
print("pyarrow", pyarrow.__version__, "| sklearn", sklearn.__version__, "| hub", hh.__version__)
from huggingface_hub import HfApi
api = HfApi()
try:
    print("whoami:", api.whoami()["name"])
except Exception as e:
    print("whoami FAILED:", e)
for d in ["ghananlpcommunity/ghana-speech-phonemes",
          "ghananlpcommunity/ghana-speech-ipa",
          "ghananlpcommunity/ghana-speech",
          "google/WaxalNLP"]:
    try:
        info = api.dataset_info(d)
        n = len(info.siblings or [])
        print(f"OK   {d}  ({n} files, gated={info.gated})")
    except Exception as e:
        print(f"FAIL {d}: {type(e).__name__} {str(e)[:110]}")
PY
echo "LEAN SETUP DONE"
