#!/usr/bin/env bash
# Separate venv for GPU decoding. fairseq2 links PyTorch's C++ ABI so the torch pin is
# exact, and it needs numpy<2 -- both incompatible with the training venv, hence the split.
set -euo pipefail
P=/mnt/volume_d2wey28/projects/ghana-speech-id
export HF_HOME=/mnt/volume_d2wey28/hf-cache
export TMPDIR=/mnt/volume_d2wey28/tmp
export PIP_CACHE_DIR=/mnt/volume_d2wey28/cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
cd "$P"
[ -d .venv-gpu ] || python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip -q install --upgrade pip

python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.8.0') else 1)" 2>/dev/null || \
  pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
pip install "numpy<2" omnilingual-asr sentencepiece pyarrow soundfile PyYAML \
  "huggingface_hub>=0.30" 2>&1 | tail -3
pip install --no-deps git+https://github.com/GhanaNLP/ghana-ipa-asr 2>&1 | tail -2

echo "=== check ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
import numpy; print("numpy", numpy.__version__)
import fairseq2; print("fairseq2", fairseq2.__version__)
from ghana_ipa_asr.batch import load_model, run_batch, resolve_model
from ghana_ipa_asr.frontend import normalize_padded_
print("ghana_ipa_asr.batch imports OK")
PY
echo "GPU SETUP DONE"
