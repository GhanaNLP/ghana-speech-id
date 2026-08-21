#!/usr/bin/env bash
set -e
P=/mnt/volume_d2wey28/projects/ghana-speech-id
cd "$P"
M=out/svm_ng5_mf200000_contiguous_twimerged
echo "=== export ==="
.venv/bin/python -u scripts/export_onnx.py --model "$M/model.joblib" --outdir "$M/onnx" --n-check 300 2>&1 | tail -12
echo
echo "=== artefacts ==="
ls -la "$M/onnx"
echo
echo "=== C++ selftest ==="
export LD_LIBRARY_PATH="$P/third_party/onnxruntime-linux-x64-1.20.1/lib:${LD_LIBRARY_PATH:-}"
GSID_MODEL_DIR="$P/$M/onnx" ./build/gsid_selftest
echo
echo "=== C++ vs Python parity on 300 validation strings ==="
.venv/bin/python -u scripts/cpp_parity.py --model "$M/model.joblib" --onnx-dir "$M/onnx" \
    --cli ./build/gsid --n 300
