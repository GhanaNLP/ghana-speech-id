#!/usr/bin/env bash
set -e
P=/mnt/volume_d2wey28/projects/ghana-speech-id
cd "$P"
tar xzf lib.tgz && rm lib.tgz

ORT_VER=1.20.1
ORT_DIR="$P/third_party/onnxruntime-linux-x64-${ORT_VER}"
if [ ! -d "$ORT_DIR" ]; then
  mkdir -p "$P/third_party" && cd "$P/third_party"
  echo "fetching onnxruntime ${ORT_VER} C/C++ release..."
  curl -sL -o ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz"
  tar xzf ort.tgz && rm ort.tgz
  cd "$P"
fi
ls "$ORT_DIR/include" | head -5
echo "--- configure ---"
cmake -S . -B build -DONNXRUNTIME_ROOT="$ORT_DIR" -DCMAKE_BUILD_TYPE=Release > /tmp/cm.log 2>&1 \
  || { tail -25 /tmp/cm.log; exit 1; }
grep -E "onnxruntime (headers|library)" /tmp/cm.log
echo "--- build ---"
cmake --build build -j8 2>&1 | grep -E "warning|error|Error|\.cpp|Linking|Built target" | tail -30
ls -la build/gsid build/gsid_selftest build/libghana_speech_id.a 2>/dev/null
echo "BUILD OK"
