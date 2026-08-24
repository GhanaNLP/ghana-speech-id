#!/usr/bin/env bash
# Does the C++ char_wb reproduce sklearn's? A mismatch does not raise -- it silently
# misclassifies -- so this is the check that decides whether the port is shippable.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
M=out/chunk40_vote
[ -s "$M/model.joblib" ] || { echo "chunked model not ready"; exit 1; }

echo "########## export ##########"
.venv/bin/python -u scripts/export_onnx.py --model "$M/model.joblib" --outdir "$M/onnx" \
    --n-check 300 2>&1 | tail -12

echo; echo "########## head_config.txt ##########"
cat "$M/onnx/head_config.txt"
echo "casefold sample:"; head -5 "$M/onnx/casefold.txt"

echo; echo "########## C++ vs Python, whole-transcript (shipping mode) ##########"
export LD_LIBRARY_PATH=$PWD/third_party/onnxruntime-linux-x64-1.20.1/lib:${LD_LIBRARY_PATH:-}
.venv/bin/python - <<'PY'
import subprocess, sys
import joblib, numpy as np, pyarrow.parquet as pq
from ghana_speech_id import GhanaSpeechId

M = "out/chunk40_vote"
b = joblib.load(f"{M}/model.joblib"); vec, clf = b["vec"], b["clf"]
lid = GhanaSpeechId.load(f"{M}/onnx")
print("package:", repr(lid), "analyzer:", lid.analyzer, "lowercase:", lid.lowercase)

t = pq.read_table("data/base_300m_noen.parquet", columns=["text"]).to_pydict()["text"]
docs = [s for s in t if s and len(s) >= 60][:200]

def chunks(s, size=40, stride=20):
    if len(s) <= size: return [s]
    out = [s[i:i+size] for i in range(0, len(s)-size+1, stride)]
    tail = s[-size:]
    if out[-1] != tail: out.append(tail)
    return out

# sklearn, voted the same way the C++ does
classes = list(clf.classes_)
# whole transcript, matching chunk_chars 0 in the exported config
sk = list(clf.predict(vec.transform(docs)))

py = [(r.language if r else "unknown") for r in (lid.classify(d) for d in docs)]

p = subprocess.run(["./build/gsid", "--model-dir", f"{M}/onnx"],
                   input="\n".join(docs), capture_output=True, text=True)
if p.returncode != 0:
    print("CLI failed:", p.stderr[:400]); raise SystemExit(1)
cpp = [l.split("\t")[0] for l in p.stdout.strip().split("\n")]

n = len(docs)
print(f"sklearn vs python : {sum(a==b for a,b in zip(sk,py))}/{n}")
print(f"sklearn vs C++    : {sum(a==b for a,b in zip(sk,cpp))}/{n}")
print(f"python  vs C++    : {sum(a==b for a,b in zip(py,cpp))}/{n}")
bad = [(d,a,b,c) for d,a,b,c in zip(docs,sk,py,cpp) if not (a==b==c)]
for d,a,b,c in bad[:4]:
    print(f"  MISMATCH sk={a} py={b} cpp={c}\n    {d[:70]}")
PY
echo "PARITY DONE"
