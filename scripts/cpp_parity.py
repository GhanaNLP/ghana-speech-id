"""Does the C++ CLI agree with the Python model it was exported from?

The ONNX graph was already checked against sklearn inside export_onnx.py. This checks the
other half -- the tokenisation and n-gram lookup that the C++ reimplements. A mismatch here
means the app would see different features from the trainer, which is the failure mode that
would otherwise only show up as unexplained accuracy loss on device.
"""
import argparse, subprocess, sys
import joblib
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--onnx-dir", required=True)
ap.add_argument("--cli", required=True)
ap.add_argument("--data", default="/mnt/volume_d2wey28/projects/ghana-speech-id/data/ipa_text.parquet")
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()

b = joblib.load(a.model)
vec, clf = b["vec"], b["clf"]

t = pq.read_table(a.data, columns=["ipa", "language"]).to_pydict()
PUNCT = set(".,!?;:\"'()-—…")
strings = []
for s in t["ipa"]:
    if not s:
        continue
    s = " ".join(u for u in s.split() if u not in PUNCT)   # head trained with --drop-punct
    if len(s.split()) >= 5:
        strings.append(s)
    if len(strings) >= a.n:
        break

py = clf.predict(vec.transform(strings))

proc = subprocess.run([a.cli, "--model-dir", a.onnx_dir],
                      input="\n".join(strings), capture_output=True, text=True)
if proc.returncode != 0:
    print("CLI failed:", proc.stderr[:500]); sys.exit(1)
cpp = [l.split("\t")[0] for l in proc.stdout.strip().split("\n")]

if len(cpp) != len(py):
    print(f"MISMATCH in count: cpp={len(cpp)} py={len(py)}"); sys.exit(1)

agree = sum(1 for x, y in zip(cpp, py) if x == y)
print(f"agreement: {agree}/{len(py)}  ({100*agree/len(py):.2f}%)")
if agree != len(py):
    for x, y in zip(cpp, py):
        if x != y:
            print(f"  cpp={x:24} py={y}")
    sys.exit(1)
print("C++ and Python produce identical predictions")
