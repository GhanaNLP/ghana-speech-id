"""Three-way agreement: sklearn (trainer) vs the pip package vs the C++ CLI.

export_onnx.py already checks the graph against sklearn. This checks the two independent
reimplementations of the tokeniser and n-gram lookup -- Python and C++ -- against it and
each other. Those are where a silent feature mismatch would live, and it would show up only
as unexplained accuracy loss in the app.
"""
import subprocess
import sys

import joblib
import pyarrow.parquet as pq
from ghana_speech_id import GhanaSpeechId

MODEL = "out/svm_ng5_mf200000_contiguous_twimerged"
N = 300
PUNCT = set(".,!?;:\"'()-—…")

b = joblib.load(f"{MODEL}/model.joblib")
vec, clf = b["vec"], b["clf"]
lid = GhanaSpeechId.load(f"{MODEL}/onnx")
print(f"package loaded: {lid!r}")

t = pq.read_table("data/ipa_text.parquet", columns=["ipa", "split"]).to_pydict()
strings = []
for s, sp in zip(t["ipa"], t["split"]):
    if sp != "validation" or not s:
        continue
    if len(s.split()) >= 5:
        strings.append(s)
    if len(strings) >= N:
        break
print(f"{len(strings)} validation strings")

sk = list(clf.predict(vec.transform(strings)))
py = [(r.language if r else "unknown") for r in (lid.classify(s) for s in strings)]

p = subprocess.run(["./build/gsid", "--model-dir", f"{MODEL}/onnx"],
                   input="\n".join(strings), capture_output=True, text=True)
if p.returncode != 0:
    print("C++ CLI failed:", p.stderr[:400]); sys.exit(1)
cpp = [l.split("\t")[0] for l in p.stdout.strip().split("\n")]

n = len(sk)
a_py = sum(x == y for x, y in zip(sk, py))
a_cpp = sum(x == y for x, y in zip(sk, cpp))
a_both = sum(x == y for x, y in zip(py, cpp))
print(f"sklearn vs python package : {a_py}/{n}")
print(f"sklearn vs C++ CLI        : {a_cpp}/{n}")
print(f"python package vs C++ CLI : {a_both}/{n}")

bad = [(s, a, b_, c) for s, a, b_, c in zip(strings, sk, py, cpp) if not (a == b_ == c)]
for s, a, b_, c in bad[:5]:
    print(f"  MISMATCH sk={a} py={b_} cpp={c}\n    {s[:70]}")
sys.exit(0 if not bad else 1)
