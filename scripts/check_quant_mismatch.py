"""Does omniASR int8 transcribe differently from fp32, and does the head care?

The training corpus was decoded with fp32, because int8 on CUDA runs at 7x against fp32's
111x. The Modal demo serves int8, because on CPU that ordering inverts. So the head is
trained on one quantisation and served another, and dynamic int8 changed 10 of 12
transcripts for both phoneme models -- enough to suspect this matters.

Measures the divergence directly, then scores the shipped head on both.
"""
from __future__ import annotations

import glob
import io

import joblib
import numpy as np
import pyarrow.parquet as pq
import sherpa_onnx as so
import soundfile as sf

MD = "/mnt/volume_d2wey28/projects/ghana-speech-id/models"
FP32 = (f"{MD}/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12", "model.onnx")
INT8 = (f"{MD}/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12",
        "model.int8.onnx")
N = 200
CONFIGS = ["lds_Asante_Twi", "waxal_Ewe_ewe", "jw_fante_fat", "waxal_Dagbani_dag",
           "unicef_dagbani"]
ISO = {"lds_Asante_Twi": "Twi_twi", "waxal_Ewe_ewe": "Ewe_ewe", "jw_fante_fat": "Fante_fat",
       "waxal_Dagbani_dag": "Dagbani_dag", "unicef_dagbani": "Dagbani_dag"}


def edit(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


root = glob.glob("/mnt/volume_d2wey28/hf-cache/hub/"
                 "datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*")[0]
clips = []
for cfg in CONFIGS:
    fs = sorted(glob.glob(f"{root}/{cfg}/*.parquet"))
    if not fs:
        continue
    t = pq.read_table(fs[0], columns=["audio"]).to_pydict()
    got = 0
    for cell in t["audio"]:
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        try:
            w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception:
            continue
        if w.ndim > 1:
            w = w.mean(axis=1)
        if 3.0 <= len(w) / sr <= 15.0:
            clips.append((cfg, w.astype(np.float32), sr)); got += 1
        if got >= N // len(CONFIGS):
            break
print(f"{len(clips)} clips from {len(set(c[0] for c in clips))} configs\n")

out = {}
for name, (d, mf) in (("fp32", FP32), ("int8", INT8)):
    rec = so.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=f"{d}/{mf}", tokens=f"{d}/tokens.txt", num_threads=8, provider="cpu")
    txt = []
    for _, w, sr in clips:
        s = rec.create_stream(); s.accept_waveform(sr, w); rec.decode_stream(s)
        txt.append(s.result.text.strip())
    out[name] = txt
    print(f"{name}: mean {np.mean([len(t) for t in txt]):.1f} chars")

same = sum(a == b for a, b in zip(out["fp32"], out["int8"]))
tot = sum(len(a) for a in out["fp32"])
ed = sum(edit(a, b) for a, b in zip(out["fp32"], out["int8"]))
print(f"\nidentical transcripts: {same}/{len(clips)} ({same/len(clips):.1%})")
print(f"character difference between the two: {ed/max(tot,1):.2%}")

b = joblib.load("out/final_300m_mf50000/model.joblib")
vec, clf = b["vec"], b["clf"]
gold = [ISO[c] for c, _, _ in clips]
print("\nshipped head scored on each:")
for name in ("fp32", "int8"):
    keep = [i for i, t in enumerate(out[name]) if len(t) >= 10]
    pred = clf.predict(vec.transform([out[name][i] for i in keep]))
    acc = np.mean([p == gold[i] for p, i in zip(pred, keep)])
    print(f"  {name}: {acc:.4f} on {len(keep)} clips")
agree = 0
kp = [i for i in range(len(clips)) if len(out['fp32'][i]) >= 10 and len(out['int8'][i]) >= 10]
p1 = clf.predict(vec.transform([out["fp32"][i] for i in kp]))
p2 = clf.predict(vec.transform([out["int8"][i] for i in kp]))
print(f"  same prediction either way: {sum(a==b for a,b in zip(p1,p2))}/{len(kp)}")
