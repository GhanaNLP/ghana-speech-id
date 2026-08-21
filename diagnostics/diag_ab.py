"""A/B the GPU path against the reference sherpa-onnx CPU decoder on the same clips.

If sherpa produces sensible IPA where the GPU path produced almost none, the fault is in
decode_gpu.py. If both are sparse, the eval audio itself is the problem.
"""
import glob, io, sys
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

CFGS = ["finance_Asante_Twi", "waxal_Ewe_ewe", "jw_fante_fat"]
N = 6

root = (glob.glob("/mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*")
        or glob.glob("/mnt/volume_d2wey28/hf_cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*"))[0]

gpu = pq.read_table("data/eval_ipa_gh.parquet", columns=["id", "group", "ipa"]).to_pydict()
gpu_by_id = dict(zip(gpu["id"], gpu["ipa"]))

from ghana_ipa_asr import GhanaIPAASR
asr = GhanaIPAASR.load()
print("sherpa CPU reference loaded\n")

for cfg in CFGS:
    f = sorted(glob.glob(f"{root}/{cfg}/*.parquet"))[0]
    t = pq.read_table(f).to_pydict()
    print(f"########## {cfg} ##########")
    for i in range(min(N, len(t["audio"]))):
        cell = t["audio"][i]
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        secs = len(w) / sr
        ref = asr.transcribe(w, sr).spaced(punctuation=False)
        got = gpu_by_id.get(f"{cfg}_{i:06d}", "<missing>")
        print(f"  [{i}] {secs:5.1f}s  sherpa {len(ref.split()):3d} units "
              f"({len(ref.split())/secs:4.1f}/s) | gpu {len(got.split()):3d} units "
              f"({len(got.split())/secs:4.1f}/s)")
        print(f"      sherpa: {ref[:80]}")
        print(f"      gpu   : {got[:80]}")
    print()
