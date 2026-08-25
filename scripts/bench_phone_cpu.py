"""CPU throughput of the phoneme front ends, against what we already ship.

The 610x and 290x figures elsewhere are CUDA. Deployment is CPU, so measure CPU: torch
fp32 here, which is the pessimistic case -- an int8 ONNX export would be faster, but
neither model has one and only wav2vec2 would export cleanly.

Reference point: omniASR CTC 300M int8 through sherpa-onnx runs at about 17x realtime on
CPU, and that is what the live demo uses.
"""
from __future__ import annotations

import glob
import io
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

SR = 16000
N = 12
THREADS = 4          # a phone has a few usable cores, not twenty


def clips():
    f = sorted(glob.glob("/mnt/volume_d2wey28/data/ghana-speech/*/*.parquet"))[0]
    t = pq.read_table(f, columns=["audio"]).to_pydict()
    out = []
    for cell in t["audio"]:
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        if 4.0 <= len(w) / sr <= 9.0:
            out.append(w.astype(np.float32))
        if len(out) >= N:
            break
    return out


def main():
    torch.set_num_threads(THREADS)
    ws = clips()
    secs = sum(len(w) / SR for w in ws)
    print(f"{len(ws)} clips, {secs:.0f}s audio, torch threads={THREADS}, CPU only\n")

    import json as _json

    from huggingface_hub import hf_hub_download
    from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2ForCTC

    W2V = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    vocab = _json.load(open(hf_hub_download(W2V, "vocab.json")))
    fe = AutoFeatureExtractor.from_pretrained(W2V)
    m = Wav2Vec2ForCTC.from_pretrained(W2V).to("cpu").eval()
    nparam = sum(p.numel() for p in m.parameters()) / 1e6
    for w in ws[:2]:                                    # warm
        with torch.inference_mode():
            m(fe(w, sampling_rate=SR, return_tensors="pt").input_values)
    t0 = time.time()
    for w in ws:
        with torch.inference_mode():
            m(fe(w, sampling_rate=SR, return_tensors="pt").input_values).logits.argmax(-1)
    dt = time.time() - t0
    print(f"wav2vec2-espeak  {nparam:.0f}M params  {secs/dt:6.1f}x RT  "
          f"({dt/len(ws)*1000:.0f} ms per clip)")
    del m

    m2 = AutoModel.from_pretrained("changelinglab/PhoneticXeus",
                                   trust_remote_code=True).to("cpu").eval()
    n2 = sum(p.numel() for p in m2.parameters()) / 1e6
    for w in ws[:2]:
        with torch.inference_mode():
            m2.transcribe(torch.from_numpy(w))
    t0 = time.time()
    for w in ws:
        with torch.inference_mode():
            m2.transcribe(torch.from_numpy(w))
    dt = time.time() - t0
    print(f"PhoneticXeus     {n2:.0f}M params  {secs/dt:6.1f}x RT  "
          f"({dt/len(ws)*1000:.0f} ms per clip)")

    print("\nfor comparison, omniASR CTC 300M int8 via sherpa-onnx on CPU: ~17x RT")
    print("these are torch fp32; int8 ONNX would be faster, but neither model has an")
    print("export and only wav2vec2 is a standard enough architecture to make one easily")


if __name__ == "__main__":
    main()
