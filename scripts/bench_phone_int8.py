"""int8 on CPU for both phoneme front ends, measured rather than extrapolated.

torch dynamic quantisation converts Linear layers to int8 with dynamic activation scaling.
It needs no ONNX export and no calibration data, so it is the cheapest honest read on what
quantisation buys. Convolutions stay fp32, so a wav2vec2 feature extractor or an
E-Branchformer's conv modules do not benefit -- an ONNX int8 export would do better, and
this is a lower bound on it.

Output is also checked, not just speed: quantisation that changes the transcript is not a
free win.
"""
from __future__ import annotations

import glob
import io
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

SR, N, THREADS = 16000, 12, 4


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


def size_mb(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6


def timeit(fn, ws, secs):
    for w in ws[:2]:
        fn(w)
    t0 = time.time()
    outs = [fn(w) for w in ws]
    dt = time.time() - t0
    return secs / dt, dt / len(ws) * 1000, outs


def main():
    torch.set_num_threads(THREADS)
    ws = clips()
    secs = sum(len(w) / SR for w in ws)
    print(f"{len(ws)} clips, {secs:.0f}s audio, {THREADS} threads, CPU\n")
    print(f"{'model':26} {'x RT':>7} {'ms/clip':>8} {'weights':>9}  transcripts match fp32")
    print("-" * 74)

    import json as _json

    from huggingface_hub import hf_hub_download
    from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2ForCTC

    # ---- wav2vec2
    W2V = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    vocab = _json.load(open(hf_hub_download(W2V, "vocab.json")))
    inv = {v: k for k, v in vocab.items()}
    pad = vocab.get("<pad>", 0)
    fe = AutoFeatureExtractor.from_pretrained(W2V)
    m = Wav2Vec2ForCTC.from_pretrained(W2V).eval()

    def dec(ids):
        o, prev = [], -1
        for i in ids:
            if i != prev and i != pad:
                t = inv.get(int(i), "")
                if t and not t.startswith("<"):
                    o.append(t)
            prev = i
        return " ".join(o)

    def run_w2v(model):
        return lambda w: dec(model(
            fe(w, sampling_rate=SR, return_tensors="pt").input_values
        ).logits.argmax(-1)[0].tolist())

    with torch.inference_mode():
        rt, ms, base = timeit(run_w2v(m), ws, secs)
        print(f"{'wav2vec2 fp32':26} {rt:7.1f} {ms:8.0f} {size_mb(m):8.0f}M  --")
        mq = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        rt, ms, q = timeit(run_w2v(mq), ws, secs)
        same = sum(a == b for a, b in zip(base, q))
        print(f"{'wav2vec2 int8 (dynamic)':26} {rt:7.1f} {ms:8.0f} {size_mb(mq):8.0f}M  "
              f"{same}/{len(ws)} identical")
    del m, mq

    # ---- PhoneticXeus
    m2 = AutoModel.from_pretrained("changelinglab/PhoneticXeus",
                                   trust_remote_code=True).eval()

    def run_x(model):
        def f(w):
            r = model.transcribe(torch.from_numpy(w))
            if isinstance(r, list) and r and isinstance(r[0], dict):
                return r[0].get("processed_transcript", "")
            return str(r)
        return f

    with torch.inference_mode():
        rt, ms, base2 = timeit(run_x(m2), ws, secs)
        print(f"{'PhoneticXeus fp32':26} {rt:7.1f} {ms:8.0f} {size_mb(m2):8.0f}M  --")
        try:
            m2q = torch.quantization.quantize_dynamic(m2, {torch.nn.Linear},
                                                      dtype=torch.qint8)
            rt, ms, q2 = timeit(run_x(m2q), ws, secs)
            same = sum(a == b for a, b in zip(base2, q2))
            print(f"{'PhoneticXeus int8 (dyn)':26} {rt:7.1f} {ms:8.0f} {size_mb(m2q):8.0f}M  "
                  f"{same}/{len(ws)} identical")
        except Exception as e:
            print(f"{'PhoneticXeus int8':26}  FAILED {type(e).__name__}: {str(e)[:60]}")

    print("\nomniASR CTC 300M int8 via sherpa-onnx, the shipping reference: ~17x RT")
    print("dynamic quantisation leaves convolutions in fp32, so an ONNX int8 export")
    print("should beat these -- this is a lower bound on what quantisation buys.")


if __name__ == "__main__":
    main()
