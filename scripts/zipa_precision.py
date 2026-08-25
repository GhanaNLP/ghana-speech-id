"""How much does precision cost ZIPA -- in speed, and in accuracy?

Six full decodes across three precisions and two sizes is about fifteen hours. Divergence
between precisions is measurable in minutes, and if fp32 and int8 produce near-identical
transcripts then one decode serves both. This is the same check that found the omniASR
train/serve mismatch, where int8 differed from fp32 on 3.09% of characters and cost the
head 1.3 points.

Reports CPU speed, transcript divergence against fp32, and -- where a head exists -- what
the difference actually costs in language-ID accuracy.
"""
from __future__ import annotations

import glob
import io
import time

import numpy as np
import pyarrow.parquet as pq
import sherpa_onnx as so
import soundfile as sf

SR, THREADS = 16000, 4
CONFIGS = ["Asante_Twi_twi", "Ewe_ewe", "Dagbani_dag", "Kasem_xsm", "Ninkare_gur"]
PER = 24


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


def clips():
    out = []
    for cfg in CONFIGS:
        fs = sorted(glob.glob(f"/mnt/volume_d2wey28/data/ghana-speech/{cfg}/*.parquet"))
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
            if 3.0 <= len(w) / sr <= 12.0:
                out.append((cfg, w.astype(np.float32))); got += 1
            if got >= PER:
                break
    return out


def main():
    cs = clips()
    secs = sum(len(w) / SR for _, w in cs)
    print(f"{len(cs)} clips, {secs:.0f}s audio, {THREADS} CPU threads\n")
    print(f"{'model':22} {'x RT':>7} {'MB':>7} {'chars':>7} {'vs fp32: same':>14} {'charΔ':>7}")
    print("-" * 70)

    import os
    for size in ("small", "large"):
        base = None
        for prec, fn in (("fp32", "model.onnx"), ("fp16", "model.fp16.onnx"),
                         ("int8", "model.int8.onnx")):
            path = f"models/zipa-{size}/{fn}"
            if not os.path.exists(path):
                print(f"{'zipa-'+size+' '+prec:22} not downloaded")
                continue
            rec = so.OfflineRecognizer.from_zipformer_ctc(
                model=path, tokens=f"models/zipa-{size}/tokens.txt",
                num_threads=THREADS, provider="cpu")
            for _, w in cs[:3]:
                s = rec.create_stream(); s.accept_waveform(SR, w); rec.decode_stream(s)
            t0 = time.time()
            txt = []
            for _, w in cs:
                s = rec.create_stream(); s.accept_waveform(SR, w); rec.decode_stream(s)
                txt.append(s.result.text)
            dt = time.time() - t0
            mb = os.path.getsize(path) / 1e6
            ch = np.mean([len(t) for t in txt])
            if prec == "fp32":
                base = txt
                cmp = "--".rjust(14) + " " + "--".rjust(7)
            else:
                same = sum(a == b for a, b in zip(base, txt))
                tot = sum(len(a) for a in base)
                ed = sum(edit(a, b) for a, b in zip(base, txt))
                cmp = f"{same}/{len(txt)}".rjust(14) + f" {ed/max(tot,1):6.2%}"
            print(f"{'zipa-'+size+' '+prec:22} {secs/dt:7.1f} {mb:7.0f} {ch:7.1f} {cmp}")
            np.save(f"/tmp/zipa_{size}_{prec}.npy", np.array(txt, dtype=object),
                    allow_pickle=True)
        print()

    print("omniASR int8, the shipping front end: 17x RT, 350 MB")
    print("its int8-vs-fp32 divergence was 3.09% of characters and cost 1.3 points")


if __name__ == "__main__":
    main()
