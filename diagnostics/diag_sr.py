"""What is actually in the eval audio cells: sample rate, format, channels, amplitude."""
import glob, io, collections
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

root = glob.glob("/mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*")
if not root:
    root = glob.glob("/mnt/volume_d2wey28/hf_cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*")
files = sorted(f for f in glob.glob(f"{root[0]}/*/*.parquet")
               if not f.rsplit("/", 2)[-2].startswith("bible_"))
print(f"{len(files)} non-bible files\n")

srs, fmts, chans = collections.Counter(), collections.Counter(), collections.Counter()
peaks = []
for f in files[:8]:
    cfg = f.rsplit("/", 2)[-2]
    t = pq.read_table(f, columns=["audio"]).to_pydict()
    for cell in t["audio"][:25]:
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        info = sf.info(io.BytesIO(raw))
        srs[info.samplerate] += 1
        fmts[f"{info.format}/{info.subtype}"] += 1
        chans[info.channels] += 1
        w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        peaks.append(float(np.abs(w).max()))
    print(f"  sampled {cfg}")

print("\nsample rates:", dict(srs))
print("formats:", dict(fmts))
print("channels:", dict(chans))
print(f"peak amplitude: min {min(peaks):.4f} median {np.median(peaks):.4f} max {max(peaks):.4f}")

print("\nEnglish source for comparison:")
ef = sorted(glob.glob("/mnt/volume_d2wey28/hf_cache/hub/datasets--ghanaopendata--ghana-english-speech-ipa/snapshots/*/English_eng/*.parquet"))[0]
t = pq.read_table(ef, columns=["audio"]).to_pydict()
esrs = collections.Counter()
for cell in t["audio"][:25]:
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    info = sf.info(io.BytesIO(raw))
    esrs[f"{info.samplerate} {info.format}/{info.subtype}"] += 1
print(" ", dict(esrs))
