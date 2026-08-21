"""Why does JW audio decode to nothing? Compare it against LDS, which decodes fine.

Peak amplitude looked healthy for both, but peak cannot tell speech from silence with a
click in it. RMS, the share of frames above a speech-level floor, and the spectral centroid
can.
"""
import glob, io
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

root = (glob.glob("/mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*")
        or glob.glob("/mnt/volume_d2wey28/hf_cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*"))[0]


def stats(cfg, n=40):
    f = sorted(glob.glob(f"{root}/{cfg}/*.parquet"))[0]
    t = pq.read_table(f, columns=["audio"]).to_pydict()
    rms, peak, active, centroid, durs, dc = [], [], [], [], [], []
    for cell in t["audio"][:n]:
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        if len(w) < sr // 4:
            continue
        durs.append(len(w) / sr)
        rms.append(float(np.sqrt(np.mean(w ** 2))))
        peak.append(float(np.abs(w).max()))
        dc.append(float(np.mean(w)))
        # share of 20 ms frames carrying speech-level energy
        fl = int(0.02 * sr)
        fr = w[: len(w) // fl * fl].reshape(-1, fl)
        fe = np.sqrt((fr ** 2).mean(axis=1))
        active.append(float((fe > 0.01).mean()))
        # spectral centroid: speech sits low, noise/music sits high
        sp = np.abs(np.fft.rfft(w[: 1 << 15] if len(w) > (1 << 15) else w))
        fq = np.fft.rfftfreq(len(w[: 1 << 15]) if len(w) > (1 << 15) else len(w), 1 / sr)
        centroid.append(float((sp * fq).sum() / max(sp.sum(), 1e-9)))
    return (len(rms), np.mean(durs), np.mean(rms), np.mean(peak), np.mean(active),
            np.mean(centroid), np.mean(dc))


print(f"{'config':22} {'n':>4} {'dur':>6} {'rms':>7} {'peak':>6} {'active':>7} "
      f"{'centroid':>9} {'dc_off':>9}")
for cfg in ["lds_Asante_Twi", "waxal_Asante_Twi", "finance_Asante_Twi",
            "jw_ewe_ewe", "jw_fante_fat", "jw_ga_gaa", "unicef_Asante_Twi"]:
    try:
        n, d, r, p, a, c, o = stats(cfg)
        print(f"{cfg:22} {n:4d} {d:6.1f} {r:7.4f} {p:6.3f} {a:6.1%} {c:8.0f}Hz {o:9.5f}")
    except Exception as e:
        print(f"{cfg:22} failed: {type(e).__name__} {str(e)[:50]}")
print("\nactive = share of 20 ms frames above a speech-level energy floor")
print("centroid = spectral centre of mass; conversational speech usually 500-1500 Hz")
