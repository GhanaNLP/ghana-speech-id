"""Re-decode Ghanaian English with ghana-ipa-asr so English can be a class of this head.

The cached ghana-english-speech-ipa `ipa` column was produced by an English phonemiser --
it contains æ ʌ ɹ ɑː θ ɜː ɒ ᵻ ɚ, none of which exist in the 176-unit Ghanaian inventory --
so it is unusable here. The head only ever sees what ghana-ipa-asr emits, so the English
class has to be built from that recogniser's output, English distortions and all. Only the
audio column of those shards is reused.

Audio is randomly cropped to a duration drawn from the Ghanaian corpus before decoding,
which does two jobs at once:
  * keeps clips inside the encoder's usable window -- batch.py measures a 30 s clip
    decoding to a single unit, with nothing raised
  * removes length as a class cue. English clips run ~15 s against a Ghanaian mean of
    ~7 s, and left alone the head would learn "long string = English".
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

SRC = ("/mnt/volume_d2wey28/hf_cache/hub/datasets--ghanaopendata--ghana-english-speech-ipa"
       "/snapshots/*/English_eng/*.parquet")
GH = "/mnt/volume_d2wey28/projects/ghana-speech-id/data/ipa_text.parquet"
LABEL = "English_eng"
SR = 16000

_asr = None


def get_asr():
    global _asr
    if _asr is None:
        from ghana_ipa_asr import GhanaIPAASR
        _asr = GhanaIPAASR.load()
    return _asr


def decode_audio(cell):
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        n = round(len(w) * SR / sr)
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
    return w


def work(job):
    shard, want, durations, seed = job
    rng = random.Random(seed)
    t0 = time.time()
    t = pq.read_table(shard, columns=["id", "audio"]).to_pydict()
    ids, cells = t["id"], t["audio"]
    idx = list(range(len(ids)))
    rng.shuffle(idx)
    idx = idx[:want]

    waves, keep_ids, keep_dur = [], [], []
    for i in idx:
        try:
            w = decode_audio(cells[i])
        except Exception:
            continue
        target = float(rng.choice(durations))
        n = int(target * SR)
        if n < len(w):
            s = rng.randint(0, len(w) - n)
            w = w[s:s + n]
        if len(w) < int(0.4 * SR):
            continue
        waves.append(w); keep_ids.append(ids[i]); keep_dur.append(len(w) / SR)
    t_read = time.time() - t0

    asr = get_asr()
    out, t0, B = [], time.time(), 16
    for i in range(0, len(waves), B):
        chunk = waves[i:i + B]
        for j, tr in enumerate(asr.transcribe_batch(chunk, sample_rate=SR)):
            out.append({"id": keep_ids[i + j], "ipa": tr.spaced(punctuation=True),
                        "duration": keep_dur[i + j]})
    dt = time.time() - t0
    secs = sum(keep_dur)
    return os.path.basename(shard), out, t_read, dt, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--gh", default=GH)
    ap.add_argument("--n", type=int, default=10000, help="clips to decode (before filtering)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="/mnt/volume_d2wey28/projects/ghana-speech-id/data/english_ipa_gh.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gh = pq.read_table(args.gh, columns=["duration"])["duration"].to_pylist()
    durations = [d for d in gh if d and 0.5 <= d <= 20.0]
    print(f"Ghanaian duration pool: n={len(durations)} mean={np.mean(durations):.1f}s "
          f"median={np.median(durations):.1f}s")

    files = sorted(glob.glob(args.src))
    files = [f for f in files if "/train-" in f]
    if not files:
        raise SystemExit(f"no shards matched {args.src}")
    per = max(1, args.n // len(files))
    print(f"{len(files)} local shards, ~{per} clips each -> ~{per*len(files)} total", flush=True)

    # sample the duration pool down; shipping 369k floats to every worker is wasteful
    pool = random.Random(args.seed).sample(durations, min(20000, len(durations)))
    jobs = [(f, per, pool, args.seed + i) for i, f in enumerate(files)]

    rows, t_all = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, out, t_read, dt, secs in ex.map(work, jobs):
            rows.extend(out)
            print(f"  {name}: {len(out):5d} clips {secs/60:6.1f} min audio | "
                  f"read {t_read:4.0f}s decode {dt:5.0f}s ({secs/max(dt,1e-9):5.1f}x RT) | "
                  f"total {len(rows)}", flush=True)

    units = [len(r["ipa"].split()) for r in rows]
    ok = sum(1 for u in units if u >= 3)
    pq.write_table(pa.table({
        "id": pa.array([r["id"] for r in rows], pa.string()),
        "ipa": pa.array([r["ipa"] for r in rows], pa.string()),
        "duration": pa.array([r["duration"] for r in rows], pa.float64()),
    }), args.out, compression="zstd")
    print(f"\nwrote {args.out}: {len(rows)} clips ({ok} with >=3 units), "
          f"mean {np.mean(units):.1f} units/clip, {(time.time()-t_all)/60:.1f} min wall")

    # sanity: everything must now be inside the 176-unit inventory
    import collections
    seen = collections.Counter(u for r in rows for u in r["ipa"].split())
    print(f"distinct units emitted: {len(seen)}")
    print("most common:", ", ".join(f"{u}({n})" for u, n in seen.most_common(12)))


if __name__ == "__main__":
    main()
