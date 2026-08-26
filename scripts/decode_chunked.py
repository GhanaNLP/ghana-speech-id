"""Decode audio in short overlapping windows, so the head trains on what inference gives it.

Transcript chunking -- cutting a whole-clip transcript into 40-character windows -- was
worth +4.7 points at one second of speech. But it trains on windows of a *well transcribed*
utterance, and that is not what a short recording produces. Measured on this corpus, the
recogniser returns nothing for 8.6% of 0-5 s clips against 0.0% of 5-10 s clips: short audio
transcribes worse, not just shorter.

Cutting the audio first reproduces that degradation. Each window is decoded independently
and becomes its own training example, carrying the clip's language label.

Decoded with the served quantisation by default. The omniASR head was trained on fp32 and
served int8, which cost 1.3 points and flipped 9% of predictions; that gap should not be
rebuilt into the replacement.
"""
from __future__ import annotations

import argparse
import glob
import io
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000


def read_audio(cell):
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        n = round(len(w) * SR / sr)
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
    return w


def windows(w, win_s, stride_s, min_s):
    """Overlapping windows. A clip shorter than one window is kept whole rather than
    dropped -- short clips are exactly the case being trained for."""
    n, step = int(win_s * SR), int(stride_s * SR)
    if len(w) <= n:
        return [w] if len(w) >= int(min_s * SR) else []
    out = [w[i:i + n] for i in range(0, len(w) - n + 1, step)]
    tail = w[-n:]
    if not np.array_equal(out[-1], tail):
        out.append(tail)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="300m-int8")
    ap.add_argument("--audio-root", default="/mnt/volume_d2wey28/data/ghana-speech")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--keep-ids",
                    default="/mnt/volume_d2wey28/projects/ghana-speech-id/data/ipa_text.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts-dir", default="")
    ap.add_argument("--window", type=float, default=3.0, help="window length in seconds")
    ap.add_argument("--stride", type=float, default=1.5, help="hop; half the window is 50%% overlap")
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    import sherpa_onnx as so
    from decode_base import PRESETS, ZIPFORMER  # one source of truth for model paths

    d, mf = PRESETS[args.model]
    factory = (so.OfflineRecognizer.from_zipformer_ctc if args.model in ZIPFORMER
               else so.OfflineRecognizer.from_omnilingual_asr_ctc)
    rec = factory(model=f"{d}/{mf}", tokens=f"{d}/tokens.txt",
                  num_threads=args.threads, provider=args.provider)
    print(f"loaded {args.model} on {args.provider}; windows {args.window}s "
          f"stride {args.stride}s", flush=True)

    keep = None
    if args.keep_ids:
        keep = set(pq.read_table(args.keep_ids, columns=["id"])["id"].to_pylist())

    shards = sorted(glob.glob(f"{args.audio_root}/*/*.parquet"))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if args.shard_count > 1:
        shards = [f for i, f in enumerate(shards)
                  if i % args.shard_count == args.shard_index]
    parts = Path(args.parts_dir) if args.parts_dir else None
    if parts:
        parts.mkdir(parents=True, exist_ok=True)
    print(f"{len(shards)} shards", flush=True)

    ids, langs, txts, durs = [], [], [], []
    t_start, done_s = time.time(), 0.0
    for si, shard in enumerate(shards, 1):
        cfg = shard.rsplit("/", 2)[-2]
        part = None
        if parts:
            stem = shard.rsplit("/", 1)[-1].replace(".parquet", "")
            part = parts / f"{cfg}__{stem}.parquet"
            if part.exists():
                continue
        t = pq.read_table(shard, columns=["id", args.audio_col]).to_pydict()
        sel = [i for i, _id in enumerate(t["id"]) if keep is None or _id in keep]
        if not sel:
            continue

        row0, t1, shard_s = len(ids), time.time(), 0.0
        pend_w, pend_id = [], []
        for i in sel:
            try:
                w = read_audio(t[args.audio_col][i])
            except Exception:
                continue
            for k, win in enumerate(windows(w, args.window, args.stride, args.min_seconds)):
                pend_w.append(win); pend_id.append(f"{t['id'][i]}_w{k:02d}")
                if len(pend_w) >= args.batch:
                    ss = []
                    for x in pend_w:
                        s = rec.create_stream(); s.accept_waveform(SR, x); ss.append(s)
                    rec.decode_streams(ss)
                    for s, wid, x in zip(ss, pend_id, pend_w):
                        ids.append(wid); langs.append(cfg)
                        txts.append(s.result.text); durs.append(len(x) / SR)
                        shard_s += len(x) / SR
                    pend_w, pend_id = [], []
        if pend_w:
            ss = []
            for x in pend_w:
                s = rec.create_stream(); s.accept_waveform(SR, x); ss.append(s)
            rec.decode_streams(ss)
            for s, wid, x in zip(ss, pend_id, pend_w):
                ids.append(wid); langs.append(cfg)
                txts.append(s.result.text); durs.append(len(x) / SR)
                shard_s += len(x) / SR

        if part is not None and len(ids) > row0:
            pq.write_table(pa.table({
                "id": pa.array(ids[row0:], pa.string()),
                "language": pa.array(langs[row0:], pa.string()),
                "text": pa.array(txts[row0:], pa.string()),
                "duration": pa.array(durs[row0:], pa.float64()),
            }), part, compression="zstd")
        done_s += shard_s
        dt = time.time() - t1
        print(f"[{si}/{len(shards)}] {cfg:24} {len(sel):5d} clips -> "
              f"{len(ids)-row0:6d} windows {shard_s/60:6.1f} min "
              f"{shard_s/max(dt,1e-9):5.0f}x RT | total {len(ids)} "
              f"in {(time.time()-t_start)/60:.1f} min", flush=True)

    tbl = pa.table({
        "id": pa.array(ids, pa.string()), "language": pa.array(langs, pa.string()),
        "text": pa.array(txts, pa.string()), "duration": pa.array(durs, pa.float64()),
    })
    pq.write_table(tbl, args.out, compression="zstd")
    ch = [len(s.replace(" ", "")) for s in txts]
    print(f"\nwrote {args.out}: {len(ids)} windows, {done_s/3600:.2f} h, "
          f"mean {np.mean(ch):.1f} chars, "
          f"{sum(1 for c in ch if c < 3)} ({sum(1 for c in ch if c < 3)/max(len(ch),1):.1%}) empty")


if __name__ == "__main__":
    main()
