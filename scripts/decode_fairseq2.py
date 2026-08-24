"""Decode with a base omniASR model through fairseq2.

sherpa-onnx publishes the 1B only as int8, and int8 has no CUDA kernels, so it runs at 4-5x
realtime on either provider -- about 200 hours for this corpus. The fairseq2 pipeline does
128x. sherpa remains the on-device runtime; this is for bulk decoding only.

`lang` is left as None so decoding stays language-agnostic. Passing a language code would
make an LID front-end circular.
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
import torch

SR = 16000
# the fairseq2 pipeline raises above 40 s; split below that and concatenate the pieces.
# Only 0.03% of clips are affected, but one of them aborts the whole run.
MAX_SEC = 35.0


def decode_cell(cell):
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    return w, int(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="omniASR_CTC_1B")
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--keep-ids", default="")
    ap.add_argument("--exclude-prefix", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard-chunk", type=int, default=2000,
                    help="clips held in memory per transcribe call")
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--parts-dir", default="",
                    help="write one parquet per shard here and skip shards already done, so "
                         "a crash resumes instead of restarting")
    args = ap.parse_args()

    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    pipe = ASRInferencePipeline(args.card, device="cuda", dtype=torch.bfloat16)
    print(f"loaded {args.card} in {time.time()-t0:.0f}s", flush=True)

    keep = None
    if args.keep_ids:
        keep = set(pq.read_table(args.keep_ids, columns=["id"])["id"].to_pylist())
        print(f"restricting to {len(keep)} ids", flush=True)

    shards = sorted(glob.glob(f"{args.audio_root}/*/*.parquet"))
    if args.exclude_prefix:
        n0 = len(shards)
        shards = [f for f in shards
                  if not f.rsplit("/", 2)[-2].startswith(args.exclude_prefix)]
        print(f"excluded {n0-len(shards)} shards under {args.exclude_prefix}*")
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    print(f"{len(shards)} shards\n", flush=True)

    parts = Path(args.parts_dir) if args.parts_dir else None
    if parts:
        parts.mkdir(parents=True, exist_ok=True)

    out_id, out_lang, out_txt, out_dur = [], [], [], []
    t_start, done_s = time.time(), 0.0

    for si, shard in enumerate(shards, 1):
        cfg = shard.rsplit("/", 2)[-2]
        part_path = None
        if parts:
            stem = shard.rsplit("/", 1)[-1].replace(".parquet", "")
            part_path = parts / f"{cfg}__{stem}.parquet"
            if part_path.exists():
                continue
        have = set(pq.ParquetFile(shard).schema_arrow.names)
        cols = [args.audio_col] + (["id"] if "id" in have else [])
        t = pq.read_table(shard, columns=cols).to_pydict()
        if "id" not in t:
            stem = shard.rsplit("/", 1)[-1].replace(".parquet", "")
            t["id"] = [f"{cfg}_{stem}_{i:06d}" for i in range(len(t[args.audio_col]))]
        sel = [i for i, _id in enumerate(t["id"]) if keep is None or _id in keep]
        if not sel:
            continue

        t1, shard_s, row0 = time.time(), 0.0, len(out_id)
        for c0 in range(0, len(sel), args.shard_chunk):
            chunk = sel[c0: c0 + args.shard_chunk]
            items, owner, ids, durs = [], [], [], []
            for i in chunk:
                try:
                    w, sr = decode_cell(t[args.audio_col][i])
                except Exception:
                    continue
                if len(w) < int(0.3 * sr):
                    continue
                k = len(ids)
                ids.append(t["id"][i]); durs.append(len(w) / sr)
                lim = int(MAX_SEC * sr)
                for st in range(0, len(w), lim) if len(w) > lim else (0,):
                    piece = w[st:st + lim] if len(w) > lim else w
                    if len(piece) < int(0.3 * sr):
                        continue
                    items.append({"waveform": piece, "sample_rate": sr})
                    owner.append(k)
            if not items:
                continue
            texts = pipe.transcribe(items, batch_size=args.batch)
            joined = [""] * len(ids)
            for o, tx in zip(owner, texts):
                joined[o] = (joined[o] + " " + tx).strip() if joined[o] else tx
            out_id.extend(ids); out_lang.extend([cfg] * len(ids))
            out_txt.extend(joined); out_dur.extend(durs)
            shard_s += sum(durs)

        if part_path is not None and len(out_id) > row0:
            pq.write_table(pa.table({
                "id": pa.array(out_id[row0:], pa.string()),
                "language": pa.array(out_lang[row0:], pa.string()),
                "text": pa.array(out_txt[row0:], pa.string()),
                "duration": pa.array(out_dur[row0:], pa.float64()),
            }), part_path, compression="zstd")
        done_s += shard_s
        dt = time.time() - t1
        print(f"[{si}/{len(shards)}] {cfg:24} {len(sel):5d} clips {shard_s/60:6.1f} min "
              f"{shard_s/max(dt,1e-9):6.0f}x RT | total {len(out_id)} clips "
              f"{done_s/3600:.1f} h in {(time.time()-t_start)/60:.1f} min", flush=True)

    tbl = pa.table({
        "id": pa.array(out_id, pa.string()),
        "language": pa.array(out_lang, pa.string()),
        "text": pa.array(out_txt, pa.string()),
        "duration": pa.array(out_dur, pa.float64()),
    })
    if parts:
        # a resumed run only decoded the shards that were missing; fold the rest back in
        done = sorted(parts.glob("*.parquet"))
        if done:
            tbl = pa.concat_tables([pq.read_table(f) for f in done], promote_options="default")
            print(f"assembled {len(done)} shard files")
    pq.write_table(tbl, args.out, compression="zstd")
    out_txt = tbl["text"].to_pylist()
    out_dur = tbl["duration"].to_pylist()
    out_id = tbl["id"].to_pylist()

    chars = [len(s.replace(" ", "")) for s in out_txt]
    empty = sum(1 for c in chars if c < 3)
    el = time.time() - t_start
    print(f"\nwrote {args.out}: {len(out_id)} clips, {done_s/3600:.2f} h in {el/60:.1f} min "
          f"({done_s/max(el,1e-9):.0f}x RT)")
    print(f"mean {np.mean(chars):.1f} chars/clip, "
          f"{empty} ({empty/max(len(chars),1):.1%}) under 3 chars")


if __name__ == "__main__":
    main()
