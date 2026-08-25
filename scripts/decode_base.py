"""Transcribe the training corpus with a base omniASR model instead of the fine-tune.

ghana-ipa-asr lost its generalisation to the Bible domain (see compare_asr.py: 1.13 chars/s
and 35% empty on JW audio where the base model it came from does 8.36 and 0%). A head
trained on its output inherits that. So the corpus is re-transcribed with the un-finetuned
base, and the head is rebuilt on top.

Audio comes from the local ghana-speech copy rather than a download, filtered by id to
exactly the clips ghana-speech-ipa used, so the new heads are comparable to the old one.

Base omniASR emits orthography, not IPA, and takes no language code -- decoding is
language-agnostic, so using it as an LID front-end is not circular.
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

MODELS = "/mnt/volume_d2wey28/projects/ghana-speech-id/models"
# int8 on CUDA runs at 7x -- slower than CPU's 17x -- because quantised operators have no
# CUDA kernels and onnxruntime places them on CPU node by node, paying a device transfer at
# every boundary. The fp32 build does 111x. Use fp32 whenever decoding on GPU; the int8
# builds are for on-device inference, where they are the right choice.
PRESETS = {
    "300m": (f"{MODELS}/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12",
             "model.onnx"),
    "300m-int8": (f"{MODELS}/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12",
                  "model.int8.onnx"),
    "1b-int8": (f"{MODELS}/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05",
                "model.int8.onnx"),
    # ZIPA: Zipformer CTC phone recognition from the same Icefall lineage as sherpa-onnx,
    # so from_zipformer_ctc loads it directly. 70.7 MB at int8 for the small model, a fifth
    # of the omniASR int8 we ship, and the exports were published for phones.
    # fp16 is lossless against fp32 here -- 119/120 transcripts identical, 0.02% of
    # characters -- while being 1.7x faster and half the size, so fp32 is never the right
    # choice. int8 is faster still but diverges on 1.39% of characters.
    "zipa-small": (f"{MODELS}/zipa-small", "model.int8.onnx"),
    "zipa-small-fp16": (f"{MODELS}/zipa-small", "model.fp16.onnx"),
    "zipa-small-fp32": (f"{MODELS}/zipa-small", "model.onnx"),
    "zipa-large": (f"{MODELS}/zipa-large", "model.int8.onnx"),
    "zipa-large-fp16": (f"{MODELS}/zipa-large", "model.fp16.onnx"),
    "zipa-large-fp32": (f"{MODELS}/zipa-large", "model.onnx"),
}
ZIPFORMER = {k for k in PRESETS if k.startswith("zipa")}
SR = 16000


def decode_cell(cell):
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        n = round(len(w) * SR / sr)
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="300m", choices=list(PRESETS))
    ap.add_argument("--audio-root", default="/mnt/volume_d2wey28/data/ghana-speech")
    ap.add_argument("--keep-ids",
                    default="/mnt/volume_d2wey28/projects/ghana-speech-id/data/ipa_text.parquet",
                    help="restrict to the ids this parquet lists; '' decodes everything")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--exclude-prefix", default="",
                    help="skip config directories starting with this; used to leave the "
                         "bible_* training domain out of the evaluation decode")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--parts-dir", default="",
                    help="one parquet per shard; lets several processes share the work and "
                         "lets a crash resume instead of restarting")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="this worker handles shards where i %% shard-count == shard-index")
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--window", type=float, default=0.0,
                    help="split clips longer than this and concatenate; 0 disables")
    args = ap.parse_args()

    import sherpa_onnx as so

    d, mf = PRESETS[args.model]
    print(f"loading {args.model} ({mf}) on {args.provider} ...", flush=True)
    factory = (so.OfflineRecognizer.from_zipformer_ctc if args.model in ZIPFORMER
               else so.OfflineRecognizer.from_omnilingual_asr_ctc)
    rec = factory(model=f"{d}/{mf}", tokens=f"{d}/tokens.txt",
                  num_threads=args.threads, provider=args.provider)
    print("ready", flush=True)

    keep = None
    if args.keep_ids:
        keep = set(pq.read_table(args.keep_ids, columns=["id"])["id"].to_pylist())
        print(f"restricting to {len(keep)} ids from {os.path.basename(args.keep_ids)}",
              flush=True)

    shards = sorted(glob.glob(f"{args.audio_root}/*/*.parquet"))
    if args.exclude_prefix:
        before = len(shards)
        shards = [f for f in shards
                  if not f.rsplit("/", 2)[-2].startswith(args.exclude_prefix)]
        print(f"excluded {before - len(shards)} shards under {args.exclude_prefix}*")
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if args.shard_count > 1:
        shards = [f for i, f in enumerate(shards) if i % args.shard_count == args.shard_index]
        print(f"worker {args.shard_index}/{args.shard_count}: {len(shards)} shards",
              flush=True)
    else:
        print(f"{len(shards)} shards under {args.audio_root}\n", flush=True)

    parts = Path(args.parts_dir) if args.parts_dir else None
    if parts:
        parts.mkdir(parents=True, exist_ok=True)

    out_id, out_lang, out_txt, out_dur = [], [], [], []
    t_start, done_s = time.time(), 0.0

    for si, shard in enumerate(shards, 1):
        cfg = shard.rsplit("/", 2)[-2]
        part = None
        if parts:
            stem = shard.rsplit("/", 1)[-1].replace(".parquet", "")
            part = parts / f"{cfg}__{stem}.parquet"
            if part.exists():
                continue
        have = set(pq.ParquetFile(shard).schema_arrow.names)
        cols = [args.audio_col] + (["id"] if "id" in have else [])
        t = pq.read_table(shard, columns=cols).to_pydict()
        if "id" not in t:
            # ghana-speech-eval carries no id column; synthesise a stable key from the
            # config and row index so downstream joins still have one
            stem = shard.rsplit("/", 1)[-1].replace(".parquet", "")
            t["id"] = [f"{cfg}_{stem}_{i:06d}" for i in range(len(t[args.audio_col]))]
        sel = [i for i, _id in enumerate(t["id"]) if keep is None or _id in keep]
        if not sel:
            continue
        t0, shard_s, row0 = time.time(), 0.0, len(out_id)
        for b0 in range(0, len(sel), args.batch):
            chunk = sel[b0: b0 + args.batch]
            streams, ids, durs = [], [], []
            for i in chunk:
                try:
                    w = decode_cell(t[args.audio_col][i])
                except Exception:
                    continue
                if len(w) < int(0.3 * SR):
                    continue
                s = rec.create_stream()
                s.accept_waveform(SR, w)
                streams.append(s); ids.append(t["id"][i]); durs.append(len(w) / SR)
            if not streams:
                continue
            rec.decode_streams(streams)
            for s, _id, dur in zip(streams, ids, durs):
                out_id.append(_id); out_lang.append(cfg)
                out_txt.append(s.result.text); out_dur.append(dur)
            shard_s += sum(durs)
        if part is not None and len(out_id) > row0:
            pq.write_table(pa.table({
                "id": pa.array(out_id[row0:], pa.string()),
                "language": pa.array(out_lang[row0:], pa.string()),
                "text": pa.array(out_txt[row0:], pa.string()),
                "duration": pa.array(out_dur[row0:], pa.float64()),
            }), part, compression="zstd")
        done_s += shard_s
        dt = time.time() - t0
        el = time.time() - t_start
        print(f"[{si}/{len(shards)}] {cfg:24} {len(sel):5d} clips {shard_s/60:6.1f} min "
              f"{shard_s/max(dt,1e-9):6.0f}x RT | total {len(out_id)} clips "
              f"{done_s/3600:.1f} h in {el/60:.1f} min", flush=True)

    tbl = pa.table({
        "id": pa.array(out_id, pa.string()),
        "language": pa.array(out_lang, pa.string()),
        "text": pa.array(out_txt, pa.string()),
        "duration": pa.array(out_dur, pa.float64()),
    })
    if parts and args.shard_count > 1:
        print(f"worker {args.shard_index} done; assemble separately once all finish")
    pq.write_table(tbl, args.out, compression="zstd")

    chars = [len(s.replace(" ", "")) for s in out_txt]
    empty = sum(1 for c in chars if c < 3)
    el = time.time() - t_start
    print(f"\nwrote {args.out}: {len(out_id)} clips, {done_s/3600:.2f} h audio "
          f"in {el/60:.1f} min ({done_s/max(el,1e-9):.0f}x RT)")
    print(f"mean {np.mean(chars):.1f} chars/clip, "
          f"{np.mean([c/max(d,.01) for c, d in zip(chars, out_dur)]):.2f} chars/s, "
          f"{empty} clips ({empty/max(len(chars),1):.1%}) under 3 chars")


if __name__ == "__main__":
    main()
