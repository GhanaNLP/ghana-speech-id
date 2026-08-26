"""Extract VoxLingua107 ECAPA embeddings, as a front end that needs no transcript at all.

Every text front end tried so far has to produce a good transcript first, and the recogniser
fails on exactly the hard cases -- it returned nothing for 30-69% of clips in some eval
domains, and the best out-of-domain accuracy on 1.6 seconds of speech is 57%. An acoustic
model skips that stage entirely.

The encoder is FROZEN. Fine-tuning it on 2,329 h of single-domain Bible audio is what
destroyed ghana-ipa-asr's ability to read anything else, and the same corpus would pose the
same risk here. Frozen is also cheap: extract once, then a head trains in minutes.

The risk this design must answer is the speaker confound. The corpus has roughly one
narrator per language, so an acoustic model can learn narrator identity and score near
perfectly in domain while being useless. That is why the earlier text pipeline exists at
all. It is now detectable: ghana-speech-eval is five unrelated recording projects with
different speakers, so a model keying on voices will show a near-perfect in-domain score and
collapse there.

Audio is optionally truncated, because the comparison criterion is 1.6 seconds -- the
acoustic equivalent of the 20 characters the text heads are judged on.
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


def read_audio(cell):
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
    ap.add_argument("--audio-root", default="/mnt/volume_d2wey28/data/ghana-speech")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--keep-ids",
                    default="/mnt/volume_d2wey28/projects/ghana-speech-id/data/ipa_text.parquet")
    ap.add_argument("--exclude-prefix", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts-dir", default="")
    ap.add_argument("--truncate", type=float, default=0.0,
                    help="seconds to keep from the centre; 0 uses the whole clip")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    from speechbrain.inference.classifiers import EncoderClassifier

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="/mnt/volume_d2wey28/hf-cache/voxlingua107",
        run_opts={"device": dev})
    enc.mods.eval()
    print(f"loaded ECAPA on {dev} in {time.time()-t0:.0f}s", flush=True)

    keep = None
    if args.keep_ids:
        keep = set(pq.read_table(args.keep_ids, columns=["id"])["id"].to_pylist())
        print(f"restricting to {len(keep)} ids", flush=True)

    shards = sorted(glob.glob(f"{args.audio_root}/*/*.parquet"))
    if args.exclude_prefix:
        shards = [f for f in shards
                  if not f.rsplit("/", 2)[-2].startswith(args.exclude_prefix)]
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    parts = Path(args.parts_dir) if args.parts_dir else None
    if parts:
        parts.mkdir(parents=True, exist_ok=True)
    print(f"{len(shards)} shards\n", flush=True)

    ids, langs, embs, durs = [], [], [], []
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
            stem = shard.rsplit("/", 1)[-1].replace(".parquet", "")
            t["id"] = [f"{cfg}_{stem}_{i:06d}" for i in range(len(t[args.audio_col]))]
        sel = [i for i, _id in enumerate(t["id"]) if keep is None or _id in keep]
        if not sel:
            continue

        row0, t1, shard_s = len(ids), time.time(), 0.0
        for b0 in range(0, len(sel), args.batch):
            waves, wid, wdur = [], [], []
            for i in sel[b0:b0 + args.batch]:
                try:
                    w = read_audio(t[args.audio_col][i])
                except Exception:
                    continue
                if args.truncate > 0:
                    n = int(args.truncate * SR)
                    if len(w) > n:
                        st = (len(w) - n) // 2
                        w = w[st:st + n]
                if len(w) < int(0.4 * SR):
                    continue
                waves.append(w); wid.append(t["id"][i]); wdur.append(len(w) / SR)
            if not waves:
                continue
            # pad to the longest, and pass relative lengths so the pooling ignores padding
            mx = max(len(w) for w in waves)
            batch = torch.zeros(len(waves), mx)
            rel = torch.zeros(len(waves))
            for k, w in enumerate(waves):
                batch[k, :len(w)] = torch.from_numpy(w)
                rel[k] = len(w) / mx
            with torch.inference_mode():
                e = enc.encode_batch(batch.to(dev), rel.to(dev)).squeeze(1).cpu().numpy()
            ids.extend(wid); langs.extend([cfg] * len(wid))
            embs.extend(e.astype(np.float32)); durs.extend(wdur)
            shard_s += sum(wdur)

        if part is not None and len(ids) > row0:
            pq.write_table(pa.table({
                "id": pa.array(ids[row0:], pa.string()),
                "language": pa.array(langs[row0:], pa.string()),
                "emb": pa.array([e.tolist() for e in embs[row0:]],
                                pa.list_(pa.float32())),
                "duration": pa.array(durs[row0:], pa.float64()),
            }), part, compression="zstd")
        done_s += shard_s
        dt = time.time() - t1
        print(f"[{si}/{len(shards)}] {cfg:24} {len(sel):5d} clips {shard_s/60:6.1f} min "
              f"{shard_s/max(dt,1e-9):6.0f}x RT | total {len(ids)} in "
              f"{(time.time()-t_start)/60:.1f} min", flush=True)

    tbl = pa.table({
        "id": pa.array(ids, pa.string()),
        "language": pa.array(langs, pa.string()),
        "emb": pa.array([e.tolist() for e in embs], pa.list_(pa.float32())),
        "duration": pa.array(durs, pa.float64()),
    })
    if parts:
        done = sorted(parts.glob("*.parquet"))
        if done:
            tbl = pa.concat_tables([pq.read_table(f) for f in done],
                                   promote_options="default")
    pq.write_table(tbl, args.out, compression="zstd")
    el = time.time() - t_start
    dim = len(tbl["emb"][0]) if tbl.num_rows else 0
    print(f"\nwrote {args.out}: {tbl.num_rows} clips, {dim}-dim embeddings, "
          f"{done_s/3600:.2f} h audio in {el/60:.1f} min ({done_s/max(el,1e-9):.0f}x RT)")


if __name__ == "__main__":
    main()
