"""Language-discriminative features from MMS-LID-4017, two ways.

Zero shot the model scored 0.6826 out of domain on full-length audio but collapsed to
0.2696 at 1.6 seconds. That says the representation carries the signal and the off-the-shelf
4017-way classifier is what fails on short audio -- so a head trained on our data, over our
41 classes, is a different question from using its own head.

Two feature sets, because they differ in how deployable they are:

  emb     mean-pooled hidden states from the encoder, about 1280-dim. The richer
          representation, but shipping it means exporting the encoder ourselves.
  logits  the 4017 language scores. Unusual as a feature vector, and defensible: "how much
          does this sound like each of 4017 languages" is about as language-discriminative
          as a vector gets. It also works today with the ONNX exports that already exist.

Both come from one forward pass, so there is no reason to choose in advance.
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
MODEL = "facebook/mms-lid-4017"


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
    ap.add_argument("--truncate", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    m = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
    print(f"loaded on {dev} in {time.time()-t0:.0f}s", flush=True)

    keep = None
    if args.keep_ids:
        keep = set(pq.read_table(args.keep_ids, columns=["id"])["id"].to_pylist())

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

    ids, langs, embs, logs, durs = [], [], [], [], []
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
            iv = fe(waves, sampling_rate=SR, return_tensors="pt", padding=True)
            am = getattr(iv, "attention_mask", None)
            with torch.inference_mode():
                o = m(iv.input_values.to(dev),
                      attention_mask=am.to(dev) if am is not None else None,
                      output_hidden_states=True)
            # last hidden state, mean-pooled; padding must not dilute the average
            h = o.hidden_states[-1]
            if am is not None:
                # the frame mask is shorter than the sample mask, so derive it by ratio
                fl = m._get_feat_extract_output_lengths(am.sum(-1)).to(h.device)
                msk = (torch.arange(h.shape[1], device=h.device)[None, :] < fl[:, None])
                pooled = (h * msk.unsqueeze(-1)).sum(1) / msk.sum(1, keepdim=True).clamp(min=1)
            else:
                pooled = h.mean(1)
            embs.extend(pooled.cpu().numpy().astype(np.float32))
            logs.extend(o.logits.cpu().numpy().astype(np.float32))
            ids.extend(wid); langs.extend([cfg] * len(wid)); durs.extend(wdur)
            shard_s += sum(wdur)

        if part is not None and len(ids) > row0:
            pq.write_table(pa.table({
                "id": pa.array(ids[row0:], pa.string()),
                "language": pa.array(langs[row0:], pa.string()),
                "emb": pa.array([e.tolist() for e in embs[row0:]], pa.list_(pa.float32())),
                "logits": pa.array([l.tolist() for l in logs[row0:]], pa.list_(pa.float32())),
                "duration": pa.array(durs[row0:], pa.float64()),
            }), part, compression="zstd")
        done_s += shard_s
        dt = time.time() - t1
        print(f"[{si}/{len(shards)}] {cfg:24} {len(sel):5d} clips {shard_s/60:6.1f} min "
              f"{shard_s/max(dt,1e-9):6.0f}x RT | total {len(ids)} in "
              f"{(time.time()-t_start)/60:.1f} min", flush=True)

    tbl = pa.table({
        "id": pa.array(ids, pa.string()), "language": pa.array(langs, pa.string()),
        "emb": pa.array([e.tolist() for e in embs], pa.list_(pa.float32())),
        "logits": pa.array([l.tolist() for l in logs], pa.list_(pa.float32())),
        "duration": pa.array(durs, pa.float64()),
    })
    if parts:
        done = sorted(parts.glob("*.parquet"))
        if done:
            tbl = pa.concat_tables([pq.read_table(f) for f in done],
                                   promote_options="default")
    pq.write_table(tbl, args.out, compression="zstd")
    print(f"\nwrote {args.out}: {tbl.num_rows} clips, "
          f"emb {len(tbl['emb'][0]) if tbl.num_rows else 0}-dim, "
          f"logits {len(tbl['logits'][0]) if tbl.num_rows else 0}-dim, "
          f"{done_s/3600:.2f} h in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
