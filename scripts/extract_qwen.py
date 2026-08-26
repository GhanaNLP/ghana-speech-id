"""Utterance embeddings from the Qwen3-ASR encoder, skipping the decoder entirely.

For language ID the text is never needed, so the 756 MB decoder is dead weight: only
conv_frontend (44 MB) and encoder (182 MB) are loaded, which is 226 MB against the 350 MB
omniASR front end we ship.

Chain, matching infer_qwen3_asr.py in the qwen3-onnx-export project:
    audio -> 128-bin log-mel -> conv_frontend -> 896-dim tokens -> encoder -> 1024-dim
    -> mean pool over valid tokens

The mel features come from Qwen's own processor rather than being reimplemented. Guessing
window and normalisation parameters would produce embeddings that look fine and mean
nothing, which is the failure mode worth spending a dependency to avoid.
"""
from __future__ import annotations

import argparse
import glob
import io
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000
QDIR = ("/mnt/volume_d2wey28/projects/ghana-speech-id/models/"
        "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25")


def feat_to_audio_tokens_len(feat_len, chunk_size=100):
    """Lifted from infer_qwen3_asr.py: how many encoder tokens a mel length produces."""
    def conv3(n):
        x = (int(n) + 1) // 2
        x = (x + 1) // 2
        return (x + 1) // 2

    def aftercnn(x):
        x = (x - 1) // 2 + 1
        x = (x - 1) // 2 + 1
        return (x - 1) // 2 + 1

    n = np.asarray(feat_len, dtype=np.int64)
    full, rem = n // chunk_size, n % chunk_size
    return np.maximum(full * conv3(chunk_size) + aftercnn(rem), 0).astype(np.int64)


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
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--provider", default="cpu",
                    help="CPU by default, and for two reasons: it is how this would be "
                         "served, and int8 has no CUDA kernels so the cuda provider falls "
                         "back per node and runs slower -- 7x against 17x when measured on "
                         "omniASR")
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoFeatureExtractor
    # Qwen3-ASR uses a WhisperFeatureExtractor: 128 mels, n_fft 400, hop 160. AutoProcessor
    # returns only the tokenizer here, so take the feature extractor directly.
    fe = AutoFeatureExtractor.from_pretrained("Qwen/Qwen3-ASR-0.6B")
    print(f"features: {type(fe).__name__}, {fe.feature_size} mels, hop {fe.hop_length}",
          flush=True)
    # Its default pads every clip to chunk_length=30 s. On a 1.6 s clip that is 95% padding
    # and the pooled embedding would be mostly silence, so pad only to the batch maximum
    # and let feature_attention_mask carry the real lengths.

    prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
            if args.provider == "cuda" else ["CPUExecutionProvider"])
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    conv = ort.InferenceSession(f"{QDIR}/conv_frontend.onnx", so, providers=prov)
    enc = ort.InferenceSession(f"{QDIR}/encoder.int8.onnx", so, providers=prov)
    print(f"conv + encoder loaded on {enc.get_providers()[0]}", flush=True)

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

            f = fe(waves, sampling_rate=SR, return_tensors="np", padding="longest",
                   return_attention_mask=True, truncation=False)
            mel = np.asarray(f["input_features"], dtype=np.float32).transpose(0, 2, 1)
            fmask = np.asarray(f.get("feature_attention_mask",
                                     np.ones(mel.shape[:2], dtype=np.int64)))
            (cout,) = conv.run(["conv_output"], {"input_features": mel})
            a_len = feat_to_audio_tokens_len((fmask != 0).sum(axis=1))
            A = cout.shape[1]
            tok = (np.arange(A)[None, :] < a_len[:, None])
            (af,) = enc.run(["audio_features"],
                            {"input_features": cout,
                             "feature_attention_mask": tok.astype(np.bool_)})
            # mean pool over valid tokens only; padding must not dilute the embedding
            m = tok[..., None].astype(np.float32)
            pooled = (af * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-6)

            ids.extend(wid); langs.extend([cfg] * len(wid))
            embs.extend(pooled.astype(np.float32)); durs.extend(wdur)
            shard_s += sum(wdur)

        if part is not None and len(ids) > row0:
            pq.write_table(pa.table({
                "id": pa.array(ids[row0:], pa.string()),
                "language": pa.array(langs[row0:], pa.string()),
                "emb": pa.array([e.tolist() for e in embs[row0:]], pa.list_(pa.float32())),
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
        "duration": pa.array(durs, pa.float64()),
    })
    if parts:
        done = sorted(parts.glob("*.parquet"))
        if done:
            tbl = pa.concat_tables([pq.read_table(f) for f in done],
                                   promote_options="default")
    pq.write_table(tbl, args.out, compression="zstd")
    print(f"\nwrote {args.out}: {tbl.num_rows} clips, "
          f"{len(tbl['emb'][0]) if tbl.num_rows else 0}-dim, "
          f"{done_s/3600:.2f} h in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
