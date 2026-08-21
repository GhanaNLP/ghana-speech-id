"""GPU phoneme decoding for the corpora this head is built and evaluated on.

sherpa-onnx on CPU manages ~2.5x realtime per worker, so ~20x with eight of them. The
fairseq2 checkpoint on the H200 does orders of magnitude better, which is the difference
between an afternoon and a coffee break on the 27 hours of audio in ghana-speech-eval.

Model loading, normalisation, batching and CTC collapse are imported from
ghana_ipa_asr.batch rather than reimplemented: that module already handles the three things
that silently corrupt output otherwise -- per-utterance normalisation the encoder requires,
slicing the 9812-wide CTC head down to the 176 real tokens, and zeroing layer_drop.

Two sources:
  --shards   local parquet with an audio column (the cached English audio)
  --hf-eval  ghananlpcommunity/ghana-speech-eval, skipping the bible_* training domain

--crop-to draws a duration from a reference corpus and randomly crops each clip to it. Used
for English, where clips run ~15 s against a Ghanaian mean of ~7 s: without it the head
learns "long string = English", and clips also drift past the window where the encoder stays
reliable. Not used for evaluation, where clips must be left as they are.
"""
from __future__ import annotations

import argparse
import glob
import io
import random
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

SR = 16000


def decode_cell(cell):
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    return w, int(sr)


def resample_to_16k(w, sr):
    if sr == SR:
        return w
    n = int(round(len(w) * SR / sr))
    return np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", help="glob of parquet files holding audio")
    ap.add_argument("--hf-eval", action="store_true",
                    help="decode ghananlpcommunity/ghana-speech-eval (bible_* skipped)")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 decodes everything")
    ap.add_argument("--crop-to", default=None,
                    help="parquet with a duration column; crop each clip to a duration "
                         "sampled from it")
    ap.add_argument("--budget", type=int, default=SR * 900,
                    help="padded samples per GPU batch (default 900 s)")
    ap.add_argument("--max-rows", type=int, default=256)
    ap.add_argument("--model", default="ghananlpcommunity/ghana-speech-phoneme-asr")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ghana_ipa_asr.batch import (ctc_collapse, load_model, make_batches, resolve_model,
                                     run_batch)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rng = random.Random(args.seed)

    root = resolve_model(args.model)
    model, keep, vocab = load_model(root, "cuda")
    print(f"model loaded, head sliced to {keep} tokens", flush=True)

    durations = None
    if args.crop_to:
        d = pq.read_table(args.crop_to, columns=["duration"])["duration"].to_pylist()
        durations = [x for x in d if x and 0.5 <= x <= 20.0]
        print(f"crop pool: n={len(durations)} mean={np.mean(durations):.1f}s", flush=True)

    # ---------------------------------------------------------------- gather audio
    items = []  # (id, wave16k, group)
    t0 = time.time()

    if args.hf_eval:
        from datasets import Audio, load_dataset
        from huggingface_hub import HfApi
        cfgs = sorted(c for c in HfApi().dataset_info(
            "ghananlpcommunity/ghana-speech-eval").config_names if not c.startswith("bible_"))
        print(f"{len(cfgs)} non-bible configs", flush=True)
        for cfg in cfgs:
            ds = load_dataset("ghananlpcommunity/ghana-speech-eval", cfg, split="eval")
            if args.n:
                ds = ds.select(range(min(args.n, len(ds))))
            ds = ds.cast_column("audio", Audio(sampling_rate=SR))
            for i, r in enumerate(ds):
                items.append((f"{cfg}_{i:06d}",
                              np.asarray(r["audio"]["array"], dtype=np.float32), cfg))
            print(f"  {cfg:26} {len(ds):5d} clips loaded", flush=True)
    else:
        files = sorted(glob.glob(args.shards))
        files = [f for f in files if "/validation-" not in f]
        if not files:
            raise SystemExit(f"no shards matched {args.shards}")
        per = (args.n // len(files)) if args.n else 0
        print(f"{len(files)} shards, {per or 'all'} clips each", flush=True)
        for f in files:
            t = pq.read_table(f, columns=["id", args.audio_col]).to_pydict()
            idx = list(range(len(t["id"])))
            if per:
                rng.shuffle(idx)
                idx = idx[:per]
            for i in idx:
                try:
                    w, sr = decode_cell(t[args.audio_col][i])
                except Exception:
                    continue
                w = resample_to_16k(w, sr)
                if durations is not None:
                    n = int(float(rng.choice(durations)) * SR)
                    if n < len(w):
                        s = rng.randint(0, len(w) - n)
                        w = w[s:s + n]
                if len(w) >= int(0.4 * SR):
                    items.append((t["id"][i], w, f.rsplit("/", 1)[-1]))

    secs = sum(len(w) for _, w, _ in items) / SR
    print(f"\n{len(items)} clips, {secs/3600:.2f} h audio, read in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    # ---------------------------------------------------------------- decode
    lengths = [len(w) for _, w, _ in items]
    batches = make_batches(lengths, list(range(len(items))), args.budget, args.max_rows)
    print(f"{len(batches)} GPU batches", flush=True)

    out = [None] * len(items)
    t0, done_s, cache = time.time(), 0.0, {}
    for b, batch in enumerate(batches, 1):
        waves = [items[i][1] for i in batch]
        ids = run_batch(model, keep, waves, [SR] * len(waves), cache, "cuda")
        for i, o in zip(batch, ids):
            out[i] = " ".join(ctc_collapse(o, vocab))
        done_s += sum(len(w) for w in waves) / SR
        if b % 20 == 0 or b == len(batches):
            dt = time.time() - t0
            print(f"  batch {b}/{len(batches)}  {done_s/3600:.2f} h decoded  "
                  f"{done_s/max(dt,1e-9):.0f}x RT  eta "
                  f"{(secs-done_s)/max(done_s/max(dt,1e-9),1e-9)/60:.1f} min", flush=True)

    units = [len(s.split()) for s in out if s]
    pq.write_table(pa.table({
        "id": pa.array([i for i, _, _ in items], pa.string()),
        "group": pa.array([g for _, _, g in items], pa.string()),
        "ipa": pa.array(out, pa.string()),
        "duration": pa.array([len(w) / SR for _, w, _ in items], pa.float64()),
    }), args.out, compression="zstd")

    dt = time.time() - t0
    print(f"\nwrote {args.out}: {len(items)} clips, {secs/3600:.2f} h audio in "
          f"{dt/60:.1f} min ({secs/max(dt,1e-9):.0f}x RT)")
    print(f"mean {np.mean(units):.1f} units/clip, "
          f"{sum(1 for u in units if u < 3)} clips under 3 units")


if __name__ == "__main__":
    main()
