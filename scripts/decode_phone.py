"""Transcribe the corpus with a phoneme-specialist front end.

Two candidates against the omniASR orthography currently shipping (95.4% in-domain,
77.7% out of domain):

  w2v   facebook/wav2vec2-xlsr-53-espeak-cv-ft -- eSpeak targets, and the sample shows an
        English-leaning inventory: it emits ɹ and no labiovelars, which Kwa languages need
  xeus  changelinglab/PhoneticXeus -- XEUS encoder, IPAPack++ targets, and the sample shows
        k͡p, t͡ʃ, β, ɸ, ɽ and nasalised vowels, which is the right inventory for these languages

Neither runs in sherpa-onnx, which has no wav2vec2 CTC loader. This is a quality question
first; deployment is only worth solving for a winner, and wav2vec2 is the one that would
export cleanly.
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
W2V = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
XEUS = "changelinglab/PhoneticXeus"
# Allosaurus: 45.7 MB against omniASR int8's 350 MB, wav2vec2's 1.26 GB and XEUS's 2.3 GB.
# If its accuracy holds it is the only candidate that is obviously phone-deployable.


def read_audio(cell):
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        n = round(len(w) * SR / sr)
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
    return w


class W2VDecoder:
    """Decoded by hand: the tokenizer class requires the phonemizer library, which exists to
    turn text INTO phonemes and pulls in an espeak-ng system package we do not need."""

    def __init__(self, dev):
        import json as _json

        from huggingface_hub import hf_hub_download
        from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC

        vocab = _json.load(open(hf_hub_download(W2V, "vocab.json")))
        self.inv = {v: k for k, v in vocab.items()}
        self.pad = vocab.get("<pad>", 0)
        self.fe = AutoFeatureExtractor.from_pretrained(W2V)
        self.m = Wav2Vec2ForCTC.from_pretrained(W2V).to(dev).eval()
        self.dev = dev

    def __call__(self, waves):
        out = []
        for w in waves:
            iv = self.fe(w, sampling_rate=SR, return_tensors="pt").input_values.to(self.dev)
            with torch.inference_mode():
                ids = self.m(iv).logits.argmax(-1)[0].tolist()
            ph, prev = [], -1
            for i in ids:
                if i != prev and i != self.pad:
                    tok = self.inv.get(int(i), "")
                    if tok and not tok.startswith("<"):
                        ph.append(tok)
                prev = i
            out.append(" ".join(ph))
        return out


class XeusDecoder:
    def __init__(self, dev):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(XEUS, trust_remote_code=True).to(dev).eval()
        self.dev = dev

    def __call__(self, waves):
        out = []
        for w in waves:
            with torch.inference_mode():
                r = self.m.transcribe(torch.from_numpy(w).to(self.dev))
            # returns a list of dicts; processed_transcript is the cleaned IPA string
            if isinstance(r, list) and r and isinstance(r[0], dict):
                out.append(r[0].get("processed_transcript")
                           or r[0].get("predicted_transcript") or "")
            else:
                out.append(r if isinstance(r, str) else str(r))
        return out


class AlloDecoder:
    """Allosaurus, fed arrays rather than files.

    Its recognize() asserts a .wav path, but that is only the first of four steps -- read,
    featurise, acoustic model, decode. Constructing its Audio object directly skips writing
    362k temporary files. verify_inmemory() checks this against the file path it bypasses,
    because silently diverging from the library's own entry point is exactly the kind of
    shortcut that produces a plausible wrong answer.

    lang_id stays "ipa", the universal inventory. Passing a language would tell the phone
    recogniser the answer the head is supposed to work out.
    """

    def __init__(self, dev):
        import numpy as _np
        from allosaurus.app import read_recognizer
        from allosaurus.audio import Audio
        self.m = read_recognizer()
        self.Audio = Audio
        self._np = _np
        self.gpu = dev == "cuda"
        if self.gpu:
            try:
                self.m.config.device_id = 0
                self.m.am.cuda()
            except Exception:
                self.gpu = False

    def _audio(self, w):
        a = self.Audio()
        a.set_header(sample_rate=SR, sample_size=len(w), channel_number=1, sample_width=2)
        # round, do not truncate: astype() truncates toward zero where soundfile rounds to
        # nearest, and that one-LSB difference is enough to change the decoded phones
        a.samples = self._np.round(
            self._np.clip(w, -1, 1) * 32767).astype(self._np.int16)
        return a

    def __call__(self, waves):
        out = []
        for w in waves:
            feat = self.m.pm.compute(self._audio(w))
            feats = self._np.expand_dims(feat, 0)
            flen = self._np.array([feat.shape[0]], dtype=self._np.int32)
            from allosaurus.am.utils import move_to_tensor
            tf, tl = move_to_tensor([feats, flen], self.m.config.device_id)
            with torch.inference_mode():
                lp = self.m.am(tf, tl)
            lp = lp.cpu().detach().numpy() if self.m.config.device_id >= 0 else lp.detach().numpy()
            out.append(self.m.lm.compute(lp[0], "ipa", 1, emit=1.0, timestamp=False))
        return out


def verify_inmemory():
    """Feeding arrays must give what the library's own file path gives."""
    import tempfile

    import soundfile as _sf
    d = AlloDecoder("cpu")
    f = sorted(glob.glob("/mnt/volume_d2wey28/data/ghana-speech/*/*.parquet"))[0]
    t = pq.read_table(f, columns=["audio"]).to_pydict()
    ok = n = 0
    for cell in t["audio"][:6]:
        w = read_audio(cell)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            _sf.write(tmp.name, w, SR, subtype="PCM_16")
            ref = d.m.recognize(tmp.name, "ipa")
        got = d([w])[0]
        n += 1; ok += (ref == got)
        if ref != got:
            print(f"  MISMATCH\n    file: {ref[:70]}\n    mem : {got[:70]}")
    print(f"in-memory matches the file path on {ok}/{n} clips")
    return ok == n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["w2v", "xeus", "allo"], required=True)
    ap.add_argument("--verify", action="store_true",
                    help="allo only: check the in-memory path against the file path")
    ap.add_argument("--audio-root", default="/mnt/volume_d2wey28/data/ghana-speech")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--keep-ids", default="")
    ap.add_argument("--exclude-prefix", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts-dir", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if args.verify and args.model == "allo":
        raise SystemExit(0 if verify_inmemory() else 1)
    dec = {"w2v": W2VDecoder, "xeus": XeusDecoder, "allo": AlloDecoder}[args.model](dev)
    print(f"loaded {args.model} on {dev} in {time.time()-t0:.0f}s", flush=True)

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
    print(f"{len(shards)} shards\n", flush=True)

    parts = Path(args.parts_dir) if args.parts_dir else None
    if parts:
        parts.mkdir(parents=True, exist_ok=True)

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
                if len(w) < int(0.3 * SR):
                    continue
                waves.append(w); wid.append(t["id"][i]); wdur.append(len(w) / SR)
            if not waves:
                continue
            texts = dec(waves)
            ids.extend(wid); langs.extend([cfg] * len(wid))
            txts.extend(texts); durs.extend(wdur)
            shard_s += sum(wdur)

        if part is not None and len(ids) > row0:
            pq.write_table(pa.table({
                "id": pa.array(ids[row0:], pa.string()),
                "language": pa.array(langs[row0:], pa.string()),
                "text": pa.array(txts[row0:], pa.string()),
                "duration": pa.array(durs[row0:], pa.float64()),
            }), part, compression="zstd")
        done_s += shard_s
        dt = time.time() - t1
        print(f"[{si}/{len(shards)}] {cfg:24} {len(sel):5d} clips {shard_s/60:6.1f} min "
              f"{shard_s/max(dt,1e-9):6.0f}x RT | total {len(ids)} "
              f"{done_s/3600:.1f} h in {(time.time()-t_start)/60:.1f} min", flush=True)

    tbl = pa.table({
        "id": pa.array(ids, pa.string()), "language": pa.array(langs, pa.string()),
        "text": pa.array(txts, pa.string()), "duration": pa.array(durs, pa.float64()),
    })
    if parts:
        done = sorted(parts.glob("*.parquet"))
        if done:
            tbl = pa.concat_tables([pq.read_table(f) for f in done],
                                   promote_options="default")
    pq.write_table(tbl, args.out, compression="zstd")
    ch = [len(s.replace(" ", "")) for s in tbl["text"].to_pylist()]
    el = time.time() - t_start
    print(f"\nwrote {args.out}: {tbl.num_rows} clips, {done_s/3600:.2f} h in {el/60:.1f} min "
          f"({done_s/max(el,1e-9):.0f}x RT)")
    print(f"mean {np.mean(ch):.1f} chars, {sum(1 for c in ch if c < 3)} under 3 chars")


if __name__ == "__main__":
    main()
