"""Do phoneme-specialist front ends transcribe Ghanaian speech better than omniASR?

omniASR is a general multilingual ASR. These two are trained specifically to emit IPA:

  facebook/wav2vec2-xlsr-53-espeak-cv-ft   XLSR-53, eSpeak phoneme targets, 53 languages
  changelinglab/PhoneticXeus               XEUS encoder + self-conditioned CTC on IPAPack++,
                                           17k hours, 70+ languages

Neither runs in sherpa-onnx -- it has no wav2vec2 CTC loader and no ONNX build exists -- so
this goes through torch. Deployment is a separate problem and only worth solving if one of
them wins.

Sampled across languages before committing to a 740-hour decode.
"""
from __future__ import annotations

import glob
import io
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

CONFIGS = ["Asante_Twi_twi", "Ewe_ewe", "Dagbani_dag", "Ga_gaa", "Kasem_xsm"]
N_PER = 4
SR = 16000


def load_clips():
    out = []
    for cfg in CONFIGS:
        fs = sorted(glob.glob(f"/mnt/volume_d2wey28/data/ghana-speech/{cfg}/*.parquet"))
        if not fs:
            continue
        t = pq.read_table(fs[0], columns=["audio", "text"]).to_pydict()
        got = 0
        for cell, ref in zip(t["audio"], t["text"]):
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            try:
                w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            except Exception:
                continue
            if w.ndim > 1:
                w = w.mean(axis=1)
            if not (4.0 <= len(w) / sr <= 10.0):
                continue
            out.append((cfg, w.astype(np.float32), sr, ref))
            got += 1
            if got >= N_PER:
                break
    return out


def main():
    clips = load_clips()
    print(f"{len(clips)} clips from {len(set(c[0] for c in clips))} languages\n", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- wav2vec2 eSpeak
    print("=" * 70)
    print("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    try:
        import json as _json

        from huggingface_hub import hf_hub_download
        from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC

        MID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        t0 = time.time()
        # The tokenizer class insists on the `phonemizer` library, which exists to turn TEXT
        # into phonemes. We go the other way -- ids to phones -- which is a vocab lookup and
        # a CTC collapse, so decode by hand and skip the dependency and its espeak-ng
        # system package.
        vocab = _json.load(open(hf_hub_download(MID, "vocab.json")))
        inv = {v: k for k, v in vocab.items()}
        pad_id = vocab.get("<pad>", 0)
        fe = AutoFeatureExtractor.from_pretrained(MID)
        m = Wav2Vec2ForCTC.from_pretrained(MID).to(dev).eval()
        print(f"  loaded in {time.time()-t0:.0f}s, vocab {len(vocab)} phones", flush=True)

        def ctc_decode(ids):
            out, prev = [], -1
            for i in ids:
                if i != prev and i != pad_id:
                    tok = inv.get(int(i), "")
                    if tok and not tok.startswith("<"):
                        out.append(tok)
                prev = i
            return out

        secs = tdec = 0.0
        for cfg, w, sr, ref in clips:
            t1 = time.time()
            iv = fe(w, sampling_rate=sr, return_tensors="pt").input_values.to(dev)
            with torch.inference_mode():
                ids = m(iv).logits.argmax(-1)[0].tolist()
            ph = ctc_decode(ids)
            tdec += time.time() - t1; secs += len(w) / sr
            print(f"  {cfg:18} {len(w)/sr:4.1f}s {len(ph):3d} phones "
                  f"({len(ph)/(len(w)/sr):4.1f}/s)  {' '.join(ph)[:58]}")
        print(f"  -> {secs/max(tdec,1e-9):.0f}x RT")
    except Exception as e:
        print(f"  FAILED {type(e).__name__}: {str(e)[:200]}")

    # ---------------- PhoneticXeus
    print("=" * 70)
    print("changelinglab/PhoneticXeus")
    try:
        from transformers import AutoModel
        t0 = time.time()
        m2 = AutoModel.from_pretrained("changelinglab/PhoneticXeus",
                                       trust_remote_code=True).to(dev).eval()
        print(f"  loaded in {time.time()-t0:.0f}s", flush=True)
        secs = tdec = 0.0
        for cfg, w, sr, ref in clips:
            t1 = time.time()
            wav = torch.from_numpy(w).to(dev)
            with torch.inference_mode():
                out = m2.transcribe(wav) if hasattr(m2, "transcribe") else m2(wav)
            txt = out if isinstance(out, str) else str(out)
            tdec += time.time() - t1; secs += len(w) / sr
            print(f"  {cfg:18} {len(w)/sr:4.1f}s  {txt[:70]}")
        print(f"  -> {secs/max(tdec,1e-9):.0f}x RT")
    except Exception as e:
        print(f"  FAILED {type(e).__name__}: {str(e)[:300]}")

    print("=" * 70)
    print("reference orthography for the same clips:")
    for cfg, w, sr, ref in clips[:5]:
        print(f"  {cfg:18} {ref[:64]}")


if __name__ == "__main__":
    main()
