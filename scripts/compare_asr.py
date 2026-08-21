"""Is the fine-tuned recogniser's out-of-domain collapse catastrophic forgetting?

ghana-ipa-asr fine-tuned omniASR_W2V_300M on 2,329 h of Bible audio with the encoder
unfrozen from step 0. On held-out Bible audio it emits ~9.4 phoneme units/s; on
ghana-speech-eval it emits ~3.7/s, and on some configs a median of zero. If the base model
it was fine-tuned from reads that same audio fine, the generalisation was trained away, and
the fix is to build the language head on a front-end that kept it.

Compared on identical clips:
  ghana-ipa-asr    the fine-tune, IPA units
  omniASR CTC 300M the base it came from, 1600+ languages
  omniASR CTC 1B   larger, newer checkpoint
  Whisper small    a control: knows none of these languages, but if it emits a full
                   transcript then the audio carries clear speech and the others' silence
                   is their own failure

Output alphabets differ, so accuracy is not comparable. Output *density* is: a model that
has kept its generalisation produces a transcript, one that has not produces near-silence.
"""
from __future__ import annotations

import argparse
import glob
import io
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

MODELS = "/mnt/volume_d2wey28/projects/ghana-speech-id/models"
OMNI300 = f"{MODELS}/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"
OMNI1B = f"{MODELS}/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05"
WHISPER = f"{MODELS}/sherpa-onnx-whisper-small"

# bible_* is the fine-tune's training domain and the reference point; the rest are not.
CONFIGS = ["bible_Fante_fat", "lds_Fante_fat", "jw_fante_fat",
           "finance_Asante_Twi", "waxal_Asante_Twi", "unicef_Asante_Twi"]


def load_clips(root, cfg, n, crop=0.0):
    """crop>0 takes a fixed-length window from the middle of each clip.

    Configs differ from 3.3 s (finance) to 22.5 s (unicef), and length is not neutral:
    Whisper pads everything to 30 s, and ghana-ipa-asr's own batch.py windows at 6 s by
    default because the encoder degrades on long audio. Without this control, a model could
    look bad on unicef for being handed 22 s clips rather than for failing the domain.
    """
    fs = sorted(glob.glob(f"{root}/{cfg}/*.parquet"))
    if not fs:
        return []
    t = pq.read_table(fs[0], columns=["audio"]).to_pydict()
    out = []
    for cell in t["audio"][: n * 3]:
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        try:
            w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception:
            continue
        if w.ndim > 1:
            w = w.mean(axis=1)
        if len(w) < sr:            # skip sub-second clips; too little to judge density on
            continue
        if crop > 0:
            want = int(crop * sr)
            if len(w) > want:                  # centre window: avoids lead-in silence
                st = (len(w) - want) // 2
                w = w[st:st + want]
            elif len(w) < want * 0.5:          # too short to compare at this length
                continue
        out.append((w, sr))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-config", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--show", type=int, default=1, help="example transcripts per config")
    ap.add_argument("--crop", type=float, default=0.0,
                    help="crop every clip to this many seconds so length is identical "
                         "across configs and models")
    args = ap.parse_args()

    import sherpa_onnx as so
    from ghana_ipa_asr import GhanaIPAASR

    root = (glob.glob("/mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*")
            or glob.glob("/mnt/volume_d2wey28/hf_cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*"))[0]

    print("loading recognisers ...", flush=True)
    rec = {"ghana-ipa-asr (finetuned)": GhanaIPAASR.load()}
    for name, d in [("omniASR CTC 300M (base)", OMNI300), ("omniASR CTC 1B v2", OMNI1B)]:
        try:
            rec[name] = so.OfflineRecognizer.from_omnilingual_asr_ctc(
                model=f"{d}/model.int8.onnx", tokens=f"{d}/tokens.txt",
                num_threads=args.threads)
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__} {str(e)[:90]}")
    try:
        rec["Whisper small"] = so.OfflineRecognizer.from_whisper(
            encoder=f"{WHISPER}/small-encoder.int8.onnx",
            decoder=f"{WHISPER}/small-decoder.int8.onnx",
            tokens=f"{WHISPER}/small-tokens.txt",
            language="", task="transcribe", num_threads=args.threads)
    except Exception as e:
        print(f"  Whisper: FAILED {type(e).__name__} {str(e)[:90]}")
    print(f"  {len(rec)} recognisers ready\n", flush=True)

    def transcribe(name, r, w, sr):
        if name.startswith("ghana-ipa-asr"):
            return r.transcribe(w, sr).spaced(punctuation=False)
        s = r.create_stream()
        s.accept_waveform(sr, w)
        r.decode_stream(s)
        return s.result.text

    rows = {}
    for cfg in CONFIGS:
        clips = load_clips(root, cfg, args.per_config, args.crop)
        if not clips:
            print(f"{cfg}: no clips"); continue
        secs = sum(len(w) / sr for w, sr in clips)
        tag = "TRAINING DOMAIN" if cfg.startswith("bible_") else ""
        if args.crop:
            tag += f"  [all clips cropped to {args.crop:g}s]"
        print(f"########## {cfg}  ({len(clips)} clips, {secs:.0f}s) {tag} ##########",
              flush=True)
        for name, r in rec.items():
            t0 = time.time()
            outs = [transcribe(name, r, w, sr) for w, sr in clips]
            dt = time.time() - t0
            # characters excluding spaces: comparable across IPA and orthography
            chars = [len(o.replace(" ", "")) for o in outs]
            rate = [c / (len(w) / sr) for c, (w, sr) in zip(chars, clips)]
            empty = sum(1 for c in chars if c < 3) / len(chars)
            rows.setdefault(cfg, {})[name] = (float(np.median(rate)), empty)
            print(f"  {name:26} {np.median(rate):6.2f} chars/s   "
                  f"{empty:5.1%} produced nothing   ({secs/max(dt,1e-9):5.1f}x RT)",
                  flush=True)
            for o in outs[: args.show]:
                print(f"      {o[:88]}")
        print(flush=True)

    print("\n########## median chars/s (excluding spaces) ##########")
    names = list(rec)
    print(f"{'config':22} " + " ".join(f"{n[:17]:>18}" for n in names))
    for cfg in CONFIGS:
        if cfg not in rows:
            continue
        cells = " ".join(f"{rows[cfg][n][0]:9.2f}/{rows[cfg][n][1]:6.0%}"
                         if n in rows[cfg] else f"{'-':>18}" for n in names)
        print(f"{cfg:22} {cells}")
    print("\nformat: chars-per-second / share of clips producing nothing")


if __name__ == "__main__":
    main()
