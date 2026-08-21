"""Character error rate of the candidate front-ends against ghana-speech-eval references.

Output density said the base omniASR models read this audio and the fine-tune does not, but
density cannot tell a full transcript from confident nonsense. The eval set carries a `text`
column, and omniASR emits orthography, so CER is measurable rather than inferred.

ghana-ipa-asr is excluded: it emits IPA, and CER against orthographic text would be
meaningless for it. The live question is which base model becomes the new front-end.

Normalisation before scoring is deliberately light -- casefold, strip punctuation, collapse
whitespace. Ghanaian orthographies use ɛ ɔ ŋ and tone-free spellings that a heavier
normaliser would mangle, and stripping them would flatter every model equally but hide the
script-drift failures we are trying to detect.
"""
from __future__ import annotations

import argparse
import glob
import io
import re
import time
import unicodedata

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

MODELS = "/mnt/volume_d2wey28/projects/ghana-speech-id/models"
OMNI300 = f"{MODELS}/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"
OMNI1B = f"{MODELS}/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05"
WHISPER = f"{MODELS}/sherpa-onnx-whisper-small"

CONFIGS = ["bible_Fante_fat", "lds_Fante_fat", "jw_fante_fat", "jw_ewe_ewe",
           "finance_Asante_Twi", "waxal_Asante_Twi", "waxal_Dagbani_dag",
           "unicef_Asante_Twi"]

PUNCT = re.compile(r"[^\w\sɛɔŋʋɣƴɖɗƙɓ']", re.UNICODE)
LATIN = re.compile(r"[a-zɛɔŋʋɣƴɖɗƙɓ]", re.IGNORECASE)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").casefold()
    s = PUNCT.sub(" ", s)
    return " ".join(s.split())


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def non_latin_share(s: str) -> float:
    """Share of letters outside the Latin/Ghanaian set -- catches script drift."""
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if not LATIN.match(c)) / len(letters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-config", type=int, default=40)
    ap.add_argument("--crop", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import sherpa_onnx as so

    root = (glob.glob("/mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*")
            or glob.glob("/mnt/volume_d2wey28/hf_cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*"))[0]

    rec = {}
    for name, d in [("omniASR 300M", OMNI300), ("omniASR 1B v2", OMNI1B)]:
        rec[name] = so.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=f"{d}/model.int8.onnx", tokens=f"{d}/tokens.txt", num_threads=args.threads)
    rec["Whisper small"] = so.OfflineRecognizer.from_whisper(
        encoder=f"{WHISPER}/small-encoder.int8.onnx",
        decoder=f"{WHISPER}/small-decoder.int8.onnx",
        tokens=f"{WHISPER}/small-tokens.txt",
        language="", task="transcribe", num_threads=args.threads)
    print(f"{len(rec)} recognisers ready\n", flush=True)

    totals = {n: [0, 0, []] for n in rec}   # edits, ref chars, per-clip drift
    print(f"{'config':22} " + " ".join(f"{n:>16}" for n in rec))
    for cfg in CONFIGS:
        fs = sorted(glob.glob(f"{root}/{cfg}/*.parquet"))
        if not fs:
            continue
        t = pq.read_table(fs[0], columns=["audio", "text"]).to_pydict()
        clips = []
        for cell, ref in zip(t["audio"], t["text"]):
            r = norm(ref)
            if len(r) < 10:
                continue
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            try:
                w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            except Exception:
                continue
            if w.ndim > 1:
                w = w.mean(axis=1)
            if args.crop > 0:
                want = int(args.crop * sr)
                if len(w) > want:
                    st = (len(w) - want) // 2
                    w = w[st:st + want]
                    # reference no longer matches a cropped clip, so cropping is only for
                    # the density comparison, never for CER
                    raise SystemExit("--crop cannot be used with CER: the reference text "
                                     "describes the whole clip")
            clips.append((w, sr, r))
            if len(clips) >= args.per_config:
                break
        if not clips:
            continue

        cells = []
        for name, r in rec.items():
            ed = nc = 0
            drift = []
            for w, sr, ref in clips:
                s = r.create_stream()
                s.accept_waveform(sr, w)
                r.decode_stream(s)
                hyp = norm(s.result.text)
                ed += edit_distance(ref, hyp)
                nc += len(ref)
                drift.append(non_latin_share(s.result.text))
            totals[name][0] += ed
            totals[name][1] += nc
            totals[name][2].extend(drift)
            cells.append(f"{ed/max(nc,1):15.1%}")
        print(f"{cfg:22} " + " ".join(cells), flush=True)

    print(f"\n{'OVERALL CER':22} " +
          " ".join(f"{totals[n][0]/max(totals[n][1],1):15.1%}" for n in rec))
    print(f"{'non-Latin chars':22} " +
          " ".join(f"{np.mean(totals[n][2]):15.2%}" for n in rec))
    print("\nCER above 100% means the model emitted more wrong characters than the reference "
          "has;\nfor languages a model does not know that is expected, not a bug.")
    print("non-Latin share flags script drift into the shared 1600-language vocabulary.")


if __name__ == "__main__":
    main()
