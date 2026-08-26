"""MMS-LID-4017 zero shot on ghana-speech-eval. No training at all.

Every front end so far has been a transcriber with a classifier bolted on. MMS-LID is a
language classifier outright, and 38 of our 41 languages are among its 4017 labels -- so it
can be pointed at the evaluation set directly.

Three things make it worth interrupting everything for:

  * no transcript, so none of the ASR failure that caps the text heads. The recogniser
    returns nothing for 30-69% of clips in the harder eval domains.
  * Ga, Ahanta and Ikposo are labels here. Those are the three languages our head cannot
    represent and currently answers with a nearest relative, and rejection has been the
    weakest part of the system.
  * Twi and Fante both fall under MMS's `aka`, coarser than our split, so that is scored
    both strictly and leniently rather than quietly picking the flattering one.

Scored three ways, because how the label space is restricted is a real deployment choice:
  free       argmax over all 4017 -- what it says unprompted
  ours       argmax restricted to the 38 it shares with us
  ours+oos   restricted to those 38 plus Ga, Ahanta and Ikposo, which is the honest setting
             for a system that should be able to say "not one of yours"
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import time
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

SR = 16000
MODEL = "facebook/mms-lid-4017"

# eval config -> the MMS label that should win
CONFIG_TO_MMS = {
    "finance_Akuapem_Twi": "aka", "finance_Asante_Twi": "aka", "finance_fante": "aka",
    "finance_ga": "gaa",
    "jw_ahanta_aha": "aha", "jw_dagaare_dga": "dga", "jw_dangme_ada": "ada",
    "jw_ewe_ewe": "ewe", "jw_fante_fat": "aka", "jw_frafra_gur": "gur",
    "jw_ga_gaa": "gaa", "jw_nzema_nzi": "nzi", "jw_sehwi_sfw": "sfw",
    "lds_Asante_Twi": "aka", "lds_Fante_fat": "aka",
    "unicef_Asante_Twi": "aka", "unicef_dagbani": "dag", "unicef_ewe": "ewe",
    "waxal_Asante_Twi": "aka", "waxal_Dagaare_dga": "dga", "waxal_Dagbani_dag": "dag",
    "waxal_Ewe_ewe": "ewe", "waxal_Ikposo_kpo": "kpo",
}
OURS = ("acd ada akp any avn bib bim biv bov bud bwu dag dga ewe gjn gur hau kbp kdh kma "
        "kus lef lip maw mzw naw ncu nko ntr nzi sfw sig sil snw tpm vag xon xsm aka").split()
OUT_OF_SET = ["gaa", "aha", "kpo"]


def clips(root, per_config, truncate):
    out = []
    for f in sorted(glob.glob(f"{root}/*/*.parquet")):
        cfg = f.rsplit("/", 2)[-2]
        if cfg.startswith("bible_") or cfg not in CONFIG_TO_MMS:
            continue
        t = pq.read_table(f, columns=["audio"]).to_pydict()
        got = 0
        for cell in t["audio"]:
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            try:
                w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            except Exception:
                continue
            if w.ndim > 1:
                w = w.mean(axis=1)
            if truncate > 0:
                n = int(truncate * sr)
                if len(w) > n:
                    st = (len(w) - n) // 2
                    w = w[st:st + n]
            if len(w) < int(0.4 * sr):
                continue
            out.append((cfg, w.astype(np.float32)))
            got += 1
            if per_config and got >= per_config:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-config", type=int, default=250)
    ap.add_argument("--truncate", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    root = glob.glob("/mnt/volume_d2wey28/hf-cache/hub/"
                     "datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*")[0]
    cs = clips(root, args.per_config, args.truncate)
    print(f"{len(cs)} clips from {len(set(c for c, _ in cs))} configs"
          f"{f', truncated to {args.truncate}s' if args.truncate else ''}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    m = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
    # transformers hands back integer keys, whatever the JSON had
    id2label = {int(k): v for k, v in m.config.id2label.items()}
    lab2id = {v: k for k, v in id2label.items()}
    print(f"loaded on {dev} in {time.time()-t0:.0f}s, {len(id2label)} labels", flush=True)

    ours_ids = [lab2id[l] for l in OURS if l in lab2id]
    oos_ids = [lab2id[l] for l in OUT_OF_SET if l in lab2id]
    print(f"restricting to {len(ours_ids)} of our labels, +{len(oos_ids)} out-of-set\n",
          flush=True)

    preds = {"free": [], "ours": [], "ours_oos": []}
    gold, cfgs = [], []
    t0, secs = time.time(), 0.0
    for b0 in range(0, len(cs), args.batch):
        chunk = cs[b0:b0 + args.batch]
        waves = [w for _, w in chunk]
        secs += sum(len(w) / SR for w in waves)
        iv = fe(waves, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = m(iv.input_values.to(dev),
                       attention_mask=getattr(iv, "attention_mask", None).to(dev)
                       if getattr(iv, "attention_mask", None) is not None else None).logits
        lg = logits.cpu().numpy()
        for row, (cfg, _) in zip(lg, chunk):
            gold.append(CONFIG_TO_MMS[cfg]); cfgs.append(cfg)
            preds["free"].append(id2label[int(row.argmax())])
            preds["ours"].append(id2label[ours_ids[int(row[ours_ids].argmax())]])
            both = ours_ids + oos_ids
            preds["ours_oos"].append(id2label[both[int(row[both].argmax())]])
        if (b0 // args.batch) % 20 == 0:
            print(f"  {b0+len(chunk)}/{len(cs)}  {secs/max(time.time()-t0,1e-9):.0f}x RT",
                  flush=True)

    print(f"\ndecoded {secs/60:.0f} min of audio in {(time.time()-t0)/60:.1f} min "
          f"({secs/(time.time()-t0):.0f}x RT)\n")

    in_set = [i for i, g in enumerate(gold) if g not in OUT_OF_SET]
    oos = [i for i, g in enumerate(gold) if g in OUT_OF_SET]

    for mode in ("free", "ours", "ours_oos"):
        p = preds[mode]
        strict = np.mean([p[i] == gold[i] for i in in_set])
        # lenient: Akan is one label in MMS, so Twi and Fante cannot be told apart
        lenient = np.mean([p[i] == gold[i] or (gold[i] == "aka" and p[i] == "aka")
                           for i in in_set])
        print(f"{mode:9} in-set accuracy  strict {strict:.4f}   "
              f"(Akan-lenient {lenient:.4f})  on {len(in_set)} clips")

    print("\nout-of-set languages -- can it say they are not ours?")
    for mode in ("free", "ours_oos"):
        p = preds[mode]
        right = np.mean([p[i] == gold[i] for i in oos]) if oos else 0
        print(f"  {mode:9} names them correctly {right:.4f} on {len(oos)} clips")

    print("\nby domain (ours_oos):")
    dom = defaultdict(lambda: [0, 0])
    for i in in_set:
        d = cfgs[i].split("_")[0]
        dom[d][0] += preds["ours_oos"][i] == gold[i]; dom[d][1] += 1
    for d in sorted(dom):
        ok, n = dom[d]
        print(f"  {d:10} {ok/n:.3f}  ({ok}/{n})")

    if args.out:
        json.dump({"n": len(gold), "truncate": args.truncate,
                   "modes": {k: float(np.mean([preds[k][i] == gold[i] for i in in_set]))
                             for k in preds}}, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
