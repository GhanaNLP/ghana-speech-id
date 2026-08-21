"""Combine base-model transcripts of the Ghanaian corpus and English into one training set.

Two corrections, both carried over from the IPA pipeline because the same traps apply.

  size    English is sampled to the largest Ghanaian class. Earlier measurement showed more
          English keeps improving recall without hurting precision, and recall is what
          matters for detecting English when spoken.

  length  decode_base.py does not crop audio, so English clips decode ~2x longer than
          Ghanaian ones. Left alone the head learns "long string = English", a cue that
          vanishes on short real-world utterances. Each English transcript is trimmed to a
          character length drawn from the Ghanaian distribution.
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LABEL = "English_eng"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gh", required=True, help="base decode of the Ghanaian corpus")
    ap.add_argument("--en", default=None, help="base decode of English; omit for no English")
    ap.add_argument("--en-n", type=int, default=0, help="0 uses the largest Ghanaian class")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gh = pq.read_table(args.gh, columns=["id", "language", "text", "duration"])
    langs = gh["language"].to_pylist()
    texts = gh["text"].to_pylist()

    counts = {}
    for l in langs:
        key = "Twi_twi" if l.endswith("_twi") else l
        counts[key] = counts.get(key, 0) + 1
    sizes = sorted(counts.values())
    print(f"{len(sizes)} Ghanaian classes after Twi merge: "
          f"min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")

    gh_lens = [len(t) for t in texts if t]
    print(f"Ghanaian transcripts: mean {np.mean(gh_lens):.0f} chars")

    tables = [gh]
    if args.en:
        en = pq.read_table(args.en, columns=["id", "text", "duration"])
        eids, etexts, edurs = (en["id"].to_pylist(), en["text"].to_pylist(),
                               en["duration"].to_pylist())
        usable = [i for i, t in enumerate(etexts) if t and len(t) >= 10]
        target = args.en_n or sizes[-1]
        rng.shuffle(usable)
        usable = usable[:target]
        print(f"English available {len(etexts)}, usable {len(usable)} after sampling to "
              f"{target}")
        print(f"  before trim: mean {np.mean([len(etexts[i]) for i in usable]):.0f} chars")

        oi, ot, od = [], [], []
        for i in usable:
            t = etexts[i]
            want = int(rng.choice(gh_lens))
            if want < len(t):
                st = rng.randint(0, len(t) - want)
                t = t[st:st + want]
            oi.append(eids[i]); ot.append(t)
            od.append(float(edurs[i]) * len(t) / max(len(etexts[i]), 1))
        print(f"  after trim:  mean {np.mean([len(t) for t in ot]):.0f} chars")
        tables.append(pa.table({
            "id": pa.array(oi, pa.string()),
            "language": pa.array([LABEL] * len(oi), pa.string()),
            "text": pa.array(ot, pa.string()),
            "duration": pa.array(od, pa.float64()),
        }))

    out = pa.concat_tables(tables, promote_options="default")
    pq.write_table(out, args.out, compression="zstd")
    n_lang = len(set(out["language"].to_pylist()))
    print(f"\nwrote {args.out}: {out.num_rows} rows, {n_lang} languages")


if __name__ == "__main__":
    main()
