"""Fold Ghanaian English in as a 42nd class.

Source is ghanaopendata/ghana-english-speech-ipa -- ghana-english-tts-clean2 already
phonemised with the released Ghana IPA CTC model. That is the right phonemiser and the only
one that matters: the head only ever sees what THIS recogniser emits, English distortions
and all. Nothing needs re-decoding.

Two corrections applied on the way in.

  size    52,855 English clips against a median Ghanaian class of 8,540 would make English
          six times the median. An oversized English class biases the head toward calling
          Ghanaian speech English, which is the costliest error here, so it is sampled down
          to roughly the median.

  length  English clips average 147 units, Ghanaian ones about 55. Left alone the head can
          learn "long string = English", a cue that does not exist at inference on short
          utterances. Each English clip is randomly cropped to a length drawn from the
          Ghanaian distribution, so length carries no class information.
"""
from __future__ import annotations
import argparse, glob, random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = ("/mnt/volume_d2wey28/hf_cache/hub/datasets--ghanaopendata--ghana-english-speech-ipa"
       "/snapshots/*/English_eng/*.parquet")
GH = "/mnt/volume_d2wey28/projects/ghana-speech-id/data/ipa_text.parquet"
LABEL = "English_eng"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="English clips to keep; 0 uses the median Ghanaian class size")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--gh", default=GH)
    ap.add_argument("--out", default="/mnt/volume_d2wey28/projects/ghana-speech-id/data/ipa_text_en.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-crop", action="store_true",
                    help="skip IPA cropping. Correct when the source came from "
                         "rephonemise_english.py, which already cropped the AUDIO to the "
                         "Ghanaian duration distribution -- cropping again would halve the "
                         "clips and reintroduce a length cue in the other direction")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gh = pq.read_table(args.gh, columns=["id", "language", "ipa", "duration", "split"])
    gh_lens, gh_counts = [], {}
    for lang, ipa in zip(gh["language"].to_pylist(), gh["ipa"].to_pylist()):
        gh_counts[lang] = gh_counts.get(lang, 0) + 1
        if ipa:
            gh_lens.append(len(ipa.split()))
    # class sizes as the head will see them, i.e. after the Twi merge
    merged = {}
    for lang, n in gh_counts.items():
        key = "Twi_twi" if lang.endswith("_twi") else lang
        merged[key] = merged.get(key, 0) + n
    sizes = sorted(merged.values())
    target = args.n or sizes[len(sizes) // 2]
    print(f"{len(sizes)} Ghanaian classes: min={sizes[0]} median={sizes[len(sizes)//2]} "
          f"max={sizes[-1]}")
    print(f"target English clips: {target}")

    files = sorted(glob.glob(args.src))
    if not files:
        raise SystemExit(f"no English shards matched {args.src}")
    en = pq.read_table(files, columns=["id", "ipa", "duration"])
    ids = en["id"].to_pylist()
    ipas = en["ipa"].to_pylist()
    durs = en["duration"].to_pylist()
    keep = [i for i, s in enumerate(ipas) if s and len(s.split()) >= 3]
    print(f"English available: {len(ipas)} clips ({len(keep)} usable)")
    rng.shuffle(keep)
    keep = keep[:target]

    # Crop to the Ghanaian length distribution so length is not a giveaway.
    gh_lens_arr = np.array(gh_lens)
    if args.no_crop:
        print("--no-crop: audio was already cropped before decoding")
    print(f"length before crop: English mean {np.mean([len(ipas[i].split()) for i in keep]):.1f} "
          f"units vs Ghanaian mean {gh_lens_arr.mean():.1f}")

    out_ids, out_ipa, out_dur = [], [], []
    for i in keep:
        units = ipas[i].split()
        if args.no_crop:
            out_ids.append(ids[i]); out_ipa.append(" ".join(units))
            out_dur.append(float(durs[i]))
            continue
        want = int(rng.choice(gh_lens_arr.tolist()))
        if want < len(units):
            start = rng.randint(0, len(units) - want)
            units = units[start:start + want]
        frac = len(units) / max(len(ipas[i].split()), 1)
        out_ids.append(ids[i])
        out_ipa.append(" ".join(units))
        out_dur.append(float(durs[i]) * frac)
    print(f"length after crop:  English mean "
          f"{np.mean([len(s.split()) for s in out_ipa]):.1f} units")

    n = len(out_ids)
    # match the corpus's own validation proportion so the split logic sees a normal class
    n_val = max(1, round(n * 0.0045))
    splits = ["validation"] * n_val + ["train"] * (n - n_val)
    rng.shuffle(splits)

    en_tbl = pa.table({
        "id": pa.array(out_ids, pa.string()),
        "language": pa.array([LABEL] * n, pa.string()),
        "ipa": pa.array(out_ipa, pa.string()),
        "duration": pa.array(out_dur, pa.float64()),
        "split": pa.array(splits, pa.string()),
    })
    combined = pa.concat_tables([gh.select(["id", "language", "ipa", "duration", "split"]),
                                 en_tbl], promote_options="default")
    pq.write_table(combined, args.out, compression="zstd")
    print(f"\nwrote {args.out}: {combined.num_rows} rows, "
          f"{len(set(combined['language'].to_pylist()))} languages "
          f"({len(sizes)} after Twi merge, +1 English)")


if __name__ == "__main__":
    main()
