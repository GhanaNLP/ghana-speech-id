"""Pull ground-truth orthography from the local ghana-speech copy.

A head trained on this is the ceiling for the orthographic approach: base omniASR emits
orthography, so its output is this text plus recognition noise. If perfect orthography does
not beat the IPA head, switching front-ends cannot help and the problem is elsewhere.

Restricted to the same ids and given the same contiguous split, so it is comparable to
every other number in this project.
"""
import glob
import time

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/mnt/volume_d2wey28/data/ghana-speech"
KEEP = "/mnt/volume_d2wey28/projects/ghana-speech-id/data/ipa_text.parquet"
OUT = "/mnt/volume_d2wey28/projects/ghana-speech-id/data/text_truth.parquet"

keep = set(pq.read_table(KEEP, columns=["id"])["id"].to_pylist())
print(f"{len(keep)} ids to keep")

shards = sorted(glob.glob(f"{ROOT}/*/*.parquet"))
print(f"{len(shards)} shards")

ids, langs, texts, durs = [], [], [], []
t0 = time.time()
for i, sh in enumerate(shards, 1):
    cfg = sh.rsplit("/", 2)[-2]
    t = pq.read_table(sh, columns=["id", "text", "duration"]).to_pydict()
    for _id, tx, du in zip(t["id"], t["text"], t["duration"]):
        if _id in keep and tx:
            ids.append(_id); langs.append(cfg); texts.append(tx); durs.append(float(du))
    if i % 100 == 0:
        print(f"  [{i}/{len(shards)}] {len(ids)} rows ({time.time()-t0:.0f}s)", flush=True)

# same contiguous holdout as everywhere else: last 15% of each language by id
order = sorted(range(len(ids)), key=lambda k: (langs[k], ids[k]))
split = [None] * len(ids)
per = {}
for k in order:
    per.setdefault(langs[k], []).append(k)
for lang, ks in per.items():
    cut = int(len(ks) * 0.85)
    for k in ks[:cut]:
        split[k] = "train"
    for k in ks[cut:]:
        split[k] = "validation"

pq.write_table(pa.table({
    "id": pa.array(ids, pa.string()),
    "language": pa.array(langs, pa.string()),
    "text": pa.array(texts, pa.string()),
    "duration": pa.array(durs, pa.float64()),
    "split": pa.array(split, pa.string()),
}), OUT, compression="zstd")
print(f"\nwrote {OUT}: {len(ids)} rows, {len(set(langs))} languages, "
      f"{time.time()-t0:.0f}s")
