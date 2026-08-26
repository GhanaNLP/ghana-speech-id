#!/usr/bin/env bash
# Post-decode stages for one ZIPA variant: assemble shards, build corpus, train, score.
#
# Separate from run_zipa.sh on purpose. Editing that file while bash was executing it is
# what broke the last run: bash reads a script incrementally from a file offset, so an edit
# shifts the content under the running interpreter and it resumes mid-token. Never edit a
# script that is running; write a new one.
set -uo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id
M=${1:?variant}

for part in train eval; do
  [ -d "data/parts_${M}_${part}" ] || { echo "no parts for $M $part"; exit 1; }
done

echo "########## $M: assemble ##########"
.venv/bin/python - "$M" <<'PY'
import glob, sys
import pyarrow as pa, pyarrow.parquet as pq
m = sys.argv[1]
for name in ("train", "eval"):
    fs = sorted(glob.glob(f"data/parts_{m}_{name}/*.parquet"))
    t = pa.concat_tables([pq.read_table(f) for f in fs], promote_options="default")
    pq.write_table(t, f"data/{m}_{name}.parquet", compression="zstd")
    ch = [len(s.replace(" ", "")) for s in t["text"].to_pylist()]
    empty = sum(1 for c in ch if c < 3)
    print(f"  {name}: {t.num_rows} clips from {len(fs)} shards, "
          f"mean {sum(ch)/max(len(ch),1):.1f} chars, {empty} ({empty/max(len(ch),1):.1%}) under 3")
PY

.venv/bin/python -u scripts/build_base_corpus.py --gh "data/${M}_train.parquet" \
    --out "data/${M}_corpus.parquet" 2>&1 | grep -E "classes|wrote"

for mf in 50000 200000; do
  tag="${M}_char_mf${mf}"
  echo; echo "########## $tag ##########"
  .venv/bin/python -u scripts/train_head.py --data "data/${M}_corpus.parquet" \
      --text-col text --analyzer char --model svm --ngram-max 5 --max-features "$mf" \
      --chunk-chars 40 --chunk-stride 20 --merge-iso --tag "$tag" 2>&1 \
    | tee "logs/${tag}.log" | grep -E "chunked train|validation accuracy|family accuracy"
  .venv/bin/python -u scripts/ood_eval.py --model "out/${tag}/model.joblib" \
      --decoded "data/${M}_eval.parquet" --text-col text --out "out/ood_${tag}.json" \
      2>&1 | grep -E "first [0-9]+ chars|whole transcript|OVERALL"
done
echo "FINISH $M DONE"
