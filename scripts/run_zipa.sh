#!/usr/bin/env bash
# ZIPA end to end: decode, train, evaluate. Both sizes.
#
# Decoded with int8 -- the artefact that actually gets served. The omniASR head was trained
# on fp32 and served int8, which cost about 1.3 points; decoding with the served
# quantisation from the start avoids repeating that.
#
# sherpa-onnx loads these natively via from_zipformer_ctc: ZIPA is Zipformer CTC from the
# same Icefall lineage. No export, no new runtime.
set -uo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id
W=${2:-5}
M=${1:-zipa-small}
EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)

# The large model is 4.6x bigger and 2.7x slower than the small one, and ZIPA's own paper
# reports the 64M model beating 300M+ baselines. Running it before knowing whether small
# suffices is twelve hours on a likely dead end. Remove SKIP_LARGE to enable it.
# fp16 is lossless against fp32 and int8 is not, so large-int8 is the variant worth
# skipping; large-fp16 is a legitimate accuracy candidate.
case "$M" in
  zipa-large) [ -f SKIP_LARGE ] && { echo "skipping $M (SKIP_LARGE present)"; exit 0; } ;;
esac

echo "########## $M: training corpus, $W workers ##########"
for i in $(seq 0 $((W-1))); do
  .venv/bin/python -u scripts/decode_base.py --model "$M" --provider cpu --threads 4 \
      --batch 16 --audio-root /mnt/volume_d2wey28/data/ghana-speech \
      --keep-ids data/ipa_text.parquet --parts-dir "data/parts_${M}_train" \
      --shard-index "$i" --shard-count "$W" --out "/tmp/${M}_tr_$i.parquet" \
      > "logs/${M}_w$i.log" 2>&1 &
done
wait
echo "########## $M: eval set ##########"
for i in $(seq 0 $((W-1))); do
  .venv/bin/python -u scripts/decode_base.py --model "$M" --provider cpu --threads 4 \
      --batch 16 --audio-root "${EV%/}" --keep-ids "" --exclude-prefix bible_ \
      --parts-dir "data/parts_${M}_eval" --shard-index "$i" --shard-count "$W" \
      --out "/tmp/${M}_ev_$i.parquet" > "logs/${M}_ev$i.log" 2>&1 &
done
wait

.venv/bin/python - "$M" <<'PY'
import glob, sys
import pyarrow as pa, pyarrow.parquet as pq
m = sys.argv[1]
for name in ("train", "eval"):
    fs = sorted(glob.glob(f"data/parts_{m}_{name}/*.parquet"))
    if not fs:
        print(f"{name}: no parts"); continue
    t = pa.concat_tables([pq.read_table(f) for f in fs], promote_options="default")
    pq.write_table(t, f"data/{m}_{name}.parquet", compression="zstd")
    ch = [len(s.replace(" ", "")) for s in t["text"].to_pylist()]
    empty = sum(1 for c in ch if c < 3)
    print(f"{name}: {t.num_rows} clips, mean {sum(ch)/max(len(ch),1):.1f} chars, "
          f"{empty} ({empty/max(len(ch),1):.1%}) under 3")
PY

.venv/bin/python -u scripts/build_base_corpus.py --gh "data/${M}_train.parquet" \
    --out "data/${M}_corpus.parquet"

for mf in 200000 50000; do
  tag="${M}_char_mf${mf}"
  echo; echo "########## $tag ##########"
  .venv/bin/python -u scripts/train_head.py --data "data/${M}_corpus.parquet" \
      --text-col text --analyzer char --model svm --ngram-max 5 --max-features "$mf" \
      --chunk-chars 40 --chunk-stride 20 --merge-iso --tag "$tag" 2>&1 \
    | tee "logs/${tag}.log" | grep -E "chunked train|validation accuracy|family accuracy|^  first "
  .venv/bin/python -u scripts/ood_eval.py --model "out/${tag}/model.joblib" \
      --decoded "data/${M}_eval.parquet" --text-col text --out "out/ood_${tag}.json" \
      2>&1 | grep -E "^  (finance|jw|lds|unicef|waxal) |OVERALL|in-set kept|^ +0\.[0-9]"
done
echo "ZIPA $M DONE"
