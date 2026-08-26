#!/usr/bin/env bash
# Audio chunking for omniASR: 3 s windows, 1.5 s stride, decoded with the served int8.
#
# Two deferred items in one pass. Transcript chunking helped short audio (+4.7 points at one
# second) but trains on windows of a well-transcribed clip; short audio genuinely
# transcribes worse -- 8.6% of 0-5 s clips come back empty against 0.0% of 5-10 s clips.
# Cutting the audio first reproduces that. And decoding with int8 rather than fp32 closes
# the train/serve gap that costs the current head 1.3 points.
#
# CPU on purpose: int8 runs at 17x there against 7x on CUDA, and it leaves the GPU for Qwen.
set -uo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id
W=${1:-5}

echo "=== training corpus, 3s/1.5s windows, $W workers ==="
for i in $(seq 0 $((W-1))); do
  .venv/bin/python -u scripts/decode_chunked.py --model 300m-int8 --provider cpu \
      --threads 4 --batch 16 --window 3.0 --stride 1.5 \
      --audio-root /mnt/volume_d2wey28/data/ghana-speech \
      --keep-ids data/ipa_text.parquet --parts-dir data/parts_chunk3_train \
      --shard-index "$i" --shard-count "$W" --out "/tmp/chunk3_$i.parquet" \
      > "logs/chunk3_w$i.log" 2>&1 &
done
wait
echo "=== assemble ==="
.venv/bin/python - <<'PY'
import glob
import pyarrow as pa, pyarrow.parquet as pq
fs = sorted(glob.glob("data/parts_chunk3_train/*.parquet"))
t = pa.concat_tables([pq.read_table(f) for f in fs], promote_options="default")
pq.write_table(t, "data/chunk3_train.parquet", compression="zstd")
ch = [len(s.replace(" ", "")) for s in t["text"].to_pylist()]
empty = sum(1 for c in ch if c < 3)
print(f"  {t.num_rows} windows from {len(fs)} shards, mean {sum(ch)/max(len(ch),1):.1f} "
      f"chars, {empty} ({empty/max(len(ch),1):.1%}) empty")
PY

.venv/bin/python -u scripts/build_base_corpus.py --gh data/chunk3_train.parquet \
    --out data/chunk3_corpus.parquet 2>&1 | grep -E "classes|wrote"

# The eval set stays whole-clip: that is what the app receives. Only training changes.
for mf in 50000 200000; do
  tag="chunk3_mf${mf}"
  echo; echo "=== $tag ==="
  .venv/bin/python -u scripts/train_head.py --data data/chunk3_corpus.parquet \
      --text-col text --analyzer char --model svm --ngram-max 5 --max-features "$mf" \
      --merge-iso --tag "$tag" 2>&1 | tee "logs/${tag}.log" \
    | grep -E "loaded|validation accuracy|family accuracy"
  .venv/bin/python -u scripts/ood_eval.py --model "out/${tag}/model.joblib" \
      --decoded data/base_300m_eval.parquet --text-col text --out "out/ood_${tag}.json" \
      2>&1 | grep -E "first [0-9]+ chars|whole transcript|OVERALL"
done
echo "CHUNK3 DONE"
