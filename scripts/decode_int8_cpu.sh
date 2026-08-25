#!/usr/bin/env bash
# Decode the corpus with the SAME quantisation that gets served.
#
# The head was trained on fp32 transcripts and is served int8, which costs about 1.3 points
# and flips 9% of predictions. int8 runs at 7x on CUDA against fp32's 111x -- quantised ops
# have no CUDA kernels -- so this is a CPU job, parallelised across shards. The GPU stays
# free for the phoneme experiment.
set -uo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id
W=${1:-5}                       # workers; 4 torch threads each on a 20-core box
EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)

echo "=== training corpus, $W workers ==="
for i in $(seq 0 $((W-1))); do
  .venv/bin/python -u scripts/decode_base.py --model 300m-int8 --provider cpu --threads 4 \
      --batch 16 --audio-root /mnt/volume_d2wey28/data/ghana-speech \
      --keep-ids data/ipa_text.parquet --parts-dir data/parts_int8_train \
      --shard-index "$i" --shard-count "$W" \
      --out "/tmp/int8_train_$i.parquet" > "logs/int8_w$i.log" 2>&1 &
done
wait
echo "=== training corpus done ==="

echo "=== eval set ==="
for i in $(seq 0 $((W-1))); do
  .venv/bin/python -u scripts/decode_base.py --model 300m-int8 --provider cpu --threads 4 \
      --batch 16 --audio-root "${EV%/}" --keep-ids "" --exclude-prefix bible_ \
      --parts-dir data/parts_int8_eval --shard-index "$i" --shard-count "$W" \
      --out "/tmp/int8_eval_$i.parquet" > "logs/int8_ev$i.log" 2>&1 &
done
wait

echo "=== assemble ==="
.venv/bin/python - <<'PY'
import glob
import pyarrow as pa, pyarrow.parquet as pq
for name, d in (("train", "data/parts_int8_train"), ("eval", "data/parts_int8_eval")):
    fs = sorted(glob.glob(f"{d}/*.parquet"))
    if not fs:
        print(f"{name}: no parts"); continue
    t = pa.concat_tables([pq.read_table(f) for f in fs], promote_options="default")
    out = f"data/int8_{name}.parquet"
    pq.write_table(t, out, compression="zstd")
    ch = [len(s.replace(" ", "")) for s in t["text"].to_pylist()]
    print(f"{name}: {t.num_rows} clips from {len(fs)} shards -> {out}, "
          f"mean {sum(ch)/max(len(ch),1):.1f} chars")
PY
echo "INT8 DECODE DONE"
