#!/usr/bin/env bash
# What does the deployed head actually score at each duration of real audio?
#
# Truncated-transcript numbers are optimistic: the recogniser still had the whole clip and
# only its output was cut. Measured on real audio, three seconds gives 0.506 where the
# 20-character truncation suggested 0.570. Since the product is about to enforce a minimum
# duration, that threshold should be chosen from real measurements.
set -euo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id
EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)

for W in 5 7 10; do
  echo "########## ${W}s windows ##########"
  for i in 0 1 2 3; do
    PYTHONPATH=scripts .venv/bin/python -u scripts/decode_chunked.py --model 300m-int8 \
        --provider cpu --threads 4 --batch 16 --window "$W" --stride "$W" \
        --min-seconds "$W" --audio-root "${EV%/}" --keep-ids "" --exclude-prefix bible_ \
        --parts-dir "data/parts_eval${W}s" --shard-index "$i" --shard-count 4 \
        --out "/tmp/eval${W}_$i.parquet" > "logs/eval${W}s_w$i.log" 2>&1 &
  done
  wait
  .venv/bin/python - "$W" <<'PY'
import glob, sys
import pyarrow as pa, pyarrow.parquet as pq
w = sys.argv[1]
fs = sorted(glob.glob(f"data/parts_eval{w}s/*.parquet"))
t = pa.concat_tables([pq.read_table(f) for f in fs], promote_options="default")
pq.write_table(t, f"data/eval{w}s_text.parquet", compression="zstd")
ch = [len(s.replace(" ", "")) for s in t["text"].to_pylist()]
print(f"  {t.num_rows} windows, mean {sum(ch)/max(len(ch),1):.1f} chars, "
      f"{sum(1 for c in ch if c<3)/max(len(ch),1):.1%} empty")
PY
  .venv/bin/python -u scripts/ood_eval.py --model out/final_300m_mf50000/model.joblib \
      --decoded "data/eval${W}s_text.parquet" --text-col text --truncate 0 \
      --out "out/ood_real_${W}s.json" 2>&1 | grep -E "OVERALL|whole transcript"
done
echo "DURATION CURVE DONE"
