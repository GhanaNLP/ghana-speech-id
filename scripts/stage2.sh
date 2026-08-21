#!/usr/bin/env bash
# Fires when the English GPU decode lands. Two tracks that do not contend: the English
# conditions are CPU-bound training, the eval decode is GPU-bound.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
while pgrep -f decode_gpu.py > /dev/null; do sleep 20; done
echo "=== english decode finished ==="
tail -6 logs/english_gpu.log
[ -s data/english_ipa_gh.parquet ] || { echo "NO ENGLISH IPA -- STOPPING"; exit 1; }

# GPU track: the evaluation corpus. No pipe, so progress actually streams this time.
setsid nohup .venv-gpu/bin/python -u scripts/decode_gpu.py --hf-eval \
    --out data/eval_ipa_gh.parquet > logs/eval_decode.log 2>&1 < /dev/null &
echo "eval decode launched on GPU"

# CPU track: how much English is the right amount. Recall answers "does English get picked
# up"; precision answers "does it start stealing Ghanaian speech", which is the costlier error.
for n in 8540 24384 50000; do
  echo "=================== english n=$n ==================="
  .venv/bin/python -u scripts/add_english.py --src data/english_ipa_gh.parquet \
      --no-crop --n "$n" --out "data/ipa_text_en${n}.parquet" 2>&1 | tail -8
  tag="svm_ng5_mf200000_contiguous_nopunct_twimerged_en${n}"
  .venv/bin/python -u scripts/train_head.py --data "data/ipa_text_en${n}.parquet" \
      --model svm --ngram-max 5 --max-features 200000 --drop-punct --merge-iso \
      --tag "$tag" > "logs/${tag}.log" 2>&1
  .venv/bin/python - "$tag" <<'PY'
import json, sys
d = json.load(open(f"out/{sys.argv[1]}/metrics.json"))
e = d["per_language"].get("English_eng", {})
print(f"  overall {d['accuracy']:.4f}  macroF1 {d['macro_f1']:.4f}")
print(f"  English recall {e.get('recall',0):.4f}  precision {e.get('precision',0):.4f}  "
      f"n={int(e.get('support',0))}")
stolen = sum(c for a, b, c in d["top_confusions"] if b == "English_eng")
print(f"  Ghanaian clips called English (top-50 confusions): {stolen}")
PY
done
echo "ENGLISH CONDITIONS DONE"
