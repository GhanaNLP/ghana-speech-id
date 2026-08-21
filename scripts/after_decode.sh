#!/usr/bin/env bash
# Fires when both the sweep and the English decode are done: build the 8.5k English corpus
# and train the two heads that include it. logreg matters most -- confidence thresholding
# for out-of-set speech needs calibrated probabilities, which LinearSVC does not give.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
while pgrep -f "rephonemise_english|run_sweep.sh|train_head.py" > /dev/null; do sleep 30; done
echo "=== upstream jobs finished ==="
[ -s data/english_ipa_gh.parquet ] || { echo "NO ENGLISH IPA PRODUCED"; exit 1; }

echo "=== build 42-class corpus (8.5k English) ==="
.venv/bin/python -u scripts/add_english.py --src data/english_ipa_gh.parquet \
    --no-crop --out data/ipa_text_en.parquet 2>&1 | tail -12

for m in svm logreg; do
  tag="${m}_ng5_mf200000_contiguous_nopunct_twimerged_en8k"
  echo "=================== $tag ==================="
  .venv/bin/python -u scripts/train_head.py --data data/ipa_text_en.parquet \
      --model "$m" --ngram-max 5 --max-features 200000 --drop-punct --merge-iso \
      --tag "$tag" 2>&1 | tee "logs/${tag}.log" | tail -30
done
echo "ENGLISH 8K DONE"
