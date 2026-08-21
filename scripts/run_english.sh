#!/usr/bin/env bash
# Heads that include English as a class. logreg matters most here: confidence thresholding
# for out-of-set speech needs calibrated probabilities, which LinearSVC does not give.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
PY=.venv/bin/python
D=data/ipa_text_en.parquet

for m in svm logreg; do
  tag="${m}_ng5_mf200000_contiguous_nopunct_twimerged_en"
  echo "=================== $tag ==================="
  $PY -u scripts/train_head.py --data "$D" --model "$m" --ngram-max 5 --max-features 200000 \
      --drop-punct --merge-iso --tag "$tag" 2>&1 | tee "logs/${tag}.log"
done
echo "ENGLISH RUNS DONE"
