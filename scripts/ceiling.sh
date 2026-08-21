#!/usr/bin/env bash
# Ceiling for the orthographic approach: char n-grams over ground-truth text.
# Base omniASR emits orthography, so its head cannot beat this. If perfect text does not
# clear the IPA head's 94.66%, changing front-ends is not the fix.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
for mf in 200000 50000; do
  tag="ceiling_char_mf${mf}_truth"
  echo "=================== $tag ==================="
  .venv/bin/python -u scripts/train_head.py --data data/text_truth.parquet \
      --text-col text --analyzer char --model svm --ngram-max 5 --max-features "$mf" \
      --merge-iso --tag "$tag" 2>&1 | grep -E "loaded|merging|train:|validation:|vectorised|fit in|validation accuracy|family accuracy|first |of .* errors"
done
echo "CEILING DONE"
