#!/usr/bin/env bash
# Final heads for both front-ends, in the configuration the measurements settled on:
#   char_wb 1-5 grams, trained on 40-char windows with stride 20, whole-transcript inference.
#
# Chunked training is worth +1.1 points out of domain and +4.7 at one second of speech.
# Voting over windows at inference was measured and rejected: -0.6 out of domain, -0.09
# in-domain, and it compresses the margins that out-of-set rejection depends on.
set -uo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id

for v in 300m 1b; do
  CORPUS=data/base_${v}_$([ "$v" = 300m ] && echo noen || echo corpus).parquet
  [ -s "$CORPUS" ] || { echo "missing $CORPUS"; continue; }
  for mf in 200000 50000; do
    tag="final_${v}_mf${mf}"
    echo; echo "########## $tag ##########"
    .venv/bin/python -u scripts/train_head.py --data "$CORPUS" \
        --text-col text --analyzer char --model svm --ngram-max 5 --max-features "$mf" \
        --chunk-chars 40 --chunk-stride 20 --merge-iso --tag "$tag" 2>&1 \
      | tee "logs/${tag}.log" \
      | grep -E "chunked train|validation accuracy|family accuracy|^  first "

    # chunk_chars is deliberately absent from the export: it controls INFERENCE windowing,
    # and inference classifies the whole transcript
    .venv/bin/python -u scripts/export_onnx.py --model "out/${tag}/model.joblib" \
        --outdir "out/${tag}/onnx" --n-check 200 2>&1 \
      | grep -E "exported|features x|casefold|parity over|WARNING"

    echo "--- out of domain ---"
    .venv/bin/python -u scripts/ood_eval.py --model "out/${tag}/model.joblib" \
        --decoded "data/base_${v}_eval.parquet" --text-col text \
        --out "out/ood_${tag}.json" 2>&1 \
      | grep -E "^  (finance|jw|lds|unicef|waxal) |OVERALL|in-set kept|^ +0\.[0-9]"
  done
done
echo; echo "FINAL BUILD DONE"
