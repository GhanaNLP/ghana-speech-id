#!/usr/bin/env bash
# 41 languages, no English class.
#
# The English class was trained on transcripts of low-passed audio: 93.6% of energy below
# 1 kHz, so the recogniser returned short garbled strings for 82% of clips. A class built
# from that may have become a "poor transcription" class rather than an English class,
# which would explain it firing on Ghanaian clips the ASR handled badly. Removing it
# isolates the 41-language system.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
.venv/bin/python -u scripts/build_base_corpus.py --gh data/base_300m_train.parquet \
    --out data/base_300m_noen.parquet

for mf in 200000 50000; do
  tag="base300m_char_mf${mf}_noen"
  echo; echo "########## $tag ##########"
  .venv/bin/python -u scripts/train_head.py --data data/base_300m_noen.parquet \
      --text-col text --analyzer char --model svm --ngram-max 5 --max-features "$mf" \
      --merge-iso --tag "$tag" 2>&1 | tee "logs/${tag}.log" | \
      grep -E "validation accuracy|family accuracy|first |of [0-9]+ errors"
done

echo; echo "########## OUT OF DOMAIN, no English ##########"
.venv/bin/python -u scripts/ood_eval.py --model out/base300m_char_mf200000_noen/model.joblib \
    --decoded data/base_300m_eval.parquet --text-col text --out out/ood_base300m_noen.json
echo "NOEN DONE"
