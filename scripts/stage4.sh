#!/usr/bin/env bash
# Fires when the base decode lands: build the corpus, train char-ngram heads, score
# out of domain. The out-of-domain number against the IPA head's 36.3% is the point.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
while pgrep -f decode_base.py > /dev/null; do sleep 60; done
echo "=== decode finished ==="
tail -4 logs/base_decode_300m.log
for f in data/base_300m_train.parquet data/base_300m_eval.parquet; do
  [ -s "$f" ] || { echo "MISSING $f"; exit 1; }
done

echo; echo "########## build corpus ##########"
.venv/bin/python -u scripts/build_base_corpus.py --gh data/base_300m_train.parquet \
    --en data/base_300m_english.parquet --out data/base_300m_corpus.parquet

for mf in 200000 50000; do
  tag="base300m_char_mf${mf}"
  echo; echo "########## train $tag ##########"
  .venv/bin/python -u scripts/train_head.py --data data/base_300m_corpus.parquet \
      --text-col text --analyzer char --model svm --ngram-max 5 --max-features "$mf" \
      --merge-iso --tag "$tag" 2>&1 | tee "logs/${tag}.log" | \
      grep -E "loaded|merging|train:|validation:|validation accuracy|family accuracy|first |of [0-9]+ errors"
done

echo; echo "########## OUT OF DOMAIN (the number that matters) ##########"
.venv/bin/python -u scripts/ood_eval.py --model out/base300m_char_mf200000/model.joblib \
    --decoded data/base_300m_eval.parquet --text-col text \
    --out out/ood_base300m.json 2>&1
echo "STAGE4 DONE"
