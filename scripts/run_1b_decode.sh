#!/usr/bin/env bash
# Full 1B decode via fairseq2. sherpa publishes the 1B only as int8, which has no CUDA
# kernels and runs at 4-5x -- about 200 hours for this corpus. fairseq2 does ~260x.
# sherpa int8 stays the on-device runtime; this is bulk decoding only.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
export HF_HOME=/mnt/volume_d2wey28/hf-cache
PY=.venv-gpu/bin/python

echo "########## 1/2 training corpus ##########"
$PY -u scripts/decode_fairseq2.py --card omniASR_CTC_1B \
    --audio-root /mnt/volume_d2wey28/data/ghana-speech \
    --keep-ids data/ipa_text.parquet --batch 32 \
    --parts-dir data/parts_1b_train \
    --out data/base_1b_train.parquet

EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)
echo; echo "########## 2/2 evaluation set ##########"
$PY -u scripts/decode_fairseq2.py --card omniASR_CTC_1B \
    --audio-root "${EV%/}" --keep-ids "" --exclude-prefix bible_ --batch 32 \
    --parts-dir data/parts_1b_eval \
    --out data/base_1b_eval.parquet

echo "1B DECODE DONE"

echo; echo "########## build corpus ##########"
.venv/bin/python -u scripts/build_base_corpus.py --gh data/base_1b_train.parquet \
    --out data/base_1b_corpus.parquet

for mf in 200000 50000; do
  tag="base1b_char_mf${mf}_noen"
  echo; echo "########## train $tag ##########"
  .venv/bin/python -u scripts/train_head.py --data data/base_1b_corpus.parquet \
      --text-col text --analyzer char --model svm --ngram-max 5 --max-features "$mf" \
      --merge-iso --tag "$tag" 2>&1 | tee "logs/${tag}.log" | \
      grep -E "validation accuracy|family accuracy|first |of [0-9]+ errors"
done

echo; echo "########## OUT OF DOMAIN, 1B ##########"
.venv/bin/python -u scripts/ood_eval.py --model out/base1b_char_mf200000_noen/model.joblib \
    --decoded data/base_1b_eval.parquet --text-col text --out out/ood_base1b.json
echo "1B PIPELINE DONE"
