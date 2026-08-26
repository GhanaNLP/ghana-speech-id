#!/usr/bin/env bash
# PhoneticXeus end to end. GPU, where it ran at 290x; on CPU it manages 3.4x fp32 / 7.6x
# int8, so a CPU decode would take days.
#
# ZIPA, the other universal phone recogniser, came in 23 points below the omniASR baseline
# on out-of-domain accuracy at 20 characters. XEUS differs in two ways that might matter: a
# richer inventory that keeps k͡p, t͡ʃ, β, ɸ and nasalised vowels as single units where ZIPA
# decomposes them, and a much larger encoder trained on 1M hours across 4000+ languages.
set -uo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id
export HF_HOME=/mnt/volume_d2wey28/hf-cache
PY=.venv-phone/bin/python
EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)

echo "########## xeus: eval set first ##########"
# Eval is 27x smaller than train, so decoding it first means a usable signal in minutes
# rather than hours if something is wrong.
$PY -u scripts/decode_phone.py --model xeus --audio-root "${EV%/}" --keep-ids "" \
    --exclude-prefix bible_ --parts-dir data/parts_xeus_eval \
    --out data/xeus_eval.parquet

echo "########## xeus: training corpus ##########"
$PY -u scripts/decode_phone.py --model xeus \
    --audio-root /mnt/volume_d2wey28/data/ghana-speech \
    --keep-ids data/ipa_text.parquet --parts-dir data/parts_xeus_train \
    --out data/xeus_train.parquet

.venv/bin/python -u scripts/build_base_corpus.py --gh data/xeus_train.parquet \
    --out data/xeus_corpus.parquet 2>&1 | grep -E "classes|wrote"

for mf in 50000 200000; do
  tag="xeus_char_mf${mf}"
  echo; echo "########## $tag ##########"
  .venv/bin/python -u scripts/train_head.py --data data/xeus_corpus.parquet \
      --text-col text --analyzer char --model svm --ngram-max 5 --max-features "$mf" \
      --chunk-chars 40 --chunk-stride 20 --merge-iso --tag "$tag" 2>&1 \
    | tee "logs/${tag}.log" | grep -E "chunked train|validation accuracy|family accuracy"
  .venv/bin/python -u scripts/ood_eval.py --model "out/${tag}/model.joblib" \
      --decoded data/xeus_eval.parquet --text-col text --out "out/ood_${tag}.json" \
      2>&1 | grep -E "first [0-9]+ chars|whole transcript|OVERALL"
done
echo "XEUS DONE"
