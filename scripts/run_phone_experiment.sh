#!/usr/bin/env bash
# Do phoneme-specialist front ends beat omniASR orthography (95.4% in-domain, 77.7% out)?
#
# XEUS emits continuous IPA, so char n-grams are the only option. wav2vec2 emits
# space-separated phones, so both are testable: char n-grams over the joined string, and
# unit n-grams over the phones themselves, which is IPA's natural representation.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
export HF_HOME=/mnt/volume_d2wey28/hf-cache
PY=.venv-phone/bin/python
EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)

for M in w2v xeus; do
  echo "################ $M ################"
  $PY -u scripts/decode_phone.py --model "$M" \
      --audio-root /mnt/volume_d2wey28/data/ghana-speech \
      --keep-ids data/ipa_text.parquet --parts-dir "data/parts_${M}_train" \
      --out "data/phone_${M}_train.parquet"
  $PY -u scripts/decode_phone.py --model "$M" --audio-root "${EV%/}" --keep-ids "" \
      --exclude-prefix bible_ --parts-dir "data/parts_${M}_eval" \
      --out "data/phone_${M}_eval.parquet"

  .venv/bin/python -u scripts/build_base_corpus.py --gh "data/phone_${M}_train.parquet" \
      --out "data/phone_${M}_corpus.parquet"

  # char n-grams on the joined phone string, the like-for-like comparison
  tag="phone_${M}_char"
  echo "--- $tag ---"
  .venv/bin/python -u scripts/train_head.py --data "data/phone_${M}_corpus.parquet" \
      --text-col text --analyzer char --join-units --model svm --ngram-max 5 \
      --max-features 200000 --chunk-chars 40 --chunk-stride 20 --merge-iso --tag "$tag" \
      2>&1 | tee "logs/${tag}.log" | grep -E "chunked train|validation accuracy|family accuracy|^  first "
  .venv/bin/python -u scripts/ood_eval.py --model "out/${tag}/model.joblib" \
      --decoded "data/phone_${M}_eval.parquet" --text-col text --out "out/ood_${tag}.json" \
      2>&1 | grep -E "^  (finance|jw|lds|unicef|waxal) |OVERALL"
done

# unit n-grams, only meaningful where phones are actually delimited
tag="phone_w2v_units"
echo "--- $tag ---"
.venv/bin/python -u scripts/train_head.py --data data/phone_w2v_corpus.parquet \
    --text-col text --analyzer word --model svm --ngram-max 5 --max-features 200000 \
    --merge-iso --tag "$tag" 2>&1 | tee "logs/${tag}.log" \
  | grep -E "validation accuracy|family accuracy|^  first "
.venv/bin/python -u scripts/ood_eval.py --model "out/${tag}/model.joblib" \
    --decoded data/phone_w2v_eval.parquet --text-col text --out "out/ood_${tag}.json" \
    2>&1 | grep -E "^  (finance|jw|lds|unicef|waxal) |OVERALL"

echo "PHONE EXPERIMENT DONE"
