#!/usr/bin/env bash
# Allosaurus as a third phoneme front end. 45.7 MB against omniASR int8's 350 MB, so if its
# accuracy holds it is the only candidate that is obviously deployable on a phone.
# lang_id stays "ipa" -- the universal inventory. Telling the phone recogniser the language
# would hand it the answer the head is meant to work out.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
export HF_HOME=/mnt/volume_d2wey28/hf-cache
PY=.venv-phone/bin/python
EV=$(ls -d /mnt/volume_d2wey28/hf-cache/hub/datasets--ghananlpcommunity--ghana-speech-eval/snapshots/*/ | head -1)

$PY -u scripts/decode_phone.py --model allo \
    --audio-root /mnt/volume_d2wey28/data/ghana-speech \
    --keep-ids data/ipa_text.parquet --parts-dir data/parts_allo_train \
    --out data/phone_allo_train.parquet
$PY -u scripts/decode_phone.py --model allo --audio-root "${EV%/}" --keep-ids "" \
    --exclude-prefix bible_ --parts-dir data/parts_allo_eval \
    --out data/phone_allo_eval.parquet

.venv/bin/python -u scripts/build_base_corpus.py --gh data/phone_allo_train.parquet \
    --out data/phone_allo_corpus.parquet

for A in char units; do
  tag="phone_allo_${A}"
  echo "--- $tag ---"
  if [ "$A" = char ]; then EXTRA="--analyzer char --join-units --chunk-chars 40 --chunk-stride 20"
  else EXTRA="--analyzer word"; fi
  .venv/bin/python -u scripts/train_head.py --data data/phone_allo_corpus.parquet \
      --text-col text --model svm --ngram-max 5 --max-features 200000 --merge-iso \
      $EXTRA --tag "$tag" 2>&1 | tee "logs/${tag}.log" \
    | grep -E "chunked train|validation accuracy|family accuracy|^  first "
  .venv/bin/python -u scripts/ood_eval.py --model "out/${tag}/model.joblib" \
      --decoded data/phone_allo_eval.parquet --text-col text --out "out/ood_${tag}.json" \
      2>&1 | grep -E "^  (finance|jw|lds|unicef|waxal) |OVERALL"
done
echo "ALLOSAURUS DONE"
