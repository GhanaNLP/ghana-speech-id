#!/usr/bin/env bash
# Out of domain with and without voting, same chunked head, so the effect of voting is
# visible on its own.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
M=out/chunk40_vote/model.joblib
[ -s "$M" ] || { echo "no chunked model yet"; exit 1; }
echo "########## chunked head, VOTED (40/20) ##########"
.venv/bin/python -u scripts/ood_eval.py --model "$M" --decoded data/base_300m_eval.parquet \
    --text-col text --chunk-chars 40 --chunk-stride 20 --out out/ood_chunk40_vote.json \
    2>&1 | grep -E "^  (finance|jw|lds|unicef|waxal) |OVERALL|in-set kept|^ +0\.|median margin"
echo; echo "########## same head, NO voting (whole transcript) ##########"
.venv/bin/python -u scripts/ood_eval.py --model "$M" --decoded data/base_300m_eval.parquet \
    --text-col text --out out/ood_chunk40_novote.json \
    2>&1 | grep -E "^  (finance|jw|lds|unicef|waxal) |OVERALL"
echo "OOD CHUNK DONE"
