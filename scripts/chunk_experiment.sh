#!/usr/bin/env bash
# Does training on short windows and voting at inference beat training on whole transcripts?
#
# Three conditions, so the two effects can be told apart:
#   whole/whole    the current head, 95.41%
#   whole/vote     voting alone, without retraining
#   chunk/vote     both -- what the app would actually do
cd /mnt/volume_d2wey28/projects/ghana-speech-id
D=data/base_300m_noen.parquet
COMMON="--data $D --text-col text --analyzer char --model svm --ngram-max 5 --merge-iso"

echo "########## chunk 40 / stride 20, voted ##########"
.venv/bin/python -u scripts/train_head.py $COMMON --max-features 200000 \
    --chunk-chars 40 --chunk-stride 20 --vote --tag chunk40_vote 2>&1 \
    | tee logs/chunk40_vote.log | grep -E "chunked train|vectorised|fit in|validation accuracy|family accuracy|first |of [0-9]+ errors"

echo; echo "########## chunk 40 / stride 20, no voting (single window at a time) ##########"
.venv/bin/python -u scripts/train_head.py $COMMON --max-features 200000 \
    --chunk-chars 40 --chunk-stride 20 --tag chunk40_novote 2>&1 \
    | tee logs/chunk40_novote.log | grep -E "chunked train|validation accuracy|first "

echo "CHUNK EXPERIMENT DONE"
