#!/usr/bin/env bash
# Score the out-of-domain set as soon as the GPU decode lands. Run against the no-English
# head first: ghana-speech-eval has no English config, so the English class cannot be
# credited or penalised here, and the 41-class head is the clean comparison.
cd /mnt/volume_d2wey28/projects/ghana-speech-id
while pgrep -f decode_gpu.py > /dev/null; do sleep 15; done
[ -s data/eval_ipa_gh.parquet ] || { echo "NO DECODED EVAL DATA"; exit 1; }
echo "=== decode finished ==="
grep -vE "^Fetching|it/s\]$" logs/eval_decode.log | tail -4

M=out/svm_ng5_mf200000_contiguous_nopunct_twimerged
echo
echo "########## OUT-OF-DOMAIN: $(basename $M) ##########"
.venv/bin/python -u scripts/ood_eval.py --model "$M/model.joblib" \
    --out out/ood_eval.json 2>&1
echo "OOD DONE"
