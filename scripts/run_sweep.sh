#!/usr/bin/env bash
# Accuracy vs deployable size. max_features drives the ONNX weight matrix directly:
# bytes ~= max_features * n_classes * 4 (fp32), so it is the knob that decides whether the
# head fits in a mobile app at all.
set -uo pipefail
cd /mnt/volume_d2wey28/projects/ghana-speech-id
PY=.venv/bin/python
mkdir -p logs out

run () {  # model ngram maxfeat extra...
  local m=$1 ng=$2 mf=$3; shift 3
  local sm=contiguous; [[ "$*" == *"--split-mode random"* ]] && sm=random
  local tag="${m}_ng${ng}_mf${mf}_${sm}$( [[ "$*" == *--drop-punct* ]] && echo _nopunct )_twimerged"
  echo "=================== $tag ==================="
  $PY -u scripts/train_head.py --model "$m" --ngram-max "$ng" --max-features "$mf" \
      --merge-iso --tag "$tag" "$@" 2>&1 | tee "logs/${tag}.log"
}

# punctuation is only ~62% accurate upstream and carries little language signal -- check
# that dropping it helps before sweeping everything else
run svm 5 200000
run svm 5 200000 --drop-punct

# the same config on a random split: the gap against contiguous is how much of the score
# comes from scoring on verses adjacent to the training passages
run svm 5 200000 --drop-punct --split-mode random

# size ladder for mobile
for mf in 50000 100000 400000; do run svm 5 "$mf" --drop-punct; done

# how much n-gram order is worth
for ng in 3 4; do run svm "$ng" 200000 --drop-punct; done

# a calibrated-probability head for the open-set probe, and the classic phonotactic LM
run logreg 5 200000 --drop-punct
run nb 5 200000 --drop-punct

echo; echo "=================== summary ==================="
$PY - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("out/*/metrics.json")):
    d = json.load(open(f))
    lc = d.get("length_curve", {})
    rows.append((d["tag"], d["accuracy"], d["macro_f1"], d["n_features"],
                 d["n_features"] * len(d["per_language"]) * 4 / 1e6,
                 lc.get("20", {}).get("acc"), lc.get("10", {}).get("acc"),
                 100 * d["errors_within_family"] / max(d["errors"], 1)))
rows.sort(key=lambda r: -r[1])
print(f"{'tag':44} {'acc':>7} {'macroF1':>8} {'feats':>8} {'fp32MB':>7} {'acc@20u':>8} {'acc@10u':>8} {'%err in-fam':>11}")
for t, a, m, nf, mb, a20, a10, wf in rows:
    f20 = f"{a20:.4f}" if a20 else "-"
    f10 = f"{a10:.4f}" if a10 else "-"
    print(f"{t:44} {a:7.4f} {m:8.4f} {nf:8d} {mb:7.1f} {f20:>8} {f10:>8} {wf:10.1f}%")
PY
