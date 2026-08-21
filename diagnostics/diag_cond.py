"""Separate two failures that the 36.3% end-to-end number conflates.

The recogniser emits ~9.4 units/s on the training corpus and ~3.7/s on this eval audio, with
many clips producing nothing. When the head is handed an empty string it cannot be right, so
the aggregate measures the ASR more than the classifier. Conditioning on phoneme rate says
how the head does when it was actually given something to work with.
"""
import joblib
import numpy as np
import pyarrow.parquet as pq

MODEL = "out/svm_ng5_mf200000_contiguous_nopunct_twimerged"
ISO_TO_LABEL = {
    "twi": "Twi_twi", "twi_asante": "Twi_twi", "twi_akuapem": "Twi_twi",
    "fat": "Fante_fat", "dga": "Dagaare_dga", "dag": "Dagbani_dag",
    "ada": "Dangme_ada", "ewe": "Ewe_ewe", "gur": "Ninkare_gur",
    "nzi": "Nzema_nzi", "sfw": "Sehwi_sfw",
}
PUNCT = set(".,!?;:\"'()-—…")

b = joblib.load(f"{MODEL}/model.joblib")
vec, clf = b["vec"], b["clf"]

t = pq.read_table("data/eval_ipa_gh.parquet").to_pydict()
rows = []
for ipa, iso, dur, grp in zip(t["ipa"], t["iso"], t["duration"], t["group"]):
    gold = ISO_TO_LABEL.get(iso)
    if gold is None:
        continue
    s = " ".join(u for u in (ipa or "").split() if u not in PUNCT)
    n = len(s.split())
    rows.append((s, gold, n, n / max(dur, 0.01), grp.split("_")[0]))

print(f"{len(rows)} in-set eval clips\n")
print("phoneme rate is the ASR's health; training corpus runs ~9.4 units/s\n")
print(f"{'units/s band':16} {'n':>6} {'share':>7} {'head accuracy':>14}")
bands = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 100)]
for lo, hi in bands:
    sub = [r for r in rows if lo <= r[3] < hi and r[2] >= 3]
    if not sub:
        print(f"{f'{lo}-{hi}':16} {0:6d}")
        continue
    pred = clf.predict(vec.transform([r[0] for r in sub]))
    acc = np.mean([p == r[1] for p, r in zip(pred, sub)])
    print(f"{f'{lo}-{hi}':16} {len(sub):6d} {len(sub)/len(rows):6.1%} {acc:13.3f}")

dead = sum(1 for r in rows if r[2] < 3)
print(f"\nclips the ASR emitted <3 units for: {dead} ({dead/len(rows):.1%}) "
      f"-- the head cannot classify these at all")

healthy = [r for r in rows if r[3] >= 6 and r[2] >= 3]
if healthy:
    pred = clf.predict(vec.transform([r[0] for r in healthy]))
    acc = np.mean([p == r[1] for p, r in zip(pred, healthy)])
    print(f"\nhead accuracy where the ASR looked healthy (>=6 units/s): "
          f"{acc:.3f} on {len(healthy)} clips ({len(healthy)/len(rows):.1%} of the set)")

print("\nphoneme rate by domain -- which domains the ASR can read at all:")
for dom in sorted({r[4] for r in rows}):
    sub = [r for r in rows if r[4] == dom]
    print(f"  {dom:10} n={len(sub):5d}  median {np.median([r[3] for r in sub]):5.2f} units/s"
          f"   {sum(1 for r in sub if r[2] < 3)/len(sub):6.1%} produced nothing")
