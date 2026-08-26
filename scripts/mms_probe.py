"""Fast read: do MMS features separate these languages at 1.6 s at all?

Splits the eval set and trains on half. Optimistic by construction -- both halves come from
the same recordings, so this is an upper bound, not a result. Its value is negative: if the
features cannot separate languages even under these generous conditions, the four-hour
training extraction has nothing to find.

Run on both feature sets and both durations, so the comparison is like for like.
"""
from __future__ import annotations

import sys

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CONFIG_TO_ISO = {
    "finance_Akuapem_Twi": "twi", "finance_Asante_Twi": "twi", "finance_fante": "fat",
    "jw_dagaare_dga": "dga", "jw_dangme_ada": "ada", "jw_ewe_ewe": "ewe",
    "jw_fante_fat": "fat", "jw_frafra_gur": "gur", "jw_nzema_nzi": "nzi",
    "jw_sehwi_sfw": "sfw", "lds_Asante_Twi": "twi", "lds_Fante_fat": "fat",
    "unicef_Asante_Twi": "twi", "unicef_dagbani": "dag", "unicef_ewe": "ewe",
    "waxal_Asante_Twi": "twi", "waxal_Dagaare_dga": "dga", "waxal_Dagbani_dag": "dag",
    "waxal_Ewe_ewe": "ewe",
}

for path, label in ((sys.argv[1], "full"), (sys.argv[2], "1.6s")):
    t = pq.read_table(path, columns=["language", "emb", "logits"]).to_pydict()
    keep = [i for i, c in enumerate(t["language"]) if c in CONFIG_TO_ISO]
    y = np.array([CONFIG_TO_ISO[t["language"][i]] for i in keep])
    # hold out whole configs, so the split is by recording project rather than by clip
    cfgs = np.array([t["language"][i] for i in keep])
    te = np.array([c.startswith(("waxal_", "unicef_")) for c in cfgs])
    for col in ("emb", "logits"):
        X = np.asarray([t[col][i] for i in keep], dtype=np.float32)
        sc = StandardScaler().fit(X[~te])
        clf = LogisticRegression(max_iter=400, n_jobs=-1).fit(sc.transform(X[~te]), y[~te])
        acc = (clf.predict(sc.transform(X[te])) == y[te]).mean()
        print(f"  {label:5} {col:7} {X.shape[1]:5}-dim  held-out-project accuracy {acc:.4f} "
              f"on {te.sum()} clips over {len(set(y))} languages")
print("\nUpper bound only -- trained and tested within the eval set. The bar for the real")
print("experiment is 0.5700 at 1.6 s.")
