"""Train a language head on frozen ECAPA embeddings, and check the speaker confound.

The corpus has roughly one narrator per language, so an acoustic model can score near
perfectly in domain by learning voices rather than languages. That is the whole reason the
text pipeline exists. This reports in-domain and out-of-domain side by side and never
in-domain alone, because the signature of the confound is a near-perfect in-domain score
next to a collapse out of domain.

Judged on the same criterion as every text head: accuracy out of domain on about 1.6
seconds of speech, the acoustic equivalent of the 20 characters those heads are scored on.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

CONFIG_TO_ISO = {
    "finance_Akuapem_Twi": "twi", "finance_Asante_Twi": "twi", "finance_fante": "fat",
    "finance_ga": "gaa", "jw_ahanta_aha": "aha", "jw_dagaare_dga": "dga",
    "jw_dangme_ada": "ada", "jw_ewe_ewe": "ewe", "jw_fante_fat": "fat",
    "jw_frafra_gur": "gur", "jw_ga_gaa": "gaa", "jw_nzema_nzi": "nzi",
    "jw_sehwi_sfw": "sfw", "lds_Asante_Twi": "twi", "lds_Fante_fat": "fat",
    "unicef_Asante_Twi": "twi", "unicef_dagbani": "dag", "unicef_ewe": "ewe",
    "waxal_Asante_Twi": "twi", "waxal_Dagaare_dga": "dga", "waxal_Dagbani_dag": "dag",
    "waxal_Ewe_ewe": "ewe", "waxal_Ikposo_kpo": "kpo",
}
ISO_TO_LABEL = {
    "twi": "Twi_twi", "fat": "Fante_fat", "dga": "Dagaare_dga", "dag": "Dagbani_dag",
    "ada": "Dangme_ada", "ewe": "Ewe_ewe", "gur": "Ninkare_gur", "nzi": "Nzema_nzi",
    "sfw": "Sehwi_sfw",
}
OUT_OF_SET = {"gaa", "aha", "kpo"}


def merge_iso(label):
    """Asante and Akuapem Twi share ISO twi and are one class, as in every text head."""
    iso = label.rsplit("_", 1)[-1]
    return "Twi_twi" if iso == "twi" else label


def load(path, col="emb"):
    t = pq.read_table(path, columns=["id", "language", col]).to_pydict()
    X = np.asarray(t[col], dtype=np.float32)
    y = np.array([merge_iso(l) for l in t["language"]])
    return t["id"], X, y, t["language"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval-full", required=True)
    ap.add_argument("--eval-short", default="", help="embeddings from truncated audio")
    ap.add_argument("--feature-col", default="emb",
                    help="emb for pooled hidden states, logits for the 4017 language "
                         "scores as a feature vector")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    ids, X, y, raw = load(args.train, args.feature_col)
    print(f"{len(y)} clips, {X.shape[1]}-dim embeddings, {len(set(y))} classes")

    # contiguous holdout by id, the same book-disjoint split the text heads use
    by = defaultdict(list)
    for i, (_id, lang) in enumerate(zip(ids, y)):
        by[lang].append((_id, i))
    tr, va = [], []
    for lang, rows in by.items():
        rows.sort()
        cut = int(len(rows) * (1 - args.test_frac))
        tr += [i for _, i in rows[:cut]]
        va += [i for _, i in rows[cut:]]
    print(f"  train {len(tr)}  validation {len(va)} (contiguous by id)")

    sc = StandardScaler().fit(X[tr])
    clf = MLPClassifier(hidden_layer_sizes=(args.hidden,), max_iter=60, early_stopping=True,
                        n_iter_no_change=5, random_state=0, verbose=False)
    clf.fit(sc.transform(X[tr]), y[tr])

    pin = clf.predict(sc.transform(X[va]))
    acc_in = accuracy_score(y[va], pin)
    print(f"\nin-domain accuracy {acc_in:.4f}   macro-F1 {f1_score(y[va], pin, average='macro'):.4f}")

    out = {"tag": args.tag, "in_domain": acc_in, "n_train": len(tr), "dim": int(X.shape[1])}

    for name, path in (("full", args.eval_full), ("1.6s", args.eval_short)):
        if not path:
            continue
        _, Xe, _, cfgs = load(path, args.feature_col)
        gold, keep = [], []
        for i, cfg in enumerate(cfgs):
            iso = CONFIG_TO_ISO.get(cfg, "")
            if iso in OUT_OF_SET or iso not in ISO_TO_LABEL:
                continue
            gold.append(ISO_TO_LABEL[iso]); keep.append(i)
        if not keep:
            continue
        pred = clf.predict(sc.transform(Xe[keep]))
        acc = accuracy_score(gold, pred)
        print(f"out-of-domain ({name:4}) {acc:.4f}  on {len(keep)} clips")
        out[f"ood_{name}"] = acc
        dom = defaultdict(lambda: [0, 0])
        for p, g, i in zip(pred, gold, keep):
            d = cfgs[i].split("_")[0]
            dom[d][0] += (p == g); dom[d][1] += 1
        for d in sorted(dom):
            ok, n = dom[d]
            print(f"    {d:10} {ok/n:.3f}  ({ok}/{n})")
        out[f"by_domain_{name}"] = {d: v[0] / v[1] for d, v in dom.items()}

    gap = acc_in - out.get("ood_full", acc_in)
    print(f"\nin-domain minus out-of-domain: {gap:.4f}")
    print("A large gap here is the signature of a model keying on narrator rather than")
    print("language -- the corpus has roughly one voice per language.")

    d = Path("out") / args.tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=2))
    import joblib
    joblib.dump({"clf": clf, "scaler": sc, "labels": list(clf.classes_)},
                d / "model.joblib", compress=3)
    print(f"\nwrote {d}/")


if __name__ == "__main__":
    main()
