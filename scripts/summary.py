import json, glob
rows = []
for f in sorted(glob.glob("out/*/metrics.json")):
    d = json.load(open(f))
    lc = d.get("length_curve", {})
    # classification_report adds accuracy / macro avg / weighted avg keys, which are
    # not classes; counting them inflated the weight-matrix size by ~7%
    nC = sum(1 for k in d["per_language"] if k not in
             ("accuracy", "macro avg", "weighted avg"))
    rows.append((d["tag"], d["accuracy"], d["macro_f1"], d["n_features"], nC,
                 d["n_features"] * nC * 4 / 1e6,
                 lc.get("20", {}).get("acc"), lc.get("40", {}).get("acc"),
                 100 * d["errors_within_family"] / max(d["errors"], 1)))
rows.sort(key=lambda r: -r[1])
hdr = f"{'tag':48} {'acc':>7} {'macroF1':>8} {'feats':>7} {'fp32MB':>7} {'@20u':>6} {'@40u':>6} {'inFam':>6}"
print(hdr); print("-" * len(hdr))
for t, a, m, nf, nc, mb, a20, a40, wf in rows:
    s20 = f"{a20:.3f}" if a20 else "-"
    s40 = f"{a40:.3f}" if a40 else "-"
    print(f"{t:48} {a:7.4f} {m:8.4f} {nf:7d} {mb:7.1f} {s20:>6} {s40:>6} {wf:5.1f}%")
