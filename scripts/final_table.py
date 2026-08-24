import json, glob
rows = []
for f in sorted(glob.glob("out/final_*/metrics.json")):
    tag = f.split("/")[1]
    d = json.load(open(f))
    try:
        ood = json.load(open(f"out/ood_{tag}.json"))
        vals = [(r["acc"], r["n"]) for r in ood.values() if isinstance(r, dict) and "acc" in r]
        o = sum(a * n for a, n in vals) / sum(n for _, n in vals)
    except Exception:
        o = None
    nC = sum(1 for k in d["per_language"]
             if k not in ("accuracy", "macro avg", "weighted avg"))
    rows.append((tag, d["accuracy"], d["macro_f1"], d["n_features"] * nC * 4 / 1e6, o,
                 d["length_curve"].get("40", {}).get("acc"),
                 d["length_curve"].get("10", {}).get("acc")))
hdr = f"{'head':26} {'in-dom':>7} {'macroF1':>8} {'MB':>6} {'OOD':>7} {'@40ch':>7} {'@10ch':>7}"
print(hdr); print("-" * len(hdr))
for t, a, m, mb, o, c40, c10 in rows:
    so = f"{o:.4f}" if o else "-"
    print(f"{t:26} {a:7.4f} {m:8.4f} {mb:6.1f} {so:>7} {c40:7.4f} {c10:7.4f}")
