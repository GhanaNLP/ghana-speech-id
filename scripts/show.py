import json, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "ceiling_char_mf200000_truth"
d = json.load(open(f"out/{tag}/metrics.json"))
print(f"{tag}")
print(f"  accuracy {d['accuracy']:.4f}  macro-F1 {d['macro_f1']:.4f}  feats {d['n_features']}")
print(f"  family accuracy {d['family_accuracy']:.4f}")
print("\naccuracy vs first-K units/chars:")
for k, v in d["length_curve"].items():
    print(f"  first {str(k):>4}: acc {v['acc']:.4f}  macroF1 {v['macro_f1']:.4f}")
per = {k: v for k, v in d["per_language"].items()
       if k not in ("accuracy", "macro avg", "weighted avg")}
print("\nweakest languages:")
for l, r in sorted(per.items(), key=lambda kv: kv[1]["f1-score"])[:6]:
    print(f"  {l:26} n={int(r['support']):5d} f1 {r['f1-score']:.4f}")
print("\ntop confusions:")
for a, b, c in d["top_confusions"][:6]:
    print(f"  {a:24} -> {b:24} {c}")
