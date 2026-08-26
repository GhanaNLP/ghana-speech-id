"""Decide which front ends to publish, by the rule: keep what earns its size.

  1. A variant below the omniASR baseline is dropped. Being smaller does not excuse being
     less accurate than what already ships.
  2. Among survivors, drop anything Pareto-dominated -- another variant that is no larger
     and no less accurate. A bigger model is kept only when it actually buys accuracy.
  3. What remains is the frontier, and every point on it is a real choice: more accuracy for
     more size, or less of both.

The criterion is out-of-domain accuracy on the first 20 characters -- about 1.6 seconds of
speech. Short audio because that is what the app gets; out of domain because the whole
project turned on that gap, and 95% in-domain meant 36% in the wild once. Overall and
in-domain figures are reported alongside but do not decide anything.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# front end -> the artefact that would be served, since size means the served size
SERVED = {
    "final_300m":  ("omniASR 300M int8",  350.0, 17.0),
    "final_1b":    ("omniASR 1B int8",    986.0, None),
    "zipa-small":       ("zipa-small int8",  70.7, 100.8),
    "zipa-small-fp16":  ("zipa-small fp16", 131.6,  77.0),
    "zipa-large":       ("zipa-large int8", 309.9,  46.1),
    "zipa-large-fp16":  ("zipa-large fp16", 604.3,  28.6),
}


def ood(tag, at="20"):
    """Out-of-domain accuracy at a transcript length. Falls back to the overall figure for
    runs scored before the length curve existed."""
    p = f"out/ood_{tag}.json"
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    v = [(r["acc"], r["n"]) for r in d.values() if isinstance(r, dict) and "acc" in r]
    overall = sum(a * n for a, n in v) / sum(n for _, n in v) if v else None
    return (d.get("length_curve_ood") or {}).get(at), overall


def indomain(tag):
    p = f"out/{tag}/metrics.json"
    return json.load(open(p))["accuracy"] if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="50000",
                    help="compare at one feature count so size differences are the front end")
    args = ap.parse_args()

    rows = []
    for prefix, (label, mb, speed) in SERVED.items():
        for tag in sorted(glob.glob(f"out/{prefix}*mf{args.features}*/metrics.json")):
            t = tag.split("/")[1]
            at20, overall = ood(t)
            i = indomain(t)
            if at20 is None or i is None:
                print(f"  {label}: not scored at 20 chars yet, re-run ood_eval")
                continue
            rows.append({"tag": t, "label": label, "mb": mb, "speed": speed,
                         "ood": at20, "overall": overall, "in": i})
            break

    if not rows:
        print("no completed runs yet"); return

    base = next((r for r in rows if r["tag"].startswith("final_300m")), None)
    if base is None:
        print("omniASR 300M baseline missing; cannot apply the rule"); return

    print(f"criterion: out-of-domain accuracy on the first 20 characters (~1.6 s)\n")
    print(f"baseline: {base['label']}  {base['ood']:.4f} at 20 chars  "
          f"({base['overall']:.4f} whole, {base['in']:.4f} in-domain)\n")
    hdr = (f"{'variant':22} {'MB':>7} {'xRT':>6} {'OOD@20':>8} {'OODall':>8} "
           f"{'in-dom':>8}  verdict")
    print(hdr); print("-" * len(hdr))

    survivors = []
    for r in sorted(rows, key=lambda x: x["mb"]):
        if r is base:
            verdict = "baseline"
        elif r["ood"] <= base["ood"]:
            verdict = f"drop (below baseline by {base['ood']-r['ood']:.4f})"
        else:
            verdict = "keep"
            survivors.append(r)
        sp = f"{r['speed']:.0f}" if r["speed"] else "-"
        ov = f"{r['overall']:.4f}" if r["overall"] else "-"
        print(f"{r['label']:22} {r['mb']:7.0f} {sp:>6} {r['ood']:8.4f} {ov:>8} "
              f"{r['in']:8.4f}  {verdict}")

    # baseline competes on the frontier too: if something smaller beats it, it is dominated
    pool = survivors + [base]
    frontier = []
    for r in pool:
        dominated = any(o is not r and o["mb"] <= r["mb"] and o["ood"] >= r["ood"]
                        and (o["mb"] < r["mb"] or o["ood"] > r["ood"]) for o in pool)
        if not dominated:
            frontier.append(r)

    print(f"\npublish ({len(frontier)}):")
    for r in sorted(frontier, key=lambda x: x["mb"]):
        print(f"  {r['label']:22} {r['mb']:6.0f} MB  {r['ood']:.4f} at 20 chars")
    dropped = [r for r in pool if r not in frontier]
    if dropped:
        print("dominated, not published:")
        for r in sorted(dropped, key=lambda x: x["mb"]):
            better = min((o for o in frontier if o["mb"] <= r["mb"]),
                         key=lambda o: -o["ood"], default=None)
            why = f"{better['label']} is no larger and more accurate" if better else "dominated"
            print(f"  {r['label']:22} -- {why}")


if __name__ == "__main__":
    main()
