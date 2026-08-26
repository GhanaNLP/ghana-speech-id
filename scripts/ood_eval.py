"""Out-of-domain evaluation on ghananlpcommunity/ghana-speech-eval.

Everything the head is trained on is scripture narration, so an in-domain score says little
about real use. This runs the whole production path -- audio, sherpa-onnx, ghana-ipa-asr,
head -- over five unrelated domains (finance, JW, LDS, UNICEF, Waxal) and reports per domain,
because a single averaged number hides which ones break.

The bible_* configs are skipped: they are the training domain.

Three of the languages here are NOT in the label set -- Ga, Ahanta and Ikposo. Those are not
scored for accuracy, since every prediction is wrong by construction. They are the material
for the rejection curve: a closed-set head will confidently name an in-set language for them,
and the question is whether a confidence threshold can catch that without throwing away
correct in-set answers.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# iso in the eval set -> our label. Frafra is Gurene/Ninkare, same ISO `gur`.
ISO_TO_LABEL = {
    "twi": "Twi_twi", "twi_asante": "Twi_twi", "twi_akuapem": "Twi_twi",
    "fat": "Fante_fat", "dga": "Dagaare_dga", "dag": "Dagbani_dag",
    "ada": "Dangme_ada", "ewe": "Ewe_ewe", "gur": "Ninkare_gur",
    "nzi": "Nzema_nzi", "sfw": "Sehwi_sfw", "eng": "English_eng",
}
# Present in the eval set, absent from the label set. Used for the rejection curve.
OUT_OF_SET = {"gaa": "Ga", "aha": "Ahanta", "kpo": "Ikposo"}

# decode_base.py records the config directory rather than an iso code, so map it back.
# Spelled out rather than parsed off the suffix: finance_ga and unicef_dagbani do not
# carry one, and a silent mismap would quietly score the wrong language as correct.
CONFIG_TO_ISO = {
    "finance_Akuapem_Twi": "twi", "finance_Asante_Twi": "twi", "finance_fante": "fat",
    "finance_ga": "gaa",
    "jw_ahanta_aha": "aha", "jw_dagaare_dga": "dga", "jw_dangme_ada": "ada",
    "jw_ewe_ewe": "ewe", "jw_fante_fat": "fat", "jw_frafra_gur": "gur",
    "jw_ga_gaa": "gaa", "jw_nzema_nzi": "nzi", "jw_sehwi_sfw": "sfw",
    "lds_Asante_Twi": "twi", "lds_Fante_fat": "fat",
    "unicef_Asante_Twi": "twi", "unicef_dagbani": "dag", "unicef_ewe": "ewe",
    "waxal_Asante_Twi": "twi", "waxal_Dagaare_dga": "dga", "waxal_Dagbani_dag": "dag",
    "waxal_Ewe_ewe": "ewe", "waxal_Ikposo_kpo": "kpo",
}

PUNCT = set(".,!?;:\"'()-—…")


def chunks(s: str, size: int, stride: int) -> list[str]:
    """Same windowing the chunked head was trained on; must match or voting is scoring a
    different distribution from the one the model learned."""
    if size <= 0 or len(s) <= size:
        return [s]
    step = max(1, stride)
    out = [s[i:i + size] for i in range(0, len(s) - size + 1, step)]
    tail = s[-size:]
    if out and out[-1] != tail:
        out.append(tail)
    return out


def strip_punct(s: str) -> str:
    """Drop punctuation units.

    IPA transcripts carry punctuation as standalone whitespace-separated units, so removing
    them is a token filter. Orthography attaches them to words, where this is a no-op --
    which is correct, since the char analyzer is trained on text that still has them.
    """
    return " ".join(u for u in s.split() if u not in PUNCT)


def load_decoded(path: Path, text_col: str = "ipa"):
    """Read decode_gpu.py's output and group it by eval config.

    Decoding lives in decode_gpu.py so it runs on the GPU: 53 hours of audio at ~900x
    realtime is three minutes there against roughly six hours across eight CPU workers.
    """
    import pyarrow.parquet as pq

    have = set(pq.ParquetFile(path).schema_arrow.names)
    # decode_gpu.py writes group/ipa/iso; decode_base.py writes language/text and no iso
    cfg_col = "group" if "group" in have else "language"
    txt_col = text_col if text_col in have else ("ipa" if "ipa" in have else "text")
    cols = ["id", cfg_col, txt_col] + (["iso"] if "iso" in have else [])
    t = pq.read_table(path, columns=cols).to_pydict()
    by_cfg = defaultdict(list)
    for i, (_id, cfg) in enumerate(zip(t["id"], t[cfg_col])):
        iso = t["iso"][i] if "iso" in t else CONFIG_TO_ISO.get(cfg, "")
        by_cfg[cfg].append({"ipa": t[txt_col][i] or "", "iso": iso})
    print(f"columns: config={cfg_col} text={txt_col} "
          f"iso={'present' if 'iso' in t else 'derived from config'}")
    print(f"loaded {len(t['id'])} decoded clips across {len(by_cfg)} configs from {path}")
    return by_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model.joblib from train_head.py")
    ap.add_argument("--decoded", default="data/eval_ipa_gh.parquet",
                    help="output of decode_gpu.py --hf-eval")
    ap.add_argument("--truncate", default="20,40,80,0",
                    help="also score with transcripts cut to these character counts; 0 is "
                         "the whole transcript. 20 chars is about 1.6 s of speech and is "
                         "the figure variants are selected on -- short audio under real "
                         "conditions, which is what the app faces.")
    ap.add_argument("--chunk-chars", type=int, default=0,
                    help="window size for voted inference; must match how the head was "
                         "trained")
    ap.add_argument("--chunk-stride", type=int, default=20)
    ap.add_argument("--text-col", default="ipa",
                    help="transcript column; base decodes use 'text'")
    ap.add_argument("--keep-punct", action="store_true",
                    help="must match the head; the sweep trains with --drop-punct")
    ap.add_argument("--out", default="out/ood_eval.json")
    args = ap.parse_args()

    import joblib

    bundle = joblib.load(args.model)
    vec, clf = bundle["vec"], bundle["clf"]
    labels = list(bundle["labels"])
    has_english = "English_eng" in labels

    cache = load_decoded(Path(args.decoded), args.text_col)
    cfgs = sorted(cache)
    print(f"{len(cfgs)} non-bible configs (bible_* skipped: that is the training domain)\n")

    def score(strings):
        strings = [s if args.keep_punct else strip_punct(s) for s in strings]
        # match the trainer: continuous phone strings have no whitespace, so counting
        # tokens would discard every clip
        char_mode = getattr(vec, "analyzer", "") == "char_wb"
        keep = [i for i, s in enumerate(strings)
                if (len(s) if char_mode else len(s.split())) >= 3]
        if not keep:
            return None, None, 0
        docs = [strings[i] for i in keep]

        if args.chunk_chars:
            # classify every window and sum decision values per document, exactly as the
            # on-device path will: a confident window should outweigh several uncertain ones
            flat, owner = [], []
            for di, d in enumerate(docs):
                for c in chunks(d, args.chunk_chars, args.chunk_stride):
                    flat.append(c); owner.append(di)
            acc = np.zeros((len(docs), len(labels)))
            step = 20000
            for b0 in range(0, len(flat), step):
                sl = slice(b0, b0 + step)
                d = clf.decision_function(vec.transform(flat[sl]))
                if d.ndim == 1:
                    d = np.stack([-d, d], 1)
                for k, row in enumerate(d):
                    acc[owner[b0 + k]] += row
            dec = acc
        else:
            dec = clf.decision_function(vec.transform(docs))
            if dec.ndim == 1:
                dec = np.stack([-dec, dec], 1)

        order = np.argsort(dec, axis=1)
        pred = [labels[i] for i in order[:, -1]]
        margin = dec[np.arange(len(dec)), order[:, -1]] - dec[np.arange(len(dec)), order[:, -2]]
        return pred, margin, len(keep)

    results, by_domain = {}, defaultdict(lambda: [0, 0])
    print("== in-set languages, by config ==")
    print(f"{'config':26} {'n':>5} {'acc':>7} {'median margin':>14}  most common error")
    in_margins = []
    for cfg in cfgs:
        rows = cache.get(cfg)
        if not rows:
            continue
        iso = rows[0]["iso"]
        if iso in OUT_OF_SET:
            continue
        gold = ISO_TO_LABEL.get(iso)
        if gold is None:
            print(f"{cfg:26} -- unmapped iso {iso!r}, skipped")
            continue
        pred, margin, n = score([r["ipa"] for r in rows])
        if not n:
            continue
        ok = sum(1 for p in pred if p == gold)
        wrong = defaultdict(int)
        for p in pred:
            if p != gold:
                wrong[p] += 1
        top_wrong = max(wrong.items(), key=lambda kv: kv[1])[0] if wrong else "-"
        dom = cfg.split("_")[0]
        by_domain[dom][0] += ok
        by_domain[dom][1] += n
        in_margins.extend(margin.tolist())
        results[cfg] = {"n": n, "gold": gold, "acc": ok / n,
                        "median_margin": float(np.median(margin)), "top_error": top_wrong}
        print(f"{cfg:26} {n:5d} {ok/n:7.3f} {np.median(margin):14.3f}  {top_wrong}")

    print("\n== by domain ==")
    for dom, (ok, n) in sorted(by_domain.items()):
        print(f"  {dom:10} {ok/n:.3f}  ({ok}/{n})")
    tot_ok = sum(v[0] for v in by_domain.values())
    tot_n = sum(v[1] for v in by_domain.values())
    print(f"  {'OVERALL':10} {tot_ok/tot_n:.3f}  ({tot_ok}/{tot_n})")

    print("\n== languages outside the label set ==")
    print("Every prediction is wrong by construction; what matters is whether the margin is")
    print("low enough to reject without discarding correct in-set answers.")
    out_margins = []
    for cfg in cfgs:
        rows = cache.get(cfg)
        if not rows:
            continue
        iso = rows[0]["iso"]
        if iso not in OUT_OF_SET:
            continue
        pred, margin, n = score([r["ipa"] for r in rows])
        if not n:
            continue
        top = sorted(((p, pred.count(p)) for p in set(pred)), key=lambda kv: -kv[1])[:3]
        out_margins.extend(margin.tolist())
        results[cfg] = {"n": n, "out_of_set": OUT_OF_SET[iso],
                        "median_margin": float(np.median(margin)),
                        "top_predictions": top}
        print(f"  {cfg:26} ({OUT_OF_SET[iso]:7}) n={n:5d} median margin "
              f"{np.median(margin):6.3f}  -> " + ", ".join(f"{k}({v})" for k, v in top))

    if in_margins and out_margins:
        print("\n== rejection curve: threshold on margin ==")
        print(f"{'threshold':>10} {'in-set kept':>12} {'out-of-set rejected':>21}")
        ins, outs = np.array(in_margins), np.array(out_margins)
        for q in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]:
            thr = float(np.quantile(ins, q))
            kept = float((ins >= thr).mean())
            rej = float((outs < thr).mean())
            print(f"{thr:10.3f} {kept:11.1%} {rej:20.1%}")
        results["rejection_curve"] = {
            "in_set_median_margin": float(np.median(ins)),
            "out_of_set_median_margin": float(np.median(outs)),
        }

    if has_english:
        print("\nNote: this head has an English class, but ghana-speech-eval has no English")
        print("config, so English recall is not measured here.")

    # accuracy against how much transcript the head is given, out of domain
    tr = [int(x) for x in args.truncate.split(",") if x.strip()]
    if tr:
        print("\n== out-of-domain accuracy vs transcript length ==")
        curve = {}
        for k in tr:
            ok = n = 0
            for cfg, rows in cache.items():
                iso = rows[0]["iso"] if rows else ""
                gold = ISO_TO_LABEL.get(iso)
                if gold is None or iso in OUT_OF_SET:
                    continue
                cut = [(r["ipa"] or "")[:k] if k else (r["ipa"] or "") for r in rows]
                pred, _, m = score(cut)
                if not m:
                    continue
                ok += sum(1 for p in pred if p == gold); n += m
            if n:
                curve[str(k) if k else "full"] = round(ok / n, 4)
                label = f"first {k} chars" if k else "whole transcript"
                print(f"  {label:20} {ok/n:.4f}  ({ok}/{n})")
        results["length_curve_ood"] = curve

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
