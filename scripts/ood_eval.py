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
import time
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

PUNCT = set(".,!?;:\"'()-—…")


def strip_punct(s: str) -> str:
    return " ".join(u for u in s.split() if u not in PUNCT)


def phonemise_all(configs, per_config, cache_path, keep_punct):
    """Decode each config to IPA once and cache it; decoding dominates the runtime."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cache = {}
    if cache_path.exists():
        t = pq.read_table(cache_path).to_pydict()
        for cfg, ipa, iso, lang in zip(t["subset"], t["ipa"], t["iso"], t["language"]):
            cache.setdefault(cfg, []).append({"ipa": ipa, "iso": iso, "language": lang})
        print(f"loaded {sum(len(v) for v in cache.values())} cached transcriptions "
              f"for {len(cache)} configs")

    todo = [c for c in configs if c not in cache]
    if todo:
        from datasets import Audio, load_dataset
        from ghana_ipa_asr import GhanaIPAASR
        asr = GhanaIPAASR.load()
        for cfg in todo:
            t0 = time.time()
            try:
                ds = load_dataset("ghananlpcommunity/ghana-speech-eval", cfg, split="eval")
            except Exception as e:
                print(f"  {cfg}: load failed ({type(e).__name__}: {str(e)[:90]})", flush=True)
                continue
            n = min(per_config, len(ds)) if per_config else len(ds)
            ds = ds.select(range(n)).cast_column("audio", Audio(sampling_rate=16000))
            rows, wavs, meta = [], [], []
            for r in ds:
                wavs.append(np.asarray(r["audio"]["array"], dtype=np.float32))
                meta.append((r.get("iso", ""), r.get("language", "")))
            secs = sum(len(w) for w in wavs) / 16000
            B = 16
            for i in range(0, len(wavs), B):
                for j, tr in enumerate(asr.transcribe_batch(wavs[i:i + B], sample_rate=16000)):
                    iso, lang = meta[i + j]
                    rows.append({"ipa": tr.spaced(punctuation=keep_punct),
                                 "iso": iso, "language": lang})
            cache[cfg] = rows
            dt = time.time() - t0
            print(f"  {cfg:26} {len(rows):5d} clips {secs/60:6.1f} min audio "
                  f"{dt:6.0f}s ({secs/max(dt,1e-9):5.1f}x RT)", flush=True)

            flat = [(c, r["ipa"], r["iso"], r["language"])
                    for c, rs in cache.items() for r in rs]
            pq.write_table(pa.table({
                "subset": [x[0] for x in flat], "ipa": [x[1] for x in flat],
                "iso": [x[2] for x in flat], "language": [x[3] for x in flat],
            }), cache_path, compression="zstd")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model.joblib from train_head.py")
    ap.add_argument("--per-config", type=int, default=0, help="0 uses every clip")
    ap.add_argument("--keep-punct", action="store_true",
                    help="must match the head; the sweep trains with --drop-punct")
    ap.add_argument("--cache", default="data/eval_ipa.parquet")
    ap.add_argument("--out", default="out/ood_eval.json")
    args = ap.parse_args()

    import joblib
    from huggingface_hub import HfApi

    bundle = joblib.load(args.model)
    vec, clf = bundle["vec"], bundle["clf"]
    labels = list(bundle["labels"])
    has_english = "English_eng" in labels

    cfgs = sorted(c for c in HfApi().dataset_info(
        "ghananlpcommunity/ghana-speech-eval").config_names if not c.startswith("bible_"))
    print(f"{len(cfgs)} non-bible configs (bible_* skipped: that is the training domain)\n")

    cache = phonemise_all(cfgs, args.per_config, Path(args.cache), args.keep_punct)

    def score(strings):
        strings = [s if args.keep_punct else strip_punct(s) for s in strings]
        keep = [i for i, s in enumerate(strings) if len(s.split()) >= 3]
        if not keep:
            return None, None, 0
        X = vec.transform([strings[i] for i in keep])
        dec = clf.decision_function(X)
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
