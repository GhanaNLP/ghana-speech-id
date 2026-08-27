"""Publish the head to one Hub repo, laid out like AfriSpeech/afrispeech-gender-id.

`config.json` at the repo root is what makes the Hub register downloads -- without it the
model shows no count however many people pull it. It also carries the feature contract the
runtimes have to reproduce exactly.

    config.json          download trigger, feature contract
    metrics.json         in-domain and out-of-domain numbers
    300m/  head.onnx  ngrams.txt  labels.txt  head_config.txt  casefold.txt

One head ships. A 1B-front-end variant was built, measured and retired -- it tied out of
domain and lost in domain -- so publishing must not recreate `1b/`.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = "ghananlpcommunity/ghana-speech-id"
MAIN = "300m"          # the only head; see the module docstring
VARIANTS = {
    "300m": {
        "run": "final_300m_mf50000",
        "front_end": "omniASR CTC 300M (sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc)",
        "note": "the head: small, fast on CPU, and the only front end sherpa-onnx decodes "
                "at a useful rate",
    },
}
FILES = ("head.onnx", "head.fp16.onnx", "ngrams.txt", "labels.txt",
         "head_config.txt", "casefold.txt")


def read_metrics(run: Path):
    m = json.loads((run / "metrics.json").read_text())
    per = {k: v for k, v in m["per_language"].items()
           if k not in ("accuracy", "macro avg", "weighted avg")}
    return m, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hf_upload")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = Path(args.out)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    labels, variants_meta, metrics_out = None, {}, {}
    for name, spec in VARIANTS.items():
        run = Path("out") / spec["run"]
        onnx = run / "onnx"
        if not (onnx / "head.onnx").exists():
            print(f"skip {name}: {onnx}/head.onnx missing")
            continue
        dst = stage / name
        dst.mkdir()
        for f in FILES:
            if (onnx / f).exists():
                shutil.copy2(onnx / f, dst / f)
        m, per = read_metrics(run)
        labels = [l for l in (onnx / "labels.txt").read_text(encoding="utf-8").split("\n") if l]

        ood_path = Path("out") / f"ood_{spec['run']}.json"
        ood = json.loads(ood_path.read_text()) if ood_path.exists() else None
        by_domain = None
        if ood:
            agg = {}
            for cfg, r in ood.items():
                if not isinstance(r, dict) or "acc" not in r:
                    continue
                d = cfg.split("_")[0]
                a, b = agg.get(d, (0, 0))
                agg[d] = (a + r["acc"] * r["n"], b + r["n"])
            by_domain = {d: round(a / b, 4) for d, (a, b) in agg.items() if b}

        variants_meta[name] = {
            "path": name,
            "front_end": spec["front_end"],
            "n_features": m["n_features"],
            "held_out_accuracy": round(m["accuracy"], 4),
            "held_out_macro_f1": round(m["macro_f1"], 4),
            "out_of_domain_accuracy": (round(sum(
                r["acc"] * r["n"] for r in ood.values()
                if isinstance(r, dict) and "acc" in r) / sum(
                r["n"] for r in ood.values()
                if isinstance(r, dict) and "acc" in r), 4) if ood else None),
        }
        metrics_out[name] = {
            "held_out_accuracy": m["accuracy"],
            "held_out_macro_f1": m["macro_f1"],
            "length_curve": m["length_curve"],
            "family_accuracy": m["family_accuracy"],
            "per_language": per,
            "top_confusions": m["top_confusions"][:25],
            "out_of_domain_by_domain": by_domain,
            "configuration": m["args"],
        }
        print(f"staged {name}: {m['accuracy']:.4f} in-domain, "
              f"{variants_meta[name]['out_of_domain_accuracy']} out of domain")

    if not variants_meta:
        raise SystemExit("nothing staged")

    cfg = json.loads(Path("hf_card/config.json").read_text(encoding="utf-8"))
    cfg["num_labels"] = len(labels)
    cfg["labels"] = labels
    # One head, so config.json carries it flat. Writing a "variants" index back would
    # resurrect a choice that was deliberately removed.
    cfg.pop("variants", None)
    cfg.pop("default_variant", None)
    cfg["model"] = variants_meta[MAIN]
    cfg["architecture"] = ("base omniASR CTC orthography (frozen) + linear head over "
                           "character n-grams")
    cfg["input"] = ("orthographic text from a base omniASR CTC model; the head is trained "
                    "on 40-character windows and classifies whole transcripts")
    cfg["features"] = {
        "type": "tf-idf over char_wb n-grams",
        "ngram_range": [1, 5],
        "sublinear_tf": True,
        "norm": "l2",
        "lowercase": True,
        "tokenisation": ("char_wb: each word padded with one space either side, n-grams "
                         "taken within the padded word over CODEPOINTS not bytes"),
        "casefold": ("casefold.txt maps every uppercase form occurring in the vocabulary; "
                     "Unicode folding (Ɛ->ɛ) that std::tolower cannot do"),
    }
    cfg.pop("phoneme_model_repo", None)
    (stage / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (stage / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    print(f"\nstaged in {stage}/")
    for p in sorted(stage.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(stage)}  {p.stat().st_size/1e6:.2f} MB")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    HfApi().upload_folder(folder_path=str(stage), repo_id=args.repo, repo_type="model",
                          commit_message="Head on the base omniASR 300M front end")
    print(f"\nuploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
