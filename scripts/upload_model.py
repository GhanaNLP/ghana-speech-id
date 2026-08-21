"""Publish a trained head to the Hugging Face Hub.

Layout follows AfriSpeech/afrispeech-gender-id, which matters for one specific reason:
`config.json` at the repo root is what makes the Hub register downloads. Without it the
model shows no download count at all, however many people pull it.

Uploads:
    config.json           architecture and the exact feature contract; download trigger
    metrics.json          held-out numbers, per language, with the split described
    onnx/head.onnx        the graph (opset 13, core operators only)
    onnx/head.fp16.onnx   half-size variant, when present
    onnx/ngrams.txt       n-gram vocabulary; index = line number
    onnx/labels.txt       class labels in model order
    onnx/head_config.txt  n-gram range for runtimes without a JSON parser
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = "ghananlpcommunity/ghana-speech-id"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="out/<tag> directory from train_head.py")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--ood", default="out/ood_eval.json",
                    help="out-of-domain results, folded into metrics.json when present")
    ap.add_argument("--message", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    onnx_dir = run / "onnx"
    metrics = json.loads((run / "metrics.json").read_text())
    labels = [l for l in (onnx_dir / "labels.txt").read_text(encoding="utf-8").split("\n") if l]

    stage = Path("hf_upload")
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "onnx").mkdir(parents=True)

    for name in ("head.onnx", "head.fp16.onnx", "ngrams.txt", "labels.txt", "head_config.txt"):
        src = onnx_dir / name
        if src.exists():
            shutil.copy2(src, stage / "onnx" / name)

    # config.json: keep the card's existing text, fill in what the run determines
    cfg = json.loads(Path("hf_card/config.json").read_text(encoding="utf-8"))
    cfg["num_labels"] = len(labels)
    cfg["labels"] = labels
    cfg["features"]["ngram_range"] = list(metrics["args"]["ngram_range"]) \
        if "ngram_range" in metrics["args"] else [1, metrics["args"]["ngram_max"]]
    cfg["features"]["max_features"] = metrics["n_features"]
    cfg["features"]["punctuation"] = "dropped" if metrics["args"]["drop_punct"] else "kept"
    cfg["classifier"] = metrics["args"]["model"]
    (stage / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")

    # metrics.json: strip sklearn's non-class rows so per_language really is per language
    per_lang = {k: v for k, v in metrics["per_language"].items()
                if k not in ("accuracy", "macro avg", "weighted avg")}
    out = {
        "held_out_accuracy": metrics["accuracy"],
        "held_out_macro_f1": metrics["macro_f1"],
        "num_labels": len(labels),
        "split": ("contiguous: the last 15% of each language by id. Because the audio is "
                  "scripture read in order, this approximates holding out whole books. A "
                  "random split scores about 2 points higher; that gap is passage-local "
                  "memorisation."),
        "configuration": metrics["args"],
        "n_features": metrics["n_features"],
        "length_curve": metrics["length_curve"],
        "family_accuracy": metrics["family_accuracy"],
        "errors": metrics["errors"],
        "errors_within_family": metrics["errors_within_family"],
        "top_confusions": metrics["top_confusions"][:25],
        "per_language": per_lang,
    }
    ood = Path(args.ood)
    if ood.exists():
        out["out_of_domain"] = json.loads(ood.read_text())
    else:
        print(f"note: {ood} not found; metrics.json will carry in-domain numbers only")
    (stage / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"staged in {stage}/")
    for p in sorted(stage.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(stage)}  {p.stat().st_size/1e6:.2f} MB")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path=str(stage), repo_id=args.repo, repo_type="model",
        commit_message=args.message or f"Head from {run.name}: "
                                       f"{metrics['accuracy']:.4f} held-out accuracy")
    print(f"\nuploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
