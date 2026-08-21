"""Language ID over IPA phoneme strings from ghana-speech-phoneme-asr.

Input at inference is whatever the CTC model emits, so training on the corpus `ipa` column
is distribution-matched by construction: the same recogniser, the same error modes. We are
deliberately NOT learning from audio -- a phoneme string carries no narrator or mic chain,
which is what makes this immune to the Bible-recording confound.

Units are space separated and several are multi-character (kʰ, k͡p, t͡ʃ, nʷ). Everything
here splits on whitespace, never on characters.
"""
from __future__ import annotations
import argparse, json, time, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Approximate genetic grouping, used only to read the confusion matrix -- close relatives
# are where LID errors concentrate and an overall accuracy number hides that.
FAMILY = {
    "Kwa_Akan": ["Akuapem_Twi_twi", "Asante_Twi_twi", "Twi_twi", "Fante_fat", "Anyin_any",
                 "Sehwi_sfw", "Nzema_nzi"],
    "Kwa_Guang": ["Gonja_gjn", "Chumburung_ncu", "Nawuri_naw", "Gikyode_acd", "Nkonya_nko"],
    "Kwa_GTM": ["Avatime_avn", "Lelemi_lef", "Sekpele_lip", "Selee_snw", "Siwu_akp", "Tuwuli_bov", "Ntrubo_ntr"],
    "Kwa_other": ["Ewe_ewe", "Dangme_ada"],
    "Gur_Oti_Volta": ["Dagaare_dga", "Dagbani_dag", "Mampruli_maw", "Kusaal_kus", "Buli_bwu",
                       "Konni_kma", "Ninkare_gur", "Birifor_Southern_biv", "Konkomba_xon",
                       "Bimoba_bim", "Bassar_Ntcham_bud"],
    "Gur_Grusi": ["Kasem_xsm", "Sisaala_Tumulung_sil", "Paasaal_sig", "Vagla_vag", "Deg_mzw",
                   "Tampulma_tpm", "Tem_kdh", "Kabiye_kbp"],
    "Mande": ["Bissa_bib"],
    "Afroasiatic": ["Hausa_hau"],
    "Atlantic": ["Fulfulde_Maasina_ffm"],
}
LANG2FAM = {l: f for f, ls in FAMILY.items() for l in ls}

PUNCT = set(".,!?;:\"'()-—…")


def iso_of(label: str) -> str:
    """Trailing segment of a label is its ISO 639-3 code (Asante_Twi_twi -> twi)."""
    return label.rsplit("_", 1)[-1]


def merge_map(labels):
    """Collapse labels that share an ISO code into one class.

    Asante Twi and Akuapem Twi are mutually intelligible dialects of Akan sharing ISO `twi`,
    and they were the only pair the head genuinely struggled with (~12% of all errors). Fante
    is ISO `fat` and separates cleanly, so it keeps its own class. `twi` is the only duplicated
    code across the 42, so this collapse is exactly the Twi merge and nothing else.
    """
    by_iso = defaultdict(list)
    for l in set(labels):
        by_iso[iso_of(l)].append(l)
    out = {}
    for iso, ls in by_iso.items():
        tgt = iso.capitalize() + "_" + iso if len(ls) > 1 else ls[0]
        for l in ls:
            out[l] = tgt
    return out


def units(s: str) -> list[str]:
    return s.split()


def strip_punct(s: str) -> str:
    return " ".join(u for u in s.split() if u not in PUNCT)


def truncate(s: str, k: int, analyzer: str = "word") -> str:
    """First k units -- simulates a shorter clip without needing the audio.

    The unit has to match the analyzer or the curves are not comparable: whitespace tokens
    are phonemes for IPA but words for orthography, and 5 phonemes is about half a second
    where 5 words is two or three. For orthography, truncate by character instead.
    """
    if k <= 0:
        return s
    return s[:k] if analyzer == "char" else " ".join(s.split()[:k])


def load(path: str, drop_punct: bool, min_units: int, split_mode: str, test_frac: float,
         merge_iso: bool = False, text_col: str = "ipa"):
    """Build train/validation.

    split_mode:
      shipped     -- the corpus's own validation split. Only 1,680 rows (40 per language),
                     fine for plumbing, too small for per-language F1 or a 42x42 matrix.
      contiguous  -- hold out the LAST test_frac of each language by id. ids are sequential
                     and the audio is scripture read in order, so a contiguous tail is
                     approximately a held-out set of different books. This is the honest
                     number: it prevents the head from being scored on neighbouring verses
                     of the passages it trained on.
      random      -- random per-language holdout. Reported only for the gap against
                     contiguous, which measures how much passage-local memorisation helps.
    """
    have = set(pq.ParquetFile(path).schema_arrow.names)
    cols = ["id", "language", text_col, "duration"]
    if "split" in have:
        cols.append("split")
    t = pq.read_table(path, columns=cols).to_pydict()
    if "split" not in t:
        # decode_base.py does not carry one; contiguous and random modes derive their own
        if split_mode == "shipped":
            raise SystemExit(f"{path} has no split column; use --split-mode contiguous")
        t["split"] = ["train"] * len(t["id"])
    mm = merge_map(t["language"]) if merge_iso else {}
    groups = defaultdict(list)
    for k, v in mm.items():
        if k != v:
            groups[v].append(k)
    for v, ks in groups.items():
        print("merging " + " + ".join(sorted(ks)) + " -> " + v)
    rows = []
    dropped = 0
    for _id, lang, ipa, dur, split in zip(t["id"], t["language"], t[text_col], t["duration"], t["split"]):
        if ipa is None:
            dropped += 1; continue
        s = strip_punct(ipa) if drop_punct else ipa
        if len(s.split()) < min_units:
            dropped += 1; continue
        rows.append((s, mm.get(lang, lang), dur, _id, split))
    print(f"loaded {len(t['id'])} rows, dropped {dropped} (empty or <{min_units} units)")

    out = {"train": [], "validation": []}
    if split_mode == "shipped":
        for s, lang, dur, _id, split in rows:
            out["validation" if split == "validation" else "train"].append((s, lang, dur))
    else:
        by_lang = defaultdict(list)
        for r in rows:
            by_lang[r[1]].append(r)
        for lang, rs in by_lang.items():
            if split_mode == "contiguous":
                rs.sort(key=lambda r: r[3])          # id order == reading order
            else:
                import random as _r
                _r.Random(hash(lang) & 0xffff).shuffle(rs)
            cut = int(len(rs) * (1 - test_frac))
            for r in rs[:cut]:
                out["train"].append((r[0], r[1], r[2]))
            for r in rs[cut:]:
                out["validation"].append((r[0], r[1], r[2]))
    for k, v in out.items():
        print(f"  {k}: {len(v)}")
    return out


def build_model(kind: str, ngram_max: int, max_features: int, min_df: int,
                analyzer: str = "word"):
    """analyzer picks how a transcript is cut into features, and it must follow the
    front-end that produced it.

      word  ghana-ipa-asr emits space-separated IPA units, several of them multi-character
            (kʰ, k͡p, t͡ʃ). Whitespace tokens ARE the phonemes, and splitting on characters
            would turn one sound into two.

      char  base omniASR emits ordinary orthography, so whitespace tokens are words.
            Word n-grams are far sparser and weaker on short utterances; character n-grams
            capture the spelling conventions and morphology that separate these languages.
    """
    if analyzer == "char":
        vec = TfidfVectorizer(
            analyzer="char_wb", lowercase=True, ngram_range=(1, ngram_max),
            max_features=max_features, min_df=min_df, sublinear_tf=True,
            use_idf=(kind != "nb"), norm="l2" if kind != "nb" else None)
        clf = _classifier(kind)
        return vec, clf
    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\S+",       # units are whitespace separated; never split kʰ or k͡p
        lowercase=False,            # IPA case is meaningful
        ngram_range=(1, ngram_max),
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        use_idf=(kind != "nb"),     # NB wants counts, not idf-weighted reals
        norm="l2" if kind != "nb" else None,
    )
    return vec, _classifier(kind)


def _classifier(kind: str):
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, C=10.0, solver="lbfgs", n_jobs=-1)
    elif kind == "sgd":
        # modified_huber gives predict_proba and scales to this feature count
        clf = SGDClassifier(loss="modified_huber", alpha=1e-6, max_iter=25,
                            tol=1e-4, n_jobs=-1, random_state=0)
    elif kind == "svm":
        clf = LinearSVC(C=1.0)
    elif kind == "nb":
        clf = MultinomialNB(alpha=0.1)
    else:
        raise SystemExit(f"unknown model {kind}")
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/volume_d2wey28/projects/ghana-speech-id/data/ipa_text.parquet")
    ap.add_argument("--out", default="/mnt/volume_d2wey28/projects/ghana-speech-id/out")
    ap.add_argument("--model", default="svm", choices=["logreg", "svm", "nb", "sgd"])
    ap.add_argument("--ngram-max", type=int, default=5)
    ap.add_argument("--max-features", type=int, default=500_000)
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--drop-punct", action="store_true",
                    help="punctuation units are only ~62%% accurate upstream and carry little "
                         "language signal; dropping them is usually the right default")
    ap.add_argument("--min-units", type=int, default=3)
    ap.add_argument("--analyzer", default="word", choices=["word", "char"],
                    help="word for IPA units from ghana-ipa-asr; char for "
                         "orthography from base omniASR")
    ap.add_argument("--text-col", default="ipa",
                    help="column holding the transcript; base decodes use 'text'")
    ap.add_argument("--split-mode", default="contiguous",
                    choices=["contiguous", "random", "shipped"],
                    help="contiguous holds out the last test-frac of each language by id, "
                         "which approximates holding out whole books")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--merge-iso", action="store_true",
                    help=("collapse labels sharing an ISO 639-3 code; across these 42 "
                          "that is exactly Asante+Akuapem Twi -> Twi_twi (41 classes)"))
    ap.add_argument("--limit", type=int, default=0,
                    help="subsample this many training rows (smoke runs)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or (f"{args.model}_ng{args.ngram_max}_mf{args.max_features}_{args.split_mode}"
                       + ("_nopunct" if args.drop_punct else "")
                       + ("_twimerged" if args.merge_iso else ""))
    outdir = Path(args.out) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    data = load(args.data, args.drop_punct, args.min_units, args.split_mode, args.test_frac,
                args.merge_iso, args.text_col)
    tr = data["train"]
    if args.limit and args.limit < len(tr):
        import random
        random.Random(0).shuffle(tr)
        tr = tr[: args.limit]
        print(f"smoke: subsampled train to {len(tr)}")
    Xtr_raw = [r[0] for r in tr]; ytr = [r[1] for r in tr]
    Xva_raw = [r[0] for r in data["validation"]]; yva = [r[1] for r in data["validation"]]

    vec, clf = build_model(args.model, args.ngram_max, args.max_features,
                           args.min_df, args.analyzer)

    t0 = time.time()
    Xtr = vec.fit_transform(Xtr_raw)
    print(f"vectorised train {Xtr.shape} in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    print(f"fit in {time.time()-t0:.0f}s", flush=True)

    Xva = vec.transform(Xva_raw)
    pred = clf.predict(Xva)
    acc = accuracy_score(yva, pred); mf1 = f1_score(yva, pred, average="macro")
    print(f"\n== {tag} ==\nvalidation accuracy {acc:.4f}   macro-F1 {mf1:.4f}\n", flush=True)

    labels = sorted(set(ytr))
    missing = sorted(set(yva) - set(ytr))
    if missing:
        print(f"WARNING: {len(missing)} languages in validation are absent from train: {missing}")
    rep = classification_report(yva, pred, labels=labels, output_dict=True, zero_division=0)
    print(f"{'language':26} {'n':>7} {'prec':>6} {'rec':>6} {'f1':>6}")
    for l in sorted(labels, key=lambda x: rep[x]["f1-score"]):
        r = rep[l]
        print(f"{l:26} {int(r['support']):7d} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1-score']:6.3f}")

    # accuracy by utterance length -- the number that decides how much audio you need
    print("\n== accuracy vs first-K units (truncated validation) ==")
    length_curve = {}
    # characters for orthography, phoneme units for IPA -- roughly comparable durations
    ks = [10, 20, 40, 80, 160, 0] if args.analyzer == "char" else [5, 10, 20, 40, 80, 0]
    for k in ks:
        Xk = vec.transform([truncate(s, k, args.analyzer) for s in Xva_raw])
        pk = clf.predict(Xk)
        a = accuracy_score(yva, pk); m = f1_score(yva, pk, average="macro")
        length_curve[k or "full"] = {"acc": a, "macro_f1": m}
        unit = "chars" if args.analyzer == "char" else "units"
        print(f"  first {str(k) if k else 'all':>4} {unit}:  acc {a:.4f}  macroF1 {m:.4f}",
              flush=True)

    # where the errors live, by family
    fam_true = [LANG2FAM.get(l, "?") for l in yva]
    fam_pred = [LANG2FAM.get(l, "?") for l in pred]
    fam_acc = accuracy_score(fam_true, fam_pred)
    within = sum(1 for t, p, ft, fp in zip(yva, pred, fam_true, fam_pred) if t != p and ft == fp)
    errs = sum(1 for t, p in zip(yva, pred) if t != p)
    print(f"\n== family-level ==\n  family accuracy {fam_acc:.4f}")
    print(f"  of {errs} errors, {within} ({100*within/max(errs,1):.1f}%) stay inside the family")

    cm = confusion_matrix(yva, pred, labels=labels)
    np.fill_diagonal(cm, 0)
    pairs = [(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels)) if cm[i, j]]
    pairs.sort(key=lambda x: -x[2])
    print("\n  top confusions (true -> pred, count):")
    for a, b, c in pairs[:15]:
        same = "same-family" if LANG2FAM.get(a) == LANG2FAM.get(b) else ""
        print(f"    {a:26} -> {b:26} {c:5d}  {same}")

    import joblib
    joblib.dump({"vec": vec, "clf": clf, "labels": labels}, outdir / "model.joblib", compress=3)
    (outdir / "metrics.json").write_text(json.dumps({
        "tag": tag, "args": vars(args), "accuracy": acc, "macro_f1": mf1,
        "family_accuracy": fam_acc, "errors": errs, "errors_within_family": within,
        "length_curve": length_curve, "per_language": rep,
        "top_confusions": pairs[:50], "n_features": Xtr.shape[1],
    }, indent=2))
    size = (outdir / "model.joblib").stat().st_size / 1e6
    print(f"\nsaved {outdir}/model.joblib  ({size:.1f} MB compressed)")


if __name__ == "__main__":
    main()
