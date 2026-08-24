"""Language identification over IPA phoneme strings.

Pure Python on purpose. The head is a vocabulary lookup and one sparse gather, so there is
nothing worth compiling, and a universal wheel installs everywhere without a build matrix.
The C++ library in this repository exists for mobile, where linking beats interpreting.

Feature extraction must match the trainer exactly or the app silently loses accuracy, so
this mirrors sklearn's TfidfVectorizer(analyzer="word", token_pattern=r"\\S+",
lowercase=False, sublinear_tf=True, norm="l2") -- and the tf-idf arithmetic itself lives
inside the ONNX graph, so what happens here is only tokenisation, n-grams and counting.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_REPO = "ghananlpcommunity/ghana-speech-id"
DEFAULT_VARIANT = "300m"


@dataclass(frozen=True)
class Prediction:
    """One classification result."""

    language: str
    """Predicted label, e.g. ``"Twi_twi"``."""
    confidence: float
    """Model score. Calibrated only for a logistic-regression head; for an SVM head prefer
    :attr:`margin`."""
    margin: float
    """Top-1 minus top-2. A usable confidence signal regardless of head type."""
    matched_ngrams: int
    """How many n-grams were found in the vocabulary."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.language} ({self.confidence:.3f})"


class GhanaSpeechId:
    """Classify a speech transcript into one of 41 Ghanaian and West African languages.

    The transcript comes from a base omniASR CTC model, which sherpa-onnx can run on
    device::

        import sherpa_onnx
        from ghana_speech_id import GhanaSpeechId

        rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model="omniasr-300m/model.int8.onnx", tokens="omniasr-300m/tokens.txt")
        lid = GhanaSpeechId.load()          # variant="300m" by default

        s = rec.create_stream(); s.accept_waveform(16000, wav); rec.decode_stream(s)
        print(lid.classify(s.result.text))

    Pass the recogniser's output unmodified.
    """

    def __init__(
        self,
        onnx_path: str | os.PathLike,
        ngrams_path: str | os.PathLike,
        labels_path: str | os.PathLike,
        config_path: str | os.PathLike | None = None,
        num_threads: int = 1,
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        self._in = [i.name for i in self._sess.get_inputs()]
        self._out = [o.name for o in self._sess.get_outputs()]

        text = Path(ngrams_path).read_text(encoding="utf-8").split("\n")
        if text and text[-1] == "":
            text.pop()
        self._vocab = {g: i for i, g in enumerate(text)}

        self.languages: list[str] = [
            l for l in Path(labels_path).read_text(encoding="utf-8").split("\n") if l
        ]

        self.ngram_min, self.ngram_max = 1, 5
        self.analyzer = "word"
        self.lowercase = True
        if config_path and Path(config_path).exists():
            for line in Path(config_path).read_text(encoding="utf-8").splitlines():
                key, _, val = line.partition(" ")
                if key == "ngram_min":
                    self.ngram_min = int(val)
                elif key == "ngram_max":
                    self.ngram_max = int(val)
                elif key == "analyzer":
                    self.analyzer = val.strip()
                elif key == "lowercase":
                    self.lowercase = val.strip() not in ("0", "false", "False")

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(
        cls,
        model: str | os.PathLike = DEFAULT_REPO,
        *,
        variant: str = DEFAULT_VARIANT,
        fp16: bool = False,
        num_threads: int = 1,
    ) -> "GhanaSpeechId":
        """Load from a local directory or pull one from the Hugging Face Hub.

        :param model: a directory holding ``head.onnx``/``head.fp16.onnx``, ``ngrams.txt``,
            ``labels.txt`` and ``head_config.txt``, or a Hub repo id.
        :param variant: which front-end the head was built on -- ``"300m"`` or ``"1b"``.
            Both live in the same Hub repo. 300m is the default: smaller, faster on device,
            and within a point of the 1b in accuracy.
        :param fp16: prefer the half-precision head, which is half the size on disk.
        """
        path = Path(model)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(
                    str(model),
                    allow_patterns=["*.onnx", "ngrams.txt", "labels.txt", "head_config.txt"],
                )
            )

        # variant subdirectory in the Hub repo; a local export may sit at the root or
        # under onnx/
        for cand in (path / variant, path, path / "onnx"):
            if (cand / "ngrams.txt").exists():
                root = cand
                break
        else:
            raise FileNotFoundError(
                f"no ngrams.txt under {path}/{variant}, {path} or {path}/onnx")
        names = ["head.fp16.onnx", "head.onnx"] if fp16 else ["head.onnx", "head.fp16.onnx"]
        onnx = next((root / n for n in names if (root / n).exists()), None)
        if onnx is None:
            raise FileNotFoundError(f"no head.onnx or head.fp16.onnx under {root}")
        for required in ("ngrams.txt", "labels.txt"):
            if not (root / required).exists():
                raise FileNotFoundError(f"missing {required} under {root}")

        cfg = root / "head_config.txt"
        return cls(onnx, root / "ngrams.txt", root / "labels.txt",
                   cfg if cfg.exists() else None, num_threads)

    # ------------------------------------------------------------------ features

    def _featurise(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        if self.analyzer == "char":
            grams = self._char_wb_grams(text)
        else:
            grams = self._unit_grams(text)
        if not grams:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        keys = np.fromiter(grams.keys(), np.int64, len(grams))
        vals = np.fromiter(grams.values(), np.float32, len(grams))
        return keys, vals

    def _unit_grams(self, text: str) -> "Counter[int]":
        """Whitespace tokens ARE the phonemes for an IPA head, several of them
        multi-character (kʰ, k͡p, t͡ʃ), so they must never be split on characters."""
        units = text.split()
        vocab = self._vocab
        hits: Counter[int] = Counter()
        for i in range(len(units)):
            gram = ""
            for n in range(1, self.ngram_max + 1):
                j = i + n - 1
                if j >= len(units):
                    break
                gram = units[j] if n == 1 else gram + " " + units[j]
                if n < self.ngram_min:
                    continue
                idx = vocab.get(gram)
                if idx is not None:
                    hits[idx] += 1
        return hits

    def _char_wb_grams(self, text: str) -> "Counter[int]":
        """Reproduce sklearn's analyzer="char_wb" exactly.

        Each whitespace-delimited word is padded with one space either side and n-grams are
        taken within that padded word only -- they never cross a word boundary. Iteration is
        over Python characters, which are codepoints: 'kɔ' is two characters and three
        bytes, and a byte-wise loop would split ɔ and generate n-grams that exist nowhere in
        the vocabulary, silently costing accuracy rather than raising.
        """
        vocab = self._vocab
        hits: Counter[int] = Counter()
        src = text.lower() if self.lowercase else text
        for word in src.split():
            padded = f" {word} "
            L = len(padded)
            for n in range(self.ngram_min, self.ngram_max + 1):
                if n > L:
                    break
                for i in range(L - n + 1):
                    idx = vocab.get(padded[i:i + n])
                    if idx is not None:
                        hits[idx] += 1
        return hits

    # ------------------------------------------------------------------ inference

    def posterior(self, ipa: str) -> dict[str, float] | None:
        """Full distribution over languages, or ``None`` when no n-gram matched."""
        idx, cnt = self._featurise(ipa)
        if idx.size == 0:
            return None
        probs = self._sess.run([self._out[1]], {self._in[0]: idx, self._in[1]: cnt})[0]
        return dict(zip(self.languages, (float(p) for p in probs)))

    def classify(self, ipa: str) -> Prediction | None:
        """Best guess, or ``None`` when there is no basis for one.

        ``None`` means every n-gram was out of vocabulary -- report it as unknown rather
        than naming whichever language scored least badly. Note that this is *not* a
        general "not confident" signal: the head is closed-set and will name an in-set
        language for speech in a language it was never trained on. Threshold on
        :attr:`Prediction.margin` if you need to reject those.
        """
        idx, cnt = self._featurise(ipa)
        if idx.size == 0:
            return None
        probs = self._sess.run([self._out[1]], {self._in[0]: idx, self._in[1]: cnt})[0]
        order = np.argsort(probs)[::-1]
        best = int(order[0])
        second = float(probs[order[1]]) if probs.size > 1 else 0.0
        return Prediction(
            language=self.languages[best],
            confidence=float(probs[best]),
            margin=float(probs[best]) - second,
            matched_ngrams=int(cnt.sum()),
        )

    def classify_batch(self, ipas: Sequence[str]) -> list[Prediction | None]:
        """Convenience loop. The head costs microseconds, so there is no batching win."""
        return [self.classify(s) for s in ipas]

    def __len__(self) -> int:
        return len(self.languages)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"GhanaSpeechId({len(self.languages)} languages, "
                f"{len(self._vocab)} n-grams, {self.ngram_min}-{self.ngram_max} grams)")
