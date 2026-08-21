---
license: cc-by-nc-4.0
pipeline_tag: audio-classification
tags:
  - language-identification
  - spoken-language-identification
  - onnx
  - sherpa-onnx
  - ipa
  - phonemes
  - ghana
  - africa
library_name: ghana-speech-id
datasets:
  - ghananlpcommunity/ghana-speech-ipa
  - ghananlpcommunity/ghana-speech-eval
  - ghanaopendata/ghana-english-tts-clean2
language:
  - twi
  - fat
  - any
  - sfw
  - nzi
  - ada
  - ewe
  - avn
  - lef
  - lip
  - snw
  - akp
  - bov
  - ntr
  - gjn
  - ncu
  - naw
  - acd
  - nko
  - dga
  - dag
  - maw
  - kus
  - bwu
  - kma
  - gur
  - biv
  - xon
  - bim
  - bud
  - xsm
  - sil
  - sig
  - vag
  - mzw
  - tpm
  - kdh
  - kbp
  - bib
  - hau
  - ffm
  - eng
---

# ghana-speech-id

Language identification for Ghanaian and West African speech, as a small classifier over
IPA phoneme strings.

It sits on top of [ghana-ipa-asr](https://huggingface.co/ghananlpcommunity/ghana-speech-phoneme-asr):
that model turns audio into IPA units, and this one says which language those units are.

```
audio ──[sherpa-onnx + ghana-speech-phoneme-asr]──▶ IPA units ──[this]──▶ language
```

> **Model artefacts are not uploaded yet.** This repository currently holds the
> configuration and card while the final head is selected. Code, training pipeline and
> evaluation live at https://github.com/GhanaNLP/ghana-speech-id

## Why phonemes and not audio

The obvious design is to pool the ASR encoder's hidden states and classify those. It was
rejected deliberately: the training corpus is Bible narration with roughly one narrator per
language, so a classifier fed continuous acoustic features learns *narrator identity*, which
correlates perfectly with the label and does not exist at inference. A phoneme string carries
no voice and no microphone, so that shortcut is unavailable by construction.

The cost is real. Collapsing 1024 floats per frame to ~10 symbols per second discards tone
(the 176-unit inventory has no tone marks, and most Kwa and Gur languages here are tonal),
rhythm, and the recogniser's uncertainty. What survives is phonotactics: length, nasalisation,
aspiration, labialisation, labiovelars, implosives, ejectives and prenasalised stops.

## Usage

```python
from ghana_ipa_asr import GhanaIPAASR
from ghana_speech_id import GhanaSpeechId

asr = GhanaIPAASR.load()
lid = GhanaSpeechId.load("ghananlpcommunity/ghana-speech-id")

ipa = asr.transcribe("clip.wav").spaced(punctuation=False)
print(lid.classify(ipa))     # Twi_twi (0.93)
```

```sh
pip install ghana-speech-id
ghana-ipa-asr transcribe clip.wav | ghana-speech-id
```

On device there is no Python: `sherpa-onnx` produces the IPA and the head runs in
onnxruntime through the C API, with Kotlin and Swift bindings in the GitHub repository.

## How much speech is needed

Accuracy saturates at about **40 phoneme units**, roughly 4 seconds.

| units | ≈ audio | accuracy |
|---|---|---|
| 10 | ~1 s | 76.0% |
| 20 | ~2 s | 89.9% |
| **40** | **~4 s** | **93.0%** |
| all | — | 94.7% |

Gate on the unit count rather than a timer: speech runs 8–13 units/s, so a fixed 4-second
window under-serves slow speakers and wastes time on fast ones.

## Evaluation

Held-out accuracy is measured on a **book-disjoint split** — the last 15% of each language
by id, which since the audio is scripture read in order approximates holding out whole books.
A random split scores 96.7% on the same configuration; that 2-point gap is passage-local
memorisation, which the contiguous split exists to avoid.

Almost every remaining error is within language family (Deg/Vagla, Chumburung/Nkonya,
Konkomba/Bassar), which is what a model that has learned real phonotactics looks like.

Asante and Akuapem Twi share ISO 639-3 `twi` and were ~12% of all errors, so they are one
class. `twi` is the only duplicated code among the languages, so the merge is exactly that
collapse. Fante is `fat` and separates cleanly.

## Limitations

* **Closed set.** The head always names one of its classes. Ga, Ahanta and Ikposo are not
  among them and will come back as a related in-set language, usually with unremarkable
  confidence. Threshold on the top-1/top-2 margin if you need to reject out-of-set speech.
* **No tone.** The phoneme inventory carries none, so tonal minimal pairs are invisible.
* **Domain.** Training audio is read scripture. Out-of-domain numbers on
  [ghana-speech-eval](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval)
  are pending and will be lower.
* **Code-switching** is common in Ghana and handled badly by a single-label classifier.

## Licence

Code Apache-2.0. Model weights and data CC BY-NC 4.0, following the source audio corpus.
