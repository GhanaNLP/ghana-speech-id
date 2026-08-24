---
license: cc-by-nc-4.0
pipeline_tag: audio-classification
tags:
  - language-identification
  - spoken-language-identification
  - onnx
  - sherpa-onnx
  - orthography
  - omnilingual-asr
  - ghana
  - africa
library_name: ghana-speech-id
datasets:
  - ghananlpcommunity/ghana-speech
  - ghananlpcommunity/ghana-speech-eval
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
---

# ghana-speech-id

Language identification for 41 Ghanaian and West African languages, over speech transcripts.

It sits on top of [Omnilingual ASR](https://huggingface.co/facebook/omnilingual-asr): that
model turns audio into text, and this one says which language the text is in.

```
audio ──[sherpa-onnx + omniASR CTC]──▶ transcript ──[this]──▶ language
```

Two variants live in this repository, differing only in which front end produced the
transcripts they were trained on.

| variant | front end | in-domain | out-of-domain | size |
|---|---|---|---|---|
| **`300m/`** | omniASR CTC 300M | 95.30% | **77.6%** | 8.2 MB |
| `1b/` | omniASR CTC 1B v2 | 95.15% | 77.4% | 8.2 MB |

**Use `300m` unless your utterances are very short.** The two are level from about three
seconds of speech onward, and the 300M front end is a third the size and the only one
sherpa-onnx decodes at a useful rate. Below that the 1B is better: at ~0.8 s of speech it
scores 74.2% against 72.1%.

## Usage

```python
import sherpa_onnx
from ghana_speech_id import GhanaSpeechId

rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
    model="omniasr-300m/model.int8.onnx", tokens="omniasr-300m/tokens.txt")
lid = GhanaSpeechId.load("ghananlpcommunity/ghana-speech-id")   # variant="300m"

s = rec.create_stream()
s.accept_waveform(16000, wav)
rec.decode_stream(s)
print(lid.classify(s.result.text))
```

On device there is no Python: sherpa-onnx produces the transcript and the head runs in
onnxruntime through a C API, with Kotlin and Swift bindings. See
[the repository](https://github.com/GhanaNLP/ghana-speech-id).

## Evaluation

**In-domain** holds out the last 15% of each language by id. The audio is scripture read in
order, so a contiguous tail approximates holding out whole books; a random split scores
about two points higher, and that gap is passage-local memorisation.

**Out-of-domain** is
[ghana-speech-eval](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval)
without its `bible_*` configs, since those are the training domain. Five unrelated domains,
13,963 scored clips.

| finance | jw | lds | unicef | waxal |
|---|---|---|---|---|
| 42% | 76% | 81% | 92% | 89% |

## How it is built

A linear classifier over character n-grams of the transcript, with tf-idf folded into the
ONNX graph. Trained on 40-character windows (about 3.3 s) rather than whole transcripts,
which is worth +1.1 points out of domain and +4.7 at one second of speech. Inference
classifies the whole transcript in one pass.

An earlier version used IPA phonemes from a recogniser fine-tuned on this corpus. Two
measurements changed it: orthography beats IPA by five points at equal size, and the
fine-tuned front end had lost the ability to read audio outside its training domain —
1.13 characters per second with 35% of clips empty where the base model manages 8.36 and
none. Rebuilding on the base model moved out-of-domain accuracy from 36.3% to 77.6%.

## Limitations

**Closed set.** The head always names one of its 41 classes. Ga, Ahanta and Ikposo are not
among them and return their nearest relative — Ga as Dangme, Ahanta as Nzema. The top-1 minus
top-2 margin gives a rejection signal, but a weak one: at 80% of in-set answers retained it
rejects about half of out-of-set speech.

**Fante collapses into Twi out of domain**, 0.12–0.52 across three configs, despite 0.98 F1
on clean text. Recognition noise erases an Akan boundary the head can otherwise learn.

**No English class.** The available Ghanaian English corpus is low-passed — 93% of its
energy below 1 kHz — and a real ASR returns nothing for 82% of it.

**Domain still matters.** 95% in-domain against 78% out of domain, and finance recordings
are the weakest at 42%.

## Licence

Code Apache-2.0. Model weights and data CC BY-NC 4.0, following the source corpora.
