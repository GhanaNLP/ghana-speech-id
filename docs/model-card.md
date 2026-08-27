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

**One head ships**, in `300m/`, built on the omniASR CTC 300M front end. `GhanaSpeechId.load()`
finds it with no argument — there is nothing to choose.

| | in-domain | out-of-domain | very short input (~0.8 s) | size |
|---|---|---|---|---|
| **`300m/`** (ships) | **95.30%** | **77.6%** | 72.1% | 8.2 MB |
| `1b/` | 95.15% | 77.4% | **74.2%** | 8.2 MB |

A 1B-front-end variant was built and measured and is kept in `1b/` for reference, but it is
not the default and the library no longer offers the choice. It ties out of domain, loses
in domain, and its one advantage — input under a second — is unreachable now that the
guidance is five seconds minimum. It also needs a front end three times the size that
sherpa-onnx cannot decode at a useful rate.

## How much audio to give it

**Five seconds minimum, ten for the best result.** This matters more than the choice of
variant, the feature count, or anything else you can tune. Measured on
[ghana-speech-eval](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval)
without its `bible_*` configs — the out-of-domain set described under Evaluation below — on
real audio of each length, not on truncated transcripts, which flatter short input by about
six points because the recogniser still had the whole clip:

| audio | accuracy | clips scored | mean characters | empty transcripts |
|---|---|---|---|---|
| 3 s | 0.506 | 57,840 | 19.7 | 4.6% |
| **5 s** | **0.657** | 32,816 | 33.4 | 0.7% |
| **7 s** | **0.759** | 20,439 | 47.7 | 0.1% |
| whole clips (mean 9.7 s) | **0.777** | 13,963 | 80.0 | 0.9% |

Five seconds is a floor, not a target — it buys 0.66, well short of what the model can do.
The curve then flattens: seven seconds is within two points of whole clips averaging 9.7 s,
so **ten seconds is where the returns run out**.

Short audio does not simply produce less text, it produces worse text. Three seconds yields
19.7 characters on average and 4.6% of clips come back empty, against 80 characters and 0.9%
for whole clips — the recogniser needs context and starves without it.

**Check that most of it is speech.** A recording that is half silence carries half the
evidence its duration suggests. The reference service runs
[silero VAD](https://github.com/snakers4/silero-vad) and rejects anything below 80% speech
before transcribing.

## Usage

```python
import sherpa_onnx
from ghana_speech_id import GhanaSpeechId

rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
    model="omniasr-300m/model.int8.onnx", tokens="omniasr-300m/tokens.txt")
lid = GhanaSpeechId.load("ghananlpcommunity/ghana-speech-id")

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
