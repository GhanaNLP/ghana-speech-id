# ghana-speech-id

Language identification for 41 Ghanaian and West African languages.

Speech goes through
[Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr), which turns audio
into text; this library says which language the text is in.

```
audio ──[sherpa-onnx + omniASR CTC]──▶ transcript ──[this]──▶ language
```

CPU only, 8.2 MB, about 0.06 ms per classification. The inference core is C++ with a C API,
so there is no Python on the device.

```sh
pip install ghana-speech-id
```

**How the model behaves, what it scores and where it fails** is on the
[model card](https://huggingface.co/ghananlpcommunity/ghana-speech-id). This file is about
using it.

## Quick start

The head classifies text and cannot read audio, so it needs a recogniser in front of it.
Both come from the same repo:

```python
import soundfile as sf
import sherpa_onnx
from ghana_speech_id import GhanaSpeechId

model, tokens = GhanaSpeechId.download_recogniser()   # omniASR, 279 MB, once
rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(model=model, tokens=tokens)
lid = GhanaSpeechId.load()                            # the head, 8.2 MB

wav, sr = sf.read("clip.wav", dtype="float32")
s = rec.create_stream()
s.accept_waveform(sr, wav)
rec.decode_stream(s)

print(lid.classify(s.result.text))            # Ewe_ewe (0.50)
```

`load()` pulls only the head. The recogniser is downloaded when you ask for it and not
before, so a process that already has transcripts never fetches 279 MB it will not use.

There is one head and nothing to configure.

`classify()` returns `None` when no n-gram matched, meaning there was no basis for a
decision. Report that as unknown rather than naming whichever language scored least badly.

```python
p = lid.classify(text)
if p is None:
    print("unknown")
else:
    print(p.language, p.confidence, p.margin)   # margin = top-1 minus top-2
```

**The head is closed-set.** It always names one of its 41 languages, including for speech in
a language it has never seen. `margin` is the signal to threshold on if you need to reject
those — a weak one, so read the model card before relying on it.

## How much audio to give it

**Five seconds minimum, ten for the best result.** This matters more than anything else you
control:

| audio | accuracy |
|---|---|
| 3 s | 0.51 |
| 5 s | 0.66 |
| 7 s | 0.76 |
| 10 s | 0.78 |

Past about ten seconds the curve is flat. Check that most of the audio is speech before
transcribing: a recording that is half silence carries half the evidence its duration
suggests. The reference service uses
[silero VAD](https://github.com/snakers4/silero-vad) (0.6 MB, bundled with sherpa-onnx) and
rejects anything below 80% speech — `demo/modal_app.py` has the whole gate.

## Command line

```sh
ghana-speech-id "obiara na enyi nyɛden dɛ ɔbɔbɔ no nkenyan"
ghana-speech-id --top 3 < transcripts.txt
ghana-speech-id --file transcripts.txt --threads 4
```

## C

```c
GsidConfig cfg;
gsid_config_init(&cfg);
cfg.onnx_path   = "300m/head.onnx";
cfg.ngrams_path = "300m/ngrams.txt";
cfg.labels_path = "300m/labels.txt";
cfg.config_path = "300m/head_config.txt";

char err[512];
GsidHead *h = gsid_create(&cfg, err, sizeof err);

GsidResult r = gsid_classify(h, transcript);
if (r.index >= 0) printf("%s %.3f\n", gsid_language(h, r.index), r.confidence);
else              printf("unknown\n");

gsid_destroy(h);
```

`index == -1` is the same "no basis for a decision" case as `None` in Python.

## Android and iOS

`bindings/android` has the JNI shim, a Kotlin wrapper and a CMake file.
`bindings/ios` has a Swift wrapper and a module map, so the C API imports with no
Objective-C shim and no bridging header.

## Speed and footprint

CPU only, and deliberately so: the head is a vocabulary lookup and one sparse gather, so a
GPU would spend longer on transfers than on arithmetic. Both runtimes pin the CPU execution
provider.

| runtime | per classification | throughput | resident |
|---|---|---|---|
| **C++** (Xeon 8558) | **0.064 ms** | 15,700/s | **36 MB** |
| Python (Ryzen 5 4500U) | 0.09–0.20 ms | 5,500–10,000/s | 90 MB |

Roughly four orders of magnitude cheaper than the speech recognition in front of it. On any
device that can run the ASR, language identification is free.

## Building from source

```sh
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build -j
GSID_MODEL_DIR=model ./build/gsid_selftest
```

Only dependency is onnxruntime. The ONNX graph uses **opset-13 core operators only** — no
`com.microsoft` contrib ops — so it runs in mobile onnxruntime builds. The tf-idf arithmetic
is inside the graph:

```
inputs   indices int64[K], counts float32[K]
         tf = 1+log(counts) → ×idf → L2 normalise → Gather(W) → ReduceSum → +b → softmax
outputs  logits float32[C], probs float32[C]
```

The caller supplies n-gram indices and counts. Reproducing scikit-learn's `char_wb` exactly
is the delicate part, and it has two traps that fail silently rather than raising — see
[docs-char-tokenisation.md](docs-char-tokenisation.md). Every release is checked with
`scripts/cpp_parity.py`: sklearn, the Python package and the C++ CLI must agree on all 200
held-out transcripts.

## Demo

[The Space](https://huggingface.co/spaces/ghananlpcommunity/ghana-speech-id) takes an
uploaded file, a microphone recording, or a one-click sample in any of the 41 languages.
`demo/modal_app.py` is the whole service — VAD gate, recogniser and head — in one file.

## Training and evaluation

`scripts/` reproduces the model.

| script | what it does |
|---|---|
| `setup_lean.sh` | venv and dependencies |
| `pull_ipa.py` | corpus text without downloading the audio |
| `decode_base.py` | transcribe with a base omniASR model via sherpa-onnx |
| `decode_chunked.py` | transcribe fixed-length windows |
| `build_base_corpus.py` | assemble the training corpus |
| `train_head.py` | train and evaluate one configuration |
| `export_onnx.py` | export to ONNX, check parity against sklearn |
| `cpp_parity.py` | check the C++ and Python runtimes against the trainer |
| `ood_eval.py` | the out-of-domain evaluation |
| `eval_duration_curve.sh` | accuracy against real audio duration |
| `publish_hf.py` | publish to the Hub |

## Licence

Code Apache-2.0. Models and data follow the source corpora, CC BY-NC 4.0.
