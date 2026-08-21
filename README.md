# ghana-speech-id

Language identification for Ghanaian and West African speech, as a small classifier over
IPA phoneme strings.

It sits on top of [ghana-ipa-asr](https://github.com/GhanaNLP/ghana-ipa-asr): that model
turns audio into IPA units, and this one says which language those units are.

```
audio ──[sherpa-onnx + ghana-speech-phoneme-asr]──▶ IPA units ──[this]──▶ language
```

The inference library is C++ with a C API. No Python on the device.

> **Status: work in progress.** Numbers below are in-domain (read scripture and broadcast
> audio). The out-of-domain evaluation on WaxalNLP is not finished, so treat these as an
> upper bound on real-world behaviour.

## Why phonemes and not audio

The obvious design is to pool the ASR encoder's hidden states and classify those. It was
rejected for one reason: the training corpus is Bible narration with roughly one narrator
per language, so a classifier fed continuous acoustic features learns *narrator identity*,
which correlates perfectly with the label and does not exist at inference. A phoneme string
carries no voice and no microphone, so that shortcut is unavailable by construction.

The cost is real and worth stating. Collapsing the encoder's 1024 floats per frame down to
~10 symbols per second discards:

* **tone** — the 176-unit inventory has no tone marks at all, and most Kwa and Gur
  languages here are tonal. This is the largest single loss.
* **timing and rhythm** — speech rate and syllable duration, which differ between families
* **uncertainty** — where the recogniser wavered between two phones

What survives is a rich phonotactic signal: length (`aː iː`), nasalisation (`ã ɛ̃`),
aspiration (`kʰ tʰ`), labialisation (`nʷ kʷ`), labiovelars (`k͡p ɡ͡b ŋ͡m`), implosives
(`ɓ ɗ`), ejectives (`kʼ`) and prenasalised stops (`ⁿd ᵐb ᵑɡ`).

## Results

Held-out accuracy over 41 classes, **book-disjoint split** — the last 15% of each language
by id, which since the audio is scripture read in order approximates holding out whole
books.

| configuration | accuracy | macro-F1 |
|---|---|---|
| 1–5 grams, 200k features, punctuation kept | 94.19% | 0.945 |
| 1–5 grams, 200k features, punctuation dropped | **94.66%** | **0.949** |
| 1–5 grams, 50k features, punctuation dropped | 94.40% | 0.947 |
| *same, random split (control, not a result)* | *96.67%* | *0.970* |

Dropping punctuation helps: it is only ~62% accurate upstream and carries little language
signal.

**The random-split row is a control.** The 2-point gap over the book-disjoint split is how
much a naive evaluation would have overstated things by scoring on verses adjacent to the
training passages.

### Accuracy against clip length

Measured by truncating validation strings, so it says how much speech is needed before a
decision is worth making.

| units | ≈ audio | accuracy |
|---|---|---|
| 5 | ~0.5 s | 53.1% |
| 10 | ~1 s | 76.0% |
| 20 | ~2 s | 89.9% |
| **40** | **~4 s** | **93.0%** |
| 80 | ~8 s | 93.3% |
| all | — | 93.4% |

**Gate on the phoneme count, not a timer.** The knee is at ~40 units; the seconds column is
derived at ~10 units/s and real speech runs 8–13/s, so a fixed 4-second timer under-serves
slow speakers and wastes time on fast ones.

### Where the errors are

Almost every confusion is within language family, which is what a model that has learned
real phonotactics looks like rather than one keying on artefacts.

```
Asante_Twi   → Akuapem_Twi   same-language dialects — now merged to Twi_twi
Deg          → Vagla         both Grusi
Chumburung   → Nkonya        both Guang
Konkomba     → Bassar        both Gurma
```

Asante and Akuapem Twi share ISO 639-3 `twi` and were ~12% of all errors, so they are
merged into one class. `twi` is the only duplicated ISO code across the 42, so the merge is
exactly that collapse and nothing else. Fante is `fat` and separates cleanly, so it stays.

### Model size

The ONNX weight matrix is `features × classes × 4` bytes, so `max_features` is the knob
that decides whether the head fits in an app.

| features | accuracy | head.onnx | fp16 | ngrams.txt |
|---|---|---|---|---|
| 50k | 94.40% | 8.2 MB | ~4.1 MB | ~0.5 MB |
| 200k | 94.66% | 33.6 MB | ~16.8 MB | 2.0 MB |

## Using the library

### C

```c
#include "ghana_speech_id.h"

GsidConfig cfg;
gsid_config_init(&cfg);
cfg.onnx_path   = "model/head.onnx";
cfg.ngrams_path = "model/ngrams.txt";
cfg.labels_path = "model/labels.txt";
cfg.config_path = "model/head_config.txt";

char err[512];
GsidHead *h = gsid_create(&cfg, err, sizeof err);

GsidResult r = gsid_classify(h, "n a e s o m e b a ɾ ɪ m ɔ");
if (r.index >= 0) printf("%s %.3f\n", gsid_language(h, r.index), r.confidence);
else              printf("unknown\n");

gsid_destroy(h);
```

`index == -1` means no n-gram matched, so there was no basis for a decision. Report it as
unknown rather than naming whichever language scored least badly.

### Command line

```sh
gsid --model-dir model "n a e s o m e b a ɾ ɪ m ɔ"
ghana-ipa-asr transcribe clip.wav | gsid --model-dir model --top 3
```

### Android and iOS

`bindings/android` has the JNI shim, a Kotlin wrapper and a CMake file to point
`externalNativeBuild` at. `bindings/ios` has a Swift wrapper and a module map, so the C API
imports directly with no Objective-C shim and no bridging header.

## Building

```sh
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build -j
GSID_MODEL_DIR=model ./build/gsid_selftest
```

Only dependency is onnxruntime. The ONNX graph uses **opset-13 core operators only** — no
`com.microsoft` contrib ops — so it runs in mobile onnxruntime builds. `skl2onnx` would
have emitted a contrib `Tokenizer` node, so the tf-idf arithmetic is built into the graph
by hand instead:

```
inputs   indices int64[K], counts float32[K]
         tf = 1+log(counts) → ×idf → L2 normalise → Gather(W) → ReduceSum → +b → softmax
outputs  logits float32[C], probs float32[C]
```

The caller only has to split on whitespace, emit 1–5-grams and look them up in
`ngrams.txt`. Units are atomic — never split `k͡p`, `kʰ` or `t͡ʃ` on characters.

## Training

See `scripts/`. In order:

| script | what it does |
|---|---|
| `setup_lean.sh` | venv and dependencies; no torch, no fairseq2, no audio |
| `pull_ipa.py` | pulls the IPA text columns from ghana-speech-ipa without the 50 GB of audio |
| `rephonemise_english.py` | re-decodes Ghanaian English through ghana-ipa-asr |
| `add_english.py` | folds English in as a class, size- and length-matched |
| `train_head.py` | trains and evaluates one configuration |
| `run_sweep.sh` | the configuration sweep and its summary table |
| `export_onnx.py` | exports to ONNX and checks parity against sklearn |
| `cpp_parity.py` | checks the C++ CLI against the Python model |
| `ood_waxal.py` | the WaxalNLP out-of-domain evaluation |

Everything is verified end to end before shipping: ONNX matches sklearn 300/300 on argmax
(max logit difference 6.3e-06), and the C++ CLI matches the Python model 300/300.

## Licence

Code Apache-2.0. Models and data follow the source corpora, CC BY-NC 4.0.
