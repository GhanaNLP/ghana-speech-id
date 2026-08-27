# ghana-speech-id

Language identification for 41 Ghanaian and West African languages, as a small classifier
over speech transcripts.

It sits on top of [Omnilingual ASR](https://huggingface.co/facebook/omnilingual-asr): that
model turns audio into text, and this one says which language the text is in.

```
audio ──[sherpa-onnx + omniASR CTC]──▶ transcript ──[this]──▶ language
```

The inference library is C++ with a C API. No Python on the device.

```sh
pip install ghana-speech-id
```

## Results

Two evaluations, and the gap between them is the point.

**In-domain** is held-out audio from the training corpus, split so that the last 15% of each
language by id is held back — the audio is scripture read in order, so that approximates
holding out whole books.

**Out-of-domain** is
[ghana-speech-eval](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval),
skipping its `bible_*` configs because those are the training domain. Five unrelated
domains, 13,963 scored clips.

| variant | features | in-domain | out-of-domain | size |
|---|---|---|---|---|
| **300m** | 50k | 95.30% | **77.6%** | **8.2 MB** |
| 300m | 200k | **95.44%** | 77.7% | 32.8 MB |
| 1b | 50k | 95.15% | 77.4% | 8.2 MB |
| 1b | 200k | 95.31% | 77.8% | 32.8 MB |

Out-of-domain by domain, for the shipped 300m/50k head:

| finance | jw | lds | unicef | waxal |
|---|---|---|---|---|
| 42% | 76% | 81% | 92% | 89% |

## Which variant

**300m unless your utterances are very short.** The two are level from about three seconds
of speech onward, and the 300m is a third the size and the only one sherpa-onnx can decode
with at a useful rate. But the 1b is meaningfully better on very short input:

| input | ≈ audio | 300m | 1b |
|---|---|---|---|
| 10 chars | ~0.8 s | 72.1% | **74.2%** |
| 40 chars | ~3.3 s | 94.6% | 94.6% |
| full | — | 95.3% | 95.2% |

## How much audio to give it

**Accuracy rises steeply with the amount of speech**, and the effect is larger than any
other choice in this system. Measured out of domain on real audio — not on truncated
transcripts, which flatter short input by about six points because the recogniser still had
the whole clip:

| audio | accuracy |
|---|---|
| 3 seconds | 0.51 |
| whole clips (mean 9.7 s) | **0.78** |

Short audio does not merely produce less text, it produces worse text: three seconds yields
a mean of 19.7 characters at 4.6% empty, against 80 characters at 0.9% for whole clips.

So: **ask for at least five seconds, and check that most of it is speech.** A recording that
is half silence carries half the evidence its duration suggests. The reference service runs
[silero VAD](https://github.com/snakers4/silero-vad) (0.6 MB, bundled with sherpa-onnx) and
rejects anything below 80% speech before transcribing.

## Speed and footprint

CPU only. There is no GPU path in the inference library and none is wanted: the head is a
vocabulary lookup and one sparse gather, so a GPU would spend more time on transfers than on
arithmetic. Both runtimes pin the CPU execution provider.

Single thread, measured end to end including tokenisation:

| runtime | per classification | throughput | resident |
|---|---|---|---|
| **C++** (Xeon 8558) | **0.064 ms** | 15,700/s | **36 MB** |
| Python (Ryzen 5 4500U) | 0.09–0.20 ms | 5,500–10,000/s | 90 MB |

The head is roughly four orders of magnitude cheaper than the speech recognition in front of
it. On any device that can run the ASR, language identification is free.

## Using it

### Python

```python
import sherpa_onnx
from ghana_speech_id import GhanaSpeechId

rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
    model="omniasr-300m/model.int8.onnx", tokens="omniasr-300m/tokens.txt")
lid = GhanaSpeechId.load()                    # variant="300m" by default

s = rec.create_stream()
s.accept_waveform(16000, wav)
rec.decode_stream(s)

print(lid.classify(s.result.text))            # Twi_twi (0.93)
```

### C

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
```

`index == -1` means no n-gram matched, so there was no basis for a decision. Report it as
unknown rather than naming whichever language scored least badly.

### Command line

```sh
ghana-speech-id "obiara na enyi nyɛden dɛ ɔbɔbɔ no nkenyan"
ghana-speech-id --variant 1b --top 3 < transcripts.txt
```

### Android and iOS

`bindings/android` has the JNI shim, a Kotlin wrapper and a CMake file.
`bindings/ios` has a Swift wrapper and a module map, so the C API imports with no
Objective-C shim and no bridging header.

## Why transcripts and not phonemes

An earlier version classified IPA from a phoneme recogniser fine-tuned on this corpus. Two
measurements changed the design.

**Orthography beats IPA by five points.** On ground-truth text the same recipe reaches
99.61% in-domain against the IPA head's 94.66% at equal model size. Spelling conventions,
function words and morphology carry language identity that phonemes discard — the 176-unit
inventory has no tone marks at all, and most of these languages are tonal.

**The fine-tuned front end had stopped generalising.** It was trained on 2,329 h of Bible
audio with the encoder unfrozen from step 0, and lost the ability to read anything else. On
identical JW recordings it produces 1.13 characters per second with 35% of clips empty,
where the base model it was fine-tuned *from* produces 8.36 and none. Rebuilding on the base
model took out-of-domain accuracy from **36.3% to 77.6%**.

## Why short training windows

The head is trained on 40-character windows with stride 20 — about 3.3 seconds of speech —
rather than whole transcripts. Training on whole clips and deploying on short utterances is
a mismatch: the model saw a mean of 87 characters and has to decide from 25–50.

Worth +1.1 points out of domain, and much more where it matters:

| input | ≈ audio | whole-trained | window-trained |
|---|---|---|---|
| 10 chars | ~0.8 s | 67.4% | **72.1%** |
| 20 chars | ~1.6 s | 86.4% | **89.4%** |
| full | — | 95.4% | 95.4% |

Inference classifies the whole transcript in one pass. Voting across windows was
implemented and measured: −0.6 out of domain, −0.09 in-domain, and it compresses the margins
that out-of-set rejection depends on. The code remains, defaulted off.

## Front ends that were tried and rejected

Six alternatives to omniASR orthography were measured on the same evaluation. All lost, and
the pattern is consistent enough to be worth stating: **discrete symbols carry more usable
language identity than pooled acoustic vectors, and the gap widens as audio gets shorter.**

| front end | why it lost |
|---|---|
| ZIPA phones (Zipformer CTC) | 0.34 against 0.57 at 1.6 s. A universal phone inventory is *designed* to be language-invariant, which is the information language ID needs |
| MMS-LID-4017 zero shot | 0.27 at 1.6 s. Collapses on short audio far worse than text does |
| MMS features + our own head | ~0.22 at 1.6 s. The representation degrades, not just its classifier |
| omniASR encoder embeddings | **0.976 in-domain, 0.108 out of domain.** Learned narrators, not languages — the corpus has roughly one voice per language, which is why this project classifies text at all |
| wav2vec2-XLSR eSpeak | eSpeak's inventory emits English `ɹ` and no labiovelars, so it flattens distinctions these languages depend on |
| Qwen3-ASR encoder | covers ~11 languages, none African |

Audio chunking — training on 3 s windows to match inference — was also measured and came out
**neutral** once training-data volume was controlled for.

## Limitations

**Closed set.** The head always names one of its 41 classes. Ga, Ahanta and Ikposo are not
among them and come back as their nearest relative — Ga as Dangme, Ahanta as Nzema. The
top-1/top-2 margin gives a rejection signal, but a weak one: at 80% of in-set answers
retained it rejects about half of out-of-set speech.

**Fante collapses into Twi out of domain**, scoring 0.12–0.52 across three configs despite
0.98 F1 on clean text. ASR noise erases an Akan boundary the head can otherwise learn.

**No English class.** The only Ghanaian English corpus available is low-passed — 93% of its
energy below 1 kHz and 1.6% in the 2–4 kHz band where consonants live — and a real ASR
returns nothing for 82% of it. English can be added from any full-band source.

**Domain still matters.** 95% in-domain against 78% out of domain. Finance recordings are
the weakest at 42%.

## Building

```sh
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build -j
GSID_MODEL_DIR=model ./build/gsid_selftest
```

Only dependency is onnxruntime. The ONNX graph uses **opset-13 core operators only** — no
`com.microsoft` contrib ops — so it runs in mobile onnxruntime builds. The tf-idf arithmetic
is built into the graph:

```
inputs   indices int64[K], counts float32[K]
         tf = 1+log(counts) → ×idf → L2 normalise → Gather(W) → ReduceSum → +b → softmax
outputs  logits float32[C], probs float32[C]
```

The caller supplies n-gram indices and counts. Reproducing scikit-learn's `char_wb` exactly
is the delicate part and has two traps that fail silently rather than raising — see
[docs-char-tokenisation.md](docs-char-tokenisation.md). Every release is checked with
`scripts/cpp_parity.py`: sklearn, the Python package and the C++ CLI must agree on every
one of 200 held-out transcripts.

## Training

See `scripts/`, and [HANDOVER.md](HANDOVER.md) for what is settled and what is open.

| script | what it does |
|---|---|
| `setup_lean.sh` | venv and dependencies |
| `pull_ipa.py` | corpus text without downloading the audio |
| `decode_base.py` | transcribe with a base omniASR model via sherpa-onnx |
| `decode_chunked.py` | transcribe fixed-length windows, for the chunking experiment |
| `build_base_corpus.py` | assemble the training corpus |
| `train_head.py` | train and evaluate one configuration |
| `export_onnx.py` | export to ONNX, check parity against sklearn |
| `cpp_parity.py` | check the C++ and Python runtimes against the trainer |
| `ood_eval.py` | the out-of-domain evaluation |
| `eval_duration_curve.sh` | accuracy against real audio duration |
| `select_variants.py` | pick which variants ship, on measured accuracy |
| `publish_hf.py` | publish both variants to the Hub |

Quantisation has to follow the device, and getting it wrong is expensive:

| | CPU | CUDA |
|---|---|---|
| int8 | 17× | 7× |
| fp32 | 13× | **111×** |

int8 on CUDA is slower than not using the GPU at all — quantised operators have no CUDA
kernels, so onnxruntime places them on CPU node by node. int8 is still right for on-device
inference.

## Licence

Code Apache-2.0. Models and data follow the source corpora, CC BY-NC 4.0.
