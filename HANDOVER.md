# Where this stands

Written at the end of an unattended session. Numbers here are measured, not estimated.

## The headline

The head was scoring 94.7% in-domain and **36.3% out of domain**. That gap traced to the
front-end, not the classifier: `ghana-ipa-asr` was fine-tuned on 2,329 h of Bible audio with
`freeze_encoder_for_n_steps: 0`, and lost the ability to read anything else. On identical JW
clips it produces 1.13 chars/s with 35% of clips empty, where the base model it was
fine-tuned *from* produces 8.36 chars/s with none.

Rebuilding the head on un-finetuned omniASR transcripts:

| domain | IPA head | base head |
|---|---|---|
| finance | 9.3% | **42.4%** |
| jw | 34.3% | **73.1%** |
| lds | 43.8% | **74.9%** |
| unicef | 35.8% | **91.7%** |
| waxal | 48.6% | **91.6%** |
| **overall** | **36.3%** | **76.6%** |

On a larger sample too — 13,963 scored clips against 9,890, because the base model
transcribes audio the fine-tune returned nothing for.

In-domain is 95.41% against 94.66%. Almost all the gain is in generalisation, which is
where the problem was.

## What to ship

`out/base300m_char_mf50000_noen` — 41 languages, char n-grams, 50k features.
95.28% in-domain, ~76% out of domain, roughly 8 MB fp32.

Feature count barely matters: 50k is within 0.13 points of 200k in every configuration
tried, so the mobile-sized head is effectively free.

## Also established

**Orthography beats IPA.** On ground-truth text the same recipe reaches 99.61% against the
IPA head's 94.66%, so the phoneme bottleneck was costing about five points. Base omniASR
emits orthography, and recognition errors give back roughly four of those five.

**Quantisation has to follow the device.** int8 on CUDA runs at 7x -- slower than CPU --
because quantised operators have no CUDA kernels and onnxruntime places them on CPU node by
node. fp32 on CUDA does 111x. int8 stays correct for on-device inference.

    　          CPU     CUDA
    int8        17x       7x
    fp32        13x     111x

**ghana-english-tts-clean2 is low-passed.** 92.7% of its energy sits below 1 kHz and 1.6% in
the 2-4 kHz band where consonants live, against 73.9% / 7.4% for ghana-speech. A real ASR
returns nothing for 82% of it. The muffling is in the source, not the speech-ipa derivative.
Fine for TTS; unusable for recognition.

## Open decisions

**Fante.** It collapses into Twi out of domain -- 0.118, 0.473 and 0.521 across three
configs -- yet reaches 0.982 F1 on clean text. ASR noise is erasing an Akan boundary the
head can otherwise learn. Merging Fante into Twi would lift the headline number and lose a
distinction that may matter to users. Your call.

**English.** Dropped, because its only training audio is the low-passed corpus above and its
recall is unmeasured -- ghana-speech-eval has no English config. Including it cost 0.2
points out of domain, so it is cheap rather than harmful. It can return from any full-band
Ghanaian English source; `africa-accents` on the H200 looked plausible. Adding it is about
twenty minutes now the pipeline exists: decode, `build_base_corpus.py --en`, retrain.

Those two are the only ones that need a decision. The rest below is work with an obvious
answer, or things simply worth knowing.

## Known, no decision needed

**Rejection is weak.** At 80% of in-set answers retained it rejects about half of
out-of-set speech. Better than the IPA head's 26.6%, not good enough to rely on. Fixing it
properly means an explicit "other" class or a background model. Out-of-set predictions are
at least coherent: Ga goes to Dangme, Ahanta to Nzema -- each language's nearest relative.

**The 1B is resolved, not open.** Built, published, characterised: it ties the 300M out of
domain, trails slightly in-domain, and wins by about two points below three seconds of
speech. sherpa-onnx ships it int8-only and int8 has no CUDA kernels, so bulk decoding needs
fairseq2 (283x) rather than sherpa (4-5x). On-device single-clip inference through sherpa
int8 is fine.

## Not done

Both runtimes now do character n-grams and are parity-checked, so that item is closed. The
Unicode folding question was solved by deriving `casefold.txt` from the vocabulary at export
time rather than depending on ICU, which means `--no-lowercase` was never needed and its
cost is still unmeasured.

`ipa_text.parquet` holds about 7,500 duplicate ids, so the *old* IPA head's numbers may
include a small train/test overlap. It does not affect anything above.

`decode_base.py` writes its output only at the end of a stage, so a crash loses the run.
`batch.py` in ghana-ipa-asr solves this with per-shard sidecars and would be worth copying.

## A finding for the ghana-ipa-asr repo

That model reports 16.64% UER on held-out Bible audio and cannot transcribe JW, UNICEF or
finance recordings at all. If it is meant as general-purpose Ghanaian phoneme ASR, the fix
is standard: freeze the encoder for the first N steps, mix in non-Bible audio, lower the
learning rate, or stop earlier. Whisper, which knows none of these languages, beat it on JW
and waxal audio.
