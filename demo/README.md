---
title: Ghana Speech ID
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Speak and find out which of 41 Ghanaian languages you spoke
---

# Ghana Speech ID

Speak into your microphone and this identifies which of 41 Ghanaian and West African
languages you spoke.

The page is static. Recognition runs on a CPU container that scales to zero, so nothing
is running while nobody is recording — the container is woken the moment you press record,
which usually covers the cold start before you have finished speaking.

- Model: [ghananlpcommunity/ghana-speech-id](https://huggingface.co/ghananlpcommunity/ghana-speech-id)
- Code: [GhanaNLP/ghana-speech-id](https://github.com/GhanaNLP/ghana-speech-id)

## How to get a good result

Speak a **full sentence**. Accuracy against how much speech the model gets, measured:
72% at about 0.8 seconds, 89% at 1.6, 95% at 3.3. A few words is not enough to separate
languages that are closely related.

The page enforces a three-second minimum, warns when the microphone level is too low, and
tells you when a sample was short enough that the answer is less reliable.

## Limitations

There is no English class, and the model always answers with one of its 41 languages —
Ga, Ahanta and Ikposo are not among them and come back as their nearest relative. Out of
domain it is right about 78% of the time, against 95% on audio like its training data.
