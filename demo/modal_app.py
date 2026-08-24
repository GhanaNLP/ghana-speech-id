"""Language identification service: audio in, language out, on Modal CPU.

CPU only. The recogniser is int8, which is the right choice off a GPU -- int8 runs at 17x
realtime on CPU against fp32's 13x, and the ordering inverts on CUDA. The language head
itself costs about 0.06 ms, four orders of magnitude less than the recognition, so the whole
request is dominated by the ASR.

Cost control is the reason for the /warm endpoint. Containers scale to zero, so the first
request after an idle period pays a cold start of several seconds. The browser calls /warm
the moment the user presses record, and by the time they have spoken the four seconds the
model needs, the container is up. Nothing runs while nobody is recording.
"""
from __future__ import annotations

import io
import json
import time

import modal

MODELS = "/models"
ASR_DIR = f"{MODELS}/omniasr-300m"
HEAD_DIR = f"{MODELS}/head"

# Both models are baked into the image rather than downloaded at run time: a cold start
# should be container boot plus model load, not a 350 MB download.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "bzip2")          # not present in debian_slim
    .pip_install("sherpa-onnx==1.13.6", "ghana-speech-id==0.1.1", "fastapi[standard]",
                 "soundfile>=0.12", "numpy<2", "huggingface_hub>=0.30")
    .run_commands(
        f"mkdir -p {MODELS}",
        # omniASR CTC 300M int8 -- the front end
        f"cd {MODELS} && curl -sL -o a.tar.bz2 "
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2 "
        f"&& tar xjf a.tar.bz2 && rm a.tar.bz2 "
        f"&& mv sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12 {ASR_DIR}",
    )
    .run_commands(
        f"python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('ghananlpcommunity/ghana-speech-id', "
        f"local_dir='{HEAD_DIR}', allow_patterns=['*.onnx','*.txt','config.json'])\"",
    )
)

app = modal.App("ghana-speech-id", image=image)

# How much speech the head needs, measured: 72% at ~0.8 s, 89% at ~1.6 s, 95% at ~3.3 s.
MIN_SECONDS = 3.0
MAX_SECONDS = 20.0


@app.cls(
    cpu=2,
    # Stay up briefly between requests so a user recording several clips pays one cold
    # start, but never idle for long: nothing should run while nobody is recording.
    scaledown_window=120,
    min_containers=0,
)
@modal.concurrent(max_inputs=4)
class Identifier:
    @modal.enter()
    def load(self):
        import sherpa_onnx
        from ghana_speech_id import GhanaSpeechId

        t0 = time.time()
        self.rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=f"{ASR_DIR}/model.int8.onnx",
            tokens=f"{ASR_DIR}/tokens.txt",
            num_threads=2,
        )
        self.lid = GhanaSpeechId.load(HEAD_DIR, num_threads=2)
        print(f"loaded in {time.time() - t0:.1f}s, {len(self.lid.languages)} languages")

    def _run(self, wav_bytes: bytes) -> dict:
        import numpy as np
        import soundfile as sf

        t0 = time.time()
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        seconds = len(audio) / sr
        if seconds > MAX_SECONDS:                 # trim rather than reject
            audio = audio[: int(MAX_SECONDS * sr)]
            seconds = MAX_SECONDS

        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        if seconds < 0.5 or rms < 0.005:
            return {"ok": False, "reason": "too quiet or too short",
                    "seconds": round(seconds, 2), "rms": round(rms, 4)}

        s = self.rec.create_stream()
        s.accept_waveform(sr, audio)
        self.rec.decode_stream(s)
        transcript = s.result.text.strip()
        t_asr = time.time() - t0

        if len(transcript) < 3:
            return {"ok": False, "reason": "no speech recognised",
                    "seconds": round(seconds, 2), "rms": round(rms, 4),
                    "transcript": transcript}

        t1 = time.time()
        post = self.lid.posterior(transcript) or {}
        top = sorted(post.items(), key=lambda kv: -kv[1])[:3]
        t_lid = time.time() - t1

        return {
            "ok": True,
            "language": top[0][0],
            "confidence": round(top[0][1], 4),
            "margin": round(top[0][1] - (top[1][1] if len(top) > 1 else 0.0), 4),
            "top": [{"language": k, "score": round(v, 4)} for k, v in top],
            "transcript": transcript,
            "seconds": round(seconds, 2),
            "chars": len(transcript),
            # below this the head is measurably unreliable; the client warns rather than
            # hiding the answer
            "short": len(transcript) < 40,
            "timing_ms": {"asr": round(t_asr * 1000), "lid": round(t_lid * 1000)},
        }

    @modal.fastapi_endpoint(method="POST", docs=True)
    def identify(self, request_body: dict):
        """Accepts {"audio": "<base64 wav>"} and returns the predicted language."""
        import base64
        try:
            raw = base64.b64decode(request_body.get("audio", ""))
        except Exception:
            return {"ok": False, "reason": "audio is not valid base64"}
        if not raw:
            return {"ok": False, "reason": "no audio supplied"}
        try:
            return self._run(raw)
        except Exception as e:  # a bad upload should not take the container down
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    @modal.fastapi_endpoint(method="GET", docs=True)
    def warm(self):
        """Cheap endpoint the browser hits on record-start, so the container is already up
        by the time the user has finished speaking."""
        return {"ok": True, "languages": len(self.lid.languages),
                "min_seconds": MIN_SECONDS, "max_seconds": MAX_SECONDS}
