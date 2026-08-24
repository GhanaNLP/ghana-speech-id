"""Language identification for Ghanaian and West African speech, over IPA phonemes.

Sits on top of ghana-ipa-asr: that model turns audio into IPA units, this one says which
language those units are.

    audio --[sherpa-onnx + ghana-speech-phoneme-asr]--> IPA units --[this]--> language
"""
from ghana_speech_id.model import DEFAULT_REPO, GhanaSpeechId, Prediction

__version__ = "0.1.0"
__all__ = ["DEFAULT_REPO", "GhanaSpeechId", "Prediction"]
