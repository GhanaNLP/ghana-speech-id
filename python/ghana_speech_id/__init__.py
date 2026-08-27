"""Language identification for 41 Ghanaian and West African languages, over transcripts.

Sits on top of Omnilingual ASR: that model turns audio into text, this one says which
language the text is in.

    audio --[sherpa-onnx + omniASR CTC]--> transcript --[this]--> language
"""
from ghana_speech_id.model import DEFAULT_REPO, GhanaSpeechId, Prediction

__version__ = "0.2.0"
__all__ = ["DEFAULT_REPO", "GhanaSpeechId", "Prediction"]
