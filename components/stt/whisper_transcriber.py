import numpy as np
import torch
from config.settings import settings

def transcribe_audio(processor, model, audio_data, sampling_rate=None):
    """Transcribes audio using Whisper.

    Args:
        processor (transformers.WhisperProcessor): Whisper processor.
        model (transformers.WhisperForConditionalGeneration): Whisper model.
        audio_data (np.ndarray or torch.Tensor): Audio data to transcribe.
        sampling_rate (int, optional): Sample rate of the audio. Defaults to settings.RATE.

    Returns:
        str: Transcribed text.
    """
    if sampling_rate is None:
        sampling_rate = settings.RATE

    if audio_data is None:
        return ""

    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.numpy()

    input_features = processor(
        audio_data, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]
