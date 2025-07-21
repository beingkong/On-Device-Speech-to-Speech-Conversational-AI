import numpy as np
import torch
from torch.nn.functional import pad
from config.settings import settings

def init_vad_pipeline(hf_token):
    """Initializes the Voice Activity Detection pipeline.

    Args:
        hf_token (str): Hugging Face API token.

    Returns:
        pyannote.audio.pipelines.VoiceActivityDetection: VAD pipeline.
    """
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    model = Model.from_pretrained(settings.VAD_MODEL, use_auth_token=hf_token)

    pipeline = VoiceActivityDetection(segmentation=model)

    HYPER_PARAMETERS = {
        "min_duration_on": settings.VAD_MIN_DURATION_ON,
        "min_duration_off": settings.VAD_MIN_DURATION_OFF,
    }
    pipeline.instantiate(HYPER_PARAMETERS)

    return pipeline


def detect_speech_segments(pipeline, audio_data, sample_rate=None):
    """Detects speech segments in audio using pyannote VAD.

    Args:
        pipeline (pyannote.audio.pipelines.VoiceActivityDetection): VAD pipeline.
        audio_data (np.ndarray or torch.Tensor): Audio data.
        sample_rate (int, optional): Sample rate of the audio. Defaults to settings.RATE.

    Returns:
        torch.Tensor or None: Concatenated speech segments as a torch tensor, or None if no speech is detected.
    """
    if sample_rate is None:
        sample_rate = settings.RATE

    if len(audio_data.shape) == 1:
        audio_data = audio_data.reshape(1, -1)

    if not isinstance(audio_data, torch.Tensor):
        audio_data = torch.from_numpy(audio_data)

    if audio_data.shape[1] < sample_rate:
        padding_size = sample_rate - audio_data.shape[1]
        audio_data = pad(audio_data, (0, padding_size))

    vad = pipeline({"waveform": audio_data, "sample_rate": sample_rate})

    speech_segments = []
    for speech in vad.get_timeline().support():
        start_sample = int(speech.start * sample_rate)
        end_sample = int(speech.end * sample_rate)
        if start_sample < audio_data.shape[1]:
            end_sample = min(end_sample, audio_data.shape[1])
            segment = audio_data[0, start_sample:end_sample]
            speech_segments.append(segment)

    if speech_segments:
        return torch.cat(speech_segments)
    return None
