import pyaudio
import numpy as np
import time
from config.settings import settings

CHUNK = settings.CHUNK
FORMAT = pyaudio.paFloat32
CHANNELS = settings.CHANNELS
RATE = settings.RATE
SILENCE_THRESHOLD = settings.SILENCE_THRESHOLD
MAX_SILENCE_DURATION = settings.MAX_SILENCE_DURATION


def record_audio(duration=None):
    """Records audio for a specified duration.

    Args:
        duration (int, optional): Recording duration in seconds. Defaults to settings.RECORD_DURATION.

    Returns:
        np.ndarray: Recorded audio data as a numpy array.
    """
    if duration is None:
        duration = settings.RECORD_DURATION

    p = pyaudio.PyAudio()

    stream = p.open(
        format=settings.FORMAT,
        channels=settings.CHANNELS,
        rate=settings.RATE,
        input=True,
        frames_per_buffer=settings.CHUNK,
    )

    print("\nRecording...")
    frames = []

    for i in range(0, int(settings.RATE / settings.CHUNK * duration)):
        data = stream.read(settings.CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.concatenate(frames, axis=0)
    return audio_data


def record_continuous_audio():
    """Continuously monitors audio and detects speech segments.

    Returns:
        np.ndarray or None: Recorded audio data as a numpy array, or None if no speech is detected.
    """
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("\nListening... (Press Ctrl+C to stop)")
    frames = []
    buffer_frames = []
    buffer_size = int(RATE * 0.5 / CHUNK)
    silence_frames = 0
    max_silence_frames = int(RATE / CHUNK * 1)
    recording = False

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)

            buffer_frames.append(audio_chunk)
            if len(buffer_frames) > buffer_size:
                buffer_frames.pop(0)

            audio_level = np.abs(np.concatenate(buffer_frames)).mean()

            if audio_level > SILENCE_THRESHOLD:
                if not recording:
                    print("\nPotential speech detected...")
                    recording = True
                    frames.extend(buffer_frames)
                frames.append(audio_chunk)
                silence_frames = 0
            elif recording:
                frames.append(audio_chunk)
                silence_frames += 1

                if silence_frames >= max_silence_frames:
                    print("Processing speech segment...")
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if frames:
        return np.concatenate(frames)
    return None
