import pyaudio
import numpy as np
import sounddevice as sd
from queue import Queue
from config.settings import settings
import time

CHUNK = settings.CHUNK
FORMAT = pyaudio.paFloat32
CHANNELS = settings.CHANNELS
RATE = settings.RATE
SPEECH_CHECK_THRESHOLD = settings.SPEECH_CHECK_THRESHOLD


def check_for_speech(timeout=0.1):
    """Checks if speech is detected in a non-blocking way.

    Args:
        timeout (float, optional): Duration to check for speech in seconds. Defaults to 0.1.

    Returns:
        tuple: A tuple containing a boolean indicating if speech was detected and the audio data as a numpy array, or (False, None) if no speech is detected.
    """
    p = pyaudio.PyAudio()

    frames = []
    is_speech = False

    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        for _ in range(int(RATE * timeout / CHUNK)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            frames.append(audio_chunk)

            audio_level = np.abs(audio_chunk).mean()
            if audio_level > SPEECH_CHECK_THRESHOLD:
                is_speech = True
                break

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if is_speech and frames:
        return True, np.concatenate(frames)
    return False, None


def play_audio_with_interrupt(audio_data, sample_rate=24000):
    """Plays audio while monitoring for speech interruption.

    Args:
        audio_data (np.ndarray): Audio data to play.
        sample_rate (int, optional): Sample rate for playback. Defaults to 24000.

    Returns:
        tuple: A tuple containing a boolean indicating if playback was interrupted and None, or (False, None) if playback completes without interruption.
    """
    interrupt_queue = Queue()

    def input_callback(indata, frames, time, status):
        """Callback for monitoring input audio."""
        if status:
            print(f"Input status: {status}")
            return

        audio_level = np.abs(indata[:, 0]).mean()
        if audio_level > settings.INTERRUPTION_THRESHOLD:
            interrupt_queue.put(True)

    def output_callback(outdata, frames, time, status):
        """Callback for output audio."""
        if status:
            print(f"Output status: {status}")
            return

        if not interrupt_queue.empty():
            raise sd.CallbackStop()

        remaining = len(audio_data) - output_callback.position
        if remaining == 0:
            raise sd.CallbackStop()
        valid_frames = min(remaining, frames)
        outdata[:valid_frames, 0] = audio_data[
            output_callback.position : output_callback.position + valid_frames
        ]
        if valid_frames < frames:
            outdata[valid_frames:] = 0
        output_callback.position += valid_frames

    output_callback.position = 0

    try:
        with sd.InputStream(
            channels=1, callback=input_callback, samplerate=settings.RATE
        ):
            with sd.OutputStream(
                channels=1, callback=output_callback, samplerate=sample_rate
            ):
                while output_callback.position < len(audio_data):
                    sd.sleep(100)
                    if not interrupt_queue.empty():
                        return True, None
        return False, None
    except sd.CallbackStop:
        return True, None
    except Exception as e:
        print(f"Error during playback: {str(e)}")
        return False, None

def play_audio(audio_data: np.ndarray, sample_rate: int = 24000):
    """
    Play audio directly using sounddevice.

    Args:
        audio_data (np.ndarray): The audio data to play.
        sample_rate (int, optional): The sample rate of the audio data. Defaults to 24000.
    """
    try:
        sd.play(audio_data, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {str(e)}")


def stream_audio_chunks(
    audio_chunks: list, sample_rate: int = 24000, pause_duration: float = 0.2
):
    """
    Stream audio chunks one after another with a small pause between them.

    Args:
        audio_chunks (list): A list of audio chunks to play.
        sample_rate (int, optional): The sample rate of the audio data. Defaults to 24000.
        pause_duration (float, optional): The duration of the pause between chunks in seconds. Defaults to 0.2.
    """
    try:
        for chunk in audio_chunks:
            if len(chunk) == 0:
                continue
            sd.play(chunk, sample_rate)
            sd.wait()
            time.sleep(pause_duration)
    except Exception as e:
        print(f"Error streaming audio chunks: {str(e)}")
    finally:
        sd.stop()