import pyaudio
import numpy as np
import torch
from torch.nn.functional import pad
import time
from queue import Queue
import sounddevice as sd
from .config import settings

def init_vad_pipeline(hf_token):
    """Initialize the Voice Activity Detection pipeline"""
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection
    
    # Load segmentation model
    model = Model.from_pretrained(
        settings.VAD_MODEL,
        use_auth_token=hf_token
    )
    
    # Create VAD pipeline
    pipeline = VoiceActivityDetection(segmentation=model)
    
    # Configure VAD parameters
    HYPER_PARAMETERS = {
        # remove speech regions shorter than that many seconds
        "min_duration_on": settings.VAD_MIN_DURATION_ON,
        # fill non-speech regions shorter than that many seconds
        "min_duration_off": settings.VAD_MIN_DURATION_OFF
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    
    return pipeline

def detect_speech_segments(pipeline, audio_data, sample_rate=None):
    """Detect speech segments in audio using pyannote VAD"""
    if sample_rate is None:
        sample_rate = settings.RATE

    # Ensure audio is the right shape (add batch dimension if needed)
    if len(audio_data.shape) == 1:
        audio_data = audio_data.reshape(1, -1)
    
    # Convert to torch tensor if needed
    if not isinstance(audio_data, torch.Tensor):
        audio_data = torch.from_numpy(audio_data)
    
    # Pad audio if too short (minimum 1 second required)
    if audio_data.shape[1] < sample_rate:
        padding_size = sample_rate - audio_data.shape[1]
        audio_data = pad(audio_data, (0, padding_size))
    
    # Get speech segments
    vad = pipeline({
        "waveform": audio_data,
        "sample_rate": sample_rate
    })
    
    # Extract speech segments
    speech_segments = []
    for speech in vad.get_timeline().support():
        start_sample = int(speech.start * sample_rate)
        end_sample = int(speech.end * sample_rate)
        if start_sample < audio_data.shape[1]:  # Ensure within bounds
            end_sample = min(end_sample, audio_data.shape[1])
            segment = audio_data[0, start_sample:end_sample]
            speech_segments.append(segment)
    
    # Concatenate all speech segments
    if speech_segments:
        return torch.cat(speech_segments)
    return None

def record_audio(duration=None):
    """Record audio for specified duration"""
    if duration is None:
        duration = settings.RECORD_DURATION

    p = pyaudio.PyAudio()
    
    stream = p.open(format=settings.FORMAT,
                   channels=settings.CHANNELS,
                   rate=settings.RATE,
                   input=True,
                   frames_per_buffer=settings.CHUNK)
    
    print("\nRecording...")
    frames = []
    
    for i in range(0, int(settings.RATE / settings.CHUNK * duration)):
        data = stream.read(settings.CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))
    
    print("Done recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convert to numpy array
    audio_data = np.concatenate(frames, axis=0)
    return audio_data

def record_continuous_audio():
    """Continuously monitor audio and detect speech segments"""
    p = pyaudio.PyAudio()
    
    stream = p.open(format=settings.FORMAT,
                   channels=settings.CHANNELS,
                   rate=settings.RATE,
                   input=True,
                   frames_per_buffer=settings.CHUNK)
    
    print("\nListening... (Press Ctrl+C to stop)")
    frames = []
    buffer_frames = []  # Rolling buffer for VAD
    buffer_size = int(settings.RATE * settings.ROLLING_BUFFER_TIME / settings.CHUNK)
    silence_frames = 0
    max_silence_frames = int(settings.RATE / settings.CHUNK * settings.MAX_SILENCE_DURATION)
    recording = False
    
    try:
        while True:
            data = stream.read(settings.CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Maintain rolling buffer
            buffer_frames.append(audio_chunk)
            if len(buffer_frames) > buffer_size:
                buffer_frames.pop(0)
            
            # Check audio level using the buffer, with clipping to prevent overflow
            buffer_data = np.concatenate(buffer_frames)
            audio_level = np.clip(buffer_data, -1.0, 1.0).mean()
            
            if audio_level > settings.SILENCE_THRESHOLD:
                if not recording:
                    print("\nPotential speech detected...")
                    recording = True
                    # Include some pre-buffer when speech starts
                    frames.extend(buffer_frames)
                frames.append(audio_chunk)
                silence_frames = 0
            elif recording:
                frames.append(audio_chunk)
                silence_frames += 1
                
                # Stop if silence is too long
                if silence_frames >= max_silence_frames:
                    print("Processing speech segment...")
                    break
            
            # Small sleep to prevent CPU overload
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    if frames:
        audio_data = np.concatenate(frames)
        # Normalize audio data to prevent overflow
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val
        return audio_data
    return None

def check_for_speech(timeout=None):
    """Check if speech is detected in a non-blocking way"""
    if timeout is None:
        timeout = settings.SPEECH_CHECK_TIMEOUT

    p = pyaudio.PyAudio()
    
    frames = []
    is_speech = False
    
    try:
        stream = p.open(format=settings.FORMAT,
                       channels=settings.CHANNELS,
                       rate=settings.RATE,
                       input=True,
                       frames_per_buffer=settings.CHUNK)
        
        # Only check for a short duration
        for _ in range(int(settings.RATE * timeout / settings.CHUNK)):
            data = stream.read(settings.CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            frames.append(audio_chunk)
            
            # Check audio level with clipping to prevent overflow
            audio_level = np.clip(audio_chunk, -1.0, 1.0).mean()
            if audio_level > settings.SPEECH_CHECK_THRESHOLD:
                is_speech = True
                break
        
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    if is_speech and frames:
        audio_data = np.concatenate(frames)
        # Normalize audio data to prevent overflow
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val
        return True, audio_data
    return False, None

def play_audio_with_interrupt(audio_data, sample_rate=None):
    """Play audio while monitoring for speech interruption"""
    if sample_rate is None:
        sample_rate = settings.OUTPUT_SAMPLE_RATE

    # Create queues for communication between callbacks
    interrupt_queue = Queue()
    speech_buffer = Queue()
    
    # Rolling buffer for capturing initial speech
    buffer_size = int(settings.RATE * settings.ROLLING_BUFFER_TIME)
    rolling_buffer = np.zeros(buffer_size)
    buffer_position = 0
    
    def input_callback(indata, frames, time, status):
        """Callback for monitoring input audio"""
        nonlocal buffer_position
        
        if status:
            print(f"Input status: {status}")
            return
            
        # Update rolling buffer
        audio_chunk = indata[:, 0]
        chunk_size = len(audio_chunk)
        
        # Roll the buffer and add new data
        rolling_buffer[:-chunk_size] = rolling_buffer[chunk_size:]
        rolling_buffer[-chunk_size:] = audio_chunk
        
        # Check audio level for potential speech with clipping to prevent overflow
        audio_level = np.clip(audio_chunk, -1.0, 1.0).mean()
        if audio_level > settings.SILENCE_THRESHOLD:
            # Normalize the rolling buffer before putting it in the queue
            buffer_copy = rolling_buffer.copy()
            max_val = np.abs(buffer_copy).max()
            if max_val > 0:
                buffer_copy = buffer_copy / max_val
            speech_buffer.put(buffer_copy)
            interrupt_queue.put(True)
    
    def output_callback(outdata, frames, time, status):
        """Callback for output audio"""
        if status:
            print(f"Output status: {status}")
            return
            
        # Check if we should interrupt
        if not interrupt_queue.empty():
            raise sd.CallbackStop()
            
        # Calculate remaining frames
        remaining = len(audio_data) - output_callback.position
        if remaining == 0:
            raise sd.CallbackStop()
        valid_frames = min(remaining, frames)
        outdata[:valid_frames, 0] = audio_data[output_callback.position:output_callback.position + valid_frames]
        if valid_frames < frames:
            outdata[valid_frames:] = 0
        output_callback.position += valid_frames
    
    # Initialize position counter
    output_callback.position = 0
    
    try:
        # Open both input and output streams
        with sd.InputStream(channels=1, callback=input_callback, samplerate=settings.RATE):
            with sd.OutputStream(channels=1, callback=output_callback, samplerate=sample_rate):
                while output_callback.position < len(audio_data):
                    sd.sleep(100)
                    if not interrupt_queue.empty():
                        # Get the initial speech that triggered interruption
                        initial_speech = speech_buffer.get() if not speech_buffer.empty() else None
                        return True, initial_speech
        return False, None
    except sd.CallbackStop:
        # Get the initial speech that triggered interruption
        initial_speech = speech_buffer.get() if not speech_buffer.empty() else None
        return True, initial_speech
    except Exception as e:
        print(f"Error during playback: {str(e)}")
        return False, None

def transcribe_audio(processor, model, audio_data, sampling_rate=None):
    """Transcribe audio using Whisper"""
    if sampling_rate is None:
        sampling_rate = settings.RATE

    if audio_data is None:
        return ""
        
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.numpy()
        
    input_features = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0] 