import os
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import traceback
import requests
import json
from src.utils import play_audio, VoiceGenerator
import re
import soundfile as sf
from datetime import datetime
import torch
import pyaudio
import wave
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyannote.audio import Pipeline
from torch.nn.functional import pad
import time

# Define base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'data' / 'models'
VOICES_DIR = BASE_DIR / 'data' / 'voices'
OUTPUT_DIR = BASE_DIR / 'output'
RECORDINGS_DIR = BASE_DIR / 'recordings'

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
_DEFAULT_MODEL_PATH = os.getenv("TTS_MODEL")
_DEFAULT_VOICE_NAME = os.getenv("VOICE_NAME")
_DEFAULT_SPEED = float(os.getenv("SPEED"))
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Add this to your .env file

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz

def init_vad_pipeline():
    """Initialize the Voice Activity Detection pipeline"""
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection
    
    # Load segmentation model
    model = Model.from_pretrained(
        "pyannote/segmentation-3.0",
        use_auth_token=HF_TOKEN
    )
    
    # Create VAD pipeline
    pipeline = VoiceActivityDetection(segmentation=model)
    
    # Configure VAD parameters
    HYPER_PARAMETERS = {
        # remove speech regions shorter than that many seconds
        "min_duration_on": 0.1,
        # fill non-speech regions shorter than that many seconds
        "min_duration_off": 0.1
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    
    return pipeline

def detect_speech_segments(pipeline, audio_data, sample_rate=RATE):
    """Detect speech segments in audio using pyannote VAD"""
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

def record_audio(duration=5):
    """Record audio for specified duration"""
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    print("\nRecording...")
    frames = []
    
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))
    
    print("Done recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convert to numpy array
    audio_data = np.concatenate(frames, axis=0)
    return audio_data

def transcribe_audio(processor, model, audio_data, sampling_rate=RATE):
    """Transcribe audio using Whisper"""
    if audio_data is None:
        return ""
        
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.numpy()
        
    input_features = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

def record_continuous_audio():
    """Continuously monitor audio and detect speech segments"""
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    print("\nListening... (Press Ctrl+C to stop)")
    frames = []
    buffer_frames = []  # Rolling buffer for VAD
    buffer_size = int(RATE * 0.5 / CHUNK)  # 0.5 second buffer
    silence_threshold = 0.01
    silence_frames = 0
    max_silence_frames = int(RATE / CHUNK * 1)  # 1 second of silence
    recording = False
    
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Maintain rolling buffer
            buffer_frames.append(audio_chunk)
            if len(buffer_frames) > buffer_size:
                buffer_frames.pop(0)
            
            # Check audio level using the buffer
            audio_level = np.abs(np.concatenate(buffer_frames)).mean()
            
            if audio_level > silence_threshold:
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
        return np.concatenate(frames)
    return None

# LM Studio API settings
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL")
DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT")
LLM_MODEL = os.getenv("LLM_MODEL")
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
# Function to filter AI response
def filter_response(response):
    # Remove markdown
    response = re.sub(r'\*\*|__|~~|`', '', response)  # Remove markdown symbols
    # Remove emojis
    response = re.sub(r'[\U00010000-\U0010ffff]', '', response, flags=re.UNICODE)  # Remove emojis
    return response

def get_ai_response(messages):
    """Get response from LM Studio API"""
    try:
        response = requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "messages": messages,
                "model": LLM_MODEL,
                "temperature": 0.7,
                "max_tokens": MAX_TOKENS,
                "stream": False
            },
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"Error communicating with LM Studio: {str(e)}")
        return None

def main():
    try:
        # Initialize the voice generator
        generator = VoiceGenerator(MODELS_DIR, VOICES_DIR)
        
        # Initialize Whisper
        print("\nInitializing Whisper model...")
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        
        # Initialize VAD pipeline
        print("\nInitializing Voice Activity Detection...")
        vad_pipeline = init_vad_pipeline()
        
        print("\n=== Voice Chat Bot Initializing ===")
        print("Device being used:", generator.device)
        
        # Initialize the model
        print("\nInitializing voice generator...")
        result = generator.initialize(_DEFAULT_MODEL_PATH, _DEFAULT_VOICE_NAME)
        print(result)
        
        # Test LM Studio connection
        print("\nTesting LM Studio connection...")
        test_response = get_ai_response([
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "Hello"}
        ])
        if test_response is None:
            print("Error: Could not connect to LM Studio. Make sure it's running and the API is accessible.")
            return
        
        print("\n=== Voice Chat Bot Ready ===")
        print("The bot is now listening for speech.")
        print("Just start speaking, and I'll respond automatically!")
        print("\nOther options:")
        print("  - Type text messages and press Enter")
        print("  - Use 'v' for manual voice recording")
        print("  - Commands: speed=X, voice=X, voices, mix=voice1,voice2")
        print("  - Type 'quit' to exit")
        print("-" * 50)
        
        # Initialize chat history
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        speed = _DEFAULT_SPEED
        
        # Chat loop
        while True:
            try:
                # Check for keyboard input (non-blocking)
                import msvcrt
                if msvcrt.kbhit():
                    user_input = input("\nYou (text): ").strip()
                    
                    # Check for exit command
                    if user_input.lower() == 'quit':
                        print("Goodbye!")
                        break
                        
                    # Handle manual voice recording
                    if user_input.lower() == 'v':
                        audio_data = record_audio()
                    else:
                        # Handle other commands
                        if handle_commands(user_input, generator, speed):
                            continue
                        # Process text input
                        process_input(user_input, messages, generator, speed)
                        continue
                
                # Continuously monitor audio
                audio_data = record_continuous_audio()
                if audio_data is not None:
                    # Detect speech segments
                    speech_segments = detect_speech_segments(vad_pipeline, audio_data)
                    
                    if speech_segments is not None:
                        print("\nTranscribing detected speech...")
                        user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                        if user_input.strip():
                            print(f"You (voice): {user_input}")
                            # Process the transcribed input
                            process_input(user_input, messages, generator, speed)
                    else:
                        print("No clear speech detected, please try again.")
                
            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()

def handle_commands(user_input, generator, speed):
    """Handle bot commands, return True if command was handled"""
    if user_input.lower() == 'voices':
        voices = generator.list_available_voices()
        print("\nAvailable voices:")
        for voice in voices:
            print(f"- {voice}")
        return True
        
    if user_input.startswith('speed='):
        try:
            speed = float(user_input.split('=')[1])
            print(f"Speed set to {speed}")
        except:
            print("Invalid speed value. Use format: speed=1.2")
        return True
        
    if user_input.startswith('voice='):
        try:
            voice = user_input.split('=')[1]
            if voice in generator.list_available_voices():
                generator.initialize(_DEFAULT_MODEL_PATH, voice)
                print(f"Switched to voice: {voice}")
            else:
                print("Voice not found. Use 'voices' to list available voices.")
        except Exception as e:
            print(f"Error changing voice: {str(e)}")
        return True
        
    if user_input.startswith('mix='):
        # ... existing mix command code ...
        return True
    
    return False

def process_input(user_input, messages, generator, speed):
    """Process user input and generate response"""
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    # Get AI response
    print("\nThinking...")
    ai_response = get_ai_response(messages)
    if ai_response is None:
        print("Failed to get response from AI. Please try again.")
        return
    
    # Filter AI response
    ai_response = filter_response(ai_response)

    # Add AI response to history
    messages.append({"role": "assistant", "content": ai_response})
    print(f"\nAI: {ai_response}")
    
    # Generate and play audio
    print("\nGenerating speech...")
    audio, _ = generator.generate(ai_response, speed=speed)
    
    # Save audio file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"output_{timestamp}.wav"
    sf.write(str(output_path), audio, 24000)
    
    # Play the audio
    play_audio(audio)

if __name__ == "__main__":
    main() 