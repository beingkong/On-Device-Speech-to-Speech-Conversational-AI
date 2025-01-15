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

# Define base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'data' / 'models'
VOICES_DIR = BASE_DIR / 'data' / 'voices'
OUTPUT_DIR = BASE_DIR / 'output'

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
_DEFAULT_MODEL_PATH = os.getenv("TTS_MODEL")
_DEFAULT_VOICE_NAME = os.getenv("VOICE_NAME")
_DEFAULT_SPEED = float(os.getenv("SPEED"))

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
    # Initialize the voice generator
    generator = VoiceGenerator(MODELS_DIR, VOICES_DIR)
    
    try:
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
        print("Type your message and press Enter. Type 'quit' to exit.")
        print("Available commands:")
        print("  speed=X : Change speech speed (e.g., speed=1.2)")
        print("  voice=X : Change voice (e.g., voice=af_sky)")
        print("  voices  : List available voices")
        print("  mix=voice1,voice2[:weight1,weight2] : Mix two voices (e.g., mix=af_sky,am_adam or mix=af_sky,am_adam:0.7,0.3)")
        print("-" * 50)
        
        # Initialize chat history
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        speed = _DEFAULT_SPEED
        
        # Chat loop
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Check for exit command
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                    
                # Check for commands
                if user_input.lower() == 'voices':
                    voices = generator.list_available_voices()
                    print("\nAvailable voices:")
                    for voice in voices:
                        print(f"- {voice}")
                    continue
                    
                if user_input.startswith('speed='):
                    try:
                        speed = float(user_input.split('=')[1])
                        print(f"Speed set to {speed}")
                    except:
                        print("Invalid speed value. Use format: speed=1.2")
                    continue
                    
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
                    continue
                
                if user_input.startswith('mix='):
                    try:
                        # Parse the mix command
                        mix_input = user_input.split('=')[1]
                        voices_weights = mix_input.split(':')
                        voices = [v.strip() for v in voices_weights[0].split(',')]
                        
                        # Check if weights are provided
                        if len(voices_weights) > 1:
                            weights = [float(w.strip()) for w in voices_weights[1].split(',')]
                        else:
                            weights = [0.5, 0.5]  # Default to equal weights
                            
                        if len(voices) != 2 or len(weights) != 2:
                            print("Mix command requires exactly two voices. Format: mix=voice1,voice2[:weight1,weight2]")
                            continue
                            
                        # Verify voices exist
                        available_voices = generator.list_available_voices()
                        if not all(voice in available_voices for voice in voices):
                            print("One or more voices not found. Use 'voices' to list available voices.")
                            continue
                            
                        # Load and mix voices
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Use 'a' prefix for mixed voices to maintain naming convention
                        output_name = f"af_mixed_voice_{timestamp}"
                        
                        voice_tensors = []
                        for voice_name in voices:
                            voice_path = VOICES_DIR / f"{voice_name}.pt"
                            voice = torch.load(voice_path, weights_only=True)
                            voice_tensors.append(voice)
                        
                        # Mix voices using quick_mix_voice
                        from src.utils.voice import quick_mix_voice
                        mixed = quick_mix_voice(output_name, VOICES_DIR, *voice_tensors, weights=weights)
                        
                        # Initialize with mixed voice
                        result = generator.initialize(_DEFAULT_MODEL_PATH, output_name)
                        print(f"Mixed voices: {voices[0]} ({weights[0]:.1f}) and {voices[1]} ({weights[1]:.1f})")
                    except Exception as e:
                        print(f"Error mixing voices: {str(e)}")
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Add user message to history
                messages.append({"role": "user", "content": user_input})
                
                # Get AI response
                print("\nThinking...")
                ai_response = get_ai_response(messages)
                if ai_response is None:
                    print("Failed to get response from AI. Please try again.")
                    continue
                
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
                print(f"Audio saved to: {output_path}")
                
                # Play the audio
                play_audio(audio)
                
            except KeyboardInterrupt:
                print("\nInterrupted by user. Type 'quit' to exit.")
            except Exception as e:
                print(f"Error: {str(e)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 