import os
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

from pathlib import Path
import traceback
from src.utils import play_audio, VoiceGenerator

# Define base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'data' / 'models'
VOICES_DIR = BASE_DIR / 'data' / 'voices'

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Constants
_DEFAULT_MODEL_PATH = 'kokoro-v0_19-half.pth'
_DEFAULT_VOICE_NAME = 'af_sky_adam'
_DEFAULT_SPEED = 1.0

def main():
    # Initialize the voice generator
    generator = VoiceGenerator(MODELS_DIR, VOICES_DIR)
    
    try:
        print("\n=== Voice Chat Initializing ===")
        print("Device being used:", generator.device)
        
        # Initialize the model
        print("\nInitializing voice generator...")
        result = generator.initialize(_DEFAULT_MODEL_PATH, _DEFAULT_VOICE_NAME)
        print(result)
        
        print("\n=== Voice Chat Ready ===")
        print("Type your text and press Enter. Type 'quit' to exit.")
        print("Available commands:")
        print("  speed=X : Change speech speed (e.g., speed=1.2)")
        print("  voice=X : Change voice (e.g., voice=af_sky)")
        print("  voices  : List available voices")
        print("-" * 50)
        
        # Chat loop
        speed = _DEFAULT_SPEED
        while True:
            try:
                # Get user input
                text = input("\nYou: ").strip()
                
                # Check for exit command
                if text.lower() == 'quit':
                    print("Goodbye!")
                    break
                    
                # Check for commands
                if text.lower() == 'voices':
                    voices = generator.list_available_voices()
                    print("\nAvailable voices:")
                    for voice in voices:
                        print(f"- {voice}")
                    continue
                    
                if text.startswith('speed='):
                    try:
                        speed = float(text.split('=')[1])
                        print(f"Speed set to {speed}")
                    except:
                        print("Invalid speed value. Use format: speed=1.2")
                    continue
                    
                if text.startswith('voice='):
                    try:
                        voice = text.split('=')[1]
                        if voice in generator.list_available_voices():
                            generator.initialize(_DEFAULT_MODEL_PATH, voice)
                            print(f"Switched to voice: {voice}")
                        else:
                            print("Voice not found. Use 'voices' to list available voices.")
                    except Exception as e:
                        print(f"Error changing voice: {str(e)}")
                    continue
                
                # Skip empty input
                if not text:
                    continue
                
                # Generate and play audio
                print("Generating audio...")
                audio, _ = generator.generate(text, speed=speed)
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