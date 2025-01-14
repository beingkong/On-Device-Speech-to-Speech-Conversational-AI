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

# Define base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'data' / 'models'
VOICES_DIR = BASE_DIR / 'data' / 'voices'

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Constants
_DEFAULT_MODEL_PATH = 'kokoro-v0_19-half.pth'
_DEFAULT_VOICE_NAME = 'af_nicole'
_DEFAULT_SPEED = 1.0

# LM Studio API settings
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL")
DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT")
LLM_MODEL = os.getenv("LLM_MODEL")

def get_ai_response(messages):
    """Get response from LM Studio API"""
    try:
        response = requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "messages": messages,
                "model": LLM_MODEL,
                "temperature": 0.7,
                "max_tokens": 500,
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
                
                # Add AI response to history
                messages.append({"role": "assistant", "content": ai_response})
                print(f"\nAI: {ai_response}")
                
                # Generate and play audio
                print("\nGenerating speech...")
                audio, _ = generator.generate(ai_response, speed=speed)
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