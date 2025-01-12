import requests
import time
from pathlib import Path
import sys
import json

BASE_URL = "http://localhost:8000"

def check_server():
    """Check if the server is running and get initial status"""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            return response.json()
        return None
    except requests.ConnectionError:
        return None

def test_voice_api():
    # Check if server is running and get status
    print("Checking server status...")
    status = check_server()
    if not status:
        print("Error: Server is not running. Please start the server with 'python app.py'")
        sys.exit(1)
    
    print(f"\nServer Status:")
    print(f"Models directory: {status['models_dir']}")
    print(f"Voices directory: {status['voices_dir']}")
    print(f"Using device: {status['device']}")
    print(f"Available models: {status['models_available']}")
    print(f"Available voices: {status['voices_available']}")

    if not status['models_available']:
        print("\nError: No model files found. Please add .pth files to the models directory.")
        sys.exit(1)
    if not status['voices_available']:
        print("\nError: No voice files found. Please add .pt files to the voices directory.")
        sys.exit(1)

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 1. List available voices
    print("\n1. Testing list voices endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/voices")
        response.raise_for_status()
        voices = response.json()["voices"]
        print(f"Available voices: {voices}")
    except requests.RequestException as e:
        print(f"Error listing voices: {str(e)}")
        if hasattr(response, 'text'):
            try:
                error_data = json.loads(response.text)
                print(f"Server error: {error_data.get('detail', 'No detail provided')}")
            except:
                print(f"Raw response: {response.text}")
        return

    # Select model and voice
    model_name = status['models_available'][0]  # First available model
    voice_name = "af_sky_adam"  # Using a specific voice that we know exists

    # 2. Initialize a voice
    print(f"\n2. Testing initialize endpoint with model '{model_name}' and voice '{voice_name}'...")
    init_data = {
        "model_path": model_name,
        "voice_name": voice_name
    }
    try:
        response = requests.post(f"{BASE_URL}/initialize", json=init_data)
        response.raise_for_status()
        print(f"Voice initialized: {response.json()['message']}")
    except requests.RequestException as e:
        print(f"Error initializing voice: {str(e)}")
        if hasattr(response, 'text'):
            try:
                error_data = json.loads(response.text)
                print(f"Server error: {error_data.get('detail', 'No detail provided')}")
                if 'traceback' in error_data:
                    print(f"Traceback:\n{error_data['traceback']}")
            except:
                print(f"Raw response: {response.text}")
        return

    # Wait a bit for initialization to complete
    print("Waiting for initialization to complete...")
    time.sleep(5)

    # 3. Generate speech with different texts
    test_texts = [
        "Hello! This is a test of the voice generation system.",
        "How are you doing today?",
        "This is a longer text that will test the system's ability to handle multiple sentences. It should work well with pauses between sentences."
    ]

    print("\n3. Testing speech generation...")
    for i, text in enumerate(test_texts):
        try:
            text_data = {
                "text": text,
                "speed": 1.0,
                "pause_duration": 4000
            }
            print(f"\nGenerating speech for text {i+1}: {text[:50]}...")
            response = requests.post(f"{BASE_URL}/generate", json=text_data)
            response.raise_for_status()
            
            # Save the audio file
            output_file = output_dir / f"test_output_{i+1}.wav"
            output_file.write_bytes(response.content)
            print(f"Audio saved to {output_file}")
            
            # Wait a bit between generations
            time.sleep(2)
        except requests.RequestException as e:
            print(f"Error generating speech for text {i+1}: {str(e)}")
            if hasattr(response, 'text'):
                try:
                    error_data = json.loads(response.text)
                    print(f"Server error: {error_data.get('detail', 'No detail provided')}")
                    if 'traceback' in error_data:
                        print(f"Traceback:\n{error_data['traceback']}")
                except:
                    print(f"Raw response: {response.text}")

    # 4. Test voice mixing (if multiple voices available)
    if len(voices) >= 2:
        print("\n4. Testing voice mixing...")
        
        # Get actual available voices
        print("\nAvailable voices:", voices)
        
        # Make sure we have at least two voices to mix
        if len(voices) < 2:
            print("Not enough voices available for mixing (need at least 2)")
            return
            
        # Select two voices that actually exist
        voice1 = voices[0]
        voice2 = voices[1]
        print(f"\nWill mix voices: {voice1} and {voice2}")
        
        # Test case 1: Mix with equal weights (no weights specified)
        mix_data = {
            "output_name": "mixed_equal_auto",
            "voice_names": [voice1, voice2],
            "weights": None  # Let the server handle equal weights
        }
        try:
            print("\nTesting mix with auto equal weights...")
            response = requests.post(f"{BASE_URL}/mix-voices", json=mix_data)
            response.raise_for_status()
            result = response.json()
            print(f"Voice mixing result: {result['message']}")
            print(f"Weights used: {result['weights_used']}")
            print(f"Mixed voice saved to: {result['voice_path']}")
            
            # Verify the file exists
            if not Path(result['voice_path']).exists():
                print(f"Warning: Mixed voice file not found at {result['voice_path']}")
            
            # Wait a bit before next test
            time.sleep(2)
        except requests.RequestException as e:
            print(f"Error mixing voices with auto equal weights: {str(e)}")
            if hasattr(response, 'text'):
                try:
                    error_data = json.loads(response.text)
                    print(f"Server error: {error_data.get('detail', 'No detail provided')}")
                    if 'traceback' in error_data:
                        print(f"Traceback:\n{error_data['traceback']}")
                except:
                    print(f"Raw response: {response.text}")

        # Test case 2: Mix with custom weights
        mix_data = {
            "output_name": "mixed_custom",
            "voice_names": [voice1, voice2],
            "weights": [0.7, 0.3]  # Custom weights
        }
        try:
            print(f"\nTesting mix with custom weights (0.7, 0.3) for voices {voice1} and {voice2}...")
            response = requests.post(f"{BASE_URL}/mix-voices", json=mix_data)
            response.raise_for_status()
            result = response.json()
            print(f"Voice mixing result: {result['message']}")
            print(f"Weights used: {result['weights_used']}")
            print(f"Mixed voice saved to: {result['voice_path']}")
            
            # Verify the file exists
            if not Path(result['voice_path']).exists():
                print(f"Warning: Mixed voice file not found at {result['voice_path']}")
            
            # Wait a bit before next test
            time.sleep(2)
        except requests.RequestException as e:
            print(f"Error mixing voices with custom weights: {str(e)}")
            if hasattr(response, 'text'):
                try:
                    error_data = json.loads(response.text)
                    print(f"Server error: {error_data.get('detail', 'No detail provided')}")
                    if 'traceback' in error_data:
                        print(f"Traceback:\n{error_data['traceback']}")
                except:
                    print(f"Raw response: {response.text}")

        # Test case 3: Mix with invalid weights (should fail gracefully)
        mix_data = {
            "output_name": "mixed_invalid",
            "voice_names": [voice1, voice2],
            "weights": [0.7]  # Invalid number of weights
        }
        try:
            print("\nTesting mix with invalid weights (should fail)...")
            response = requests.post(f"{BASE_URL}/mix-voices", json=mix_data)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Expected error with invalid weights: {str(e)}")
            if hasattr(response, 'text'):
                try:
                    error_data = json.loads(response.text)
                    print(f"Server error (expected): {error_data.get('detail', 'No detail provided')}")
                except:
                    print(f"Raw response: {response.text}")

if __name__ == "__main__":
    print("Testing Voice Chat API endpoints...")
    test_voice_api() 