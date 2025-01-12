import asyncio
import aiohttp
from pathlib import Path
import sys
import json

BASE_URL = "http://localhost:8000"

async def check_server():
    """Check if the server is running and get initial status"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/") as response:
                if response.status == 200:
                    return await response.json()
        return None
    except aiohttp.ClientError:
        return None

async def test_voice_api():
    # Check if server is running and get status
    print("Checking server status...")
    status = await check_server()
    if not status:
        print("Error: Server is not running. Please start the server with 'python app.py'")
        sys.exit(1)
    
    print(f"\nServer Status:")
    print(f"Models directory: {status['models_dir']}")
    print(f"Voices directory: {status['voices_dir']}")
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

    async with aiohttp.ClientSession() as session:
        # 1. List available voices
        print("\n1. Testing list voices endpoint...")
        try:
            async with session.get(f"{BASE_URL}/voices") as response:
                response.raise_for_status()
                data = await response.json()
                voices = data["voices"]
                print(f"Available voices: {voices}")
        except aiohttp.ClientError as e:
            print(f"Error listing voices: {str(e)}")
            return

        # Select model and voice
        model_name = status['models_available'][0]  # First available model
        voice_name = "af_sky_adam"  # Using a specific voice that we know exists

        # 2. Initialize voice
        print(f"\n2. Testing initialize endpoint with model '{model_name}' and voice '{voice_name}'...")
        init_data = {
            "model_path": model_name,
            "voice_name": voice_name
        }
        try:
            async with session.post(f"{BASE_URL}/initialize", json=init_data) as response:
                response.raise_for_status()
                result = await response.json()
                print(f"Voice initialized: {result['message']}")
        except aiohttp.ClientResponseError as e:
            print(f"Error initializing voice: {str(e)}")
            try:
                error_text = await response.text()
                error_json = json.loads(error_text)
                print(f"Details: {error_json['detail']}")
            except:
                print(f"Raw response: {error_text if 'error_text' in locals() else 'No response text available'}")
            return

        # 3. Generate speech with different texts
        test_texts = [
            "Hello! This is an async test of the voice generation system.",
            "How are you doing today? Testing async generation.",
            "This is a longer text that will test the async system's ability to handle multiple sentences. It should work well with pauses between sentences."
        ]

        print("\n3. Testing speech generation...")
        for i, text in enumerate(test_texts):
            try:
                text_data = {
                    "text": text,
                    "speed": 1.0,
                    "pause_duration": 4000
                }
                async with session.post(f"{BASE_URL}/generate", json=text_data) as response:
                    response.raise_for_status()
                    output_file = output_dir / f"test_output_async_{i+1}.wav"
                    output_file.write_bytes(await response.read())
                    print(f"Audio saved to {output_file}")
            except aiohttp.ClientError as e:
                print(f"Error generating speech for text {i+1}: {str(e)}")
                try:
                    error_text = await response.text()
                    error_json = json.loads(error_text)
                    print(f"Details: {error_json.get('detail', 'No details available')}")
                except:
                    pass

        # 4. Test voice mixing (if multiple voices available)
        if len(voices) >= 2:
            print("\n4. Testing voice mixing...")
            mix_data = {
                "output_name": "mixed_test_voice_async",
                "voice_names": ["af_sky_adam", "am_adam"],  # Using specific voices we know exist
                "weights": [0.6, 0.4]
            }
            try:
                async with session.post(f"{BASE_URL}/mix-voices", json=mix_data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    print(f"Voice mixing result: {result['message']}")
            except aiohttp.ClientError as e:
                print(f"Error mixing voices: {str(e)}")
                try:
                    error_text = await response.text()
                    error_json = json.loads(error_text)
                    print(f"Details: {error_json.get('detail', 'No details available')}")
                except:
                    pass

async def main():
    print("Testing Voice Chat API endpoints (async version)...")
    await test_voice_api()

if __name__ == "__main__":
    asyncio.run(main()) 