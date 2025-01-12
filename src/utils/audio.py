import numpy as np
import sounddevice as sd

def play_audio(audio_data, sample_rate=24000):
    """Play audio directly using sounddevice"""
    try:
        # List available devices
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        
        # Find Realtek speakers
        speaker_idx = None
        for i, dev in enumerate(devices):
            if "Realtek" in dev['name'] and dev['max_output_channels'] > 0:
                speaker_idx = i
                break
        
        if speaker_idx is None:
            print("Could not find Realtek speakers, using default device")
        else:
            print(f"\nUsing device {speaker_idx}: {devices[speaker_idx]['name']}")
            sd.default.device = speaker_idx
        
        # Print audio stats for debugging
        print(f"\nAudio shape: {audio_data.shape}")
        print(f"Audio min/max values: {audio_data.min()}, {audio_data.max()}")
        print(f"Sample rate: {sample_rate}")
        
        # Ensure audio is in the correct format (float32 between -1 and 1)
        audio_float = audio_data.astype(np.float32)
        if audio_data.dtype == np.int16:
            audio_float /= 32767.0
        elif audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_float /= max(abs(audio_data.max()), abs(audio_data.min()))
            
        # Print final audio stats
        print(f"Normalized audio min/max: {audio_float.min()}, {audio_float.max()}")
        
        # Play audio with blocking
        sd.play(audio_float, sample_rate, blocking=True)
        
    except Exception as e:
        print(f"Error playing audio: {str(e)}")
        print("Audio device info:")
        print(sd.query_devices()) 