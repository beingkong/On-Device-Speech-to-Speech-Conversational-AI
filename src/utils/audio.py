import numpy as np
import sounddevice as sd
import time

def play_audio(audio_data, sample_rate=24000):
    """Play audio directly using sounddevice"""
    try:
        # Play audio
        sd.play(audio_data, sample_rate)
        sd.wait()  # Wait until audio is finished playing
    except Exception as e:
        print(f"Error playing audio: {str(e)}")

def stream_audio_chunks(audio_chunks, sample_rate=24000, pause_duration=0.2):
    """Stream audio chunks one after another with a small pause between them"""
    try:
        for chunk in audio_chunks:
            if len(chunk) == 0:
                continue
                
            # Play the current chunk
            sd.play(chunk, sample_rate)
            sd.wait()  # Wait until chunk is finished playing
            
            # Small pause between chunks for natural speech flow
            time.sleep(pause_duration)
            
    except Exception as e:
        print(f"Error streaming audio chunks: {str(e)}")
        
    finally:
        # Ensure the stream is stopped
        sd.stop() 