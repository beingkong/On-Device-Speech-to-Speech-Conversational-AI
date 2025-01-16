import numpy as np
import soundfile as sf
import sounddevice as sd
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

def save_audio_file(audio_data: np.ndarray, output_dir: Path, sample_rate: int = 24000) -> Path:
    """Save audio data to a WAV file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"output_{timestamp}.wav"
    
    # Handle both single audio and list of segments
    if isinstance(audio_data, list):
        audio_data = np.concatenate(audio_data)
        
    sf.write(str(output_path), audio_data, sample_rate)
    print(f"Audio saved to: {output_path}")
    return output_path

def play_audio(audio_data: np.ndarray, sample_rate: int = 24000) -> Tuple[bool, Optional[np.ndarray]]:
    """Play audio data using sounddevice"""
    sd.play(audio_data, sample_rate)
    sd.wait()
    return False, None  # Return tuple (was_interrupted, interrupt_audio) 