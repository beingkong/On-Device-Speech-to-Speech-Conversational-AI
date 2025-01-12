import torch
from pathlib import Path
import json
import os

def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path) as f:
        return json.load(f)

def get_available_voices(voices_dir):
    """Get list of available voice names without .pt extension"""
    voices_dir = Path(voices_dir)
    if not voices_dir.exists():
        return []
    return [f.stem for f in voices_dir.glob("*.pt")]

def validate_voice_name(voice_name, voices_dir):
    """Validate that a voice name exists in the voices directory"""
    available_voices = get_available_voices(voices_dir)
    if voice_name not in available_voices:
        raise ValueError(
            f"Voice '{voice_name}' not found. Available voices: {', '.join(available_voices)}"
        )
    return True

def load_voice(voice_name, voices_dir):
    """Load a voice from the voices directory."""
    voices_dir = Path(voices_dir)
    assert voices_dir.exists(), f"Voices directory does not exist: {voices_dir}"
    assert voices_dir.is_dir(), f"Voices path is not a directory: {voices_dir}"
    
    # Validate voice name exists
    validate_voice_name(voice_name, voices_dir)
    
    voice_path = voices_dir / f'{voice_name}.pt'
    assert voice_path.exists(), f"Voice file not found: {voice_path}"
    assert voice_path.is_file(), f"Voice path is not a file: {voice_path}"
    
    try:
        voice = torch.load(voice_path, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Error loading voice file {voice_path}: {str(e)}")
    
    # Ensure voice is a tensor
    if not isinstance(voice, torch.Tensor):
        try:
            voice = torch.tensor(voice)
        except Exception as e:
            raise RuntimeError(f"Could not convert voice to tensor: {str(e)}")
    
    return voice

def quick_mix_voice(output_name, voices_dir, *voices, weights=None):
    """Quick function to mix and save voices with weights.
    Example usage:
        sky = load_voice('af_sky', voices_dir)
        adam = load_voice('am_adam', voices_dir)
        mixed = quick_mix_voice('af_mix', voices_dir, sky, adam, weights=[0.7, 0.3])
    """
    voices_dir = Path(voices_dir)
    assert voices_dir.exists(), f"Voices directory does not exist: {voices_dir}"
    assert voices_dir.is_dir(), f"Voices path is not a directory: {voices_dir}"
    
    if not voices:
        raise ValueError("Must provide at least one voice")
    
    # Verify all voices are tensors and have the same shape
    base_shape = voices[0].shape
    for i, voice in enumerate(voices):
        if not isinstance(voice, torch.Tensor):
            raise ValueError(f"Voice {i} is not a tensor")
        if voice.shape != base_shape:
            raise ValueError(f"Voice {i} has shape {voice.shape}, but expected {base_shape} (same as first voice)")
    
    # Handle weights
    if weights is None:
        # Equal weights if none provided
        weights = [1.0 / len(voices)] * len(voices)
    else:
        # Validate weights
        if len(weights) != len(voices):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of voices ({len(voices)})")
        # Normalize weights to sum to 1.0
        weights_sum = sum(weights)
        if weights_sum <= 0:
            raise ValueError("Sum of weights must be positive")
        weights = [w / weights_sum for w in weights]
    
    # Ensure all voices are on the same device
    device = voices[0].device
    voices = [v.to(device) for v in voices]
    
    # Stack voices and convert weights to tensor
    stacked = torch.stack(voices)  # Shape: [num_voices, style_dim]
    weights = torch.tensor(weights, device=device)
    
    # Mix voices
    mixed = torch.zeros_like(voices[0])  # Shape: [style_dim]
    for i, weight in enumerate(weights):
        mixed += stacked[i] * weight
    
    # Save mixed voice
    output_path = voices_dir / f'{output_name}.pt'
    torch.save(mixed, output_path)
    print(f"Created mixed voice: {output_name}.pt")
    return mixed

def split_into_sentences(text):
    """Split text into sentences using punctuation rules."""
    import re
    # Remove square brackets content and split on sentence endings
    text = re.sub(r'\[.*?\]', '', text)
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()] 