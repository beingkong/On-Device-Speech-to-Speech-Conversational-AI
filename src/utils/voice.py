import torch
from pathlib import Path

def load_voice(voice_name, voices_dir):
    """Load a voice from the voices directory."""
    voice_path = voices_dir / f'{voice_name}.pt'
    if not voice_path.exists():
        raise FileNotFoundError(f"Voice {voice_name} not found at {voice_path}")
    return torch.load(voice_path, weights_only=True)

def quick_mix_voice(output_name, voices_dir, *voices, weights=None):
    """Quick function to mix and save voices with weights.
    Example usage:
        sky = load_voice('af_sky', voices_dir)
        adam = load_voice('am_adam', voices_dir)
        mixed = quick_mix_voice('af_mix', voices_dir, sky, adam, weights=[0.7, 0.3])
    """
    if not voices:
        raise ValueError("Must provide at least one voice")
    
    if weights is None:
        # Equal weights if none provided
        weights = [1.0 / len(voices)] * len(voices)
    else:
        # Validate weights
        if len(weights) != len(voices):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of voices ({len(voices)})")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
    # Stack voices
    stacked = torch.stack(voices)  # Shape: [num_voices, style_dim]
    
    # Convert weights to tensor and apply to each voice
    weights = torch.tensor(weights, device=stacked.device)
    mixed = torch.zeros_like(voices[0])  # Shape: [style_dim]
    
    # Mix each voice according to its weight
    for i, weight in enumerate(weights):
        mixed += stacked[i] * weight
    
    # Save mixed voice
    torch.save(mixed, voices_dir / f'{output_name}.pt')
    print(f"Created mixed voice: {output_name}.pt")
    return mixed

def split_into_sentences(text):
    """Split text into sentences using punctuation rules."""
    import re
    # Remove square brackets content and split on sentence endings
    text = re.sub(r'\[.*?\]', '', text)
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()] 