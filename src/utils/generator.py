import torch
import numpy as np
from pathlib import Path
from src.models.models import build_model
from src.core.kokoro import generate
from .voice import split_into_sentences

class VoiceGenerator:
    def __init__(self, models_dir, voices_dir):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.voicepack = None
        self.voice_name = None
        self.models_dir = models_dir
        self.voices_dir = voices_dir
        self._initialized = False
        
    def initialize(self, model_path, voice_name):
        """Initialize the model and voice pack."""
        model_file = self.models_dir / model_path
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at {model_file}. Please place the model file in the 'models' directory.")
            
        self.model = build_model(str(model_file), self.device)
        self.voice_name = voice_name
        
        voice_path = self.voices_dir / f'{voice_name}.pt'
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice pack not found at {voice_path}. Please place voice files in the 'data/voices' directory.")
            
        self.voicepack = torch.load(voice_path, weights_only=True).to(self.device)
        self._initialized = True
        return f'Loaded voice: {voice_name}'
        
    def list_available_voices(self):
        """List all available voice packs."""
        if not self.voices_dir.exists():
            return []
        return [f.stem for f in self.voices_dir.glob('*.pt')]

    def is_initialized(self):
        """Check if the generator is properly initialized."""
        return self._initialized and self.model is not None and self.voicepack is not None

    def generate(self, text, lang=None, speed=1.0, pause_duration=4000, short_text_limit=200):
        """Generate speech from text - handles both short and long-form text."""
        if not self.is_initialized():
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        if lang is None:
            lang = self.voice_name[0]

        # For short text, generate directly
        if len(text.strip()) < short_text_limit:
            audio, phonemes = generate(self.model, text, self.voicepack, lang=lang, speed=speed)
            return audio, phonemes

        # For long text, split and process each sentence
        sentences = split_into_sentences(text)
        if not sentences:
            return None, []

        audio_segments = []
        phonemes_list = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Add pause between sentences
            if audio_segments:
                audio_segments.append(np.zeros(pause_duration))
            
            # Generate and process audio for sentence
            audio, phonemes = generate(self.model, sentence, self.voicepack, lang=lang, speed=speed)
            audio_segments.append(audio)
            phonemes_list.extend(phonemes)

        return np.concatenate(audio_segments), phonemes_list 