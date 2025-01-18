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

    def generate(self, text, lang=None, speed=1.0, pause_duration=4000, short_text_limit=200, return_chunks=False):
        """Generate speech from text - handles both short and long-form text."""
        if not self.is_initialized():
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        if lang is None:
            lang = self.voice_name[0]

        # Clean and normalize the text
        text = text.strip()
        if not text:
            return (None, []) if not return_chunks else ([], [])

        try:
            # For short text, generate directly
            if len(text) < short_text_limit:
                try:
                    audio, phonemes = generate(self.model, text, self.voicepack, lang=lang, speed=speed)
                    if audio is None or len(audio) == 0:
                        raise ValueError(f"Failed to generate audio for text: {text}")
                    return (audio, phonemes) if not return_chunks else ([audio], phonemes)
                except Exception as e:
                    raise ValueError(f"Error generating audio for text: {text}. Error: {str(e)}")

            # For long text, split and process each sentence
            sentences = split_into_sentences(text)
            if not sentences:
                return (None, []) if not return_chunks else ([], [])

            audio_segments = []
            phonemes_list = []
            failed_sentences = []
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                try:
                    # Add pause between sentences
                    if audio_segments and not return_chunks:
                        audio_segments.append(np.zeros(pause_duration))
                    
                    # Generate and process audio for sentence
                    audio, phonemes = generate(self.model, sentence, self.voicepack, lang=lang, speed=speed)
                    if audio is not None and len(audio) > 0:
                        audio_segments.append(audio)
                        phonemes_list.extend(phonemes)
                    else:
                        failed_sentences.append((i, sentence, "Generated audio is empty"))
                except Exception as e:
                    failed_sentences.append((i, sentence, str(e)))
                    continue
            
            if failed_sentences:
                error_msg = "\n".join([f"Sentence {i+1}: '{s}' - {e}" for i, s, e in failed_sentences])
                raise ValueError(f"Failed to generate audio for some sentences:\n{error_msg}")
            
            if not audio_segments:
                return (None, []) if not return_chunks else ([], [])

            if return_chunks:
                return audio_segments, phonemes_list
            return np.concatenate(audio_segments), phonemes_list
            
        except Exception as e:
            raise ValueError(f"Error in audio generation: {str(e)}") 