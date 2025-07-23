import torch
import numpy as np
from pathlib import Path
import json
import re

# Removed imports for build_model and kokoro_tts as model loading is now external.

def get_device():
    """Gets the device to use for torch operations.

    Returns:
        str: The device to use ('cuda' or 'cpu').
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_available_voices(voices_dir):
    """Gets a list of available voice names without the .pt extension.

    Args:
        voices_dir (str): The path to the directory containing voice files.

    Returns:
        list: A list of voice names (strings).
    """
    voices_dir = Path(voices_dir)
    if not voices_dir.exists():
        return []
    return [f.stem for f in voices_dir.glob("*.pt")]


def validate_voice_name(voice_name, voices_dir):
    """Validates that a voice name exists in the voices directory.

    Args:
        voice_name (str): The name of the voice to validate.
        voices_dir (str): The path to the directory containing voice files.

    Returns:
        bool: True if the voice name is valid.

    Raises:
        ValueError: If the voice name is not found in the voices directory.
    """
    available_voices = get_available_voices(voices_dir)
    if voice_name not in available_voices:
        raise ValueError(
            f"Voice '{voice_name}' not found. Available voices: {', '.join(available_voices)}"
        )
    return True


def load_voice(voice_name, voices_dir):
    """Loads a voice from the voices directory.

    Args:
        voice_name (str): The name of the voice to load.
        voices_dir (str): The path to the directory containing voice files.

    Returns:
        torch.Tensor: The loaded voice as a torch tensor.

    Raises:
        AssertionError: If the voices directory or voice file does not exist, or if the voice path is not a file.
        RuntimeError: If there is an error loading the voice file or converting it to a tensor.
    """
    voices_dir = Path(voices_dir)
    assert voices_dir.exists(), f"Voices directory does not exist: {voices_dir}"
    assert voices_dir.is_dir(), f"Voices path is not a directory: {voices_dir}"

    validate_voice_name(voice_name, voices_dir)

    voice_path = voices_dir / f"{voice_name}.pt"
    assert voice_path.exists(), f"Voice file not found: {voice_path}"
    assert voice_path.is_file(), f"Voice path is not a file: {voice_path}"

    try:
        voice = torch.load(voice_path, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Error loading voice file {voice_path}: {str(e)}")

    if not isinstance(voice, torch.Tensor):
        try:
            voice = torch.tensor(voice)
        except Exception as e:
            raise RuntimeError(f"Could not convert voice to tensor: {str(e)}")

    return voice


def quick_mix_voice(output_name, voices_dir, *voices, weights=None):
    """Mixes and saves voices with specified weights.

    Args:
        output_name (str): The name of the output mixed voice file (without extension).
        voices_dir (str): The path to the directory containing voice files.
        *voices (torch.Tensor): Variable number of voice tensors to mix.
        weights (list, optional): List of weights for each voice. Defaults to equal weights if None.

    Returns:
        torch.Tensor: The mixed voice as a torch tensor.

    Raises:
        ValueError: If no voices are provided, if the number of weights does not match the number of voices, or if the sum of weights is not positive.
        AssertionError: If the voices directory does not exist or is not a directory.
    """
    voices_dir = Path(voices_dir)
    assert voices_dir.exists(), f"Voices directory does not exist: {voices_dir}"
    assert voices_dir.is_dir(), f"Voices path is not a directory: {voices_dir}"

    if not voices:
        raise ValueError("Must provide at least one voice")

    base_shape = voices[0].shape
    for i, voice in enumerate(voices):
        if not isinstance(voice, torch.Tensor):
            raise ValueError(f"Voice {i} is not a tensor")
        if voice.shape != base_shape:
            raise ValueError(
                f"Voice {i} has shape {voice.shape}, but expected {base_shape} (same as first voice)"
            )

    if weights is None:
        weights = [1.0 / len(voices)] * len(voices)
    else:
        if len(weights) != len(voices):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of voices ({len(voices)})"
            )
        weights_sum = sum(weights)
        if weights_sum <= 0:
            raise ValueError("Sum of weights must be positive")
        weights = [w / weights_sum for w in weights]

    device = voices[0].device
    voices = [v.to(device) for v in voices]

    stacked = torch.stack(voices)
    weights = torch.tensor(weights, device=device)

    mixed = torch.zeros_like(voices[0])
    for i, weight in enumerate(weights):
        mixed += stacked[i] * weight

    output_path = voices_dir / f"{output_name}.pt"
    torch.save(mixed, output_path)
    print(f"Created mixed voice: {output_name}.pt")
    return mixed


def split_into_sentences(text):
    """Splits text into sentences using more robust rules.

    Args:
        text (str): The input text to split.

    Returns:
        list: A list of sentences (strings).
    """
    text = text.strip()
    if not text:
        return []

    abbreviations = {
        "Mr.": "Mr",
        "Mrs.": "Mrs",
        "Dr.": "Dr",
        "Ms.": "Ms",
        "Prof.": "Prof",
        "Sr.": "Sr",
        "Jr.": "Jr",
        "vs.": "vs",
        "etc.": "etc",
        "i.e.": "ie",
        "e.g.": "eg",
        "a.m.": "am",
        "p.m.": "pm",
    }

    for abbr, repl in abbreviations.items():
        text = text.replace(abbr, repl)

    sentences = []
    current = []

    words = re.findall(r"\S+|\s+", text)

    for word in words:
        current.append(word)

        if re.search(r"[.!?]+$", word):
            if not re.match(r"^[A-Z][a-z]{1,2}$", word[:-1]):
                sentence = "".join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
                continue

    if current:
        sentence = "".join(current).strip()
        if sentence:
            sentences.append(sentence)

    for abbr, repl in abbreviations.items():
        sentences = [s.replace(repl, abbr) for s in sentences]

    sentences = [s.strip() for s in sentences if s.strip()]

    final_sentences = []
    for s in sentences:
        if len(s) > 200:
            parts = s.split(",")
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                final_sentences.extend(parts)
            else:
                final_sentences.append(s)
        else:
            final_sentences.append(s)

    return final_sentences


class VoiceGenerator:
    """
    A class to manage voice generation using a pre-trained model.
    """

    def __init__(self, models_dir, voices_dir):
        """
        Initializes the VoiceGenerator with model and voice directories.

        Args:
            models_dir (Path): Path to the directory containing model files.
            voices_dir (Path): Path to the directory containing voice pack files.
        """
        self.device = get_device()
        self.model = None
        self.voicepack = None
        self.voice_name = None
        self.models_dir = models_dir
        self.voices_dir = voices_dir
        self._initialized = False

    def initialize(self, voice_name):
        """
        Initializes the model and voice pack for audio generation.

        Args:
            model_path (str): The filename of the model.
            voice_name (str): The name of the voice pack.

        Returns:
            str: A message indicating the voice has been loaded.

        Raises:
            FileNotFoundError: If the model or voice pack file is not found.
        """
        # Model loading is now handled by ModelServer, so we only load the voice pack here.
        self.voice_name = voice_name

        voice_path = self.voices_dir / f"{voice_name}.pt"
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Voice pack not found at {voice_path}. Please place voice files in the 'data/voices' directory."
            )

        self.voicepack = torch.load(voice_path, weights_only=False).to(self.device)
        self._initialized = True
        return f"Loaded voice: {voice_name}"

    def list_available_voices(self):
        """
        Lists all available voice packs in the voices directory.

        Returns:
            list: A list of voice pack names (without the .pt extension).
        """
        if not self.voices_dir.exists():
            return []
        return [f.stem for f in self.voices_dir.glob("*.pt")]

    def is_initialized(self):
        """
        Checks if the generator is properly initialized.

        Returns:
            bool: True if the model and voice pack are loaded, False otherwise.
        """
        return self._initialized and self.voicepack is not None

    def generate(
        self, tts_model, text, lang=None, speed=1.0, pause_duration=4000, short_text_limit=200, return_chunks=False,
    ):
        """
        Generates speech from the given text.

        Handles both short and long-form text by splitting long text into sentences.

        Args:
            text (str): The text to generate speech from.
            lang (str, optional): The language of the text. Defaults to None.
            speed (float, optional): The speed of speech generation. Defaults to 1.0.
            pause_duration (int, optional): The duration of pause between sentences in milliseconds. Defaults to 4000.
            short_text_limit (int, optional): The character limit for considering text as short. Defaults to 200.
            return_chunks (bool, optional): If True, returns a list of audio chunks instead of concatenated audio. Defaults to False.

        Returns:
            tuple: A tuple containing the generated audio (numpy array or list of numpy arrays) and a list of phonemes.

        Raises:
            RuntimeError: If the model is not initialized.
            ValueError: If there is an error during audio generation.
        """
        if not self.is_initialized() or tts_model is None:
            raise RuntimeError("Model not initialized. Call initialize() first and pass a valid model to generate().")

        if lang is None:
            lang = self.voice_name[0]

        text = text.strip()
        if not text:
            return (None, []) if not return_chunks else ([], [])

        try:
            if len(text) < short_text_limit:
                try:
                    audio, phonemes = tts_model.generate(
                        text, self.voicepack, lang=lang, speed=speed
                    )
                    if audio is None or len(audio) == 0:
                        raise ValueError(f"Failed to generate audio for text: {text}")
                    return (
                        (audio, phonemes) if not return_chunks else ([audio], phonemes)
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error generating audio for text: {text}. Error: {str(e)}"
                    )

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
                    if audio_segments and not return_chunks:
                        audio_segments.append(np.zeros(pause_duration))

                    audio, phonemes = tts_model.generate(
                        sentence, self.voicepack, lang=lang, speed=speed
                    )
                    if audio is not None and len(audio) > 0:
                        audio_segments.append(audio)
                        phonemes_list.extend(phonemes)
                    else:
                        failed_sentences.append(
                            (i, sentence, "Generated audio is empty")
                        )
                except Exception as e:
                    failed_sentences.append((i, sentence, str(e)))
                    continue

            if failed_sentences:
                error_msg = "\n".join(
                    [f"Sentence {i+1}: '{s}' - {e}" for i, s, e in failed_sentences]
                )
                raise ValueError(
                    f"Failed to generate audio for some sentences:\n{error_msg}"
                )

            if not audio_segments:
                return (None, []) if not return_chunks else ([], [])

            if return_chunks:
                return audio_segments, phonemes_list
            return np.concatenate(audio_segments), phonemes_list

        except Exception as e:
            raise ValueError(f"Error in audio generation: {str(e)}")
