import torch
from transformers import pipeline, AutoProcessor, VoxtralForConditionalGeneration
from components.voice.voice_manager import VoiceGenerator
from config.settings import settings

class ModelServer:
    """
    A singleton class to load and hold all the heavy ML models in memory.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to use {self.device}")

        print("Loading VAD model...")
        self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.vad_model.to(self.device)
        print("VAD model loaded.")

        print("Loading STT model (Voxtral-Mini)...")
        model_name = "mistralai/Voxtral-Mini-3B-2507"
        self.stt_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.stt_model = VoxtralForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        print("STT model loaded.")

        print("Loading TTS model...")
        self.voice_generator = VoiceGenerator(settings.MODELS_DIR, settings.VOICES_DIR)
        self.voice_generator.initialize(settings.TTS_MODEL, settings.VOICE_NAME)
        print("TTS model loaded.")
