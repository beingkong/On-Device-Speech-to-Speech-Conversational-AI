import os
import torch
from transformers import pipeline, AutoProcessor, VoxtralForConditionalGeneration
from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel, HiggsAudioConfig
from src.config.settings import settings

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

        print("Loading Higgs-Audio TTS model...")
        from huggingface_hub import snapshot_download
        model_dir = "data/tts_model"
        model_repo = getattr(settings, "HIGGS_MODEL_REPO", None)
        # 如果模型目录不存在或为空，则自动下载
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            if not model_repo:
                raise RuntimeError("HIGGS_MODEL_REPO 环境变量未设置，无法自动下载模型！")
            print(f"Model not found in {model_dir}, downloading from {model_repo} ...")
            snapshot_download(repo_id=model_repo, local_dir=model_dir, local_dir_use_symlinks=False)
            print("Download complete.")
        # 加载配置和模型
        higgs_config = HiggsAudioConfig.from_pretrained(model_dir)
        self.tts_engine = HiggsAudioModel.from_pretrained(model_dir, config=higgs_config)
        self.tts_engine.to(self.device)
        print("Higgs-Audio TTS model loaded.")
