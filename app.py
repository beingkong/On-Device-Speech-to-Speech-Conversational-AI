import os
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.name == 'nt':
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
import numpy as np
import io
import soundfile as sf
import torch
from src.utils.generator import VoiceGenerator
from src.utils.voice import quick_mix_voice, load_voice
from src.utils.audio import play_audio

app = FastAPI(title="Voice Chat API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize paths
MODELS_DIR = Path("data/models").absolute()
VOICES_DIR = Path("data/voices").absolute()

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the voice generator with device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = VoiceGenerator(MODELS_DIR, VOICES_DIR)

class TextRequest(BaseModel):
    text: str
    lang: str | None = None
    speed: float = 1.0
    pause_duration: int = 4000

class VoiceInitRequest(BaseModel):
    model_path: str
    voice_name: str

class VoiceMixRequest(BaseModel):
    output_name: str
    voice_names: list[str]
    weights: list[float] | None = None

@app.get("/")
async def root():
    """Root endpoint that also checks if required directories and files exist"""
    status = {
        "message": "Voice Chat API is running",
        "models_dir": str(MODELS_DIR),
        "voices_dir": str(VOICES_DIR),
        "models_available": [],
        "voices_available": [],
        "device": str(device),
        "initialized": generator.is_initialized()
    }
    
    # Check for model files
    if MODELS_DIR.exists():
        status["models_available"] = [f.name for f in MODELS_DIR.glob("*.pth")]
    
    # Check for voice files
    if VOICES_DIR.exists():
        status["voices_available"] = [f.stem for f in VOICES_DIR.glob("*.pt")]
    
    return status

@app.get("/models")
async def list_models():
    """List available model files"""
    try:
        models = [f.name for f in MODELS_DIR.glob("*.pth")]
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def list_voices():
    """List available voice files"""
    try:
        voices = generator.list_available_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/initialize")
async def initialize_voice(request: VoiceInitRequest):
    """Initialize the voice generator with a model and voice"""
    try:
        # Check if model file exists
        model_path = MODELS_DIR / request.model_path
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file '{request.model_path}' not found in {MODELS_DIR}"
            )
        
        # Check if voice file exists
        voice_path = VOICES_DIR / f"{request.voice_name}.pt"
        if not voice_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Voice file '{request.voice_name}.pt' not found in {VOICES_DIR}"
            )
        
        # Initialize with proper error handling
        try:
            message = generator.initialize(request.model_path, request.voice_name)
            logger.info(f"Successfully initialized model {request.model_path} with voice {request.voice_name}")
            return {"message": message}
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Initialization error: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing voice: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate")
async def generate_speech(request: TextRequest):
    """Generate speech from text"""
    try:
        if not generator.is_initialized():
            raise HTTPException(status_code=400, detail="Voice not initialized. Call /initialize first.")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        logger.info(f"Generating speech for text: {request.text[:50]}...")
        
        try:
            # Generate audio
            audio, phonemes = generator.generate(
                request.text,
                lang=request.lang,
                speed=request.speed,
                pause_duration=request.pause_duration
            )
            
            if audio is None or len(audio) == 0:
                raise HTTPException(status_code=500, detail="Generated audio is empty")
            
            # Convert to WAV format
            buffer = io.BytesIO()
            sf.write(buffer, audio, 24000, format='WAV')
            buffer.seek(0)
            
            logger.info("Successfully generated speech")
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=generated_speech.wav",
                    "X-Phonemes": phonemes  # Include phonemes in response headers if needed
                }
            )
        except Exception as e:
            logger.error(f"Error in generation process: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/mix-voices")
async def mix_voices(request: VoiceMixRequest):
    """Mix multiple voices together"""
    try:
        # Check if all voice files exist
        missing_voices = []
        for voice_name in request.voice_names:
            voice_path = VOICES_DIR / f"{voice_name}.pt"
            if not voice_path.exists():
                missing_voices.append(voice_name)
        
        if missing_voices:
            raise HTTPException(
                status_code=404,
                detail=f"Voice files not found: {', '.join(missing_voices)}"
            )

        # Validate number of voices
        if len(request.voice_names) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two voices are required for mixing"
            )

        # Load all voices first to validate they can be loaded
        voices = []
        try:
            for name in request.voice_names:
                voice = load_voice(name, VOICES_DIR)
                voice = voice.to(device)  # Move to correct device
                voices.append(voice)
                logger.info(f"Loaded voice {name} with shape {voice.shape}")
        except Exception as e:
            logger.error(f"Error loading voices: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error loading voices: {str(e)}"
            )

        # Create output name with timestamp to avoid conflicts
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{request.output_name}_{timestamp}"

        try:
            # Mix voices
            logger.info(f"Mixing {len(voices)} voices with weights: {request.weights}")
            mixed = quick_mix_voice(output_name, VOICES_DIR, *voices, weights=request.weights)
            logger.info(f"Successfully created mixed voice: {output_name}")
            
            # Verify the mixed voice file exists
            mixed_path = VOICES_DIR / f"{output_name}.pt"
            if not mixed_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail=f"Mixed voice file was not created at {mixed_path}"
                )
            
            # Get the actual weights used (in case they were normalized)
            actual_weights = [1.0/len(voices)] * len(voices) if request.weights is None else request.weights
            if actual_weights:
                weight_sum = sum(actual_weights)
                if weight_sum > 0:
                    actual_weights = [w/weight_sum for w in actual_weights]
            
            return {
                "message": f"Created mixed voice: {output_name}",
                "voice_name": output_name,
                "weights_used": actual_weights,
                "voice_path": str(mixed_path),
                "voice_names": request.voice_names
            }
        except Exception as e:
            logger.error(f"Error during voice mixing: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error during voice mixing: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error mixing voices: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Print startup information
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info(f"Voices directory: {VOICES_DIR}")
    logger.info(f"Using device: {device}")
    
    # Print available models and voices
    logger.info("\nAvailable models:")
    models = list(MODELS_DIR.glob("*.pth"))
    for model in models:
        logger.info(f"- {model.name}")
    
    logger.info("\nAvailable voices:")
    voices = list(VOICES_DIR.glob("*.pt"))
    for voice in voices:
        logger.info(f"- {voice.stem}")
    
    if not models:
        logger.warning("No model files found. Please place .pth files in the models directory.")
    if not voices:
        logger.warning("No voice files found. Please place .pt files in the voices directory.")
    
    # Print CUDA availability
    logger.info(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    uvicorn.run(app, host="localhost", port=8000)
