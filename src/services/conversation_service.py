import time
import requests
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config.settings import settings
from components.voice.voice_manager import VoiceGenerator
from components.llm.llm_client import get_ai_response, parse_stream_chunk
from components.audio.player import save_audio
from components.stt.whisper_transcriber import transcribe_audio
from components.text_processing.chunker import TextChunker
from components.commands.command_handler import CommandHandler

class ConversationService:
    def __init__(self):
        settings.setup_directories()
        self.session = requests.Session()
        self.generator = VoiceGenerator(settings.MODELS_DIR, settings.VOICES_DIR)
        self.command_handler = CommandHandler(self)

        print("\nInitializing Whisper model...")
        self.whisper_processor = WhisperProcessor.from_pretrained(settings.WHISPER_MODEL)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(settings.WHISPER_MODEL)
        
        print("\n=== Voice Chat Bot Initializing ===")
        print("Device being used:", self.generator.device)
        print("\nInitializing voice generator...")
        # Store the model path for later use, e.g., when changing voices
        self.generator.model_path = settings.TTS_MODEL
        result = self.generator.initialize(self.generator.model_path, settings.VOICE_NAME)
        print(result)

        self.messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]
        self.speed = settings.SPEED
        self._warmup_llm()

    def _warmup_llm(self):
        try:
            print("\nWarming up the LLM model...")
            health = self.session.get("http://localhost:11434", timeout=3)
            if health.status_code != 200:
                print("Ollama not running! Start it first.")
                return
            get_ai_response(
                session=self.session,
                messages=[
                    {"role": "system", "content": "Hi"},
                    {"role": "user", "content": "Hi!"},
                ],
                llm_model=settings.LLM_MODEL,
                llm_url=settings.OLLAMA_URL,
                max_tokens=10,
                stream=False,
            )
        except requests.RequestException as e:
            print(f"Warmup failed: {str(e)}")

    def process_audio(self, audio_data):
        print("\nTranscribing detected speech...")
        transcription_start_time = time.perf_counter()
        
        user_input = transcribe_audio(
            self.whisper_processor, self.whisper_model, audio_data
        )
        
        transcription_duration = time.perf_counter() - transcription_start_time
        print(f"Transcription took: {transcription_duration:.2f}s")

        if not user_input.strip():
            return {"transcribed_text": "", "response_text": "No speech detected.", "audio_path": None}

        print(f"User (voice): {user_input}")
        self.messages.append({"role": "user", "content": user_input})
        
        print("\nThinking...")
        try:
            response_stream = get_ai_response(
                session=self.session,
                messages=self.messages,
                llm_model=settings.LLM_MODEL,
                llm_url=settings.OLLAMA_URL,
                max_tokens=settings.MAX_TOKENS,
                stream=True,
            )

            if not response_stream:
                return {"transcribed_text": user_input, "response_text": "Failed to get AI response.", "audio_path": None}

            # Simplified handling for streaming response
            full_response_text = ""
            for chunk in response_stream:
                data = parse_stream_chunk(chunk)
                if data and "choices" in data and "delta" in data["choices"][0] and "content" in data["choices"][0]["delta"]:
                    content = data["choices"][0]["delta"]["content"]
                    if content:
                        full_response_text += content
            
            print(f"Assistant: {full_response_text}")

            # Check if the response is a command
            if full_response_text.strip().startswith("[COMMAND:"):
                command_result = self.command_handler.handle_command(full_response_text.strip())
                response_text = command_result
                self.messages.append({"role": "assistant", "content": response_text})
                # Generate audio for the command's result
                audio_segment, _ = self.generator.generate(response_text, speed=self.speed)
                if audio_segment is None:
                    return {"transcribed_text": user_input, "response_text": response_text, "audio_path": None}
                full_audio_segments = [torch.from_numpy(audio_segment)]
            else:
                # Process for TTS
                response_text = full_response_text
                self.messages.append({"role": "assistant", "content": response_text})
                audio_segment, _ = self.generator.generate(response_text, speed=self.speed)
                if audio_segment is None:
                    return {"transcribed_text": user_input, "response_text": response_text, "audio_path": None}
                full_audio_segments = [torch.from_numpy(audio_segment)]

            # Save and return audio
            final_audio = torch.cat(full_audio_segments, dim=-1)
            output_filename = f"output_{int(time.time())}.wav"
            
            # Ensure the output directory exists
            output_dir = settings.BASE_DIR / "src" / "web" / "static" / "audio" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / output_filename
            save_audio(final_audio, str(output_path), 24000)
            
            static_path = f"static/audio/output/{output_filename}"

            return {"transcribed_text": user_input, "response_text": response_text, "audio_path": static_path}

        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            return {"transcribed_text": user_input, "response_text": f"An error occurred: {e}", "audio_path": None}

