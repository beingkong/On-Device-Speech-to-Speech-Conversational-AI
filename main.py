import keyboard
import traceback
import time
import requests
import threading
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config.settings import settings
from components.voice.voice_manager import VoiceGenerator
from components.llm.llm_client import get_ai_response, parse_stream_chunk
from components.audio.player import play_audio_with_interrupt, check_for_speech
from components.vad.vad_pipeline import init_vad_pipeline, detect_speech_segments
from components.audio.recorder import record_continuous_audio
from components.stt.whisper_transcriber import transcribe_audio
from components.audio.queue import AudioGenerationQueue
from components.text_processing.chunker import TextChunker

settings.setup_directories()

class ConversationManager:
    def __init__(
        self,
        session: requests.Session,
        generator: VoiceGenerator,
        whisper_processor: WhisperProcessor,
        whisper_model: WhisperForConditionalGeneration,
        vad_pipeline,
    ):
        self.session = session
        self.generator = generator
        self.whisper_processor = whisper_processor
        self.whisper_model = whisper_model
        self.vad_pipeline = vad_pipeline
        self.messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]
        self.speed = settings.SPEED
        self.timing_info = {
            "vad_start": None,
            "transcription_start": None,
            "llm_first_token": None,
            "audio_queued": None,
            "first_audio_play": None,
            "playback_start": None,
            "end": None,
            "transcription_duration": None,
        }

    def _audio_playback_worker(self, audio_queue) -> tuple[bool, None]:
        was_interrupted = False
        interrupt_audio = None

        try:
            while True:
                speech_detected, audio_data = check_for_speech()
                if speech_detected:
                    was_interrupted = True
                    interrupt_audio = audio_data
                    break

                audio_data, _ = audio_queue.get_next_audio()
                if audio_data is not None:
                    if not self.timing_info["first_audio_play"]:
                        self.timing_info["first_audio_play"] = time.perf_counter()

                    was_interrupted, interrupt_data = play_audio_with_interrupt(audio_data)
                    if was_interrupted:
                        interrupt_audio = interrupt_data
                        break
                else:
                    time.sleep(settings.PLAYBACK_DELAY)

                if (
                    not audio_queue.is_running
                    and audio_queue.sentence_queue.empty()
                    and audio_queue.audio_queue.empty()
                ):
                    break

        except Exception as e:
            print(f"Error in audio playback: {str(e)}")

        return was_interrupted, interrupt_audio

    def process_input(self, user_input: str) -> tuple[bool, None]:
        self.timing_info = {k: None for k in self.timing_info}
        self.timing_info["vad_start"] = time.perf_counter()

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
                print("Failed to get AI response stream.")
                return False, None

            audio_queue = AudioGenerationQueue(self.generator, self.speed)
            audio_queue.start()
            chunker = TextChunker()
            complete_response = []

            playback_thread = threading.Thread(
                target=lambda: self._audio_playback_worker(audio_queue)
            )
            playback_thread.daemon = True
            playback_thread.start()

            for chunk in response_stream:
                data = parse_stream_chunk(chunk)
                if not data or "choices" not in data:
                    continue

                choice = data["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    content = choice["delta"]["content"]
                    if content:
                        if not self.timing_info["llm_first_token"]:
                            self.timing_info["llm_first_token"] = time.perf_counter()
                        print(content, end="", flush=True)
                        chunker.current_text.append(content)

                        text = "".join(chunker.current_text)
                        if chunker.should_process(text):
                            if not self.timing_info["audio_queued"]:
                                self.timing_info["audio_queued"] = time.perf_counter()
                            remaining = chunker.process(text, audio_queue)
                            chunker.current_text = [remaining]
                            complete_response.append(text[: len(text) - len(remaining)])

                if choice.get("finish_reason") == "stop":
                    final_text = "".join(chunker.current_text).strip()
                    if final_text:
                        chunker.process(final_text, audio_queue)
                        complete_response.append(final_text)
                    break

            self.messages.append({"role": "assistant", "content": " ".join(complete_response)})
            print()

            time.sleep(0.1)
            audio_queue.stop()
            playback_thread.join()

            self.timing_info["end"] = time.perf_counter()
            self._print_timing_chart(self.timing_info)
            return False, None

        except Exception as e:
            print(f"\nError during streaming: {str(e)}")
            if "audio_queue" in locals():
                audio_queue.stop()
            return False, None

    def _print_timing_chart(self, metrics):
        base_time = metrics["vad_start"]
        events = [
            ("User stopped speaking", metrics["vad_start"]),
            ("VAD started", metrics["vad_start"]),
            ("Transcription started", metrics["transcription_start"]),
            ("LLM first token", metrics["llm_first_token"]),
            ("Audio queued", metrics["audio_queued"]),
            ("First audio played", metrics["first_audio_play"]),
            ("Playback started", metrics["playback_start"]),
            ("End-to-end response", metrics["end"]),
        ]

        print("\nTiming Chart:")
        print(f"{'Event':<25} | {'Time (s)':>9} | {'Δ+':>6}")
        print("-" * 45)

        prev_time = base_time
        for name, t in events:
            if t is None:
                continue
            elapsed = t - base_time
            delta = t - prev_time
            print(f"{name:<25} | {elapsed:9.2f} | {delta:6.2f}")
            prev_time = t


def main():
    with requests.Session() as session:
        try:
            generator = VoiceGenerator(settings.MODELS_DIR, settings.VOICES_DIR)
            print("\nInitializing Whisper model...")
            whisper_processor = WhisperProcessor.from_pretrained(settings.WHISPER_MODEL)
            whisper_model = WhisperForConditionalGeneration.from_pretrained(
                settings.WHISPER_MODEL
            )
            print("\nInitializing Voice Activity Detection...")
            vad_pipeline = init_vad_pipeline(settings.HUGGINGFACE_TOKEN)
            print("\n=== Voice Chat Bot Initializing ===")
            print("Device being used:", generator.device)
            print("\nInitializing voice generator...")
            result = generator.initialize(settings.TTS_MODEL, settings.VOICE_NAME)
            print(result)

            conversation_manager = ConversationManager(
                session,
                generator,
                whisper_processor,
                whisper_model,
                vad_pipeline,
            )

            try:
                print("\nWarming up the LLM model...")
                health = session.get("http://localhost:11434", timeout=3)
                if health.status_code != 200:
                    print("Ollama not running! Start it first.")
                    return
                response_stream = get_ai_response(
                    session=session,
                    messages=[
                        {"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": "Hi!"},
                    ],
                    llm_model=settings.LLM_MODEL,
                    llm_url=settings.OLLAMA_URL,
                    max_tokens=settings.MAX_TOKENS,
                    stream=False,
                )
                if not response_stream:
                    print("Failed to initialized the AI model!")
                    return
            except requests.RequestException as e:
                print(f"Warmup failed: {str(e)}")

            print("\n\n=== Voice Chat Bot Ready ===")
            print("The bot is now listening for speech.")
            print("Just start speaking, and I'll respond automatically!")
            print("You can interrupt me anytime by starting to speak.")
            while True:
                try:
                    if keyboard.is_pressed("enter"):
                        user_input = input("\nYou (text): ").strip()

                        if user_input.lower() == "quit":
                            print("Goodbye!")
                            break

                    audio_data = record_continuous_audio()
                    if audio_data is not None:
                        speech_segments = detect_speech_segments(
                            conversation_manager.vad_pipeline, audio_data
                        )

                        if speech_segments is not None:
                            print("\nTranscribing detected speech...")
                            conversation_manager.timing_info["transcription_start"] = time.perf_counter()

                            user_input = transcribe_audio(
                                conversation_manager.whisper_processor, conversation_manager.whisper_model, speech_segments
                            )

                            conversation_manager.timing_info["transcription_duration"] = (
                                time.perf_counter() - conversation_manager.timing_info["transcription_start"]
                            )
                            if user_input.strip():
                                print(f"You (voice): {user_input}")
                                was_interrupted, speech_data = conversation_manager.process_input(
                                    user_input
                                )
                                if was_interrupted and speech_data is not None:
                                    speech_segments = detect_speech_segments(
                                        conversation_manager.vad_pipeline, speech_data
                                    )
                                    if speech_segments is not None:
                                        print("\nTranscribing interrupted speech...")
                                        user_input = transcribe_audio(
                                            conversation_manager.whisper_processor,
                                            conversation_manager.whisper_model,
                                            speech_segments,
                                        )
                                        if user_input.strip():
                                            print(f"You (voice): {user_input}")
                                            conversation_manager.process_input(
                                                user_input
                                            )
                        else:
                            print("No clear speech detected, please try again.")
                    if session is not None:
                        session.headers.update({"Connection": "keep-alive"})
                        if hasattr(session, "connection_pool"):
                            session.connection_pool.clear()

                except KeyboardInterrupt:
                    print("\nStopping...")
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()


if __name__ == "__main__":
    main()