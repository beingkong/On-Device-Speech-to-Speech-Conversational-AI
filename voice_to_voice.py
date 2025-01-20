import os
import msvcrt
import traceback
import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from src.utils.config import settings
from src.utils import (
    VoiceGenerator, split_into_sentences, filter_response,
    get_ai_response, play_audio_with_interrupt, handle_commands,
    init_vad_pipeline, detect_speech_segments, record_audio,
    record_continuous_audio, check_for_speech, transcribe_audio,
    generate_and_play_sentences
)
from src.utils.audio_queue import AudioGenerationQueue
from src.utils.llm import parse_stream_chunk
import threading

settings.setup_directories()

def process_input(user_input: str, messages: list, generator: VoiceGenerator, speed: float) -> tuple[bool, None]:
    """Processes user input, generates a response, and handles audio output.

    The function sends user input to the AI, processes the streaming response in chunks,
    and plays the audio. It prioritizes a quick initial response by using a smaller
    target size for the first sentence. It also allows for interruptions during playback.

    Args:
        user_input: The input text from the user.
        messages: The list of chat messages for context.
        generator: The voice generator object.
        speed: The speed at which to generate audio.

    Returns:
        A tuple indicating if the process was interrupted and None.
    """
    messages.append({"role": "user", "content": user_input})
    
    print("\nThinking...")
    ai_response = []
    complete_response = []
    current_text = []
    found_first_sentence = False
    
    try:
        response_stream = get_ai_response(
            messages=messages,
            llm_model=settings.LLM_MODEL,
            lm_studio_url=settings.LM_STUDIO_URL,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.LM_STUDIO_TEMPERATURE,
            stream=True
        )
        
        if response_stream is None:
            print("Failed to get AI response stream.")
            return False, None
        audio_queue = AudioGenerationQueue(generator, speed)
        audio_queue.start()
        
        def should_process_chunk(text: str) -> bool:
            """Check if we have enough text to process a chunk.
            
            Args:
                text: Current accumulated text
                
            Returns:
                bool: True if we should process the chunk
            """
            is_sentence_end = any(text.endswith(p) for p in ('.', '!', '?',';', ':'))
            if is_sentence_end:
                return True
                
            words = text.split()
            if not found_first_sentence:
                return len(words) >= settings.FIRST_SENTENCE_SIZE
            return len(words) >= settings.TARGET_SIZE
        
        def process_chunk(text: str) -> str:
            """Process text chunk and return remaining text.
            
            Args:
                text: Text to process
            
            Returns:
                Remaining unprocessed text
            """
            nonlocal found_first_sentence
            
            words = text.split()
            
            if len(words) == 0:
                return ""
                
            target_size = settings.FIRST_SENTENCE_SIZE if not found_first_sentence else settings.TARGET_SIZE
            

            if len(words) <= target_size and found_first_sentence:
                chunk = ' '.join(words)
                if chunk.strip() and any(c.isalnum() for c in chunk):
                    ai_response.append(chunk)
                    complete_response.append(chunk)
                    audio_queue.add_sentences([chunk])
                return ""
            
            connectors = {'and', 'but', 'because', 'then'}
            
            break_points = []
            for i, word in enumerate(words[:target_size+3]):
                if any(word.endswith(p) for p in ('.', '!', '?', ';', ':')) or any(p in word for p in ('\n', '\r')):
                    break_points.append((i, 2))
                elif word.lower() in connectors:
                    break_points.append((i, 1))
            
            found_first_sentence = True
            break_points.sort(key=lambda x: (x[1], x[0]), reverse=True)
            
            if break_points and break_points[0][0] >= target_size-2:
                split_point = break_points[0][0] + 1
            else:
                split_point = target_size
            
            chunk = ' '.join(words[:split_point])
            if chunk.strip() and any(c.isalnum() for c in chunk):
                ai_response.append(chunk)
                complete_response.append(chunk)
                audio_queue.add_sentences([chunk])
            
            return ' '.join(words[split_point:])

        def audio_playback_worker() -> tuple[bool, None]:
            """Manages audio playback in a separate thread, handling interruptions.

            Returns:
                A tuple indicating if the playback was interrupted and the interrupt audio data.
            """
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
                        was_interrupted, interrupt_data = play_audio_with_interrupt(audio_data)
                        if was_interrupted:
                            interrupt_audio = interrupt_data
                            break
                    else:
                        time.sleep(settings.PLAYBACK_DELAY)
                        
                    if not audio_queue.is_running and audio_queue.sentence_queue.empty() and audio_queue.audio_queue.empty():
                        break
                        
            except Exception as e:
                print(f"Error in audio playback: {str(e)}")
                
            return was_interrupted, interrupt_audio
            
        playback_thread = threading.Thread(target=audio_playback_worker)
        playback_thread.daemon = True
        playback_thread.start()
        
        for chunk in response_stream:
            data = parse_stream_chunk(chunk)
            if not data:
                continue
                
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                
                if "delta" in choice:
                    delta = choice["delta"]
                    if "content" in delta:
                        content = delta["content"]
                        if content:
                            print(content, end='', flush=True)
                            current_text.append(content)
                            
                            text = ''.join(current_text)
                            if should_process_chunk(text):
                                remaining = process_chunk(text)
                                current_text = [remaining]
                if choice.get("finish_reason") == "stop":
                    text = ''.join(current_text).strip()
                    if text:
                        process_chunk(text)
                    break
        
        messages.append({"role": "assistant", "content": ' '.join(complete_response)})
        print()
        
        time.sleep(0.1)
        audio_queue.stop()
        playback_thread.join()
        
        return False, None
        
    except Exception as e:
        print(f"\nError during streaming: {str(e)}")
        if 'audio_queue' in locals():
            audio_queue.stop()
        return False, None

def main():
    """Main function to run the voice chat bot."""
    try:
        generator = VoiceGenerator(settings.MODELS_DIR, settings.VOICES_DIR)
        
        print("\nInitializing Whisper model...")
        whisper_processor = WhisperProcessor.from_pretrained(settings.WHISPER_MODEL)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(settings.WHISPER_MODEL)
        
        print("\nInitializing Voice Activity Detection...")
        vad_pipeline = init_vad_pipeline(settings.HUGGINGFACE_TOKEN)
        
        print("\n=== Voice Chat Bot Initializing ===")
        print("Device being used:", generator.device)
        
        print("\nInitializing voice generator...")
        result = generator.initialize(settings.TTS_MODEL, settings.VOICE_NAME)
        print(result)
        
        print("\n=== Voice Chat Bot Ready ===")
        print("The bot is now listening for speech.")
        print("Just start speaking, and I'll respond automatically!")
        print("You can interrupt me anytime by starting to speak.")
        print("\nOther options:")
        print("  - Type text messages and press Enter")
        print("  - Use 'v' for manual voice recording")
        print("  - Commands: speed=X, voice=X, voices, mix=voice1,voice2")
        print("  - Type 'quit' to exit")
        print("-" * 50)
        
        messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]
        speed = settings.SPEED
        
        while True:
            try:
                if msvcrt.kbhit():
                    user_input = input("\nYou (text): ").strip()
                    
                    if user_input.lower() == 'quit':
                        print("Goodbye!")
                        break
                        
                    if user_input.lower() == 'v':
                        audio_data = record_audio()
                        if audio_data is not None:
                            speech_segments = detect_speech_segments(vad_pipeline, audio_data)
                            if speech_segments is not None:
                                print("\nTranscribing recorded speech...")
                                user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                                if user_input.strip():
                                    print(f"You (voice): {user_input}")
                                    was_interrupted, speech_data = process_input(user_input, messages, generator, speed)
                                    if was_interrupted and speech_data is not None:
                                        speech_segments = detect_speech_segments(vad_pipeline, speech_data)
                                        if speech_segments is not None:
                                            print("\nTranscribing interrupted speech...")
                                            user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                                            if user_input.strip():
                                                print(f"You (voice): {user_input}")
                                                process_input(user_input, messages, generator, speed)
                    else:
                        if handle_commands(user_input, generator, speed, settings.TTS_MODEL):
                            continue
                        was_interrupted, speech_data = process_input(user_input, messages, generator, speed)
                        if was_interrupted and speech_data is not None:
                            speech_segments = detect_speech_segments(vad_pipeline, speech_data)
                            if speech_segments is not None:
                                print("\nTranscribing interrupted speech...")
                                user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                                if user_input.strip():
                                    print(f"You (voice): {user_input}")
                                    process_input(user_input, messages, generator, speed)
                        continue
                
                audio_data = record_continuous_audio()
                if audio_data is not None:
                    speech_segments = detect_speech_segments(vad_pipeline, audio_data)
                    
                    if speech_segments is not None:
                        print("\nTranscribing detected speech...")
                        user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                        if user_input.strip():
                            print(f"You (voice): {user_input}")
                            was_interrupted, speech_data = process_input(user_input, messages, generator, speed)
                            if was_interrupted and speech_data is not None:
                                speech_segments = detect_speech_segments(vad_pipeline, speech_data)
                                if speech_segments is not None:
                                    print("\nTranscribing interrupted speech...")
                                    user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                                    if user_input.strip():
                                        print(f"You (voice): {user_input}")
                                        process_input(user_input, messages, generator, speed)
                    else:
                        print("No clear speech detected, please try again.")
                
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
