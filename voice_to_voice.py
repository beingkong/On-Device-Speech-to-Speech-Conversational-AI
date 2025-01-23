import os
import msvcrt
import traceback
import time
import torch
import logging
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
from src.utils.text_chunker import TextChunker

settings.setup_directories()
logger = logging.getLogger('voice_chat')

def process_input(user_input: str, messages: list, generator: VoiceGenerator, speed: float) -> tuple[bool, None]:
    """Processes user input, generates a response, and handles audio output."""
    start_time = time.time()
    messages.append({"role": "user", "content": user_input})
    print("\nThinking...")
    
    try:
        # Store all timing and response info to log at the end
        timing_info = []
        complete_response = []
        
        llm_start_time = time.time()
        response_stream = get_ai_response(
            messages=messages,
            llm_model=settings.LLM_MODEL,
            lm_studio_url=settings.LM_STUDIO_URL,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.LM_STUDIO_TEMPERATURE,
            stream=True
        )
        
        if not response_stream:
            print("Failed to get AI response stream.")
            return False, None
            
        llm_first_token_time = time.time()
        timing_info.append(f"1. Time to first LLM token: {llm_first_token_time - start_time:.2f} seconds")
            
        audio_queue = AudioGenerationQueue(generator, speed)
        audio_queue.start()
        chunker = TextChunker()
        first_audio_generated = False
        first_token_received = False
        current_sentence = []
        
        # Start audio playback thread
        playback_thread = threading.Thread(
            target=lambda: audio_playback_worker(audio_queue)
        )
        playback_thread.daemon = True
        playback_thread.start()
        
        # Process streaming response
        for chunk in response_stream:
            data = parse_stream_chunk(chunk)
            if not data or "choices" not in data:
                continue
                
            choice = data["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                content = choice["delta"]["content"]
                if content:
                    if not first_token_received:
                        first_token_received = True
                        timing_info.append(f"2. Time to first text token: {time.time() - start_time:.2f} seconds")
                    
                    print(content, end='', flush=True)
                    current_sentence.append(content)
                    text = ''.join(current_sentence)
                    
                    # Check if we should send this for audio generation
                    should_process = False
                    if not chunker.found_first_sentence:
                        # For first sentence, be aggressive - break at any semantic break or punctuation
                        if any(break_char in content for break_char in chunker.semantic_breaks) or \
                           any(break_char in content for break_char in chunker.sentence_breaks):
                            should_process = True
                    else:
                        # For subsequent sentences, only break at sentence breaks or if over target size
                        words = text.split()
                        if any(break_char in content for break_char in chunker.sentence_breaks) and \
                           len(words) > settings.TARGET_SIZE/2:
                            should_process = True
                        elif len(words) > settings.TARGET_SIZE and \
                             any(break_char in content for break_char in chunker.semantic_breaks):
                            should_process = True
                    
                    if should_process and any(c.isalnum() for c in text):
                        # Keep track of where we found the break
                        break_idx = -1
                        # Find the last occurrence of any break character
                        for break_char in (chunker.sentence_breaks | chunker.semantic_breaks):
                            idx = text.rfind(break_char)
                            if idx > break_idx:
                                break_idx = idx
                        
                        if break_idx >= 0:
                            # Include the break character in the current chunk
                            chunk = text[:break_idx + 1]
                            # Keep the rest for the next sentence
                            remaining = text[break_idx + 1:]
                            
                            if any(c.isalnum() for c in chunk):
                                audio_queue.add_sentences([chunk])
                                complete_response.append(chunk)
                                chunker.found_first_sentence = True
                                
                                if not first_audio_generated and audio_queue.audio_generated > 0:
                                    first_audio_time = time.time()
                                    timing_info.append(f"3. Time to first audio generation: {first_audio_time - start_time:.2f} seconds")
                                    first_audio_generated = True
                                
                            # Start new sentence with any remaining text
                            current_sentence = [remaining] if remaining.strip() else []
                        else:
                            # If no break found but we should process, send the whole text
                            audio_queue.add_sentences([text])
                            complete_response.append(text)
                            current_sentence = []
                            chunker.found_first_sentence = True
            
            if choice.get("finish_reason") == "stop":
                # Process any remaining text
                text = ''.join(current_sentence)
                if text.strip() and any(c.isalnum() for c in text):
                    audio_queue.add_sentences([text])
                    complete_response.append(text)
                break
        
        messages.append({"role": "assistant", "content": ' '.join(complete_response)})
        print()
        
        time.sleep(0.1)
        audio_queue.stop()
        playback_thread.join()
        
        end_time = time.time()
        timing_info.append(f"4. Total processing time: {end_time - start_time:.2f} seconds")
        
        # Log everything at once after processing is complete
        logger.info("\n" + "=" * 50)
        logger.info("User Question: " + user_input)
        logger.info("-" * 50)
        logger.info("\nAI Response:")
        logger.info(' '.join(complete_response))
        logger.info("\nTiming Results:")
        for timing in timing_info:
            logger.info(timing)
        logger.info("=" * 50 + "\n")
        
        return False, None
        
    except Exception as e:
        print(f"\nError during streaming: {str(e)}")
        if 'audio_queue' in locals():
            audio_queue.stop()
        return False, None

def audio_playback_worker(audio_queue) -> tuple[bool, None]:
    """Manages audio playback in a separate thread, handling interruptions."""
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
