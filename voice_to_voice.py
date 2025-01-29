import os
import msvcrt
import traceback
import time
import requests
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
import inspect

settings.setup_directories()
logger = logging.getLogger('voice_chat')

def process_input(session: requests.Session, user_input: str, messages: list, generator: VoiceGenerator, speed: float, 
                  process_start_time: float, vad_time: float, transcribe_time: float) -> tuple[bool, None]:
    """Processes user input with timing parameters"""
    start_time = process_start_time
    messages.append({"role": "user", "content": user_input})
    

    try:
        timing_info = []
        complete_response = []
        
        # Initialize with VAD and transcription times
        timing_info.append(f"0. User stopped speaking: {0:.2f}s (Δ+0.00s)")
        timing_info.append(f"1. VAD started: {vad_time:.2f}s (Δ+{vad_time:.2f}s)")
        timing_info.append(f"2. Transcription started: {(transcribe_time - vad_time):.2f}s (Δ+{transcribe_time - vad_time:.2f}s)")


        last_timing_step = start_time + vad_time + transcribe_time
        
        llm_start = time.time()
        response_stream = get_ai_response(
            session=session,
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
        delta = llm_first_token_time - last_timing_step
        timing_info.append(f"3. LLM processing started: {(llm_first_token_time - start_time):.2f}s (Δ+{delta:.2f}s)")
        last_timing_step = llm_first_token_time
            

        audio_queue = AudioGenerationQueue(generator, speed)
        audio_queue.start()
        chunker = TextChunker()
        first_audio_generated = False
        first_token_received = False
        current_sentence = []
        
        # Store first generation time
        first_generation_time = None
        
        # Create shared timing structure
        timing_data = {
            'start_time': start_time,
            'last_step': last_timing_step
        }
        
        # Pass to playback thread
        playback_thread = threading.Thread(
            target=audio_playback_worker,
            args=(audio_queue, timing_info, timing_data)
        )
        playback_thread.daemon = True
        playback_thread.start()
        
        for chunk in response_stream:
            # Add heartbeat every 2 seconds
            if time.time() - last_timing_step > 2:
                chunk += b' '  # Add empty space to keep connection alive
            data = parse_stream_chunk(chunk)
            if not data or "choices" not in data:
                continue
                
            choice = data["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                content = choice["delta"]["content"]
                if content:
                    if not first_token_received:
                        first_token_received = True
                        text_token_time = time.time()
                        delta = text_token_time - last_timing_step
                        timing_info.append(f"4. First text token: {text_token_time - start_time:.2f}s (Δ+{delta:.2f}s)")
                        last_timing_step = text_token_time
                    
                    print(content, end='', flush=True)
                    current_sentence.append(content)
                    text = ''.join(current_sentence)
                    
                    should_process = False
                    words = text.split()
                    
                    if not chunker.found_first_sentence:
                        if len(words) >= 2:
                            for break_char in chunker.sentence_breaks:
                                if text.endswith(break_char):
                                    should_process = True
                                    break
                            if not should_process:
                                for break_char in chunker.semantic_breaks:
                                    if text.endswith(break_char) and len(words) >= 2:
                                        should_process = True
                                        break
                    else:
                        if any(text.endswith(p) for p in chunker.sentence_breaks) and len(words) > settings.TARGET_SIZE/2:
                            should_process = True
                        elif len(words) > settings.TARGET_SIZE:
                            last_word = words[-1]
                            if last_word.lower() in chunker.semantic_breaks or \
                               any(last_word.endswith(p) for p in chunker.sentence_breaks):
                                should_process = True
                    
                    if should_process and any(c.isalnum() for c in text):
                        break_idx = -1
                        for break_char in (chunker.sentence_breaks | chunker.semantic_breaks):
                            idx = text.rfind(break_char)
                            if idx > break_idx:
                                next_char = text[idx+1] if idx+1 < len(text) else ' '
                                if next_char.isspace() or idx == len(text)-1:
                                    break_idx = idx
                        
                        if break_idx >= 0:
                            chunk = text[:break_idx + 1].strip()
                            remaining = text[break_idx + 1:].strip()
                            
                            if chunk and any(c.isalnum() for c in chunk):
                                audio_queue.add_sentences([chunk])
                                complete_response.append(chunk)
                                chunker.found_first_sentence = True
                                
                                # Track first generation time
                                if not first_generation_time:
                                    first_generation_time = time.time()
                                    delta = first_generation_time - last_timing_step
                                    timing_info.append(f"5. Audio queued: {first_generation_time - start_time:.2f}s (Δ+{delta:.2f}s)")
                                    last_timing_step = first_generation_time
                                
                                current_sentence = [remaining] if remaining else []
                            else:
                                pass
                        else:
                            pass
            
            if choice.get("finish_reason") == "stop":
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
        delta = end_time - start_time
        timing_info.append(f"6. End-to-end response time: {end_time - start_time:.2f}s (Δ+{delta:.2f}s)")
        
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

def audio_playback_worker(audio_queue, timing_info, timing_data):
    """Manages audio playback with shared timing data"""
    was_interrupted = False
    first_audio_played = False
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
                play_start = time.time()
                was_interrupted, interrupt_data = play_audio_with_interrupt(audio_data)
                timing_info.append(f"6. Audio playback started: {play_start - timing_data['start_time']:.2f}s (Δ+{play_start - timing_data['last_step']:.2f}s)")
                timing_data['last_step'] = play_start
                if not first_audio_played:
                    first_audio_played = True
                    current_time = time.time()
                    elapsed = current_time - timing_data['start_time']
                    delta = current_time - timing_data['last_step']
                    timing_info.append(f"6. First audio playback completed: {elapsed:.2f}s (Δ+{delta:.2f}s)")
                    timing_data['last_step'] = current_time
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
    session = None  # Will be properly initialized in the loop
    generator = None
    
    try:
        # Persistent session outside the loop
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=100,
            max_retries=3
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)

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
        
        messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]
        speed = settings.SPEED
        
        while True:
            try:
                if msvcrt.kbhit():
                    user_input = input("\nYou (text): ").strip()
                    
                    if user_input.lower() == 'quit':
                        print("Goodbye!")
                        break
                audio_data = record_continuous_audio()
                if audio_data is not None:
                    vad_start = time.time()
                    speech_segments = detect_speech_segments(vad_pipeline, audio_data)
                    vad_time = time.time() - vad_start
                    
                    if speech_segments is not None:
                        transcribe_start = time.time()
                        print("\nTranscribing detected speech...")
                        user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                        transcribe_time = time.time() - transcribe_start
                        
                        process_start = time.time()
                        if user_input.strip():
                            print(f"You (voice): {user_input}")
                            was_interrupted, speech_data = process_input(
                                session=session,
                                user_input=user_input,
                                messages=messages,
                                generator=generator,
                                speed=speed,
                                process_start_time=process_start,
                                vad_time=vad_time,
                                transcribe_time=transcribe_time
                            )
                            # Initialize timing_info before concatenation
                            timing_info = []
                            if was_interrupted:
                                timing_info = [
                                    f"1. Voice activity detection: {vad_time:.2f}s",
                                    f"2. Audio transcription: {transcribe_time:.2f}s"
                                ] + timing_info
                            if was_interrupted and speech_data is not None:
                                speech_segments = detect_speech_segments(vad_pipeline, speech_data)
                                if speech_segments is not None:
                                    print("\nTranscribing interrupted speech...")
                                    user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                                    if user_input.strip():
                                        print(f"You (voice): {user_input}")
                                        process_input(session, user_input, messages, generator, speed, time.time())
                    else:
                        print("No clear speech detected, please try again.")
                
                # Add connection maintenance
                if session is not None:
                    # Reset connection pool between interactions
                    session.headers.update({'Connection': 'keep-alive'})
                    if hasattr(session, 'connection_pool'):
                        session.connection_pool.clear()
                
            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
                
    finally:
        # Clean up resources
        if session:
            session.close()
        if generator:
            generator.cleanup()

if __name__ == "__main__":
    main()
